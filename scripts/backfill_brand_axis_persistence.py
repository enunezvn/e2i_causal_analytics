#!/usr/bin/env python3
"""Surgical, idempotent re-derivation of a BRAND's persistence labels to plant its
brand-distinct clinical-axis differential (issue #1321) — that brand's rows ONLY.

WHY THIS SCRIPT EXISTS
----------------------
Each brand's distinct axis (Fabhalta prior-C5 switch, Kisqali advanced-line burden,
Remibrutinib uncontrolled CSU) is a REAL, recoverable causal axis on persistence. The
committed DGP (cohort_outcomes.generate_discontinuation_outcomes +
patient_generator._BRAND_AXIS_DIFFERENTIALS) plants it at synthetic-generation time,
but the live droplet is FROZEN (full reseed = disaster recovery only; the live labels
are only ~63% bit-reproducible — the ad-hoc boosted UPDATE's RNG seed was never
committed). Like the shipped Fabhalta pilot's backfill, this brings the EXISTING rows
into line IN PLACE: it reads their covariates (never re-drawn) and re-derives
persistent_180d / discontinued_180d from the committed DGP WITH the axis differential,
keyed deterministically so re-runs are idempotent.

The axis is derived from an ALREADY-POPULATED, brand-gated eligibility column — no new
column, no migration. Only that brand's rows are touched; every other brand is left
byte-identical.

DESIGN (mean-preserving, so the marginal is not disturbed)
----------------------------------------------------------
  * MAIN effect: axis=1 -> more discontinuation, MEAN-CENTERED so the brand's
    persistence PREVALENCE is preserved while the axis contrast becomes recoverable.
  * CATE modifier: axis x treatment_arm, mean-preserving — axis=1 patients respond
    less; population ATE unchanged. main_pull / exp_mult MATCH the generator's
    _BRAND_AXIS_DIFFERENTIALS for the brand (so the substrate matches the DGP design).
  * copay_support + psp_enrolled are RE-PASSED so their planted persistence pulls
    (COMM-ARMS) are preserved, not clobbered.

BLAST-RADIUS DISPROOF (printed by the default dry-run, BEFORE any write)
-----------------------------------------------------------------------
The dry-run re-derives in memory and reports, against the LIVE labels: persistence
prevalence (band [0.05,0.60]); the axis RD (must be clearly negative & recoverable); a
leakage-safe proxy AUC (RandomForest 3-fold) on BOTH current-live and re-derived labels
— the DELTA gauges the WS1 goldstd persistence-AUC blast radius (pins ~0.79-0.80; a
large drop is a STOP signal); per-row flip rate + complement property. It also writes a
TSV backup of the current live labels. Writes NOTHING.

USAGE
-----
    # DEFAULT: dry-run (read-only + backup + disproof).  Run in the API container:
    docker exec e2i_api python scripts/backfill_brand_axis_persistence.py --brand Kisqali

    # WRITE PATH -- UPDATE that brand's persistence labels in the live table:
    docker exec e2i_api python scripts/backfill_brand_axis_persistence.py --brand Kisqali --execute
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

from src.ml.synthetic.config import Brand  # noqa: E402
from src.ml.synthetic.dgp.treatment_arm import _BRAND_CATE_SCALE  # noqa: E402
from src.ml.synthetic.generators.cohort_outcomes import (  # noqa: E402
    generate_discontinuation_outcomes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TABLE = "patient_journeys"
KEY = "patient_id"
# Fixed seed on the patient_id-sorted brand partition -> deterministic & idempotent
# (re-runs write identical labels). Not required to bit-match the generator's
# frame-order spawn-key plant (that is for fresh loads/tests); this is the
# live-substrate analogue. 1321 = the issue number, for traceability.
SEED = 1321
BATCH_SIZE = 500


@dataclass(frozen=True)
class AxisConfig:
    brand: str
    brand_enum: Brand
    axis_column: str  # already-populated brand-gated eligibility column
    experienced: Callable[[pd.Series], np.ndarray]  # column -> per-row 0/1 axis
    main_pull: float  # MUST match patient_generator._BRAND_AXIS_DIFFERENTIALS
    exp_mult: float
    axis_label: str  # human tag for the axis=1 contrast


_AXES: dict[str, AxisConfig] = {
    "Fabhalta": AxisConfig(
        brand="Fabhalta",
        brand_enum=Brand.FABHALTA,
        axis_column="complement_inhibitor_status",
        experienced=lambda s: (s.astype("object") == "prior").to_numpy(dtype=int),
        main_pull=0.55,
        exp_mult=0.40,
        axis_label="prior-C5 switch (prior vs current)",
    ),
    "Kisqali": AxisConfig(
        brand="Kisqali",
        brand_enum=Brand.KISQALI,
        axis_column="disease_stage",
        experienced=lambda s: s.astype(str)
        .str.lower()
        .isin({"metastatic", "stage_iv"})
        .to_numpy(dtype=int),
        main_pull=0.55,
        exp_mult=0.55,
        axis_label="advanced line (metastatic/stage_iv vs earlier)",
    ),
    "Remibrutinib": AxisConfig(
        brand="Remibrutinib",
        brand_enum=Brand.REMIBRUTINIB,
        axis_column="urticaria_severity_uas7",
        experienced=lambda s: (pd.to_numeric(s, errors="coerce") >= 28).to_numpy(dtype=int),
        main_pull=0.55,
        exp_mult=0.48,
        axis_label="uncontrolled CSU (UAS7 >= 28 vs controlled)",
    ),
}

_BASE_COVARIATE_COLS = [
    "patient_id",
    "brand",
    "treatment_arm",
    "disease_severity",
    "academic_hcp",
    "geographic_region",
    "segment_assignment",
    "insurance_type",
    "age_at_diagnosis",
    "comorbidity_burden",
    "prior_therapy_lines",
    "copay_support",
    "psp_enrolled",
    "persistent_180d",  # current live labels (backup + comparison)
    "discontinued_180d",
]

# Leakage-safe proxy-AUC features (base covariates + the axis). A comparable, NOT
# identical, signal to the WS1 goldstd eval — used only to bound the AUC blast radius.
_AUC_NUM = [
    "disease_severity",
    "academic_hcp",
    "age_at_diagnosis",
    "comorbidity_burden",
    "prior_therapy_lines",
    "copay_support",
    "psp_enrolled",
    "axis",
]
_AUC_CAT = ["geographic_region", "insurance_type"]


def fetch(client: Any, cfg: AxisConfig) -> Optional[pd.DataFrame]:
    """Read all synthetic rows for the brand (covariates + axis column + labels), paged."""
    cols = _BASE_COVARIATE_COLS + [cfg.axis_column]
    try:
        rows: List[dict] = []
        page, page_size = 0, 1000
        while True:
            resp = (
                client.table(TABLE)
                .select(",".join(cols))
                .eq("is_synthetic", True)
                .eq("brand", cfg.brand)
                .order(KEY)
                .range(page * page_size, (page + 1) * page_size - 1)
                .execute()
            )
            batch = resp.data or []
            rows.extend(batch)
            if len(batch) < page_size:
                break
            page += 1
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as e:  # pragma: no cover - network/permission edge
        logger.warning("Could not read live %s: %s", TABLE, e)
        return None


def regenerate(cov: pd.DataFrame, cfg: AxisConfig, *, seed: int = SEED) -> pd.DataFrame:
    """Re-derive persistent_180d / discontinued_180d from the committed DGP WITH the
    axis differential. patient_id-sorted, single deterministic stream -> idempotent.
    copay/psp re-passed so their pulls survive."""
    df = cov.copy().sort_values(KEY).reset_index(drop=True)
    experienced = cfg.experienced(df[cfg.axis_column])
    scale = float(_BRAND_CATE_SCALE.get(cfg.brand_enum, 1.0))
    rng = np.random.default_rng(seed)
    res = generate_discontinuation_outcomes(
        rng=rng,
        treatment_arm=df["treatment_arm"].fillna(0).to_numpy(dtype=int),
        disease_severity=df["disease_severity"].to_numpy(dtype=float),
        academic_hcp=df["academic_hcp"].fillna(0).to_numpy(dtype=int),
        geographic_region=df["geographic_region"].astype(str).to_numpy(),
        insurance_type=df["insurance_type"].astype(str).to_numpy(),
        age_at_diagnosis=df["age_at_diagnosis"].to_numpy(dtype=int),
        comorbidity_burden=df["comorbidity_burden"].to_numpy(dtype=int),
        prior_therapy_lines=df["prior_therapy_lines"].to_numpy(dtype=int),
        segment=df["segment_assignment"].astype(str).to_numpy(),
        brand_cate_scale=scale,
        copay_support=df["copay_support"].fillna(0).to_numpy(dtype=int),
        psp_enrolled=df["psp_enrolled"].fillna(0).to_numpy(dtype=int),
        axis_experienced=experienced,
        axis_main_pull=cfg.main_pull,
        axis_exp_mult=cfg.exp_mult,
    )
    out = df[[KEY]].copy()
    out["persistent_180d"] = res["persistent_180d"].astype(int)
    out["discontinued_180d"] = res["discontinued_180d"].astype(int)
    out["axis"] = experienced
    out["_axis_rd_attr"] = res["axis_persistent_rd"]
    return out


def _proxy_auc(cov: pd.DataFrame, labels: pd.Series, cfg: AxisConfig) -> float:
    """RandomForest 3-fold leakage-safe proxy AUC of persistence over the base
    covariates + the axis. Same model/features for live & regen -> the delta isolates
    the label change."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    frame = cov.copy()
    frame["axis"] = cfg.experienced(frame[cfg.axis_column])
    num = frame[_AUC_NUM].apply(pd.to_numeric, errors="coerce")
    feat = pd.get_dummies(pd.concat([num, frame[_AUC_CAT].astype(str)], axis=1), columns=_AUC_CAT)
    y = labels.to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    clf = RandomForestClassifier(n_estimators=120, min_samples_leaf=20, random_state=0)
    return float(cross_val_score(clf, feat, y, cv=3, scoring="roc_auc").mean())


def verify(regen: pd.DataFrame, live: pd.DataFrame, cfg: AxisConfig) -> None:
    m = regen.merge(live[[KEY, "persistent_180d"]], on=KEY, suffixes=("_gen", "_live"))
    viol = int((regen["persistent_180d"] + regen["discontinued_180d"] != 1).sum())
    logger.info("--- COMPLEMENT PROPERTY --- persist == 1-disc : %s (viol=%d)", viol == 0, viol)

    prev_gen = float(regen["persistent_180d"].mean())
    prev_live = float(live["persistent_180d"].mean())
    logger.info(
        "--- PREVALENCE --- regen persist=%.4f  live persist=%.4f  (delta %+.4f; disc band [0.05,0.60])",
        prev_gen,
        prev_live,
        prev_gen - prev_live,
    )

    def _rd(frame: pd.DataFrame, col: str) -> float:
        pr = frame.loc[frame["axis"] == 1, col].mean()
        cu = frame.loc[frame["axis"] == 0, col].mean()
        return float(pr - cu)

    logger.info(
        "--- AXIS RD (%s; persist: axis=1 - axis=0) --- regen=%.4f  attr=%.4f",
        cfg.axis_label,
        _rd(regen, "persistent_180d"),
        -float(regen["_axis_rd_attr"].iloc[0]),  # attr is on DISC; persist = -disc
    )
    live_ax = live.copy()
    live_ax["axis"] = cfg.experienced(live_ax[cfg.axis_column])
    logger.info(
        "    live naive RD=%.4f  ->  regen strengthens the (weak) live contrast",
        _rd(live_ax, "persistent_180d"),
    )

    auc_live = _proxy_auc(live, live["persistent_180d"], cfg)
    auc_regen = _proxy_auc(live, regen.set_index(KEY).loc[live[KEY], "persistent_180d"], cfg)
    logger.info(
        "--- PROXY LEAKAGE-SAFE AUC --- live=%.4f  regen=%.4f  (delta %+.4f; WS1 pins ~0.79-0.80, floor 0.78)",
        auc_live,
        auc_regen,
        auc_regen - auc_live,
    )

    flips = int((m["persistent_180d_gen"] != m["persistent_180d_live"]).sum())
    logger.info(
        "--- PER-ROW --- persistence flips vs live: %d/%d (%.1f%%)",
        flips,
        len(m),
        100.0 * flips / max(len(m), 1),
    )


def write_backup(live: pd.DataFrame, out_dir: Path, cfg: AxisConfig) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"{cfg.brand.lower()}_axis_persistence_backup_{ts}.tsv"
    cols = [c for c in (KEY, "brand", "persistent_180d", "discontinued_180d") if c in live.columns]
    live[cols].to_csv(path, sep="\t", index=False)
    logger.info("Wrote backup of %d live %s label rows to %s", len(live), cfg.brand, path)
    return path


def update_labels(client: Any, regen: pd.DataFrame, *, batch_size: int = BATCH_SIZE) -> int:
    written = 0
    for rec in regen[[KEY, "persistent_180d", "discontinued_180d"]].to_dict(orient="records"):
        client.table(TABLE).update(
            {
                "persistent_180d": int(rec["persistent_180d"]),
                "discontinued_180d": int(rec["discontinued_180d"]),
            }
        ).eq(KEY, rec[KEY]).execute()
        written += 1
        if written % batch_size == 0:
            logger.info("  updated %d/%d rows", written, len(regen))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--brand", required=True, choices=sorted(_AXES), help="Brand to backfill.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="WRITE PATH: UPDATE the brand's persistent_180d/discontinued_180d. "
        "Omit (default) for a read-only dry-run + backup + disproof.",
    )
    parser.add_argument(
        "--backup-dir", default=str(_PROJECT_ROOT / "data" / "backups"), help="Backup dir."
    )
    args = parser.parse_args()
    cfg = _AXES[args.brand]
    dry_run = not args.execute

    logger.info("=" * 70)
    logger.info(
        "%s axis persistence backfill  (%s)  axis=%s  seed=%d",
        cfg.brand,
        "DRY RUN" if dry_run else "EXECUTE",
        cfg.axis_column,
        SEED,
    )
    logger.info("=" * 70)

    try:
        from src.memory.services.factories import get_supabase_client

        client = get_supabase_client()
    except Exception as e:
        logger.error("No Supabase client (%s). Run from the prod box / API container.", e)
        return 1
    if client is None:
        logger.error("No Supabase client. Set SUPABASE_URL + a service key.")
        return 1

    live = fetch(client, cfg)
    if live is None or live.empty:
        logger.error("No live %s rows to backfill.", cfg.brand)
        return 1
    logger.info("Read %d %s patient rows.", len(live), cfg.brand)

    write_backup(live, Path(args.backup_dir), cfg)
    regen = regenerate(live, cfg)
    verify(regen, live, cfg)

    if dry_run:
        logger.info("DRY RUN complete. No rows written. Re-run with --execute to write.")
        return 0

    n = update_labels(client, regen)
    logger.info(
        "EXECUTE complete: updated %d %s rows in %s (idempotent on %s).", n, cfg.brand, TABLE, KEY
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
