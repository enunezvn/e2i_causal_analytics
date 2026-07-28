#!/usr/bin/env python3
"""Surgical, idempotent re-derivation of FABHALTA persistence labels to plant the
prior-C5-inhibitor differential (issue #1321 pilot) — Fabhalta rows ONLY.

WHY THIS SCRIPT EXISTS
----------------------
The Fabhalta pilot makes ``complement_inhibitor_status`` (prior C5-inhibitor
switch) a REAL, recoverable causal axis on persistence — the first brand-DISTINCT
variable in the gold standard. The committed DGP (cohort_outcomes.
generate_discontinuation_outcomes + priorc5_cate_modifier + the mean-centered
_PRIORC5_MAIN_PULL) plants it at synthetic-generation time, but the live droplet
is FROZEN (full reseed = disaster recovery only). Like regenerate_cohort_outcomes.py
(the live labels are only ~63% bit-reproducible — the ad-hoc boosted UPDATE's RNG
seed was never committed), this brings the EXISTING Fabhalta rows into line IN
PLACE: it reads their covariates (never re-drawn) and re-derives persistent_180d /
discontinued_180d from the committed DGP WITH the prior-C5 differential, keyed
deterministically so re-runs are idempotent.

DESIGN (mean-preserving, so the marginal is not disturbed)
----------------------------------------------------------
  * MAIN effect: prior-C5 -> more discontinuation, MEAN-CENTERED (_PRIORC5_MAIN_PULL
    * (c5 - prev)) so the Fabhalta persistence PREVALENCE is preserved while the
    prior-vs-current contrast becomes recoverable (~-0.11 RD).
  * CATE modifier: prior-C5 x treatment_arm, mean-preserving (priorc5_cate_modifier)
    — prior-C5-experienced respond less to iptacopan; population ATE unchanged.
  * copay_support + psp_enrolled are RE-PASSED so their planted persistence pulls
    (COMM-ARMS) are preserved, not clobbered.
Only Fabhalta rows are touched; every other brand is left byte-identical.

BLAST-RADIUS DISPROOF (printed by the default dry-run, BEFORE any write)
-----------------------------------------------------------------------
The dry-run re-derives in memory and reports, against the LIVE labels:
  * persistence prevalence (must stay ~0.56, in the [0.05,0.60] disc band);
  * prior-C5 RD (must be clearly negative & recoverable);
  * a leakage-safe proxy AUC (RandomForest 3-fold) on BOTH current-live and
    re-derived labels — the DELTA gauges the WS1 goldstd Fabhalta-persistence-AUC
    blast radius (the pin is ~0.804; a large drop is a STOP signal);
  * per-row flip rate + complement property.
It also writes a TSV backup of the current live Fabhalta labels. Writes NOTHING.

USAGE
-----
    # DEFAULT: dry-run (read-only + backup + disproof).  Run in the API container:
    docker exec e2i_api python scripts/backfill_fabhalta_priorc5_persistence.py

    # WRITE PATH -- UPDATE the ~8.6k Fabhalta persistence labels in the live table:
    docker exec e2i_api python scripts/backfill_fabhalta_priorc5_persistence.py --execute
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

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
BRAND = "Fabhalta"
KEY = "patient_id"
# Fixed seed on the patient_id-sorted Fabhalta partition -> deterministic &
# idempotent (re-runs write identical labels). Not required to bit-match the
# generator's frame-order spawn-key plant (that is for fresh loads/tests); this is
# the live-substrate analogue, exactly as backfill_biologic_ige.py is to the
# generator's biologic draw. 1321 = the issue number, for traceability.
SEED = 1321
BATCH_SIZE = 500

_COVARIATE_COLS = [
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
    "complement_inhibitor_status",  # the pilot axis (prior/current)
    "copay_support",
    "psp_enrolled",
    "persistent_180d",  # current live labels (backup + comparison)
    "discontinued_180d",
]

# Leakage-safe proxy-AUC features (base covariates + the pilot axis). Mirrors the
# cheapest-disproof harness; a comparable, NOT identical, signal to the WS1 goldstd
# eval — used only to bound the AUC blast radius (delta live-vs-regen).
_AUC_NUM = [
    "disease_severity",
    "academic_hcp",
    "age_at_diagnosis",
    "comorbidity_burden",
    "prior_therapy_lines",
    "copay_support",
    "psp_enrolled",
    "priorc5",
]
_AUC_CAT = ["geographic_region", "insurance_type"]


def fetch_fabhalta(client: Any) -> Optional[pd.DataFrame]:
    """Read all synthetic Fabhalta covariates + current labels (paged)."""
    try:
        rows: List[dict] = []
        page, page_size = 0, 1000
        while True:
            resp = (
                client.table(TABLE)
                .select(",".join(_COVARIATE_COLS))
                .eq("is_synthetic", True)
                .eq("brand", BRAND)
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


def regenerate(cov: pd.DataFrame, *, seed: int = SEED) -> pd.DataFrame:
    """Re-derive Fabhalta persistent_180d / discontinued_180d from the committed
    DGP WITH the prior-C5 differential. patient_id-sorted, single deterministic
    stream -> idempotent. copay/psp re-passed so their pulls survive."""
    df = cov.copy().sort_values(KEY).reset_index(drop=True)
    experienced = (df["complement_inhibitor_status"].astype("object") == "prior").astype(int)
    scale = float(_BRAND_CATE_SCALE.get(Brand.FABHALTA, 0.7))
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
        priorc5_experienced=experienced.to_numpy(dtype=int),
    )
    out = df[[KEY]].copy()
    out["persistent_180d"] = res["persistent_180d"].astype(int)
    out["discontinued_180d"] = res["discontinued_180d"].astype(int)
    out["priorc5"] = experienced.to_numpy()
    out["_priorc5_rd_attr"] = res["priorc5_persistent_rd"]
    return out


def _proxy_auc(cov: pd.DataFrame, labels: pd.Series) -> float:
    """RandomForest 3-fold leakage-safe proxy AUC of persistence over the base
    covariates + the pilot axis. Same model/features for live & regen -> the
    delta isolates the label change."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    frame = cov.copy()
    frame["priorc5"] = (frame["complement_inhibitor_status"].astype("object") == "prior").astype(
        int
    )
    num = frame[_AUC_NUM].apply(pd.to_numeric, errors="coerce")
    feat = pd.get_dummies(pd.concat([num, frame[_AUC_CAT].astype(str)], axis=1), columns=_AUC_CAT)
    y = labels.to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    clf = RandomForestClassifier(n_estimators=120, min_samples_leaf=20, random_state=0)
    return float(cross_val_score(clf, feat, y, cv=3, scoring="roc_auc").mean())


def verify(regen: pd.DataFrame, live: pd.DataFrame) -> None:
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

    # prior-C5 RD (recovered contrast), regen vs live.
    def _rd(frame: pd.DataFrame, col: str) -> float:
        pr = frame.loc[frame["priorc5"] == 1, col].mean()
        cu = frame.loc[frame["priorc5"] == 0, col].mean()
        return float(pr - cu)

    logger.info(
        "--- PRIOR-C5 RD (persist: prior - current) --- regen=%.4f  attr=%.4f",
        _rd(regen, "persistent_180d"),
        -float(regen["_priorc5_rd_attr"].iloc[0]),  # attr is on DISC; persist = -disc
    )
    live_pc = live.copy()
    live_pc["priorc5"] = (
        live_pc["complement_inhibitor_status"].astype("object") == "prior"
    ).astype(int)
    logger.info(
        "    live naive RD=%.4f  ->  regen strengthens the (weak) live contrast",
        _rd(live_pc.rename(columns={"persistent_180d": "persistent_180d"}), "persistent_180d"),
    )

    # Blast-radius: leakage-safe proxy AUC, live labels vs regen labels.
    auc_live = _proxy_auc(live, live["persistent_180d"])
    auc_regen = _proxy_auc(live, regen.set_index(KEY).loc[live[KEY], "persistent_180d"])
    logger.info(
        "--- PROXY LEAKAGE-SAFE AUC --- live=%.4f  regen=%.4f  (delta %+.4f; WS1 pin ~0.804, floor 0.78)",
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


def write_backup(live: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"fabhalta_priorc5_persistence_backup_{ts}.tsv"
    cols = [c for c in (KEY, "brand", "persistent_180d", "discontinued_180d") if c in live.columns]
    live[cols].to_csv(path, sep="\t", index=False)
    logger.info("Wrote backup of %d live Fabhalta label rows to %s", len(live), path)
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
    parser.add_argument(
        "--execute",
        action="store_true",
        help="WRITE PATH: UPDATE Fabhalta persistent_180d/discontinued_180d. "
        "Omit (default) for a read-only dry-run + backup + disproof.",
    )
    parser.add_argument(
        "--backup-dir", default=str(_PROJECT_ROOT / "data" / "backups"), help="Backup dir."
    )
    args = parser.parse_args()
    dry_run = not args.execute

    logger.info("=" * 70)
    logger.info(
        "Fabhalta prior-C5 persistence backfill  (%s)  seed=%d",
        "DRY RUN" if dry_run else "EXECUTE",
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

    live = fetch_fabhalta(client)
    if live is None or live.empty:
        logger.error("No live Fabhalta rows to backfill.")
        return 1
    logger.info("Read %d Fabhalta patient rows.", len(live))

    write_backup(live, Path(args.backup_dir))
    regen = regenerate(live)
    verify(regen, live)

    if dry_run:
        logger.info("DRY RUN complete. No rows written. Re-run with --execute to write.")
        return 0

    n = update_labels(client, regen)
    logger.info(
        "EXECUTE complete: updated %d Fabhalta rows in %s (idempotent on %s).", n, TABLE, KEY
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
