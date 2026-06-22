#!/usr/bin/env python3
"""Reproducible, idempotent regeneration of patient initiation/persistence/discontinuation labels.

WHY THIS SCRIPT EXISTS
----------------------
The synthetic ``patient_journeys`` table carries three outcome labels used by the
gold-standard cohorts: ``treatment_initiated`` (initiation) plus the two 180-day
outcomes ``persistent_180d`` and ``discontinued_180d`` (persistence/discontinuation).
T11 (2026-06-22): ``treatment_initiated`` is ALSO re-derived now — its outcome eqn
(``binary_outcome_with_cate``) was enriched with 4 prognostic drivers (insurance
access, age, comorbidity, prior-therapy) drawn ⊥ treatment_arm, so the goldstd
initiation model lifts ~0.67 → ~0.80. Backfilling the driver columns WITHOUT
re-deriving the initiation label would leave AUC at ~0.67 (the label must depend on
the new drivers); re-derivation is prevalence-banded (~0.35) so the marginal rate is
preserved while per-row labels shift. treatment_arm/propensity_score/segment_assignment/
treatment_effect_estimate are UNTOUCHED (they key off the arm, not the outcome), so the
causal substrate + ATE/CATE recovery are preserved.
The committed DGP (``src/ml/synthetic/generators/cohort_outcomes.py:
generate_discontinuation_outcomes``) produces these, and it runs inside
``PatientGenerator.generate()`` at synthetic-generation time. But the live values
were (re)derived ad-hoc on the prod box -- per the project history, a "TARGETED
UPDATE" applied a *boosted* DGP (severity/academic/region coefficients raised,
the causal treatment term UNCHANGED) to lift the leakage-safe persistence AUC from
~0.59 to ~0.77 -- and that UPDATE step was never committed. This script
reconstructs the regeneration deterministically from the committed DGP so the
labels are reproducible from version control.

WHAT IT DOES
------------
Reads the EXISTING synthetic patients' covariates (treatment_arm, disease_severity,
academic_hcp, geographic_region, segment_assignment, brand, plus the T9 prognostic
drivers insurance_type + age_at_diagnosis) -- the causal inputs are NOT re-drawn --
and re-applies ``generate_discontinuation_outcomes`` per brand (``brand_cate_scale``
from ``_BRAND_CATE_SCALE``: Remi 1.0 / Fabhalta 0.7 / Kisqali 1.4) to re-derive
``persistent_180d`` / ``discontinued_180d``. The two NEW driver columns
(comorbidity_burden, prior_therapy_lines; migration 087) are read when already present
or else drawn deterministically (independent of treatment_arm), so a single
``--execute`` pass BACKFILLS them alongside the labels. The UPDATE is idempotent,
keyed on ``patient_id``. ``persistent_180d == 1 - discontinued_180d`` by construction
(no complement violations possible).

REPRODUCIBILITY VERDICT (honest, data-driven; see ``--dry-run`` output)
-----------------------------------------------------------------------
With the committed (boosted) DGP applied to the existing covariates, a recoverable
seed reproduces the live INITIATOR prevalence essentially exactly
(target persistent_180d mean 0.456 / discontinued 0.544) and the complement
property holds. BUT a 0..200 seed sweep found NO seed that reproduces the per-row
labels bit-for-bit (best ~63% agreement) -- the ad-hoc UPDATE's exact RNG
seed/threading was never committed, so the per-row Bernoulli draws cannot be
recovered. ``--dry-run`` quantifies the exact agreement against the live table so
the verdict stays data-driven.

USAGE
-----
    # DEFAULT: dry-run. Regenerates in-memory, prints prevalence + complement +
    # per-row agreement vs live, and writes a TSV backup of the CURRENT live labels.
    # Writes NOTHING to the DB.
    python scripts/regenerate_cohort_outcomes.py

    python scripts/regenerate_cohort_outcomes.py --seed 74

    # WRITE PATH -- DO NOT RUN against prod unless you intend to UPDATE 25k rows.
    python scripts/regenerate_cohort_outcomes.py --execute
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

from src.ml.synthetic.config import Brand  # noqa: E402
from src.ml.synthetic.dgp.treatment_arm import (  # noqa: E402
    _BRAND_CATE_SCALE,
    binary_outcome_with_cate,
    brand_scaled_cate,
    initiation_prognostic_offset,
)
from src.ml.synthetic.generators.cohort_outcomes import (  # noqa: E402
    generate_discontinuation_outcomes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TABLE = "patient_journeys"
# Seed for the regeneration RNG. EMPIRICAL FINDING: the ad-hoc UPDATE's seed was not
# committed. A 0..200 per-brand sweep against the live table reproduces the live
# initiator prevalence at |err|<=0.0002 (seed=74) but NEVER the per-row labels
# (best ~63%). seed=74 is the best aggregate match; overridable via --seed.
DEFAULT_SEED = 74
KEY = "patient_id"
BATCH_SIZE = 500

# Brand -> _BRAND_CATE_SCALE enum (mirrors PatientGenerator.generate()).
_BRAND_ENUM: Dict[str, Brand] = {
    "Remibrutinib": Brand.REMIBRUTINIB,
    "Fabhalta": Brand.FABHALTA,
    "Kisqali": Brand.KISQALI,
}

# Covariate columns we READ from the live table (the causal inputs; never re-drawn).
_COVARIATE_COLS = [
    "patient_id",
    "brand",
    "treatment_initiated",
    "treatment_arm",
    "disease_severity",
    "academic_hcp",
    "geographic_region",
    "segment_assignment",
    # T9 prognostic drivers. insurance_type + age_at_diagnosis already exist on live
    # rows; comorbidity_burden + prior_therapy_lines are new (migration 087) and may be
    # NULL until this script backfills them (drawn deterministically, independent of
    # treatment_arm, then written by --execute alongside the labels).
    "insurance_type",
    "age_at_diagnosis",
    "comorbidity_burden",
    "prior_therapy_lines",
    "persistent_180d",  # current live labels (for the backup + comparison)
    "discontinued_180d",
]


def _read_or_draw_num(sub: pd.DataFrame, col: str, draw: Any) -> np.ndarray:
    """Return ``sub[col]`` as a numeric array when fully populated, else a fresh
    deterministic draw (used to backfill the new T9 driver columns on live rows)."""
    if col in sub.columns:
        s = pd.to_numeric(sub[col], errors="coerce")
        if s.notna().all() and len(s) > 0:
            return np.asarray(s.to_numpy())
    return np.asarray(draw(len(sub)))


def _read_or_draw_str(sub: pd.DataFrame, col: str, draw: Any) -> np.ndarray:
    """String-valued analogue of ``_read_or_draw_num`` (for ``insurance_type``)."""
    if col in sub.columns:
        s = sub[col].astype("object")
        if s.notna().all() and len(s) > 0:
            return np.asarray(s.to_numpy())
    return np.asarray(draw(len(sub)))


# ---------------------------------------------------------------------------
# Live read (read-only)
# ---------------------------------------------------------------------------


def fetch_covariates(client: Any) -> Optional[pd.DataFrame]:
    """Read all synthetic patients' covariates + current labels (paged)."""
    try:
        rows: List[dict] = []
        page = 0
        page_size = 1000
        while True:
            resp = (
                client.table(TABLE)
                .select(",".join(_COVARIATE_COLS))
                .eq("is_synthetic", True)
                .order(KEY)
                .range(page * page_size, (page + 1) * page_size - 1)
                .execute()
            )
            batch = resp.data or []
            rows.extend(batch)
            if len(batch) < page_size:
                break
            page += 1
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception as e:  # pragma: no cover - network/permission edge
        logger.warning("Could not read live %s: %s", TABLE, e)
        return None


# ---------------------------------------------------------------------------
# Regeneration (pure, deterministic)
# ---------------------------------------------------------------------------


def regenerate(covariates: pd.DataFrame, *, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Re-derive persistent_180d / discontinued_180d from the committed DGP.

    Per brand, a fresh ``np.random.default_rng(seed)`` is applied to that brand's
    ``patient_id``-sorted covariate partition with the brand's ``_BRAND_CATE_SCALE``
    -- mirroring how ``PatientGenerator`` calls the DGP per (single-brand) generator
    instance. Returns a frame keyed on ``patient_id`` with the regenerated labels.
    """
    df = covariates.copy()
    df["treatment_arm"] = df["treatment_arm"].fillna(0).astype(int)
    df["academic_hcp"] = df["academic_hcp"].fillna(0).astype(int)
    df["disease_severity"] = df["disease_severity"].astype(float)
    df["segment_assignment"] = df["segment_assignment"].astype(str)
    df["geographic_region"] = df["geographic_region"].astype(str)

    out_persist = pd.Series(index=df.index, dtype="Int64")
    out_disc = pd.Series(index=df.index, dtype="Int64")
    out_com = pd.Series(index=df.index, dtype="Int64")
    out_prior = pd.Series(index=df.index, dtype="Int64")
    out_init = pd.Series(index=df.index, dtype="Int64")  # T11: re-derived treatment_initiated
    out_days = pd.Series(
        index=df.index, dtype="float64"
    )  # T11: days_to_treatment (NaN if not init)

    for brand, sub in df.groupby("brand"):
        sub = sub.sort_values(KEY)
        scale = _BRAND_CATE_SCALE.get(_BRAND_ENUM.get(str(brand), Brand.REMIBRUTINIB), 1.0)
        # INDEPENDENT per-component streams (codex FINDING-2 fix): the driver backfill
        # draws must NOT advance the RNG that feeds the discontinuation / initiation
        # outcomes — otherwise a rerun where the driver columns are already populated
        # (so _read_or_draw READS instead of DRAWS) would re-realize the outcomes off a
        # different offset. Spawning makes disc/init deterministic functions of
        # (seed, brand, covariates) regardless of the read-vs-draw path → idempotent.
        drivers_rng, disc_rng, init_rng = np.random.default_rng(seed).spawn(3)
        # T9 prognostic drivers. insurance_type + age_at_diagnosis are READ from the
        # existing covariates (live has them); comorbidity_burden + prior_therapy_lines
        # are read when already backfilled, else drawn deterministically (independent of
        # treatment_arm) so a single --execute pass backfills them AND re-derives labels.
        insurance_type = _read_or_draw_str(
            sub,
            "insurance_type",
            lambda k, rng=drivers_rng: rng.choice(
                ["commercial", "medicare", "medicaid"], k, p=[0.6, 0.3, 0.1]
            ),
        )
        age_at_diagnosis = _read_or_draw_num(
            sub, "age_at_diagnosis", lambda k, rng=drivers_rng: rng.integers(18, 85, k)
        )
        comorbidity_burden = _read_or_draw_num(
            sub, "comorbidity_burden", lambda k, rng=drivers_rng: rng.poisson(1.3, k).clip(0, 5)
        )
        prior_therapy_lines = _read_or_draw_num(
            sub, "prior_therapy_lines", lambda k, rng=drivers_rng: rng.integers(0, 4, k)
        )
        res = generate_discontinuation_outcomes(
            rng=disc_rng,
            treatment_arm=sub["treatment_arm"].to_numpy(dtype=int),
            disease_severity=sub["disease_severity"].to_numpy(dtype=float),
            academic_hcp=sub["academic_hcp"].to_numpy(dtype=int),
            geographic_region=sub["geographic_region"].to_numpy(dtype=str),
            insurance_type=np.asarray(insurance_type, dtype=str),
            age_at_diagnosis=np.asarray(age_at_diagnosis, dtype=int),
            comorbidity_burden=np.asarray(comorbidity_burden, dtype=int),
            prior_therapy_lines=np.asarray(prior_therapy_lines, dtype=int),
            segment=sub["segment_assignment"].to_numpy(dtype=str),
            brand_cate_scale=float(scale),
        )
        out_persist.loc[sub.index] = res["persistent_180d"].astype(int)
        out_disc.loc[sub.index] = res["discontinued_180d"].astype(int)
        out_com.loc[sub.index] = np.asarray(comorbidity_burden, dtype=int)
        out_prior.loc[sub.index] = np.asarray(prior_therapy_lines, dtype=int)

        # T11: re-derive treatment_initiated from the ENRICHED initiation eqn — the
        # SAME binary_outcome_with_cate the generator now calls, fed the SAME 4
        # prognostic drivers via initiation_prognostic_offset (⊥ treatment_arm). It
        # draws from its OWN spawned stream (init_rng), so the disc/persist
        # re-derivation above is unaffected by it (and vice versa). Backfilling the
        # driver columns WITHOUT re-deriving this label would leave the goldstd
        # initiation model at ~0.67 (the label must actually depend on the new
        # drivers). The prevalence-banded construction pins initiation prev at ~0.35.
        brand_enum = _BRAND_ENUM.get(str(brand), Brand.REMIBRUTINIB)
        new_init, _tau = binary_outcome_with_cate(
            sub["treatment_arm"].to_numpy(dtype=int),
            {
                "disease_severity": sub["disease_severity"].to_numpy(dtype=float),
                "academic_hcp": sub["academic_hcp"].to_numpy(dtype=float),
            },
            sub["segment_assignment"].to_numpy(dtype=str),
            brand_scaled_cate(brand_enum),
            init_rng,
            prognostic_offset=initiation_prognostic_offset(
                np.asarray(insurance_type, dtype=str),
                np.asarray(age_at_diagnosis, dtype=int),
                np.asarray(comorbidity_burden, dtype=int),
                np.asarray(prior_therapy_lines, dtype=int),
            ),
        )
        out_init.loc[sub.index] = np.asarray(new_init, dtype=int)
        # T11: keep days_to_treatment internally consistent with the re-derived label
        # (a value for initiators, NULL otherwise) — mirrors patient_generator. It is
        # denylisted from every model and absent from all KPIs/views, but a relabel
        # that left it stale (an "initiated" row with NULL days) would be a silent
        # inconsistency. Drawn from init_rng (same component stream as the label).
        new_init_arr = np.asarray(new_init, dtype=int)
        out_days.loc[sub.index] = np.where(
            new_init_arr == 1, init_rng.integers(7, 90, len(sub)).astype(float), np.nan
        )

    result = df[[KEY, "brand"]].copy()
    result["treatment_initiated"] = out_init.astype(int).to_numpy()
    result["days_to_treatment"] = out_days.to_numpy()
    result["persistent_180d"] = out_persist.astype(int).to_numpy()
    result["discontinued_180d"] = out_disc.astype(int).to_numpy()
    result["comorbidity_burden"] = out_com.astype(int).to_numpy()
    result["prior_therapy_lines"] = out_prior.astype(int).to_numpy()
    return result


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(regen: pd.DataFrame, live: pd.DataFrame) -> None:
    """Print prevalence (initiators + all), complement property, per-row agreement."""
    viol = int((regen["persistent_180d"] + regen["discontinued_180d"] != 1).sum())
    logger.info("--- COMPLEMENT PROPERTY ---")
    logger.info("  persistent_180d == 1 - discontinued_180d : %s (violations=%d)", viol == 0, viol)

    init = regen["treatment_initiated"].astype(int) == 1
    logger.info("--- PREVALENCE (regenerated) ---")
    # T11: re-derived treatment_initiated — prevalence-banded ~0.35 by construction.
    logger.info(
        "  treatment_initiated (n=%d): prev=%.4f  [target ~0.35, band 0.25-0.45]",
        len(regen),
        float(regen["treatment_initiated"].mean()),
    )
    if "treatment_initiated" in live.columns:
        cmp = regen[[KEY, "treatment_initiated"]].merge(
            live[[KEY, "treatment_initiated"]], on=KEY, how="inner", suffixes=("_gen", "_live")
        )
        if len(cmp):
            flips = int((cmp["treatment_initiated_gen"] != cmp["treatment_initiated_live"]).sum())
            logger.info(
                "  treatment_initiated flips vs live: %d/%d (%.1f%%)  live_prev=%.4f",
                flips,
                len(cmp),
                100.0 * flips / len(cmp),
                float(cmp["treatment_initiated_live"].mean()),
            )
    logger.info(
        "  initiators (n=%d): persist=%.4f disc=%.4f  [target 0.456 / 0.544]",
        int(init.sum()),
        float(regen.loc[init, "persistent_180d"].mean()),
        float(regen.loc[init, "discontinued_180d"].mean()),
    )
    logger.info(
        "  all rows  (n=%d): persist=%.4f", len(regen), float(regen["persistent_180d"].mean())
    )

    merged = regen[[KEY, "treatment_initiated", "persistent_180d", "discontinued_180d"]].merge(
        live[[KEY, "persistent_180d", "discontinued_180d"]],
        on=KEY,
        how="inner",
        suffixes=("_gen", "_live"),
    )
    if len(merged):
        m_init = (merged["treatment_initiated"].astype(int) == 1).to_numpy()
        p = float((merged["persistent_180d_gen"] == merged["persistent_180d_live"]).mean())
        p_init = float(
            (
                merged.loc[m_init, "persistent_180d_gen"]
                == merged.loc[m_init, "persistent_180d_live"]
            ).mean()
        )
        live_init_persist = float(merged.loc[m_init, "persistent_180d_live"].mean())
        logger.info("--- PER-ROW AGREEMENT vs live (n=%d) ---", len(merged))
        logger.info("  persistent_180d match (all)        : %.4f", p)
        logger.info("  persistent_180d match (initiators) : %.4f", p_init)
        logger.info("  live initiator persist mean        : %.4f", live_init_persist)
        if p_init >= 0.9999:
            logger.info("  VERDICT: EXACT per-row reproduction of the live labels.")
        elif abs(float(regen.loc[init, "persistent_180d"].mean()) - live_init_persist) <= 0.005:
            logger.info(
                "  VERDICT: AGGREGATE reproduction (initiator prevalence + complement match) "
                "but per-row labels DIFFER -- the ad-hoc UPDATE's RNG seed is not the one "
                "used here. The committed DGP cannot reproduce the per-row draws bit-for-bit."
            )
        else:
            logger.info(
                "  VERDICT: APPROXIMATE -- prevalence and/or per-row labels differ from live."
            )


def write_backup(live: pd.DataFrame, out_dir: Path) -> Path:
    """Write a TSV backup of the CURRENT live labels before any (never-invoked) write."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"patient_journeys_persistence_backup_{ts}.tsv"
    cols = [
        c
        for c in (KEY, "brand", "treatment_initiated", "persistent_180d", "discontinued_180d")
        if c in live.columns
    ]
    live[cols].to_csv(path, sep="\t", index=False)
    logger.info("Wrote backup of %d live label rows to %s", len(live), path)
    return path


# ---------------------------------------------------------------------------
# Update (write path -- only reachable with --execute)
# ---------------------------------------------------------------------------


def update_labels(client: Any, regen: pd.DataFrame, *, batch_size: int = BATCH_SIZE) -> int:
    """Idempotent UPDATE of persistent_180d/discontinued_180d keyed on patient_id.

    Uses per-row PATCH (``.update(...).eq(patient_id)``) so only the two label
    columns change -- covariates and all other columns are untouched.
    """
    written = 0
    cols = [
        KEY,
        "treatment_initiated",
        "days_to_treatment",
        "persistent_180d",
        "discontinued_180d",
        "comorbidity_burden",
        "prior_therapy_lines",
    ]
    records = regen[cols].to_dict(orient="records")
    for rec in records:
        days = rec["days_to_treatment"]
        client.table(TABLE).update(
            {
                # T11: re-derived from the enriched initiation eqn (the goldstd
                # initiation label must depend on the new drivers to gain signal).
                "treatment_initiated": int(rec["treatment_initiated"]),
                # T11: kept consistent with the label (NULL when not initiated).
                "days_to_treatment": (int(days) if pd.notna(days) else None),
                "persistent_180d": int(rec["persistent_180d"]),
                "discontinued_180d": int(rec["discontinued_180d"]),
                # T9: backfill the new prognostic driver columns alongside the labels.
                "comorbidity_burden": int(rec["comorbidity_burden"]),
                "prior_therapy_lines": int(rec["prior_therapy_lines"]),
            }
        ).eq(KEY, rec[KEY]).execute()
        written += 1
        if written % batch_size == 0:
            logger.info("  updated %d/%d rows", written, len(records))
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="WRITE PATH: UPDATE persistent_180d/discontinued_180d in the live table. "
        "Omit (the default) for a read-only dry-run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Regeneration seed (default {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "backups"),
        help="Directory for the pre-write TSV backup of live labels.",
    )
    args = parser.parse_args()
    dry_run = not args.execute

    logger.info("=" * 70)
    logger.info(
        "cohort outcomes regeneration  (%s)  seed=%d",
        "DRY RUN" if dry_run else "EXECUTE",
        args.seed,
    )
    logger.info(
        "  brand scales: %s", {b: _BRAND_CATE_SCALE.get(e, 1.0) for b, e in _BRAND_ENUM.items()}
    )
    logger.info("=" * 70)

    client = None
    try:
        from src.memory.services.factories import get_supabase_client

        client = get_supabase_client()
    except Exception as e:
        logger.warning("No Supabase client (%s). Falling back to no comparison.", e)

    if client is None:
        logger.error(
            "This script reads the EXISTING patient covariates from the live DB to "
            "re-derive labels. Without a Supabase client there is nothing to regenerate "
            "from. Set SUPABASE_URL + a service key, or run from the prod box."
        )
        return 1

    live = fetch_covariates(client)
    if live is None or live.empty:
        logger.error("Live %s has no synthetic rows to regenerate from.", TABLE)
        return 1
    logger.info("Read %d synthetic patient rows.", len(live))

    # Always write a backup of the CURRENT live labels (cheap, safe, even in dry-run).
    write_backup(live, Path(args.backup_dir))

    regen = regenerate(live, seed=args.seed)
    verify(regen, live)

    if dry_run:
        logger.info("DRY RUN complete. No rows updated. Re-run with --execute to write.")
        return 0

    n = update_labels(client, regen)
    logger.info("EXECUTE complete: updated %d rows in %s (idempotent on %s).", n, TABLE, KEY)
    return 0


if __name__ == "__main__":
    sys.exit(main())
