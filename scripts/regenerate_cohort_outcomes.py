#!/usr/bin/env python3
"""Reproducible, idempotent regeneration of patient persistence/discontinuation labels.

WHY THIS SCRIPT EXISTS
----------------------
The synthetic ``patient_journeys`` table carries two 180-day outcome labels used by
the gold-standard persistence/discontinuation cohorts:
``persistent_180d`` and ``discontinued_180d`` (8,750 initiators among 25,000 rows).
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
academic_hcp, geographic_region, segment_assignment, brand) -- the causal inputs are
NOT re-drawn -- and re-applies ``generate_discontinuation_outcomes`` per brand
(``brand_cate_scale`` from ``_BRAND_CATE_SCALE``: Remi 1.0 / Fabhalta 0.7 /
Kisqali 1.4) to re-derive ``persistent_180d`` / ``discontinued_180d``. The UPDATE
is idempotent, keyed on ``patient_id``. ``persistent_180d == 1 - discontinued_180d``
by construction (no complement violations possible).

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
from src.ml.synthetic.dgp.treatment_arm import _BRAND_CATE_SCALE  # noqa: E402
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
    "persistent_180d",  # current live labels (for the backup + comparison)
    "discontinued_180d",
]


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

    for brand, sub in df.groupby("brand"):
        sub = sub.sort_values(KEY)
        scale = _BRAND_CATE_SCALE.get(_BRAND_ENUM.get(str(brand), Brand.REMIBRUTINIB), 1.0)
        rng = np.random.default_rng(seed)
        res = generate_discontinuation_outcomes(
            rng=rng,
            treatment_arm=sub["treatment_arm"].to_numpy(dtype=int),
            disease_severity=sub["disease_severity"].to_numpy(dtype=float),
            academic_hcp=sub["academic_hcp"].to_numpy(dtype=int),
            geographic_region=sub["geographic_region"].to_numpy(dtype=str),
            segment=sub["segment_assignment"].to_numpy(dtype=str),
            brand_cate_scale=float(scale),
        )
        out_persist.loc[sub.index] = res["persistent_180d"].astype(int)
        out_disc.loc[sub.index] = res["discontinued_180d"].astype(int)

    result = df[[KEY, "brand", "treatment_initiated"]].copy()
    result["persistent_180d"] = out_persist.astype(int).to_numpy()
    result["discontinued_180d"] = out_disc.astype(int).to_numpy()
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
    records = regen[[KEY, "persistent_180d", "discontinued_180d"]].to_dict(orient="records")
    for rec in records:
        client.table(TABLE).update(
            {
                "persistent_180d": int(rec["persistent_180d"]),
                "discontinued_180d": int(rec["discontinued_180d"]),
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
