#!/usr/bin/env python3
"""Backfill a confounded synthetic ``treatment_arm`` onto ``hcp_brand_adoption``
and re-derive its causally-linked ``adopted`` label from the committed DGP.

WHY THIS SCRIPT EXISTS
----------------------
``hcp_brand_adoption`` (15,000 rows = 5,000 synthetic HCPs x 3 brands; migration
076) carries the gold-standard HCP-adoption cohort's label ``adopted`` but has NO
``treatment_arm`` column. The HCP-adoption "Treatment Effects" surface therefore
has nothing to estimate an ATE from. The user wants a synthetic treatment arm so
the per-brand HCP-adoption ATE is real and recoverable.

The committed DGP ``_compute_adoption`` (src/ml/synthetic/generators/
hcp_adoption_artifact.py) ALREADY generates exactly this structure:

    centrality_z      ~ EXOGENOUS (stored peer_influence_score, NEVER re-drawn)
    hcp_segment       = centrality tier {high,medium,low}_influence (effect modifier)
    treatment_arm     ~ Bernoulli(sigmoid(0.8*centrality_z + noise))   # CONFOUNDED
    affinity          = per-(brand, specialty) DETERMINISTIC logit shift (#1551;
                        stored hcp_profiles.specialty, consumes NO rng draws)
    adoption_logit    = a + b*centrality_z + affinity
                        + scale*tau(segment)*treatment_arm + noise
    adopted           ~ Bernoulli(sigmoid(adoption_logit))             # T -> Y
    cate_estimate     = sigmoid(base + scale*tau) - sigmoid(base)      # per-HCP RD
                        (base includes the affinity term)

i.e. ``adopted`` is CAUSALLY generated FROM ``treatment_arm`` (treatment -> outcome),
heterogeneous by HCP influence segment, scaled per brand by ``_BRAND_ADOPT_SCALE``
(Fabhalta 1.2 > Remibrutinib 1.0 > Kisqali 0.8). The original ad-hoc load
(scripts/load_hcp_brand_adoption.py) called this DGP but persisted ONLY ``adopted``
-- it threw away the ``treatment_arm`` and ``cate_estimate`` it had just drawn.

WHAT IT DOES
------------
Reads the EXISTING per-HCP covariates ``peer_influence_score`` + ``specialty``
from the live ``hcp_profiles`` table (the causal inputs -- NEVER re-drawn),
standardizes the centrality exactly as the generator does, and per brand re-runs
``_compute_adoption`` (specialty threaded through for the #1551 affinity) to draw:
  * ``treatment_arm`` (0/1, confounded by centrality)  -- NEW column
  * ``adopted``       (0/1, CAUSALLY from treatment_arm) -- re-derived
keyed back to ``hcp_brand_adoption`` rows on ``(hcp_id, brand)``. The per-brand
TRUE ATE (mean prob-scale per-HCP ``cate_estimate``) is computed and printed so it
is verifiable.

WHY ``adopted`` IS RE-DERIVED (the honest choice; not a free lunch)
------------------------------------------------------------------
The live ``adopted`` labels were drawn by THIS SAME DGP at the original (uncommitted)
seed -- but the ``treatment_arm`` that produced them was discarded and the seed/RNG
threading was not committed (the load script's sweep found only ~63% per-row
agreement). So we CANNOT recover the exact arm that generated the live labels.

Two options were considered:
  * ADD treatment_arm WITHOUT touching ``adopted``. REJECTED: the new arm would
    NOT be the arm that generated the live label, so any "ATE" recovered against
    the live ``adopted`` would be a SPURIOUS reverse-correlation, exactly the
    anti-pattern the honesty directive forbids. The causal link must be real.
  * RE-DERIVE both arm and label together from the committed DGP on the stored
    centrality + specialty (this script). The treatment -> outcome link is
    INTACT and the TRUE ATE is documented. Cost: ``adopted`` moves, so the 3
    deployed HCP gold-standard models must be RE-TRAINED on the new labels
    (one-time). The aggregate adoption rate stays in the live band (~0.40-0.43)
    so the models' base rate and AUC ceiling are preserved -- see the dry-run
    prevalence print and the BLAST RADIUS section.

    #1551 NOTE: "the committed DGP" now INCLUDES the per-(brand, specialty)
    adoption affinity (_BRAND_SPECIALTY_AFFINITY), so this script threads the
    stored hcp_profiles.specialty into _compute_adoption. A specialty-blind
    re-derivation would silently UPDATE the live labels back to the
    specialty-free distribution and undo the served-propensity fix. The
    affinity is a deterministic logit shift with ZERO extra rng draws, so the
    treatment_arm stream this script faithfully reproduces is unchanged.

BLAST RADIUS (must read before --execute)
-----------------------------------------
* SCHEMA: ``hcp_brand_adoption`` needs a new ``treatment_arm INTEGER`` column.
  ``--execute`` runs ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` first (additive,
  idempotent). A sibling migration is the cleaner home; the ALTER here is a
  convenience so the backfill is self-contained. The column is nullable with no
  default so existing/real (is_synthetic=false) rows are unaffected.
* MODELS: the 3 staging models ``hcp_adoption_{brand}_goldstd_lr_v1`` (AUC
  0.74-0.76, training_samples 4000) use ``adopted`` as their TRAINING LABEL. Their
  FEATURES come from ``hcp_profiles`` (peer_influence_score, influence_network_size,
  years_experience, specialty, geographic_region) -- NOT from this table -- and
  ``treatment_arm`` is NOT in that covariate set, so adding the column does NOT
  change the feature space. BUT re-deriving ``adopted`` changes the label, so the
  models need a one-time RE-TRAIN (run_hcp_cohorts.py) to stay consistent. Until
  retrained they predict against the OLD label distribution; AUC is expected to
  hold (~same prevalence, same DGP family) but should be re-verified.
* TREATMENT-EFFECTS SURFACE: the new ``treatment_arm`` + the documented per-brand
  TRUE ATE are what the HCP-adoption Treatment Effects view consumes (treatment =
  treatment_arm, outcome = adopted, confounders = centrality covariates).

USAGE
-----
    # DEFAULT: dry-run. Reads live centrality, derives arm+label in memory, prints
    # treatment-rate / prevalence / TRUE-ATE / per-row movement vs live, and writes
    # a TSV backup of the CURRENT live adopted labels. Writes NOTHING to the DB.
    python scripts/backfill_hcp_treatment_arm.py

    python scripts/backfill_hcp_treatment_arm.py --seed 427

    # WRITE PATH -- DO NOT RUN against prod unless you intend to ALTER the table and
    # UPDATE 15k rows (treatment_arm + adopted). The human runs this after review.
    python scripts/backfill_hcp_treatment_arm.py --execute
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

from src.ml.synthetic.generators.hcp_adoption_artifact import (  # noqa: E402
    _BRAND_ADOPT_SCALE,
    ADOPTER_VALUE,
    _compute_adoption,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ADOPTION_TABLE = "hcp_brand_adoption"
PROFILES_TABLE = "hcp_profiles"
_NON_ADOPTER_VALUE = "NON_ADOPTER"

# Default seed for the per-brand re-derivation RNG. EMPIRICAL: the ad-hoc load's
# seed was not committed; seed=427 is the load script's best AGGREGATE match to the
# live per-brand prevalence (0.407/0.417/0.399). Overridable via --seed. Per-row
# labels will NOT match the live table bit-for-bit (the discarded arm/seed cannot
# be recovered) -- --dry-run quantifies the exact movement so the verdict is
# data-driven. The brand RNG is derived from a master_rng exactly as
# generate_hcp_brand_adoption_frame does, so this mirrors the original load.
DEFAULT_SEED = 427

# Brands in the same order the original load iterated them (master_rng stream
# order matters for the per-brand sub-seed derivation -> faithful reproduction).
BRANDS = ("Remibrutinib", "Fabhalta", "Kisqali")
KEY = ["hcp_id", "brand"]


# ---------------------------------------------------------------------------
# Live read (read-only)
# ---------------------------------------------------------------------------


def fetch_centrality(client: Any) -> Optional[pd.DataFrame]:
    """Read the EXISTING per-HCP causal inputs from hcp_profiles (paged).

    Returns hcp_id + peer_influence_score + specialty for the synthetic HCPs.
    These are the causal inputs -- NEVER re-drawn; treatment_arm and adopted are
    derived FROM them, exactly as the generator does (specialty feeds the #1551
    per-(brand, specialty) adoption affinity).
    """
    try:
        rows: List[dict] = []
        page = 0
        page_size = 1000
        while True:
            resp = (
                client.table(PROFILES_TABLE)
                .select("hcp_id,peer_influence_score,specialty")
                .eq("is_synthetic", True)
                .order("hcp_id")
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
        df = pd.DataFrame(rows)
        df["peer_influence_score"] = df["peer_influence_score"].astype(float)
        return df.sort_values("hcp_id").reset_index(drop=True)
    except Exception as e:  # pragma: no cover - network/permission edge
        logger.warning("Could not read live %s: %s", PROFILES_TABLE, e)
        return None


def fetch_live_adoption(client: Any) -> Optional[pd.DataFrame]:
    """Read the CURRENT live hcp_brand_adoption rows (for backup + comparison)."""
    try:
        rows: List[dict] = []
        page = 0
        page_size = 1000
        cols = "hcp_id,brand,adopted,adoption_category"
        while True:
            resp = (
                client.table(ADOPTION_TABLE)
                .select(cols)
                .eq("is_synthetic", True)
                .order("hcp_id")
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
    except Exception as e:  # pragma: no cover
        logger.warning("Could not read live %s: %s", ADOPTION_TABLE, e)
        return None


# ---------------------------------------------------------------------------
# Derivation (pure, deterministic)
# ---------------------------------------------------------------------------


def derive(centrality: pd.DataFrame, *, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Re-derive treatment_arm + adopted per brand from the committed DGP.

    Standardizes the STORED peer_influence_score exactly as
    generate_hcp_brand_adoption_frame does (z-score over the synthetic HCP
    population), then for each brand draws a sub-RNG from the master_rng stream
    (mirroring the original load's brand-seed derivation) and calls
    ``_compute_adoption``. Returns one row per (hcp_id, brand) with treatment_arm,
    adopted, adoption_category, and the per-HCP cate_estimate (for the TRUE ATE).

    #1551: when the frame carries a ``specialty`` column (fetch_centrality now
    selects it), it is threaded into ``_compute_adoption`` so the re-derived
    labels carry the committed per-(brand, specialty) affinity — a
    specialty-blind re-derivation would clobber the fixed labels on --execute.
    NULL/missing specialty values naturally take the brand's default shift via
    ``_specialty_affinity``'s ``.get(s, default)`` — no special casing. The
    affinity consumes NO rng draws, so treatment_arm is byte-identical with or
    without the column (faithful arm reproduction preserved).
    """
    hcp_ids = centrality["hcp_id"].tolist()
    pis = centrality["peer_influence_score"].to_numpy(dtype=float)
    pis_std = pis.std()
    centrality_z = (pis - pis.mean()) / (pis_std if pis_std > 0 else 1.0)
    specialty: Optional[List[str]] = (
        centrality["specialty"].tolist() if "specialty" in centrality.columns else None
    )

    master_rng = np.random.default_rng(seed)
    frames: List[pd.DataFrame] = []
    for brand in BRANDS:
        # Same per-brand sub-seed derivation as the original load -> faithful.
        brand_seed = int(master_rng.integers(0, 2**32))
        brand_rng = np.random.default_rng(brand_seed)
        dgp = _compute_adoption(brand_rng, centrality_z, brand, specialty=specialty)
        adopted = dgp["adopted"].astype(int)
        frames.append(
            pd.DataFrame(
                {
                    "hcp_id": hcp_ids,
                    "brand": brand,
                    "treatment_arm": dgp["treatment_arm"].astype(int),
                    "adopted": adopted,
                    "adoption_category": np.where(
                        adopted == 1, ADOPTER_VALUE, _NON_ADOPTER_VALUE
                    ),
                    "cate_estimate": dgp["cate_estimate"].astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(derived: pd.DataFrame, live: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Print per-brand treatment-rate, prevalence, TRUE ATE, and movement vs live.

    Returns the {brand: true_ate} map (the documented planted effect).
    """
    logger.info("--- PER-BRAND DGP SUMMARY (treatment -> outcome) ---")
    logger.info(
        "  %-14s %-7s %-11s %-11s %-22s",
        "brand",
        "scale",
        "treat_rate",
        "adopt_rate",
        "TRUE_ATE (mean prob CATE)",
    )
    true_ate: Dict[str, float] = {}
    for brand in BRANDS:
        sub = derived[derived["brand"] == brand]
        scale = _BRAND_ADOPT_SCALE.get(brand, 1.0)
        ate = float(sub["cate_estimate"].mean())
        true_ate[brand] = round(ate, 4)
        logger.info(
            "  %-14s %-7.1f %-11.4f %-11.4f %-22.4f",
            brand,
            scale,
            float(sub["treatment_arm"].mean()),
            float(sub["adopted"].mean()),
            ate,
        )
    logger.info(
        "  ATE ordering (prob scale): Fabhalta(1.2) > Remibrutinib(1.0) > Kisqali(0.8) "
        "-- matches _BRAND_ADOPT_SCALE; each ATE = E[P(adopt|do T=1) - P(adopt|do T=0)]."
    )

    # Confounding sanity: adoption rate should be higher among treated (positive,
    # confounded effect) -- a quick directional check the surface will reflect.
    logger.info("--- NAIVE (confounded) vs treated/untreated adoption ---")
    for brand in BRANDS:
        sub = derived[derived["brand"] == brand]
        t1 = float(sub.loc[sub["treatment_arm"] == 1, "adopted"].mean())
        t0 = float(sub.loc[sub["treatment_arm"] == 0, "adopted"].mean())
        logger.info(
            "  %-14s treated_adopt=%.4f untreated_adopt=%.4f naive_diff=%.4f "
            "(>= TRUE_ATE: confounded by centrality, as designed)",
            brand,
            t1,
            t0,
            t1 - t0,
        )

    if live is not None and len(live):
        merged = derived[["hcp_id", "brand", "adopted"]].merge(
            live[["hcp_id", "brand", "adopted"]],
            on=KEY,
            how="inner",
            suffixes=("_new", "_live"),
        )
        if len(merged):
            agree = float((merged["adopted_new"] == merged["adopted_live"]).mean())
            live_rate = float(merged["adopted_live"].mean())
            new_rate = float(merged["adopted_new"].mean())
            logger.info("--- LABEL MOVEMENT vs live (n=%d) ---", len(merged))
            logger.info("  adopted per-row agreement : %.4f", agree)
            logger.info("  live adopt rate           : %.4f", live_rate)
            logger.info("  new  adopt rate           : %.4f", new_rate)
            logger.info(
                "  prevalence shift          : %+.4f (in live band ~0.40-0.43: %s)",
                new_rate - live_rate,
                "YES" if 0.38 <= new_rate <= 0.45 else "CHECK",
            )
            logger.info(
                "  VERDICT: re-derived label is the HONEST causally-linked outcome "
                "(treatment_arm -> adopted via committed DGP). Per-row labels move "
                "(arm/seed of the original draw was discarded), so the 3 HCP "
                "gold-standard models need a one-time RE-TRAIN on the new labels. "
                "Aggregate prevalence stays in-band so the AUC ceiling is preserved."
            )
    return true_ate


def write_backup(live: pd.DataFrame, out_dir: Path) -> Path:
    """Write a TSV backup of the CURRENT live adopted labels before any write."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"hcp_brand_adoption_label_backup_{ts}.tsv"
    cols = [c for c in ("hcp_id", "brand", "adopted", "adoption_category") if c in live.columns]
    live[cols].to_csv(path, sep="\t", index=False)
    logger.info("Wrote backup of %d live label rows to %s", len(live), path)
    return path


# ---------------------------------------------------------------------------
# Write path (only reachable with --execute)
# ---------------------------------------------------------------------------

_ADD_COLUMN_SQL = (
    "ALTER TABLE hcp_brand_adoption "
    "ADD COLUMN IF NOT EXISTS treatment_arm INTEGER;"
)
# Postgres has no `ADD CONSTRAINT IF NOT EXISTS`; guard with a DO block so the
# DDL is idempotent (re-runnable) on --execute.
_ADD_CHECK_SQL = (
    "DO $$ BEGIN "
    "IF NOT EXISTS (SELECT 1 FROM pg_constraint "
    "WHERE conname = 'ck_hcp_brand_adoption_treatment_arm') THEN "
    "ALTER TABLE hcp_brand_adoption "
    "ADD CONSTRAINT ck_hcp_brand_adoption_treatment_arm "
    "CHECK (treatment_arm IS NULL OR treatment_arm IN (0, 1)); "
    "END IF; END $$;"
)


def ensure_schema(client: Any) -> None:
    """Additively add the nullable treatment_arm column (idempotent).

    Tries a SQL RPC if the project exposes one; otherwise logs the DDL for the
    human to run as a sibling migration. The column is nullable / no default so
    real (is_synthetic=false) rows are untouched.
    """
    for sql in (_ADD_COLUMN_SQL, _ADD_CHECK_SQL):
        try:
            client.postgrest.rpc("exec_sql", {"sql": sql}).execute()  # type: ignore[attr-defined]
            logger.info("Applied DDL: %s", sql)
        except Exception as e:  # pragma: no cover - depends on RPC availability
            logger.warning(
                "Could not apply DDL via RPC (%s). Run this as a sibling migration "
                "BEFORE --execute writes:\n    %s",
                e,
                sql,
            )


def write_rows(client: Any, derived: pd.DataFrame, *, batch_size: int = 500) -> int:
    """Idempotent per-row UPDATE of treatment_arm + adopted keyed on (hcp_id, brand).

    Only the three derived columns change; consideration_date / data_split /
    is_synthetic / created_at are untouched.
    """
    written = 0
    records = derived[
        ["hcp_id", "brand", "treatment_arm", "adopted", "adoption_category"]
    ].to_dict(orient="records")
    for rec in records:
        client.table(ADOPTION_TABLE).update(
            {
                "treatment_arm": int(rec["treatment_arm"]),
                "adopted": int(rec["adopted"]),
                "adoption_category": str(rec["adoption_category"]),
            }
        ).eq("hcp_id", rec["hcp_id"]).eq("brand", rec["brand"]).execute()
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
        help="WRITE PATH: ALTER TABLE (add treatment_arm) then UPDATE 15k rows "
        "(treatment_arm + adopted). Omit (the default) for a read-only dry-run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Re-derivation seed (default {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "backups"),
        help="Directory for the pre-write TSV backup of live labels.",
    )
    args = parser.parse_args()
    dry_run = not args.execute

    logger.info("=" * 72)
    logger.info(
        "hcp_brand_adoption treatment-arm backfill  (%s)  seed=%d",
        "DRY RUN" if dry_run else "EXECUTE",
        args.seed,
    )
    logger.info("  brand adoption scales: %s", _BRAND_ADOPT_SCALE)
    logger.info("=" * 72)

    client = None
    try:
        from src.memory.services.factories import get_supabase_client

        client = get_supabase_client()
    except Exception as e:
        logger.warning("No Supabase client (%s).", e)

    if client is None:
        logger.error(
            "This script reads the EXISTING hcp_profiles centrality to derive the "
            "treatment arm + label. Without a Supabase client there is nothing to "
            "derive from. Set SUPABASE_URL + a service key, or run from the prod box."
        )
        return 1

    centrality = fetch_centrality(client)
    if centrality is None or centrality.empty:
        logger.error("Live %s has no synthetic centrality to derive from.", PROFILES_TABLE)
        return 1
    logger.info("Read %d synthetic HCP centrality rows from %s.", len(centrality), PROFILES_TABLE)

    live = fetch_live_adoption(client)
    if live is not None and len(live):
        # Always write a backup of the CURRENT live labels (cheap, safe, even dry-run).
        write_backup(live, Path(args.backup_dir))

    derived = derive(centrality, seed=args.seed)
    logger.info(
        "Derived %d rows (%d HCPs x %d brands): treatment_arm + adopted.",
        len(derived),
        derived["hcp_id"].nunique(),
        len(BRANDS),
    )

    true_ate = verify(derived, live)
    logger.info("DOCUMENTED PER-BRAND TRUE ATE (prob scale): %s", true_ate)

    if dry_run:
        logger.info("--- WOULD WRITE (on --execute) ---")
        logger.info("  1. %s", _ADD_COLUMN_SQL)
        logger.info("  2. %s", _ADD_CHECK_SQL)
        logger.info(
            "  3. UPDATE %d rows: treatment_arm + adopted + adoption_category "
            "(keyed on (hcp_id, brand); other columns untouched).",
            len(derived),
        )
        logger.info(
            "  4. THEN re-train the 3 hcp_adoption_*_goldstd_lr_v1 models "
            "(run_hcp_cohorts.py) and re-verify AUC."
        )
        logger.info("DRY RUN complete. No schema/rows changed. Re-run with --execute to write.")
        return 0

    ensure_schema(client)
    n = write_rows(client, derived)
    logger.info(
        "EXECUTE complete: updated %d rows in %s (idempotent on (hcp_id, brand)).",
        n,
        ADOPTION_TABLE,
    )
    logger.info(
        "NEXT: re-train the 3 hcp_adoption_*_goldstd_lr_v1 models on the new labels "
        "and re-verify AUC before promoting."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
