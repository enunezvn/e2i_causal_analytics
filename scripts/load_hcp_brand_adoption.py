#!/usr/bin/env python3
"""Reproducible, idempotent loader for the ``hcp_brand_adoption`` table (migration 076).

WHY THIS SCRIPT EXISTS
----------------------
The gold-standard model-eval suite (``src/mlops/gold_standard_eval/``) serves a
4th HCP-grain cohort: brand adoption x 3 brands. Its source table,
``hcp_brand_adoption`` (15,000 rows = 5,000 synthetic HCPs x 3 brands), was
populated ad-hoc on the prod box: the *generator* and the *migration* are
committed, but the *load step* (regenerate the HCP frame -> call the generator ->
upsert) was never committed. This script reconstructs exactly that load step so
the table is reproducible from version control.

DETERMINISM CHAIN (all committed):
  1. ``HCPGenerator(seed=42, id_prefix="scv", n_records=5000).generate()``
     reproduces the live ``hcp_profiles`` synthetic HCPs EXACTLY -- verified:
     all 5,000 ``hcp_id`` (scvhcp_00000..scvhcp_04999) AND
     ``influence_network_size`` match the live table digit-for-digit. This is the
     same seed/tag ``scripts/load_synthetic_data.py`` uses (seed=42, --tag scv).
  2. ``generate_hcp_brand_adoption_frame(hcp_df, seed=<ADOPTION_SEED>,
     end_date=date(2026, 6, 1), n_months=37)`` derives one row per (hcp_id, brand)
     with the leakage-safe ``_compute_adoption`` DGP. consideration_date spans the
     trailing 37 monthly buckets 2023-06-01..2026-06-01.
  3. Upsert on the natural key ``(hcp_id, brand)`` (table constraint
     ``uq_hcp_brand_adoption``) with ``is_synthetic=True`` so a re-run is a no-op.

REPRODUCIBILITY VERDICT (see ``--dry-run`` output and the script's module
docstring in the delivery report): with the committed generator + seed=42 HCPs,
the live table's row COUNT, per-brand PREVALENCE band, MONTH RANGE, and SPLIT
ratios are reproduced; whether the per-row ``adopted`` labels match bit-for-bit
depends on the adoption-frame seed, which was NOT committed by the ad-hoc op.
``--dry-run`` quantifies the exact agreement against the live table so the verdict
is data-driven, never assumed.

USAGE
-----
    # DEFAULT: dry-run. Generates in-memory, prints what it WOULD write, and
    # (if a DB is reachable) a full comparison vs the live table. Writes nothing.
    python scripts/load_hcp_brand_adoption.py

    # Explicit adoption-frame seed (sweep/override):
    python scripts/load_hcp_brand_adoption.py --adoption-seed 427

    # WRITE PATH -- DO NOT RUN against prod unless you intend to upsert 15k rows.
    python scripts/load_hcp_brand_adoption.py --execute
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path so ``src`` imports resolve when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

from src.ml.synthetic.generators import GeneratorConfig, HCPGenerator  # noqa: E402
from src.ml.synthetic.generators.hcp_brand_adoption_generator import (  # noqa: E402
    generate_hcp_brand_adoption_frame,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Reproduction parameters (committed; match load_synthetic_data.py + migration 076) ---
TABLE = "hcp_brand_adoption"
N_HCPS = 5000
HCP_SEED = 42  # PROVEN: reproduces live hcp_profiles ids + influence_network_size exactly
HCP_ID_PREFIX = "scv"  # CSU id-namespace tag (project convention)
BRANDS: Tuple[str, ...] = ("Remibrutinib", "Fabhalta", "Kisqali")
N_MONTHS = 37  # live consideration_date spans 37 monthly buckets
END_DATE = date(2026, 6, 1)  # live max(consideration_date) == 2026-06-01
# Adoption-frame seed. EMPIRICAL FINDING: the ad-hoc load's seed was not committed.
# A 0..1000 sweep against the live table (using the seed=42 HCP cohort + the 2dp DB
# peer_influence_score the loader read) found NO seed that reproduces the per-row
# ``adopted`` labels bit-for-bit (best ~63% across all brand orders x pis sources).
# seed=427 reproduces the live AGGREGATE most faithfully: per-brand prevalence
# 0.407/0.417/0.399 (L1 err 0.0006) + correct counts/month-range/splits. Overridable
# via --adoption-seed; --dry-run quantifies the exact agreement against live.
DEFAULT_ADOPTION_SEED = 427  # best aggregate match to live (see note above)
ON_CONFLICT = "hcp_id,brand"  # uq_hcp_brand_adoption UNIQUE(hcp_id, brand)
BATCH_SIZE = 500  # mirrors load_synthetic_data.py LoaderConfig.batch_size

# Columns the DB table carries (excluding server-defaulted id/created_at/updated_at).
_LOAD_COLUMNS = [
    "hcp_id",
    "brand",
    "consideration_date",
    "adopted",
    "adoption_category",
    "data_split",
    "is_synthetic",
]


# ---------------------------------------------------------------------------
# Generation (pure, deterministic, no DB)
# ---------------------------------------------------------------------------


def build_hcp_frame(seed: int = HCP_SEED, n_hcps: int = N_HCPS) -> pd.DataFrame:
    """Regenerate the synthetic HCP cohort that the adoption frame joins to.

    Uses the same generator/seed/tag as ``scripts/load_synthetic_data.py`` so the
    in-memory ``peer_influence_score`` reproduces the values fed to the adoption
    DGP at the original (ad-hoc) load time.
    """
    cfg = GeneratorConfig(id_prefix=HCP_ID_PREFIX, seed=seed, n_records=n_hcps)
    return HCPGenerator(cfg).generate()


def build_adoption_frame(
    *,
    adoption_seed: int = DEFAULT_ADOPTION_SEED,
    hcp_seed: int = HCP_SEED,
    end_date: date = END_DATE,
    n_months: int = N_MONTHS,
) -> pd.DataFrame:
    """Deterministically build the full ``hcp_brand_adoption`` frame (15k rows)."""
    hcp_df = build_hcp_frame(seed=hcp_seed)
    frame = generate_hcp_brand_adoption_frame(
        hcp_df,
        seed=adoption_seed,
        end_date=end_date,
        brands=BRANDS,
        n_months=n_months,
    )
    # is_synthetic is already True from the generator; re-affirm for safety.
    frame["is_synthetic"] = True
    return frame[_LOAD_COLUMNS].copy()


# ---------------------------------------------------------------------------
# Live comparison (read-only)
# ---------------------------------------------------------------------------


def _fetch_live(client: Any) -> Optional[pd.DataFrame]:
    """Read the full live ``hcp_brand_adoption`` table (paged). None on failure."""
    try:
        rows: List[dict] = []
        page = 0
        page_size = 1000
        while True:
            resp = (
                client.table(TABLE)
                .select("hcp_id,brand,consideration_date,adopted,adoption_category,data_split")
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
        df["consideration_date"] = df["consideration_date"].astype(str).str.slice(0, 10)
        return df
    except Exception as e:  # pragma: no cover - network/permission edge
        logger.warning("Could not read live %s: %s", TABLE, e)
        return None


def _summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute the comparison summary (counts, per-brand prevalence, splits, months)."""
    g = df.groupby("brand")["adopted"].agg(["count", "mean"])
    return {
        "total": int(len(df)),
        "n_hcp": int(df["hcp_id"].nunique()),
        "per_brand": {
            b: (int(g.loc[b, "count"]), round(float(g.loc[b, "mean"]), 3)) for b in g.index
        },
        "splits": {k: int(v) for k, v in df["data_split"].value_counts().items()},
        "month_min": str(df["consideration_date"].min()),
        "month_max": str(df["consideration_date"].max()),
        "n_buckets": int(df["consideration_date"].nunique()),
    }


def compare_vs_live(gen: pd.DataFrame, live: pd.DataFrame) -> None:
    """Print an aggregate + per-row agreement comparison gen-vs-live."""
    logger.info("--- AGGREGATE COMPARISON (generated vs live) ---")
    gs, ls = _summary(gen), _summary(live)
    for k in ("total", "n_hcp", "per_brand", "splits", "month_min", "month_max", "n_buckets"):
        flag = "OK" if gs[k] == ls[k] else "DIFF"
        logger.info("  %-11s generated=%s | live=%s  [%s]", k, gs[k], ls[k], flag)

    # Per-row agreement on the natural key (hcp_id, brand).
    key = ["hcp_id", "brand"]
    merged = gen.merge(live, on=key, how="inner", suffixes=("_gen", "_live"))
    if len(merged):
        adop = float((merged["adopted_gen"] == merged["adopted_live"]).mean())
        cd = float(
            (
                merged["consideration_date_gen"].astype(str) == merged["consideration_date_live"]
            ).mean()
        )
        sp = float((merged["data_split_gen"] == merged["data_split_live"]).mean())
        logger.info("--- PER-ROW AGREEMENT (inner-joined on %s, n=%d) ---", key, len(merged))
        logger.info("  adopted match        : %.4f", adop)
        logger.info("  consideration_date   : %.4f", cd)
        logger.info("  data_split match     : %.4f", sp)
        if adop >= 0.9999 and cd >= 0.9999 and sp >= 0.9999:
            logger.info("  VERDICT: EXACT per-row reproduction of the live table.")
        elif gs["total"] == ls["total"] and gs["per_brand"] == ls["per_brand"]:
            logger.info(
                "  VERDICT: AGGREGATE reproduction (counts/prevalence/splits/months match) "
                "but per-row labels DIFFER -- the ad-hoc load's adoption-frame seed is not "
                "the one used here. Try --adoption-seed sweep; see delivery note."
            )
        else:
            logger.info(
                "  VERDICT: APPROXIMATE -- aggregates and/or per-row labels differ from live."
            )


# ---------------------------------------------------------------------------
# Upsert (write path -- only reachable with --execute)
# ---------------------------------------------------------------------------


def _records(df: pd.DataFrame) -> List[dict]:
    """DataFrame -> JSON-safe upsert records (dates -> ISO str, int coercion)."""
    out: List[dict] = []
    for rec in df.to_dict(orient="records"):
        clean: dict = {}
        for k, v in rec.items():
            if isinstance(v, (date,)):
                clean[k] = v.isoformat()
            elif isinstance(v, np.generic):
                clean[k] = v.item()
            elif isinstance(v, float) and float(v).is_integer():
                clean[k] = int(v)
            else:
                clean[k] = v
        out.append(clean)
    return out


def upsert(client: Any, df: pd.DataFrame, *, batch_size: int = BATCH_SIZE) -> int:
    """Idempotent batched upsert on (hcp_id, brand). Returns rows written."""
    records = _records(df)
    written = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        client.table(TABLE).upsert(batch, on_conflict=ON_CONFLICT).execute()
        written += len(batch)
        logger.info("  upserted %d/%d rows", written, len(records))
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
        help="WRITE PATH: upsert the generated rows into the live table. "
        "Omit (the default) for a read-only dry-run.",
    )
    parser.add_argument(
        "--adoption-seed",
        type=int,
        default=DEFAULT_ADOPTION_SEED,
        help=f"Seed for the adoption frame (default {DEFAULT_ADOPTION_SEED}).",
    )
    args = parser.parse_args()
    dry_run = not args.execute

    logger.info("=" * 70)
    logger.info("hcp_brand_adoption loader  (%s)", "DRY RUN" if dry_run else "EXECUTE")
    logger.info("  hcp_seed=%d tag=%s n_hcps=%d", HCP_SEED, HCP_ID_PREFIX, N_HCPS)
    logger.info(
        "  adoption_seed=%d end_date=%s n_months=%d brands=%s",
        args.adoption_seed,
        END_DATE,
        N_MONTHS,
        BRANDS,
    )
    logger.info("=" * 70)

    gen = build_adoption_frame(adoption_seed=args.adoption_seed)
    logger.info(
        "Generated %d rows (%d HCPs x %d brands).", len(gen), gen["hcp_id"].nunique(), len(BRANDS)
    )
    gs = _summary(gen)
    logger.info(
        "WOULD WRITE: total=%d per_brand=%s splits=%s months=%s..%s (%d buckets)",
        gs["total"],
        gs["per_brand"],
        gs["splits"],
        gs["month_min"],
        gs["month_max"],
        gs["n_buckets"],
    )

    # Resolve a client for the live comparison / write.
    client = None
    try:
        from src.memory.services.factories import get_supabase_client

        client = get_supabase_client()
    except Exception as e:
        logger.warning("No Supabase client (%s). Comparison/write skipped.", e)

    if client is not None:
        live = _fetch_live(client)
        if live is not None and len(live):
            compare_vs_live(gen, live)
        elif live is not None:
            logger.info("Live %s is EMPTY -- nothing to compare against.", TABLE)

    if dry_run:
        logger.info("DRY RUN complete. No rows written. Re-run with --execute to write.")
        return 0

    if client is None:
        logger.error("Cannot --execute without a Supabase client.")
        return 1
    n = upsert(client, gen)
    logger.info(
        "EXECUTE complete: upserted %d rows into %s (idempotent on %s).", n, TABLE, ON_CONFLICT
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
