#!/usr/bin/env python3
"""Idempotent loader for ``optum_biologic_persistence_causal`` (migration 148).

Lane A (spec docs/superpowers/specs/2026-09-22-real-data-causal-estimation-design.md
§3A.2): reads the causal export written by
``scripts/convert_optum_mart.py --cohort persistence_causal`` and upserts it on
``patient_id`` through the PostgREST client (the same path
``scripts/load_hcp_brand_adoption.py`` uses — the worker containers mount no
``data/rwd`` volume, so this runs on the host venv).

Fail-loud validation BEFORE any write: required columns present, every row
``is_synthetic == False``, ``patient_id`` unique, the treatment coding agrees with
``index_biologic_brand`` and only the observed two-arm contrast is present, every
outcome is 0/1. After ``--execute`` the live table's arm split, treatment-column
counts, and per-outcome positives are re-read and compared with the parquet; a
disagreement is printed as MISMATCH and the exit code is 1 — the load is never
reported as verified on the strength of the write call alone.

USAGE
-----
    # DEFAULT: dry run. Validates the parquet, prints the arm split it WOULD
    # write and (if reachable) the live table's current split. Writes nothing.
    # --dry-run is an explicit alias for this default (mutually exclusive with
    # --execute).
    python -m scripts.load_optum_causal_cohort --input data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet

    # WRITE PATH — owner-GO step (spec §7 records the GO for the production load).
    python -m scripts.load_optum_causal_cohort --input <parquet> --execute
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TABLE = "optum_biologic_persistence_causal"
ON_CONFLICT = "patient_id"
BATCH_SIZE = 500
DEFAULT_INPUT = "data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet"
TREATMENT = "treatment_dupixent"
BRAND = "index_biologic_brand"
ARMS = ("XOLAIR", "DUPIXENT")
OUTCOME_COLUMNS = (
    "persistent_at_180d_g28",
    "discontinued_180d",
    "biologic_switch_180d_flag",
    "persistent_at_180d",
)
REQUIRED_COLUMNS = (
    "patient_id",
    "patient_journey_id",
    "patient_hash",
    "index_date",
    "journey_start_date",
    BRAND,
    TREATMENT,
    "treatment_start_date",
    *OUTCOME_COLUMNS,
    "is_synthetic",
)


# ---------------------------------------------------------------------------
# Validation (pure)
# ---------------------------------------------------------------------------


def load_frame(path: Path | str) -> pd.DataFrame:
    """Read the export and refuse anything that is not the causal cohort contract."""
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"export is missing required column(s) {missing}")
    if df["is_synthetic"].astype(bool).any():
        n = int(df["is_synthetic"].astype(bool).sum())
        raise ValueError(
            f"{n} row(s) carry is_synthetic=True; the causal cohort is real claims data only"
        )
    dups = df["patient_id"].duplicated()
    if dups.any():
        raise ValueError(f"patient_id is not unique: {int(dups.sum())} duplicate(s)")
    off_contrast = ~df[BRAND].isin(ARMS)
    if off_contrast.any():
        raise ValueError(
            f"{int(off_contrast.sum())} row(s) have an index_biologic_brand outside {ARMS}: "
            f"{sorted(df.loc[off_contrast, BRAND].astype(str).unique())}"
        )
    expected = df[BRAND].eq("DUPIXENT").astype(int)
    if not df[TREATMENT].astype(int).eq(expected).all():
        raise ValueError(f"{TREATMENT} disagrees with index_biologic_brand on some rows")
    for col in OUTCOME_COLUMNS:
        values = set(pd.unique(df[col].dropna()))
        if not values <= {0, 1}:
            raise ValueError(f"{col} is not 0/1: found {sorted(values)}")
        if df[col].isna().any():
            raise ValueError(
                f"{col} has NULLs; the export fills the switch flag and derives the rest"
            )
    return df


def arm_split(df: pd.DataFrame) -> Dict[str, Any]:
    """Counts by arm, by the treatment column the causal run reads, and per-outcome
    positives by arm — the verification unit."""
    arms = {arm: int((df[BRAND] == arm).sum()) for arm in ARMS}
    positives = {
        col: {arm: int(df.loc[df[BRAND] == arm, col].astype(int).sum()) for arm in ARMS}
        for col in OUTCOME_COLUMNS
    }
    treatment_int = df[TREATMENT].astype(int)
    treatment = {str(v): int((treatment_int == v).sum()) for v in (0, 1)}
    return {"n": int(len(df)), "arms": arms, "outcome_positives": positives, "treatment": treatment}


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.date().isoformat() if not pd.isna(value) else None
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, float):
        return None if math.isnan(value) else value
    return value


def to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame -> JSON-safe upsert records (dates -> 'YYYY-MM-DD', NaN -> null,
    numpy scalars -> python). Deterministic for a given frame."""
    out: List[Dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        out.append({k: _json_safe(v) for k, v in rec.items()})
    return out


# ---------------------------------------------------------------------------
# DB (only reachable with a client)
# ---------------------------------------------------------------------------


def _client() -> Any:
    from src.memory.services.factories import get_supabase_client

    return get_supabase_client()


def upsert(client: Any, records: List[Dict[str, Any]], *, batch_size: int = BATCH_SIZE) -> int:
    """Batched idempotent upsert on ``patient_id``. Returns rows written."""
    written = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        client.table(TABLE).upsert(batch, on_conflict=ON_CONFLICT).execute()
        written += len(batch)
        logger.info("  upserted %d/%d rows", written, len(records))
    return written


def fetch_live_split(client: Any) -> Optional[Dict[str, Any]]:
    """The live table's arm split via exact counts. None when the table is unreachable
    (e.g. migration 148 not yet applied) — the caller reports that, never a zero."""
    try:
        arms: Dict[str, int] = {}
        positives: Dict[str, Dict[str, int]] = {col: {} for col in OUTCOME_COLUMNS}
        for arm in ARMS:
            arms[arm] = int(
                client.table(TABLE)
                .select("patient_id", count="exact")
                .eq(BRAND, arm)
                .execute()
                .count
            )
            for col in OUTCOME_COLUMNS:
                positives[col][arm] = int(
                    client.table(TABLE)
                    .select("patient_id", count="exact")
                    .eq(BRAND, arm)
                    .eq(col, 1)
                    .execute()
                    .count
                )
        treatment: Dict[str, int] = {}
        for v in (0, 1):
            treatment[str(v)] = int(
                client.table(TABLE)
                .select("patient_id", count="exact")
                .eq(TREATMENT, v)
                .execute()
                .count
            )
        total = int(client.table(TABLE).select("patient_id", count="exact").execute().count)
        return {"n": total, "arms": arms, "outcome_positives": positives, "treatment": treatment}
    except Exception as e:  # noqa: BLE001 — a missing relation / store hiccup is reported, not hidden
        logger.warning("Could not read live %s: %s", TABLE, e)
        return None


def verify(expected: Dict[str, Any], live: Dict[str, Any]) -> List[str]:
    """Every disagreement between the parquet split and the live split, as text."""
    problems: List[str] = []
    if expected["n"] != live["n"]:
        problems.append(f"n: parquet {expected['n']} vs live {live['n']}")
    for v in ("0", "1"):
        e, l = expected["treatment"][v], live["treatment"][v]
        if e != l:
            problems.append(f"treatment={v}: parquet {e} vs live {l}")
    for arm in ARMS:
        if expected["arms"][arm] != live["arms"][arm]:
            problems.append(
                f"arm {arm}: parquet {expected['arms'][arm]} vs live {live['arms'][arm]}"
            )
        for col in OUTCOME_COLUMNS:
            e, l = expected["outcome_positives"][col][arm], live["outcome_positives"][col][arm]
            if e != l:
                problems.append(f"{col} positives in {arm}: parquet {e} vs live {l}")
    return problems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _print_split(label: str, split: Dict[str, Any]) -> None:
    print(f"{label}: n={split['n']} arms={split['arms']}")
    for col in OUTCOME_COLUMNS:
        pos = split["outcome_positives"][col]
        rates = {
            arm: (round(pos[arm] / split["arms"][arm], 4) if split["arms"][arm] else None)
            for arm in ARMS
        }
        print(f"  {col}: positives={pos} rate={rates}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="causal export parquet")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--execute",
        action="store_true",
        help="WRITE PATH: upsert the rows into the live table. Omit (default) for a dry run.",
    )
    run_mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicit dry run (the default when neither flag is given). "
        "Mutually exclusive with --execute.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)
    dry_run = not args.execute

    print("=" * 70)
    print(f"{TABLE} loader ({'DRY RUN' if dry_run else 'EXECUTE'})  input={args.input}")
    print("=" * 70)
    df = load_frame(args.input)
    expected = arm_split(df)
    _print_split("WOULD WRITE" if dry_run else "WRITING", expected)

    client = None
    try:
        client = _client()
    except Exception as e:  # noqa: BLE001 — no client => dry-run still useful
        logger.warning("No Supabase client (%s).", e)

    live_before = fetch_live_split(client) if client is not None else None
    if live_before is None:
        print("LIVE TABLE: unreachable (migration 148 not applied, or no client)")
    else:
        _print_split("LIVE BEFORE", live_before)

    if dry_run:
        print("DRY RUN complete. No rows written. Re-run with --execute to write.")
        return 0
    if client is None:
        print("ERROR: cannot --execute without a Supabase client.")
        return 1

    n = upsert(client, to_records(df), batch_size=args.batch_size)
    print(f"EXECUTE: upserted {n} rows into {TABLE} (idempotent on {ON_CONFLICT}).")
    live_after = fetch_live_split(client)
    if live_after is None:
        print("MISMATCH: could not re-read the live table after the write.")
        return 1
    _print_split("LIVE AFTER", live_after)
    problems = verify(expected, live_after)
    if problems:
        print("MISMATCH between the parquet and the live table:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("VERIFIED: live arm split and per-outcome positives equal the parquet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
