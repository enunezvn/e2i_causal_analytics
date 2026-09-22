#!/usr/bin/env python3
"""Idempotent loader for ``optum_biologic_persistence_causal`` (migration 148).

Lane A (spec docs/superpowers/specs/2026-09-22-real-data-causal-estimation-design.md
§3A.2): reads the causal export written by
``scripts/convert_optum_mart.py --cohort persistence_causal`` and upserts it on
``patient_id`` through the PostgREST client (the same path
``scripts/load_hcp_brand_adoption.py`` uses — the worker containers mount no
``data/rwd`` volume, so this runs on the host venv).

Fail-loud validation BEFORE any write: the frame must carry EXACTLY the export's
81-column contract (7 journey-metadata keys + the primary outcome + the 64
``MART_SAFE_FEATURES`` baseline covariates + ``data_quality_score`` + the 6
``CAUSAL_EXTRA_COLS`` + ``is_synthetic`` + ``data_split`` -- no missing column, no
unexpected extra one, so a frame missing baseline confounders can never silently
upsert NULLs into them), every row ``is_synthetic == False``, ``patient_id``
unique, the treatment coding agrees with ``index_biologic_brand`` and only the
observed two-arm contrast is present, every outcome is 0/1. After ``--execute``
the live table's arm split, treatment-column counts, and per-outcome positives are
re-read and compared with the parquet, AND every exported field of every row is
re-read and compared patient-by-patient (``fetch_live_rows``/``verify_rows``) --
matching aggregate margins alone cannot hide wrong patient_ids or corrupted
covariates. A disagreement at either level is printed as MISMATCH and the exit
code is 1 — the load is never reported as verified on the strength of the write
call alone.

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

from scripts.convert_optum_mart import CAUSAL_EXTRA_COLS, TARGET_PERSISTENT_G28  # noqa: E402
from src.data.manifests import MART_SAFE_FEATURES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TABLE = "optum_biologic_persistence_causal"
ON_CONFLICT = "patient_id"
BATCH_SIZE = 500
ROW_PAGE_SIZE = 1000
DEFAULT_INPUT = "data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet"
TREATMENT = "treatment_dupixent"
BRAND = "index_biologic_brand"
ARMS = ("XOLAIR", "DUPIXENT")
OUTCOME_COLUMNS = (
    TARGET_PERSISTENT_G28,
    "discontinued_180d",
    "biologic_switch_180d_flag",
    "persistent_at_180d",
)
# The journey-metadata keys the converter emits for every cohort (not part of the
# owner-approved feature allow-list, not an outcome, not the treatment).
JOURNEY_METADATA_COLUMNS = (
    "patient_journey_id",
    "patient_id",
    "patient_hash",
    "index_date",
    "journey_start_date",
    "journey_status",
    "discontinuation_flag",
)
# The EXACT causal export contract (81 columns), derived from the converter's own
# constants rather than hand-listed, so this loader cannot drift from what
# ``scripts/convert_optum_mart.py --cohort persistence_causal`` actually emits:
# 7 journey-metadata keys + the primary outcome + the 64 owner-approved pre-index
# baseline features (``MART_SAFE_FEATURES``, includes geographic_region and
# enrollment_duration_days) + data_quality_score + the 6 CAUSAL_EXTRA_COLS
# (treatment + brand + the 3 remaining outcomes) + is_synthetic + data_split.
# ``load_frame`` refuses BOTH a missing column (e.g. a dropped baseline
# confounder, which would otherwise upsert as a silent NULL) and an unexpected
# extra one.
REQUIRED_COLUMNS = (
    *JOURNEY_METADATA_COLUMNS,
    TARGET_PERSISTENT_G28,
    *MART_SAFE_FEATURES,
    "data_quality_score",
    *CAUSAL_EXTRA_COLS,
    "is_synthetic",
    "data_split",
)


# ---------------------------------------------------------------------------
# Validation (pure)
# ---------------------------------------------------------------------------


def load_frame(path: Path | str) -> pd.DataFrame:
    """Read the export and refuse anything that is not EXACTLY the causal cohort
    contract -- missing OR unexpected extra columns, so a frame lacking baseline
    confounders can never silently upsert NULLs into them."""
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"export is missing required column(s) {missing}")
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra:
        raise ValueError(f"export has unexpected column(s) not in the causal contract: {extra}")
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


def fetch_live_rows(client: Any) -> Optional[List[Dict[str, Any]]]:
    """Page the whole live table back, ``ROW_PAGE_SIZE`` rows at a time, until a
    short page. None when the table is unreachable — never a partial or empty
    result mistaken for the truth (mirrors ``fetch_live_split``'s honesty)."""
    try:
        rows: List[Dict[str, Any]] = []
        start = 0
        while True:
            page = (
                client.table(TABLE)
                .select("*")
                .range(start, start + ROW_PAGE_SIZE - 1)
                .execute()
                .data
            )
            rows.extend(page)
            if len(page) < ROW_PAGE_SIZE:
                break
            start += ROW_PAGE_SIZE
        return rows
    except Exception as e:  # noqa: BLE001 — a missing relation / store hiccup is reported, not hidden
        logger.warning("Could not read live %s rows: %s", TABLE, e)
        return None


def _same(a: Any, b: Any) -> bool:
    """Value equality across the JSON round-trip: PostgREST returns NUMERIC/INTEGER
    columns as JSON numbers, DATE as 'YYYY-MM-DD' strings, and BOOLEAN as bools --
    none of which necessarily share a Python type with the exported record."""
    if a is None and b is None:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


_LIVE_ONLY_COLUMNS = ("created_at", "updated_at")


def verify_rows(
    expected_records: List[Dict[str, Any]], live_rows: List[Dict[str, Any]]
) -> List[str]:
    """Row-level diff keyed on ``patient_id``: missing/extra ids (counts + first 5)
    and, for every exported field of every row present in both, a value mismatch
    (first 10 reported, plus a total). Aggregate margins (``verify``) can match
    while patient_ids or covariates are wrong; this is the check that cannot be
    fooled that way. Live-only bookkeeping columns (created_at/updated_at) are
    never compared -- the export never emits them."""
    problems: List[str] = []
    expected_by_id = {r["patient_id"]: r for r in expected_records}
    live_by_id = {
        r["patient_id"]: {k: v for k, v in r.items() if k not in _LIVE_ONLY_COLUMNS}
        for r in live_rows
    }
    expected_ids, live_ids = set(expected_by_id), set(live_by_id)

    missing_ids = sorted(expected_ids - live_ids)
    if missing_ids:
        problems.append(
            f"{len(missing_ids)} patient_id(s) in the export missing from the live table "
            f"(first 5): {missing_ids[:5]}"
        )
    extra_ids = sorted(live_ids - expected_ids)
    if extra_ids:
        problems.append(
            f"{len(extra_ids)} patient_id(s) in the live table not present in the export "
            f"(first 5): {extra_ids[:5]}"
        )

    mismatches: List[str] = []
    for pid in sorted(expected_ids & live_ids):
        exp_row, live_row = expected_by_id[pid], live_by_id[pid]
        for field, exp_val in exp_row.items():
            if not _same(exp_val, live_row.get(field)):
                mismatches.append(
                    f"{pid}.{field}: parquet {exp_val!r} vs live {live_row.get(field)!r}"
                )
    if mismatches:
        problems.append(f"{len(mismatches)} field value mismatch(es) (first 10 shown):")
        problems.extend(mismatches[:10])
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

    expected_records = to_records(df)
    n = upsert(client, expected_records, batch_size=args.batch_size)
    print(f"EXECUTE: upserted {n} rows into {TABLE} (idempotent on {ON_CONFLICT}).")

    live_after = fetch_live_split(client)
    if live_after is None:
        print("MISMATCH: could not re-read the live table after the write.")
        return 1
    _print_split("LIVE AFTER", live_after)
    problems = verify(expected, live_after)

    live_rows = fetch_live_rows(client)
    if live_rows is None:
        print("MISMATCH: could not re-read the live table's rows for row-level verification.")
        return 1
    row_problems = verify_rows(expected_records, live_rows)
    print(
        f"ROW VERIFICATION: compared {len(expected_records)} exported rows against {len(live_rows)} live rows."
    )

    all_problems = problems + row_problems
    if all_problems:
        print("MISMATCH between the parquet and the live table:")
        for p in all_problems:
            print(f"  - {p}")
        return 1
    print(
        "VERIFIED: live arm split, treatment counts, per-outcome positives, and "
        "every exported field of every row equal the parquet."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
