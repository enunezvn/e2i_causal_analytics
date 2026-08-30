#!/usr/bin/env python
"""Reseed the business_metrics AGGREGATE rows in place under the current DGP (#1833).

WHY: the frozen base (2013-01..2026-07, 9,780 ``metric_<12hex>`` rows loaded
2026-07-03) and the monthly frontier cohorts (``m2608_NNNN``, ...) were emitted
by the pre-#1833 generator, whose only regional structure was the market-size
factor on both value and target — no brand-specific geography. The Mon-3AM
cron (``scripts/reseed_synthetic.sh`` -> ``--append-frontier``) regenerates
ONLY the cohort months from ``BM_EPOCH``; the base never regenerates, so
after deploying the brand x region DGP the base would stay on the old formula
while cohorts moved to the new one. This script brings every aggregate row
onto the new formula at once.

HOW (Step 0 of #1833, measured 2026-08-30): the base identity
(``frontier_append.base_business_metrics_frame``: seed 42, n=10000, start
pinned to 2013-01-01) reproduces all 9,780 DB base rows byte-for-byte on 16
columns, and ``generate_month_cohort(2026-08-01)`` reproduces the 60 ``m2608``
rows likewise. The #1833 terms are value-only and consume no RNG, so ids,
dates, targets and every other column are unchanged and the reseed is an
IN-PLACE UPSERT ON ``metric_id`` (the PK) through the loader's own
``BatchLoader.load_table`` path — the same path the cron uses. Nothing is
deleted; the 12,078 ``per_hcp_rollup`` rows (``metric_name`` NULL) are never
touched because only regenerated ids are written.

DEFAULT is ``--dry-run``: reads the live aggregate rows, regenerates the frame,
and prints the diff summary (rows to upsert, id-set drift, values changed /
unchanged, per-brand national TRx scale before -> after). ``--execute``
performs the upsert and then re-reads the rows to verify. The execute path
FAILS CLOSED on id drift in either direction (see ``execute_refusal``): stale
DB ids need ``--allow-id-drift`` (the identity assumption no longer holds for
them — consider the issue's delete+reinsert path); regenerated cohort ids the
Mon-3AM cron has not appended yet need ``--allow-new-cohorts`` (or run after
the cron / with an earlier ``--frontier``); target drift is never allowed.

Usage (from the checkout root, with the project env)::

    .venv/bin/dotenv run -- env PYTHONPATH=. .venv/bin/python \\
        scripts/reseed_business_metrics_aggregate.py            # dry-run
    .venv/bin/dotenv run -- env PYTHONPATH=. .venv/bin/python \\
        scripts/reseed_business_metrics_aggregate.py --execute

After ``--execute``: re-run ``POST /gaps/analyze`` per brand (persisted
``gap_analyses`` rows and the /gap-analysis page read those, not the metrics
table directly). kpi_history needs no refresh — the only business_metrics KPI
(WS3-BI-010 ROI) reads the ``roi`` column, which is an RNG draw this reseed
does not change.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.ml.synthetic.frontier_append import (
    base_business_metrics_frame,
    generate_month_cohort,
    iter_month_starts,
)
from src.ml.synthetic.loaders import BatchLoader, LoaderConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per page otherwise
logger = logging.getLogger(__name__)

# The 5 gap-connector keys (BusinessMetricsGenerator.METRIC_CONFIGS). Rows with
# any other metric_name (or NULL — the per_hcp_rollup rows) are out of scope.
AGGREGATE_METRIC_NAMES = ("trx", "nrx", "market_share", "conversion_rate", "hcp_engagement_score")
DIFF_COLUMNS = ["metric_id", "metric_date", "brand", "region", "metric_name", "value", "target"]
PAGE_SIZE = 1000


def build_reseed_frame(frontier: date) -> pd.DataFrame:
    """Every aggregate row the DB should hold at ``frontier`` under the
    current DGP: the frozen base plus the cohort months BM_EPOCH..frontier
    (exactly the months the cron's ``iter_month_starts`` emits)."""
    frames = [base_business_metrics_frame()]
    for month_start in iter_month_starts(frontier):
        frames.append(generate_month_cohort(month_start)["business_metrics"])
    frame = pd.concat(frames, ignore_index=True)
    frame["is_synthetic"] = True
    return frame


def fetch_db_aggregate_rows(client: Any, page_size: int = PAGE_SIZE) -> pd.DataFrame:
    """Read the live aggregate rows (PK-ordered ``.range()`` paging to an
    empty page — the repository's cap-agnostic idiom, #931/#938)."""
    rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        page = (
            client.table("business_metrics")
            .select(",".join(DIFF_COLUMNS))
            .in_("metric_name", list(AGGREGATE_METRIC_NAMES))
            .order("metric_id")
            .range(offset, offset + page_size - 1)
            .execute()
        ).data
        if not page:
            break
        rows.extend(page)
        offset += len(page)
    df = pd.DataFrame(rows, columns=DIFF_COLUMNS)
    df["value"] = pd.to_numeric(df["value"])
    df["target"] = pd.to_numeric(df["target"])
    df["metric_date"] = df["metric_date"].astype(str).str[:10]
    return df


def _national_trx(df: pd.DataFrame, month: Optional[str]) -> Dict[str, float]:
    d = df[df["metric_name"] == "trx"]
    if month is not None:
        d = d[d["metric_date"] == month]
    return {str(b): float(v) for b, v in d.groupby("brand")["value"].sum().items()}


def diff_summary(db: pd.DataFrame, regen: pd.DataFrame, scale_month: str) -> Dict[str, Any]:
    """What ``--execute`` would change. Pure; both frames carry DIFF_COLUMNS."""
    db_ids, regen_ids = set(db["metric_id"]), set(regen["metric_id"])
    merged = regen[DIFF_COLUMNS].merge(
        db[["metric_id", "value", "target"]], on="metric_id", suffixes=("", "_db")
    )
    value_changed = int(((merged["value"] - merged["value_db"]).abs() > 0.005).sum())
    target_changed = int(((merged["target"] - merged["target_db"]).abs() > 0.005).sum())

    def scale(month: Optional[str]) -> Dict[str, Dict[str, float]]:
        before, after = _national_trx(db, month), _national_trx(regen, month)
        out: Dict[str, Dict[str, float]] = {}
        for brand in sorted(set(before) | set(after)):
            b, a = before.get(brand, 0.0), after.get(brand, 0.0)
            out[brand] = {"before": b, "after": a, "ratio": round(a / b, 4) if b else float("nan")}
        return out

    regional = {}
    for (brand, region), grp in (
        regen[(regen["metric_name"] == "trx") & (regen["metric_date"] == scale_month)]
        .groupby(["brand", "region"])["value"]
        .sum()
        .items()
    ):
        dbv = db[
            (db["metric_name"] == "trx")
            & (db["metric_date"] == scale_month)
            & (db["brand"] == brand)
            & (db["region"] == region)
        ]["value"].sum()
        regional[f"{brand}/{region}"] = {"before": float(dbv), "after": float(grp)}

    return {
        "rows_to_upsert": int(len(regen)),
        "db_aggregate_rows": int(len(db)),
        "ids_only_in_regen": sorted(regen_ids - db_ids),
        "ids_only_in_db": sorted(db_ids - regen_ids),
        "value_changed": value_changed,
        "value_unchanged": int(len(merged) - value_changed),
        "target_changed": target_changed,
        "scale_month": scale_month,
        "national_trx": scale(scale_month),
        "national_trx_all_months": scale(None),
        "regional_trx": regional,
    }


def execute_refusal(
    summary: Dict[str, Any], allow_id_drift: bool = False, allow_new_cohorts: bool = False
) -> Optional[str]:
    """Why ``--execute`` must NOT proceed, or None. Fails closed on id drift in
    EITHER direction — each direction has its own explicit opt-in, because
    they mean different things — and always on target drift.

    * ids only in the DB: the in-place identity assumption (#1833 Step 0) no
      longer holds for those rows; upserting would leave them stale beside
      the regenerated ones. ``--allow-id-drift`` leaves them in place.
    * ids only in the regeneration: cohort months the Mon-3AM cron has not
      appended yet (e.g. ``--frontier 2026-09-01`` before the first September
      cron). Upserting would INSERT them — byte-identical to what the cron
      will emit, but not the in-place reseed this script promises.
      ``--allow-new-cohorts`` inserts them now.
    * targets differ: the RNG stream moved; this is not a value-only reseed.
      No flag — the delete+reinsert path in the issue applies instead.
    """
    only_db, only_regen = summary["ids_only_in_db"], summary["ids_only_in_regen"]
    if summary["target_changed"]:
        return (
            f"{summary['target_changed']} targets differ — the RNG stream moved; this is not "
            "the value-only reseed this script is for (see the issue's delete+reinsert path)."
        )
    if only_db and not allow_id_drift:
        return (
            f"{len(only_db)} aggregate ids in the DB are not reproduced by the regeneration "
            f"(e.g. {only_db[:3]}); the in-place identity assumption (#1833 Step 0) no longer "
            "holds for them. Investigate, or pass --allow-id-drift to leave them in place."
        )
    if only_regen and not allow_new_cohorts:
        return (
            f"{len(only_regen)} regenerated ids are absent from the DB (e.g. {only_regen[:3]}): "
            "cohort months the Mon-3AM cron has not appended yet. Run after the cron, use an "
            "earlier --frontier, or pass --allow-new-cohorts to insert them now."
        )
    return None


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"rows to upsert            : {summary['rows_to_upsert']}")
    print(f"db aggregate rows         : {summary['db_aggregate_rows']}")
    only_regen, only_db = summary["ids_only_in_regen"], summary["ids_only_in_db"]
    print(f"ids only in regeneration  : {len(only_regen)} {only_regen[:5]}")
    print(f"ids only in db (stale)    : {len(only_db)} {only_db[:5]}")
    print(f"value changed / unchanged : {summary['value_changed']} / {summary['value_unchanged']}")
    print(f"target changed            : {summary['target_changed']}  (must be 0: RNG untouched)")
    print(f"per-brand national TRx, {summary['scale_month']} (before -> after, ratio):")
    for brand, s in summary["national_trx"].items():
        print(f"  {brand:13s} {s['before']:>14,.2f} -> {s['after']:>14,.2f}  x{s['ratio']:.4f}")
    print("per-brand national TRx, all months (before -> after, ratio):")
    for brand, s in summary["national_trx_all_months"].items():
        print(f"  {brand:13s} {s['before']:>16,.2f} -> {s['after']:>16,.2f}  x{s['ratio']:.4f}")
    print(f"brand/region TRx, {summary['scale_month']} (before -> after):")
    for key, s in summary["regional_trx"].items():
        ratio = s["after"] / s["before"] if s["before"] else float("nan")
        print(f"  {key:24s} {s['before']:>12,.2f} -> {s['after']:>12,.2f}  x{ratio:.4f}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True, help="(default) diff only")
    mode.add_argument("--execute", action="store_true", help="perform the in-place upsert")
    parser.add_argument(
        "--frontier",
        type=date.fromisoformat,
        default=date.today(),
        help="regenerate cohort months through this date (default: today)",
    )
    parser.add_argument(
        "--scale-month",
        default=None,
        help="YYYY-MM-01 month for the national-scale readout (default: latest month in the DB)",
    )
    parser.add_argument(
        "--allow-id-drift",
        action="store_true",
        help="execute even if the DB holds aggregate ids the regeneration does not (stale rows stay)",
    )
    parser.add_argument(
        "--allow-new-cohorts",
        action="store_true",
        help=(
            "execute even if the regeneration holds cohort ids the DB does not yet (a month "
            "the Mon-3AM cron has not appended); they are INSERTED, byte-identical to what "
            "the cron will upsert"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args(argv)

    # Same convention as scripts/load_synthetic_data.py: the checkout's .env
    # (a no-op under the cron wrapper's `dotenv run`, or in a bare worktree).
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    loader = BatchLoader(LoaderConfig(batch_size=args.batch_size, dry_run=False))
    client = loader.client
    if client is None:
        logger.error("no Supabase client (SUPABASE_URL / key missing) — nothing read or written")
        return 2

    logger.info("reading live aggregate rows ...")
    db = fetch_db_aggregate_rows(client)
    logger.info("regenerating base + cohort months through %s ...", args.frontier)
    regen = build_reseed_frame(args.frontier)
    scale_month = args.scale_month or (
        db["metric_date"].max() if len(db) else regen["metric_date"].max()
    )
    summary = diff_summary(db, regen, scale_month=scale_month)
    print_summary(summary)

    if not args.execute:
        print("\nDRY RUN — nothing written. Re-run with --execute to upsert.")
        return 0

    refusal = execute_refusal(
        summary, allow_id_drift=args.allow_id_drift, allow_new_cohorts=args.allow_new_cohorts
    )
    if refusal is not None:
        logger.error("REFUSING: %s", refusal)
        return 3

    logger.info("upserting %d rows on metric_id via BatchLoader.load_table ...", len(regen))
    result = loader.load_table("business_metrics", regen)
    print(
        f"\nloaded {result.records_loaded} rows, {result.records_failed} failed "
        f"({result.total_batches} batches)"
    )
    for err in result.errors[:10]:
        print(f"  error: {err}")
    if result.records_failed:
        return 1

    logger.info("verifying: re-reading live rows ...")
    after = fetch_db_aggregate_rows(client)
    post = diff_summary(after, regen, scale_month=scale_month)
    print(
        f"post-upsert verification: value mismatches={post['value_changed']} "
        f"target mismatches={post['target_changed']} missing ids={len(post['ids_only_in_regen'])}"
    )
    return 0 if (post["value_changed"] == 0 and not post["ids_only_in_regen"]) else 1


if __name__ == "__main__":
    sys.exit(main())
