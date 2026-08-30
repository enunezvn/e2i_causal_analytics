#!/usr/bin/env python
"""#1833 empirical arbiter — does the planted brand x region geography WIN the
real gap-analyzer ranking, not just exist in the data?

Regenerates business_metrics offline under the CURRENT DGP (the frozen base
identity + the monthly frontier cohorts the Mon-3AM cron will emit), then runs
the REAL gap arithmetic and the REAL ROI ranking over that frame for every
brand at the current frontier and the next six monthly frontier positions:

* ``SupabaseDataConnector.fetch_performance_data`` / ``fetch_prior_period``
  (the production 90-day current window and the 90-day prior window, exactly
  as ``_parse_time_period``'s ``current_quarter`` fallback computes them),
* ``BenchmarkStore.get_targets`` / ``get_peer_benchmarks`` / ``get_top_decile``
  (all-history per-region means, P75, P90 — un-windowed, as in production),
* ``GapDetectorNode._detect_segment_gaps`` -> ``_calculate_gap`` with the
  production request defaults (metrics ``["trx", "market_share"]``, segments
  ``["region"]``, ``min_gap_threshold`` 5.0, gap_type ``all``),
* ``ROICalculatorNode._calculate_roi`` (brand-scoped $/TRx from
  config/agents/gap_analyzer.yaml) and ``PrioritizerNode.execute`` (rank by
  expected_roi, low-value suppression) — ranking is by ROI, not gap %.

Nothing in the arithmetic is mocked. The ONLY substitution is the storage
layer: ``FrameRepository`` serves the generated frame with the same filter
semantics as ``BusinessMetricRepository`` (metric_name / brand / region
equality, inclusive metric_date bounds, ``.limit(1000)`` on the time series),
and only rows dated on or before the frontier are visible — the DB at frontier
``f`` holds only the months loaded through ``f``.

PASS per brand: the #1 prioritized opportunity's segment is the planted region
at the current frontier AND at >= 5 of the next 6 monthly positions.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/gap_arbiter_1833.py [--frontier 2026-08-30]

Exit status 0 iff every brand passes.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import pandas as pd

from src.agents.gap_analyzer.connectors.benchmark_store import BenchmarkStore
from src.agents.gap_analyzer.connectors.supabase_connector import SupabaseDataConnector
from src.agents.gap_analyzer.nodes.gap_detector import GapDetectorNode
from src.agents.gap_analyzer.nodes.prioritizer import PrioritizerNode
from src.agents.gap_analyzer.nodes.roi_calculator import ROICalculatorNode
from src.ml.synthetic.frontier_append import (
    BM_EPOCH,
    base_business_metrics_frame,
    generate_month_cohort,
)
from src.ml.synthetic.generators.business_metrics_generator import BusinessMetricsGenerator

# Production request defaults (src/api/routes/gaps.py RunGapAnalysisRequest).
METRICS = ["trx", "market_share"]
SEGMENTS = ["region"]
MIN_GAP_THRESHOLD = 5.0
MAX_OPPORTUNITIES = 10
BRANDS = ("Kisqali", "Fabhalta", "Remibrutinib")
NEXT_POSITIONS = 6
REQUIRED_NEXT_PASSES = 5


def planted_region(brand: str) -> str:
    """The brand's weakest execution region — the planted story."""
    row = BusinessMetricsGenerator.BRAND_REGION_PERFORMANCE[brand]
    return min(row, key=lambda region: row[region])


class FrameRepository:
    """business_metrics rows served from an in-memory frame with the SAME
    filter semantics as ``BusinessMetricRepository`` (see module docstring).
    Rows dated after ``loaded_through`` are invisible."""

    def __init__(self, frame: pd.DataFrame, loaded_through: date):
        self.df = frame[frame["metric_date"] <= loaded_through.isoformat()].reset_index(drop=True)

    async def get_time_series(
        self,
        kpi_name: str,
        brand: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_synthetic: bool = False,
    ) -> List[Dict[str, Any]]:
        d = self.df[(self.df["metric_name"] == kpi_name) & (self.df["brand"] == brand)]
        if start_date:
            d = d[d["metric_date"] >= start_date]
        if end_date:
            d = d[d["metric_date"] <= end_date]
        records: List[Dict[str, Any]] = (
            d.sort_values("metric_date", kind="stable").head(1000).to_dict("records")
        )
        return records

    async def get_distinct_values(
        self, column: str, brand: Optional[str] = None, include_synthetic: bool = False, **_: Any
    ) -> List[str]:
        if column not in self.df.columns:
            return []
        d = self.df if brand is None else self.df[self.df["brand"] == brand]
        return list(dict.fromkeys(d[column].dropna().tolist()))

    async def get_by_region_paged(
        self,
        region: str,
        brand: Optional[str] = None,
        include_synthetic: bool = False,
        columns: str = "*",
        **_: Any,
    ) -> List[Dict[str, Any]]:
        d = self.df[self.df["region"] == region]
        if brand is not None:
            d = d[d["brand"] == brand]
        records: List[Dict[str, Any]] = d.to_dict("records")
        return records


def build_frame(last_month: date) -> pd.DataFrame:
    """Frozen base (2013-01..2026-07) + monthly cohorts BM_EPOCH..last_month,
    exactly what the DB holds after the reseed + successive cron appends."""
    frames = [base_business_metrics_frame()]
    ms = BM_EPOCH
    while ms <= last_month:
        frames.append(generate_month_cohort(ms)["business_metrics"])
        ms = (ms + timedelta(days=32)).replace(day=1)
    return pd.concat(frames, ignore_index=True)


def production_time_period(frontier: date) -> str:
    """The window ``_parse_time_period`` derives for the ``current_quarter``
    default: last 90 days ending today."""
    return f"{(frontier - timedelta(days=90)).isoformat()}_{frontier.isoformat()}"


@dataclass
class Ranked:
    brand: str
    frontier: date
    segment: Optional[str]
    metric: Optional[str]
    gap_type: Optional[str]
    gap_pct: Optional[float]
    expected_roi: Optional[float]
    best_other: Optional[str]  # highest-ROI opportunity in a NON-planted region
    best_other_roi: Optional[float]
    n_gaps: int


async def rank_one(
    frame: pd.DataFrame,
    brand: str,
    frontier: date,
    detector: GapDetectorNode,
    roi_node: ROICalculatorNode,
    prioritizer: PrioritizerNode,
) -> Ranked:
    repo = FrameRepository(frame, loaded_through=frontier)
    # Storage substitution only (see module docstring): the connector and the
    # store resolve ``_repository`` lazily; pre-seeding it is the one seam.
    connector = SupabaseDataConnector(include_synthetic=True)
    cast(Any, connector)._repository = repo
    store = BenchmarkStore(include_synthetic=True)
    cast(Any, store)._repository = repo

    time_period = production_time_period(frontier)
    current = await connector.fetch_performance_data(
        brand=brand, metrics=METRICS, segments=SEGMENTS, time_period=time_period, filters=None
    )
    comparison = await detector._get_comparison_data(
        gap_type="all",
        brand=brand,
        metrics=METRICS,
        segments=SEGMENTS,
        time_period=time_period,
        data_connector=connector,
        benchmark_store=store,
    )
    _, gaps = await detector._detect_segment_gaps(
        current_data=current,
        comparison_data=comparison,
        segment="region",
        metrics=METRICS,
        gap_type="all",
        min_gap_threshold=MIN_GAP_THRESHOLD,
    )
    gaps.sort(key=lambda g: g["gap_percentage"], reverse=True)
    value_per_trx = roi_node._resolve_value_per_trx(brand)
    roi_estimates = [roi_node._calculate_roi(g, value_per_trx=value_per_trx) for g in gaps]
    state: Dict[str, Any] = {
        "brand": brand,
        "gaps_detected": gaps,
        "roi_estimates": roi_estimates,
        "max_opportunities": MAX_OPPORTUNITIES,
    }
    result = await prioritizer.execute(state)  # type: ignore[arg-type]
    if result.get("status") != "completed":
        raise RuntimeError(f"prioritizer failed: {result.get('errors')}")
    opps = result["prioritized_opportunities"]
    if not opps:
        return Ranked(brand, frontier, None, None, None, None, None, None, None, len(gaps))
    top = opps[0]
    planted = planted_region(brand)
    other = next((o for o in opps if o["gap"]["segment_value"] != planted), None)
    return Ranked(
        brand=brand,
        frontier=frontier,
        segment=top["gap"]["segment_value"],
        metric=top["gap"]["metric"],
        gap_type=top["gap"]["gap_type"],
        gap_pct=top["gap"]["gap_percentage"],
        expected_roi=top["roi_estimate"]["expected_roi"],
        best_other=(
            f"{other['gap']['segment_value']}/{other['gap']['metric']}/{other['gap']['gap_type']}"
            if other
            else None
        ),
        best_other_roi=(other["roi_estimate"]["expected_roi"] if other else None),
        n_gaps=len(gaps),
    )


def frontier_positions(current: date, n_next: int = NEXT_POSITIONS) -> List[date]:
    """``current`` plus the same day-of-month in each of the next ``n_next``
    months (clamped to month end)."""
    out = [current]
    y, m = current.year, current.month
    for _ in range(n_next):
        m += 1
        if m > 12:
            m, y = 1, y + 1
        last_day = ((date(y, m, 1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)).day
        out.append(date(y, m, min(current.day, last_day)))
    return out


async def run(
    frontier: date, brands: Sequence[str] = BRANDS, frame: Optional[pd.DataFrame] = None
) -> Tuple[Dict[str, bool], List[Ranked]]:
    positions = frontier_positions(frontier)
    last_month = positions[-1].replace(day=1)
    if frame is None:
        frame = build_frame(last_month)
    detector = GapDetectorNode(use_mock=False)
    roi_node = ROICalculatorNode()
    prioritizer = PrioritizerNode()
    rows: List[Ranked] = []
    verdict: Dict[str, bool] = {}
    for brand in brands:
        target = planted_region(brand)
        hits = []
        for pos in positions:
            r = await rank_one(frame, brand, pos, detector, roi_node, prioritizer)
            rows.append(r)
            hits.append(r.segment == target)
        verdict[brand] = bool(hits[0]) and sum(hits[1:]) >= REQUIRED_NEXT_PASSES
    return verdict, rows


def format_table(rows: List[Ranked], verdict: Dict[str, bool]) -> str:
    lines = [
        f"   {'brand':12s} {'frontier':10s} {'planted':9s} {'top':9s} {'metric':12s} "
        f"{'type':12s} {'gap%':>6s} {'roi':>7s}  best other region (roi)         gaps"
    ]
    for r in rows:
        planted = planted_region(r.brand)
        mark = "OK " if r.segment == planted else "XX "
        other_roi = f"({r.best_other_roi:.2f})" if r.best_other_roi is not None else ""
        lines.append(
            f"{mark}{r.brand:12s} {r.frontier.isoformat():10s} {planted:9s} "
            f"{(r.segment or '-'):9s} {(r.metric or '-'):12s} {(r.gap_type or '-'):12s} "
            f"{(r.gap_pct if r.gap_pct is not None else float('nan')):6.1f} "
            f"{(r.expected_roi if r.expected_roi is not None else float('nan')):7.2f}  "
            f"{(r.best_other or 'none'):32s}{other_roi:8s} {r.n_gaps}"
        )
    for brand, ok in verdict.items():
        lines.append(f"{brand}: {'PASS' if ok else 'FAIL'} (planted={planted_region(brand)})")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--frontier", type=date.fromisoformat, default=date.today())
    args = parser.parse_args(argv)
    verdict, rows = asyncio.run(run(args.frontier))
    print(format_table(rows, verdict))
    return 0 if all(verdict.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
