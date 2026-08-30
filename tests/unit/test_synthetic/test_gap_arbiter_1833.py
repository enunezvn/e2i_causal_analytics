"""#1833 gap arbiter — the claims that make it FAITHFUL to production.

The full arbiter (scripts/gap_arbiter_1833.py) regenerates 2013..2027 and runs
the real gap/ROI/prioritizer code at 21 (brand, frontier) positions — ~2-3 min,
so it ships as a script with its output in the PR. These tests pin the parts
that make its verdict transferable to the live site: the window arithmetic
matches the connector's ``current_quarter`` fallback, the frame-backed
repository has the repository's filter semantics, and the frontier positions
are the ones the acceptance criterion names.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from scripts.gap_arbiter_1833 import (
    METRICS,
    MIN_GAP_THRESHOLD,
    SEGMENTS,
    FrameRepository,
    frontier_positions,
    planted_region,
    production_time_period,
)
from src.agents.gap_analyzer.connectors.supabase_connector import SupabaseDataConnector


def test_window_matches_the_connectors_current_quarter_fallback():
    # The production request default ``time_period="current_quarter"`` is not a
    # format _parse_time_period knows, so it falls through to "last 90 days".
    today = date.today()
    start, end = SupabaseDataConnector()._parse_time_period("current_quarter")
    assert production_time_period(today) == f"{start}_{end}"
    # and the explicit form round-trips through the connector unchanged
    assert SupabaseDataConnector()._parse_time_period(production_time_period(today)) == (
        (today - timedelta(days=90)).isoformat(),
        today.isoformat(),
    )


def test_request_defaults_match_the_api_model():
    from src.api.routes.gaps import RunGapAnalysisRequest

    req = RunGapAnalysisRequest(query="q", brand="Kisqali")
    assert req.metrics == METRICS
    assert req.segments == SEGMENTS
    assert req.min_gap_threshold == MIN_GAP_THRESHOLD
    assert req.time_period == "current_quarter"


def test_frontier_positions_are_current_plus_next_six_months():
    assert frontier_positions(date(2026, 8, 30)) == [
        date(2026, 8, 30),
        date(2026, 9, 30),
        date(2026, 10, 30),
        date(2026, 11, 30),
        date(2026, 12, 30),
        date(2027, 1, 30),
        date(2027, 2, 28),  # clamped to month end
    ]


def test_planted_region_is_distinct_per_brand():
    regions = {b: planted_region(b) for b in ("Kisqali", "Fabhalta", "Remibrutinib")}
    assert regions == {"Kisqali": "midwest", "Fabhalta": "south", "Remibrutinib": "west"}


@pytest.fixture
def frame() -> pd.DataFrame:
    rows = []
    for i, d in enumerate(("2026-06-01", "2026-07-01", "2026-08-01", "2026-09-01")):
        for region in ("west", "midwest"):
            rows.append(
                {
                    "metric_id": f"m{i}{region}",
                    "metric_date": d,
                    "metric_name": "trx",
                    "brand": "Kisqali",
                    "region": region,
                    "value": 1.0,
                    "target": 2.0,
                }
            )
    rows.append(
        {
            "metric_id": "other",
            "metric_date": "2026-07-01",
            "metric_name": "trx",
            "brand": "Fabhalta",
            "region": "west",
            "value": 9.0,
            "target": 9.0,
        }
    )
    return pd.DataFrame(rows)


async def test_rows_after_the_frontier_are_invisible(frame):
    repo = FrameRepository(frame, loaded_through=date(2026, 8, 15))
    rows = await repo.get_time_series("trx", "Kisqali")
    assert {r["metric_date"] for r in rows} == {"2026-06-01", "2026-07-01", "2026-08-01"}


async def test_time_series_bounds_are_inclusive_and_brand_scoped(frame):
    repo = FrameRepository(frame, loaded_through=date(2026, 12, 31))
    rows = await repo.get_time_series(
        "trx", "Kisqali", start_date="2026-07-01", end_date="2026-08-01"
    )
    assert sorted(r["metric_date"] for r in rows) == ["2026-07-01"] * 2 + ["2026-08-01"] * 2
    assert all(r["brand"] == "Kisqali" for r in rows)


async def test_region_page_and_distinct_values_are_brand_scoped(frame):
    repo = FrameRepository(frame, loaded_through=date(2026, 12, 31))
    west = await repo.get_by_region_paged("west", brand="Fabhalta")
    assert [r["metric_id"] for r in west] == ["other"]
    assert await repo.get_distinct_values("region", brand="Kisqali") == ["west", "midwest"]
    assert await repo.get_distinct_values("specialty", brand="Kisqali") == []
