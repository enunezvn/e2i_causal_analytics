"""#1833 gap arbiter — the claims that make it FAITHFUL to production.

The full arbiter (scripts/gap_arbiter_1833.py) regenerates 2013..2027 and runs
the real gap/ROI/prioritizer code at 21 (brand, frontier) positions — ~2-3 min,
so it ships as a script with its output in the PR. These tests pin the parts
that make its verdict transferable to the live site: the current AND prior
windows are whatever the shared #1834 grammar resolves for the API default
label with its clock pinned to the frontier (helper AND call site), the
frame-backed repository has the repository's filter semantics, and the frontier
positions are the ones the acceptance criterion names.

"""

from datetime import date

import pandas as pd
import pytest

import src.utils.gap_time_period as tp
from scripts.gap_arbiter_1833 import (
    METRICS,
    MIN_GAP_THRESHOLD,
    SEGMENTS,
    TIME_PERIOD,
    FrameRepository,
    frontier_positions,
    pinned_clock,
    planted_region,
    resolve_production_window,
)
from src.agents.gap_analyzer.connectors.supabase_connector import SupabaseDataConnector
from src.utils.gap_time_period import resolve_time_period


@pytest.mark.parametrize(
    "frontier",
    [date(2026, 8, 30), date(2026, 10, 30), date(2027, 1, 30), date(2027, 2, 28)],
)
def test_window_is_the_grammars_resolution_of_the_api_default_at_the_frontier(frontier):
    # No hand-computed range anywhere: the arbiter's window IS what the shared
    # grammar (#1834) resolves for the API default label with the clock pinned
    # to the frontier — current AND prior — so quarter boundaries (2026-10-30:
    # a one-row October vs a three-row Q3 prior) are whatever prod does.
    assert resolve_production_window(frontier) == resolve_time_period(TIME_PERIOD, today=frontier)


def test_window_does_not_depend_on_the_wall_clock():
    # The frontier, not date.today(), drives the resolution (the arbiter walks
    # six months into the future).
    far = date(2031, 5, 17)
    assert resolve_production_window(far).period_end == far
    assert resolve_production_window(far) != resolve_time_period(TIME_PERIOD)


def test_pinned_clock_is_scoped_to_the_block():
    before = tp._today
    with pinned_clock(date(2030, 3, 3)):
        assert tp._today() == date(2030, 3, 3)
        # the real connector shim reads the same seam
        assert SupabaseDataConnector()._parse_time_period(TIME_PERIOD)[1] == "2030-03-03"
    assert tp._today is before


async def test_rank_one_reads_through_the_grammars_windows_at_a_quarter_boundary():
    """codex iter-3 LOW: pin the CALL SITE, not just the helper. A tiny Kisqali
    frame Jul..Oct 2026 where only midwest/October drops; at frontier
    2026-10-30 the grammar's current window is the single October row and the
    prior is the full Q3, so the temporal gap must read current=50,000 vs
    prior-mean=100,000 (50%). A regression to the old hand-computed
    (frontier-90d) window would read a three-row Aug..Oct mean of ~83,333
    against a one-row July prior and could not produce these numbers."""
    from scripts.gap_arbiter_1833 import rank_one
    from src.agents.gap_analyzer.nodes.gap_detector import GapDetectorNode
    from src.agents.gap_analyzer.nodes.prioritizer import PrioritizerNode
    from src.agents.gap_analyzer.nodes.roi_calculator import ROICalculatorNode

    rows = []
    for month in ("2026-07-01", "2026-08-01", "2026-09-01", "2026-10-01"):
        for region in ("northeast", "south", "midwest", "west"):
            trx = 50_000.0 if (region == "midwest" and month == "2026-10-01") else 100_000.0
            for metric, value in (("trx", trx), ("market_share", 0.2)):
                rows.append(
                    {
                        "metric_id": f"{month}-{region}-{metric}",
                        "metric_date": month,
                        "metric_type": metric,
                        "metric_name": metric,
                        "brand": "Kisqali",
                        "region": region,
                        "value": value,
                        "target": value,  # vs_target is zero everywhere by construction
                    }
                )
    frame = pd.DataFrame(rows)
    frontier = date(2026, 10, 30)

    ranked = await rank_one(
        frame,
        "Kisqali",
        frontier,
        GapDetectorNode(use_mock=False),
        ROICalculatorNode(),
        PrioritizerNode(),
    )

    assert ranked.window == resolve_time_period(TIME_PERIOD, today=frontier)
    current, reference = ranked.gaps["region_midwest_trx_temporal"]
    assert current == pytest.approx(50_000.0)  # the ONE October row
    assert reference == pytest.approx(100_000.0)  # mean of the three Q3 rows
    # and the planted shortfall is what gets ranked
    assert ranked.segment == "midwest"


def test_request_defaults_match_the_api_model():
    from src.api.routes.gaps import RunGapAnalysisRequest

    req = RunGapAnalysisRequest(query="q", brand="Kisqali")
    assert req.metrics == METRICS
    assert req.segments == SEGMENTS
    assert req.min_gap_threshold == MIN_GAP_THRESHOLD
    assert req.time_period == TIME_PERIOD == "current_quarter"


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
