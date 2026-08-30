"""#1834 — the gap_detector node resolves ``time_period`` once, surfaces the window
in state, and fails CLOSED on an unparseable period (like every other connector
error: ``status='failed'`` + an ``errors`` entry — never a silent 90-day window).
"""

from __future__ import annotations

from datetime import date

import pytest

from src.agents.gap_analyzer.graph import create_gap_analyzer_graph
from src.agents.gap_analyzer.nodes.gap_detector import GapDetectorNode
from src.agents.gap_analyzer.state import GapAnalyzerState

TODAY = date(2026, 8, 30)
EXPECTED_WINDOW = {
    "time_period": "current_quarter",
    "period_start": "2026-07-01",
    "period_end": "2026-08-30",
    "prior_start": "2026-04-01",
    "prior_end": "2026-06-30",
}


@pytest.fixture
def frozen_today(monkeypatch):
    import src.utils.gap_time_period as tp

    monkeypatch.setattr(tp, "_today", lambda: TODAY)
    return TODAY


def _state(time_period: str, gap_type: str = "temporal") -> GapAnalyzerState:
    return {
        "query": "identify trx gaps",
        "metrics": ["trx", "nrx"],
        "segments": ["region"],
        "brand": "kisqali",
        "time_period": time_period,
        "filters": None,
        "tier0_frame_ref": None,
        "instrument_specs": None,
        "instrument_strength_by_feature": None,
        "gap_type": gap_type,  # type: ignore[typeddict-item]
        "min_gap_threshold": 5.0,
        "max_opportunities": 10,
        "gaps_detected": None,
        "gaps_by_segment": None,
        "total_gap_value": None,
        "roi_estimates": None,
        "total_addressable_value": None,
        "prioritized_opportunities": None,
        "quick_wins": None,
        "strategic_bets": None,
        "executive_summary": None,
        "key_insights": None,
        "detection_latency_ms": 0,
        "roi_latency_ms": 0,
        "total_latency_ms": 0,
        "segments_analyzed": 0,
        "errors": [],
        "warnings": [],
        "status": "pending",
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_node_fails_closed_on_unknown_time_period(frozen_today):
    node = GapDetectorNode(use_mock=True)

    result = await node.execute(_state("bogus"))

    assert result["status"] == "failed"
    assert result.get("gaps_detected") is None
    assert len(result["errors"]) == 1
    err = result["errors"][0]
    assert err["node"] == "gap_detector"
    assert "'bogus'" in err["error"]
    assert "current_quarter" in err["error"]  # the accepted forms travel with the error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_node_surfaces_the_resolved_window_in_state(frozen_today):
    node = GapDetectorNode(use_mock=True)

    result = await node.execute(_state("current_quarter"))

    assert result["status"] == "calculating"
    assert result["resolved_period"] == EXPECTED_WINDOW


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graph_carries_the_resolved_window_to_the_final_state(frozen_today):
    graph = create_gap_analyzer_graph(use_mock=True)

    final_state = await graph.ainvoke(_state("current_quarter", gap_type="vs_target"))

    assert final_state.get("errors") == []
    assert final_state["resolved_period"] == EXPECTED_WINDOW


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graph_fails_closed_end_to_end_on_unknown_time_period(frozen_today):
    graph = create_gap_analyzer_graph(use_mock=True)

    final_state = await graph.ainvoke(_state("2024Q3", gap_type="vs_target"))

    assert final_state["status"] == "failed"
    detector_errors = [e for e in final_state["errors"] if e.get("node") == "gap_detector"]
    assert len(detector_errors) == 1
    # Positive control: the failure is the GRAMMAR, not an unrelated connector error.
    assert "'2024Q3'" in detector_errors[0]["error"]
    assert "Accepted forms" in detector_errors[0]["error"]
    assert final_state.get("resolved_period") is None


@pytest.mark.unit
def test_agent_output_carries_the_resolved_window():
    """``GapAnalyzerAgent._build_output`` is the chat/dispatcher surface — the window
    must reach it, not only the HTTP response."""
    from src.agents.gap_analyzer.agent import GapAnalyzerAgent

    agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)
    state = _state("current_quarter")
    state["resolved_period"] = dict(EXPECTED_WINDOW)  # type: ignore[typeddict-unknown-key]
    state["status"] = "completed"

    output = agent._build_output(state)

    assert output["resolved_period"] == EXPECTED_WINDOW


# ---------------------------------------------------------------------------
# codex iter-1 MEDIUM: the node's resolved window must be the window QUERIED.
# ---------------------------------------------------------------------------


class _WindowRecordingRepo:
    """Replays monthly rows (1st of each month) and records every window asked for."""

    ROWS = {
        "2026-04-01": 66412.17,
        "2026-05-01": 49893.14,
        "2026-06-01": 34561.03,
        "2026-07-01": 49585.25,
        "2026-08-01": 60885.99,
        "2026-09-01": 52000.00,
        "2026-10-01": 58000.00,
    }

    def __init__(self) -> None:
        self.windows: list[tuple[str, str]] = []

    async def get_time_series(self, kpi_name, brand, start_date, end_date, include_synthetic=False):
        self.windows.append((start_date, end_date))
        return [
            {"metric_date": d, "value": v, "target": None, "region": "west"}
            for d, v in self.ROWS.items()
            if start_date <= d <= end_date
        ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_connector_queries_the_window_the_node_resolved_even_across_midnight(
    monkeypatch,
):
    """Resolve ONCE. If the clock flips from Sep 30 to Oct 1 between the node's
    resolution and the connector's reads, the persisted ``resolved_period`` must
    still be the window that was actually queried (Q3 vs Q2), not Q4-to-date vs Q3."""
    from unittest.mock import MagicMock

    import src.utils.gap_time_period as tp
    from src.agents.gap_analyzer.connectors.supabase_connector import SupabaseDataConnector

    clock = iter([date(2026, 9, 30)])  # first read: Sep 30; every later read: Oct 1

    def _flipping_today() -> date:
        return next(clock, date(2026, 10, 1))

    monkeypatch.setattr(tp, "_today", _flipping_today)

    repo = _WindowRecordingRepo()
    connector = SupabaseDataConnector(supabase_client=MagicMock(), include_synthetic=True)
    connector._repository = repo
    node = GapDetectorNode(use_mock=True)
    node._connector_pairs = {False: (connector, node.benchmark_store)}

    state = _state("current_quarter", gap_type="temporal")
    state["metrics"] = ["trx"]
    result = await node.execute(state)

    assert result["status"] == "calculating", result.get("errors")
    assert result["resolved_period"] == {
        "time_period": "current_quarter",
        "period_start": "2026-07-01",
        "period_end": "2026-09-30",
        "prior_start": "2026-04-01",
        "prior_end": "2026-06-30",
    }
    # The windows the repository was actually asked for == the resolved window.
    assert repo.windows == [("2026-07-01", "2026-09-30"), ("2026-04-01", "2026-06-30")]
