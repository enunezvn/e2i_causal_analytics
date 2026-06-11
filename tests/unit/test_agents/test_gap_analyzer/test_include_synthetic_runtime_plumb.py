"""Issue #874: the include_synthetic opt-in must REACH the agent on the chat path.

The agent factory (``src/agents/factory.py``) registers ``GapAnalyzerAgent()`` with
constructor defaults, and the #872/#851 ``include_synthetic`` flag was baked into the
compiled graph at construction (GapDetectorNode builds its connectors in __init__).
A per-dispatch opt-in therefore could NOT reach the connectors — an opted-in chat
dispatch would silently read the real-mode (empty) substrate and report "no gaps".

Fix: ``include_synthetic`` is now a per-run input — ``run(input_data)`` threads it
into ``GapAnalyzerState`` (defaulting to the constructor flag), and GapDetectorNode
resolves a per-run connector pair honoring it.
"""

from __future__ import annotations

import pytest

from src.agents.gap_analyzer.agent import GapAnalyzerAgent
from src.agents.gap_analyzer.nodes.gap_detector import GapDetectorNode


def test_initialize_state_threads_include_synthetic_from_input() -> None:
    agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)
    state = agent._initialize_state(
        {
            "query": "q",
            "metrics": ["trx"],
            "segments": ["region"],
            "brand": "Kisqali",
            "include_synthetic": True,
        }
    )
    assert state["include_synthetic"] is True


def test_initialize_state_defaults_to_constructor_flag() -> None:
    # Constructor default (False) when the input omits the flag.
    agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)
    state = agent._initialize_state(
        {"query": "q", "metrics": ["trx"], "segments": ["region"], "brand": "Kisqali"}
    )
    assert state["include_synthetic"] is False

    # A validation-constructed agent (include_synthetic=True) keeps its opt-in.
    agent_synth = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False, include_synthetic=True)
    state = agent_synth._initialize_state(
        {"query": "q", "metrics": ["trx"], "segments": ["region"], "brand": "Kisqali"}
    )
    assert state["include_synthetic"] is True


def test_gap_detector_resolves_per_run_connectors() -> None:
    """A real-mode-constructed node must serve an OPT-IN connector pair on demand
    (and cache it), while the default request returns the constructed pair."""
    node = GapDetectorNode(use_mock=False, include_synthetic=False)

    # Default: the constructed (real-mode) pair.
    data_conn, bench = node._connectors_for(False)
    assert data_conn is node.data_connector
    assert bench is node.benchmark_store
    assert data_conn.include_synthetic is False

    # Opt-in: a synthetic-opted pair, built lazily and cached.
    synth_conn, synth_bench = node._connectors_for(True)
    assert synth_conn.include_synthetic is True
    assert synth_bench.include_synthetic is True
    again_conn, again_bench = node._connectors_for(True)
    assert again_conn is synth_conn and again_bench is synth_bench


@pytest.mark.asyncio
async def test_gap_detector_execute_honors_state_opt_in(monkeypatch) -> None:
    """``execute`` must route fetches through the connector pair matching the
    state's include_synthetic — proven by capturing which pair is used."""
    import pandas as pd

    node = GapDetectorNode(use_mock=True, include_synthetic=False)

    captured: dict = {}

    class _SpyConnector:
        include_synthetic = True

        async def fetch_performance_data(self, **kwargs):
            captured["used_opt_in_connector"] = True
            return pd.DataFrame()

        async def fetch_prior_period(self, **kwargs):
            return pd.DataFrame()

    class _SpyStore:
        include_synthetic = True

        async def get_top_decile(self, **kwargs):
            return pd.DataFrame()

    node._connector_pairs[True] = (_SpyConnector(), _SpyStore())

    state = {
        "query": "q",
        "metrics": ["trx"],
        "segments": ["region"],
        "brand": "Kisqali",
        "time_period": "current_quarter",
        "filters": None,
        "tier0_data": None,
        "gap_type": "vs_potential",
        "min_gap_threshold": 5.0,
        "max_opportunities": 10,
        "include_synthetic": True,
        "errors": [],
        "warnings": [],
        "status": "pending",
    }
    out = await node.execute(state)  # type: ignore[arg-type]
    assert captured.get("used_opt_in_connector") is True
    assert out.get("status") != "failed", out.get("errors")
