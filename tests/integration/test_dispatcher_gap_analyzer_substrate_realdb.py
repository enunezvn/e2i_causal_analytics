"""Issue #874 — faithful (real docker-Supabase) proof that the gap_analyzer
dispatcher resolver binds REAL business_metrics substrate at dispatch time.

Load-bearing, NO mocks: a chat-shaped dispatch through the real ``DispatcherNode``
with a real ``GapAnalyzerAgent`` instance, against the live synthetic substrate
(``include_synthetic`` opt-in, the #872 channels), must produce a SUCCESSFUL run
grounded in real rows — and a real-mode dispatch against a synthetic-only brand
must fail closed with NeedsStructuredInput semantics, never the raw 7ms
``Missing required field: metrics`` ValueError.

Gated behind ``E2I_DB_INTEGRATION=1``; run with ``-n0``. Read-only: no rows are
created, so there is nothing to clean up.
"""

from __future__ import annotations

import os

import pytest

from src.agents.orchestrator.nodes import dispatcher as disp
from src.agents.orchestrator.nodes.dispatcher import DispatcherNode, NeedsStructuredInput

_RUN = os.environ.get("E2I_DB_INTEGRATION") == "1"

if _RUN:
    from dotenv import load_dotenv

    load_dotenv()

pytestmark = pytest.mark.skipif(
    not _RUN, reason="set E2I_DB_INTEGRATION=1 to run faithful real-DB tests"
)


@pytest.fixture(autouse=True)
def _reset_async_client_cache():
    """Reset the module-global async Supabase client cache around EACH test.

    ``get_async_supabase_client`` caches a client bound to the event loop that
    created it; pytest-asyncio gives each test its own loop (see the #851
    integration suite for the same fixture).
    """
    import src.memory.services.factories as factories

    factories._async_supabase_client = None
    yield
    factories._async_supabase_client = None


# Kisqali exists ONLY as synthetic rows in the prod docker DB (Shard 05) — the
# clean substrate state the 2026-06-11 synthetic-gold cleanup certified.
_SYNTH_BRAND = "Kisqali"


def _dispatch(params=None, timeout_ms: int = 180000):
    return {
        "agent_name": "gap_analyzer",
        "priority": "high",
        "parameters": params or {},
        "timeout_ms": timeout_ms,
        "fallback_agent": None,
        "execution_mode": "parallel",
    }


def _state(query: str, user_context: dict, params=None):
    return {
        "query": query,
        "user_context": user_context,
        "session_id": "itest-874",
        "parsed_query": {"intent": "performance_gap", "entities": []},
        "dispatch_plan": [_dispatch(params)],
        "parallel_groups": [["gap_analyzer"]],
    }


def test_resolver_binds_real_substrate_with_opt_in() -> None:
    """Resolver-level: an opted-in dispatch derives metrics/segments/brand from
    the live synthetic business_metrics rows (real metric names, real segment)."""
    resolved = disp.INPUT_RESOLVERS["gap_analyzer"](
        {
            "query": f"where are the biggest performance gaps for {_SYNTH_BRAND}?",
            "session_id": "itest-874",
            "user_context": {"brand": _SYNTH_BRAND, "include_synthetic": True},
            "parsed_query": {"entities": []},
        },
        _dispatch(),
    )
    assert not isinstance(resolved, NeedsStructuredInput), getattr(resolved, "reason", "")
    assert resolved["brand"] == _SYNTH_BRAND
    assert resolved["segments"] == ["region"]
    assert resolved["include_synthetic"] is True
    # Real metric names from the live substrate — non-empty, all real columns.
    assert resolved["metrics"], "expected >=1 real metric_name from business_metrics"
    known = {"trx", "nrx", "conversion_rate", "market_share", "hcp_engagement_score"}
    assert set(resolved["metrics"]) <= known, f"unexpected metric names: {resolved['metrics']}"


def test_resolver_real_mode_fails_closed_for_synthetic_only_brand() -> None:
    """Real mode (no opt-in) on a synthetic-only brand → the provenance-filtered
    probe finds NO rows → NeedsStructuredInput, not fabricated inputs."""
    resolved = disp.INPUT_RESOLVERS["gap_analyzer"](
        {
            "query": f"gaps for {_SYNTH_BRAND}",
            "session_id": "itest-874",
            "user_context": {"brand": _SYNTH_BRAND},
            "parsed_query": {"entities": []},
        },
        _dispatch(),
    )
    assert isinstance(resolved, NeedsStructuredInput)
    assert "business_metrics" in resolved.reason


@pytest.mark.asyncio
async def test_chat_dispatch_opt_in_runs_gap_analysis_on_real_substrate() -> None:
    """LOAD-BEARING: the full dispatcher path (resolver -> real GapAnalyzerAgent ->
    production connector with the per-run include_synthetic opt-in) completes a
    REAL gap analysis on the live synthetic substrate."""
    from src.agents.gap_analyzer.agent import GapAnalyzerAgent

    agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)
    node = DispatcherNode(agent_registry={"gap_analyzer": agent})

    out = await node.execute(
        _state(
            f"where are the biggest performance gaps for {_SYNTH_BRAND} by region?",
            {"brand": _SYNTH_BRAND, "include_synthetic": True},
            params={"gap_type": "all", "time_period": "2012-01-01_2026-12-31"},
        )
    )
    res = out["agent_results"][0]
    assert res["success"] is True, f"dispatch failed: {res.get('error')}"
    result = res["result"]
    assert result.get("status") == "completed", result.get("errors")
    # Real analysis happened: the single requested segment dimension was analyzed
    # and real gaps/opportunities were recovered from the substrate.
    assert result.get("segments_analyzed") == 1
    assert (result.get("total_gap_value") or 0) > 0, "expected real gaps from the substrate"
    assert result.get("prioritized_opportunities"), "expected >=1 real opportunity"
    # Latency sanity: this was a real run, not the old 7ms validation crash.
    assert res["latency_ms"] > 100


@pytest.mark.asyncio
async def test_chat_dispatch_real_mode_fails_closed_not_raw_valueerror() -> None:
    """The original #874 symptom, end-to-end: a real-mode chat dispatch must fail
    CLOSED with the structured resolver error — never the raw
    ``Missing required field: metrics`` ValueError."""
    from src.agents.gap_analyzer.agent import GapAnalyzerAgent

    agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)
    node = DispatcherNode(agent_registry={"gap_analyzer": agent})

    out = await node.execute(_state(f"gaps for {_SYNTH_BRAND}", {"brand": _SYNTH_BRAND}))
    res = out["agent_results"][0]
    assert res["success"] is False
    err = (res["error"] or "").lower()
    assert "missing required field" not in err, f"raw field-validation crash leaked: {err}"
    assert "fabricat" in err
    assert "business_metrics" in err
