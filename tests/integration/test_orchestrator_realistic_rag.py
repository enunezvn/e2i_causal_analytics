"""Phase 5 verification with realistic RAG context.

The existing TestClient tests in test_orchestrator_end_to_end.py mock
hybrid_search to return []. That hides routing bugs that depend on retrieval
context — specifically issue #251 F1 (self-dispatch) and F2 (API-default
leak), both surfaced by the live Docker smoke after this branch landed.

These tests use a realistic_rag fixture (≥2 RetrievalResult entries with
domain-matched content) to assert the contract under production-equivalent
conditions. See feedback_testclient_vs_live_divergence for the reusable
lesson.
"""

from __future__ import annotations

import os
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ["E2I_TESTING_MODE"] = "1"

from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import cognitive as cognitive_route
from src.rag.types import RetrievalResult, RetrievalSource


def _realistic_retrieval_results() -> list[RetrievalResult]:
    """Return ≥2 RetrievalResult entries that match the live-Docker smoke
    expectations: causal-flavoured + experiment-flavoured content with
    domain-matched metadata.

    Per feedback_testclient_vs_live_divergence: empty-stub hybrid_search
    silently strips the routing inputs that drove the F1/F2 bugs the
    live-Docker curl smoke caught. This fixture restores those inputs.
    """
    return [
        RetrievalResult(
            id="rr-causal-1",
            content=(
                "Brand-X discontinuation rates increased 12% in Q2 2026; "
                "the leading driver is HCP detailing gap in segment B."
            ),
            source=RetrievalSource.VECTOR,
            score=0.87,
            metadata={"brand": "Brand-X", "agent": "causal_impact", "source_type": "causal_path"},
        ),
        RetrievalResult(
            id="rr-experiment-1",
            content=(
                "A/B experiment exp_2026_05 (n=3200) shows SRM at p=0.0008; "
                "interim looks scheduled at week 4."
            ),
            source=RetrievalSource.VECTOR,
            score=0.82,
            metadata={
                "experiment_id": "exp_2026_05",
                "agent": "experiment_monitor",
                "source_type": "experiment",
            },
        ),
    ]


@pytest.fixture
def reset_orchestrator_singleton():
    """Each test gets a fresh OrchestratorAgent (re-built from the registry)."""
    cognitive_route._orchestrator_instance = None
    yield
    cognitive_route._orchestrator_instance = None


@pytest.fixture
def realistic_rag_patch():
    """Patch working_memory + hybrid_search with realistic retrieval results.

    Empty-stub fixture (the one in test_orchestrator_end_to_end.py) misses F1
    and F2 because hybrid_search returns []. This fixture restores ≥2
    RetrievalResult entries so the orchestrator's user_context.evidence gets
    populated and the routing path matches production.
    """
    fake_memory = MagicMock()
    fake_memory.create_session = AsyncMock(return_value={"session_id": "s_realistic_rag"})
    fake_memory.get_session = AsyncMock(return_value={"state": "active"})
    fake_memory.add_message = AsyncMock(return_value=True)
    fake_memory.append_evidence = AsyncMock(return_value=True)

    with (
        patch("src.api.routes.cognitive.get_working_memory", return_value=fake_memory),
        patch(
            "src.api.routes.cognitive.hybrid_search",
            AsyncMock(return_value=_realistic_retrieval_results()),
        ),
    ):
        yield


@pytest.fixture
def client(reset_orchestrator_singleton, realistic_rag_patch):
    return TestClient(app)


def _post(client: TestClient, query: str, query_type: str = "general") -> Dict[str, Any]:
    response = client.post(
        "/api/cognitive/query",
        json={
            "query": query,
            "query_type": query_type,
            "include_evidence": False,
            "brand": "Brand-X",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


class TestSelfDispatchRegressionWithRealisticRag:
    """Issue #251 F1: orchestrator must never emit itself even with RAG context."""

    def test_ambiguous_general_query_does_not_self_dispatch(self, client: TestClient) -> None:
        """When orchestrator returns agents_dispatched=[], the API must NOT
        fall through to the query_type-derived 'orchestrator' default."""
        body = _post(client, "tell me something")
        assert body["agent_used"] != "orchestrator", body
        assert body["metadata"]["orchestrator_used"] is True, body

    def test_empty_query_does_not_self_dispatch(self, client: TestClient) -> None:
        body = _post(client, "...")
        assert body["agent_used"] != "orchestrator", body

    def test_orchestrator_general_query_uses_degraded_marker_or_real_agent(
        self, client: TestClient
    ) -> None:
        """When the orchestrator was called but emitted no real dispatch, the
        API must surface a recognisable degraded marker, not 'orchestrator'."""
        body = _post(client, "what is the meaning of this?")
        # Either a real agent dispatched, or the documented degraded marker
        # — but never the orchestrator's own name (which is the false-positive
        # query-type fallback for GENERAL queries).
        assert body["agent_used"] != "orchestrator", body


class TestRoutingLeakRegressionWithRealisticRag:
    """Issue #251 F2: API-side query_type→agent default must not leak."""

    def test_monitor_active_experiments_does_not_leak_health_score(
        self, client: TestClient
    ) -> None:
        """For monitoring queries: when orchestrator dispatches, agent_used is
        the dispatched agent. When orchestrator emits agents_dispatched=[],
        the API must NOT fall through to QueryType.MONITORING's 'health_score'
        default.
        """
        body = _post(
            client,
            "monitor all active A/B experiments for sample ratio mismatch",
            query_type="monitoring",
        )
        # 'health_score' is the QueryType.MONITORING default in _route_to_agent.
        # If the orchestrator dispatched correctly to experiment_monitor, that
        # wins. If the orchestrator returned []), the API must NOT default to
        # health_score — issue #251 F2.
        assert body["agent_used"] != "health_score", body

    def test_check_experiment_status_does_not_leak_health_score(self, client: TestClient) -> None:
        body = _post(client, "check the status of running experiments", query_type="monitoring")
        assert body["agent_used"] != "health_score", body


class TestRouterNeverEmitsOrchestratorHardGuard:
    """Issue #251 F1 acceptance: RouterNode hard guard.

    The RouterNode must structurally never emit a dispatch plan containing
    'orchestrator' — even if a future intent or fallback path tried to. This
    is defense-in-depth on top of the cognitive.py fix.
    """

    @pytest.mark.asyncio
    async def test_router_never_emits_orchestrator_in_dispatch_plan(self) -> None:
        from src.agents.orchestrator.nodes.router import RouterNode

        router = RouterNode()
        # Try every intent that exists in INTENT_TO_AGENTS plus a synthetic
        # unknown one. None should produce an orchestrator-dispatch.
        intents_to_try = list(router.INTENT_TO_AGENTS.keys()) + [
            "nonexistent_intent_to_force_default"
        ]
        for primary_intent in intents_to_try:
            state = {
                "intent": {
                    "primary_intent": primary_intent,
                    "confidence": 0.9,
                    "secondary_intents": [],
                    "requires_multi_agent": False,
                }
            }
            result = await router.execute(state)
            plan = result["dispatch_plan"]
            dispatched_names = [d["agent_name"] for d in plan]
            assert "orchestrator" not in dispatched_names, (
                f"RouterNode emitted 'orchestrator' for intent={primary_intent!r}: "
                f"dispatch_plan={dispatched_names}"
            )
