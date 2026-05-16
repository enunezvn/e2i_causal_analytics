"""Phase 5 verification: /api/cognitive/query routes through real orchestrator.

These tests assert that the cognitive endpoint constructs an OrchestratorAgent
with a real agent registry (via ``create_agent_registry``) and that the
resulting dispatch reaches real agents — not the canned mock narratives that
``dispatcher._mock_agent_execution`` used to emit when the registry was empty.

The memory / RAG side of the request is mocked because it is not the Phase-5
concern; the test concentrates on:

* ``metadata.orchestrator_used == True``  (registry wired)
* ``agent_used`` is one of the real agent names in the registry
* causal queries → ``causal_impact`` / ``gap_analyzer``
* A/B-design queries → ``experiment_designer``
* monitor / SRM queries → ``experiment_monitor``
* response body is non-empty
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


@pytest.fixture
def reset_orchestrator_singleton():
    """Each test gets a fresh OrchestratorAgent (re-built from the registry)."""
    cognitive_route._orchestrator_instance = None
    yield
    cognitive_route._orchestrator_instance = None


def _fake_memory() -> MagicMock:
    """Shared working_memory mock for all fixtures."""
    fake_memory = MagicMock()
    fake_memory.create_session = AsyncMock(return_value={"session_id": "s_test"})
    fake_memory.get_session = AsyncMock(return_value={"state": "active"})
    fake_memory.add_message = AsyncMock(return_value=True)
    fake_memory.append_evidence = AsyncMock(return_value=True)
    return fake_memory


@pytest.fixture
def mock_memory_and_rag():
    """Stub working_memory + hybrid_search → [] so the test focuses on dispatch.

    This is the ``empty_rag`` fixture per memory feedback
    [[feedback-testclient-vs-live-divergence]] — fast, hides the RAG-context
    code path. Use ``realistic_rag`` for tests that must catch runtime
    composition gaps (Issue #251).
    """
    with (
        patch("src.api.routes.cognitive.get_working_memory", return_value=_fake_memory()),
        patch("src.api.routes.cognitive.hybrid_search", AsyncMock(return_value=[])),
    ):
        yield


def _realistic_retrieval_results() -> list[RetrievalResult]:
    """≥2 domain-matched RetrievalResult entries with non-trivial content + score.

    Surfaces the RAG-context injection code path in
    ``src/api/routes/cognitive.py`` ``user_context.evidence``. Live Docker
    behaves identically — only the data source differs.
    """
    return [
        RetrievalResult(
            id="rag_remi_csu_1",
            content=(
                "Remibrutinib CSU launch readiness — KPI scorecard for HCP detailing "
                "cadence; baseline conversion rate 0.42; high-priority HCP cohort "
                "n=312; experiment design eligible."
            ),
            source=RetrievalSource.VECTOR,
            score=0.91,
            metadata={"retrieval_method": "vector", "brand": "Remibrutinib"},
            raw_score=0.91,
        ),
        RetrievalResult(
            id="rag_ab_test_srm_1",
            content=(
                "A/B experiment monitor: sample ratio mismatch (SRM) flagged on "
                "Trial-2024-Q3 (expected 50/50, observed 47/53, p=0.018). Interim "
                "analysis recommended."
            ),
            source=RetrievalSource.FULLTEXT,
            score=0.87,
            metadata={"retrieval_method": "fulltext"},
            raw_score=0.87,
        ),
    ]


@pytest.fixture
def realistic_rag():
    """Realistic ≥2 RetrievalResult entries — for routing/composition tests.

    Memory note [[feedback-testclient-vs-live-divergence]]: TestClient with
    ``hybrid_search=[]`` can pass while live Docker fails the same routing
    path because the orchestrator's ``user_context.evidence`` field is empty.
    This fixture reproduces the live-Docker context shape.
    """
    with (
        patch("src.api.routes.cognitive.get_working_memory", return_value=_fake_memory()),
        patch(
            "src.api.routes.cognitive.hybrid_search",
            AsyncMock(return_value=_realistic_retrieval_results()),
        ),
    ):
        yield


@pytest.fixture
def client(reset_orchestrator_singleton, mock_memory_and_rag):
    return TestClient(app)


@pytest.fixture
def client_realistic_rag(reset_orchestrator_singleton, realistic_rag):
    """TestClient with realistic_rag fixture wired."""
    return TestClient(app)


def _post_query(client: TestClient, query: str, query_type: str = "causal") -> Dict[str, Any]:
    response = client.post(
        "/api/cognitive/query",
        json={
            "query": query,
            "query_type": query_type,
            "include_evidence": False,
            "brand": "TestBrand",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


class TestOrchestratorEndToEnd:
    def test_causal_query_uses_real_orchestrator(self, client: TestClient) -> None:
        body = _post_query(
            client,
            "What drives discontinuation for patients on Brand-X?",
            query_type="causal",
        )

        assert body["metadata"]["orchestrator_used"] is True, body
        assert body["response"], "response text must not be empty"
        # Real registry dispatches to causal_impact (the orchestrator
        # consistently picks this for causal-flavoured queries in the live
        # Docker smoke; gap_analyzer is the alternate when wording is gap-
        # flavoured, but for this canonical phrasing causal_impact is the
        # contract).
        assert body["agent_used"] == "causal_impact", body

    def test_experiment_design_query_routes_to_experiment_designer(
        self, client: TestClient
    ) -> None:
        body = _post_query(
            client,
            "Design an A/B test to evaluate a new HCP detailing cadence.",
            query_type="general",
        )
        assert body["metadata"]["orchestrator_used"] is True, body
        # Tight: only experiment_designer is acceptable. tool_composer is
        # acceptable when the multi_faceted detector fires. The orchestrator
        # must never name itself as the responding agent (see
        # ``test_orchestrator_never_dispatches_to_itself`` regression guard).
        assert body["agent_used"] in {"experiment_designer", "tool_composer"}, body

    def test_experiment_monitor_query_routes_to_experiment_monitor(
        self, client: TestClient
    ) -> None:
        body = _post_query(
            client,
            "Check all active A/B experiments for sample ratio mismatch.",
            query_type="monitoring",
        )
        assert body["metadata"]["orchestrator_used"] is True, body
        # Tight: experiment_monitor is the correct answer. drift_monitor is
        # acceptable as a near miss (both tier-3 monitors); health_score is
        # explicitly NOT — the live-Docker smoke surfaced that path as a
        # latent API-default-leak bug (tracked in issue #251).
        assert body["agent_used"] in {"experiment_monitor", "drift_monitor"}, body
        assert body["agent_used"] != "health_score", body

    def test_orchestrator_never_dispatches_to_itself(self, client: TestClient) -> None:
        """Regression guard: 'orchestrator' must never appear as agent_used.

        Even when intent classification fails or the dispatch plan is empty,
        the API must not surface the orchestrator's own name as the
        responding agent. The live-Docker smoke caught the orchestrator
        emitting itself for ambiguous 'general' queries when real RAG context
        was attached — this guard locks the contract.
        """
        body = _post_query(client, "Tell me something.", query_type="general")
        assert body["metadata"]["orchestrator_used"] is True, body
        assert body["agent_used"] != "orchestrator", body


# ---------------------------------------------------------------------------
# Issue #251 — realistic_rag regression guards.
# These mirror the live-Docker behaviour by injecting non-empty RAG context,
# closing the TestClient↔live divergence loop.
# ---------------------------------------------------------------------------


class TestOrchestratorEndToEndWithRealisticRAG:
    """Same three smoke queries but with realistic_rag fixture wired.

    F1: ``orchestrator`` must never leak as ``agent_used``.
    F2: API ``query_type``-derived default must not leak when orchestrator
    returns ``agents_dispatched=[]``.
    F3: TestClient + realistic_rag reproduces the live-Docker context shape.
    """

    def test_orchestrator_never_dispatches_to_itself_with_real_rag(
        self, client_realistic_rag: TestClient
    ) -> None:
        body = _post_query(client_realistic_rag, "Tell me something.", query_type="general")
        assert body["metadata"]["orchestrator_used"] is True, body
        # F1: orchestrator must never name itself, even with real RAG evidence.
        assert body["agent_used"] != "orchestrator", body

    def test_experiment_design_query_routes_to_experiment_designer_with_real_rag(
        self, client_realistic_rag: TestClient
    ) -> None:
        body = _post_query(
            client_realistic_rag,
            "Design an A/B test to evaluate a new HCP detailing cadence.",
            query_type="general",
        )
        assert body["metadata"]["orchestrator_used"] is True, body
        # F1 + F2 acceptance: should resolve to a real Tier-3 designer agent.
        # Acceptable: experiment_designer (canonical), tool_composer (multi-
        # faceted). Forbidden: 'orchestrator' (F1) and the API GENERAL default
        # 'orchestrator' (F2 — same string).
        assert body["agent_used"] in {"experiment_designer", "tool_composer"}, body
        assert body["agent_used"] != "orchestrator", body

    def test_experiment_monitor_query_routes_to_experiment_monitor_with_real_rag(
        self, client_realistic_rag: TestClient
    ) -> None:
        body = _post_query(
            client_realistic_rag,
            "Check all active A/B experiments for sample ratio mismatch.",
            query_type="monitoring",
        )
        assert body["metadata"]["orchestrator_used"] is True, body
        # F2 acceptance: must NOT silently fall to ``health_score`` (the
        # MONITORING API default) when the orchestrator's dispatch is empty.
        assert body["agent_used"] != "health_score", body
        assert body["agent_used"] in {"experiment_monitor", "drift_monitor"}, body

    def test_monitoring_query_does_not_leak_api_default_when_dispatch_empty(
        self, client_realistic_rag: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Force orchestrator to emit ``agents_dispatched=[]`` and assert the
        API default ``health_score`` does NOT leak through.

        With the F2 fix in place, ``agent_used`` must be either ``null`` or
        the recognisable degraded marker ``"orchestrator_degraded"`` — never
        the ``_route_to_agent(query_type)`` value.
        """
        # Reset singleton so our patched orchestrator instance is fresh.
        cognitive_route._orchestrator_instance = None

        fake_orchestrator = MagicMock()
        fake_orchestrator.run = AsyncMock(
            return_value={
                "response_text": "degraded path",
                "response_confidence": 0.0,
                "agents_dispatched": [],
                "status": "completed",
            }
        )
        monkeypatch.setattr(cognitive_route, "get_orchestrator", lambda: fake_orchestrator)

        body = _post_query(
            client_realistic_rag,
            "Check all active A/B experiments for sample ratio mismatch.",
            query_type="monitoring",
        )
        # F2: must NOT silently fall to ``health_score`` (the API MONITORING
        # default) when orchestrator returns empty dispatch.
        assert body["agent_used"] != "health_score", body
        # Acceptable: null or recognisable degraded marker.
        assert body["agent_used"] in (None, "orchestrator_degraded"), body

    def test_general_query_does_not_leak_orchestrator_default_when_dispatch_empty(
        self, client_realistic_rag: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F1+F2 corner: GENERAL query_type's API default is the string
        ``"orchestrator"``. When the orchestrator emits empty dispatch, that
        default must not leak as ``agent_used`` either.
        """
        cognitive_route._orchestrator_instance = None

        fake_orchestrator = MagicMock()
        fake_orchestrator.run = AsyncMock(
            return_value={
                "response_text": "degraded path",
                "response_confidence": 0.0,
                "agents_dispatched": [],
                "status": "completed",
            }
        )
        monkeypatch.setattr(cognitive_route, "get_orchestrator", lambda: fake_orchestrator)

        body = _post_query(client_realistic_rag, "Tell me something.", query_type="general")
        # F1: never the literal 'orchestrator' string (the GENERAL API default).
        assert body["agent_used"] != "orchestrator", body
        assert body["agent_used"] in (None, "orchestrator_degraded"), body


class TestOrchestratorSingletonContract:
    """The singleton must be built with a non-empty real registry."""

    def test_orchestrator_singleton_has_real_registry(self, reset_orchestrator_singleton) -> None:
        orchestrator = cognitive_route.get_orchestrator()
        assert orchestrator is not None
        # Real registry attribute name varies; check both common spellings.
        registry: Dict[str, Any] = (
            getattr(orchestrator, "agent_registry", None)
            or getattr(orchestrator, "agents", None)
            or {}
        )
        assert isinstance(registry, dict)
        assert len(registry) >= 5, f"registry too small: {sorted(registry)}"
        # The orchestrator must NOT include itself in the dispatch registry.
        assert "orchestrator" not in registry


# ---------------------------------------------------------------------------
# Issue #251 — codex MED-2 regression guards.
#
# The original F2 fix routed the *empty-dispatch* path through the
# ``"orchestrator_degraded"`` marker. The codex follow-up gate caught
# two surviving leaks for ``query_type=GENERAL`` queries:
#
# * cognitive.py:408 — orchestrator threw → ``agent_name = _route_to_agent(query_type)``
# * cognitive.py:416 — orchestrator is None → same fall-through
#
# ``_route_to_agent`` at :630 maps ``QueryType.GENERAL → "orchestrator"``,
# so both paths re-introduce the F1 violation at the API boundary. The
# fix changes the GENERAL mapping (and any other agent-name string that
# matches ``SELF_AGENT_NAME``) to ``"orchestrator_degraded"`` / ``None``.
# ---------------------------------------------------------------------------


class TestApiFallbackNeverLeaksSelfDispatch:
    """codex MED-2 — `_route_to_agent` GENERAL default must not leak."""

    def test_route_to_agent_general_does_not_return_orchestrator(self) -> None:
        """Unit-level invariant on the helper itself."""
        from src.api.routes.cognitive import QueryType, _route_to_agent

        result = _route_to_agent(QueryType.GENERAL)
        assert result != "orchestrator", result

    def test_route_to_agent_never_returns_self_agent_name(self) -> None:
        """Audit ALL QueryType enum values — none may map to the
        forbidden self literal. This guards against future enum additions
        that pick up the historic GENERAL behaviour by copy-paste."""
        from src.agents.orchestrator._self_dispatch_guard import SELF_AGENT_NAME
        from src.api.routes.cognitive import QueryType, _route_to_agent

        for qt in QueryType:
            mapped = _route_to_agent(qt)
            assert mapped != SELF_AGENT_NAME, (
                f"QueryType.{qt.name} → {mapped!r} re-introduces F1 self-dispatch leak"
            )

    def test_general_query_with_orchestrator_exception_does_not_leak_orchestrator(
        self, client_realistic_rag: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """codex MED-2: cognitive.py:408 exception path with
        ``query_type=GENERAL`` previously returned ``agent_used="orchestrator"``
        via ``_route_to_agent(GENERAL) == "orchestrator"``. The fix must
        surface the degraded marker (or null), NEVER the self literal."""
        cognitive_route._orchestrator_instance = None

        fake_orchestrator = MagicMock()
        fake_orchestrator.run = AsyncMock(side_effect=RuntimeError("orchestrator crashed"))
        monkeypatch.setattr(cognitive_route, "get_orchestrator", lambda: fake_orchestrator)

        body = _post_query(
            client_realistic_rag,
            "Tell me something.",
            query_type="general",
        )
        # F1 invariant: must never leak the self literal.
        assert body["agent_used"] != "orchestrator", body
        # Acceptable: null or degraded marker.
        assert body["agent_used"] in (None, "orchestrator_degraded"), body

    def test_general_query_with_null_orchestrator_does_not_leak_orchestrator(
        self, client_realistic_rag: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """codex MED-2: cognitive.py:416 null-orchestrator path with
        ``query_type=GENERAL`` previously returned ``agent_used="orchestrator"``
        via the same ``_route_to_agent`` mapping."""
        cognitive_route._orchestrator_instance = None
        monkeypatch.setattr(cognitive_route, "get_orchestrator", lambda: None)

        body = _post_query(
            client_realistic_rag,
            "Tell me something.",
            query_type="general",
        )
        assert body["agent_used"] != "orchestrator", body
        assert body["agent_used"] in (None, "orchestrator_degraded"), body
