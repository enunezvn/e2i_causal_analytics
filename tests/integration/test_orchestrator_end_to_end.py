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


@pytest.fixture
def reset_orchestrator_singleton():
    """Each test gets a fresh OrchestratorAgent (re-built from the registry)."""
    cognitive_route._orchestrator_instance = None
    yield
    cognitive_route._orchestrator_instance = None


@pytest.fixture
def mock_memory_and_rag():
    """Stub working_memory + hybrid_search so the test focuses on dispatch."""
    fake_memory = MagicMock()
    fake_memory.create_session = AsyncMock(return_value={"session_id": "s_test"})
    fake_memory.get_session = AsyncMock(return_value={"state": "active"})
    fake_memory.add_message = AsyncMock(return_value=True)
    fake_memory.append_evidence = AsyncMock(return_value=True)

    with (
        patch("src.api.routes.cognitive.get_working_memory", return_value=fake_memory),
        patch("src.api.routes.cognitive.hybrid_search", AsyncMock(return_value=[])),
    ):
        yield


@pytest.fixture
def client(reset_orchestrator_singleton, mock_memory_and_rag):
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
