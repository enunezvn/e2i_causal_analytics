"""Falsifiability-verified unit tests for cognitive.py's #251 F1+F2 fix.

These tests force the EXACT code paths that the cognitive.py degraded-marker
change introduced:

* When ``orchestrator.run()`` returns ``agents_dispatched=[]`` (empty),
  ``agent_used`` must be ``"orchestrator_degraded"`` instead of the
  query_type-derived default (``_route_to_agent(query_type)``).

* When ``orchestrator.run()`` raises, same contract: ``agent_used`` must be
  ``"orchestrator_degraded"`` instead of the query_type-derived default.

The realistic_rag tests in ``test_orchestrator_realistic_rag.py`` pass for
an INDEPENDENT reason (the router's explainer fallback handles the
"general" intent so ``agents_dispatched=["explainer"]`` is non-empty), so
the new branches are NEVER exercised there. The ralph-loop gate flagged
these as HEDGE with a falsifiability check: "revert cognitive.py:365-372;
F1 tests still pass."

These tests are the load-bearing assertion. They mock get_orchestrator()
directly so the orchestrator's internal routing is not on the path —
exactly the branches the fix added are exercised.

See feedback_verification_step_evidence_gate.
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
    cognitive_route._orchestrator_instance = None
    yield
    cognitive_route._orchestrator_instance = None


@pytest.fixture
def empty_dispatch_orchestrator():
    """Mock get_orchestrator() to return an orchestrator whose .run() returns
    ``agents_dispatched=[]``. Forces the new ``else: agent_name =
    "orchestrator_degraded"`` branch at cognitive.py:365-372.
    """
    fake_orch = MagicMock()
    fake_orch.run = AsyncMock(
        return_value={
            "response_text": "(degraded — no dispatch produced)",
            "response_confidence": 0.4,
            "agents_dispatched": [],
        }
    )
    with patch("src.api.routes.cognitive.get_orchestrator", return_value=fake_orch):
        yield fake_orch


@pytest.fixture
def raising_orchestrator():
    """Mock get_orchestrator() to return one whose .run() raises. Forces the
    cognitive.py except-block fix to set agent_name = 'orchestrator_degraded'.
    """
    fake_orch = MagicMock()
    fake_orch.run = AsyncMock(side_effect=RuntimeError("graph compile failed"))
    with patch("src.api.routes.cognitive.get_orchestrator", return_value=fake_orch):
        yield fake_orch


@pytest.fixture
def mocked_memory_and_rag():
    """Standard memory/RAG stubs — these aren't the variable under test here."""
    fake_memory = MagicMock()
    fake_memory.create_session = AsyncMock(return_value={"session_id": "s_degraded"})
    fake_memory.get_session = AsyncMock(return_value={"state": "active"})
    fake_memory.add_message = AsyncMock(return_value=True)
    fake_memory.append_evidence = AsyncMock(return_value=True)
    with (
        patch("src.api.routes.cognitive.get_working_memory", return_value=fake_memory),
        patch("src.api.routes.cognitive.hybrid_search", AsyncMock(return_value=[])),
    ):
        yield


@pytest.fixture
def client(reset_orchestrator_singleton, mocked_memory_and_rag):
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


class TestEmptyDispatchSurfacesDegradedMarker:
    """Issue #251 F1+F2: orchestrator.run() returned agents_dispatched=[].

    Falsifiability: revert cognitive.py's ``else: agent_name =
    "orchestrator_degraded"`` branch and these tests trip with
    ``agent_used == "orchestrator"`` (F1) or ``"health_score"`` (F2).
    """

    def test_general_query_with_empty_dispatch_returns_degraded_marker(
        self, client: TestClient, empty_dispatch_orchestrator
    ) -> None:
        """F1 path: QueryType.GENERAL → _route_to_agent → "orchestrator".

        Without the fix: agent_used="orchestrator" (the leak).
        With the fix: agent_used="orchestrator_degraded".
        """
        body = _post(client, "tell me something", query_type="general")
        assert body["agent_used"] == "orchestrator_degraded", body
        assert body["metadata"]["orchestrator_used"] is True, body

    def test_monitoring_query_with_empty_dispatch_returns_degraded_marker(
        self, client: TestClient, empty_dispatch_orchestrator
    ) -> None:
        """F2 path: QueryType.MONITORING → _route_to_agent → "health_score".

        Without the fix: agent_used="health_score" (the leak).
        With the fix: agent_used="orchestrator_degraded".
        """
        body = _post(client, "monitor all active experiments", query_type="monitoring")
        assert body["agent_used"] == "orchestrator_degraded", body
        assert body["agent_used"] != "health_score", body

    def test_causal_query_with_empty_dispatch_returns_degraded_marker(
        self, client: TestClient, empty_dispatch_orchestrator
    ) -> None:
        """F2 path: QueryType.CAUSAL → _route_to_agent → "causal_impact".

        Without the fix: agent_used="causal_impact" (mislabel — orchestrator
        actually did NOT dispatch causal_impact).
        With the fix: agent_used="orchestrator_degraded".
        """
        body = _post(client, "what drives discontinuation?", query_type="causal")
        assert body["agent_used"] == "orchestrator_degraded", body


class TestRaisedOrchestratorSurfacesDegradedMarker:
    """Issue #251 F1 live-Docker path: orchestrator.run() raised an exception.

    cognitive.py's except block previously set response_text via placeholder
    but left agent_name at the _route_to_agent default. Live Docker exposed
    this as agent_used="orchestrator" with the placeholder response text.

    Falsifiability: revert cognitive.py's ``agent_name =
    "orchestrator_degraded"`` line in the except block and these tests trip.
    """

    def test_general_query_with_raised_orchestrator_returns_degraded_marker(
        self, client: TestClient, raising_orchestrator
    ) -> None:
        """The exact live Docker symptom from issue #251 F1."""
        body = _post(client, "tell me something", query_type="general")
        assert body["agent_used"] == "orchestrator_degraded", body
        assert body["agent_used"] != "orchestrator", body
        # Response text should be the placeholder (kept for backward compat).
        assert body["response"], "response_text must not be empty"

    def test_monitoring_query_with_raised_orchestrator_returns_degraded_marker(
        self, client: TestClient, raising_orchestrator
    ) -> None:
        """F2-equivalent for the exception path."""
        body = _post(client, "check SRM in active experiments", query_type="monitoring")
        assert body["agent_used"] == "orchestrator_degraded", body
        assert body["agent_used"] != "health_score", body
