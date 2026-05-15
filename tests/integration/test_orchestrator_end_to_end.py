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
        # Real registry dispatches to causal_impact (or gap_analyzer for
        # gap-flavoured causal queries).
        assert body["agent_used"] in {
            "causal_impact",
            "gap_analyzer",
            "tool_composer",  # multi_faceted fan-in is also acceptable
        }, body

    def test_experiment_design_query_routes_to_experiment_designer(
        self, client: TestClient
    ) -> None:
        body = _post_query(
            client,
            "Design an A/B test to evaluate a new HCP detailing cadence.",
            query_type="general",
        )
        assert body["metadata"]["orchestrator_used"] is True, body
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
        assert body["agent_used"] in {
            "experiment_monitor",
            "drift_monitor",
            "tool_composer",
        }, body


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
