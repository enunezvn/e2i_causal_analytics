"""
Tests for Memory API endpoints.

Tests the memory system endpoints:
- POST /memory/search
- POST /memory/episodic
- GET /memory/episodic/{id}
- POST /memory/procedural/feedback
- GET /memory/semantic/paths
- GET /memory/stats
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.rag.models.retrieval_models import RetrievalResult

client = TestClient(app)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_hybrid_search():
    """Mock hybrid_search function."""
    with patch("src.api.routes.memory.hybrid_search") as mock:
        mock.return_value = [
            RetrievalResult(
                content="TRx dropped due to HCP engagement decline",
                source="episodic_memories",
                source_id="mem_1",
                score=0.85,
                retrieval_method="dense",
                metadata={"brand": "Kisqali"},
            ),
            RetrievalResult(
                content="Causal path: HCP visits → Script volume → TRx",
                source="causal_paths",
                source_id="path_1",
                score=0.75,
                retrieval_method="sparse",
                metadata={},
            ),
        ]
        yield mock


@pytest.fixture
def mock_episodic_memory_functions():
    """Mock episodic memory functions."""
    with (
        patch("src.api.routes.memory.insert_episodic_memory_with_text") as mock_insert,
        patch("src.api.routes.memory.get_memory_by_id") as mock_get,
    ):
        mock_insert.return_value = "mem_123"
        mock_get.return_value = {
            "memory_id": "mem_123",
            "description": "Test memory content",
            "event_type": "query",
            "session_id": "sess_abc",
            "agent_name": "orchestrator",
            "brand": "Kisqali",
            "region": "northeast",
            "raw_content": {},
            "occurred_at": "2025-01-01T00:00:00",
        }
        yield {"insert": mock_insert, "get": mock_get}


@pytest.fixture
def mock_procedural_memory_functions():
    """Mock procedural memory functions."""
    with (
        patch("src.api.routes.memory.update_procedure_outcome") as mock_update,
        patch("src.api.routes.memory.get_procedure_by_id") as mock_get,
        patch("src.api.routes.memory.record_learning_signal") as mock_signal,
    ):
        mock_update.return_value = None
        mock_get.return_value = {"procedure_id": "proc_001", "usage_count": 10, "success_count": 9}
        mock_signal.return_value = "signal_123"
        yield {"update": mock_update, "get": mock_get, "signal": mock_signal}


@pytest.fixture
def mock_semantic_memory():
    """Mock semantic memory."""
    memory = MagicMock()
    memory.find_causal_paths_for_kpi = MagicMock(
        return_value=[
            {
                "nodes": ["HCP engagement", "Script volume", "TRx"],
                "confidence": 0.85,
                "path_id": "path_1",
            }
        ]
    )
    memory.traverse_causal_chain = MagicMock(
        return_value=[{"path": ["Entity A", "Entity B"], "confidence": 0.8}]
    )
    return memory


# =============================================================================
# SEARCH ENDPOINT TESTS
# =============================================================================


class TestMemorySearch:
    """Tests for POST /memory/search."""

    def test_search_returns_results(self, mock_hybrid_search):
        """search should return hybrid search results."""
        response = client.post(
            "/memory/search", json={"query": "Why did TRx drop in northeast?", "k": 10}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] == 2
        assert len(data["results"]) == 2
        assert data["query"] == "Why did TRx drop in northeast?"

    def test_search_with_filters(self, mock_hybrid_search):
        """search should pass filters to hybrid_search."""
        response = client.post(
            "/memory/search",
            json={
                "query": "TRx trends",
                "k": 5,
                "filters": {"brand": "Kisqali", "region": "northeast"},
            },
        )

        assert response.status_code == 200
        mock_hybrid_search.assert_called_once()
        call_kwargs = mock_hybrid_search.call_args[1]
        assert call_kwargs["filters"] == {"brand": "Kisqali", "region": "northeast"}

    def test_search_with_kpi_name(self, mock_hybrid_search):
        """search should pass kpi_name for targeted retrieval."""
        response = client.post(
            "/memory/search", json={"query": "What impacts TRx?", "kpi_name": "TRx"}
        )

        assert response.status_code == 200
        call_kwargs = mock_hybrid_search.call_args[1]
        assert call_kwargs["kpi_name"] == "TRx"

    def test_search_filters_by_min_score(self, mock_hybrid_search):
        """search should filter results below min_score."""
        response = client.post("/memory/search", json={"query": "TRx analysis", "min_score": 0.8})

        assert response.status_code == 200
        data = response.json()
        # Only the first result (0.85) should pass the 0.8 threshold
        assert data["total_results"] == 1
        assert data["results"][0]["score"] >= 0.8

    def test_search_includes_latency(self, mock_hybrid_search):
        """search should include search latency in response."""
        response = client.post("/memory/search", json={"query": "test query"})

        assert response.status_code == 200
        data = response.json()
        assert "search_latency_ms" in data
        assert data["search_latency_ms"] >= 0

    def test_search_validates_query_length(self):
        """search should reject empty queries."""
        response = client.post("/memory/search", json={"query": ""})

        assert response.status_code == 422  # Validation error


# =============================================================================
# EPISODIC MEMORY TESTS
# =============================================================================


class TestEpisodicMemory:
    """Tests for episodic memory endpoints."""

    def test_create_episodic_memory(self, mock_episodic_memory_functions):
        """POST /memory/episodic should create a new memory."""
        response = client.post(
            "/memory/episodic",
            json={
                "content": "User asked about TRx trends",
                "event_type": "query",
                "session_id": "sess_123",
                "agent_name": "orchestrator",
                "brand": "Kisqali",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "mem_123"
        assert data["content"] == "User asked about TRx trends"
        assert data["event_type"] == "query"

    def test_create_episodic_memory_minimal(self, mock_episodic_memory_functions):
        """POST /memory/episodic should work with minimal fields."""
        response = client.post(
            "/memory/episodic", json={"content": "Minimal memory", "event_type": "action"}
        )

        assert response.status_code == 200

    def test_get_episodic_memory_by_id(self, mock_episodic_memory_functions):
        """GET /memory/episodic/{id} should retrieve a memory."""
        response = client.get("/memory/episodic/mem_123")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "mem_123"
        assert data["content"] == "Test memory content"
        assert data["brand"] == "Kisqali"

    def test_get_episodic_memory_not_found(self):
        """GET /memory/episodic/{id} should return 404 for missing memory."""
        with patch("src.api.routes.memory.get_memory_by_id") as mock_get:
            mock_get.return_value = None
            response = client.get("/memory/episodic/nonexistent")

        assert response.status_code == 404


# =============================================================================
# PROCEDURAL MEMORY TESTS
# =============================================================================


class TestProceduralFeedback:
    """Tests for POST /memory/procedural/feedback."""

    def test_record_feedback_success(self, mock_procedural_memory_functions):
        """Should record feedback and return new success rate."""
        response = client.post(
            "/memory/procedural/feedback",
            json={
                "procedure_id": "proc_001",
                "outcome": "success",
                "score": 0.9,
                "feedback_text": "Analysis was accurate",
                "agent_name": "feedback_learner",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["procedure_id"] == "proc_001"
        assert data["feedback_recorded"] is True
        # 9 successes / 10 usage = 0.9 success rate
        assert data["new_success_rate"] == 0.9
        # Verify mocks were called
        mock_procedural_memory_functions["update"].assert_called_once()
        mock_procedural_memory_functions["signal"].assert_called_once()

    def test_record_feedback_minimal(self, mock_procedural_memory_functions):
        """Should work with minimal required fields."""
        response = client.post(
            "/memory/procedural/feedback",
            json={"procedure_id": "proc_002", "outcome": "partial", "score": 0.6},
        )

        assert response.status_code == 200


# =============================================================================
# SEMANTIC PATH TESTS
# =============================================================================


class TestSemanticPaths:
    """Tests for GET /memory/semantic/paths."""

    def test_query_paths_by_kpi(self, mock_semantic_memory):
        """Should find paths for a given KPI."""
        with patch("src.api.routes.memory.get_semantic_memory", return_value=mock_semantic_memory):
            response = client.get(
                "/memory/semantic/paths", params={"kpi_name": "TRx", "min_confidence": 0.6}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_paths"] == 1
        assert "query_latency_ms" in data
        mock_semantic_memory.find_causal_paths_for_kpi.assert_called_once_with(
            kpi_name="TRx", min_confidence=0.6
        )

    def test_query_paths_by_entity(self, mock_semantic_memory):
        """Should traverse from a starting entity."""
        with patch("src.api.routes.memory.get_semantic_memory", return_value=mock_semantic_memory):
            response = client.get(
                "/memory/semantic/paths", params={"start_entity_id": "ent_001", "max_depth": 2}
            )

        assert response.status_code == 200
        mock_semantic_memory.traverse_causal_chain.assert_called_once()


# =============================================================================
# STATS ENDPOINT TESTS
# =============================================================================


class TestMemoryStats:
    """Tests for GET /memory/stats."""

    def test_get_stats_returns_structure(self):
        """Should return stats for all memory types."""
        response = client.get("/memory/stats")

        assert response.status_code == 200
        data = response.json()
        assert "episodic" in data
        assert "procedural" in data
        assert "semantic" in data
        assert "last_updated" in data
