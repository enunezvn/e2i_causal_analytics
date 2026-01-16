"""
Tests for Cognitive Workflow API endpoints.

Tests the cognitive workflow endpoints:
- POST /cognitive/query
- GET /cognitive/session/{id}
- POST /cognitive/session
- DELETE /cognitive/session/{id}
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.rag.models.retrieval_models import RetrievalResult

client = TestClient(app)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_working_memory():
    """Mock working memory."""
    memory = MagicMock()
    memory.create_session = AsyncMock(return_value={"session_id": "sess_123"})
    memory.get_session = AsyncMock(
        return_value={
            "user_id": "user_001",
            "context": {"brand": "Kisqali", "region": "northeast"},
            "state": "active",
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
        }
    )
    memory.add_message = AsyncMock(return_value=True)
    memory.get_messages = AsyncMock(
        return_value=[
            {
                "role": "user",
                "content": "Test query",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {},
            },
            {
                "role": "assistant",
                "content": "Test response",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {"agent_name": "orchestrator"},
            },
        ]
    )
    memory.append_evidence = AsyncMock(return_value=True)
    memory.get_evidence_trail = AsyncMock(
        return_value=[
            {
                "content": "Evidence 1",
                "source": "episodic_memories",
                "score": 0.85,
                "retrieval_method": "dense",
            }
        ]
    )
    memory.delete_session = AsyncMock(return_value=True)
    return memory


@pytest.fixture
def mock_hybrid_search():
    """Mock hybrid_search function."""
    with patch("src.api.routes.cognitive.hybrid_search") as mock:
        mock.return_value = [
            RetrievalResult(
                content="Relevant memory content",
                source="episodic_memories",
                source_id="mem_1",
                score=0.85,
                retrieval_method="dense",
                metadata={},
            )
        ]
        yield mock


# =============================================================================
# COGNITIVE QUERY TESTS
# =============================================================================


class TestCognitiveQuery:
    """Tests for POST /cognitive/query."""

    def test_process_query_new_session(self, mock_working_memory, mock_hybrid_search):
        """Should process query and create new session."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = client.post(
                "/api/cognitive/query",
                json={
                    "query": "Why did TRx drop in northeast region?",
                    "brand": "Kisqali",
                    "region": "northeast",
                    "include_evidence": True,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["query"] == "Why did TRx drop in northeast region?"
        assert data["query_type"] == "causal"  # Auto-detected from "why"
        assert data["agent_used"] == "causal_impact"
        assert "phases_completed" in data
        assert len(data["phases_completed"]) == 5  # All phases

    def test_process_query_existing_session(self, mock_working_memory, mock_hybrid_search):
        """Should continue existing session."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = client.post(
                "/api/cognitive/query",
                json={"query": "What else impacts this?", "session_id": "sess_existing"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "sess_existing"

    def test_process_query_with_evidence(self, mock_working_memory, mock_hybrid_search):
        """Should include evidence when requested."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = client.post(
                "/api/cognitive/query", json={"query": "Analyze TRx trends", "include_evidence": True}
            )

        assert response.status_code == 200
        data = response.json()
        assert "evidence" in data
        assert data["evidence"] is not None
        assert len(data["evidence"]) > 0

    def test_process_query_without_evidence(self, mock_working_memory, mock_hybrid_search):
        """Should exclude evidence when not requested."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = client.post(
                "/api/cognitive/query", json={"query": "Quick question", "include_evidence": False}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["evidence"] is None

    def test_process_query_includes_latency(self, mock_working_memory, mock_hybrid_search):
        """Should include processing time."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = client.post("/api/cognitive/query", json={"query": "Test query"})

        assert response.status_code == 200
        data = response.json()
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] >= 0

    def test_query_type_detection_prediction(self, mock_working_memory, mock_hybrid_search):
        """Should detect prediction query type."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = client.post(
                "/api/cognitive/query", json={"query": "What will TRx be next quarter?"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["query_type"] == "prediction"
        assert data["agent_used"] == "prediction_synthesizer"

    def test_query_type_detection_optimization(self, mock_working_memory, mock_hybrid_search):
        """Should detect optimization query type."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = client.post(
                "/api/cognitive/query", json={"query": "How can we optimize resource allocation?"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["query_type"] == "optimization"
        assert data["agent_used"] == "resource_optimizer"

    def test_query_type_explicit(self, mock_working_memory, mock_hybrid_search):
        """Should use explicit query type when provided."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = client.post(
                "/api/cognitive/query", json={"query": "Analyze this data", "query_type": "monitoring"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["query_type"] == "monitoring"


# =============================================================================
# SESSION ENDPOINT TESTS
# =============================================================================


class TestSessionManagement:
    """Tests for session management endpoints."""

    def test_create_session(self, mock_working_memory):
        """POST /cognitive/session should create new session."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = client.post(
                "/api/cognitive/session",
                json={"user_id": "user_001", "brand": "Kisqali", "region": "northeast"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["state"] == "active"
        assert "expires_at" in data

    def test_create_session_minimal(self, mock_working_memory):
        """POST /cognitive/session should work with no body."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = client.post("/api/cognitive/session", json={})

        assert response.status_code == 200

    def test_get_session(self, mock_working_memory):
        """GET /cognitive/session/{id} should return session state."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = client.get("/api/cognitive/session/sess_123")

        assert response.status_code == 200
        data = response.json()
        assert "context" in data
        assert "messages" in data
        assert "evidence_trail" in data
        assert data["context"]["session_id"] == "sess_123"

    def test_get_session_not_found(self, mock_working_memory):
        """GET /cognitive/session/{id} should return 404 for missing session."""
        mock_working_memory.get_session.return_value = None

        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = client.get("/api/cognitive/session/nonexistent")

        assert response.status_code == 404

    def test_get_session_includes_messages(self, mock_working_memory):
        """GET /cognitive/session/{id} should include message history."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = client.get("/api/cognitive/session/sess_123")

        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"

    def test_delete_session(self, mock_working_memory):
        """DELETE /cognitive/session/{id} should delete session."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = client.delete("/api/cognitive/session/sess_123")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "sess_123"
        assert data["deleted"] is True
        mock_working_memory.delete_session.assert_called_once_with("sess_123")


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestHelperFunctions:
    """Tests for cognitive endpoint helper functions."""

    def test_kpi_extraction(self, mock_working_memory, mock_hybrid_search):
        """Should extract KPI from query for targeted retrieval."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = client.post("/api/cognitive/query", json={"query": "Why did TRx drop?"})

        assert response.status_code == 200
        # Verify hybrid_search was called with kpi_name
        mock_hybrid_search.assert_called_once()
        call_kwargs = mock_hybrid_search.call_args[1]
        assert call_kwargs["kpi_name"] == "TRx"

    def test_filter_building(self, mock_working_memory, mock_hybrid_search):
        """Should build filters from brand and region."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = client.post(
                "/api/cognitive/query",
                json={"query": "Analyze data", "brand": "Kisqali", "region": "northeast"},
            )

        assert response.status_code == 200
        call_kwargs = mock_hybrid_search.call_args[1]
        assert call_kwargs["filters"] == {"brand": "Kisqali", "region": "northeast"}


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_query_validation_error(self):
        """Should return 422 for invalid request."""
        response = client.post("/api/cognitive/query", json={"query": ""})  # Empty query

        assert response.status_code == 422

    def test_query_handles_memory_error(self, mock_working_memory, mock_hybrid_search):
        """Should return 500 when memory operations fail."""
        mock_working_memory.create_session.side_effect = Exception("Memory error")

        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = client.post("/api/cognitive/query", json={"query": "Test query"})

        assert response.status_code == 500
        assert "Query processing failed" in response.json()["detail"]
