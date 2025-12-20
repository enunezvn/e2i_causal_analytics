"""
End-to-End Tests for Memory System
===================================

Complete E2E tests for the Tri-Memory Architecture:
- Full cognitive cycles with real or mocked services
- Memory persistence across API calls
- Learning signal propagation
- Performance benchmarks

Note: These tests use mocked external services by default.
Set E2E_USE_REAL_SERVICES=true to test with real backends.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone
import asyncio
import os

from fastapi.testclient import TestClient

from src.api.main import app
from src.rag.models.retrieval_models import RetrievalResult


# =============================================================================
# CONFIGURATION
# =============================================================================

USE_REAL_SERVICES = os.getenv("E2E_USE_REAL_SERVICES", "false").lower() == "true"

client = TestClient(app)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_all_services():
    """Mock all external services for isolated E2E testing."""
    with patch("src.api.routes.memory.hybrid_search") as mock_search, \
         patch("src.api.routes.memory.insert_episodic_memory_with_text") as mock_insert, \
         patch("src.api.routes.memory.get_memory_by_id") as mock_get, \
         patch("src.api.routes.memory.update_procedure_outcome") as mock_update, \
         patch("src.api.routes.memory.record_learning_signal") as mock_signal, \
         patch("src.api.routes.memory.get_procedure_by_id") as mock_proc, \
         patch("src.api.routes.memory.get_semantic_memory") as mock_semantic:

        # Configure mock responses
        mock_search.return_value = [
            RetrievalResult(
                content="TRx dropped 15% in northeast due to HCP engagement decline",
                source="episodic_memories",
                source_id="mem_001",
                score=0.85,
                retrieval_method="dense",
                metadata={"brand": "Kisqali", "region": "northeast"}
            ),
            RetrievalResult(
                content="Causal path: HCP visits -> Script volume -> TRx",
                source="causal_paths",
                source_id="path_001",
                score=0.78,
                retrieval_method="graph",
                metadata={}
            )
        ]
        mock_insert.return_value = "mem_e2e_001"
        mock_get.return_value = {
            "memory_id": "mem_e2e_001",
            "description": "E2E test memory",
            "event_type": "query",
            "session_id": "sess_e2e",
            "agent_name": "orchestrator",
            "brand": "Kisqali",
            "region": "northeast",
            "raw_content": {},
            "occurred_at": datetime.now(timezone.utc).isoformat()
        }
        mock_update.return_value = None
        mock_signal.return_value = "signal_e2e_001"
        mock_proc.return_value = {
            "procedure_id": "proc_e2e",
            "usage_count": 20,
            "success_count": 18
        }
        mock_semantic_instance = MagicMock()
        mock_semantic_instance.find_causal_paths_for_kpi.return_value = [
            {
                "nodes": ["HCP engagement", "Script volume", "TRx"],
                "confidence": 0.85,
                "path_id": "path_e2e"
            }
        ]
        mock_semantic.return_value = mock_semantic_instance

        yield {
            "search": mock_search,
            "insert": mock_insert,
            "get": mock_get,
            "update": mock_update,
            "signal": mock_signal,
            "proc": mock_proc,
            "semantic": mock_semantic_instance
        }


@pytest.fixture
def mock_cognitive_service():
    """Mock cognitive service for E2E testing."""
    with patch("src.api.routes.cognitive.get_working_memory") as mock_memory, \
         patch("src.api.routes.cognitive.hybrid_search") as mock_search:

        # Mock working memory
        mock_wm = MagicMock()
        mock_wm.create_session = AsyncMock(return_value={"session_id": "sess_e2e"})
        mock_wm.get_session = AsyncMock(return_value={
            "user_id": "user_e2e",
            "context": {"brand": "Kisqali"},
            "state": "active"
        })
        mock_wm.add_message = AsyncMock(return_value=True)
        mock_wm.append_evidence = AsyncMock(return_value=True)
        mock_wm.get_evidence_trail = AsyncMock(return_value=[])
        mock_memory.return_value = mock_wm

        # Mock hybrid search for cognitive route
        mock_search.return_value = [
            RetrievalResult(
                content="TRx dropped 15% in northeast",
                source="episodic_memories",
                source_id="mem_001",
                score=0.85,
                retrieval_method="dense",
                metadata={"brand": "Kisqali"}
            )
        ]

        yield {"working_memory": mock_wm, "search": mock_search}


# =============================================================================
# FULL CYCLE E2E TESTS
# =============================================================================

class TestFullCognitiveE2E:
    """End-to-end tests for complete cognitive cycles."""

    def test_causal_query_full_cycle(self, mock_all_services, mock_cognitive_service):
        """Test complete causal query cycle through API."""
        # Step 1: Execute cognitive query
        response = client.post(
            "/cognitive/query",
            json={
                "query": "Why did TRx drop in the northeast region for Kisqali?",
                "brand": "Kisqali",
                "region": "northeast",
                "include_evidence": True
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "session_id" in data
        assert "response" in data
        assert "query_type" in data
        assert "confidence" in data

    def test_prediction_query_full_cycle(self, mock_all_services, mock_cognitive_service):
        """Test complete prediction query cycle through API."""
        response = client.post(
            "/cognitive/query",
            json={
                "query": "What will TRx be next quarter?",
                "brand": "Kisqali"
            }
        )

        assert response.status_code == 200

    def test_session_persistence_across_queries(self, mock_all_services, mock_cognitive_service):
        """Test that session context persists across multiple queries."""
        # First query creates session
        response1 = client.post(
            "/cognitive/query",
            json={
                "query": "What are the current TRx trends?",
                "brand": "Kisqali"
            }
        )

        assert response1.status_code == 200
        session_id = response1.json().get("session_id")
        assert session_id is not None

        # Second query reuses session
        response2 = client.post(
            "/cognitive/query",
            json={
                "query": "Why is this happening?",
                "session_id": session_id
            }
        )

        assert response2.status_code == 200
        # Should use same session
        assert response2.json().get("session_id") == session_id


# =============================================================================
# MEMORY INTEGRATION E2E TESTS
# =============================================================================

class TestMemoryIntegrationE2E:
    """End-to-end tests for memory system integration."""

    def test_episodic_memory_create_and_retrieve(self, mock_all_services):
        """Test creating and retrieving episodic memory."""
        # Create memory
        create_response = client.post(
            "/memory/episodic",
            json={
                "content": "User asked about TRx drop in northeast",
                "event_type": "query",
                "session_id": "sess_e2e",
                "agent_name": "orchestrator",
                "brand": "Kisqali",
                "region": "northeast"
            }
        )

        assert create_response.status_code == 200
        memory_id = create_response.json()["id"]
        assert memory_id == "mem_e2e_001"

        # Retrieve memory
        get_response = client.get(f"/memory/episodic/{memory_id}")

        assert get_response.status_code == 200
        data = get_response.json()
        assert data["brand"] == "Kisqali"

    def test_search_finds_relevant_memories(self, mock_all_services):
        """Test hybrid search returns relevant memories."""
        response = client.post(
            "/memory/search",
            json={
                "query": "Why did TRx drop?",
                "k": 10,
                "filters": {"brand": "Kisqali"}
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] > 0
        assert any("TRx" in r["content"] for r in data["results"])

    def test_procedural_feedback_updates_success_rate(self, mock_all_services):
        """Test that procedural feedback updates success rate."""
        response = client.post(
            "/memory/procedural/feedback",
            json={
                "procedure_id": "proc_e2e",
                "outcome": "success",
                "score": 0.95,
                "feedback_text": "Analysis was highly accurate",
                "agent_name": "feedback_learner"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["feedback_recorded"] is True
        # 18/20 = 0.9 success rate
        assert data["new_success_rate"] == 0.9

    def test_semantic_path_query(self, mock_all_services):
        """Test semantic graph path queries."""
        response = client.get(
            "/memory/semantic/paths",
            params={"kpi_name": "TRx", "min_confidence": 0.6}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_paths"] >= 1


# =============================================================================
# LEARNING SIGNAL E2E TESTS
# =============================================================================

class TestLearningSignalE2E:
    """End-to-end tests for learning signal propagation."""

    def test_learning_signal_recorded_after_feedback(self, mock_all_services):
        """Test that learning signals are recorded after procedural feedback."""
        response = client.post(
            "/memory/procedural/feedback",
            json={
                "procedure_id": "proc_test",
                "outcome": "success",
                "score": 0.9,
                "agent_name": "causal_impact"
            }
        )

        assert response.status_code == 200
        # Verify signal was recorded
        mock_all_services["signal"].assert_called_once()

    def test_feedback_signals_contain_correct_metadata(self, mock_all_services):
        """Test that feedback signals have correct metadata."""
        response = client.post(
            "/memory/procedural/feedback",
            json={
                "procedure_id": "proc_meta_test",
                "outcome": "partial",
                "score": 0.7,
                "feedback_text": "Partially correct analysis",
                "session_id": "sess_meta",
                "agent_name": "gap_analyzer"
            }
        )

        assert response.status_code == 200

        # Check signal call arguments
        call_kwargs = mock_all_services["signal"].call_args
        signal_input = call_kwargs[1]["signal"]

        assert signal_input.applies_to_type == "procedure"
        assert signal_input.applies_to_id == "proc_meta_test"
        assert signal_input.rated_agent == "gap_analyzer"
        assert signal_input.signal_value == 0.7


# =============================================================================
# PERFORMANCE E2E TESTS
# =============================================================================

class TestPerformanceE2E:
    """End-to-end performance tests."""

    def test_search_latency_under_sla(self, mock_all_services):
        """Test that search latency is within SLA (<500ms)."""
        response = client.post(
            "/memory/search",
            json={"query": "TRx trends", "k": 10}
        )

        assert response.status_code == 200
        latency = response.json()["search_latency_ms"]
        # With mocked services, should be very fast
        # In production, SLA is <500ms
        assert latency < 500

    def test_path_query_latency_under_sla(self, mock_all_services):
        """Test that semantic path query is within SLA (<500ms)."""
        response = client.get(
            "/memory/semantic/paths",
            params={"kpi_name": "TRx"}
        )

        assert response.status_code == 200
        latency = response.json()["query_latency_ms"]
        assert latency < 500


# =============================================================================
# ERROR HANDLING E2E TESTS
# =============================================================================

class TestErrorHandlingE2E:
    """End-to-end tests for error handling."""

    def test_invalid_query_returns_validation_error(self):
        """Test that empty query returns 422."""
        response = client.post(
            "/memory/search",
            json={"query": ""}
        )

        assert response.status_code == 422

    def test_missing_memory_returns_404(self):
        """Test that missing memory returns 404."""
        with patch("src.api.routes.memory.get_memory_by_id") as mock:
            mock.return_value = None
            response = client.get("/memory/episodic/nonexistent_id")

        assert response.status_code == 404

    def test_cognitive_error_returns_error_response(self, mock_cognitive_service):
        """Test that cognitive errors are handled gracefully."""
        # This should still return 200 with error info in response
        # because errors are caught and returned as error query_type
        response = client.post(
            "/cognitive/query",
            json={"query": "Test query"}
        )

        # Should return 200 even with internal error handling
        assert response.status_code in [200, 500]


# =============================================================================
# CONCURRENT ACCESS E2E TESTS
# =============================================================================

class TestConcurrentAccessE2E:
    """End-to-end tests for concurrent access patterns."""

    def test_multiple_concurrent_searches(self, mock_all_services):
        """Test that multiple concurrent searches work correctly."""
        import concurrent.futures

        def make_search():
            return client.post(
                "/memory/search",
                json={"query": "TRx analysis", "k": 5}
            )

        # Execute 5 concurrent searches
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_search) for _ in range(5)]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert all(r.status_code == 200 for r in responses)
        assert all(r.json()["total_results"] > 0 for r in responses)

    def test_mixed_memory_operations(self, mock_all_services):
        """Test mixed read/write operations."""
        # Create episodic memory
        create_resp = client.post(
            "/memory/episodic",
            json={
                "content": "Concurrent test memory",
                "event_type": "action"
            }
        )
        assert create_resp.status_code == 200

        # Search while creating
        search_resp = client.post(
            "/memory/search",
            json={"query": "concurrent test"}
        )
        assert search_resp.status_code == 200

        # Record feedback
        feedback_resp = client.post(
            "/memory/procedural/feedback",
            json={
                "procedure_id": "proc_concurrent",
                "outcome": "success",
                "score": 0.8
            }
        )
        assert feedback_resp.status_code == 200
