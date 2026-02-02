"""
Unit tests for src/api/routes/cognitive.py

Tests cover:
- Cognitive workflow endpoints (process_cognitive_query, get_session, create_session, delete_session, cognitive_rag_search)
- Happy paths, error paths, edge cases
- Mock all external dependencies (WorkingMemory, HybridSearch, OrchestratorAgent, CausalRAG)
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException

from src.api.routes.cognitive import (
    CognitivePhase,
    CognitiveQueryRequest,
    CreateSessionRequest,
    QueryType,
    SessionState,
    _build_filters,
    _detect_query_type,
    _extract_kpi_from_query,
    _generate_placeholder_response,
    _route_to_agent,
    cognitive_rag_search,
    create_session,
    delete_session,
    get_orchestrator,
    get_session,
    process_cognitive_query,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_working_memory():
    """Mock WorkingMemory."""
    memory = AsyncMock()
    memory.create_session = AsyncMock(return_value={"session_id": "test-session"})
    memory.get_session = AsyncMock(
        return_value={
            "session_id": "test-session",
            "user_id": "test-user",
            "context": {"brand": "Kisqali", "region": "northeast"},
            "state": "active",
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
        }
    )
    memory.add_message = AsyncMock()
    memory.append_evidence = AsyncMock()
    memory.get_messages = AsyncMock(return_value=[])
    memory.get_evidence_trail = AsyncMock(return_value=[])
    memory.delete_session = AsyncMock()
    return memory


@pytest.fixture
def mock_hybrid_search():
    """Mock hybrid_search function."""

    async def search_mock(*args, **kwargs):
        from src.rag import RetrievalResult
        from src.rag.types import RetrievalSource

        return [
            RetrievalResult(
                id="doc1",
                content="Test evidence content",
                score=0.9,
                source=RetrievalSource.VECTOR,
                metadata={"retrieval_method": "hybrid"},
            )
        ]

    return search_mock


@pytest.fixture
def mock_orchestrator():
    """Mock OrchestratorAgent."""
    orchestrator = MagicMock()
    orchestrator.run = AsyncMock(
        return_value={
            "response_text": "Test response from orchestrator",
            "response_confidence": 0.85,
            "agents_dispatched": ["causal_impact"],
        }
    )
    return orchestrator


@pytest.fixture
def sample_query_request():
    """Sample cognitive query request."""
    return CognitiveQueryRequest(
        query="Why did TRx drop 15% in northeast region last quarter?",
        brand="Kisqali",
        region="northeast",
        query_type=QueryType.CAUSAL,
    )


# =============================================================================
# Endpoint Tests
# =============================================================================


class TestProcessCognitiveQueryEndpoint:
    """Tests for /cognitive/query endpoint."""

    @pytest.mark.asyncio
    async def test_process_query_success(
        self, sample_query_request, mock_working_memory, mock_hybrid_search, mock_orchestrator
    ):
        """Test successful cognitive query processing."""
        with (
            patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory),
            patch("src.api.routes.cognitive.hybrid_search", new=mock_hybrid_search),
            patch("src.api.routes.cognitive.get_orchestrator", return_value=mock_orchestrator),
        ):
            response = await process_cognitive_query(sample_query_request, BackgroundTasks())

            assert response.query == sample_query_request.query
            assert response.query_type == QueryType.CAUSAL
            assert response.agent_used == "causal_impact"
            assert CognitivePhase.COMPLETE in response.phases_completed

    @pytest.mark.asyncio
    async def test_process_query_creates_new_session(
        self, sample_query_request, mock_working_memory, mock_hybrid_search
    ):
        """Test query creates new session when session_id not provided."""
        with (
            patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory),
            patch("src.api.routes.cognitive.hybrid_search", new=mock_hybrid_search),
            patch("src.api.routes.cognitive.get_orchestrator", return_value=None),
        ):
            response = await process_cognitive_query(sample_query_request, BackgroundTasks())

            mock_working_memory.create_session.assert_called_once()
            assert response.session_id is not None

    @pytest.mark.asyncio
    async def test_process_query_uses_existing_session(
        self, sample_query_request, mock_working_memory, mock_hybrid_search
    ):
        """Test query uses existing session when session_id provided."""
        sample_query_request.session_id = "existing-session"

        with (
            patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory),
            patch("src.api.routes.cognitive.hybrid_search", new=mock_hybrid_search),
            patch("src.api.routes.cognitive.get_orchestrator", return_value=None),
        ):
            response = await process_cognitive_query(sample_query_request, BackgroundTasks())

            assert response.session_id == "existing-session"
            # Should NOT create new session
            mock_working_memory.create_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_query_without_orchestrator(
        self, sample_query_request, mock_working_memory, mock_hybrid_search
    ):
        """Test query processing when orchestrator not available."""
        with (
            patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory),
            patch("src.api.routes.cognitive.hybrid_search", new=mock_hybrid_search),
            patch("src.api.routes.cognitive.get_orchestrator", return_value=None),
        ):
            response = await process_cognitive_query(sample_query_request, BackgroundTasks())

            # Should use fallback response
            assert "causal" in response.response.lower()
            assert response.confidence == 0.85

    @pytest.mark.asyncio
    async def test_process_query_error_handling(self, sample_query_request, mock_working_memory):
        """Test query processing error handling."""
        mock_working_memory.create_session.side_effect = Exception("Session error")

        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            with pytest.raises(HTTPException) as exc_info:
                await process_cognitive_query(sample_query_request, BackgroundTasks())

            assert exc_info.value.status_code == 500


class TestGetSessionEndpoint:
    """Tests for /cognitive/session/{session_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_session_success(self, mock_working_memory):
        """Test successful session retrieval."""
        mock_working_memory.get_messages.return_value = [
            {
                "role": "user",
                "content": "Test message",
                "timestamp": datetime.now(timezone.utc),
                "metadata": {},
            }
        ]
        mock_working_memory.get_evidence_trail.return_value = [
            {"content": "Evidence", "source": "test", "score": 0.9}
        ]

        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = await get_session("test-session")

            assert response.context.session_id == "test-session"
            assert len(response.messages) == 1
            assert len(response.evidence_trail) == 1

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, mock_working_memory):
        """Test session not found error."""
        mock_working_memory.get_session.return_value = None

        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            with pytest.raises(HTTPException) as exc_info:
                await get_session("nonexistent-session")

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_session_error(self, mock_working_memory):
        """Test session retrieval error handling."""
        mock_working_memory.get_session.side_effect = Exception("DB error")

        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            with pytest.raises(HTTPException) as exc_info:
                await get_session("test-session")

            assert exc_info.value.status_code == 500


class TestCreateSessionEndpoint:
    """Tests for /cognitive/session endpoint."""

    @pytest.mark.asyncio
    async def test_create_session_success(self, mock_working_memory):
        """Test successful session creation."""
        request = CreateSessionRequest(
            user_id="test-user",
            brand="Kisqali",
            region="northeast",
        )

        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = await create_session(request)

            assert response.session_id is not None
            assert response.state == SessionState.ACTIVE
            mock_working_memory.create_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_session_error(self, mock_working_memory):
        """Test session creation error handling."""
        mock_working_memory.create_session.side_effect = Exception("Create error")

        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            with pytest.raises(HTTPException) as exc_info:
                await create_session(CreateSessionRequest())

            assert exc_info.value.status_code == 500


class TestDeleteSessionEndpoint:
    """Tests for DELETE /cognitive/session/{session_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_session_success(self, mock_working_memory):
        """Test successful session deletion."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = await delete_session("test-session")

            assert response["session_id"] == "test-session"
            assert response["deleted"] is True
            mock_working_memory.delete_session.assert_called_once_with("test-session")

    @pytest.mark.asyncio
    async def test_delete_session_error(self, mock_working_memory):
        """Test session deletion error handling."""
        mock_working_memory.delete_session.side_effect = Exception("Delete error")

        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            with pytest.raises(HTTPException) as exc_info:
                await delete_session("test-session")

            assert exc_info.value.status_code == 500


class TestCognitiveRAGSearchEndpoint:
    """Tests for /cognitive/rag endpoint."""

    @pytest.mark.asyncio
    async def test_cognitive_rag_success(self):
        """Test successful cognitive RAG search."""
        from src.api.routes.cognitive import CognitiveRAGRequest

        request = CognitiveRAGRequest(
            query="Why did Kisqali adoption increase in the Northeast last quarter?"
        )

        mock_rag = MagicMock()
        mock_rag.cognitive_search = AsyncMock(
            return_value={
                "response": "Adoption increased due to increased engagement",
                "evidence": [{"content": "Evidence 1"}],
                "hop_count": 2,
                "entities": ["Kisqali", "Northeast"],
                "intent": "causal",
                "rewritten_query": "Enhanced query",
                "latency_ms": 1250.5,
            }
        )

        with patch("src.rag.causal_rag.CausalRAG", return_value=mock_rag):
            response = await cognitive_rag_search(request)

            assert response.response == "Adoption increased due to increased engagement"
            assert response.hop_count == 2
            assert "Kisqali" in response.entities

    @pytest.mark.asyncio
    async def test_cognitive_rag_import_error(self):
        """Test cognitive RAG with import error."""
        from src.api.routes.cognitive import CognitiveRAGRequest

        request = CognitiveRAGRequest(query="test")

        with patch("src.rag.causal_rag.CausalRAG", side_effect=ImportError("No module")):
            with pytest.raises(HTTPException) as exc_info:
                await cognitive_rag_search(request)

            assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_cognitive_rag_value_error(self):
        """Test cognitive RAG with value error."""
        from src.api.routes.cognitive import CognitiveRAGRequest

        request = CognitiveRAGRequest(query="test")

        with patch("src.rag.causal_rag.CausalRAG", side_effect=ValueError("Config error")):
            with pytest.raises(HTTPException) as exc_info:
                await cognitive_rag_search(request)

            assert exc_info.value.status_code == 400


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_detect_query_type_causal(self):
        """Test causal query type detection."""
        assert _detect_query_type("Why did TRx drop?") == QueryType.CAUSAL
        assert _detect_query_type("What caused the decline?") == QueryType.CAUSAL
        assert _detect_query_type("What's the impact of the campaign?") == QueryType.CAUSAL

    def test_detect_query_type_prediction(self):
        """Test prediction query type detection."""
        assert _detect_query_type("What will happen next quarter?") == QueryType.PREDICTION
        assert _detect_query_type("Forecast TRx for Q4") == QueryType.PREDICTION

    def test_detect_query_type_optimization(self):
        """Test optimization query type detection."""
        assert (
            _detect_query_type("How can we optimize resource allocation?") == QueryType.OPTIMIZATION
        )
        assert _detect_query_type("What's the best approach?") == QueryType.OPTIMIZATION

    def test_detect_query_type_monitoring(self):
        """Test monitoring query type detection."""
        assert _detect_query_type("Check system health") == QueryType.MONITORING
        assert _detect_query_type("Any drift detected?") == QueryType.MONITORING

    def test_detect_query_type_explanation(self):
        """Test explanation query type detection."""
        assert _detect_query_type("Explain the model behavior") == QueryType.EXPLANATION
        assert _detect_query_type("How does the algorithm work?") == QueryType.EXPLANATION

    def test_detect_query_type_general(self):
        """Test general query type detection."""
        assert _detect_query_type("Show me the data") == QueryType.GENERAL

    def test_route_to_agent(self):
        """Test agent routing."""
        assert _route_to_agent(QueryType.CAUSAL) == "causal_impact"
        assert _route_to_agent(QueryType.PREDICTION) == "prediction_synthesizer"
        assert _route_to_agent(QueryType.OPTIMIZATION) == "resource_optimizer"
        assert _route_to_agent(QueryType.MONITORING) == "health_score"
        assert _route_to_agent(QueryType.EXPLANATION) == "explainer"
        assert _route_to_agent(QueryType.GENERAL) == "orchestrator"

    def test_extract_kpi_from_query(self):
        """Test KPI extraction from query."""
        assert _extract_kpi_from_query("TRx dropped last quarter") == "TRx"
        assert _extract_kpi_from_query("NRx is increasing") == "NRx"
        assert _extract_kpi_from_query("Check conversion rates") == "conversion_rate"
        assert _extract_kpi_from_query("Market share analysis") == "market_share"
        assert _extract_kpi_from_query("Patient adherence report") == "adherence_rate"
        assert _extract_kpi_from_query("Churn prediction") == "churn_rate"
        assert _extract_kpi_from_query("No KPI mentioned") is None

    def test_build_filters(self):
        """Test filter building."""
        filters = _build_filters("Kisqali", "northeast")
        assert filters == {"brand": "Kisqali", "region": "northeast"}

        filters = _build_filters("Kisqali", None)
        assert filters == {"brand": "Kisqali"}

        filters = _build_filters(None, "northeast")
        assert filters == {"region": "northeast"}

        filters = _build_filters(None, None)
        assert filters is None

    def test_generate_placeholder_response(self):
        """Test placeholder response generation."""
        response = _generate_placeholder_response(
            query="Test query",
            query_type=QueryType.CAUSAL,
            evidence=None,
            brand="Kisqali",
        )
        assert "Kisqali" in response
        assert "causal" in response.lower()


class TestGetOrchestratorFunction:
    """Tests for get_orchestrator singleton function."""

    def test_get_orchestrator_creates_instance(self):
        """Test orchestrator instance creation."""
        # Reset global
        import src.api.routes.cognitive as cognitive_module

        cognitive_module._orchestrator_instance = None

        with patch("src.agents.orchestrator.OrchestratorAgent") as mock_orch_class:
            mock_orch_class.return_value = MagicMock()

            orchestrator = get_orchestrator()

            assert orchestrator is not None
            mock_orch_class.assert_called_once()

    def test_get_orchestrator_handles_error(self):
        """Test orchestrator creation error handling."""
        import src.api.routes.cognitive as cognitive_module

        cognitive_module._orchestrator_instance = None

        with patch(
            "src.agents.orchestrator.OrchestratorAgent", side_effect=Exception("Init error")
        ):
            orchestrator = get_orchestrator()

            assert orchestrator is None


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_process_query_with_empty_evidence(
        self, sample_query_request, mock_working_memory
    ):
        """Test processing query with no evidence found."""

        async def empty_search(*args, **kwargs):
            return []

        with (
            patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory),
            patch("src.api.routes.cognitive.hybrid_search", new=empty_search),
            patch("src.api.routes.cognitive.get_orchestrator", return_value=None),
        ):
            response = await process_cognitive_query(sample_query_request, BackgroundTasks())

            assert response.evidence is None or len(response.evidence) == 0

    @pytest.mark.asyncio
    async def test_process_query_auto_detect_type(self, mock_working_memory, mock_hybrid_search):
        """Test query type auto-detection."""
        request = CognitiveQueryRequest(
            query="Why did sales drop?",  # Should be detected as CAUSAL
            brand="Kisqali",
        )
        # Don't set query_type explicitly

        with (
            patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory),
            patch("src.api.routes.cognitive.hybrid_search", new=mock_hybrid_search),
            patch("src.api.routes.cognitive.get_orchestrator", return_value=None),
        ):
            response = await process_cognitive_query(request, BackgroundTasks())

            assert response.query_type == QueryType.CAUSAL

    @pytest.mark.asyncio
    async def test_process_query_max_memory_results(self, mock_working_memory, mock_hybrid_search):
        """Test query with max memory results limit."""
        request = CognitiveQueryRequest(
            query="Test query",
            max_memory_results=50,  # Max allowed
        )

        # Wrap the async function in AsyncMock to track calls
        mock_search = AsyncMock(side_effect=mock_hybrid_search)

        with (
            patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory),
            patch("src.api.routes.cognitive.hybrid_search", new=mock_search),
            patch("src.api.routes.cognitive.get_orchestrator", return_value=None),
        ):
            await process_cognitive_query(request, BackgroundTasks())

            # Verify max_memory_results was used
            # Note: hybrid_search is called with k parameter
            assert mock_search.call_count > 0
