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
    list_sessions,
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


@pytest.fixture
def admin_user():
    """Authenticated principal with admin role (bypasses ownership checks)."""
    return {
        "id": "admin-001",
        "email": "admin@e2i-analytics.com",
        "app_metadata": {"role": "admin"},
    }


@pytest.fixture
def owner_user():
    """Authenticated principal who OWNS the mock session (user_id='test-user')."""
    return {
        "id": "test-user",
        "email": "owner@e2i-analytics.com",
        "app_metadata": {"role": "viewer"},
    }


@pytest.fixture
def other_user():
    """Authenticated principal who does NOT own the mock session."""
    return {
        "id": "attacker-999",
        "email": "attacker@e2i-analytics.com",
        "app_metadata": {"role": "viewer"},
    }


# =============================================================================
# Endpoint Tests
# =============================================================================


class TestProcessCognitiveQueryEndpoint:
    """Tests for /cognitive/query endpoint."""

    @pytest.mark.asyncio
    async def test_process_query_success(
        self,
        sample_query_request,
        mock_working_memory,
        mock_hybrid_search,
        mock_orchestrator,
        admin_user,
    ):
        """Test successful cognitive query processing."""
        with (
            patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory),
            patch("src.api.routes.cognitive.hybrid_search", new=mock_hybrid_search),
            patch("src.api.routes.cognitive.get_orchestrator", return_value=mock_orchestrator),
        ):
            response = await process_cognitive_query(
                sample_query_request, BackgroundTasks(), user=admin_user
            )

            assert response.query == sample_query_request.query
            assert response.query_type == QueryType.CAUSAL
            assert response.agent_used == "causal_impact"
            assert CognitivePhase.COMPLETE in response.phases_completed

    @pytest.mark.asyncio
    async def test_process_query_creates_new_session(
        self, sample_query_request, mock_working_memory, mock_hybrid_search, admin_user
    ):
        """Test query creates new session when session_id not provided."""
        with (
            patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory),
            patch("src.api.routes.cognitive.hybrid_search", new=mock_hybrid_search),
            patch("src.api.routes.cognitive.get_orchestrator", return_value=None),
        ):
            response = await process_cognitive_query(
                sample_query_request, BackgroundTasks(), user=admin_user
            )

            mock_working_memory.create_session.assert_called_once()
            assert response.session_id is not None

    @pytest.mark.asyncio
    async def test_process_query_uses_existing_session(
        self, sample_query_request, mock_working_memory, mock_hybrid_search, admin_user
    ):
        """Test query uses existing session when session_id provided."""
        sample_query_request.session_id = "existing-session"

        with (
            patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory),
            patch("src.api.routes.cognitive.hybrid_search", new=mock_hybrid_search),
            patch("src.api.routes.cognitive.get_orchestrator", return_value=None),
        ):
            response = await process_cognitive_query(
                sample_query_request, BackgroundTasks(), user=admin_user
            )

            assert response.session_id == "existing-session"
            # Should NOT create new session
            mock_working_memory.create_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_query_without_orchestrator(
        self, sample_query_request, mock_working_memory, mock_hybrid_search, admin_user
    ):
        """Test query processing when orchestrator not available."""
        with (
            patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory),
            patch("src.api.routes.cognitive.hybrid_search", new=mock_hybrid_search),
            patch("src.api.routes.cognitive.get_orchestrator", return_value=None),
        ):
            response = await process_cognitive_query(
                sample_query_request, BackgroundTasks(), user=admin_user
            )

            # Should use fallback response
            assert "causal" in response.response.lower()
            # FINDING #2: the degraded/placeholder path must NOT emit a
            # fabricated 0.85 confidence. No real orchestrator ran here.
            assert response.confidence is None

    @pytest.mark.asyncio
    async def test_process_query_error_handling(
        self, sample_query_request, mock_working_memory, admin_user
    ):
        """Test query processing error handling."""
        mock_working_memory.create_session.side_effect = Exception("Session error")

        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            with pytest.raises(HTTPException) as exc_info:
                await process_cognitive_query(
                    sample_query_request, BackgroundTasks(), user=admin_user
                )

            assert exc_info.value.status_code == 500
            # FINDING #3: 500 detail must be generic, not leak str(e).
            assert "Session error" not in str(exc_info.value.detail)


class TestGetSessionEndpoint:
    """Tests for /cognitive/session/{session_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_session_success(self, mock_working_memory, admin_user):
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
            response = await get_session("test-session", user=admin_user)

            assert response.context.session_id == "test-session"
            assert len(response.messages) == 1
            assert len(response.evidence_trail) == 1

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, mock_working_memory, admin_user):
        """Test session not found error."""
        mock_working_memory.get_session.return_value = None

        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            with pytest.raises(HTTPException) as exc_info:
                await get_session("nonexistent-session", user=admin_user)

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_session_error(self, mock_working_memory, admin_user):
        """Test session retrieval error handling."""
        mock_working_memory.get_session.side_effect = Exception("DB error")

        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            with pytest.raises(HTTPException) as exc_info:
                await get_session("test-session", user=admin_user)

            assert exc_info.value.status_code == 500
            # FINDING #3: 500 detail must be generic, not leak str(e).
            assert "DB error" not in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_session_owner_allowed(self, mock_working_memory, owner_user):
        """FINDING #1: the session owner can read their own session."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = await get_session("test-session", user=owner_user)
            assert response.context.session_id == "test-session"

    @pytest.mark.asyncio
    async def test_get_session_idor_blocked(self, mock_working_memory, other_user):
        """FINDING #1 [CRITICAL IDOR]: a non-owner non-admin cannot read
        another user's session. Must return 404 (no existence leak)."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            with pytest.raises(HTTPException) as exc_info:
                await get_session("test-session", user=other_user)

            assert exc_info.value.status_code == 404


class TestCreateSessionEndpoint:
    """Tests for /cognitive/session endpoint."""

    @pytest.mark.asyncio
    async def test_create_session_success(self, mock_working_memory, owner_user):
        """Test successful session creation."""
        request = CreateSessionRequest(
            user_id="ignored-client-value",
            brand="Kisqali",
            region="northeast",
        )

        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = await create_session(request, user=owner_user)

            assert response.session_id is not None
            assert response.state == SessionState.ACTIVE
            mock_working_memory.create_session.assert_called_once()
            # FINDING #1: session must be owned by the AUTHENTICATED caller,
            # not the client-supplied user_id in the body.
            _, call_kwargs = mock_working_memory.create_session.call_args
            assert call_kwargs["user_id"] == "test-user"

    @pytest.mark.asyncio
    async def test_create_session_error(self, mock_working_memory, owner_user):
        """Test session creation error handling."""
        mock_working_memory.create_session.side_effect = Exception("Create error")

        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            with pytest.raises(HTTPException) as exc_info:
                await create_session(CreateSessionRequest(), user=owner_user)

            assert exc_info.value.status_code == 500
            # FINDING #3: 500 detail must be generic, not leak str(e).
            assert "Create error" not in str(exc_info.value.detail)


class TestDeleteSessionEndpoint:
    """Tests for DELETE /cognitive/session/{session_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_session_success(self, mock_working_memory, admin_user):
        """Test successful session deletion."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = await delete_session("test-session", user=admin_user)

            assert response["session_id"] == "test-session"
            assert response["deleted"] is True
            mock_working_memory.delete_session.assert_called_once_with("test-session")

    @pytest.mark.asyncio
    async def test_delete_session_error(self, mock_working_memory, admin_user):
        """Test session deletion error handling."""
        mock_working_memory.delete_session.side_effect = Exception("Delete error")

        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            with pytest.raises(HTTPException) as exc_info:
                await delete_session("test-session", user=admin_user)

            assert exc_info.value.status_code == 500
            # FINDING #3: 500 detail must be generic, not leak str(e).
            assert "Delete error" not in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_delete_session_owner_allowed(self, mock_working_memory, owner_user):
        """FINDING #1: the session owner can delete their own session."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            response = await delete_session("test-session", user=owner_user)
            assert response["deleted"] is True
            mock_working_memory.delete_session.assert_called_once_with("test-session")

    @pytest.mark.asyncio
    async def test_delete_session_idor_blocked(self, mock_working_memory, other_user):
        """FINDING #1 [CRITICAL IDOR]: a non-owner non-admin cannot delete
        another user's session. Must 404 and NEVER call delete_session."""
        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            with pytest.raises(HTTPException) as exc_info:
                await delete_session("test-session", user=other_user)

            assert exc_info.value.status_code == 404
            mock_working_memory.delete_session.assert_not_called()


class TestCognitiveRAGSearchEndpoint:
    """Tests for /cognitive/rag endpoint."""

    @pytest.mark.asyncio
    async def test_cognitive_rag_success(self, admin_user):
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
            response = await cognitive_rag_search(request, user=admin_user)

            assert response.response == "Adoption increased due to increased engagement"
            assert response.hop_count == 2
            assert "Kisqali" in response.entities

    @pytest.mark.asyncio
    async def test_cognitive_rag_import_error(self, admin_user):
        """Test cognitive RAG with import error."""
        from src.api.routes.cognitive import CognitiveRAGRequest

        request = CognitiveRAGRequest(query="test")

        with patch("src.rag.causal_rag.CausalRAG", side_effect=ImportError("No module")):
            with pytest.raises(HTTPException) as exc_info:
                await cognitive_rag_search(request, user=admin_user)

            assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_cognitive_rag_value_error(self, admin_user):
        """Test cognitive RAG with value error."""
        from src.api.routes.cognitive import CognitiveRAGRequest

        request = CognitiveRAGRequest(query="test")

        with patch("src.rag.causal_rag.CausalRAG", side_effect=ValueError("Config error")):
            with pytest.raises(HTTPException) as exc_info:
                await cognitive_rag_search(request, user=admin_user)

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
        self, sample_query_request, mock_working_memory, admin_user
    ):
        """Test processing query with no evidence found."""

        async def empty_search(*args, **kwargs):
            return []

        with (
            patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory),
            patch("src.api.routes.cognitive.hybrid_search", new=empty_search),
            patch("src.api.routes.cognitive.get_orchestrator", return_value=None),
        ):
            response = await process_cognitive_query(
                sample_query_request, BackgroundTasks(), user=admin_user
            )

            assert response.evidence is None or len(response.evidence) == 0

    @pytest.mark.asyncio
    async def test_process_query_auto_detect_type(
        self, mock_working_memory, mock_hybrid_search, admin_user
    ):
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
            response = await process_cognitive_query(request, BackgroundTasks(), user=admin_user)

            assert response.query_type == QueryType.CAUSAL

    @pytest.mark.asyncio
    async def test_process_query_max_memory_results(
        self, mock_working_memory, mock_hybrid_search, admin_user
    ):
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
            await process_cognitive_query(request, BackgroundTasks(), user=admin_user)

            # Verify max_memory_results was used
            # Note: hybrid_search is called with k parameter
            assert mock_search.call_count > 0


# =============================================================================
# Authorization / IDOR Tests (Security Review Findings #1, #2)
# =============================================================================


class TestProcessQueryAuthorization:
    """FINDING #1 [CRITICAL IDOR] + #2 for POST /cognitive/query."""

    @pytest.mark.asyncio
    async def test_new_session_owned_by_token_not_body(
        self, mock_working_memory, mock_hybrid_search, owner_user
    ):
        """FINDING #1: user_id is derived from the authenticated token, NOT
        from the request body (which is ignored). A new session is created
        under the caller's id even if the body claims another user."""
        request = CognitiveQueryRequest(
            query="Why did TRx drop?",
            user_id="victim-spoofed-id",  # attacker-supplied; must be ignored
            query_type=QueryType.CAUSAL,
        )

        with (
            patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory),
            patch("src.api.routes.cognitive.hybrid_search", new=mock_hybrid_search),
            patch("src.api.routes.cognitive.get_orchestrator", return_value=None),
        ):
            await process_cognitive_query(request, BackgroundTasks(), user=owner_user)

            _, call_kwargs = mock_working_memory.create_session.call_args
            assert call_kwargs["user_id"] == "test-user"

    @pytest.mark.asyncio
    async def test_continue_others_session_blocked(
        self, mock_working_memory, mock_hybrid_search, other_user
    ):
        """FINDING #1 [CRITICAL IDOR]: continuing an existing session you do
        not own must be rejected with 404, and must NOT add a message."""
        request = CognitiveQueryRequest(
            query="leak this user's data",
            session_id="test-session",  # owned by user_id='test-user'
            query_type=QueryType.CAUSAL,
        )

        with (
            patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory),
            patch("src.api.routes.cognitive.hybrid_search", new=mock_hybrid_search),
            patch("src.api.routes.cognitive.get_orchestrator", return_value=None),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await process_cognitive_query(request, BackgroundTasks(), user=other_user)

            assert exc_info.value.status_code == 404
            mock_working_memory.add_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_continue_own_session_allowed(
        self, mock_working_memory, mock_hybrid_search, owner_user
    ):
        """FINDING #1: the owner may continue their own existing session."""
        request = CognitiveQueryRequest(
            query="follow-up question",
            session_id="test-session",
            query_type=QueryType.CAUSAL,
        )

        with (
            patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory),
            patch("src.api.routes.cognitive.hybrid_search", new=mock_hybrid_search),
            patch("src.api.routes.cognitive.get_orchestrator", return_value=None),
        ):
            response = await process_cognitive_query(request, BackgroundTasks(), user=owner_user)
            assert response.session_id == "test-session"

    @pytest.mark.asyncio
    async def test_degraded_path_emits_null_confidence(
        self, sample_query_request, mock_working_memory, mock_hybrid_search, admin_user
    ):
        """FINDING #2: when the orchestrator runs but produces no dispatch
        (degraded), confidence is the orchestrator's real value, not a
        fabricated default. When NO orchestrator runs, confidence is None."""
        # Orchestrator present but returns no dispatch and no confidence.
        degraded_orch = MagicMock()
        degraded_orch.run = AsyncMock(
            return_value={"response_text": "partial", "agents_dispatched": []}
        )

        with (
            patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory),
            patch("src.api.routes.cognitive.hybrid_search", new=mock_hybrid_search),
            patch("src.api.routes.cognitive.get_orchestrator", return_value=degraded_orch),
        ):
            response = await process_cognitive_query(
                sample_query_request, BackgroundTasks(), user=admin_user
            )
            assert response.agent_used == "orchestrator_degraded"
            # No real confidence was produced -> must not fabricate 0.85.
            assert response.confidence is None


class TestListSessionsAuthorization:
    """FINDING #1 [CRITICAL IDOR] for GET /cognitive/sessions."""

    @pytest.mark.asyncio
    async def test_list_sessions_scoped_to_caller(self, mock_working_memory, other_user):
        """A non-admin caller's list is ALWAYS scoped to their own id; the
        client-supplied ?user_id is ignored (no cross-user enumeration)."""
        mock_working_memory.list_sessions = AsyncMock(return_value=[])

        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            await list_sessions(user_id="someone-else", limit=50, user=other_user)

            _, call_kwargs = mock_working_memory.list_sessions.call_args
            assert call_kwargs["user_id"] == "attacker-999"

    @pytest.mark.asyncio
    async def test_list_sessions_admin_may_filter_any_user(self, mock_working_memory, admin_user):
        """An admin may pass ?user_id to inspect any user's sessions."""
        mock_working_memory.list_sessions = AsyncMock(return_value=[])

        with patch("src.api.routes.cognitive.get_working_memory", return_value=mock_working_memory):
            await list_sessions(user_id="some-user", limit=50, user=admin_user)

            _, call_kwargs = mock_working_memory.list_sessions.call_args
            assert call_kwargs["user_id"] == "some-user"
