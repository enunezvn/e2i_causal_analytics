"""
Tests for src/api/routes/copilotkit.py

Covers:
- GET /copilotkit/status endpoint
- POST /copilotkit/chat endpoint
- POST /copilotkit/chat/stream endpoint
- POST /copilotkit/feedback endpoint
- GET /copilotkit/feedback/stats endpoint
- GET /copilotkit/analytics/usage endpoint
- GET /copilotkit/analytics/agents endpoint
- GET /copilotkit/analytics/errors endpoint
- GET /copilotkit/analytics/hourly endpoint
- ChatRequest model validation
- ChatResponse model validation
- FeedbackRequest model validation
- FeedbackResponse model validation
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.copilotkit import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    router,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_client():
    """Create a FastAPI test client with the copilotkit router."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def mock_llm_provider():
    """Mock the LLM provider lookup."""
    with (
        patch("src.api.routes.copilotkit.get_llm_provider") as mock_provider,
        patch(
            "src.api.routes.copilotkit.MODEL_MAPPINGS",
            {"anthropic": {"standard": "claude-3-sonnet"}},
        ),
    ):
        mock_provider.return_value = "anthropic"
        yield mock_provider


@pytest.fixture
def mock_copilot_actions():
    """Mock the COPILOT_ACTIONS list."""
    mock_action = MagicMock()
    mock_action.name = "test_action"
    with patch("src.api.routes.copilotkit.COPILOT_ACTIONS", [mock_action]):
        yield [mock_action]


# =============================================================================
# ChatRequest Model Tests
# =============================================================================


class TestChatRequest:
    """Tests for ChatRequest model."""

    def test_create_valid_request(self):
        """Test creating a valid ChatRequest."""
        request = ChatRequest(
            query="What is the TRx for Kisqali?",
            user_id="user-123",
            request_id="req-456",
        )
        assert request.query == "What is the TRx for Kisqali?"
        assert request.user_id == "user-123"
        assert request.request_id == "req-456"

    def test_request_with_optional_fields(self):
        """Test ChatRequest with all optional fields."""
        request = ChatRequest(
            query="Show metrics",
            user_id="user-123",
            request_id="req-456",
            session_id="session-789",
            brand_context="Kisqali",
            region_context="US",
        )
        assert request.session_id == "session-789"
        assert request.brand_context == "Kisqali"
        assert request.region_context == "US"

    def test_request_defaults_optional_to_none(self):
        """Test that optional fields default to None."""
        request = ChatRequest(
            query="Test query",
            user_id="user-1",
            request_id="req-1",
        )
        assert request.session_id is None
        assert request.brand_context is None
        assert request.region_context is None

    def test_request_requires_query(self):
        """Test that query is required."""
        with pytest.raises(ValueError):
            ChatRequest(
                user_id="user-123",
                request_id="req-456",
            )

    def test_request_requires_user_id(self):
        """Test that user_id is required.

        IDOR fix: user_id remains a required body field for backward
        compatibility, but it is NON-AUTHORITATIVE — the server derives the
        caller's identity from the authenticated token and 403s on mismatch.
        See TestChatIdentityFromToken.
        """
        with pytest.raises(ValueError):
            ChatRequest(
                query="Test query",
                request_id="req-456",
            )

    def test_request_request_id_is_optional(self):
        """Test that request_id is optional (auto-extracted from header if not provided)."""
        request = ChatRequest(
            query="Test query",
            user_id="user-123",
        )
        assert request.request_id is None


# =============================================================================
# ChatResponse Model Tests
# =============================================================================


class TestChatResponse:
    """Tests for ChatResponse model."""

    def test_create_success_response(self):
        """Test creating a successful ChatResponse."""
        response = ChatResponse(
            success=True,
            session_id="session-123",
            response="The TRx for Kisqali is 1,234 units.",
            conversation_title="TRx Query",
            agent_name="tool_composer",
        )
        assert response.success is True
        assert response.session_id == "session-123"
        assert response.response == "The TRx for Kisqali is 1,234 units."
        assert response.conversation_title == "TRx Query"
        assert response.agent_name == "tool_composer"
        assert response.error is None

    def test_create_error_response(self):
        """Test creating an error ChatResponse."""
        response = ChatResponse(
            success=False,
            session_id="",
            response="",
            error="Internal server error",
        )
        assert response.success is False
        assert response.error == "Internal server error"

    def test_response_optional_fields(self):
        """Test that optional fields default correctly."""
        response = ChatResponse(
            success=True,
            session_id="session-1",
            response="Test response",
        )
        assert response.conversation_title is None
        assert response.agent_name is None
        assert response.error is None

    def test_dispatch_observability_fields_default(self):
        """Test that dispatch observability fields have correct defaults."""
        response = ChatResponse(
            success=True,
            session_id="session-1",
            response="Test response",
        )
        # Phase 1 System Evaluation fields
        assert response.orchestrator_used is False
        assert response.agents_dispatched == []
        assert response.routed_agent is None
        assert response.response_confidence is None
        assert response.execution_time_ms is None
        assert response.intent is None
        assert response.intent_confidence is None

    def test_dispatch_observability_fields_populated(self):
        """Test ChatResponse with all dispatch observability fields."""
        response = ChatResponse(
            success=True,
            session_id="session-123",
            response="Causal analysis shows...",
            conversation_title="Causal Query",
            agent_name="orchestrator",
            # Phase 1 System Evaluation fields
            orchestrator_used=True,
            agents_dispatched=["causal_impact", "gap_analyzer"],
            routed_agent="causal_impact",
            response_confidence=0.87,
            execution_time_ms=1523.45,
            intent="causal_analysis",
            intent_confidence=0.92,
        )
        assert response.orchestrator_used is True
        assert response.agents_dispatched == ["causal_impact", "gap_analyzer"]
        assert response.routed_agent == "causal_impact"
        assert response.response_confidence == 0.87
        assert response.execution_time_ms == 1523.45
        assert response.intent == "causal_analysis"
        assert response.intent_confidence == 0.92

    def test_error_response_includes_execution_time(self):
        """Test that error responses include execution_time_ms."""
        response = ChatResponse(
            success=False,
            session_id="",
            response="",
            error="Agent timeout",
            execution_time_ms=30045.12,
        )
        assert response.success is False
        assert response.error == "Agent timeout"
        assert response.execution_time_ms == 30045.12


# =============================================================================
# FeedbackRequest Model Tests
# =============================================================================


class TestFeedbackRequest:
    """Tests for FeedbackRequest model."""

    def test_create_valid_feedback(self):
        """Test creating a valid FeedbackRequest."""
        request = FeedbackRequest(
            message_id=123,
            rating="thumbs_up",
        )
        assert request.message_id == 123
        assert request.rating == "thumbs_up"

    def test_feedback_with_all_fields(self):
        """Test FeedbackRequest with all optional fields."""
        request = FeedbackRequest(
            message_id=456,
            session_id="session-789",
            rating="thumbs_down",
            comment="Response was not helpful",
            query_text="What is the TRx?",
            response_preview="The TRx for...",
            agent_name="causal_impact",
            tools_used=["query_kpi", "analyze_trend"],
        )
        assert request.message_id == 456
        assert request.session_id == "session-789"
        assert request.rating == "thumbs_down"
        assert request.comment == "Response was not helpful"
        assert request.query_text == "What is the TRx?"
        assert request.response_preview == "The TRx for..."
        assert request.agent_name == "causal_impact"
        assert request.tools_used == ["query_kpi", "analyze_trend"]

    def test_feedback_defaults_optional_to_none(self):
        """Test that optional fields default to None."""
        request = FeedbackRequest(
            message_id=1,
            rating="thumbs_up",
        )
        assert request.session_id is None
        assert request.comment is None
        assert request.query_text is None
        assert request.response_preview is None
        assert request.agent_name is None
        assert request.tools_used is None

    def test_feedback_message_id_now_optional(self):
        """message_id is optional: the live CopilotKit stream only knows its
        AG-UI uuid, so the server resolves the DB row from session_id +
        response_preview instead (the old client fabricated ids via
        parseInt(uuid)||Date.now(), which could attach feedback to a row from a
        DIFFERENT session)."""
        request = FeedbackRequest(
            rating="thumbs_up",
            session_id="thread-uuid",
            response_preview="The TRx for...",
            message_uuid="ag-ui-uuid",
        )
        assert request.message_id is None
        assert request.message_uuid == "ag-ui-uuid"

    def test_feedback_requires_rating(self):
        """Test that rating is required."""
        with pytest.raises(ValueError):
            FeedbackRequest(message_id=123)


# =============================================================================
# FeedbackResponse Model Tests
# =============================================================================


class TestFeedbackResponse:
    """Tests for FeedbackResponse model."""

    def test_create_success_response(self):
        """Test creating a successful FeedbackResponse."""
        response = FeedbackResponse(
            success=True,
            feedback_id=789,
            message="Feedback submitted successfully",
        )
        assert response.success is True
        assert response.feedback_id == 789
        assert response.message == "Feedback submitted successfully"
        assert response.error is None

    def test_create_error_response(self):
        """Test creating an error FeedbackResponse."""
        response = FeedbackResponse(
            success=False,
            error="Invalid rating value",
        )
        assert response.success is False
        assert response.error == "Invalid rating value"
        assert response.feedback_id is None
        assert response.message is None

    def test_response_optional_fields(self):
        """Test that optional fields default correctly."""
        response = FeedbackResponse(success=True)
        assert response.feedback_id is None
        assert response.message is None
        assert response.error is None


# =============================================================================
# GET /copilotkit/status Endpoint Tests
# =============================================================================


class TestGetStatusEndpoint:
    """Tests for GET /copilotkit/status endpoint."""

    def test_get_status_success(self, test_client, mock_llm_provider, mock_copilot_actions):
        """Test successful status retrieval."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            response = test_client.get("/copilotkit/status")

        assert response.status_code == 200

    def test_status_has_required_fields(self, test_client, mock_llm_provider, mock_copilot_actions):
        """Test that status response has all required fields."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            response = test_client.get("/copilotkit/status")
            data = response.json()

        assert "status" in data
        assert "version" in data
        assert "agents_available" in data
        assert "agent_names" in data
        assert "actions_available" in data
        assert "action_names" in data
        assert "llm_provider" in data
        assert "llm_model" in data
        assert "llm_configured" in data
        assert "timestamp" in data

    def test_status_values(self, test_client, mock_llm_provider, mock_copilot_actions):
        """Test status response values."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            response = test_client.get("/copilotkit/status")
            data = response.json()

        assert data["status"] == "active"
        assert data["llm_provider"] == "anthropic"
        assert data["llm_model"] == "claude-3-sonnet"
        assert data["llm_configured"] is True

    def test_status_without_api_key(self, test_client, mock_llm_provider, mock_copilot_actions):
        """Test status when no API key is configured."""
        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""},
            clear=True,
        ):
            response = test_client.get("/copilotkit/status")
            data = response.json()

        assert data["llm_configured"] is False

    def test_status_timestamp_format(self, test_client, mock_llm_provider, mock_copilot_actions):
        """Test that timestamp is valid ISO format."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            response = test_client.get("/copilotkit/status")
            data = response.json()

        timestamp = data["timestamp"]
        # Should be parseable as datetime
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert parsed is not None


# =============================================================================
# POST /copilotkit/chat Endpoint Tests
# =============================================================================


class TestChatEndpoint:
    """Tests for POST /copilotkit/chat endpoint."""

    def test_chat_success(self, test_client):
        """Test successful chat request."""
        mock_result = {
            "response_text": "The TRx for Kisqali is 1,234 units.",
            "session_id": "session-123",
            "agent_name": "tool_composer",
        }

        with patch(
            "src.api.routes.chatbot_graph.run_chatbot",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = test_client.post(
                "/copilotkit/chat",
                json={
                    "query": "What is the TRx for Kisqali?",
                    "user_id": "user-123",
                    "request_id": "req-456",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["response"] == "The TRx for Kisqali is 1,234 units."
        assert data["session_id"] == "session-123"
        assert data["agent_name"] == "tool_composer"

    def test_chat_returns_dispatch_observability(self, test_client):
        """Test that chat returns dispatch observability fields from orchestrator."""
        mock_result = {
            "response_text": "Based on causal analysis...",
            "session_id": "session-456",
            "agent_name": "orchestrator",
            # Phase 1 System Evaluation fields
            "orchestrator_used": True,
            "agents_dispatched": ["causal_impact", "gap_analyzer"],
            "routed_agent": "causal_impact",
            "response_confidence": 0.87,
            "intent": "causal_analysis",
            "intent_confidence": 0.92,
        }

        with patch(
            "src.api.routes.chatbot_graph.run_chatbot",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = test_client.post(
                "/copilotkit/chat",
                json={
                    "query": "What caused the drop in TRx?",
                    "user_id": "user-123",
                    "request_id": "req-456",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        # Verify dispatch observability fields
        assert data["orchestrator_used"] is True
        assert data["agents_dispatched"] == ["causal_impact", "gap_analyzer"]
        assert data["routed_agent"] == "causal_impact"
        assert data["response_confidence"] == 0.87
        assert data["intent"] == "causal_analysis"
        assert data["intent_confidence"] == 0.92
        # execution_time_ms is always returned
        assert "execution_time_ms" in data
        assert data["execution_time_ms"] > 0

    def test_chat_execution_time_always_present(self, test_client):
        """Test that execution_time_ms is always returned, even without orchestrator."""
        mock_result = {
            "response_text": "Hello! How can I help?",
            "session_id": "session-789",
            "agent_name": "chatbot",
            "intent": "greeting",
        }

        with patch(
            "src.api.routes.chatbot_graph.run_chatbot",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = test_client.post(
                "/copilotkit/chat",
                json={
                    "query": "Hello",
                    "user_id": "user-123",
                    "request_id": "req-456",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        # execution_time_ms is always calculated
        assert "execution_time_ms" in data
        assert isinstance(data["execution_time_ms"], (int, float))
        assert data["execution_time_ms"] > 0
        # When orchestrator not used, these should be defaults
        assert data["orchestrator_used"] is False
        assert data["agents_dispatched"] == []

    def test_chat_generates_title(self, test_client):
        """Test that chat generates a conversation title."""
        mock_result = {
            "response_text": "Test response",
            "session_id": "session-123",
        }

        with patch(
            "src.api.routes.chatbot_graph.run_chatbot",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = test_client.post(
                "/copilotkit/chat",
                json={
                    "query": "What is the TRx for Kisqali?",
                    "user_id": "user-123",
                    "request_id": "req-456",
                },
            )

        data = response.json()
        assert data["conversation_title"] is not None

    def test_chat_truncates_long_title(self, test_client):
        """Test that long queries get truncated for title."""
        long_query = "A" * 100  # 100 character query
        mock_result = {
            "response_text": "Response",
            "session_id": "session-123",
        }

        with patch(
            "src.api.routes.chatbot_graph.run_chatbot",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = test_client.post(
                "/copilotkit/chat",
                json={
                    "query": long_query,
                    "user_id": "user-123",
                    "request_id": "req-456",
                },
            )

        data = response.json()
        # Title should be 50 chars + "..."
        assert len(data["conversation_title"]) == 53
        assert data["conversation_title"].endswith("...")

    def test_chat_error_handling(self, test_client):
        """Test chat error handling.

        The internal exception text must NOT be leaked to the client
        (Finding 3). Only a generic error message is returned.
        """
        with patch(
            "src.api.routes.chatbot_graph.run_chatbot",
            new_callable=AsyncMock,
            side_effect=Exception("Database connection error"),
        ):
            response = test_client.post(
                "/copilotkit/chat",
                json={
                    "query": "Test query",
                    "user_id": "test-user-id",
                    "request_id": "req-456",
                },
            )

        assert response.status_code == 200  # Endpoint returns 200 with error in body
        data = response.json()
        assert data["success"] is False
        # Internal detail must not leak; a generic message is returned instead.
        assert "Database connection error" not in data["error"]
        assert data["error"]

    def test_chat_with_brand_context(self, test_client):
        """Test chat with brand context."""
        mock_result = {
            "response_text": "Kisqali metrics...",
            "session_id": "session-123",
        }

        with patch(
            "src.api.routes.chatbot_graph.run_chatbot",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_run:
            test_client.post(
                "/copilotkit/chat",
                json={
                    "query": "Show metrics",
                    "user_id": "user-123",
                    "request_id": "req-456",
                    "brand_context": "Kisqali",
                    "region_context": "US",
                },
            )

        # Verify context was passed to run_chatbot
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["brand_context"] == "Kisqali"
        assert call_kwargs["region_context"] == "US"

    def test_chat_missing_required_fields(self, test_client):
        """Test chat with missing required fields."""
        response = test_client.post(
            "/copilotkit/chat",
            json={
                "query": "Test query",
                # Missing user_id and request_id
            },
        )

        assert response.status_code == 422  # Validation error


# =============================================================================
# POST /copilotkit/feedback Endpoint Tests
# =============================================================================


class TestFeedbackEndpoint:
    """Tests for POST /copilotkit/feedback endpoint."""

    def test_feedback_invalid_rating(self, test_client):
        """Test feedback with invalid rating value."""
        response = test_client.post(
            "/copilotkit/feedback",
            json={
                "message_id": 123,
                "rating": "invalid_rating",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "Invalid rating" in data["error"]

    def test_feedback_requires_thumbs_up_or_down(self, test_client):
        """Test that only thumbs_up or thumbs_down are valid."""
        # Valid thumbs_up
        for rating in ["thumbs_up", "thumbs_down"]:
            with (
                patch.dict(
                    "os.environ",
                    {
                        "SUPABASE_URL": "https://test.supabase.co",
                        "SUPABASE_SERVICE_KEY": "test-key",
                    },
                ),
                patch("supabase.create_client") as mock_create,
            ):
                # Mock message lookup
                mock_client = MagicMock()
                mock_table = MagicMock()
                mock_select = MagicMock()
                mock_eq = MagicMock()
                mock_limit = MagicMock()
                MagicMock()

                mock_create.return_value = mock_client
                mock_client.table.return_value = mock_table
                mock_table.select.return_value = mock_select
                mock_select.eq.return_value = mock_eq
                mock_eq.limit.return_value = mock_limit
                mock_limit.execute.return_value = MagicMock(
                    data=[{"id": 123, "session_id": "session-123"}]
                )

                with (
                    patch(
                        "src.memory.services.factories.get_async_supabase_client",
                        new_callable=AsyncMock,
                    ),
                    patch("src.repositories.get_chatbot_feedback_repository") as mock_repo,
                ):
                    mock_repo_instance = MagicMock()
                    mock_repo_instance.add_feedback = AsyncMock(return_value={"id": 1})
                    mock_repo.return_value = mock_repo_instance

                    response = test_client.post(
                        "/copilotkit/feedback",
                        json={
                            "message_id": 123,
                            "rating": rating,
                        },
                    )

                    # Should not fail validation
                    assert response.status_code == 200

    def test_feedback_missing_supabase_config(self, test_client):
        """Test feedback when Supabase config is missing."""
        with patch.dict(
            "os.environ",
            {"SUPABASE_URL": "", "SUPABASE_SERVICE_KEY": ""},
            clear=True,
        ):
            response = test_client.post(
                "/copilotkit/feedback",
                json={
                    "message_id": 123,
                    "rating": "thumbs_up",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "configuration error" in data["error"]

    @staticmethod
    def _chain_client(rows, uuid_raises=False):
        """Supabase client fake that honors the pass-0 resolution filter.

        The route now runs a metadata->>frontend_message_id eq-query BEFORE
        content matching; a fake that returned the same rows for every query
        would make pass 0 spuriously match the newest row and hide content-
        resolution behavior from the tests. Other filters (session_id, role)
        are ignored — rows are already scoped per test. uuid_raises simulates
        a pass-0 query failure (the guard must fall back to content passes).
        """

        class _Query:
            def __init__(self, all_rows):
                self._rows = all_rows
                self._uuid_filter = None
                self._limit = None

            def select(self, *_args, **_kwargs):
                return self

            def order(self, *_args, **_kwargs):
                return self

            def limit(self, n):
                self._limit = n
                return self

            def eq(self, key, value):
                if key == "metadata->>frontend_message_id":
                    self._uuid_filter = value
                return self

            def execute(self):
                result = self._rows
                if self._uuid_filter is not None:
                    if uuid_raises:
                        raise RuntimeError("jsonb filter rejected")
                    result = [
                        r
                        for r in result
                        if (r.get("metadata") or {}).get("frontend_message_id") == self._uuid_filter
                    ]
                if self._limit is not None:
                    result = result[: self._limit]
                return MagicMock(data=result)

        client = MagicMock()
        client.table.side_effect = lambda _name: _Query(rows)
        return client

    def _post_resolution(self, test_client, rows, payload, uuid_raises=False):
        with (
            patch.dict(
                "os.environ",
                {
                    "SUPABASE_URL": "https://test.supabase.co",
                    "SUPABASE_SERVICE_KEY": "test-key",
                },
            ),
            patch(
                "supabase.create_client",
                return_value=self._chain_client(rows, uuid_raises=uuid_raises),
            ),
            patch(
                "src.memory.services.factories.get_async_supabase_client",
                new_callable=AsyncMock,
            ),
            patch("src.repositories.get_chatbot_feedback_repository") as mock_repo,
        ):
            repo = MagicMock()
            repo.add_feedback = AsyncMock(return_value={"id": 42})
            mock_repo.return_value = repo
            response = test_client.post("/copilotkit/feedback", json=payload)
        return response, repo

    def test_feedback_resolves_by_session_and_preview(self, test_client):
        """Without a DB message_id, the server must resolve the rated message by
        exact response-prefix match within the session — the live CopilotKit
        stream only knows its AG-UI uuid, never the DB id."""
        preview = "The TRx performance for Remibrutinib is strong"
        rows = [
            {"id": 9, "session_id": "thread-1", "content": "A different, newer response"},
            {"id": 7, "session_id": "thread-1", "content": preview + " across all regions."},
        ]
        response, repo = self._post_resolution(
            test_client,
            rows,
            {
                "session_id": "thread-1",
                "response_preview": preview,
                "message_uuid": "ag-ui-uuid-1",
                "rating": "thumbs_up",
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        kwargs = repo.add_feedback.call_args.kwargs
        assert kwargs["message_id"] == 7
        assert kwargs["session_id"] == "thread-1"
        assert kwargs["metadata"] == {"message_uuid": "ag-ui-uuid-1"}

    def test_feedback_resolution_no_match_fails_closed(self, test_client):
        """A preview matching NO persisted assistant message must fail honestly —
        never fall back to 'newest row' (that would attach the rating to the
        wrong message)."""
        rows = [{"id": 9, "session_id": "thread-1", "content": "Entirely different content"}]
        response, repo = self._post_resolution(
            test_client,
            rows,
            {
                "session_id": "thread-1",
                "response_preview": "The TRx performance...",
                "rating": "thumbs_down",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "No persisted assistant message" in data["error"]
        repo.add_feedback.assert_not_called()

    def test_feedback_exact_content_match_beats_shared_prefix(self, test_client):
        """Two responses sharing the same 500-char prefix are indistinguishable
        to prefix matching — response_text (full content) must resolve the
        exact row, not the newest prefix match."""
        shared = "x" * 500
        rows = [
            {"id": 9, "session_id": "thread-1", "content": shared + " newer tail"},
            {"id": 7, "session_id": "thread-1", "content": shared + " the rated tail"},
        ]
        response, repo = self._post_resolution(
            test_client,
            rows,
            {
                "session_id": "thread-1",
                "response_preview": shared,
                "response_text": shared + " the rated tail",
                "rating": "thumbs_up",
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert repo.add_feedback.call_args.kwargs["message_id"] == 7

    def test_feedback_attribution_derived_from_matched_row(self, test_client):
        """The persisted row is the authority on who responded: a client-sent
        agent_name (the old sidebar hardcoded 'copilotkit') must NOT override
        the row's agent_name, and tools come from the row's tool_results."""
        content = "The gap analysis shows Kisqali underperforming in the West."
        rows = [
            {
                "id": 5,
                "session_id": "thread-1",
                "content": content,
                "agent_name": "gap_analyzer",
                "metadata": {"tool_results": [{"tool": "run_gap_analysis", "result": "..."}]},
            }
        ]
        response, repo = self._post_resolution(
            test_client,
            rows,
            {
                "session_id": "thread-1",
                "response_preview": content[:500],
                "response_text": content,
                "agent_name": "copilotkit",
                "rating": "thumbs_down",
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        kwargs = repo.add_feedback.call_args.kwargs
        assert kwargs["agent_name"] == "gap_analyzer"
        assert kwargs["tools_used"] == ["run_gap_analysis"]

    def test_feedback_client_agent_hint_is_fallback_only(self, test_client):
        """When the matched row has no agent_name, the client hint survives."""
        content = "A response persisted without attribution."
        rows = [{"id": 6, "session_id": "thread-1", "content": content}]
        response, repo = self._post_resolution(
            test_client,
            rows,
            {
                "session_id": "thread-1",
                "response_text": content,
                "agent_name": "orchestrator",
                "rating": "thumbs_up",
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert repo.add_feedback.call_args.kwargs["agent_name"] == "orchestrator"

    def test_feedback_resolves_by_stamped_frontend_message_id(self, test_client):
        """Pass 0: the SSE layer stamps metadata.frontend_message_id on the
        persisted row, so the client's message_uuid resolves the exact row
        even when content matching would pick a different (newer) one."""
        preview = "Identical preview text for both rows"
        rows = [
            {"id": 9, "session_id": "thread-1", "content": preview + " newer"},
            {
                "id": 7,
                "session_id": "thread-1",
                "content": preview + " rated",
                "agent_name": "copilotkit",
                "metadata": {"frontend_message_id": "ag-ui-uuid-7"},
            },
        ]
        response, repo = self._post_resolution(
            test_client,
            rows,
            {
                "session_id": "thread-1",
                "message_uuid": "ag-ui-uuid-7",
                "response_preview": preview,
                "rating": "thumbs_up",
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert repo.add_feedback.call_args.kwargs["message_id"] == 7

    def test_feedback_unstamped_uuid_falls_back_to_content(self, test_client):
        """A message_uuid with no stamped row (stamping is best-effort) must
        fall through to content matching, not fail."""
        content = "Fallback-resolved response."
        rows = [{"id": 4, "session_id": "thread-1", "content": content}]
        response, repo = self._post_resolution(
            test_client,
            rows,
            {
                "session_id": "thread-1",
                "message_uuid": "never-stamped",
                "response_text": content,
                "rating": "thumbs_down",
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert repo.add_feedback.call_args.kwargs["message_id"] == 4

    def test_feedback_row_tool_columns_beat_client_hint(self, test_client):
        """Orchestrator-flow rows store tools in top-level tool_results/
        tool_calls columns (not metadata); those must be selected and must
        win over a client-supplied tools_used hint."""
        content = "Causal analysis complete."
        rows = [
            {
                "id": 3,
                "session_id": "thread-1",
                "content": content,
                "agent_name": "causal_impact",
                "tool_results": [{"tool": "run_causal_analysis", "result": "..."}],
            }
        ]
        response, repo = self._post_resolution(
            test_client,
            rows,
            {
                "session_id": "thread-1",
                "response_text": content,
                "tools_used": ["client-invented-tool"],
                "rating": "thumbs_up",
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert repo.add_feedback.call_args.kwargs["tools_used"] == ["run_causal_analysis"]

    def test_feedback_tool_name_shape_resolved_from_row_tool_calls(self, test_client):
        """chatbot_graph.finalize_node persists top-level tool_calls entries
        keyed "tool_name" — when tool_results is empty, that shape is the only
        authoritative source and must be extracted."""
        content = "Query answered via data tool."
        rows = [
            {
                "id": 5,
                "session_id": "thread-1",
                "content": content,
                "agent_name": "orchestrator",
                "tool_results": [],
                "tool_calls": [{"tool_name": "e2i_data_query_tool", "args": {}}],
            }
        ]
        response, repo = self._post_resolution(
            test_client,
            rows,
            {
                "session_id": "thread-1",
                "response_text": content,
                "rating": "thumbs_up",
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert repo.add_feedback.call_args.kwargs["tools_used"] == ["e2i_data_query_tool"]

    def test_feedback_pass0_error_falls_back_to_content(self, test_client):
        """A pass-0 (stamped-id) query failure must degrade to content
        matching, not abort resolution."""
        content = "Resolved despite pass-0 failure."
        rows = [{"id": 6, "session_id": "thread-1", "content": content}]
        response, repo = self._post_resolution(
            test_client,
            rows,
            {
                "session_id": "thread-1",
                "message_uuid": "uuid-whose-query-breaks",
                "response_text": content,
                "rating": "thumbs_up",
            },
            uuid_raises=True,
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert repo.add_feedback.call_args.kwargs["message_id"] == 6

    def test_feedback_requires_id_or_session_preview(self, test_client):
        """Neither message_id nor (session_id + response_preview) → honest error."""
        response, repo = self._post_resolution(
            test_client,
            [],
            {"rating": "thumbs_up"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "message_id or (session_id + response_preview)" in data["error"]
        repo.add_feedback.assert_not_called()


# =============================================================================
# Frontend messageId tracking + stamping (feedback resolution pass-0 key)
# =============================================================================


class TestFrontendMessageIdStamping:
    """The SSE layer rebuilds (messageId -> text) from TEXT_MESSAGE_* events
    and stamps metadata.frontend_message_id onto the persisted rows — the
    stable key /copilotkit/feedback resolves by before content matching."""

    def test_tracker_accumulates_lifecycle(self):
        from src.api.routes.copilotkit import _track_text_message_event

        deltas: dict = {}
        completed: dict = {}
        events = [
            {"type": "TEXT_MESSAGE_START", "messageId": "m1"},
            {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m1", "delta": "Hello "},
            # snake_case key — native ag_ui events serialize by alias, but
            # defensive handling covers both spellings
            {"type": "TEXT_MESSAGE_CONTENT", "message_id": "m1", "delta": "world"},
            {"type": "TEXT_MESSAGE_END", "messageId": "m1"},
        ]
        for event in events:
            _track_text_message_event(event, deltas, completed)
        assert completed == {"m1": "Hello world"}
        assert deltas == {}

    def test_tracker_ignores_malformed_events(self):
        from src.api.routes.copilotkit import _track_text_message_event

        deltas: dict = {}
        completed: dict = {}
        for event in [
            {"type": "TEXT_MESSAGE_CONTENT", "delta": "no message id"},
            {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m2", "delta": 42},
            {"type": "TEXT_MESSAGE_END", "messageId": "m2"},  # no content accumulated
            {"type": "RUN_FINISHED"},
        ]:
            _track_text_message_event(event, deltas, completed)
        assert completed == {}

    def _stamp_client(self, rows):
        """Sync-client fake recording metadata updates and honoring the
        metadata->>run_id filter (run-scoped stamping)."""
        updates = []

        class _Query:
            def __init__(self):
                self._update_payload = None
                self._update_id = None
                self._run_id_filter = None

            def select(self, *_a, **_k):
                return self

            def eq(self, key, value):
                if self._update_payload is not None and key == "id":
                    self._update_id = value
                if key == "metadata->>run_id":
                    self._run_id_filter = value
                return self

            def order(self, *_a, **_k):
                return self

            def limit(self, *_a):
                return self

            def update(self, payload):
                self._update_payload = payload
                return self

            def execute(self):
                if self._update_payload is not None:
                    updates.append((self._update_id, self._update_payload))
                    return MagicMock(data=[])
                data = rows
                if self._run_id_filter is not None:
                    data = [
                        r
                        for r in rows
                        if (r.get("metadata") or {}).get("run_id") == self._run_id_filter
                    ]
                return MagicMock(data=data)

        client = MagicMock()
        client.table.side_effect = lambda _name: _Query()
        return client, updates

    def test_stamps_newest_matching_row(self):
        from src.api.routes.copilotkit import _stamp_frontend_message_ids

        rows = [
            {"id": 9, "content": "The answer.", "metadata": {"source": "copilotkit"}},
            {"id": 7, "content": "The answer.", "metadata": {}},  # older duplicate
        ]
        client, updates = self._stamp_client(rows)
        with patch("src.api.dependencies.supabase_client.get_supabase", return_value=client):
            _stamp_frontend_message_ids("thread-1", {"m1": "The answer."})
        assert updates == [(9, {"metadata": {"source": "copilotkit", "frontend_message_id": "m1"}})]

    def test_already_stamped_row_skipped_for_next_match(self):
        """An identical-content row already mapped to another messageId must
        not be re-stamped — the next-newest matching row takes the id."""
        from src.api.routes.copilotkit import _stamp_frontend_message_ids

        rows = [
            {"id": 9, "content": "Same.", "metadata": {"frontend_message_id": "m-old"}},
            {"id": 7, "content": "Same.", "metadata": {}},
        ]
        client, updates = self._stamp_client(rows)
        with patch("src.api.dependencies.supabase_client.get_supabase", return_value=client):
            _stamp_frontend_message_ids("thread-1", {"m-new": "Same."})
        assert updates == [(7, {"metadata": {"frontend_message_id": "m-new"}})]

    def test_stamping_is_run_scoped(self):
        """Two overlapping same-session runs with identical final content
        must not cross-stamp: the run_id filter scopes the content match to
        rows this run persisted, so run A cannot take run B's newer row."""
        from src.api.routes.copilotkit import _stamp_frontend_message_ids

        rows = [
            {"id": 9, "content": "Same.", "metadata": {"run_id": "run-B"}},
            {"id": 7, "content": "Same.", "metadata": {"run_id": "run-A"}},
        ]
        client, updates = self._stamp_client(rows)
        with patch("src.api.dependencies.supabase_client.get_supabase", return_value=client):
            _stamp_frontend_message_ids("thread-1", {"m-a": "Same."}, run_id="run-A")
        assert updates == [(7, {"metadata": {"run_id": "run-A", "frontend_message_id": "m-a"}})]

    def test_persist_message_sync_attaches_run_id(self):
        """Assistant rows persisted during a run must carry metadata.run_id
        (from the run contextvar) so stamping can scope to them; the caller's
        metadata dict must not be mutated."""
        from src.api.routes.copilotkit import _persist_message_sync, _run_id_context

        inserted = []

        class _Insert:
            def insert(self, payload):
                inserted.append(payload)
                return self

            def execute(self):
                return MagicMock(data=[{"id": 1}])

        client = MagicMock()
        client.table.return_value = _Insert()
        caller_metadata = {"source": "copilotkit"}
        token = _run_id_context.set("run-X")
        try:
            with patch("src.api.dependencies.supabase_client.get_supabase", return_value=client):
                _persist_message_sync(
                    "thread-1",
                    "assistant",
                    "hi",
                    agent_name="copilotkit",
                    metadata=caller_metadata,
                )
        finally:
            _run_id_context.reset(token)
        assert inserted[0]["metadata"] == {"source": "copilotkit", "run_id": "run-X"}
        assert caller_metadata == {"source": "copilotkit"}

    def test_persist_message_sync_no_run_id_outside_run(self):
        from src.api.routes.copilotkit import _persist_message_sync, _run_id_context

        inserted = []

        class _Insert:
            def insert(self, payload):
                inserted.append(payload)
                return self

            def execute(self):
                return MagicMock(data=[{"id": 1}])

        client = MagicMock()
        client.table.return_value = _Insert()
        token = _run_id_context.set(None)
        try:
            with patch("src.api.dependencies.supabase_client.get_supabase", return_value=client):
                _persist_message_sync("thread-1", "assistant", "hi", metadata={})
        finally:
            _run_id_context.reset(token)
        assert "run_id" not in inserted[0]["metadata"]

    def test_persist_message_sync_state_run_id_when_context_lost(self):
        """run_id must ride the same two-channel ladder as session_id: when
        the contextvar is lost but the node still has state's run_id, the row
        must carry it — otherwise the stamp filter hides the row forever and
        feedback degrades to content heuristics (codex round-4 MED)."""
        from src.api.routes.copilotkit import _persist_message_sync, _run_id_context

        inserted = []

        class _Insert:
            def insert(self, payload):
                inserted.append(payload)
                return self

            def execute(self):
                return MagicMock(data=[{"id": 1}])

        client = MagicMock()
        client.table.return_value = _Insert()
        token = _run_id_context.set(None)
        try:
            with patch("src.api.dependencies.supabase_client.get_supabase", return_value=client):
                _persist_message_sync(
                    "thread-1", "assistant", "hi", metadata={}, run_id="run-from-state"
                )
        finally:
            _run_id_context.reset(token)
        assert inserted[0]["metadata"]["run_id"] == "run-from-state"

    def test_persist_message_sync_context_var_wins_over_state_param(self):
        """Ladder order mirrors session_id: context var preferred, state
        param is the fallback."""
        from src.api.routes.copilotkit import _persist_message_sync, _run_id_context

        inserted = []

        class _Insert:
            def insert(self, payload):
                inserted.append(payload)
                return self

            def execute(self):
                return MagicMock(data=[{"id": 1}])

        client = MagicMock()
        client.table.return_value = _Insert()
        token = _run_id_context.set("run-ctx")
        try:
            with patch("src.api.dependencies.supabase_client.get_supabase", return_value=client):
                _persist_message_sync(
                    "thread-1", "assistant", "hi", metadata={}, run_id="run-stale"
                )
        finally:
            _run_id_context.reset(token)
        assert inserted[0]["metadata"]["run_id"] == "run-ctx"

    def test_stamping_failure_is_swallowed(self):
        """Stamping is best-effort: a DB failure must never propagate into the
        SSE generator (feedback falls back to content matching)."""
        from src.api.routes.copilotkit import _stamp_frontend_message_ids

        with patch(
            "src.api.dependencies.supabase_client.get_supabase",
            side_effect=RuntimeError("db down"),
        ):
            _stamp_frontend_message_ids("thread-1", {"m1": "text"})  # no raise


# =============================================================================
# GET /copilotkit/feedback/stats Endpoint Tests
# =============================================================================


class TestFeedbackStatsEndpoint:
    """Tests for GET /copilotkit/feedback/stats endpoint."""

    def test_feedback_stats_success(self, test_client):
        """Test successful feedback stats retrieval."""
        mock_agent_stats = [{"agent_name": "tool_composer", "thumbs_up": 10, "thumbs_down": 2}]
        mock_summary = {"total": 12, "positive_rate": 0.83}

        with (
            patch(
                "src.memory.services.factories.get_async_supabase_client",
                new_callable=AsyncMock,
            ),
            patch("src.repositories.get_chatbot_feedback_repository") as mock_repo,
        ):
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_agent_stats = AsyncMock(return_value=mock_agent_stats)
            mock_repo_instance.get_feedback_summary = AsyncMock(return_value=mock_summary)
            mock_repo.return_value = mock_repo_instance

            response = test_client.get("/copilotkit/feedback/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "summary" in data
        assert "agent_stats" in data

    def test_feedback_stats_with_filter(self, test_client):
        """Test feedback stats with agent filter."""
        with (
            patch(
                "src.memory.services.factories.get_async_supabase_client",
                new_callable=AsyncMock,
            ),
            patch("src.repositories.get_chatbot_feedback_repository") as mock_repo,
        ):
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_agent_stats = AsyncMock(return_value=[])
            mock_repo_instance.get_feedback_summary = AsyncMock(return_value={})
            mock_repo.return_value = mock_repo_instance

            response = test_client.get("/copilotkit/feedback/stats?agent_name=causal_impact&days=7")

        assert response.status_code == 200

    def test_feedback_stats_error_handling(self, test_client):
        """Test feedback stats error handling."""
        with patch(
            "src.memory.services.factories.get_async_supabase_client",
            new_callable=AsyncMock,
            side_effect=Exception("Database error"),
        ):
            response = test_client.get("/copilotkit/feedback/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data


# =============================================================================
# GET /copilotkit/analytics/usage Endpoint Tests
# =============================================================================


class TestUsageAnalyticsEndpoint:
    """Tests for GET /copilotkit/analytics/usage endpoint."""

    def test_usage_analytics_success(self, test_client):
        """Test successful usage analytics retrieval."""
        mock_summary = {"total_queries": 100, "avg_response_time": 1.5}
        mock_query_types = [{"type": "kpi_query", "count": 50}]
        mock_tool_usage = [{"tool": "query_kpi", "count": 75}]

        with patch("src.repositories.get_chatbot_analytics_repository") as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_usage_summary = AsyncMock(return_value=mock_summary)
            mock_repo_instance.get_query_type_distribution = AsyncMock(
                return_value=mock_query_types
            )
            mock_repo_instance.get_tool_usage_stats = AsyncMock(return_value=mock_tool_usage)
            mock_repo.return_value = mock_repo_instance

            response = test_client.get("/copilotkit/analytics/usage?days=7")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["period_days"] == 7
        assert "summary" in data
        assert "query_types" in data
        assert "tool_usage" in data

    def test_usage_analytics_default_days(self, test_client):
        """Test usage analytics with default days parameter."""
        with patch("src.repositories.get_chatbot_analytics_repository") as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_usage_summary = AsyncMock(return_value={})
            mock_repo_instance.get_query_type_distribution = AsyncMock(return_value=[])
            mock_repo_instance.get_tool_usage_stats = AsyncMock(return_value=[])
            mock_repo.return_value = mock_repo_instance

            response = test_client.get("/copilotkit/analytics/usage")

        data = response.json()
        assert data["period_days"] == 7  # Default value

    def test_usage_analytics_error(self, test_client):
        """Test usage analytics error handling."""
        with patch(
            "src.repositories.get_chatbot_analytics_repository",
            side_effect=Exception("Repository error"),
        ):
            response = test_client.get("/copilotkit/analytics/usage")

        data = response.json()
        assert data["success"] is False
        assert "error" in data


# =============================================================================
# GET /copilotkit/analytics/agents Endpoint Tests
# =============================================================================


class TestAgentAnalyticsEndpoint:
    """Tests for GET /copilotkit/analytics/agents endpoint."""

    def test_agent_analytics_success(self, test_client):
        """Test successful agent analytics retrieval."""
        mock_stats = [
            {
                "agent_name": "tool_composer",
                "query_count": 100,
                "avg_response_time": 1.2,
                "error_rate": 0.02,
            }
        ]

        with patch("src.repositories.get_chatbot_analytics_repository") as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_agent_performance = AsyncMock(return_value=mock_stats)
            mock_repo.return_value = mock_repo_instance

            response = test_client.get("/copilotkit/analytics/agents")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "agent_stats" in data

    def test_agent_analytics_with_filter(self, test_client):
        """Test agent analytics with agent name filter."""
        with patch("src.repositories.get_chatbot_analytics_repository") as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_agent_performance = AsyncMock(return_value=[])
            mock_repo.return_value = mock_repo_instance

            response = test_client.get(
                "/copilotkit/analytics/agents?agent_name=causal_impact&days=14"
            )

        data = response.json()
        assert data["agent_name"] == "causal_impact"
        assert data["period_days"] == 14


# =============================================================================
# GET /copilotkit/analytics/errors Endpoint Tests
# =============================================================================


class TestErrorAnalyticsEndpoint:
    """Tests for GET /copilotkit/analytics/errors endpoint."""

    def test_error_analytics_success(self, test_client):
        """Test successful error analytics retrieval."""
        mock_errors = [
            {
                "error_type": "LLMError",
                "message": "Rate limit exceeded",
                "session_id": "session-123",
                "timestamp": "2024-01-15T10:00:00Z",
            }
        ]

        with patch("src.repositories.get_chatbot_analytics_repository") as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_recent_errors = AsyncMock(return_value=mock_errors)
            mock_repo.return_value = mock_repo_instance

            response = test_client.get("/copilotkit/analytics/errors?limit=10")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 1
        assert len(data["errors"]) == 1

    def test_error_analytics_default_limit(self, test_client):
        """Test error analytics with default limit."""
        with patch("src.repositories.get_chatbot_analytics_repository") as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_recent_errors = AsyncMock(return_value=[])
            mock_repo.return_value = mock_repo_instance

            response = test_client.get("/copilotkit/analytics/errors")

        # Should use default limit of 20
        assert response.status_code == 200


# =============================================================================
# GET /copilotkit/analytics/hourly Endpoint Tests
# =============================================================================


class TestHourlyAnalyticsEndpoint:
    """Tests for GET /copilotkit/analytics/hourly endpoint."""

    def test_hourly_analytics_success(self, test_client):
        """Test successful hourly analytics retrieval."""
        mock_pattern = [
            {"hour": 9, "count": 50},
            {"hour": 10, "count": 75},
            {"hour": 14, "count": 100},
        ]

        with patch("src.repositories.get_chatbot_analytics_repository") as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_hourly_pattern = AsyncMock(return_value=mock_pattern)
            mock_repo.return_value = mock_repo_instance

            response = test_client.get("/copilotkit/analytics/hourly?days=7")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["period_days"] == 7
        assert "hourly_pattern" in data
        assert len(data["hourly_pattern"]) == 3

    def test_hourly_analytics_error(self, test_client):
        """Test hourly analytics error handling."""
        with patch(
            "src.repositories.get_chatbot_analytics_repository",
            side_effect=Exception("Database error"),
        ):
            response = test_client.get("/copilotkit/analytics/hourly")

        data = response.json()
        assert data["success"] is False
        assert "error" in data


# =============================================================================
# Integration Tests
# =============================================================================


class TestCopilotKitIntegration:
    """Integration tests for CopilotKit endpoints."""

    def test_multiple_endpoints_available(self, test_client):
        """Test that all endpoints are accessible."""
        # Status endpoint - should work without mocks
        with (
            patch("src.api.routes.copilotkit.get_llm_provider", return_value="anthropic"),
            patch(
                "src.api.routes.copilotkit.MODEL_MAPPINGS",
                {"anthropic": {"standard": "claude-3-sonnet"}},
            ),
            patch("src.api.routes.copilotkit.COPILOT_ACTIONS", []),
        ):
            status_response = test_client.get("/copilotkit/status")
            assert status_response.status_code == 200

    def test_chat_request_validation(self, test_client):
        """Test that chat request validates input properly."""
        # Valid request
        with patch(
            "src.api.routes.chatbot_graph.run_chatbot",
            new_callable=AsyncMock,
            return_value={
                "response_text": "OK",
                "session_id": "s1",
            },
        ):
            response = test_client.post(
                "/copilotkit/chat",
                json={
                    "query": "Test",
                    "user_id": "u1",
                    "request_id": "r1",
                },
            )
            assert response.status_code == 200

        # Invalid request - missing fields
        response = test_client.post(
            "/copilotkit/chat",
            json={"query": "Test"},
        )
        assert response.status_code == 422

    def test_feedback_rating_validation(self, test_client):
        """Test that feedback validates rating values."""
        # Invalid rating
        response = test_client.post(
            "/copilotkit/feedback",
            json={
                "message_id": 1,
                "rating": "5_stars",  # Invalid
            },
        )
        data = response.json()
        assert data["success"] is False
        assert "Invalid rating" in data["error"]


# =============================================================================
# Security fixes (API code review)
# =============================================================================


class TestChatIdentityFromToken:
    """Finding 1 [HIGH IDOR]: chat identity must come from the authenticated
    token, NOT the request body. A caller must not be able to pass an
    arbitrary ``user_id`` to impersonate another user / read their
    sessions and memory.
    """

    def _client_with_user(self, user):
        """Build a test client whose ``require_viewer`` returns ``user``."""
        from src.api.dependencies.auth import require_viewer
        from src.api.routes.copilotkit import router

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[require_viewer] = lambda: user
        return TestClient(app)

    def test_chat_uses_token_identity_not_body_user_id(self):
        """run_chatbot must receive the authenticated user id, never the body value."""
        token_user = {"id": "real-token-user", "app_metadata": {"role": "admin"}}
        client = self._client_with_user(token_user)

        with patch(
            "src.api.routes.chatbot_graph.run_chatbot",
            new_callable=AsyncMock,
            return_value={"response_text": "ok", "session_id": "s1"},
        ) as mock_run:
            response = client.post(
                "/copilotkit/chat",
                json={
                    "query": "Show my sessions",
                    # Attacker tries to impersonate another user via the body.
                    "user_id": "victim-user",
                    "request_id": "req-1",
                },
            )

        assert response.status_code == 200
        call_kwargs = mock_run.call_args.kwargs
        # The downstream call must use the TOKEN identity, not the body value.
        assert call_kwargs["user_id"] == "real-token-user"
        assert call_kwargs["user_id"] != "victim-user"

    def test_chat_mismatched_body_user_id_rejected_in_production(self):
        """In production (TESTING_MODE off), a body user_id that disagrees with
        the token identity is an impersonation attempt and must be rejected 403.
        """
        token_user = {"id": "real-token-user", "app_metadata": {"role": "admin"}}
        client = self._client_with_user(token_user)

        with (
            patch("src.api.routes.copilotkit.TESTING_MODE", False),
            patch(
                "src.api.routes.chatbot_graph.run_chatbot",
                new_callable=AsyncMock,
                return_value={"response_text": "ok", "session_id": "s1"},
            ) as mock_run,
        ):
            response = client.post(
                "/copilotkit/chat",
                json={
                    "query": "Show victim sessions",
                    "user_id": "victim-user",
                    "request_id": "req-1",
                },
            )

        assert response.status_code == 403
        mock_run.assert_not_called()

    def test_chat_matching_body_user_id_allowed(self):
        """A body user_id that matches the token identity is allowed (compat)."""
        token_user = {"id": "real-token-user", "app_metadata": {"role": "admin"}}
        client = self._client_with_user(token_user)

        with (
            patch("src.api.routes.copilotkit.TESTING_MODE", False),
            patch(
                "src.api.routes.chatbot_graph.run_chatbot",
                new_callable=AsyncMock,
                return_value={"response_text": "ok", "session_id": "s1"},
            ) as mock_run,
        ):
            response = client.post(
                "/copilotkit/chat",
                json={
                    "query": "Show my sessions",
                    "user_id": "real-token-user",
                    "request_id": "req-1",
                },
            )

        assert response.status_code == 200
        assert mock_run.call_args.kwargs["user_id"] == "real-token-user"

    def test_stream_chat_uses_token_identity_not_body_user_id(self):
        """The streaming endpoint must also derive identity from the token."""
        token_user = {"id": "real-token-user", "app_metadata": {"role": "admin"}}
        client = self._client_with_user(token_user)

        async def _fake_stream(*args, **kwargs):
            # Record the user_id the stream was invoked with, then yield nothing.
            _fake_stream.seen_user_id = kwargs.get("user_id")
            if False:  # pragma: no cover - generator with no yields
                yield {}

        with patch(
            "src.api.routes.chatbot_graph.stream_chatbot",
            side_effect=_fake_stream,
        ):
            response = client.post(
                "/copilotkit/chat/stream",
                json={
                    "query": "Show my sessions",
                    "user_id": "victim-user",
                    "request_id": "req-1",
                },
            )
            # Consume the streaming body so the generator actually runs.
            _ = response.text

        assert response.status_code == 200
        assert getattr(_fake_stream, "seen_user_id", None) == "real-token-user"


class TestPlaceholderActionProvenance:
    """Finding 2 [HIGH silent-mock]: get_recommendations / search_insights are
    scaffolded placeholders returning fabricated pharma numbers. They must not
    silently present fake values as real AI analysis. They are gated behind a
    feature flag (default OFF -> fail closed) and, when enabled for dev, carry
    explicit provenance markers so the UI can disclose they are placeholders.
    """

    @pytest.mark.asyncio
    async def test_get_recommendations_fails_closed_by_default(self, monkeypatch):
        from src.api.routes import copilotkit

        monkeypatch.delenv("E2I_ENABLE_PLACEHOLDER_ACTIONS", raising=False)
        result = await copilotkit.get_recommendations("Kisqali")

        # Must NOT present fabricated recommendations as real data by default.
        assert result.get("data_source") in ("unavailable", "not_implemented")
        assert not result.get("recommendations")
        assert result.get("success") is False

    @pytest.mark.asyncio
    async def test_get_recommendations_placeholder_is_disclosed_when_enabled(self, monkeypatch):
        from src.api.routes import copilotkit

        monkeypatch.setenv("E2I_ENABLE_PLACEHOLDER_ACTIONS", "1")
        result = await copilotkit.get_recommendations("Kisqali")

        # When enabled for dev, the provenance MUST be explicitly marked so the
        # UI cannot mistake placeholder advice for real analysis.
        assert result.get("data_source") == "placeholder"
        assert result.get("is_placeholder") is True
        assert "placeholder" in (result.get("disclaimer") or "").lower()

    @pytest.mark.asyncio
    async def test_search_insights_fails_closed_by_default(self, monkeypatch):
        from src.api.routes import copilotkit

        monkeypatch.delenv("E2I_ENABLE_PLACEHOLDER_ACTIONS", raising=False)
        result = await copilotkit.search_insights("HCP engagement")

        assert result.get("data_source") in ("unavailable", "not_implemented")
        assert not result.get("results")
        assert result.get("success") is False

    @pytest.mark.asyncio
    async def test_search_insights_placeholder_is_disclosed_when_enabled(self, monkeypatch):
        from src.api.routes import copilotkit

        monkeypatch.setenv("E2I_ENABLE_PLACEHOLDER_ACTIONS", "1")
        result = await copilotkit.search_insights("HCP engagement")

        assert result.get("data_source") == "placeholder"
        assert result.get("is_placeholder") is True
        assert "placeholder" in (result.get("disclaimer") or "").lower()


class TestChatErrorDoesNotLeakInternals:
    """Finding 3 [MEDIUM info-disclosure]: the chat error field must not leak
    internal exception text to the client.
    """

    def test_chat_error_returns_generic_message(self, test_client):
        with patch(
            "src.api.routes.chatbot_graph.run_chatbot",
            new_callable=AsyncMock,
            side_effect=Exception("psql://secret:pw@db: relation does not exist"),
        ):
            response = test_client.post(
                "/copilotkit/chat",
                json={
                    "query": "Test query",
                    "user_id": "test-user-id",
                    "request_id": "req-456",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        # Internal exception text must NOT be present in the client response.
        assert "psql" not in (data.get("error") or "")
        assert "secret" not in (data.get("error") or "")
        assert "relation does not exist" not in (data.get("error") or "")
        # A generic message should be returned instead.
        assert data["error"]


# =============================================================================
# Bounded analytics query-limit tests (disputed-sweep finding #4)
# =============================================================================


class TestAnalyticsBoundedLimits:
    """Out-of-range days/limit query params must be rejected with 422.

    Previously these were bare ``int`` defaults (days=30/7, limit=20) with no
    bounds, so a caller could request an arbitrarily large/zero/negative window
    and force an unbounded scan. They are now ``Query(..., ge=1, le=N)``.
    """

    def test_feedback_stats_days_too_large(self, test_client):
        response = test_client.get("/copilotkit/feedback/stats?days=10000")
        assert response.status_code == 422

    def test_feedback_stats_days_zero(self, test_client):
        response = test_client.get("/copilotkit/feedback/stats?days=0")
        assert response.status_code == 422

    def test_usage_analytics_days_too_large(self, test_client):
        response = test_client.get("/copilotkit/analytics/usage?days=10000")
        assert response.status_code == 422

    def test_agent_analytics_days_negative(self, test_client):
        response = test_client.get("/copilotkit/analytics/agents?days=-1")
        assert response.status_code == 422

    def test_error_analytics_limit_too_large(self, test_client):
        response = test_client.get("/copilotkit/analytics/errors?limit=100000")
        assert response.status_code == 422

    def test_error_analytics_limit_zero(self, test_client):
        response = test_client.get("/copilotkit/analytics/errors?limit=0")
        assert response.status_code == 422

    def test_hourly_pattern_days_too_large(self, test_client):
        response = test_client.get("/copilotkit/analytics/hourly?days=10000")
        assert response.status_code == 422

    def test_within_bounds_still_accepted(self, test_client):
        """A within-bounds request must NOT 422 (no over-tightening)."""
        with patch("src.repositories.get_chatbot_analytics_repository") as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_recent_errors = AsyncMock(return_value=[])
            mock_repo.return_value = mock_repo_instance
            response = test_client.get("/copilotkit/analytics/errors?limit=200")
        assert response.status_code == 200


class TestResolveChatBrand:
    """H1 (#694): chat brand_context must be within the caller's grants, else a
    user could poison another tenant's scoped causal-graph view via the analysis
    write path. TESTING_MODE bypasses real auth, so these patch it off."""

    @staticmethod
    def _viewer(*brands: str) -> dict:
        return {"id": "u1", "role": "viewer", "brands": list(brands)}

    @staticmethod
    def _admin() -> dict:
        return {"id": "a1", "role": "admin", "brands": []}

    def test_granted_brand_allowed(self) -> None:
        from src.api.routes.copilotkit import _resolve_chat_brand

        with patch("src.api.routes.copilotkit.TESTING_MODE", False):
            assert _resolve_chat_brand(self._viewer("Kisqali"), "Kisqali") == "Kisqali"

    def test_cross_grant_brand_rejected(self) -> None:
        from fastapi import HTTPException

        from src.api.routes.copilotkit import _resolve_chat_brand

        with patch("src.api.routes.copilotkit.TESTING_MODE", False):
            with pytest.raises(HTTPException) as exc:
                _resolve_chat_brand(self._viewer("Kisqali"), "Fabhalta")
            assert exc.value.status_code == 403

    def test_admin_any_brand_allowed(self) -> None:
        from src.api.routes.copilotkit import _resolve_chat_brand

        with patch("src.api.routes.copilotkit.TESTING_MODE", False):
            assert _resolve_chat_brand(self._admin(), "Fabhalta") == "Fabhalta"

    def test_empty_brand_allowed(self) -> None:
        from src.api.routes.copilotkit import _resolve_chat_brand

        with patch("src.api.routes.copilotkit.TESTING_MODE", False):
            assert _resolve_chat_brand(self._viewer("Kisqali"), "") == ""
            assert _resolve_chat_brand(self._viewer("Kisqali"), None) == ""


# =============================================================================
# TESTS - #1240 Copilot learning-signal collection
# =============================================================================


class TestCopilotLearningSignals:
    """#1240: every completed copilot turn must produce an honestly graded
    learning_signals row matching the feedback learner's contract
    (LearningSignalsFeedbackStore: signal_details carries type/query/response/
    reward/metadata + domain_signal='dspy_signal'; is_synthetic=false).

    Only the 'agent' (synthesis) component is graded — the copilot path runs
    no summarizer/investigator, so fabricating those signals would be
    dishonest attribution. The reward derivation mirrors the cognitive
    workflow's agent_reward observables with tool results as the evidence
    analog. Faithful end-to-end flow (real chat turn -> real row) is verified
    live post-deploy per the issue's evidence protocol.
    """

    def test_grade_zero_when_no_response(self) -> None:
        from src.api.routes.copilotkit import _grade_copilot_turn

        assert _grade_copilot_turn(response="", tool_count=3) == 0.0

    def test_grade_direct_answer_baseline(self) -> None:
        """A substantive direct answer (no tools): (base 0.5 + 0.1 length)
        rescaled by the 0.8 surface max (codex-1240 M2 calibration)."""
        from src.api.routes.copilotkit import _grade_copilot_turn

        reward = _grade_copilot_turn(response="x" * 250, tool_count=0)
        assert reward == pytest.approx(0.75)

    def test_grade_tool_grounded_answer_reaches_full_band(self) -> None:
        """Tool results are the evidence analog: 0.2 * min(1, n/4). A
        best-possible copilot turn (synthesis + full grounding + substantive
        length) must map to 1.0 — the copilot surface has no evidence-board/
        visualization analog, so its raw composite max is 0.8; both surfaces
        feed the same rating_1to5 = 1 + 4*reward mapping and the same
        avg_rating < 3.0 low-ratings gate, so an uncalibrated 0.8 cap would
        structurally depress the shared agent-quality average (codex-1240 M2)."""
        from src.api.routes.copilotkit import _grade_copilot_turn

        reward = _grade_copilot_turn(response="x" * 250, tool_count=4)
        assert reward == pytest.approx(1.0)
        # Never exceeds 1.0
        assert _grade_copilot_turn(response="x" * 250, tool_count=12) <= 1.0

    def test_grade_synthesis_error_drops_base(self) -> None:
        """A failed synthesis serves a raw tool dump — not a synthesis. The
        0.5 base is forfeited; only the observable components remain (raw
        0.3), still normalized by the FULL surface max so degraded turns
        can never reach the top band."""
        from src.api.routes.copilotkit import _grade_copilot_turn

        reward = _grade_copilot_turn(response="x" * 250, tool_count=4, synthesis_error=True)
        assert reward == pytest.approx(0.375)

    @pytest.mark.asyncio
    async def test_collect_signal_matches_learner_contract(self, monkeypatch) -> None:
        """The persisted row must be exactly what LearningSignalsFeedbackStore
        consumes: details.type/query/response/reward/metadata.routed_agents/
        conversation_id + domain_signal, via record_learning_signal."""
        import src.api.routes.copilotkit as mod

        captured = {}

        async def _fake_record(signal, cycle_id=None, session_id=None):
            captured["signal"] = signal
            captured["session_id"] = session_id
            return "sig-id"

        monkeypatch.setattr("src.memory.procedural_memory.record_learning_signal", _fake_record)

        await mod._collect_copilot_learning_signal(
            query="What drives Kisqali TRx?",
            response="Kisqali TRx is driven by...",
            tool_names=["kpi_calculate_tool", "causal_analysis_tool"],
            conversation_id="thread-abc",
            synthesis_error=False,
        )

        sig = captured["signal"]
        details = sig.signal_details
        assert details["domain_signal"] == "dspy_signal"
        assert details["type"] == "agent"
        assert details["query"] == "What drives Kisqali TRx?"
        assert details["response"].startswith("Kisqali TRx")
        assert isinstance(details["reward"], float)
        meta = details["metadata"]
        assert meta["routed_agents"] == ["copilotkit"]
        assert meta["tools_invoked"] == ["kpi_calculate_tool", "causal_analysis_tool"]
        assert meta["conversation_id"] == "thread-abc"
        assert meta["source_path"] == "copilotkit"
        assert sig.signal_type == "rating"
        assert sig.is_training_example is True
        assert sig.signal_value == details["reward"]

    @pytest.mark.asyncio
    async def test_collect_signal_never_raises(self, monkeypatch) -> None:
        """Signal collection is best-effort: a writer failure must never
        unwind into the chat response path."""

        import src.api.routes.copilotkit as mod

        async def _boom(signal, cycle_id=None, session_id=None):
            raise RuntimeError("db down")

        monkeypatch.setattr("src.memory.procedural_memory.record_learning_signal", _boom)

        # Must not raise.
        await mod._collect_copilot_learning_signal(
            query="q",
            response="r" * 300,
            tool_names=[],
            conversation_id="t",
        )

    def test_graph_nodes_wire_signal_collection(self) -> None:
        """Wiring guard: ALL THREE terminal paths call the collector —
        synthesize success, synthesize error-fallback (the degraded-turn
        site that prevents selection bias; codex-1240 LOW: a >= 2 guard let
        it be silently removed), and chat_node's direct answer. Source-level
        check because the nodes are closures inside create_e2i_chat_agent;
        the live flow is verified post-deploy."""
        import inspect

        import src.api.routes.copilotkit as mod

        source = inspect.getsource(mod.create_e2i_chat_agent)
        assert source.count("_collect_copilot_learning_signal") == 3, (
            "synthesize_node (success AND error-fallback) and chat_node's "
            "direct-answer path must each collect a learning signal"
        )
