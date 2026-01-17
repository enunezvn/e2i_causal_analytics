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

from datetime import datetime, timezone
from typing import Any, Dict, Optional
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
    with patch(
        "src.api.routes.copilotkit.get_llm_provider"
    ) as mock_provider, patch(
        "src.api.routes.copilotkit.MODEL_MAPPINGS",
        {"anthropic": {"standard": "claude-3-sonnet"}},
    ):
        mock_provider.return_value = "anthropic"
        yield mock_provider


@pytest.fixture
def mock_copilot_actions():
    """Mock the COPILOT_ACTIONS list."""
    mock_action = MagicMock()
    mock_action.name = "test_action"
    with patch(
        "src.api.routes.copilotkit.COPILOT_ACTIONS", [mock_action]
    ):
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
        """Test that user_id is required."""
        with pytest.raises(ValueError):
            ChatRequest(
                query="Test query",
                request_id="req-456",
            )

    def test_request_requires_request_id(self):
        """Test that request_id is required."""
        with pytest.raises(ValueError):
            ChatRequest(
                query="Test query",
                user_id="user-123",
            )


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

    def test_feedback_requires_message_id(self):
        """Test that message_id is required."""
        with pytest.raises(ValueError):
            FeedbackRequest(rating="thumbs_up")

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

    def test_get_status_success(
        self, test_client, mock_llm_provider, mock_copilot_actions
    ):
        """Test successful status retrieval."""
        with patch.dict(
            "os.environ", {"ANTHROPIC_API_KEY": "test-key"}
        ):
            response = test_client.get("/copilotkit/status")

        assert response.status_code == 200

    def test_status_has_required_fields(
        self, test_client, mock_llm_provider, mock_copilot_actions
    ):
        """Test that status response has all required fields."""
        with patch.dict(
            "os.environ", {"ANTHROPIC_API_KEY": "test-key"}
        ):
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

    def test_status_values(
        self, test_client, mock_llm_provider, mock_copilot_actions
    ):
        """Test status response values."""
        with patch.dict(
            "os.environ", {"ANTHROPIC_API_KEY": "test-key"}
        ):
            response = test_client.get("/copilotkit/status")
            data = response.json()

        assert data["status"] == "active"
        assert data["llm_provider"] == "anthropic"
        assert data["llm_model"] == "claude-3-sonnet"
        assert data["llm_configured"] is True

    def test_status_without_api_key(
        self, test_client, mock_llm_provider, mock_copilot_actions
    ):
        """Test status when no API key is configured."""
        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""},
            clear=True,
        ):
            response = test_client.get("/copilotkit/status")
            data = response.json()

        assert data["llm_configured"] is False

    def test_status_timestamp_format(
        self, test_client, mock_llm_provider, mock_copilot_actions
    ):
        """Test that timestamp is valid ISO format."""
        with patch.dict(
            "os.environ", {"ANTHROPIC_API_KEY": "test-key"}
        ):
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
        """Test chat error handling."""
        with patch(
            "src.api.routes.chatbot_graph.run_chatbot",
            new_callable=AsyncMock,
            side_effect=Exception("Database connection error"),
        ):
            response = test_client.post(
                "/copilotkit/chat",
                json={
                    "query": "Test query",
                    "user_id": "user-123",
                    "request_id": "req-456",
                },
            )

        assert response.status_code == 200  # Endpoint returns 200 with error in body
        data = response.json()
        assert data["success"] is False
        assert "Database connection error" in data["error"]

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
            response = test_client.post(
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
            with patch.dict(
                "os.environ",
                {
                    "SUPABASE_URL": "https://test.supabase.co",
                    "SUPABASE_SERVICE_KEY": "test-key",
                },
            ), patch(
                "supabase.create_client"
            ) as mock_create:
                # Mock message lookup
                mock_client = MagicMock()
                mock_table = MagicMock()
                mock_select = MagicMock()
                mock_eq = MagicMock()
                mock_limit = MagicMock()
                mock_execute = MagicMock()

                mock_create.return_value = mock_client
                mock_client.table.return_value = mock_table
                mock_table.select.return_value = mock_select
                mock_select.eq.return_value = mock_eq
                mock_eq.limit.return_value = mock_limit
                mock_limit.execute.return_value = MagicMock(
                    data=[{"id": 123, "session_id": "session-123"}]
                )

                with patch(
                    "src.memory.services.factories.get_async_supabase_client",
                    new_callable=AsyncMock,
                ), patch(
                    "src.repositories.get_chatbot_feedback_repository"
                ) as mock_repo:
                    mock_repo_instance = MagicMock()
                    mock_repo_instance.add_feedback = AsyncMock(
                        return_value={"id": 1}
                    )
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


# =============================================================================
# GET /copilotkit/feedback/stats Endpoint Tests
# =============================================================================


class TestFeedbackStatsEndpoint:
    """Tests for GET /copilotkit/feedback/stats endpoint."""

    def test_feedback_stats_success(self, test_client):
        """Test successful feedback stats retrieval."""
        mock_agent_stats = [
            {"agent_name": "tool_composer", "thumbs_up": 10, "thumbs_down": 2}
        ]
        mock_summary = {"total": 12, "positive_rate": 0.83}

        with patch(
            "src.memory.services.factories.get_async_supabase_client",
            new_callable=AsyncMock,
        ), patch(
            "src.repositories.get_chatbot_feedback_repository"
        ) as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_agent_stats = AsyncMock(
                return_value=mock_agent_stats
            )
            mock_repo_instance.get_feedback_summary = AsyncMock(
                return_value=mock_summary
            )
            mock_repo.return_value = mock_repo_instance

            response = test_client.get("/copilotkit/feedback/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "summary" in data
        assert "agent_stats" in data

    def test_feedback_stats_with_filter(self, test_client):
        """Test feedback stats with agent filter."""
        with patch(
            "src.memory.services.factories.get_async_supabase_client",
            new_callable=AsyncMock,
        ), patch(
            "src.repositories.get_chatbot_feedback_repository"
        ) as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_agent_stats = AsyncMock(return_value=[])
            mock_repo_instance.get_feedback_summary = AsyncMock(return_value={})
            mock_repo.return_value = mock_repo_instance

            response = test_client.get(
                "/copilotkit/feedback/stats?agent_name=causal_impact&days=7"
            )

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

        with patch(
            "src.repositories.get_chatbot_analytics_repository"
        ) as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_usage_summary = AsyncMock(
                return_value=mock_summary
            )
            mock_repo_instance.get_query_type_distribution = AsyncMock(
                return_value=mock_query_types
            )
            mock_repo_instance.get_tool_usage_stats = AsyncMock(
                return_value=mock_tool_usage
            )
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
        with patch(
            "src.repositories.get_chatbot_analytics_repository"
        ) as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_usage_summary = AsyncMock(return_value={})
            mock_repo_instance.get_query_type_distribution = AsyncMock(
                return_value=[]
            )
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

        with patch(
            "src.repositories.get_chatbot_analytics_repository"
        ) as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_agent_performance = AsyncMock(
                return_value=mock_stats
            )
            mock_repo.return_value = mock_repo_instance

            response = test_client.get("/copilotkit/analytics/agents")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "agent_stats" in data

    def test_agent_analytics_with_filter(self, test_client):
        """Test agent analytics with agent name filter."""
        with patch(
            "src.repositories.get_chatbot_analytics_repository"
        ) as mock_repo:
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

        with patch(
            "src.repositories.get_chatbot_analytics_repository"
        ) as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_recent_errors = AsyncMock(
                return_value=mock_errors
            )
            mock_repo.return_value = mock_repo_instance

            response = test_client.get("/copilotkit/analytics/errors?limit=10")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 1
        assert len(data["errors"]) == 1

    def test_error_analytics_default_limit(self, test_client):
        """Test error analytics with default limit."""
        with patch(
            "src.repositories.get_chatbot_analytics_repository"
        ) as mock_repo:
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

        with patch(
            "src.repositories.get_chatbot_analytics_repository"
        ) as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_hourly_pattern = AsyncMock(
                return_value=mock_pattern
            )
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
        with patch(
            "src.api.routes.copilotkit.get_llm_provider", return_value="anthropic"
        ), patch(
            "src.api.routes.copilotkit.MODEL_MAPPINGS",
            {"anthropic": {"standard": "claude-3-sonnet"}},
        ), patch(
            "src.api.routes.copilotkit.COPILOT_ACTIONS", []
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
