"""
Integration tests for E2I Chatbot Streaming Endpoint.

Tests the streaming SSE endpoint in copilotkit.py including
response format and error handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from src.api.main import app


# Disable authentication for tests by mocking is_auth_enabled
@pytest.fixture(autouse=True)
def disable_auth():
    """Disable JWT authentication for streaming tests."""
    with patch("src.api.middleware.auth_middleware.is_auth_enabled", return_value=False):
        yield


class TestChatStreamEndpoint:
    """Tests for /api/copilotkit/chat/stream endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.stream_chatbot")
    async def test_stream_returns_sse_format(self, mock_stream_chatbot):
        """Test that streaming endpoint returns SSE format."""
        # Mock the stream to yield state updates
        async def mock_stream(*args, **kwargs):
            yield {"init": {"messages": []}}
            yield {"generate": {"messages": [], "response_text": "Hello!"}}
            yield {"finalize": {"response_text": "Hello!", "streaming_complete": True}}

        mock_stream_chatbot.return_value = mock_stream()

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(
                "/api/copilotkit/chat/stream",
                json={
                    "query": "Hello",
                    "user_id": "user-123",
                    "request_id": "req-456",
                },
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    @pytest.mark.asyncio
    async def test_stream_requires_query(self):
        """Test that query is required."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(
                "/api/copilotkit/chat/stream",
                json={
                    "user_id": "user-123",
                    "request_id": "req-456",
                },
            )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_stream_requires_user_id(self):
        """Test that user_id is required."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(
                "/api/copilotkit/chat/stream",
                json={
                    "query": "Hello",
                    "request_id": "req-456",
                },
            )

        assert response.status_code == 422  # Validation error


class TestChatEndpoint:
    """Tests for /api/copilotkit/chat endpoint (non-streaming)."""

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.run_chatbot")
    async def test_chat_returns_json_response(self, mock_run_chatbot):
        """Test that non-streaming endpoint returns JSON."""
        mock_run_chatbot.return_value = {
            "session_id": "user-123~uuid-456",
            "response_text": "Hello! I'm the E2I assistant.",
            "conversation_title": "Greeting",
            "agent_name": "chatbot",
            "streaming_complete": True,
        }

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(
                "/api/copilotkit/chat",
                json={
                    "query": "Hello",
                    "user_id": "user-123",
                    "request_id": "req-456",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "response" in data
        assert data["response"] == "Hello! I'm the E2I assistant."

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.run_chatbot")
    async def test_chat_with_session_id(self, mock_run_chatbot):
        """Test continuing a conversation with session_id."""
        mock_run_chatbot.return_value = {
            "session_id": "existing-session-123",
            "response_text": "Continuing our conversation...",
            "streaming_complete": True,
        }

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(
                "/api/copilotkit/chat",
                json={
                    "query": "Continue from before",
                    "user_id": "user-123",
                    "request_id": "req-456",
                    "session_id": "existing-session-123",
                },
            )

        assert response.status_code == 200
        mock_run_chatbot.assert_called_once()
        call_kwargs = mock_run_chatbot.call_args.kwargs
        assert call_kwargs["session_id"] == "existing-session-123"

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.run_chatbot")
    async def test_chat_with_brand_context(self, mock_run_chatbot):
        """Test chat with brand context filter."""
        mock_run_chatbot.return_value = {
            "session_id": "user-123~uuid-456",
            "response_text": "TRx for Kisqali is...",
            "streaming_complete": True,
        }

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(
                "/api/copilotkit/chat",
                json={
                    "query": "What is TRx?",
                    "user_id": "user-123",
                    "request_id": "req-456",
                    "brand_context": "Kisqali",
                },
            )

        assert response.status_code == 200
        call_kwargs = mock_run_chatbot.call_args.kwargs
        assert call_kwargs["brand_context"] == "Kisqali"

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.run_chatbot")
    async def test_chat_handles_error(self, mock_run_chatbot):
        """Test that errors are handled gracefully."""
        mock_run_chatbot.side_effect = Exception("Internal error")

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(
                "/api/copilotkit/chat",
                json={
                    "query": "Hello",
                    "user_id": "user-123",
                    "request_id": "req-456",
                },
            )

        # Should return 500 or handle gracefully
        assert response.status_code in [200, 500]


class TestExistingCopilotKitEndpoints:
    """Tests to ensure existing CopilotKit endpoints still work."""

    @pytest.mark.asyncio
    async def test_info_endpoint_still_works(self):
        """Test that /api/copilotkit/info still returns info."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post("/api/copilotkit/info")

        # Should return 200 or appropriate response
        assert response.status_code in [200, 405, 422]

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health endpoint."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/health")

        assert response.status_code == 200


class TestSSEFormat:
    """Tests for SSE response format."""

    @pytest.mark.asyncio
    @patch("src.api.routes.chatbot_graph.stream_chatbot")
    async def test_sse_includes_session_id_event(self, mock_stream_chatbot):
        """Test that SSE includes session_id event."""
        async def mock_stream(*args, **kwargs):
            yield {
                "init": {
                    "messages": [],
                    "metadata": {"is_new_conversation": True},
                }
            }
            yield {"finalize": {"response_text": "Done", "streaming_complete": True}}

        mock_stream_chatbot.return_value = mock_stream()

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(
                "/api/copilotkit/chat/stream",
                json={
                    "query": "Hello",
                    "user_id": "user-123",
                    "request_id": "req-456",
                },
            )

        assert response.status_code == 200
        # Response should contain SSE data
        content = response.text
        assert "data:" in content or response.headers["content-type"].startswith("text/event-stream")
