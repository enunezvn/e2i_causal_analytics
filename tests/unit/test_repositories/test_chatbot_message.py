"""
Unit tests for ChatbotMessageRepository.

Tests message management including creation, retrieval, and statistics.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.repositories.chatbot_message import (
    ChatbotMessageRepository,
    get_chatbot_message_repository,
)


class TestChatbotMessageRepository:
    """Tests for ChatbotMessageRepository."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Supabase client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def repo(self, mock_client):
        """Create repository with mock client."""
        return ChatbotMessageRepository(mock_client)

    @pytest.fixture
    def sample_user_message(self):
        """Sample user message data."""
        return {
            "id": "uuid-msg-1",
            "session_id": "user-123~uuid-456",
            "role": "user",
            "content": "What is the TRx trend for Kisqali?",
            "agent_name": None,
            "agent_tier": None,
            "tool_calls": [],
            "tool_results": [],
            "rag_context": [],
            "rag_sources": [],
            "model_used": None,
            "tokens_used": None,
            "latency_ms": None,
            "confidence_score": None,
            "metadata": {"request_id": "req-123"},
            "created_at": "2025-01-05T10:00:00Z",
        }

    @pytest.fixture
    def sample_assistant_message(self):
        """Sample assistant message data."""
        return {
            "id": "uuid-msg-2",
            "session_id": "user-123~uuid-456",
            "role": "assistant",
            "content": "TRx for Kisqali shows 12% growth over the last quarter...",
            "agent_name": "chatbot",
            "agent_tier": 1,
            "tool_calls": [{"tool_name": "e2i_data_query_tool", "args": {"query_type": "kpi"}}],
            "tool_results": [{"kpi": "trx", "value": 1500}],
            "rag_context": [{"source": "business_metrics", "content": "..."}],
            "rag_sources": ["business_metrics"],
            "model_used": "claude-sonnet-4-20250514",
            "tokens_used": 450,
            "latency_ms": 1200,
            "confidence_score": 0.95,
            "metadata": {"request_id": "req-123", "intent": "kpi_query"},
            "created_at": "2025-01-05T10:00:05Z",
        }


class TestAddMessage(TestChatbotMessageRepository):
    """Tests for add_message method."""

    @pytest.mark.asyncio
    async def test_adds_user_message_successfully(self, repo, mock_client, sample_user_message):
        """Test adding a user message."""
        mock_result = MagicMock()
        mock_result.data = [sample_user_message]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.insert.return_value.execute = mock_execute

        result = await repo.add_message(
            session_id="user-123~uuid-456",
            role="user",
            content="What is the TRx trend for Kisqali?",
            metadata={"request_id": "req-123"},
        )

        assert result is not None
        assert result["role"] == "user"
        assert result["content"] == "What is the TRx trend for Kisqali?"

    @pytest.mark.asyncio
    async def test_adds_assistant_message_with_full_context(
        self, repo, mock_client, sample_assistant_message
    ):
        """Test adding an assistant message with all metadata."""
        mock_result = MagicMock()
        mock_result.data = [sample_assistant_message]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.insert.return_value.execute = mock_execute

        result = await repo.add_message(
            session_id="user-123~uuid-456",
            role="assistant",
            content="TRx for Kisqali shows 12% growth over the last quarter...",
            agent_name="chatbot",
            agent_tier=1,
            tool_calls=[{"tool_name": "e2i_data_query_tool", "args": {"query_type": "kpi"}}],
            tool_results=[{"kpi": "trx", "value": 1500}],
            rag_context=[{"source": "business_metrics", "content": "..."}],
            rag_sources=["business_metrics"],
            model_used="claude-sonnet-4-20250514",
            tokens_used=450,
            latency_ms=1200,
            confidence_score=0.95,
            metadata={"request_id": "req-123", "intent": "kpi_query"},
        )

        assert result is not None
        assert result["role"] == "assistant"
        assert result["agent_name"] == "chatbot"
        assert result["tokens_used"] == 450

    @pytest.mark.asyncio
    async def test_returns_none_when_no_client(self):
        """Test that None is returned when client is None."""
        repo = ChatbotMessageRepository(None)

        result = await repo.add_message(
            session_id="user-123~uuid-456",
            role="user",
            content="Test message",
        )

        assert result is None


class TestGetSessionMessages(TestChatbotMessageRepository):
    """Tests for get_session_messages method."""

    @pytest.mark.asyncio
    async def test_returns_session_messages(
        self, repo, mock_client, sample_user_message, sample_assistant_message
    ):
        """Test that session messages are returned."""
        mock_result = MagicMock()
        mock_result.data = [sample_user_message, sample_assistant_message]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.offset.return_value.execute = mock_execute

        result = await repo.get_session_messages("user-123~uuid-456")

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_client(self):
        """Test that empty list is returned when client is None."""
        repo = ChatbotMessageRepository(None)

        result = await repo.get_session_messages("user-123~uuid-456")

        assert result == []


class TestGetRecentMessages(TestChatbotMessageRepository):
    """Tests for get_recent_messages method."""

    @pytest.mark.asyncio
    async def test_returns_recent_messages_in_chronological_order(
        self, repo, mock_client, sample_user_message, sample_assistant_message
    ):
        """Test that recent messages are returned in chronological order."""
        # Mock returns newest first (desc order)
        mock_result = MagicMock()
        mock_result.data = [sample_assistant_message, sample_user_message]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_recent_messages("user-123~uuid-456", count=10)

        # Should be reversed to chronological order
        assert len(result) == 2
        assert result[0]["role"] == "user"  # Older message first
        assert result[1]["role"] == "assistant"  # Newer message second


class TestGetByRole(TestChatbotMessageRepository):
    """Tests for get_by_role method."""

    @pytest.mark.asyncio
    async def test_returns_messages_by_role(self, repo, mock_client, sample_assistant_message):
        """Test filtering messages by role."""
        mock_result = MagicMock()
        mock_result.data = [sample_assistant_message]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_by_role("user-123~uuid-456", "assistant")

        assert len(result) == 1
        assert result[0]["role"] == "assistant"


class TestGetMessageCount(TestChatbotMessageRepository):
    """Tests for get_message_count method."""

    @pytest.mark.asyncio
    async def test_returns_message_count(self, repo, mock_client):
        """Test getting message count."""
        mock_result = MagicMock()
        mock_result.count = 10
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.execute = mock_execute

        result = await repo.get_message_count("user-123~uuid-456")

        assert result == 10

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_client(self):
        """Test that zero is returned when client is None."""
        repo = ChatbotMessageRepository(None)

        result = await repo.get_message_count("user-123~uuid-456")

        assert result == 0


class TestGetToolUsageStats(TestChatbotMessageRepository):
    """Tests for get_tool_usage_stats method."""

    @pytest.mark.asyncio
    async def test_returns_tool_usage_stats(self, repo, mock_client):
        """Test getting tool usage statistics."""
        mock_result = MagicMock()
        mock_result.data = [
            {"tool_calls": [{"tool_name": "e2i_data_query_tool"}]},
            {
                "tool_calls": [
                    {"tool_name": "e2i_data_query_tool"},
                    {"tool_name": "causal_analysis_tool"},
                ]
            },
        ]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.neq.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_tool_usage_stats()

        assert "e2i_data_query_tool" in result
        assert result["e2i_data_query_tool"] == 2
        assert "causal_analysis_tool" in result
        assert result["causal_analysis_tool"] == 1


class TestSearchMessages(TestChatbotMessageRepository):
    """Tests for search_messages method."""

    @pytest.mark.asyncio
    async def test_searches_messages_by_content(self, repo, mock_client, sample_assistant_message):
        """Test searching messages by content."""
        mock_result = MagicMock()
        mock_result.data = [sample_assistant_message]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.ilike.return_value.order.return_value.limit.return_value.execute = mock_execute

        result = await repo.search_messages("TRx")

        assert len(result) == 1
        assert "TRx" in result[0]["content"]


class TestFactoryFunction:
    """Tests for factory function."""

    def test_creates_repository_with_client(self):
        """Test factory creates repository with client."""
        mock_client = MagicMock()
        repo = get_chatbot_message_repository(mock_client)

        assert isinstance(repo, ChatbotMessageRepository)
        assert repo.client == mock_client

    def test_creates_repository_without_client(self):
        """Test factory creates repository without client."""
        repo = get_chatbot_message_repository(None)

        assert isinstance(repo, ChatbotMessageRepository)
        assert repo.client is None
