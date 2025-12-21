"""
Unit tests for ConversationRepository.

Tests vector similarity search and feedback retrieval using pgvector RPC.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.repositories.conversation import ConversationRepository


class TestConversationRepository:
    """Tests for ConversationRepository."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Supabase client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def repo(self, mock_client):
        """Create repository with mock client."""
        return ConversationRepository(supabase_client=mock_client)

    @pytest.fixture
    def sample_embedding(self):
        """Sample 1536-dimension embedding."""
        return [0.1] * 1536

    @pytest.fixture
    def sample_conversations(self):
        """Sample conversation data."""
        return [
            {
                "cycle_id": "uuid-1",
                "session_id": "session-1",
                "user_id": "user-1",
                "user_query": "What is TRx trend for Kisqali?",
                "detected_intent": "trend_analysis",
                "detected_entities": {"brand": "kisqali"},
                "agent_response": "TRx for Kisqali shows 12% growth...",
                "response_type": "chart",
                "feedback_type": None,
                "feedback_text": None,
                "similarity": 0.92,
                "created_at": "2025-01-15T10:30:00Z",
            },
            {
                "cycle_id": "uuid-2",
                "session_id": "session-1",
                "user_id": "user-1",
                "user_query": "Show Kisqali market share",
                "detected_intent": "market_analysis",
                "detected_entities": {"brand": "kisqali"},
                "agent_response": "Market share is currently at 15%...",
                "response_type": "chart",
                "feedback_type": "positive",
                "feedback_text": "Great visualization!",
                "similarity": 0.85,
                "created_at": "2025-01-15T11:00:00Z",
            },
        ]


class TestGetSimilarQueries(TestConversationRepository):
    """Tests for get_similar_queries method."""

    @pytest.mark.asyncio
    async def test_returns_similar_conversations(
        self, repo, mock_client, sample_embedding, sample_conversations
    ):
        """Test that similar conversations are returned."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.data = sample_conversations
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.rpc.return_value.execute = mock_execute

        # Call method
        result = await repo.get_similar_queries(
            query_embedding=sample_embedding,
            top_k=5,
            min_similarity=0.5,
        )

        # Verify
        assert len(result) == 2
        assert result[0]["user_query"] == "What is TRx trend for Kisqali?"
        assert result[0]["similarity"] == 0.92

    @pytest.mark.asyncio
    async def test_calls_rpc_with_correct_params(self, repo, mock_client, sample_embedding):
        """Test that RPC is called with correct parameters."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.rpc.return_value.execute = mock_execute

        await repo.get_similar_queries(
            query_embedding=sample_embedding,
            top_k=10,
            min_similarity=0.7,
        )

        # Verify RPC call
        mock_client.rpc.assert_called_once_with(
            "search_similar_conversations",
            {
                "query_embedding": sample_embedding,
                "match_count": 10,
                "min_similarity": 0.7,
            },
        )

    @pytest.mark.asyncio
    async def test_includes_user_filter(self, repo, mock_client, sample_embedding):
        """Test that user_id filter is passed to RPC."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.rpc.return_value.execute = mock_execute

        await repo.get_similar_queries(
            query_embedding=sample_embedding,
            top_k=5,
            user_id="user-123",
        )

        call_args = mock_client.rpc.call_args
        assert call_args[0][1]["filter_user_id"] == "user-123"

    @pytest.mark.asyncio
    async def test_includes_session_filter(self, repo, mock_client, sample_embedding):
        """Test that session_id filter is passed to RPC."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.rpc.return_value.execute = mock_execute

        await repo.get_similar_queries(
            query_embedding=sample_embedding,
            top_k=5,
            session_id="session-456",
        )

        call_args = mock_client.rpc.call_args
        assert call_args[0][1]["filter_session_id"] == "session-456"

    @pytest.mark.asyncio
    async def test_validates_embedding_dimensions(self, repo, mock_client):
        """Test that wrong embedding dimensions raise ValueError."""
        wrong_embedding = [0.1] * 512  # Wrong size

        with pytest.raises(ValueError) as exc_info:
            await repo.get_similar_queries(
                query_embedding=wrong_embedding,
                top_k=5,
            )

        assert "1536 dimensions" in str(exc_info.value)
        assert "512" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_returns_empty_list_without_client(self, sample_embedding):
        """Test that empty list is returned when client is None."""
        repo = ConversationRepository(supabase_client=None)

        result = await repo.get_similar_queries(
            query_embedding=sample_embedding,
            top_k=5,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_handles_no_results(self, repo, mock_client, sample_embedding):
        """Test handling when no similar conversations found."""
        mock_result = MagicMock()
        mock_result.data = None
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.rpc.return_value.execute = mock_execute

        result = await repo.get_similar_queries(
            query_embedding=sample_embedding,
            top_k=5,
        )

        assert result == []


class TestGetWithFeedback(TestConversationRepository):
    """Tests for get_with_feedback method."""

    @pytest.fixture
    def feedback_conversations(self):
        """Sample conversations with feedback."""
        return [
            {
                "cycle_id": "uuid-1",
                "user_query": "Show TRx trend",
                "feedback_type": "positive",
                "feedback_text": "Very helpful!",
                "feedback_score": 5,
            },
            {
                "cycle_id": "uuid-2",
                "user_query": "Market share analysis",
                "feedback_type": "positive",
                "feedback_text": None,
                "feedback_score": 4,
            },
        ]

    @pytest.mark.asyncio
    async def test_returns_conversations_with_feedback(
        self, repo, mock_client, feedback_conversations
    ):
        """Test that conversations with feedback are returned."""
        mock_result = MagicMock()
        mock_result.data = feedback_conversations
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.rpc.return_value.execute = mock_execute

        result = await repo.get_with_feedback(limit=100)

        assert len(result) == 2
        assert result[0]["feedback_type"] == "positive"

    @pytest.mark.asyncio
    async def test_filters_by_feedback_type(self, repo, mock_client):
        """Test filtering by feedback type."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.rpc.return_value.execute = mock_execute

        await repo.get_with_feedback(
            feedback_type="negative",
            limit=50,
        )

        call_args = mock_client.rpc.call_args
        assert call_args[0][1]["p_feedback_type"] == "negative"
        assert call_args[0][1]["p_limit"] == 50

    @pytest.mark.asyncio
    async def test_filters_by_user(self, repo, mock_client):
        """Test filtering by user_id."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.rpc.return_value.execute = mock_execute

        await repo.get_with_feedback(user_id="user-123")

        call_args = mock_client.rpc.call_args
        assert call_args[0][1]["p_user_id"] == "user-123"

    @pytest.mark.asyncio
    async def test_filters_by_date(self, repo, mock_client):
        """Test filtering by since date."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.rpc.return_value.execute = mock_execute

        since_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        await repo.get_with_feedback(since=since_date)

        call_args = mock_client.rpc.call_args
        assert "p_since" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_returns_empty_list_without_client(self):
        """Test that empty list is returned when client is None."""
        repo = ConversationRepository(supabase_client=None)

        result = await repo.get_with_feedback()

        assert result == []


class TestGetConversationContext(TestConversationRepository):
    """Tests for get_conversation_context method."""

    @pytest.mark.asyncio
    async def test_formats_context_for_rag(
        self, repo, mock_client, sample_embedding, sample_conversations
    ):
        """Test that context is formatted correctly for RAG."""
        mock_result = MagicMock()
        mock_result.data = sample_conversations
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.rpc.return_value.execute = mock_execute

        result = await repo.get_conversation_context(
            query_embedding=sample_embedding,
            context_size=3,
        )

        assert len(result) == 2
        assert "query" in result[0]
        assert "response" in result[0]
        assert "intent" in result[0]
        assert "similarity" in result[0]
        assert result[0]["query"] == "What is TRx trend for Kisqali?"

    @pytest.mark.asyncio
    async def test_uses_correct_similarity_threshold(self, repo, mock_client, sample_embedding):
        """Test that min_similarity is passed correctly."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.rpc.return_value.execute = mock_execute

        await repo.get_conversation_context(
            query_embedding=sample_embedding,
            min_similarity=0.8,
        )

        call_args = mock_client.rpc.call_args
        assert call_args[0][1]["min_similarity"] == 0.8


class TestRecordFeedback(TestConversationRepository):
    """Tests for record_feedback method."""

    @pytest.mark.asyncio
    async def test_records_positive_feedback(self, repo, mock_client):
        """Test recording positive feedback."""
        mock_result = MagicMock()
        mock_result.data = [{"cycle_id": "uuid-1", "feedback_type": "positive"}]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

        result = await repo.record_feedback(
            cycle_id="uuid-1",
            feedback_type="positive",
            feedback_text="Great analysis!",
            feedback_score=5,
        )

        assert result is not None
        assert result["feedback_type"] == "positive"

    @pytest.mark.asyncio
    async def test_records_feedback_with_score(self, repo, mock_client):
        """Test recording feedback with numeric score."""
        mock_result = MagicMock()
        mock_result.data = [{"cycle_id": "uuid-1"}]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_update = MagicMock()
        mock_update.eq.return_value.execute = mock_execute
        mock_client.table.return_value.update.return_value = mock_update

        await repo.record_feedback(
            cycle_id="uuid-1",
            feedback_type="negative",
            feedback_score=2,
        )

        # Verify update was called
        mock_client.table.assert_called_with("cognitive_cycles")

    @pytest.mark.asyncio
    async def test_returns_none_without_client(self):
        """Test that None is returned when client is None."""
        repo = ConversationRepository(supabase_client=None)

        result = await repo.record_feedback(
            cycle_id="uuid-1",
            feedback_type="positive",
        )

        assert result is None


class TestGetBySession(TestConversationRepository):
    """Tests for get_by_session method."""

    @pytest.mark.asyncio
    async def test_queries_by_session_id(self, repo, mock_client, sample_conversations):
        """Test that session_id filter is applied."""
        mock_result = MagicMock()
        mock_result.data = sample_conversations
        mock_query = MagicMock()
        mock_query.limit.return_value.offset.return_value.execute = AsyncMock(
            return_value=mock_result
        )
        mock_client.table.return_value.select.return_value.eq.return_value = mock_query

        result = await repo.get_by_session(session_id="session-1")

        assert len(result) == 2
        mock_client.table.assert_called_with("cognitive_cycles")


class TestGetByUser(TestConversationRepository):
    """Tests for get_by_user method."""

    @pytest.mark.asyncio
    async def test_queries_by_user_id(self, repo, mock_client, sample_conversations):
        """Test that user_id filter is applied."""
        mock_result = MagicMock()
        mock_result.data = sample_conversations
        mock_query = MagicMock()
        mock_query.limit.return_value.offset.return_value.execute = AsyncMock(
            return_value=mock_result
        )
        mock_client.table.return_value.select.return_value.eq.return_value = mock_query

        result = await repo.get_by_user(user_id="user-1")

        assert len(result) == 2
        mock_client.table.assert_called_with("cognitive_cycles")
