"""
Unit tests for ChatbotConversationRepository.

Tests conversation management including creation, retrieval, and updates.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.repositories.chatbot_conversation import (
    ChatbotConversationRepository,
    get_chatbot_conversation_repository,
)


class TestChatbotConversationRepository:
    """Tests for ChatbotConversationRepository."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Supabase client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def repo(self, mock_client):
        """Create repository with mock client."""
        return ChatbotConversationRepository(mock_client)

    @pytest.fixture
    def sample_conversation(self):
        """Sample conversation data."""
        return {
            "id": "uuid-conv-123",
            "user_id": "user-123",
            "session_id": "user-123~uuid-456",
            "title": "KPI Analysis for Kisqali",
            "brand_context": "Kisqali",
            "region_context": "US",
            "query_type": "kpi_query",
            "is_archived": False,
            "is_pinned": False,
            "message_count": 4,
            "last_message_at": "2025-01-05T15:30:00Z",
            "created_at": "2025-01-05T10:00:00Z",
            "updated_at": "2025-01-05T15:30:00Z",
        }


class TestCreateConversation(TestChatbotConversationRepository):
    """Tests for create_conversation method."""

    @pytest.mark.asyncio
    async def test_creates_conversation_successfully(self, repo, mock_client, sample_conversation):
        """Test creating a new conversation."""
        mock_result = MagicMock()
        mock_result.data = [sample_conversation]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.insert.return_value.execute = mock_execute

        result = await repo.create_conversation(
            user_id="user-123",
            session_id="user-123~uuid-456",
            brand_context="Kisqali",
            region_context="US",
            query_type="kpi_query",
        )

        assert result is not None
        assert result["session_id"] == "user-123~uuid-456"
        assert result["brand_context"] == "Kisqali"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_client(self):
        """Test that None is returned when client is None."""
        repo = ChatbotConversationRepository(None)

        result = await repo.create_conversation(
            user_id="user-123",
            session_id="user-123~uuid-456",
        )

        assert result is None


class TestGetBySessionId(TestChatbotConversationRepository):
    """Tests for get_by_session_id method."""

    @pytest.mark.asyncio
    async def test_returns_conversation_when_found(self, repo, mock_client, sample_conversation):
        """Test that conversation is returned when found."""
        mock_result = MagicMock()
        # .limit(1) returns a LIST of rows (#1335: was .single()).
        mock_result.data = [sample_conversation]

        # Mock the chain: table().select().eq().limit().execute()
        mock_limit = MagicMock()
        mock_limit.execute = AsyncMock(return_value=mock_result)
        mock_eq = MagicMock()
        mock_eq.limit.return_value = mock_limit
        mock_select = MagicMock()
        mock_select.eq.return_value = mock_eq
        mock_table = MagicMock()
        mock_table.select.return_value = mock_select
        mock_client.table.return_value = mock_table

        result = await repo.get_by_session_id("user-123~uuid-456")

        assert result is not None
        assert result["session_id"] == "user-123~uuid-456"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, repo, mock_client):
        """Test that None is returned when not found.

        #1335: models the REAL PostgREST 0-row contract -- ``.limit(1)`` returns
        an EMPTY LIST (the prior ``.single()`` mock returned ``data=None``, a
        shape that never occurs in prod because ``.single()`` *raises* PGRST116
        on 0 rows; that mock-vs-reality gap hid the FK-violation bug).
        """
        mock_result = MagicMock()
        mock_result.data = []

        # Mock the chain: table().select().eq().limit().execute()
        mock_limit = MagicMock()
        mock_limit.execute = AsyncMock(return_value=mock_result)
        mock_eq = MagicMock()
        mock_eq.limit.return_value = mock_limit
        mock_select = MagicMock()
        mock_select.eq.return_value = mock_eq
        mock_table = MagicMock()
        mock_table.select.return_value = mock_select
        mock_client.table.return_value = mock_table

        result = await repo.get_by_session_id("nonexistent-session")

        assert result is None


class TestGetUserConversations(TestChatbotConversationRepository):
    """Tests for get_user_conversations method."""

    @pytest.mark.asyncio
    async def test_returns_user_conversations(self, repo, mock_client, sample_conversation):
        """Test that user conversations are returned."""
        mock_result = MagicMock()
        mock_result.data = [sample_conversation]

        # Mock the chain: table().select().eq().order().limit().eq().execute()
        # (second eq is for is_archived filter when include_archived=False)
        mock_eq2 = MagicMock()
        mock_eq2.execute = AsyncMock(return_value=mock_result)
        mock_limit = MagicMock()
        mock_limit.eq.return_value = mock_eq2
        mock_order = MagicMock()
        mock_order.limit.return_value = mock_limit
        mock_eq1 = MagicMock()
        mock_eq1.order.return_value = mock_order
        mock_select = MagicMock()
        mock_select.eq.return_value = mock_eq1
        mock_table = MagicMock()
        mock_table.select.return_value = mock_select
        mock_client.table.return_value = mock_table

        result = await repo.get_user_conversations("user-123")

        assert len(result) == 1
        assert result[0]["user_id"] == "user-123"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_client(self):
        """Test that empty list is returned when client is None."""
        repo = ChatbotConversationRepository(None)

        result = await repo.get_user_conversations("user-123")

        assert result == []


class TestUpdateTitle(TestChatbotConversationRepository):
    """Tests for update_title method."""

    @pytest.mark.asyncio
    async def test_updates_title_successfully(self, repo, mock_client, sample_conversation):
        """Test updating conversation title."""
        updated_conv = {**sample_conversation, "title": "New Title"}
        mock_result = MagicMock()
        mock_result.data = [updated_conv]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

        result = await repo.update_title("user-123~uuid-456", "New Title")

        assert result is not None
        assert result["title"] == "New Title"


class TestArchiveConversation(TestChatbotConversationRepository):
    """Tests for archive_conversation method."""

    @pytest.mark.asyncio
    async def test_archives_conversation_successfully(self, repo, mock_client, sample_conversation):
        """Test archiving a conversation."""
        archived_conv = {**sample_conversation, "is_archived": True}
        mock_result = MagicMock()
        mock_result.data = [archived_conv]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

        result = await repo.archive_conversation("user-123~uuid-456")

        assert result is not None
        assert result["is_archived"] is True


class TestPinConversation(TestChatbotConversationRepository):
    """Tests for pin_conversation method."""

    @pytest.mark.asyncio
    async def test_pins_conversation_successfully(self, repo, mock_client, sample_conversation):
        """Test pinning a conversation."""
        pinned_conv = {**sample_conversation, "is_pinned": True}
        mock_result = MagicMock()
        mock_result.data = [pinned_conv]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

        result = await repo.pin_conversation("user-123~uuid-456", pinned=True)

        assert result is not None
        assert result["is_pinned"] is True


class TestUpdateMetadata(TestChatbotConversationRepository):
    """Tests for update_metadata method (#1407 multi-turn clarification home)."""

    def _mock_update_chain(self, mock_client, returned_row):
        """Wire table().update().eq().execute() and return the update MagicMock."""
        mock_result = MagicMock()
        mock_result.data = [returned_row]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_update = MagicMock()
        mock_update.return_value.eq.return_value.execute = mock_execute
        mock_client.table.return_value.update = mock_update
        return mock_update

    @pytest.mark.asyncio
    async def test_merges_patch_preserving_other_keys(self, repo, mock_client, sample_conversation):
        """A patch merges into existing metadata without dropping other keys."""
        existing = {**sample_conversation, "metadata": {"keep": 1}}
        repo.get_by_session_id = AsyncMock(return_value=existing)
        mock_update = self._mock_update_chain(mock_client, existing)

        pending = {"original_query": "why did it drop?", "asked_at": "2026-08-02T00:00:00+00:00"}
        await repo.update_metadata("user-123~uuid-456", {"pending_clarification": pending})

        written = mock_update.call_args.args[0]["metadata"]
        assert written == {"keep": 1, "pending_clarification": pending}

    @pytest.mark.asyncio
    async def test_none_value_pops_key(self, repo, mock_client, sample_conversation):
        """A None value in the patch POPS that key (used to CLEAR pending)."""
        existing = {
            **sample_conversation,
            "metadata": {"keep": 1, "pending_clarification": {"x": 1}},
        }
        repo.get_by_session_id = AsyncMock(return_value=existing)
        mock_update = self._mock_update_chain(mock_client, existing)

        await repo.update_metadata("user-123~uuid-456", {"pending_clarification": None})

        written = mock_update.call_args.args[0]["metadata"]
        assert written == {"keep": 1}

    @pytest.mark.asyncio
    async def test_returns_none_when_no_client(self):
        repo = ChatbotConversationRepository(None)
        result = await repo.update_metadata("s", {"a": 1})
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_conversation_missing(self, repo):
        repo.get_by_session_id = AsyncMock(return_value=None)
        result = await repo.update_metadata("missing", {"a": 1})
        assert result is None


class TestFactoryFunction:
    """Tests for factory function."""

    def test_creates_repository_with_client(self):
        """Test factory creates repository with client."""
        mock_client = MagicMock()
        repo = get_chatbot_conversation_repository(mock_client)

        assert isinstance(repo, ChatbotConversationRepository)
        assert repo.client == mock_client

    def test_creates_repository_without_client(self):
        """Test factory creates repository without client."""
        repo = get_chatbot_conversation_repository(None)

        assert isinstance(repo, ChatbotConversationRepository)
        assert repo.client is None
