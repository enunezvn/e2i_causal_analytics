"""
Unit tests for ChatbotUserProfileRepository.

Tests user profile management including preferences and expertise level.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.repositories.chatbot_user_profile import (
    ChatbotUserProfileRepository,
    get_chatbot_user_profile_repository,
)


class TestChatbotUserProfileRepository:
    """Tests for ChatbotUserProfileRepository."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Supabase client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def repo(self, mock_client):
        """Create repository with mock client."""
        return ChatbotUserProfileRepository(mock_client)

    @pytest.fixture
    def sample_profile(self):
        """Sample user profile data."""
        return {
            "id": "uuid-123",
            "user_id": "user-123",
            "display_name": "Test User",
            "brand_preference": "Kisqali",
            "region_preference": "US",
            "expertise_level": "intermediate",
            "notification_preferences": {"email": True, "push": False},
            "ui_preferences": {"theme": "dark", "dashboard_layout": "compact"},
            "created_at": "2025-01-01T10:00:00Z",
            "updated_at": "2025-01-05T15:30:00Z",
        }


class TestGetByUserId(TestChatbotUserProfileRepository):
    """Tests for get_by_user_id method."""

    @pytest.mark.asyncio
    async def test_returns_profile_when_found(self, repo, mock_client, sample_profile):
        """Test that profile is returned when found."""
        mock_result = MagicMock()
        mock_result.data = [sample_profile]

        # Mock the chain: table().select().eq().execute()
        mock_eq = MagicMock()
        mock_eq.execute = AsyncMock(return_value=mock_result)
        mock_select = MagicMock()
        mock_select.eq.return_value = mock_eq
        mock_table = MagicMock()
        mock_table.select.return_value = mock_select
        mock_client.table.return_value = mock_table

        result = await repo.get_by_user_id("user-123")

        assert result is not None
        assert result["user_id"] == "user-123"
        assert result["brand_preference"] == "Kisqali"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, repo, mock_client):
        """Test that None is returned when profile not found."""
        mock_result = MagicMock()
        mock_result.data = []

        # Mock the chain: table().select().eq().execute()
        mock_eq = MagicMock()
        mock_eq.execute = AsyncMock(return_value=mock_result)
        mock_select = MagicMock()
        mock_select.eq.return_value = mock_eq
        mock_table = MagicMock()
        mock_table.select.return_value = mock_select
        mock_client.table.return_value = mock_table

        result = await repo.get_by_user_id("nonexistent-user")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_client(self):
        """Test that None is returned when client is None."""
        repo = ChatbotUserProfileRepository(None)

        result = await repo.get_by_user_id("user-123")

        assert result is None


class TestUpdatePreferences(TestChatbotUserProfileRepository):
    """Tests for update_preferences method."""

    @pytest.mark.asyncio
    async def test_updates_brand_preference(self, repo, mock_client, sample_profile):
        """Test updating brand preference."""
        updated_profile = {**sample_profile, "brand_preference": "Fabhalta"}
        mock_result = MagicMock()
        mock_result.data = [updated_profile]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

        result = await repo.update_preferences("user-123", brand_preference="Fabhalta")

        assert result is not None
        assert result["brand_preference"] == "Fabhalta"

    @pytest.mark.asyncio
    async def test_updates_expertise_level(self, repo, mock_client, sample_profile):
        """Test updating expertise level."""
        updated_profile = {**sample_profile, "expertise_level": "expert"}
        mock_result = MagicMock()
        mock_result.data = [updated_profile]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.update.return_value.eq.return_value.execute = mock_execute

        result = await repo.update_preferences("user-123", expertise_level="expert")

        assert result is not None
        assert result["expertise_level"] == "expert"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_client(self):
        """Test that None is returned when client is None."""
        repo = ChatbotUserProfileRepository(None)

        result = await repo.update_preferences("user-123", brand_preference="Kisqali")

        assert result is None


class TestFactoryFunction:
    """Tests for factory function."""

    def test_creates_repository_with_client(self):
        """Test factory creates repository with client."""
        mock_client = MagicMock()
        repo = get_chatbot_user_profile_repository(mock_client)

        assert isinstance(repo, ChatbotUserProfileRepository)
        assert repo.client == mock_client

    def test_creates_repository_without_client(self):
        """Test factory creates repository without client."""
        repo = get_chatbot_user_profile_repository(None)

        assert isinstance(repo, ChatbotUserProfileRepository)
        assert repo.client is None
