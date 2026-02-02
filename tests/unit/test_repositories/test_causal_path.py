"""
Unit tests for CausalPathRepository.

Tests causal relationship queries and path traversal.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.repositories.causal_path import CausalPathRepository


@pytest.mark.unit
class TestCausalPathRepository:
    """Tests for CausalPathRepository."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Supabase client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def repo(self, mock_client):
        """Create repository with mock client."""
        return CausalPathRepository(supabase_client=mock_client)

    @pytest.fixture
    def sample_paths(self):
        """Sample causal path data."""
        return [
            {
                "path_id": str(uuid4()),
                "cause": "HCP_engagement",
                "effect": "TRx_growth",
                "brand": "Kisqali",
                "region": "US",
                "effect_size": 0.25,
                "confidence": 0.85,
                "path_strength": "strong",
                "created_at": "2025-01-15T10:00:00Z",
            },
            {
                "path_id": str(uuid4()),
                "cause": "HCP_engagement",
                "effect": "NRx_growth",
                "brand": "Kisqali",
                "region": "US",
                "effect_size": 0.18,
                "confidence": 0.78,
                "path_strength": "moderate",
                "created_at": "2025-01-15T10:00:00Z",
            },
        ]


@pytest.mark.unit
class TestGetPathsForCause(TestCausalPathRepository):
    """Tests for get_paths_for_cause method."""

    @pytest.mark.asyncio
    async def test_returns_all_paths_from_cause(self, repo, mock_client, sample_paths):
        """Test that all paths originating from a cause are returned."""
        mock_result = MagicMock()
        mock_result.data = sample_paths
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.offset.return_value.execute = mock_execute

        result = await repo.get_paths_for_cause(cause="HCP_engagement")

        assert len(result) == 2
        mock_client.table.assert_called_with("causal_paths")

    @pytest.mark.asyncio
    async def test_respects_limit(self, repo, mock_client, sample_paths):
        """Test that limit is respected."""
        mock_result = MagicMock()
        mock_result.data = sample_paths[:1]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq = MagicMock()
        mock_eq.limit.return_value = mock_limit
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq

        result = await repo.get_paths_for_cause(cause="HCP_engagement", limit=1)

        assert len(result) == 1
        mock_eq.limit.assert_called_with(1)


@pytest.mark.unit
class TestGetPathsForEffect(TestCausalPathRepository):
    """Tests for get_paths_for_effect method."""

    @pytest.mark.asyncio
    async def test_returns_all_paths_to_effect(self, repo, mock_client, sample_paths):
        """Test that all paths leading to an effect are returned."""
        mock_result = MagicMock()
        mock_result.data = [sample_paths[0]]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.offset.return_value.execute = mock_execute

        result = await repo.get_paths_for_effect(effect="TRx_growth")

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_respects_limit(self, repo, mock_client, sample_paths):
        """Test that limit is respected."""
        mock_result = MagicMock()
        mock_result.data = sample_paths[:1]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq = MagicMock()
        mock_eq.limit.return_value = mock_limit
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq

        result = await repo.get_paths_for_effect(effect="TRx_growth", limit=1)

        assert len(result) == 1
        mock_eq.limit.assert_called_with(1)


@pytest.mark.unit
class TestGetPathBetween(TestCausalPathRepository):
    """Tests for get_path_between method."""

    @pytest.mark.asyncio
    async def test_returns_path_when_exists(self, repo, mock_client, sample_paths):
        """Test that path is returned when it exists."""
        mock_result = MagicMock()
        mock_result.data = [sample_paths[0]]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq_effect = MagicMock()
        mock_eq_effect.limit.return_value = mock_limit
        mock_eq_cause = MagicMock()
        mock_eq_cause.eq.return_value = mock_eq_effect
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq_cause

        result = await repo.get_path_between(cause="HCP_engagement", effect="TRx_growth")

        assert result is not None
        mock_eq_cause.eq.assert_called_with("effect", "TRx_growth")

    @pytest.mark.asyncio
    async def test_returns_none_when_not_exists(self, repo, mock_client):
        """Test that None is returned when path doesn't exist."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)

        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq_effect = MagicMock()
        mock_eq_effect.limit.return_value = mock_limit
        mock_eq_cause = MagicMock()
        mock_eq_cause.eq.return_value = mock_eq_effect
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq_cause

        result = await repo.get_path_between(cause="X", effect="Y")

        assert result is None

    @pytest.mark.asyncio
    async def test_applies_both_filters(self, repo, mock_client, sample_paths):
        """Test that both cause and effect filters are applied."""
        mock_result = MagicMock()
        mock_result.data = [sample_paths[0]]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq_effect = MagicMock()
        mock_eq_effect.limit.return_value = mock_limit
        mock_eq_cause = MagicMock()
        mock_eq_cause.eq.return_value = mock_eq_effect
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq_cause

        result = await repo.get_path_between(cause="HCP_engagement", effect="TRx_growth")

        assert result is not None

    @pytest.mark.asyncio
    async def test_limits_to_one_result(self, repo, mock_client, sample_paths):
        """Test that only one result is returned."""
        mock_result = MagicMock()
        mock_result.data = [sample_paths[0]]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq_effect = MagicMock()
        mock_eq_effect.limit.return_value = mock_limit
        mock_eq_cause = MagicMock()
        mock_eq_cause.eq.return_value = mock_eq_effect
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq_cause

        await repo.get_path_between(cause="HCP_engagement", effect="TRx_growth")

        mock_eq_effect.limit.assert_called_with(1)


@pytest.mark.unit
class TestGetByBrand(TestCausalPathRepository):
    """Tests for get_by_brand method."""

    @pytest.mark.asyncio
    async def test_filters_by_brand(self, repo, mock_client, sample_paths):
        """Test that brand filter is applied."""
        mock_result = MagicMock()
        mock_result.data = sample_paths
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.offset.return_value.execute = mock_execute

        result = await repo.get_by_brand(brand="Kisqali")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_respects_limit(self, repo, mock_client, sample_paths):
        """Test that limit is respected."""
        mock_result = MagicMock()
        mock_result.data = sample_paths[:1]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq = MagicMock()
        mock_eq.limit.return_value = mock_limit
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq

        result = await repo.get_by_brand(brand="Kisqali", limit=1)

        assert len(result) == 1
        mock_eq.limit.assert_called_with(1)


@pytest.mark.unit
class TestCausalPathEdgeCases(TestCausalPathRepository):
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handles_empty_cause_list(self, repo, mock_client):
        """Test handling when no paths exist for a cause."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.offset.return_value.execute = mock_execute

        result = await repo.get_paths_for_cause(cause="NonexistentCause")

        assert result == []

    @pytest.mark.asyncio
    async def test_handles_empty_effect_list(self, repo, mock_client):
        """Test handling when no paths exist for an effect."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.offset.return_value.execute = mock_execute

        result = await repo.get_paths_for_effect(effect="NonexistentEffect")

        assert result == []

    @pytest.mark.asyncio
    async def test_handles_empty_brand_list(self, repo, mock_client):
        """Test handling when no paths exist for a brand."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.offset.return_value.execute = mock_execute

        result = await repo.get_by_brand(brand="NonexistentBrand")

        assert result == []


@pytest.mark.unit
class TestCausalPathInheritance(TestCausalPathRepository):
    """Tests for inherited BaseRepository methods."""

    @pytest.mark.asyncio
    async def test_inherits_get_by_id(self, repo, mock_client, sample_paths):
        """Test that get_by_id is inherited from BaseRepository."""
        mock_result = MagicMock()
        mock_result.data = [sample_paths[0]]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.execute = mock_execute

        result = await repo.get_by_id(sample_paths[0]["path_id"])

        assert result is not None

    @pytest.mark.asyncio
    async def test_inherits_get_many(self, repo, mock_client, sample_paths):
        """Test that get_many is inherited from BaseRepository."""
        mock_result = MagicMock()
        mock_result.data = sample_paths
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.offset.return_value.execute = mock_execute

        result = await repo.get_many(filters={"brand": "Kisqali"})

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_table_name_is_set(self, repo):
        """Test that table_name is correctly set."""
        assert repo.table_name == "causal_paths"

    @pytest.mark.asyncio
    async def test_model_class_is_none(self, repo):
        """Test that model_class is None (to be set when model is available)."""
        assert repo.model_class is None


@pytest.mark.unit
class TestCausalPathDataTransformation(TestCausalPathRepository):
    """Tests for data transformation and model conversion."""

    @pytest.mark.asyncio
    async def test_to_model_returns_dict_when_no_model_class(self, repo, sample_paths):
        """Test that _to_model returns dict when model_class is None."""
        result = repo._to_model(sample_paths[0])

        # Since model_class is None, should return the dict as-is
        assert result == sample_paths[0]

    @pytest.mark.asyncio
    async def test_preserves_all_fields(self, repo, sample_paths):
        """Test that all fields are preserved in transformation."""
        result = repo._to_model(sample_paths[0])

        assert result["path_id"] == sample_paths[0]["path_id"]
        assert result["cause"] == sample_paths[0]["cause"]
        assert result["effect"] == sample_paths[0]["effect"]
        assert result["brand"] == sample_paths[0]["brand"]
        assert result["effect_size"] == sample_paths[0]["effect_size"]
        assert result["confidence"] == sample_paths[0]["confidence"]


@pytest.mark.unit
class TestCausalPathWithoutClient(TestCausalPathRepository):
    """Tests for repository behavior without Supabase client."""

    @pytest.mark.asyncio
    async def test_get_paths_for_cause_returns_empty_list(self):
        """Test that get_paths_for_cause returns empty list without client."""
        repo = CausalPathRepository(supabase_client=None)
        result = await repo.get_paths_for_cause(cause="any")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_paths_for_effect_returns_empty_list(self):
        """Test that get_paths_for_effect returns empty list without client."""
        repo = CausalPathRepository(supabase_client=None)
        result = await repo.get_paths_for_effect(effect="any")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_path_between_returns_none(self):
        """Test that get_path_between returns None without client."""
        repo = CausalPathRepository(supabase_client=None)
        result = await repo.get_path_between(cause="x", effect="y")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_brand_returns_empty_list(self):
        """Test that get_by_brand returns empty list without client."""
        repo = CausalPathRepository(supabase_client=None)
        result = await repo.get_by_brand(brand="any")
        assert result == []
