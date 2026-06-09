"""
Unit tests for TriggerRepository.

Covers provenance read-path enforcement (Shard 07, R4): every explicit-query
read default-excludes ``is_synthetic`` rows and opts in via ``include_synthetic``.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.repositories.trigger import TriggerRepository


@pytest.mark.unit
class TestTriggerRepository:
    """Base fixtures for TriggerRepository tests."""

    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.fixture
    def repo(self, mock_client):
        return TriggerRepository(supabase_client=mock_client)


@pytest.mark.unit
class TestProvenanceDefaultExclude(TestTriggerRepository):
    """Provenance read-path enforcement (R4)."""

    def test_repo_declares_provenance(self):
        assert TriggerRepository.HAS_PROVENANCE is True

    @pytest.mark.asyncio
    async def test_get_recent_triggers_excludes_synthetic_by_default(self, repo, mock_client):
        result = MagicMock()
        result.data = []
        # chain: select().gte().eq(is_synthetic).order().limit().execute()
        chain = mock_client.table.return_value.select.return_value.gte.return_value
        chain.eq.return_value.order.return_value.limit.return_value.execute = AsyncMock(
            return_value=result
        )

        await repo.get_recent_triggers()

        chain.eq.assert_called_with("is_synthetic", False)

    @pytest.mark.asyncio
    async def test_get_recent_triggers_includes_synthetic_when_opted_in(self, repo, mock_client):
        result = MagicMock()
        result.data = []
        chain = mock_client.table.return_value.select.return_value.gte.return_value
        chain.order.return_value.limit.return_value.execute = AsyncMock(return_value=result)

        await repo.get_recent_triggers(include_synthetic=True)

        chain.eq.assert_not_called()
        chain.order.assert_called()

    @pytest.mark.asyncio
    async def test_get_by_patient_excludes_synthetic_by_default(self, repo, mock_client):
        result = MagicMock()
        result.data = []
        # chain: select().eq(patient_id).eq(is_synthetic).order().limit().execute()
        chain = mock_client.table.return_value.select.return_value.eq.return_value
        chain.eq.return_value.order.return_value.limit.return_value.execute = AsyncMock(
            return_value=result
        )

        await repo.get_by_patient("patient-1")

        chain.eq.assert_called_with("is_synthetic", False)

    @pytest.mark.asyncio
    async def test_get_by_patient_includes_synthetic_when_opted_in(self, repo, mock_client):
        result = MagicMock()
        result.data = []
        chain = mock_client.table.return_value.select.return_value.eq.return_value
        chain.order.return_value.limit.return_value.execute = AsyncMock(return_value=result)

        await repo.get_by_patient("patient-1", include_synthetic=True)

        chain.eq.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_change_fail_rate_excludes_synthetic_by_default(self, repo, mock_client):
        result = MagicMock()
        result.data = []
        # chain: select().gte().not_.is_().eq(is_synthetic).limit().execute()
        chain = mock_client.table.return_value.select.return_value.gte.return_value.not_.is_.return_value
        chain.eq.return_value.limit.return_value.execute = AsyncMock(return_value=result)

        await repo.get_change_fail_rate()

        chain.eq.assert_called_with("is_synthetic", False)

    @pytest.mark.asyncio
    async def test_get_change_fail_rate_includes_synthetic_when_opted_in(self, repo, mock_client):
        result = MagicMock()
        result.data = []
        chain = mock_client.table.return_value.select.return_value.gte.return_value.not_.is_.return_value
        chain.limit.return_value.execute = AsyncMock(return_value=result)

        await repo.get_change_fail_rate(include_synthetic=True)

        chain.eq.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_trigger_acceptance_rate_excludes_synthetic_by_default(
        self, repo, mock_client
    ):
        result = MagicMock()
        result.data = []
        # chain: select().gte().eq(delivery_status).eq(is_synthetic).limit().execute()
        chain = mock_client.table.return_value.select.return_value.gte.return_value.eq.return_value
        chain.eq.return_value.limit.return_value.execute = AsyncMock(return_value=result)

        await repo.get_trigger_acceptance_rate()

        chain.eq.assert_called_with("is_synthetic", False)

    @pytest.mark.asyncio
    async def test_get_triggers_in_range_excludes_synthetic_by_default(self, repo, mock_client):
        result = MagicMock()
        result.data = []
        # chain: select().gte().lte().eq(is_synthetic).order().limit().execute()
        chain = mock_client.table.return_value.select.return_value.gte.return_value.lte.return_value
        chain.eq.return_value.order.return_value.limit.return_value.execute = AsyncMock(
            return_value=result
        )

        await repo.get_triggers_in_range(
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 31, tzinfo=timezone.utc),
        )

        chain.eq.assert_called_with("is_synthetic", False)

    @pytest.mark.asyncio
    async def test_get_triggers_in_range_includes_synthetic_when_opted_in(self, repo, mock_client):
        result = MagicMock()
        result.data = []
        chain = mock_client.table.return_value.select.return_value.gte.return_value.lte.return_value
        chain.order.return_value.limit.return_value.execute = AsyncMock(return_value=result)

        await repo.get_triggers_in_range(
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 31, tzinfo=timezone.utc),
            include_synthetic=True,
        )

        chain.eq.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_by_hcp_excludes_synthetic_via_get_many(self, repo, mock_client):
        """Dict-filter path inherits provenance via base.get_many + HAS_PROVENANCE."""
        result = MagicMock()
        result.data = []
        # chain: select().eq(hcp_id).eq(is_synthetic).limit().offset().execute()
        chain = mock_client.table.return_value.select.return_value.eq.return_value
        chain.eq.return_value.limit.return_value.offset.return_value.execute = AsyncMock(
            return_value=result
        )

        await repo.get_by_hcp("hcp-1")

        chain.eq.assert_called_with("is_synthetic", False)
