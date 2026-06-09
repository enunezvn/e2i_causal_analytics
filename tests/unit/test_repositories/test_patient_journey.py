"""
Unit tests for PatientJourneyRepository.

Covers provenance read-path enforcement (Shard 07, R5): every explicit-query
read default-excludes ``is_synthetic`` rows and opts in via ``include_synthetic``.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.repositories.patient_journey import PatientJourneyRepository


@pytest.mark.unit
class TestPatientJourneyRepository:
    """Base fixtures for PatientJourneyRepository tests."""

    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.fixture
    def repo(self, mock_client):
        return PatientJourneyRepository(supabase_client=mock_client)


@pytest.mark.unit
class TestProvenanceDefaultExclude(TestPatientJourneyRepository):
    """Provenance read-path enforcement (R5)."""

    def test_repo_declares_provenance(self):
        assert PatientJourneyRepository.HAS_PROVENANCE is True

    @pytest.mark.asyncio
    async def test_get_data_freshness_excludes_synthetic_by_default(self, repo, mock_client):
        result = MagicMock()
        result.data = []
        # chain: select().eq(brand).not_.is_().eq(is_synthetic).limit().execute()
        chain = (
            mock_client.table.return_value.select.return_value.eq.return_value.not_.is_.return_value
        )
        chain.eq.return_value.limit.return_value.execute = AsyncMock(return_value=result)

        await repo.get_data_freshness("Kisqali")

        chain.eq.assert_called_with("is_synthetic", False)

    @pytest.mark.asyncio
    async def test_get_data_freshness_includes_synthetic_when_opted_in(self, repo, mock_client):
        result = MagicMock()
        result.data = []
        chain = (
            mock_client.table.return_value.select.return_value.eq.return_value.not_.is_.return_value
        )
        chain.limit.return_value.execute = AsyncMock(return_value=result)

        await repo.get_data_freshness("Kisqali", include_synthetic=True)

        chain.eq.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_freshness_by_source_excludes_synthetic_by_default(self, repo, mock_client):
        result = MagicMock()
        result.data = []
        # no brand -> chain: select().not_.is_().eq(is_synthetic).limit().execute()
        chain = mock_client.table.return_value.select.return_value.not_.is_.return_value
        chain.eq.return_value.limit.return_value.execute = AsyncMock(return_value=result)

        await repo.get_freshness_by_source()

        chain.eq.assert_called_with("is_synthetic", False)

    @pytest.mark.asyncio
    async def test_get_journey_stage_distribution_excludes_synthetic_by_default(
        self, repo, mock_client
    ):
        result = MagicMock()
        result.data = []
        # chain: select().eq(brand).eq(is_synthetic).limit().execute()
        chain = mock_client.table.return_value.select.return_value.eq.return_value
        chain.eq.return_value.limit.return_value.execute = AsyncMock(return_value=result)

        await repo.get_journey_stage_distribution("Kisqali")

        chain.eq.assert_called_with("is_synthetic", False)

    @pytest.mark.asyncio
    async def test_get_journey_stage_distribution_includes_synthetic_when_opted_in(
        self, repo, mock_client
    ):
        result = MagicMock()
        result.data = []
        chain = mock_client.table.return_value.select.return_value.eq.return_value
        chain.limit.return_value.execute = AsyncMock(return_value=result)

        await repo.get_journey_stage_distribution("Kisqali", include_synthetic=True)

        chain.eq.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_source_stacking_metrics_excludes_synthetic_by_default(
        self, repo, mock_client
    ):
        result = MagicMock()
        result.data = []
        # chain: select().eq(brand).eq(is_synthetic).limit().execute()
        chain = mock_client.table.return_value.select.return_value.eq.return_value
        chain.eq.return_value.limit.return_value.execute = AsyncMock(return_value=result)

        await repo.get_source_stacking_metrics("Kisqali")

        chain.eq.assert_called_with("is_synthetic", False)

    @pytest.mark.asyncio
    async def test_get_by_brand_excludes_synthetic_via_get_many(self, repo, mock_client):
        """Dict-filter path inherits provenance via base.get_many + HAS_PROVENANCE."""
        result = MagicMock()
        result.data = []
        # chain: select().eq(brand).eq(is_synthetic).limit().offset().execute()
        chain = mock_client.table.return_value.select.return_value.eq.return_value
        chain.eq.return_value.limit.return_value.offset.return_value.execute = AsyncMock(
            return_value=result
        )

        await repo.get_by_brand("Kisqali")

        chain.eq.assert_called_with("is_synthetic", False)
