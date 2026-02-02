"""
Unit tests for BusinessMetricRepository.

Tests KPI queries, time series retrieval, and achievement summaries.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.repositories.business_metric import BusinessMetricRepository


@pytest.mark.unit
class TestBusinessMetricRepository:
    """Tests for BusinessMetricRepository."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Supabase client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def repo(self, mock_client):
        """Create repository with mock client."""
        return BusinessMetricRepository(supabase_client=mock_client)

    @pytest.fixture
    def sample_metrics(self):
        """Sample business metrics data."""
        return [
            {
                "metric_id": str(uuid4()),
                "metric_date": "2025-01-15",
                "metric_name": "TRx",
                "brand": "Kisqali",
                "region": "US",
                "value": 1200.0,
                "target": 1000.0,
                "achievement_rate": 1.2,
                "roi": 2.5,
            },
            {
                "metric_id": str(uuid4()),
                "metric_date": "2025-01-14",
                "metric_name": "TRx",
                "brand": "Kisqali",
                "region": "US",
                "value": 1100.0,
                "target": 1000.0,
                "achievement_rate": 1.1,
                "roi": 2.3,
            },
        ]


@pytest.mark.unit
class TestGetByKpi(TestBusinessMetricRepository):
    """Tests for get_by_kpi method."""

    @pytest.mark.asyncio
    async def test_returns_metrics_for_kpi(self, repo, mock_client, sample_metrics):
        """Test that metrics for a specific KPI are returned."""
        mock_result = MagicMock()
        mock_result.data = sample_metrics
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.offset.return_value.execute = mock_execute

        result = await repo.get_by_kpi(kpi_name="TRx")

        assert len(result) == 2
        mock_client.table.assert_called_with("business_metrics")

    @pytest.mark.asyncio
    async def test_filters_by_brand(self, repo, mock_client, sample_metrics):
        """Test that brand filter is applied."""
        mock_result = MagicMock()
        mock_result.data = sample_metrics
        mock_execute = AsyncMock(return_value=mock_result)

        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq_brand = MagicMock()
        mock_eq_brand.limit.return_value = mock_limit
        mock_eq_kpi = MagicMock()
        mock_eq_kpi.eq.return_value = mock_eq_brand
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq_kpi

        result = await repo.get_by_kpi(kpi_name="TRx", brand="Kisqali")

        assert len(result) == 2
        mock_eq_kpi.eq.assert_called_with("brand", "Kisqali")

    @pytest.mark.asyncio
    async def test_respects_limit(self, repo, mock_client, sample_metrics):
        """Test that limit is respected."""
        mock_result = MagicMock()
        mock_result.data = sample_metrics[:1]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq = MagicMock()
        mock_eq.limit.return_value = mock_limit
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq

        result = await repo.get_by_kpi(kpi_name="TRx", limit=1)

        assert len(result) == 1
        mock_eq.limit.assert_called_with(1)


@pytest.mark.unit
class TestGetTimeSeries(TestBusinessMetricRepository):
    """Tests for get_time_series method."""

    @pytest.mark.asyncio
    async def test_returns_time_ordered_data(self, repo, mock_client, sample_metrics):
        """Test that time series data is returned in order."""
        mock_result = MagicMock()
        mock_result.data = sample_metrics
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_time_series(kpi_name="TRx", brand="Kisqali")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_applies_date_filters(self, repo, mock_client, sample_metrics):
        """Test that start and end date filters are applied."""
        mock_result = MagicMock()
        mock_result.data = sample_metrics
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock()
        mock_limit.execute = mock_execute
        mock_order = MagicMock()
        mock_order.limit.return_value = mock_limit
        mock_lte = MagicMock()
        mock_lte.order.return_value = mock_order
        mock_gte = MagicMock()
        mock_gte.lte.return_value = mock_lte
        mock_eq_brand = MagicMock()
        mock_eq_brand.gte.return_value = mock_gte
        mock_eq_kpi = MagicMock()
        mock_eq_kpi.eq.return_value = mock_eq_brand
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq_kpi

        result = await repo.get_time_series(
            kpi_name="TRx",
            brand="Kisqali",
            start_date="2025-01-01",
            end_date="2025-01-31",
        )

        assert len(result) == 2
        mock_eq_brand.gte.assert_called_with("metric_date", "2025-01-01")
        mock_gte.lte.assert_called_with("metric_date", "2025-01-31")

    @pytest.mark.asyncio
    async def test_orders_by_date_ascending(self, repo, mock_client, sample_metrics):
        """Test that results are ordered by date ascending."""
        mock_result = MagicMock()
        mock_result.data = sample_metrics
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock()
        mock_limit.execute = mock_execute
        mock_order = MagicMock()
        mock_order.limit.return_value = mock_limit
        mock_eq_brand = MagicMock()
        mock_eq_brand.order.return_value = mock_order
        mock_eq_kpi = MagicMock()
        mock_eq_kpi.eq.return_value = mock_eq_brand
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq_kpi

        result = await repo.get_time_series(kpi_name="TRx", brand="Kisqali")

        assert len(result) == 2
        mock_eq_brand.order.assert_called_with("metric_date", desc=False)

    @pytest.mark.asyncio
    async def test_returns_empty_list_without_client(self):
        """Test that empty list is returned when client is None."""
        repo = BusinessMetricRepository(supabase_client=None)
        result = await repo.get_time_series(kpi_name="TRx", brand="Kisqali")
        assert result == []


@pytest.mark.unit
class TestGetLatestSnapshot(TestBusinessMetricRepository):
    """Tests for get_latest_snapshot method."""

    @pytest.fixture
    def snapshot_metrics(self):
        """Sample metrics for snapshot testing."""
        return [
            {
                "metric_name": "TRx",
                "metric_date": "2025-01-15",
                "value": 1200.0,
                "target": 1000.0,
                "achievement_rate": 1.2,
                "roi": 2.5,
            },
            {
                "metric_name": "TRx",
                "metric_date": "2025-01-14",
                "value": 1100.0,
                "target": 1000.0,
                "achievement_rate": 1.1,
                "roi": 2.3,
            },
            {
                "metric_name": "NRx",
                "metric_date": "2025-01-15",
                "value": 300.0,
                "target": 250.0,
                "achievement_rate": 1.2,
                "roi": 3.0,
            },
        ]

    @pytest.mark.asyncio
    async def test_returns_latest_value_per_metric(self, repo, mock_client, snapshot_metrics):
        """Test that latest value for each metric is returned."""
        mock_result = MagicMock()
        mock_result.data = snapshot_metrics
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.order.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_latest_snapshot(brand="Kisqali")

        assert len(result) == 2  # TRx and NRx
        assert "TRx" in result
        assert "NRx" in result
        # Latest TRx should be from 2025-01-15
        assert result["TRx"]["value"] == 1200.0
        assert result["TRx"]["date"] == "2025-01-15"

    @pytest.mark.asyncio
    async def test_deduplicates_metrics(self, repo, mock_client, snapshot_metrics):
        """Test that duplicate metrics are deduplicated to latest."""
        mock_result = MagicMock()
        mock_result.data = snapshot_metrics
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.order.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_latest_snapshot(brand="Kisqali")

        # Should have only one entry per metric_name
        metric_names = list(result.keys())
        assert len(metric_names) == len(set(metric_names))

    @pytest.mark.asyncio
    async def test_includes_all_fields(self, repo, mock_client, snapshot_metrics):
        """Test that all required fields are included."""
        mock_result = MagicMock()
        mock_result.data = snapshot_metrics
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.order.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_latest_snapshot(brand="Kisqali")

        assert "TRx" in result
        snapshot = result["TRx"]
        assert "value" in snapshot
        assert "target" in snapshot
        assert "achievement_rate" in snapshot
        assert "roi" in snapshot
        assert "date" in snapshot

    @pytest.mark.asyncio
    async def test_returns_empty_dict_without_client(self):
        """Test that empty dict is returned when client is None."""
        repo = BusinessMetricRepository(supabase_client=None)
        result = await repo.get_latest_snapshot(brand="Kisqali")
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_dict_when_no_data(self, repo, mock_client):
        """Test that empty dict is returned when no data exists."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.order.return_value.limit.return_value.execute = mock_execute

        result = await repo.get_latest_snapshot(brand="NonexistentBrand")

        assert result == {}


@pytest.mark.unit
class TestGetByRegion(TestBusinessMetricRepository):
    """Tests for get_by_region method."""

    @pytest.mark.asyncio
    async def test_filters_by_region(self, repo, mock_client, sample_metrics):
        """Test that region filter is applied."""
        mock_result = MagicMock()
        mock_result.data = sample_metrics
        mock_execute = AsyncMock(return_value=mock_result)
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.offset.return_value.execute = mock_execute

        result = await repo.get_by_region(region="US")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_filters_by_region_and_brand(self, repo, mock_client, sample_metrics):
        """Test that both region and brand filters are applied."""
        mock_result = MagicMock()
        mock_result.data = sample_metrics
        mock_execute = AsyncMock(return_value=mock_result)

        mock_offset = MagicMock()
        mock_offset.execute = mock_execute
        mock_limit = MagicMock()
        mock_limit.offset.return_value = mock_offset
        mock_eq_brand = MagicMock()
        mock_eq_brand.limit.return_value = mock_limit
        mock_eq_region = MagicMock()
        mock_eq_region.eq.return_value = mock_eq_brand
        mock_client.table.return_value.select.return_value.eq.return_value = mock_eq_region

        result = await repo.get_by_region(region="US", brand="Kisqali")

        assert len(result) == 2
        mock_eq_region.eq.assert_called_with("brand", "Kisqali")


@pytest.mark.unit
class TestGetAchievementSummary(TestBusinessMetricRepository):
    """Tests for get_achievement_summary method."""

    @pytest.fixture
    def achievement_snapshot(self):
        """Sample snapshot for achievement testing."""
        return {
            "metric1": {"achievement_rate": 1.2, "value": 100},
            "metric2": {"achievement_rate": 0.9, "value": 200},
            "metric3": {"achievement_rate": 1.1, "value": 150},
            "metric4": {"achievement_rate": 0.8, "value": 120},
        }

    @pytest.mark.asyncio
    async def test_calculates_average_achievement(self, repo, mock_client, achievement_snapshot):
        """Test that average achievement rate is calculated correctly."""
        # Mock get_latest_snapshot
        repo.get_latest_snapshot = AsyncMock(return_value=achievement_snapshot)

        result = await repo.get_achievement_summary(brand="Kisqali")

        expected_avg = (1.2 + 0.9 + 1.1 + 0.8) / 4
        assert result["avg_achievement"] == expected_avg

    @pytest.mark.asyncio
    async def test_counts_metrics_at_target(self, repo, mock_client, achievement_snapshot):
        """Test that metrics at or above target are counted."""
        repo.get_latest_snapshot = AsyncMock(return_value=achievement_snapshot)

        result = await repo.get_achievement_summary(brand="Kisqali")

        # Metrics at or above 1.0: metric1 (1.2), metric3 (1.1)
        assert result["metrics_at_target"] == 2

    @pytest.mark.asyncio
    async def test_counts_metrics_below_target(self, repo, mock_client, achievement_snapshot):
        """Test that metrics below target are counted."""
        repo.get_latest_snapshot = AsyncMock(return_value=achievement_snapshot)

        result = await repo.get_achievement_summary(brand="Kisqali")

        # Metrics below 1.0: metric2 (0.9), metric4 (0.8)
        assert result["metrics_below_target"] == 2

    @pytest.mark.asyncio
    async def test_counts_total_metrics(self, repo, mock_client, achievement_snapshot):
        """Test that total metric count is correct."""
        repo.get_latest_snapshot = AsyncMock(return_value=achievement_snapshot)

        result = await repo.get_achievement_summary(brand="Kisqali")

        assert result["total_metrics"] == 4

    @pytest.mark.asyncio
    async def test_handles_empty_snapshot(self, repo, mock_client):
        """Test handling when no metrics exist."""
        repo.get_latest_snapshot = AsyncMock(return_value={})

        result = await repo.get_achievement_summary(brand="Kisqali")

        assert result["avg_achievement"] == 0
        assert result["metrics_at_target"] == 0
        assert result["metrics_below_target"] == 0
        assert result["total_metrics"] == 0

    @pytest.mark.asyncio
    async def test_returns_default_without_client(self):
        """Test that default values are returned when client is None."""
        repo = BusinessMetricRepository(supabase_client=None)
        result = await repo.get_achievement_summary(brand="Kisqali")

        assert result["avg_achievement"] == 0
        assert result["metrics_at_target"] == 0
        assert result["metrics_below_target"] == 0
        assert result["total_metrics"] == 0

    @pytest.mark.asyncio
    async def test_ignores_null_achievement_rates(self, repo, mock_client):
        """Test that metrics with null achievement rates are ignored."""
        snapshot_with_nulls = {
            "metric1": {"achievement_rate": 1.2, "value": 100},
            "metric2": {"achievement_rate": None, "value": 200},
            "metric3": {"achievement_rate": 0.9, "value": 150},
        }
        repo.get_latest_snapshot = AsyncMock(return_value=snapshot_with_nulls)

        result = await repo.get_achievement_summary(brand="Kisqali")

        # Should only use metric1 and metric3
        expected_avg = (1.2 + 0.9) / 2
        assert result["avg_achievement"] == expected_avg


@pytest.mark.unit
class TestGetRoiSummary(TestBusinessMetricRepository):
    """Tests for get_roi_summary method."""

    @pytest.fixture
    def roi_snapshot(self):
        """Sample snapshot for ROI testing."""
        return {
            "metric1": {"roi": 2.5, "value": 100},
            "metric2": {"roi": 3.0, "value": 200},
            "metric3": {"roi": 1.5, "value": 150},
            "metric4": {"roi": 2.0, "value": 120},
        }

    @pytest.mark.asyncio
    async def test_calculates_average_roi(self, repo, mock_client, roi_snapshot):
        """Test that average ROI is calculated correctly."""
        repo.get_latest_snapshot = AsyncMock(return_value=roi_snapshot)

        result = await repo.get_roi_summary(brand="Kisqali")

        expected_avg = (2.5 + 3.0 + 1.5 + 2.0) / 4
        assert result["avg_roi"] == expected_avg

    @pytest.mark.asyncio
    async def test_finds_max_roi(self, repo, mock_client, roi_snapshot):
        """Test that maximum ROI is identified."""
        repo.get_latest_snapshot = AsyncMock(return_value=roi_snapshot)

        result = await repo.get_roi_summary(brand="Kisqali")

        assert result["max_roi"] == 3.0

    @pytest.mark.asyncio
    async def test_finds_min_roi(self, repo, mock_client, roi_snapshot):
        """Test that minimum ROI is identified."""
        repo.get_latest_snapshot = AsyncMock(return_value=roi_snapshot)

        result = await repo.get_roi_summary(brand="Kisqali")

        assert result["min_roi"] == 1.5

    @pytest.mark.asyncio
    async def test_sums_total_value(self, repo, mock_client, roi_snapshot):
        """Test that total value is summed correctly."""
        repo.get_latest_snapshot = AsyncMock(return_value=roi_snapshot)

        result = await repo.get_roi_summary(brand="Kisqali")

        expected_total = 100 + 200 + 150 + 120
        assert result["total_value"] == expected_total

    @pytest.mark.asyncio
    async def test_handles_empty_snapshot(self, repo, mock_client):
        """Test handling when no metrics exist."""
        repo.get_latest_snapshot = AsyncMock(return_value={})

        result = await repo.get_roi_summary(brand="Kisqali")

        assert result["avg_roi"] == 0
        assert result["max_roi"] == 0
        assert result["min_roi"] == 0
        assert result["total_value"] == 0

    @pytest.mark.asyncio
    async def test_returns_default_without_client(self):
        """Test that default values are returned when client is None."""
        repo = BusinessMetricRepository(supabase_client=None)
        result = await repo.get_roi_summary(brand="Kisqali")

        assert result["avg_roi"] == 0
        assert result["max_roi"] == 0
        assert result["min_roi"] == 0
        assert result["total_value"] == 0

    @pytest.mark.asyncio
    async def test_ignores_null_values(self, repo, mock_client):
        """Test that metrics with null ROI or value are ignored."""
        snapshot_with_nulls = {
            "metric1": {"roi": 2.5, "value": 100},
            "metric2": {"roi": None, "value": 200},
            "metric3": {"roi": 3.0, "value": None},
            "metric4": {"roi": 1.5, "value": 150},
        }
        repo.get_latest_snapshot = AsyncMock(return_value=snapshot_with_nulls)

        result = await repo.get_roi_summary(brand="Kisqali")

        # ROI list: metric1 (2.5), metric3 (3.0), metric4 (1.5) - excludes metric2 (roi is None)
        expected_avg_roi = (2.5 + 3.0 + 1.5) / 3
        assert (
            abs(result["avg_roi"] - expected_avg_roi) < 0.01
        )  # Allow small floating point differences

        # Value list: metric1 (100), metric2 (200), metric4 (150) - excludes metric3 (value is None)
        # The function filters ROI and value independently
        expected_total_value = 100 + 200 + 150
        assert result["total_value"] == expected_total_value
