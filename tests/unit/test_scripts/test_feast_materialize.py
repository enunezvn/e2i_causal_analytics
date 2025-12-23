"""Tests for Feast Feature Materialization Script.

Tests cover:
- MaterializationJob initialization
- Full materialization mode
- Incremental materialization mode
- Feature freshness checking
- Dry run validation
- Error handling
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.feast_materialize import MaterializationJob


class TestMaterializationJobInit:
    """Test MaterializationJob initialization."""

    def test_job_creation_defaults(self):
        """Test job can be created with defaults."""
        job = MaterializationJob()

        assert job.feast_client is None
        assert job.config is not None
        assert job._initialized is False

    def test_job_creation_with_client(self):
        """Test job with provided Feast client."""
        mock_client = MagicMock()
        job = MaterializationJob(feast_client=mock_client)

        assert job.feast_client is mock_client


class TestMaterializationJobInitialize:
    """Test MaterializationJob initialization."""

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()

        job = MaterializationJob(feast_client=mock_client)
        result = await job.initialize()

        assert result is True
        assert job._initialized is True
        mock_client.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        """Test that initialization is skipped if already done."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()

        job = MaterializationJob(feast_client=mock_client)
        job._initialized = True

        result = await job.initialize()

        assert result is True
        mock_client.initialize.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test initialization failure handling."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock(side_effect=Exception("Init failed"))

        job = MaterializationJob(feast_client=mock_client)
        result = await job.initialize()

        assert result is False
        assert job._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_creates_client_if_none(self):
        """Test that initialization creates client if none provided."""
        job = MaterializationJob()

        with patch(
            "scripts.feast_materialize.get_feast_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.initialize = AsyncMock()
            mock_get_client.return_value = mock_client

            result = await job.initialize()

            assert result is True
            mock_get_client.assert_called_once()


class TestFullMaterialization:
    """Test full materialization mode."""

    @pytest.mark.asyncio
    async def test_full_materialization_success(self):
        """Test successful full materialization."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()
        mock_client.materialize = AsyncMock(
            return_value={
                "status": "completed",
                "duration_seconds": 5.0,
            }
        )

        job = MaterializationJob(feast_client=mock_client)

        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)

        result = await job.run_full_materialization(
            start_date=start_date,
            end_date=end_date,
        )

        assert result["status"] == "completed"
        assert result["mode"] == "full"
        assert "start_date" in result
        assert "end_date" in result
        mock_client.materialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_materialization_with_feature_views(self):
        """Test full materialization with specific feature views."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()
        mock_client.materialize = AsyncMock(
            return_value={"status": "completed", "duration_seconds": 3.0}
        )

        job = MaterializationJob(feast_client=mock_client)

        result = await job.run_full_materialization(
            start_date=datetime.now(timezone.utc) - timedelta(days=1),
            end_date=datetime.now(timezone.utc),
            feature_views=["hcp_conversion_features", "patient_journey_features"],
        )

        assert result["status"] == "completed"
        call_args = mock_client.materialize.call_args
        assert call_args.kwargs["feature_views"] == [
            "hcp_conversion_features",
            "patient_journey_features",
        ]

    @pytest.mark.asyncio
    async def test_full_materialization_init_failure(self):
        """Test full materialization when init fails."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock(side_effect=Exception("Init failed"))

        job = MaterializationJob(feast_client=mock_client)

        result = await job.run_full_materialization(
            start_date=datetime.now(timezone.utc) - timedelta(days=1),
            end_date=datetime.now(timezone.utc),
        )

        assert result["status"] == "failed"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_full_materialization_error(self):
        """Test full materialization error handling."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()
        mock_client.materialize = AsyncMock(
            side_effect=Exception("Materialization error")
        )

        job = MaterializationJob(feast_client=mock_client)

        result = await job.run_full_materialization(
            start_date=datetime.now(timezone.utc) - timedelta(days=1),
            end_date=datetime.now(timezone.utc),
        )

        assert result["status"] == "failed"
        assert "Materialization error" in result["error"]
        assert result["mode"] == "full"


class TestIncrementalMaterialization:
    """Test incremental materialization mode."""

    @pytest.mark.asyncio
    async def test_incremental_materialization_success(self):
        """Test successful incremental materialization."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()
        mock_client.materialize_incremental = AsyncMock(
            return_value={
                "status": "completed",
                "duration_seconds": 2.0,
                "incremental": True,
            }
        )

        job = MaterializationJob(feast_client=mock_client)

        result = await job.run_incremental_materialization()

        assert result["status"] == "completed"
        assert result["mode"] == "incremental"
        assert "end_date" in result
        mock_client.materialize_incremental.assert_called_once()

    @pytest.mark.asyncio
    async def test_incremental_materialization_with_end_date(self):
        """Test incremental materialization with specific end date."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()
        mock_client.materialize_incremental = AsyncMock(
            return_value={"status": "completed", "duration_seconds": 1.5}
        )

        job = MaterializationJob(feast_client=mock_client)
        end_date = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        result = await job.run_incremental_materialization(end_date=end_date)

        assert result["status"] == "completed"
        call_args = mock_client.materialize_incremental.call_args
        assert call_args.kwargs["end_date"] == end_date

    @pytest.mark.asyncio
    async def test_incremental_materialization_with_feature_views(self):
        """Test incremental materialization with specific feature views."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()
        mock_client.materialize_incremental = AsyncMock(
            return_value={"status": "completed"}
        )

        job = MaterializationJob(feast_client=mock_client)

        result = await job.run_incremental_materialization(
            feature_views=["hcp_conversion_features"]
        )

        assert result["status"] == "completed"
        call_args = mock_client.materialize_incremental.call_args
        assert call_args.kwargs["feature_views"] == ["hcp_conversion_features"]

    @pytest.mark.asyncio
    async def test_incremental_materialization_error(self):
        """Test incremental materialization error handling."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()
        mock_client.materialize_incremental = AsyncMock(
            side_effect=Exception("Incremental error")
        )

        job = MaterializationJob(feast_client=mock_client)

        result = await job.run_incremental_materialization()

        assert result["status"] == "failed"
        assert "Incremental error" in result["error"]
        assert result["mode"] == "incremental"


class TestDryRun:
    """Test dry run validation."""

    @pytest.mark.asyncio
    async def test_dry_run_full_materialization(self):
        """Test dry run for full materialization."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()
        mock_client.list_feature_views = AsyncMock(
            return_value=[
                {"name": "hcp_conversion_features"},
                {"name": "patient_journey_features"},
            ]
        )

        job = MaterializationJob(feast_client=mock_client)

        result = await job.run_full_materialization(
            start_date=datetime.now(timezone.utc) - timedelta(days=1),
            end_date=datetime.now(timezone.utc),
            dry_run=True,
        )

        assert result["status"] == "validated"
        assert result["dry_run"] is True
        assert result["total_views"] == 2
        # Materialize should NOT be called in dry run
        mock_client.materialize.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_incremental_materialization(self):
        """Test dry run for incremental materialization."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()
        mock_client.list_feature_views = AsyncMock(
            return_value=[{"name": "hcp_conversion_features"}]
        )

        job = MaterializationJob(feast_client=mock_client)

        result = await job.run_incremental_materialization(dry_run=True)

        assert result["status"] == "validated"
        assert result["dry_run"] is True
        mock_client.materialize_incremental.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_with_invalid_feature_views(self):
        """Test dry run catches invalid feature views."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()
        mock_client.list_feature_views = AsyncMock(
            return_value=[{"name": "hcp_conversion_features"}]
        )

        job = MaterializationJob(feast_client=mock_client)

        result = await job.run_full_materialization(
            start_date=datetime.now(timezone.utc) - timedelta(days=1),
            end_date=datetime.now(timezone.utc),
            feature_views=["nonexistent_feature_view"],
            dry_run=True,
        )

        assert result["status"] == "failed"
        assert "not found" in result["error"]


class TestFeatureFreshness:
    """Test feature freshness checking."""

    @pytest.mark.asyncio
    async def test_check_freshness_all_fresh(self):
        """Test freshness check when all features are fresh."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()
        mock_client.list_feature_views = AsyncMock(
            return_value=[{"name": "hcp_conversion_features"}]
        )

        # Feature updated 1 hour ago
        from src.feature_store.feast_client import FeatureStatistics

        stats = FeatureStatistics(
            feature_view="hcp_conversion_features",
            feature_name="*",
            count=1000,
            null_count=0,
            last_updated=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        mock_client.get_feature_statistics = AsyncMock(return_value=stats)

        job = MaterializationJob(feast_client=mock_client)

        result = await job.check_feature_freshness(
            max_staleness_hours=24.0,
        )

        assert result["status"] == "completed"
        assert result["fresh"] is True
        assert len(result["stale_features"]) == 0
        assert len(result["fresh_features"]) == 1

    @pytest.mark.asyncio
    async def test_check_freshness_with_stale_features(self):
        """Test freshness check when features are stale."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()
        mock_client.list_feature_views = AsyncMock(
            return_value=[{"name": "hcp_conversion_features"}]
        )

        # Feature updated 48 hours ago
        from src.feature_store.feast_client import FeatureStatistics

        stats = FeatureStatistics(
            feature_view="hcp_conversion_features",
            feature_name="*",
            count=1000,
            null_count=0,
            last_updated=datetime.now(timezone.utc) - timedelta(hours=48),
        )
        mock_client.get_feature_statistics = AsyncMock(return_value=stats)

        job = MaterializationJob(feast_client=mock_client)

        result = await job.check_feature_freshness(
            max_staleness_hours=24.0,
        )

        assert result["status"] == "completed"
        assert result["fresh"] is False
        assert len(result["stale_features"]) == 1
        assert result["stale_features"][0]["feature_view"] == "hcp_conversion_features"

    @pytest.mark.asyncio
    async def test_check_freshness_specific_feature_views(self):
        """Test freshness check for specific feature views."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()

        from src.feature_store.feast_client import FeatureStatistics

        stats = FeatureStatistics(
            feature_view="hcp_conversion_features",
            feature_name="*",
            count=500,
            null_count=0,
            last_updated=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        mock_client.get_feature_statistics = AsyncMock(return_value=stats)

        job = MaterializationJob(feast_client=mock_client)

        result = await job.check_feature_freshness(
            feature_views=["hcp_conversion_features"],
            max_staleness_hours=24.0,
        )

        assert result["status"] == "completed"
        assert result["fresh"] is True
        # list_feature_views should NOT be called when views are specified
        mock_client.list_feature_views.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_freshness_error_handling(self):
        """Test freshness check error handling."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()
        mock_client.list_feature_views = AsyncMock(
            return_value=[{"name": "hcp_conversion_features"}]
        )
        mock_client.get_feature_statistics = AsyncMock(
            side_effect=Exception("Stats error")
        )

        job = MaterializationJob(feast_client=mock_client)

        result = await job.check_feature_freshness()

        assert result["status"] == "completed"
        # Should have errors list
        assert len(result["errors"]) == 1
        assert "Stats error" in result["errors"][0]["error"]

    @pytest.mark.asyncio
    async def test_check_freshness_no_stats_available(self):
        """Test freshness check when no stats available."""
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()
        mock_client.list_feature_views = AsyncMock(
            return_value=[{"name": "hcp_conversion_features"}]
        )
        mock_client.get_feature_statistics = AsyncMock(return_value=None)

        job = MaterializationJob(feast_client=mock_client)

        result = await job.check_feature_freshness()

        assert result["status"] == "completed"
        assert len(result["errors"]) == 1
        assert "No statistics available" in result["errors"][0]["error"]


class TestJobLifecycle:
    """Test job lifecycle operations."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing the job."""
        mock_client = MagicMock()
        mock_client.close = AsyncMock()

        job = MaterializationJob(feast_client=mock_client)
        job._initialized = True

        await job.close()

        mock_client.close.assert_called_once()
        assert job._initialized is False

    @pytest.mark.asyncio
    async def test_close_no_client(self):
        """Test closing when no client."""
        job = MaterializationJob()

        # Should not raise
        await job.close()
