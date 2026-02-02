"""Unit tests for Feast Tracking Repositories.

Tests cover:
- FeastFeatureViewRepository CRUD operations
- FeastMaterializationRepository job tracking
- FeastFreshnessRepository monitoring operations
- Data class serialization/deserialization
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from src.repositories.feast_tracking import (
    FeastFeatureFreshness,
    FeastFeatureView,
    FeastFeatureViewRepository,
    FeastFreshnessRepository,
    FeastMaterializationJob,
    FeastMaterializationRepository,
    FreshnessStatus,
    MaterializationJobType,
    MaterializationStatus,
    SourceType,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client."""
    client = MagicMock()
    return client


@pytest.fixture
def view_repo(mock_supabase_client):
    """Create FeastFeatureViewRepository with mock client."""
    return FeastFeatureViewRepository(mock_supabase_client)


@pytest.fixture
def materialization_repo(mock_supabase_client):
    """Create FeastMaterializationRepository with mock client."""
    return FeastMaterializationRepository(mock_supabase_client)


@pytest.fixture
def freshness_repo(mock_supabase_client):
    """Create FeastFreshnessRepository with mock client."""
    return FeastFreshnessRepository(mock_supabase_client)


@pytest.fixture
def sample_feature_view_data():
    """Sample feature view data from database."""
    return {
        "id": str(uuid4()),
        "name": "patient_features",
        "project": "e2i_causal_analytics",
        "description": "Patient journey features for ML models",
        "entities": ["patient_id"],
        "entity_join_keys": ["patient_id"],
        "features": [
            {"name": "visit_count", "dtype": "INT64"},
            {"name": "last_visit_days", "dtype": "INT64"},
            {"name": "avg_engagement_score", "dtype": "FLOAT"},
        ],
        "feature_count": 3,
        "source_type": "batch",
        "source_name": "patient_journey_source",
        "source_config": {"table": "patient_journeys"},
        "ttl_seconds": 86400,
        "online_enabled": True,
        "batch_source_enabled": True,
        "tags": {"team": "ml", "tier": "production"},
        "owner": "ml_team",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "deleted_at": None,
    }


@pytest.fixture
def sample_materialization_data():
    """Sample materialization job data from database."""
    return {
        "id": str(uuid4()),
        "feature_view_id": str(uuid4()),
        "feature_view_name": "patient_features",
        "job_id": "mat-job-12345",
        "job_type": "incremental",
        "start_time": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "error_message": None,
        "rows_materialized": 50000,
        "bytes_written": 2500000,
        "duration_seconds": 45.5,
        "online_store_rows_written": 50000,
        "online_store_latency_ms": 120.0,
        "cpu_seconds": 30.0,
        "memory_peak_mb": 512.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_freshness_data():
    """Sample freshness data from database."""
    return {
        "id": str(uuid4()),
        "feature_view_id": str(uuid4()),
        "feature_view_name": "patient_features",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "last_materialization_time": (
            datetime.now(timezone.utc) - timedelta(minutes=30)
        ).isoformat(),
        "staleness_seconds": 1800,
        "data_lag_seconds": 300,
        "null_rate": 0.02,
        "unique_count": 10000,
        "freshness_status": "fresh",
        "staleness_threshold_seconds": 3600,
        "critical_threshold_seconds": 86400,
    }


# ============================================================================
# DATA CLASS TESTS
# ============================================================================


class TestFeastFeatureViewDataClass:
    """Tests for FeastFeatureView dataclass."""

    def test_default_initialization(self):
        """Test default values."""
        view = FeastFeatureView()
        assert view.id is None
        assert view.name == ""
        assert view.project == "e2i_causal_analytics"
        assert view.online_enabled is True
        assert view.entities == []
        assert view.feature_count == 0

    def test_from_dict(self, sample_feature_view_data):
        """Test creation from dictionary."""
        view = FeastFeatureView.from_dict(sample_feature_view_data)
        assert view.name == "patient_features"
        assert view.project == "e2i_causal_analytics"
        assert view.feature_count == 3
        assert view.source_type == "batch"
        assert isinstance(view.id, UUID)
        assert view.entities == ["patient_id"]

    def test_to_dict(self, sample_feature_view_data):
        """Test conversion to dictionary."""
        view = FeastFeatureView.from_dict(sample_feature_view_data)
        result = view.to_dict()
        assert result["name"] == "patient_features"
        assert result["feature_count"] == 3
        assert result["online_enabled"] is True
        assert isinstance(result["id"], str)

    def test_roundtrip_conversion(self, sample_feature_view_data):
        """Test dict -> model -> dict roundtrip."""
        view = FeastFeatureView.from_dict(sample_feature_view_data)
        result = view.to_dict()
        view2 = FeastFeatureView.from_dict(result)
        assert view.name == view2.name
        assert view.feature_count == view2.feature_count


class TestFeastMaterializationJobDataClass:
    """Tests for FeastMaterializationJob dataclass."""

    def test_default_initialization(self):
        """Test default values."""
        job = FeastMaterializationJob()
        assert job.id is None
        assert job.status == "pending"
        assert job.job_type == "incremental"
        assert job.rows_materialized == 0

    def test_from_dict(self, sample_materialization_data):
        """Test creation from dictionary."""
        job = FeastMaterializationJob.from_dict(sample_materialization_data)
        assert job.feature_view_name == "patient_features"
        assert job.status == "success"
        assert job.rows_materialized == 50000
        assert job.duration_seconds == 45.5
        assert isinstance(job.id, UUID)

    def test_to_dict(self, sample_materialization_data):
        """Test conversion to dictionary."""
        job = FeastMaterializationJob.from_dict(sample_materialization_data)
        result = job.to_dict()
        assert result["feature_view_name"] == "patient_features"
        assert result["rows_materialized"] == 50000
        assert isinstance(result["id"], str)


class TestFeastFeatureFreshnessDataClass:
    """Tests for FeastFeatureFreshness dataclass."""

    def test_default_initialization(self):
        """Test default values."""
        freshness = FeastFeatureFreshness()
        assert freshness.id is None
        assert freshness.freshness_status == "unknown"
        assert freshness.staleness_seconds is None

    def test_from_dict(self, sample_freshness_data):
        """Test creation from dictionary."""
        freshness = FeastFeatureFreshness.from_dict(sample_freshness_data)
        assert freshness.feature_view_name == "patient_features"
        assert freshness.freshness_status == "fresh"
        assert freshness.staleness_seconds == 1800
        assert freshness.null_rate == 0.02

    def test_to_dict(self, sample_freshness_data):
        """Test conversion to dictionary."""
        freshness = FeastFeatureFreshness.from_dict(sample_freshness_data)
        result = freshness.to_dict()
        assert result["freshness_status"] == "fresh"
        assert result["staleness_seconds"] == 1800


# ============================================================================
# FEATURE VIEW REPOSITORY TESTS
# ============================================================================


class TestFeastFeatureViewRepository:
    """Tests for FeastFeatureViewRepository."""

    @pytest.mark.asyncio
    async def test_create_feature_view(
        self, view_repo, mock_supabase_client, sample_feature_view_data
    ):
        """Test feature view creation."""
        mock_supabase_client.table.return_value.insert.return_value.execute.return_value.data = [
            sample_feature_view_data
        ]

        view = await view_repo.create_feature_view(
            name="patient_features",
            description="Patient journey features",
            entities=["patient_id"],
            features=[{"name": "visit_count", "dtype": "INT64"}],
            source_type="batch",
            ttl_seconds=86400,
        )

        assert view is not None
        assert view.name == "patient_features"
        mock_supabase_client.table.assert_called_with("ml_feast_feature_views")

    @pytest.mark.asyncio
    async def test_create_feature_view_no_client(self):
        """Test feature view creation without client."""
        repo = FeastFeatureViewRepository(None)
        view = await repo.create_feature_view(name="test_view")
        assert view is None

    @pytest.mark.asyncio
    async def test_get_by_name(self, view_repo, mock_supabase_client, sample_feature_view_data):
        """Test getting feature view by name."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.eq.return_value.is_.return_value.limit.return_value.execute.return_value.data = [
            sample_feature_view_data
        ]

        view = await view_repo.get_by_name("patient_features")
        assert view is not None
        assert view.name == "patient_features"

    @pytest.mark.asyncio
    async def test_get_active_views(
        self, view_repo, mock_supabase_client, sample_feature_view_data
    ):
        """Test getting active feature views."""
        mock_supabase_client.table.return_value.select.return_value.is_.return_value.execute.return_value.data = [
            sample_feature_view_data
        ]

        views = await view_repo.get_active_views()
        assert len(views) == 1
        assert views[0].name == "patient_features"

    @pytest.mark.asyncio
    async def test_get_active_views_online_only(
        self, view_repo, mock_supabase_client, sample_feature_view_data
    ):
        """Test getting online-enabled feature views."""
        mock_supabase_client.table.return_value.select.return_value.is_.return_value.eq.return_value.execute.return_value.data = [
            sample_feature_view_data
        ]

        views = await view_repo.get_active_views(online_only=True)
        assert len(views) == 1
        assert views[0].online_enabled is True

    @pytest.mark.asyncio
    async def test_update_feature_view(
        self, view_repo, mock_supabase_client, sample_feature_view_data
    ):
        """Test updating feature view."""
        sample_feature_view_data["ttl_seconds"] = 172800
        mock_supabase_client.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [
            sample_feature_view_data
        ]

        view_id = UUID(sample_feature_view_data["id"])
        view = await view_repo.update_feature_view(view_id=view_id, ttl_seconds=172800)

        assert view is not None
        assert view.ttl_seconds == 172800

    @pytest.mark.asyncio
    async def test_soft_delete(self, view_repo, mock_supabase_client, sample_feature_view_data):
        """Test soft deleting feature view."""
        sample_feature_view_data["deleted_at"] = datetime.now(timezone.utc).isoformat()
        mock_supabase_client.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [
            sample_feature_view_data
        ]

        view_id = UUID(sample_feature_view_data["id"])
        result = await view_repo.soft_delete(view_id)

        assert result is True


# ============================================================================
# MATERIALIZATION REPOSITORY TESTS
# ============================================================================


class TestFeastMaterializationRepository:
    """Tests for FeastMaterializationRepository."""

    @pytest.mark.asyncio
    async def test_create_job(
        self, materialization_repo, mock_supabase_client, sample_materialization_data
    ):
        """Test materialization job creation."""
        mock_supabase_client.table.return_value.insert.return_value.execute.return_value.data = [
            sample_materialization_data
        ]

        feature_view_id = UUID(sample_materialization_data["feature_view_id"])
        job = await materialization_repo.create_job(
            feature_view_id=feature_view_id,
            feature_view_name="patient_features",
            start_time=datetime.now(timezone.utc) - timedelta(hours=2),
            end_time=datetime.now(timezone.utc),
            job_type="incremental",
        )

        assert job is not None
        assert job.feature_view_name == "patient_features"
        mock_supabase_client.table.assert_called_with("ml_feast_materialization_jobs")

    @pytest.mark.asyncio
    async def test_create_job_no_client(self):
        """Test job creation without client."""
        repo = FeastMaterializationRepository(None)
        job = await repo.create_job(
            feature_view_id=uuid4(),
            feature_view_name="test",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
        )
        assert job is None

    @pytest.mark.asyncio
    async def test_update_status_running(
        self, materialization_repo, mock_supabase_client, sample_materialization_data
    ):
        """Test updating job to running status."""
        sample_materialization_data["status"] = "running"
        mock_supabase_client.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [
            sample_materialization_data
        ]

        job_id = UUID(sample_materialization_data["id"])
        job = await materialization_repo.update_status(
            job_id=job_id,
            status=MaterializationStatus.RUNNING,
        )

        assert job is not None
        assert job.status == "running"

    @pytest.mark.asyncio
    async def test_update_status_success(
        self, materialization_repo, mock_supabase_client, sample_materialization_data
    ):
        """Test updating job to success status with metrics."""
        sample_materialization_data["status"] = "success"
        sample_materialization_data["rows_materialized"] = 100000
        mock_supabase_client.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [
            sample_materialization_data
        ]

        job_id = UUID(sample_materialization_data["id"])
        job = await materialization_repo.update_status(
            job_id=job_id,
            status=MaterializationStatus.SUCCESS,
            rows_materialized=100000,
            duration_seconds=60.0,
        )

        assert job is not None
        assert job.status == "success"
        assert job.rows_materialized == 100000

    @pytest.mark.asyncio
    async def test_update_status_failed(
        self, materialization_repo, mock_supabase_client, sample_materialization_data
    ):
        """Test updating job to failed status with error."""
        sample_materialization_data["status"] = "failed"
        sample_materialization_data["error_message"] = "Connection timeout"
        mock_supabase_client.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [
            sample_materialization_data
        ]

        job_id = UUID(sample_materialization_data["id"])
        job = await materialization_repo.update_status(
            job_id=job_id,
            status=MaterializationStatus.FAILED,
            error_message="Connection timeout",
        )

        assert job is not None
        assert job.status == "failed"
        assert job.error_message == "Connection timeout"

    @pytest.mark.asyncio
    async def test_get_jobs_for_view(
        self, materialization_repo, mock_supabase_client, sample_materialization_data
    ):
        """Test getting jobs for a feature view."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [
            sample_materialization_data
        ]

        feature_view_id = UUID(sample_materialization_data["feature_view_id"])
        jobs = await materialization_repo.get_jobs_for_view(feature_view_id)

        assert len(jobs) == 1
        assert jobs[0].feature_view_name == "patient_features"

    @pytest.mark.asyncio
    async def test_get_recent_jobs(
        self, materialization_repo, mock_supabase_client, sample_materialization_data
    ):
        """Test getting recent jobs."""
        mock_supabase_client.table.return_value.select.return_value.gte.return_value.order.return_value.execute.return_value.data = [
            sample_materialization_data
        ]

        jobs = await materialization_repo.get_recent_jobs(days=7)

        assert len(jobs) == 1

    @pytest.mark.asyncio
    async def test_get_summary_with_rpc(
        self, materialization_repo, mock_supabase_client, sample_materialization_data
    ):
        """Test getting materialization summary using RPC."""
        mock_supabase_client.rpc.return_value.execute.return_value.data = {
            "total_jobs": 10,
            "successful_jobs": 9,
            "failed_jobs": 1,
            "total_rows_materialized": 500000,
            "avg_duration_seconds": 45.0,
            "success_rate": 90.0,
        }

        feature_view_id = UUID(sample_materialization_data["feature_view_id"])
        summary = await materialization_repo.get_summary(feature_view_id=feature_view_id, days=7)

        assert summary["total_jobs"] == 10
        assert summary["success_rate"] == 90.0


# ============================================================================
# FRESHNESS REPOSITORY TESTS
# ============================================================================


class TestFeastFreshnessRepository:
    """Tests for FeastFreshnessRepository."""

    @pytest.mark.asyncio
    async def test_record_freshness(
        self, freshness_repo, mock_supabase_client, sample_freshness_data
    ):
        """Test recording freshness check."""
        mock_supabase_client.table.return_value.insert.return_value.execute.return_value.data = [
            sample_freshness_data
        ]

        feature_view_id = UUID(sample_freshness_data["feature_view_id"])
        freshness = await freshness_repo.record_freshness(
            feature_view_id=feature_view_id,
            feature_view_name="patient_features",
            staleness_seconds=1800,
            freshness_status="fresh",
        )

        assert freshness is not None
        assert freshness.freshness_status == "fresh"
        mock_supabase_client.table.assert_called_with("ml_feast_feature_freshness")

    @pytest.mark.asyncio
    async def test_record_freshness_no_client(self):
        """Test recording freshness without client."""
        repo = FeastFreshnessRepository(None)
        freshness = await repo.record_freshness(
            feature_view_id=uuid4(),
            feature_view_name="test",
        )
        assert freshness is None

    @pytest.mark.asyncio
    async def test_get_latest(self, freshness_repo, mock_supabase_client, sample_freshness_data):
        """Test getting latest freshness record."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [
            sample_freshness_data
        ]

        feature_view_id = UUID(sample_freshness_data["feature_view_id"])
        freshness = await freshness_repo.get_latest(feature_view_id)

        assert freshness is not None
        assert freshness.freshness_status == "fresh"

    @pytest.mark.asyncio
    async def test_get_stale_views(
        self, freshness_repo, mock_supabase_client, sample_freshness_data
    ):
        """Test getting stale feature views."""
        sample_freshness_data["freshness_status"] = "stale"
        mock_supabase_client.table.return_value.select.return_value.in_.return_value.order.return_value.execute.return_value.data = [
            sample_freshness_data
        ]

        stale = await freshness_repo.get_stale_views()

        assert len(stale) == 1
        assert stale[0].freshness_status == "stale"

    @pytest.mark.asyncio
    async def test_get_freshness_history(
        self, freshness_repo, mock_supabase_client, sample_freshness_data
    ):
        """Test getting freshness history."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.order.return_value.execute.return_value.data = [
            sample_freshness_data
        ]

        feature_view_id = UUID(sample_freshness_data["feature_view_id"])
        history = await freshness_repo.get_freshness_history(feature_view_id, hours=24)

        assert len(history) == 1


# ============================================================================
# ENUM TESTS
# ============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_materialization_job_type_values(self):
        """Test MaterializationJobType enum values."""
        assert MaterializationJobType.FULL.value == "full"
        assert MaterializationJobType.INCREMENTAL.value == "incremental"

    def test_materialization_status_values(self):
        """Test MaterializationStatus enum values."""
        assert MaterializationStatus.PENDING.value == "pending"
        assert MaterializationStatus.RUNNING.value == "running"
        assert MaterializationStatus.SUCCESS.value == "success"
        assert MaterializationStatus.FAILED.value == "failed"

    def test_freshness_status_values(self):
        """Test FreshnessStatus enum values."""
        assert FreshnessStatus.FRESH.value == "fresh"
        assert FreshnessStatus.STALE.value == "stale"
        assert FreshnessStatus.CRITICAL.value == "critical"
        assert FreshnessStatus.UNKNOWN.value == "unknown"

    def test_source_type_values(self):
        """Test SourceType enum values."""
        assert SourceType.BATCH.value == "batch"
        assert SourceType.STREAM.value == "stream"
        assert SourceType.REQUEST.value == "request"
