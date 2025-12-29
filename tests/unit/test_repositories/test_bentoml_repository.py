"""Unit tests for BentoML Service Repository.

Tests cover:
- BentoMLServiceRepository CRUD operations
- BentoMLMetricsRepository time-series operations
- Health status management
- Metrics aggregation
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4, UUID

from src.repositories.bentoml_service import (
    BentoMLService,
    BentoMLServingMetrics,
    BentoMLServiceRepository,
    BentoMLMetricsRepository,
    ServiceHealthStatus,
    ServiceStatus,
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
def service_repo(mock_supabase_client):
    """Create BentoMLServiceRepository with mock client."""
    return BentoMLServiceRepository(mock_supabase_client)


@pytest.fixture
def metrics_repo(mock_supabase_client):
    """Create BentoMLMetricsRepository with mock client."""
    return BentoMLMetricsRepository(mock_supabase_client)


@pytest.fixture
def sample_service_data():
    """Sample service data from database."""
    return {
        "id": str(uuid4()),
        "service_name": "test_service",
        "bento_tag": "test_model:v1",
        "bento_version": "v1",
        "model_registry_id": str(uuid4()),
        "deployment_id": str(uuid4()),
        "container_image": "test-repo/test-image",
        "container_tag": "latest",
        "replicas": 2,
        "resources": {"cpu": "2", "memory": "4Gi"},
        "environment": "staging",
        "health_status": "healthy",
        "last_health_check": datetime.now(timezone.utc).isoformat(),
        "health_check_failures": 0,
        "serving_endpoint": "https://test.example.com/predict",
        "internal_endpoint": "http://test-service:8080",
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stopped_at": None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "test_user",
        "labels": {"app": "e2i", "tier": "ml"},
        "annotations": {"version": "1.0"},
    }


@pytest.fixture
def sample_metrics_data():
    """Sample metrics data from database."""
    return {
        "id": str(uuid4()),
        "service_id": str(uuid4()),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "requests_total": 1000,
        "requests_per_second": 50.5,
        "successful_requests": 990,
        "failed_requests": 10,
        "avg_latency_ms": 25.3,
        "p50_latency_ms": 20.0,
        "p95_latency_ms": 45.0,
        "p99_latency_ms": 80.0,
        "max_latency_ms": 150.0,
        "error_rate": 1.0,
        "error_types": {"ValidationError": 5, "TimeoutError": 5},
        "memory_mb": 512.5,
        "memory_percent": 25.0,
        "cpu_percent": 15.0,
        "predictions_count": 1000,
        "batch_size_avg": 1.0,
        "model_load_time_ms": 250.0,
        "inference_time_avg_ms": 20.0,
    }


# ============================================================================
# DATA CLASS TESTS
# ============================================================================


class TestBentoMLServiceDataClass:
    """Tests for BentoMLService dataclass."""

    def test_default_initialization(self):
        """Test default values."""
        service = BentoMLService()
        assert service.id is None
        assert service.service_name == ""
        assert service.replicas == 1
        assert service.environment == "staging"
        assert service.health_status == "unknown"
        assert service.status == "pending"

    def test_from_dict(self, sample_service_data):
        """Test creation from dictionary."""
        service = BentoMLService.from_dict(sample_service_data)
        assert service.service_name == "test_service"
        assert service.bento_tag == "test_model:v1"
        assert service.replicas == 2
        assert service.health_status == "healthy"
        assert isinstance(service.id, UUID)
        assert isinstance(service.model_registry_id, UUID)

    def test_to_dict(self, sample_service_data):
        """Test conversion to dictionary."""
        service = BentoMLService.from_dict(sample_service_data)
        result = service.to_dict()
        assert result["service_name"] == "test_service"
        assert result["bento_tag"] == "test_model:v1"
        assert result["replicas"] == 2
        assert isinstance(result["id"], str)

    def test_roundtrip_conversion(self, sample_service_data):
        """Test dict -> model -> dict roundtrip."""
        service = BentoMLService.from_dict(sample_service_data)
        result = service.to_dict()
        service2 = BentoMLService.from_dict(result)
        assert service.service_name == service2.service_name
        assert service.bento_tag == service2.bento_tag


class TestBentoMLServingMetricsDataClass:
    """Tests for BentoMLServingMetrics dataclass."""

    def test_default_initialization(self):
        """Test default values."""
        metrics = BentoMLServingMetrics()
        assert metrics.requests_total == 0
        assert metrics.successful_requests == 0
        assert metrics.error_rate is None

    def test_from_dict(self, sample_metrics_data):
        """Test creation from dictionary."""
        metrics = BentoMLServingMetrics.from_dict(sample_metrics_data)
        assert metrics.requests_total == 1000
        assert metrics.requests_per_second == 50.5
        assert metrics.error_rate == 1.0
        assert metrics.error_types == {"ValidationError": 5, "TimeoutError": 5}

    def test_to_dict(self, sample_metrics_data):
        """Test conversion to dictionary."""
        metrics = BentoMLServingMetrics.from_dict(sample_metrics_data)
        result = metrics.to_dict()
        assert result["requests_total"] == 1000
        assert result["requests_per_second"] == 50.5


# ============================================================================
# SERVICE REPOSITORY TESTS
# ============================================================================


class TestBentoMLServiceRepository:
    """Tests for BentoMLServiceRepository."""

    @pytest.mark.asyncio
    async def test_create_service(self, service_repo, mock_supabase_client, sample_service_data):
        """Test service creation."""
        mock_supabase_client.table.return_value.insert.return_value.execute.return_value.data = [
            sample_service_data
        ]

        service = await service_repo.create_service(
            service_name="test_service",
            bento_tag="test_model:v1",
            environment="staging",
        )

        assert service is not None
        assert service.service_name == "test_service"
        mock_supabase_client.table.assert_called_with("ml_bentoml_services")

    @pytest.mark.asyncio
    async def test_create_service_no_client(self):
        """Test service creation without client."""
        repo = BentoMLServiceRepository(None)
        service = await repo.create_service(
            service_name="test",
            bento_tag="test:v1",
        )
        assert service is None

    @pytest.mark.asyncio
    async def test_update_status(self, service_repo, mock_supabase_client, sample_service_data):
        """Test status update."""
        sample_service_data["status"] = "active"
        mock_supabase_client.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [
            sample_service_data
        ]

        service_id = UUID(sample_service_data["id"])
        service = await service_repo.update_status(
            service_id=service_id,
            status=ServiceStatus.ACTIVE,
            started_at=datetime.now(timezone.utc),
        )

        assert service is not None
        assert service.status == "active"

    @pytest.mark.asyncio
    async def test_update_health(self, service_repo, mock_supabase_client, sample_service_data):
        """Test health update."""
        sample_service_data["health_status"] = "healthy"
        sample_service_data["health_check_failures"] = 0
        mock_supabase_client.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [
            sample_service_data
        ]

        service_id = UUID(sample_service_data["id"])
        service = await service_repo.update_health(
            service_id=service_id,
            health_status=ServiceHealthStatus.HEALTHY,
        )

        assert service is not None
        assert service.health_status == "healthy"

    @pytest.mark.asyncio
    async def test_get_active_services(self, service_repo, mock_supabase_client, sample_service_data):
        """Test getting active services."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
            sample_service_data
        ]

        services = await service_repo.get_active_services()
        assert len(services) == 1
        assert services[0].status == "active"

    @pytest.mark.asyncio
    async def test_get_active_services_with_env_filter(
        self, service_repo, mock_supabase_client, sample_service_data
    ):
        """Test getting active services with environment filter."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
            sample_service_data
        ]

        services = await service_repo.get_active_services(environment="staging")
        assert len(services) == 1

    @pytest.mark.asyncio
    async def test_get_by_bento_tag(self, service_repo, mock_supabase_client, sample_service_data):
        """Test getting service by bento tag."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [
            sample_service_data
        ]

        service = await service_repo.get_by_bento_tag("test_model:v1")
        assert service is not None
        assert service.bento_tag == "test_model:v1"

    @pytest.mark.asyncio
    async def test_get_unhealthy_services(self, service_repo, mock_supabase_client, sample_service_data):
        """Test getting unhealthy services."""
        sample_service_data["health_status"] = "unhealthy"
        sample_service_data["health_check_failures"] = 5
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.neq.return_value.gte.return_value.execute.return_value.data = [
            sample_service_data
        ]

        services = await service_repo.get_unhealthy_services(failure_threshold=3)
        assert len(services) == 1
        assert services[0].health_status == "unhealthy"


# ============================================================================
# METRICS REPOSITORY TESTS
# ============================================================================


class TestBentoMLMetricsRepository:
    """Tests for BentoMLMetricsRepository."""

    @pytest.mark.asyncio
    async def test_record_metrics(self, metrics_repo, mock_supabase_client, sample_metrics_data):
        """Test recording metrics."""
        mock_supabase_client.table.return_value.insert.return_value.execute.return_value.data = [
            sample_metrics_data
        ]

        service_id = UUID(sample_metrics_data["service_id"])
        metrics = await metrics_repo.record_metrics(
            service_id=service_id,
            requests_total=1000,
            requests_per_second=50.5,
            error_rate=1.0,
        )

        assert metrics is not None
        assert metrics.requests_total == 1000
        mock_supabase_client.table.assert_called_with("ml_bentoml_serving_metrics")

    @pytest.mark.asyncio
    async def test_record_metrics_no_client(self):
        """Test recording metrics without client."""
        repo = BentoMLMetricsRepository(None)
        metrics = await repo.record_metrics(
            service_id=uuid4(),
            requests_total=100,
        )
        assert metrics is None

    @pytest.mark.asyncio
    async def test_get_latest_metrics(self, metrics_repo, mock_supabase_client, sample_metrics_data):
        """Test getting latest metrics."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [
            sample_metrics_data
        ]

        service_id = UUID(sample_metrics_data["service_id"])
        metrics = await metrics_repo.get_latest_metrics(service_id)

        assert metrics is not None
        assert metrics.requests_total == 1000

    @pytest.mark.asyncio
    async def test_get_metrics_range(self, metrics_repo, mock_supabase_client, sample_metrics_data):
        """Test getting metrics in time range."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.lte.return_value.order.return_value.limit.return_value.execute.return_value.data = [
            sample_metrics_data
        ]

        service_id = UUID(sample_metrics_data["service_id"])
        metrics_list = await metrics_repo.get_metrics_range(
            service_id=service_id,
            start_time=datetime.now(timezone.utc) - timedelta(hours=1),
            end_time=datetime.now(timezone.utc),
        )

        assert len(metrics_list) == 1

    @pytest.mark.asyncio
    async def test_get_metrics_range_default_times(
        self, metrics_repo, mock_supabase_client, sample_metrics_data
    ):
        """Test getting metrics with default time range."""
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.gte.return_value.lte.return_value.order.return_value.limit.return_value.execute.return_value.data = [
            sample_metrics_data
        ]

        service_id = UUID(sample_metrics_data["service_id"])
        metrics_list = await metrics_repo.get_metrics_range(service_id=service_id)

        assert len(metrics_list) == 1

    @pytest.mark.asyncio
    async def test_get_metrics_summary_with_rpc(
        self, metrics_repo, mock_supabase_client, sample_metrics_data
    ):
        """Test getting metrics summary using RPC."""
        mock_supabase_client.rpc.return_value.execute.return_value.data = {
            "total_requests": 5000,
            "avg_rps": 50.0,
            "avg_latency_ms": 25.0,
            "avg_error_rate": 1.0,
            "avg_memory_mb": 500.0,
            "avg_cpu_percent": 15.0,
            "data_points": 100,
        }

        service_id = UUID(sample_metrics_data["service_id"])
        summary = await metrics_repo.get_metrics_summary(service_id=service_id, hours=1)

        assert summary["total_requests"] == 5000
        assert summary["avg_rps"] == 50.0

    @pytest.mark.asyncio
    async def test_get_metrics_summary_no_client(self):
        """Test getting metrics summary without client."""
        repo = BentoMLMetricsRepository(None)
        summary = await repo.get_metrics_summary(uuid4())
        assert summary == {}


# ============================================================================
# ENUM TESTS
# ============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_service_health_status_values(self):
        """Test ServiceHealthStatus enum values."""
        assert ServiceHealthStatus.HEALTHY.value == "healthy"
        assert ServiceHealthStatus.UNHEALTHY.value == "unhealthy"
        assert ServiceHealthStatus.DEGRADED.value == "degraded"
        assert ServiceHealthStatus.UNKNOWN.value == "unknown"

    def test_service_status_values(self):
        """Test ServiceStatus enum values."""
        assert ServiceStatus.PENDING.value == "pending"
        assert ServiceStatus.ACTIVE.value == "active"
        assert ServiceStatus.DRAINING.value == "draining"
        assert ServiceStatus.ROLLED_BACK.value == "rolled_back"
