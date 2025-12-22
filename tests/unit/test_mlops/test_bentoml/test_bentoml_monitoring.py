"""Tests for BentoML monitoring module."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mlops.bentoml_monitoring import (
    Alert,
    AlertSeverity,
    BentoMLHealthMonitor,
    HealthCheckResult,
    MetricSnapshot,
    PrometheusMetrics,
    ServiceStatus,
    create_health_monitor,
    quick_health_check,
)


class TestServiceStatus:
    """Test ServiceStatus enum."""

    def test_all_values(self):
        """Should have all expected status values."""
        assert ServiceStatus.HEALTHY.value == "healthy"
        assert ServiceStatus.DEGRADED.value == "degraded"
        assert ServiceStatus.UNHEALTHY.value == "unhealthy"
        assert ServiceStatus.UNKNOWN.value == "unknown"


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass."""

    def test_creation(self):
        """Should create result with all fields."""
        result = HealthCheckResult(
            service_name="test-service",
            status=ServiceStatus.HEALTHY,
            timestamp=datetime.now(timezone.utc),
            latency_ms=50.5,
        )

        assert result.service_name == "test-service"
        assert result.status == ServiceStatus.HEALTHY
        assert result.latency_ms == 50.5
        assert result.error is None

    def test_with_error(self):
        """Should store error message."""
        result = HealthCheckResult(
            service_name="test-service",
            status=ServiceStatus.UNHEALTHY,
            timestamp=datetime.now(timezone.utc),
            latency_ms=0,
            error="Connection refused",
        )

        assert result.error == "Connection refused"


class TestAlert:
    """Test Alert dataclass."""

    def test_creation(self):
        """Should create alert with required fields."""
        alert = Alert(
            service_name="test-service",
            severity=AlertSeverity.ERROR,
            message="Service is unhealthy",
            timestamp=datetime.now(timezone.utc),
        )

        assert alert.service_name == "test-service"
        assert alert.severity == AlertSeverity.ERROR
        assert alert.metric_name is None

    def test_with_metric(self):
        """Should store metric details."""
        alert = Alert(
            service_name="test-service",
            severity=AlertSeverity.WARNING,
            message="High latency",
            timestamp=datetime.now(timezone.utc),
            metric_name="latency_ms",
            metric_value=1500.0,
            threshold=1000.0,
        )

        assert alert.metric_name == "latency_ms"
        assert alert.metric_value == 1500.0
        assert alert.threshold == 1000.0


class TestPrometheusMetrics:
    """Test PrometheusMetrics class."""

    def test_initialization_without_prometheus(self):
        """Should handle missing prometheus_client gracefully."""
        with patch("src.mlops.bentoml_monitoring.PROMETHEUS_AVAILABLE", False):
            metrics = PrometheusMetrics()
            # Should not raise
            metrics.record_request("svc", "predict", "success", 0.1)
            metrics.record_prediction("svc", "classification", 10)
            metrics.record_error("svc", "timeout")
            metrics.set_health_status("svc", True)

    @patch("src.mlops.bentoml_monitoring.PROMETHEUS_AVAILABLE", True)
    def test_initialization_with_prometheus(self):
        """Should create metrics when prometheus available."""
        with patch("src.mlops.bentoml_monitoring.Counter") as mock_counter, \
             patch("src.mlops.bentoml_monitoring.Histogram") as mock_histogram, \
             patch("src.mlops.bentoml_monitoring.Gauge") as mock_gauge:
            metrics = PrometheusMetrics(namespace="test")

            assert mock_counter.called
            assert mock_histogram.called
            assert mock_gauge.called


class TestBentoMLHealthMonitor:
    """Test BentoMLHealthMonitor class."""

    def test_initialization(self):
        """Should initialize with defaults."""
        monitor = BentoMLHealthMonitor()

        assert monitor.check_interval == 30
        assert len(monitor._services) == 0

    def test_register_service(self):
        """Should register service correctly."""
        monitor = BentoMLHealthMonitor()
        monitor.register_service(
            name="test-service",
            url="http://localhost:3000",
            model_type="classification",
        )

        assert "test-service" in monitor._services
        assert monitor._services["test-service"]["url"] == "http://localhost:3000"
        assert "test-service" in monitor._health_history

    def test_unregister_service(self):
        """Should unregister service correctly."""
        monitor = BentoMLHealthMonitor()
        monitor.register_service(
            name="test-service",
            url="http://localhost:3000",
        )
        monitor.unregister_service("test-service")

        assert "test-service" not in monitor._services
        assert "test-service" not in monitor._health_history

    def test_set_threshold(self):
        """Should update threshold value."""
        monitor = BentoMLHealthMonitor()
        monitor.set_threshold("latency_warning_ms", 1000)

        assert monitor._thresholds["latency_warning_ms"] == 1000

    def test_add_alert_handler(self):
        """Should add alert handler."""
        monitor = BentoMLHealthMonitor()
        handler = MagicMock()
        monitor.add_alert_handler(handler)

        assert handler in monitor.alert_handlers

    @pytest.mark.asyncio
    async def test_check_health_unregistered_service(self):
        """Should return unknown status for unregistered service."""
        monitor = BentoMLHealthMonitor()

        result = await monitor.check_health("unknown-service")

        assert result.status == ServiceStatus.UNKNOWN
        assert "not registered" in result.error

    @pytest.mark.asyncio
    async def test_check_health_success(self):
        """Should return healthy status for successful check."""
        monitor = BentoMLHealthMonitor()
        monitor.register_service(
            name="test-service",
            url="http://localhost:3000",
        )

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "healthy"})

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response

            mock_session_instance = MagicMock()
            mock_session_instance.get.return_value = mock_context
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock()
            mock_session.return_value = mock_session_instance

            result = await monitor.check_health("test-service")

            assert result.status == ServiceStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_health_timeout(self):
        """Should return unhealthy status on timeout."""
        monitor = BentoMLHealthMonitor()
        monitor.register_service(
            name="test-service",
            url="http://localhost:3000",
        )

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.side_effect = asyncio.TimeoutError()

            result = await monitor.check_health("test-service")

            assert result.status == ServiceStatus.UNHEALTHY
            assert "timed out" in result.error.lower()

    def test_get_health_summary_empty(self):
        """Should return empty summary when no services."""
        monitor = BentoMLHealthMonitor()

        summary = monitor.get_health_summary()

        assert summary["overall_status"] == "healthy"
        assert len(summary["services"]) == 0

    def test_get_health_summary_with_services(self):
        """Should return summary with service status."""
        monitor = BentoMLHealthMonitor()
        monitor.register_service("svc1", "http://localhost:3001")

        # Add a health result
        monitor._health_history["svc1"].append(
            HealthCheckResult(
                service_name="svc1",
                status=ServiceStatus.HEALTHY,
                timestamp=datetime.now(timezone.utc),
                latency_ms=50,
            )
        )

        summary = monitor.get_health_summary()

        assert "svc1" in summary["services"]
        assert summary["services"]["svc1"]["status"] == "healthy"


class TestCreateHealthMonitor:
    """Test create_health_monitor factory."""

    def test_creates_monitor(self):
        """Should create monitor with defaults."""
        monitor = create_health_monitor()

        assert isinstance(monitor, BentoMLHealthMonitor)
        assert monitor.check_interval == 30

    def test_creates_with_services(self):
        """Should register provided services."""
        services = [
            {"name": "svc1", "url": "http://localhost:3001"},
            {"name": "svc2", "url": "http://localhost:3002"},
        ]

        monitor = create_health_monitor(services=services)

        assert "svc1" in monitor._services
        assert "svc2" in monitor._services

    def test_creates_with_custom_interval(self):
        """Should use custom check interval."""
        monitor = create_health_monitor(check_interval=60)

        assert monitor.check_interval == 60


class TestQuickHealthCheck:
    """Test quick_health_check function."""

    @pytest.mark.asyncio
    async def test_healthy_response(self):
        """Should return healthy status."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "healthy"})

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response

            mock_session_instance = MagicMock()
            mock_session_instance.get.return_value = mock_context
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock()
            mock_session.return_value = mock_session_instance

            result = await quick_health_check("http://localhost:3000")

            assert result["status"] == "healthy"
            assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_error_response(self):
        """Should return unhealthy on error."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.side_effect = Exception("Connection refused")

            result = await quick_health_check("http://localhost:3000")

            assert result["status"] == "unhealthy"
            assert "error" in result
