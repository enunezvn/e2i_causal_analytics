"""Unit tests for self-monitoring module.

Tests the SelfMonitor class and related components for tracking
internal observability metrics and health status.
"""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.ml_foundation.observability_connector.self_monitor import (
    AsyncLatencyContext,
    ComponentHealth,
    HealthStatus,
    LatencyContext,
    LatencyStats,
    LatencyThresholds,
    LatencyTracker,
    MetricType,
    OverallHealth,
    SelfMonitor,
    SelfMonitorConfig,
    get_self_monitor,
    reset_self_monitor,
)


# ============================================================================
# LATENCY THRESHOLDS TESTS
# ============================================================================


class TestLatencyThresholds:
    """Tests for LatencyThresholds class."""

    def test_default_values(self):
        """Test default threshold values."""
        thresholds = LatencyThresholds()
        assert thresholds.warning_ms == 100.0
        assert thresholds.critical_ms == 500.0

    def test_custom_values(self):
        """Test custom threshold values."""
        thresholds = LatencyThresholds(warning_ms=50.0, critical_ms=200.0)
        assert thresholds.warning_ms == 50.0
        assert thresholds.critical_ms == 200.0

    def test_check_healthy(self):
        """Test check returns healthy for low latency."""
        thresholds = LatencyThresholds(warning_ms=100.0, critical_ms=500.0)
        assert thresholds.check(50.0) == HealthStatus.HEALTHY
        assert thresholds.check(99.9) == HealthStatus.HEALTHY

    def test_check_degraded(self):
        """Test check returns degraded for warning latency."""
        thresholds = LatencyThresholds(warning_ms=100.0, critical_ms=500.0)
        assert thresholds.check(100.0) == HealthStatus.DEGRADED
        assert thresholds.check(250.0) == HealthStatus.DEGRADED
        assert thresholds.check(499.9) == HealthStatus.DEGRADED

    def test_check_unhealthy(self):
        """Test check returns unhealthy for critical latency."""
        thresholds = LatencyThresholds(warning_ms=100.0, critical_ms=500.0)
        assert thresholds.check(500.0) == HealthStatus.UNHEALTHY
        assert thresholds.check(1000.0) == HealthStatus.UNHEALTHY


# ============================================================================
# LATENCY STATS TESTS
# ============================================================================


class TestLatencyStats:
    """Tests for LatencyStats dataclass."""

    def test_default_values(self):
        """Test default stats values."""
        stats = LatencyStats()
        assert stats.count == 0
        assert stats.min_ms == 0.0
        assert stats.max_ms == 0.0
        assert stats.avg_ms == 0.0
        assert stats.status == HealthStatus.UNKNOWN

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = LatencyStats(
            count=10,
            min_ms=5.0,
            max_ms=100.0,
            avg_ms=50.0,
            p50_ms=45.0,
            p95_ms=90.0,
            p99_ms=98.0,
            last_ms=55.0,
            last_recorded=datetime(2025, 1, 1, tzinfo=timezone.utc),
            status=HealthStatus.HEALTHY,
        )
        result = stats.to_dict()
        assert result["count"] == 10
        assert result["min_ms"] == 5.0
        assert result["max_ms"] == 100.0
        assert result["avg_ms"] == 50.0
        assert result["status"] == "healthy"
        assert "2025-01-01" in result["last_recorded"]


# ============================================================================
# LATENCY TRACKER TESTS
# ============================================================================


class TestLatencyTracker:
    """Tests for LatencyTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        thresholds = LatencyThresholds()
        tracker = LatencyTracker(
            metric_type=MetricType.SPAN_EMISSION,
            thresholds=thresholds,
            window_size=100,
        )
        assert tracker.metric_type == MetricType.SPAN_EMISSION
        assert tracker.window_size == 100

    def test_record_single_sample(self):
        """Test recording a single sample."""
        tracker = LatencyTracker(
            metric_type=MetricType.OPIK_API,
            thresholds=LatencyThresholds(),
        )
        tracker.record(50.0)

        stats = tracker.get_stats()
        assert stats.count == 1
        assert stats.min_ms == 50.0
        assert stats.max_ms == 50.0
        assert stats.avg_ms == 50.0
        assert stats.last_ms == 50.0

    def test_record_multiple_samples(self):
        """Test recording multiple samples."""
        tracker = LatencyTracker(
            metric_type=MetricType.DATABASE_WRITE,
            thresholds=LatencyThresholds(),
        )
        for latency in [10.0, 20.0, 30.0, 40.0, 50.0]:
            tracker.record(latency)

        stats = tracker.get_stats()
        assert stats.count == 5
        assert stats.min_ms == 10.0
        assert stats.max_ms == 50.0
        assert stats.avg_ms == 30.0

    def test_record_rolling_window(self):
        """Test rolling window behavior."""
        tracker = LatencyTracker(
            metric_type=MetricType.SPAN_EMISSION,
            thresholds=LatencyThresholds(),
            window_size=5,
        )

        # Add 10 samples (window size is 5)
        for i in range(10):
            tracker.record(float(i * 10))

        stats = tracker.get_stats()
        # Should only have last 5 samples: 50, 60, 70, 80, 90
        assert stats.min_ms == 50.0
        assert stats.max_ms == 90.0
        # Total count should be 10 (all recorded)
        assert stats.count == 10

    def test_record_error(self):
        """Test recording errors."""
        tracker = LatencyTracker(
            metric_type=MetricType.OPIK_API,
            thresholds=LatencyThresholds(),
        )
        tracker.record_error("Connection timeout")

        health = tracker.get_component_health()
        assert health.error_count == 1
        assert health.last_error == "Connection timeout"
        assert health.last_error_time is not None

    def test_get_stats_empty(self):
        """Test get_stats with no samples."""
        tracker = LatencyTracker(
            metric_type=MetricType.SPAN_EMISSION,
            thresholds=LatencyThresholds(),
        )
        stats = tracker.get_stats()
        assert stats.count == 0
        assert stats.status == HealthStatus.UNKNOWN

    def test_get_stats_with_thresholds(self):
        """Test stats status reflects threshold violations."""
        tracker = LatencyTracker(
            metric_type=MetricType.OPIK_API,
            thresholds=LatencyThresholds(warning_ms=50.0, critical_ms=100.0),
        )

        # Add samples with high average
        for _ in range(5):
            tracker.record(75.0)

        stats = tracker.get_stats()
        assert stats.avg_ms == 75.0
        assert stats.status == HealthStatus.DEGRADED

    def test_get_component_health(self):
        """Test getting component health."""
        tracker = LatencyTracker(
            metric_type=MetricType.DATABASE_WRITE,
            thresholds=LatencyThresholds(),
        )
        tracker.record(25.0)
        tracker.record(35.0)

        health = tracker.get_component_health()
        assert health.component == "database_write"
        assert health.status == HealthStatus.HEALTHY
        assert health.latency_stats.count == 2

    def test_reset(self):
        """Test resetting tracker."""
        tracker = LatencyTracker(
            metric_type=MetricType.SPAN_EMISSION,
            thresholds=LatencyThresholds(),
        )
        tracker.record(50.0)
        tracker.record_error("Error")

        tracker.reset()

        stats = tracker.get_stats()
        health = tracker.get_component_health()
        assert stats.count == 0
        assert health.error_count == 0


# ============================================================================
# SELF MONITOR CONFIG TESTS
# ============================================================================


class TestSelfMonitorConfig:
    """Tests for SelfMonitorConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SelfMonitorConfig()
        assert config.health_emission_interval_seconds == 60.0
        assert config.window_size == 100
        assert config.min_samples_for_stats == 5
        assert config.emit_health_spans is True
        assert config.log_degraded_status is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SelfMonitorConfig(
            health_emission_interval_seconds=30.0,
            window_size=50,
            emit_health_spans=False,
        )
        assert config.health_emission_interval_seconds == 30.0
        assert config.window_size == 50
        assert config.emit_health_spans is False

    def test_default_thresholds_initialized(self):
        """Test that default thresholds are initialized."""
        config = SelfMonitorConfig()
        assert MetricType.SPAN_EMISSION in config.thresholds
        assert MetricType.OPIK_API in config.thresholds
        assert MetricType.DATABASE_WRITE in config.thresholds

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        custom_thresholds = {
            MetricType.SPAN_EMISSION: LatencyThresholds(
                warning_ms=50.0, critical_ms=200.0
            ),
        }
        config = SelfMonitorConfig(thresholds=custom_thresholds)
        assert config.thresholds[MetricType.SPAN_EMISSION].warning_ms == 50.0


# ============================================================================
# SELF MONITOR TESTS
# ============================================================================


class TestSelfMonitor:
    """Tests for SelfMonitor class."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_self_monitor()

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = SelfMonitor()
        assert monitor.config is not None
        assert len(monitor._trackers) == len(MetricType)

    def test_initialization_with_config(self):
        """Test monitor initialization with custom config."""
        config = SelfMonitorConfig(window_size=50)
        monitor = SelfMonitor(config)
        assert monitor.config.window_size == 50

    def test_record_span_emission_latency(self):
        """Test recording span emission latency."""
        monitor = SelfMonitor()
        monitor.record_span_emission_latency(15.5)

        stats = monitor.get_latency_stats(MetricType.SPAN_EMISSION)
        assert stats.count == 1
        assert stats.last_ms == 15.5

    def test_record_opik_latency(self):
        """Test recording Opik API latency."""
        monitor = SelfMonitor()
        monitor.record_opik_latency(45.2)

        stats = monitor.get_latency_stats(MetricType.OPIK_API)
        assert stats.count == 1
        assert stats.last_ms == 45.2

    def test_record_database_latency(self):
        """Test recording database latency."""
        monitor = SelfMonitor()
        monitor.record_database_latency(8.3)

        stats = monitor.get_latency_stats(MetricType.DATABASE_WRITE)
        assert stats.count == 1
        assert stats.last_ms == 8.3

    def test_record_batch_flush_latency(self):
        """Test recording batch flush latency."""
        monitor = SelfMonitor()
        monitor.record_batch_flush_latency(120.0)

        stats = monitor.get_latency_stats(MetricType.BATCH_FLUSH)
        assert stats.count == 1
        assert stats.last_ms == 120.0

    def test_record_cache_latency(self):
        """Test recording cache latency."""
        monitor = SelfMonitor()
        monitor.record_cache_latency(2.5)

        stats = monitor.get_latency_stats(MetricType.CACHE_OPERATION)
        assert stats.count == 1
        assert stats.last_ms == 2.5

    def test_record_errors(self):
        """Test recording errors for different components."""
        monitor = SelfMonitor()

        monitor.record_span_emission_error("Span error")
        monitor.record_opik_error("Opik error")
        monitor.record_database_error("DB error")
        monitor.record_batch_flush_error("Batch error")
        monitor.record_cache_error("Cache error")

        health = monitor.get_health_status()
        for component_name, component in health.components.items():
            assert component.error_count == 1

    def test_get_health_status_healthy(self):
        """Test health status when all metrics are healthy."""
        config = SelfMonitorConfig(min_samples_for_stats=2)
        monitor = SelfMonitor(config)

        # Record some healthy latencies
        for _ in range(5):
            monitor.record_span_emission_latency(10.0)
            monitor.record_opik_latency(20.0)
            monitor.record_database_latency(5.0)

        health = monitor.get_health_status()
        assert health.status == HealthStatus.HEALTHY
        assert len(health.alerts) == 0

    def test_get_health_status_degraded(self):
        """Test health status when metrics are degraded."""
        config = SelfMonitorConfig(
            min_samples_for_stats=2,
            thresholds={
                MetricType.OPIK_API: LatencyThresholds(warning_ms=50.0, critical_ms=200.0),
                MetricType.SPAN_EMISSION: LatencyThresholds(),
                MetricType.DATABASE_WRITE: LatencyThresholds(),
                MetricType.BATCH_FLUSH: LatencyThresholds(),
                MetricType.CACHE_OPERATION: LatencyThresholds(),
            },
        )
        monitor = SelfMonitor(config)

        # Record degraded Opik latencies
        for _ in range(5):
            monitor.record_opik_latency(100.0)  # Above warning threshold
            monitor.record_span_emission_latency(10.0)

        health = monitor.get_health_status()
        assert health.status == HealthStatus.DEGRADED
        assert len(health.alerts) > 0
        assert any("opik_api" in alert for alert in health.alerts)

    def test_get_health_status_unhealthy(self):
        """Test health status when metrics are critical."""
        config = SelfMonitorConfig(
            min_samples_for_stats=2,
            thresholds={
                MetricType.DATABASE_WRITE: LatencyThresholds(
                    warning_ms=50.0, critical_ms=100.0
                ),
                MetricType.SPAN_EMISSION: LatencyThresholds(),
                MetricType.OPIK_API: LatencyThresholds(),
                MetricType.BATCH_FLUSH: LatencyThresholds(),
                MetricType.CACHE_OPERATION: LatencyThresholds(),
            },
        )
        monitor = SelfMonitor(config)

        # Record critical database latencies
        for _ in range(5):
            monitor.record_database_latency(200.0)  # Above critical threshold
            monitor.record_span_emission_latency(10.0)

        health = monitor.get_health_status()
        assert health.status == HealthStatus.UNHEALTHY

    def test_get_health_status_unknown_insufficient_data(self):
        """Test health status is unknown with insufficient data."""
        config = SelfMonitorConfig(min_samples_for_stats=10)
        monitor = SelfMonitor(config)

        # Only record 2 samples
        monitor.record_span_emission_latency(10.0)
        monitor.record_span_emission_latency(15.0)

        health = monitor.get_health_status()
        assert health.status == HealthStatus.UNKNOWN

    def test_get_health_status_high_error_rate(self):
        """Test alert for high error rate."""
        config = SelfMonitorConfig(min_samples_for_stats=2)
        monitor = SelfMonitor(config)

        # Record some latencies and many errors
        for _ in range(5):
            monitor.record_opik_latency(20.0)

        for _ in range(10):
            monitor.record_opik_error("Connection failed")

        health = monitor.get_health_status()
        assert any("error rate" in alert.lower() for alert in health.alerts)

    def test_get_all_latency_stats(self):
        """Test getting all latency statistics."""
        monitor = SelfMonitor()
        monitor.record_span_emission_latency(10.0)
        monitor.record_opik_latency(20.0)

        all_stats = monitor.get_all_latency_stats()
        assert "span_emission" in all_stats
        assert "opik_api" in all_stats
        assert all_stats["span_emission"].last_ms == 10.0
        assert all_stats["opik_api"].last_ms == 20.0

    def test_uptime_seconds(self):
        """Test uptime tracking."""
        monitor = SelfMonitor()
        time.sleep(0.1)
        assert monitor.uptime_seconds >= 0.1

    def test_reset(self):
        """Test resetting the monitor."""
        monitor = SelfMonitor()
        monitor.record_span_emission_latency(10.0)
        monitor.record_opik_error("Error")

        monitor.reset()

        stats = monitor.get_latency_stats(MetricType.SPAN_EMISSION)
        assert stats.count == 0
        assert monitor._health_span_count == 0

    def test_set_span_emitter(self):
        """Test setting span emitter callback."""
        monitor = SelfMonitor()
        emitter = AsyncMock()

        monitor.set_span_emitter(emitter)
        assert monitor._span_emitter == emitter


# ============================================================================
# HEALTH EMISSION TESTS
# ============================================================================


class TestHealthEmission:
    """Tests for health span emission."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_self_monitor()

    @pytest.mark.asyncio
    async def test_emit_health_span_now(self):
        """Test immediate health span emission."""
        config = SelfMonitorConfig(emit_health_spans=True)
        monitor = SelfMonitor(config)
        emitter = AsyncMock()
        monitor.set_span_emitter(emitter)

        # Record some data
        monitor.record_span_emission_latency(10.0)

        health = await monitor.emit_health_span_now()

        assert health is not None
        emitter.assert_called_once()
        call_args = emitter.call_args[0][0]
        assert call_args["agent_name"] == "observability_connector"
        assert call_args["operation"] == "health_check"

    @pytest.mark.asyncio
    async def test_emit_health_span_disabled(self):
        """Test health span emission when disabled."""
        config = SelfMonitorConfig(emit_health_spans=False)
        monitor = SelfMonitor(config)
        emitter = AsyncMock()
        monitor.set_span_emitter(emitter)

        await monitor.emit_health_span_now()

        emitter.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_stop_health_emission(self):
        """Test starting and stopping health emission loop."""
        config = SelfMonitorConfig(health_emission_interval_seconds=0.1)
        monitor = SelfMonitor(config)

        await monitor.start_health_emission()
        assert monitor.is_running is True

        await asyncio.sleep(0.05)

        await monitor.stop_health_emission()
        assert monitor.is_running is False

    @pytest.mark.asyncio
    async def test_start_health_emission_already_running(self):
        """Test starting when already running."""
        config = SelfMonitorConfig(health_emission_interval_seconds=60.0)
        monitor = SelfMonitor(config)

        await monitor.start_health_emission()
        await monitor.start_health_emission()  # Should log warning

        await monitor.stop_health_emission()

    @pytest.mark.asyncio
    async def test_health_span_count_increments(self):
        """Test health span count increments on emission."""
        config = SelfMonitorConfig(emit_health_spans=True)
        monitor = SelfMonitor(config)
        emitter = AsyncMock()
        monitor.set_span_emitter(emitter)

        assert monitor._health_span_count == 0

        await monitor.emit_health_span_now()
        assert monitor._health_span_count == 1

        await monitor.emit_health_span_now()
        assert monitor._health_span_count == 2


# ============================================================================
# SINGLETON TESTS
# ============================================================================


class TestSingleton:
    """Tests for singleton access pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_self_monitor()

    def test_get_self_monitor_default(self):
        """Test getting default singleton."""
        monitor = get_self_monitor()
        assert monitor is not None
        assert isinstance(monitor, SelfMonitor)

    def test_get_self_monitor_singleton(self):
        """Test singleton behavior."""
        monitor1 = get_self_monitor()
        monitor2 = get_self_monitor()
        assert monitor1 is monitor2

    def test_get_self_monitor_with_config(self):
        """Test singleton with custom config."""
        config = SelfMonitorConfig(window_size=50)
        monitor = get_self_monitor(config)
        assert monitor.config.window_size == 50

    def test_get_self_monitor_force_new(self):
        """Test forcing new instance."""
        monitor1 = get_self_monitor()
        monitor1.record_span_emission_latency(100.0)

        monitor2 = get_self_monitor(force_new=True)
        stats = monitor2.get_latency_stats(MetricType.SPAN_EMISSION)
        assert stats.count == 0
        assert monitor1 is not monitor2

    def test_reset_self_monitor(self):
        """Test resetting singleton."""
        monitor1 = get_self_monitor()
        reset_self_monitor()
        monitor2 = get_self_monitor()
        assert monitor1 is not monitor2


# ============================================================================
# LATENCY CONTEXT TESTS
# ============================================================================


class TestLatencyContext:
    """Tests for synchronous LatencyContext."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_self_monitor()

    def test_latency_context_records(self):
        """Test context manager records latency."""
        monitor = SelfMonitor()

        with LatencyContext(monitor, MetricType.SPAN_EMISSION) as ctx:
            time.sleep(0.01)  # 10ms

        stats = monitor.get_latency_stats(MetricType.SPAN_EMISSION)
        assert stats.count == 1
        assert stats.last_ms >= 10.0  # At least 10ms
        assert ctx.latency_ms >= 10.0

    def test_latency_context_records_error(self):
        """Test context manager records errors."""
        monitor = SelfMonitor()

        try:
            with LatencyContext(monitor, MetricType.OPIK_API):
                raise ValueError("Test error")
        except ValueError:
            pass

        health = monitor._trackers[MetricType.OPIK_API].get_component_health()
        assert health.error_count == 1
        assert "ValueError" in health.last_error

    def test_latency_context_no_error_recording(self):
        """Test context manager with error recording disabled."""
        monitor = SelfMonitor()

        try:
            with LatencyContext(monitor, MetricType.OPIK_API, record_errors=False):
                raise ValueError("Test error")
        except ValueError:
            pass

        health = monitor._trackers[MetricType.OPIK_API].get_component_health()
        assert health.error_count == 0

    def test_latency_context_does_not_suppress_exceptions(self):
        """Test context manager doesn't suppress exceptions."""
        monitor = SelfMonitor()

        with pytest.raises(ValueError):
            with LatencyContext(monitor, MetricType.SPAN_EMISSION):
                raise ValueError("Test")


# ============================================================================
# ASYNC LATENCY CONTEXT TESTS
# ============================================================================


class TestAsyncLatencyContext:
    """Tests for asynchronous AsyncLatencyContext."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_self_monitor()

    @pytest.mark.asyncio
    async def test_async_latency_context_records(self):
        """Test async context manager records latency."""
        monitor = SelfMonitor()

        async with AsyncLatencyContext(monitor, MetricType.DATABASE_WRITE) as ctx:
            await asyncio.sleep(0.01)  # 10ms

        stats = monitor.get_latency_stats(MetricType.DATABASE_WRITE)
        assert stats.count == 1
        assert stats.last_ms >= 10.0
        assert ctx.latency_ms >= 10.0

    @pytest.mark.asyncio
    async def test_async_latency_context_records_error(self):
        """Test async context manager records errors."""
        monitor = SelfMonitor()

        try:
            async with AsyncLatencyContext(monitor, MetricType.CACHE_OPERATION):
                raise RuntimeError("Async error")
        except RuntimeError:
            pass

        health = monitor._trackers[MetricType.CACHE_OPERATION].get_component_health()
        assert health.error_count == 1
        assert "RuntimeError" in health.last_error

    @pytest.mark.asyncio
    async def test_async_latency_context_does_not_suppress_exceptions(self):
        """Test async context manager doesn't suppress exceptions."""
        monitor = SelfMonitor()

        with pytest.raises(RuntimeError):
            async with AsyncLatencyContext(monitor, MetricType.BATCH_FLUSH):
                raise RuntimeError("Test")


# ============================================================================
# COMPONENT HEALTH TESTS
# ============================================================================


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = LatencyStats(
            count=5,
            avg_ms=25.0,
            status=HealthStatus.HEALTHY,
        )
        health = ComponentHealth(
            component="test_component",
            status=HealthStatus.HEALTHY,
            latency_stats=stats,
            error_count=2,
            last_error="Test error",
            last_error_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        result = health.to_dict()
        assert result["component"] == "test_component"
        assert result["status"] == "healthy"
        assert result["error_count"] == 2
        assert result["last_error"] == "Test error"
        assert result["latency"]["count"] == 5


# ============================================================================
# OVERALL HEALTH TESTS
# ============================================================================


class TestOverallHealth:
    """Tests for OverallHealth dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = LatencyStats(count=1, status=HealthStatus.HEALTHY)
        component = ComponentHealth(
            component="test",
            status=HealthStatus.HEALTHY,
            latency_stats=stats,
        )
        health = OverallHealth(
            status=HealthStatus.HEALTHY,
            components={"test": component},
            uptime_seconds=100.5,
            last_health_check=datetime(2025, 1, 1, tzinfo=timezone.utc),
            health_span_count=5,
            alerts=["Test alert"],
        )

        result = health.to_dict()
        assert result["status"] == "healthy"
        assert result["uptime_seconds"] == 100.5
        assert result["health_span_count"] == 5
        assert result["alerts"] == ["Test alert"]
        assert "test" in result["components"]


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for self-monitoring."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_self_monitor()

    def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        config = SelfMonitorConfig(
            min_samples_for_stats=3,
            thresholds={
                MetricType.SPAN_EMISSION: LatencyThresholds(
                    warning_ms=50.0, critical_ms=100.0
                ),
                MetricType.OPIK_API: LatencyThresholds(
                    warning_ms=100.0, critical_ms=500.0
                ),
                MetricType.DATABASE_WRITE: LatencyThresholds(
                    warning_ms=25.0, critical_ms=100.0
                ),
                MetricType.BATCH_FLUSH: LatencyThresholds(),
                MetricType.CACHE_OPERATION: LatencyThresholds(),
            },
        )
        monitor = SelfMonitor(config)

        # Simulate normal operation
        for _ in range(10):
            monitor.record_span_emission_latency(15.0)
            monitor.record_opik_latency(50.0)
            monitor.record_database_latency(10.0)

        health = monitor.get_health_status()
        assert health.status == HealthStatus.HEALTHY

        # Simulate degradation
        for _ in range(10):
            monitor.record_opik_latency(300.0)

        health = monitor.get_health_status()
        assert health.status == HealthStatus.DEGRADED
        assert any("opik_api" in alert for alert in health.alerts)

    def test_monitoring_with_errors_and_latency(self):
        """Test monitoring with mixed errors and latency."""
        monitor = SelfMonitor()

        # Record latencies
        for _ in range(5):
            monitor.record_span_emission_latency(20.0)

        # Record errors
        monitor.record_span_emission_error("Connection lost")
        monitor.record_span_emission_error("Timeout")

        health = monitor.get_health_status()
        component = health.components["span_emission"]
        assert component.latency_stats.count == 5
        assert component.error_count == 2

    @pytest.mark.asyncio
    async def test_async_context_integration(self):
        """Test async context with real async operations."""
        monitor = SelfMonitor()

        async def simulated_db_write():
            await asyncio.sleep(0.005)  # 5ms
            return True

        async with AsyncLatencyContext(monitor, MetricType.DATABASE_WRITE):
            result = await simulated_db_write()

        stats = monitor.get_latency_stats(MetricType.DATABASE_WRITE)
        assert stats.count == 1
        assert stats.last_ms >= 5.0
        assert result is True
