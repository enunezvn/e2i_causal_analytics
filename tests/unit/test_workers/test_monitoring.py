"""Unit tests for Celery Queue Monitoring module.

Tests cover:
- QueueMetrics dataclass initialization
- CeleryQueueMonitor queue depth tracking
- Worker statistics collection
- AutoscalerMetricsProvider recommendations
- Convenience functions

G12/G19 from observability audit remediation plan.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.workers.monitoring import (
    AutoscalerMetricsProvider,
    AutoscalerRecommendation,
    CeleryQueueMonitor,
    QueueMetrics,
    get_monitoring_summary,
    get_queue_depths,
    get_queue_monitor,
    queue_metrics,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_celery_app():
    """Mock Celery application."""
    mock = MagicMock()
    mock.conf.broker_url = "redis://localhost:6379/0"
    mock.control.inspect.return_value = MagicMock(
        ping=MagicMock(return_value={"worker-1": {"ok": "pong"}}),
        active=MagicMock(return_value={}),
        reserved=MagicMock(return_value={}),
        scheduled=MagicMock(return_value={}),
    )
    return mock


@pytest.fixture
def mock_prometheus_registry():
    """Mock Prometheus registry."""
    return MagicMock()


@pytest.fixture
def monitor(mock_celery_app, mock_prometheus_registry):
    """Create CeleryQueueMonitor instance."""
    with patch("src.workers.monitoring.PROMETHEUS_AVAILABLE", False):
        return CeleryQueueMonitor(mock_celery_app, registry=mock_prometheus_registry)


# =============================================================================
# QUEUE METRICS TESTS
# =============================================================================


class TestQueueMetrics:
    """Tests for QueueMetrics dataclass."""

    def test_metrics_default_not_initialized(self):
        """Test metrics are not initialized by default."""
        metrics = QueueMetrics()
        assert metrics._initialized is False
        assert metrics.queue_length is None

    def test_metrics_initialize_without_prometheus(self):
        """Test metrics initialization when Prometheus unavailable."""
        metrics = QueueMetrics()
        with patch("src.workers.monitoring.PROMETHEUS_AVAILABLE", False):
            metrics.initialize()
            assert metrics._initialized is False

    def test_metrics_double_initialization_is_idempotent(self):
        """Test that double initialization does nothing."""
        metrics = QueueMetrics()
        metrics._initialized = True
        metrics.initialize()
        # Should not raise


# =============================================================================
# CELERY QUEUE MONITOR TESTS
# =============================================================================


class TestCeleryQueueMonitor:
    """Tests for CeleryQueueMonitor class."""

    def test_monitor_creation(self, monitor):
        """Test monitor can be created."""
        assert monitor is not None
        assert isinstance(monitor, CeleryQueueMonitor)

    def test_queue_names_defined(self, monitor):
        """Test QUEUE_NAMES is defined."""
        assert len(monitor.QUEUE_NAMES) > 0
        assert "default" in monitor.QUEUE_NAMES

    def test_queue_tiers_defined(self, monitor):
        """Test QUEUE_TIERS is defined."""
        assert "light" in monitor.QUEUE_TIERS
        assert "medium" in monitor.QUEUE_TIERS
        assert "heavy" in monitor.QUEUE_TIERS


class TestGetQueueDepths:
    """Tests for get_queue_depths method."""

    def test_get_queue_depths_returns_dict(self, monitor):
        """Test get_queue_depths returns dictionary."""
        with patch.object(monitor, "_get_redis_client", return_value=None):
            depths = monitor.get_queue_depths()
            assert isinstance(depths, dict)

    def test_get_queue_depths_includes_all_queues(self, monitor):
        """Test get_queue_depths includes all monitored queues."""
        with patch.object(monitor, "_get_redis_client", return_value=None):
            depths = monitor.get_queue_depths()
            for queue_name in monitor.QUEUE_NAMES:
                assert queue_name in depths

    def test_get_queue_depths_via_redis(self, monitor):
        """Test get_queue_depths via Redis client."""
        mock_redis = MagicMock()
        mock_redis.llen.return_value = 5
        with patch.object(monitor, "_get_redis_client", return_value=mock_redis):
            depths = monitor.get_queue_depths()
            assert all(d == 5 for d in depths.values())

    def test_get_queue_depths_redis_failure_fallback(self, monitor):
        """Test get_queue_depths falls back on Redis failure."""
        mock_redis = MagicMock()
        mock_redis.llen.side_effect = Exception("Redis connection failed")
        with patch.object(monitor, "_get_redis_client", return_value=mock_redis):
            depths = monitor.get_queue_depths()
            assert isinstance(depths, dict)


class TestGetQueueDepthsByTier:
    """Tests for get_queue_depths_by_tier method."""

    def test_get_depths_by_tier_returns_dict(self, monitor):
        """Test get_queue_depths_by_tier returns dictionary."""
        with patch.object(monitor, "get_queue_depths", return_value={"default": 5, "quick": 3}):
            tier_depths = monitor.get_queue_depths_by_tier()
            assert isinstance(tier_depths, dict)

    def test_get_depths_by_tier_aggregates_correctly(self, monitor):
        """Test tier depths aggregate queue depths correctly."""
        mock_depths = {
            "default": 1,
            "quick": 2,
            "api": 3,
            "analytics": 4,
            "reports": 5,
            "aggregations": 6,
            "shap": 7,
            "causal": 8,
            "ml": 9,
            "twins": 10,
        }
        with patch.object(monitor, "get_queue_depths", return_value=mock_depths):
            tier_depths = monitor.get_queue_depths_by_tier()
            # light tier: default + quick + api = 1 + 2 + 3 = 6
            assert tier_depths["light"] == 6
            # medium tier: analytics + reports + aggregations = 4 + 5 + 6 = 15
            assert tier_depths["medium"] == 15
            # heavy tier: shap + causal + ml + twins = 7 + 8 + 9 + 10 = 34
            assert tier_depths["heavy"] == 34


class TestGetWorkerStats:
    """Tests for get_worker_stats method."""

    def test_get_worker_stats_returns_dict(self, monitor):
        """Test get_worker_stats returns dictionary."""
        stats = monitor.get_worker_stats()
        assert isinstance(stats, dict)
        assert "total_workers" in stats
        assert "workers_by_tier" in stats
        assert "workers" in stats

    def test_get_worker_stats_counts_workers(self, monitor, mock_celery_app):
        """Test get_worker_stats counts workers correctly."""
        mock_celery_app.control.inspect.return_value.ping.return_value = {
            "worker-light-1": {"ok": "pong"},
            "worker-medium-1": {"ok": "pong"},
        }
        stats = monitor.get_worker_stats()
        assert stats["total_workers"] == 2

    def test_get_worker_stats_groups_by_tier(self, monitor, mock_celery_app):
        """Test get_worker_stats groups workers by tier."""
        mock_celery_app.control.inspect.return_value.ping.return_value = {
            "worker-light-1": {"ok": "pong"},
            "worker-heavy-1": {"ok": "pong"},
        }
        stats = monitor.get_worker_stats()
        assert stats["workers_by_tier"]["light"] >= 0
        assert stats["workers_by_tier"]["heavy"] >= 0

    def test_get_worker_stats_handles_inspect_failure(self, monitor, mock_celery_app):
        """Test get_worker_stats handles inspect failure gracefully."""
        mock_celery_app.control.inspect.return_value.ping.side_effect = Exception(
            "Connection failed"
        )
        stats = monitor.get_worker_stats()
        assert stats["total_workers"] == 0


class TestGetScheduledTaskCount:
    """Tests for get_scheduled_task_count method."""

    def test_get_scheduled_count_returns_int(self, monitor):
        """Test get_scheduled_task_count returns integer."""
        count = monitor.get_scheduled_task_count()
        assert isinstance(count, int)

    def test_get_scheduled_count_sums_workers(self, monitor, mock_celery_app):
        """Test get_scheduled_task_count sums across workers."""
        mock_celery_app.control.inspect.return_value.scheduled.return_value = {
            "worker-1": [{"task": "task1"}, {"task": "task2"}],
            "worker-2": [{"task": "task3"}],
        }
        count = monitor.get_scheduled_task_count()
        assert count == 3


class TestGetReservedTasks:
    """Tests for get_reserved_tasks method."""

    def test_get_reserved_tasks_returns_dict(self, monitor):
        """Test get_reserved_tasks returns dictionary."""
        reserved = monitor.get_reserved_tasks()
        assert isinstance(reserved, dict)

    def test_get_reserved_tasks_counts_per_worker(self, monitor, mock_celery_app):
        """Test get_reserved_tasks counts tasks per worker."""
        mock_celery_app.control.inspect.return_value.reserved.return_value = {
            "worker-1": [{"task": "task1"}, {"task": "task2"}],
            "worker-2": [{"task": "task3"}],
        }
        reserved = monitor.get_reserved_tasks()
        assert reserved["worker-1"] == 2
        assert reserved["worker-2"] == 1


class TestUpdateMetrics:
    """Tests for update_metrics method."""

    def test_update_metrics_without_initialization(self, monitor):
        """Test update_metrics when metrics not initialized."""
        # Should not raise
        monitor.update_metrics()


class TestGetMonitoringSummary:
    """Tests for get_monitoring_summary method."""

    def test_get_summary_returns_dict(self, monitor):
        """Test get_monitoring_summary returns dictionary."""
        with patch.object(monitor, "get_queue_depths", return_value={"default": 0}):
            with patch.object(monitor, "get_queue_depths_by_tier", return_value={"light": 0}):
                with patch.object(
                    monitor,
                    "get_worker_stats",
                    return_value={"total_workers": 0, "workers_by_tier": {}, "workers": []},
                ):
                    summary = monitor.get_monitoring_summary()
                    assert isinstance(summary, dict)

    def test_get_summary_includes_expected_keys(self, monitor):
        """Test get_monitoring_summary includes expected keys."""
        with patch.object(monitor, "get_queue_depths", return_value={"default": 0}):
            with patch.object(monitor, "get_queue_depths_by_tier", return_value={"light": 0}):
                with patch.object(
                    monitor,
                    "get_worker_stats",
                    return_value={"total_workers": 0, "workers_by_tier": {}, "workers": []},
                ):
                    summary = monitor.get_monitoring_summary()
                    assert "timestamp" in summary
                    assert "queues" in summary
                    assert "workers" in summary
                    assert "health" in summary


# =============================================================================
# AUTOSCALER RECOMMENDATION TESTS
# =============================================================================


class TestAutoscalerRecommendation:
    """Tests for AutoscalerRecommendation dataclass."""

    def test_recommendation_creation(self):
        """Test AutoscalerRecommendation creation."""
        rec = AutoscalerRecommendation(
            tier="light",
            current_workers=2,
            recommended_workers=3,
            reason="High load",
            queue_depth=25,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        assert rec.tier == "light"
        assert rec.current_workers == 2
        assert rec.recommended_workers == 3


class TestAutoscalerMetricsProvider:
    """Tests for AutoscalerMetricsProvider class."""

    def test_provider_creation(self, mock_celery_app):
        """Test AutoscalerMetricsProvider creation."""
        with patch("src.workers.monitoring.PROMETHEUS_AVAILABLE", False):
            provider = AutoscalerMetricsProvider(mock_celery_app)
            assert provider is not None

    def test_get_scaling_recommendations_returns_list(self, mock_celery_app):
        """Test get_scaling_recommendations returns list."""
        with patch("src.workers.monitoring.PROMETHEUS_AVAILABLE", False):
            provider = AutoscalerMetricsProvider(mock_celery_app)
            with patch.object(
                provider.monitor,
                "get_queue_depths_by_tier",
                return_value={"light": 5, "medium": 10, "heavy": 0},
            ):
                with patch.object(
                    provider.monitor,
                    "get_worker_stats",
                    return_value={"workers_by_tier": {"light": 1, "medium": 2, "heavy": 0}},
                ):
                    recs = provider.get_scaling_recommendations()
                    assert isinstance(recs, list)
                    assert len(recs) == 3  # One per tier

    def test_scale_up_recommendation(self, mock_celery_app):
        """Test scale up recommendation for high load."""
        with patch("src.workers.monitoring.PROMETHEUS_AVAILABLE", False):
            provider = AutoscalerMetricsProvider(mock_celery_app)
            # High load: 50 tasks / 1 worker = 50 tasks per worker
            with patch.object(
                provider.monitor,
                "get_queue_depths_by_tier",
                return_value={"light": 50, "medium": 0, "heavy": 0},
            ):
                with patch.object(
                    provider.monitor,
                    "get_worker_stats",
                    return_value={"workers_by_tier": {"light": 1, "medium": 0, "heavy": 0}},
                ):
                    recs = provider.get_scaling_recommendations()
                    light_rec = next(r for r in recs if r.tier == "light")
                    assert light_rec.recommended_workers > light_rec.current_workers

    def test_scale_down_recommendation(self, mock_celery_app):
        """Test scale down recommendation for low load."""
        with patch("src.workers.monitoring.PROMETHEUS_AVAILABLE", False):
            provider = AutoscalerMetricsProvider(mock_celery_app)
            # Low load: 1 task / 4 workers = 0.25 tasks per worker
            with patch.object(
                provider.monitor,
                "get_queue_depths_by_tier",
                return_value={"light": 1, "medium": 0, "heavy": 0},
            ):
                with patch.object(
                    provider.monitor,
                    "get_worker_stats",
                    return_value={"workers_by_tier": {"light": 4, "medium": 0, "heavy": 0}},
                ):
                    recs = provider.get_scaling_recommendations()
                    light_rec = next(r for r in recs if r.tier == "light")
                    assert light_rec.recommended_workers <= light_rec.current_workers


# =============================================================================
# CONVENIENCE FUNCTIONS TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_queue_monitor(self, mock_celery_app):
        """Test get_queue_monitor function."""
        with patch("src.workers.monitoring.PROMETHEUS_AVAILABLE", False):
            monitor = get_queue_monitor(mock_celery_app)
            assert isinstance(monitor, CeleryQueueMonitor)

    def test_get_queue_depths_function(self, mock_celery_app):
        """Test get_queue_depths convenience function."""
        with patch("src.workers.monitoring.PROMETHEUS_AVAILABLE", False):
            with patch(
                "src.workers.monitoring.CeleryQueueMonitor.get_queue_depths",
                return_value={"default": 0},
            ):
                depths = get_queue_depths(mock_celery_app)
                assert isinstance(depths, dict)

    def test_get_monitoring_summary_function(self, mock_celery_app):
        """Test get_monitoring_summary convenience function."""
        with patch("src.workers.monitoring.PROMETHEUS_AVAILABLE", False):
            with patch(
                "src.workers.monitoring.CeleryQueueMonitor.get_monitoring_summary", return_value={}
            ):
                summary = get_monitoring_summary(mock_celery_app)
                assert isinstance(summary, dict)


# =============================================================================
# GLOBAL METRICS TESTS
# =============================================================================


class TestGlobalMetrics:
    """Tests for global metrics instance."""

    def test_global_queue_metrics_exists(self):
        """Test global queue_metrics instance exists."""
        assert queue_metrics is not None
        assert isinstance(queue_metrics, QueueMetrics)
