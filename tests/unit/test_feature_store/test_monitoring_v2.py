"""
Comprehensive unit tests for src/feature_store/monitoring.py

Tests cover:
- Metrics initialization
- Latency tracking context managers
- Metrics recording functions
- LatencyStats dataclass
- LatencyTracker class
- Decorator instrumentation
"""

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.feature_store.monitoring import (
    FeatureRetrievalMetrics,
    LatencyStats,
    LatencyTracker,
    get_latency_tracker,
    init_feature_store_metrics,
    record_batch_size,
    record_cache_hit,
    record_cache_miss,
    record_error,
    track_cache_operation,
    track_db_operation,
    track_retrieval,
)


# =============================================================================
# Test Metrics Initialization
# =============================================================================


@patch("src.feature_store.monitoring.PROMETHEUS_AVAILABLE", True)
@patch("src.feature_store.monitoring.Counter")
@patch("src.feature_store.monitoring.Histogram")
@patch("src.feature_store.monitoring.Summary")
def test_init_feature_store_metrics(mock_summary, mock_histogram, mock_counter):
    """Test metrics initialization."""
    # Reset global state
    import src.feature_store.monitoring as mon

    mon._fs_metrics_initialized = False

    init_feature_store_metrics()

    # Verify metrics were created
    assert mock_counter.call_count >= 2  # cache_hits, cache_misses
    assert mock_histogram.call_count >= 3  # retrieval, cache, db latency
    assert mock_summary.call_count >= 1  # feature_count


@patch("src.feature_store.monitoring.PROMETHEUS_AVAILABLE", False)
def test_init_feature_store_metrics_no_prometheus():
    """Test metrics initialization when Prometheus is not available."""
    import src.feature_store.monitoring as mon

    mon._fs_metrics_initialized = False

    # Should not raise
    init_feature_store_metrics()

    assert mon._fs_metrics_initialized is False


# =============================================================================
# Test FeatureRetrievalMetrics Dataclass
# =============================================================================


def test_feature_retrieval_metrics_creation():
    """Test FeatureRetrievalMetrics creation."""
    metrics = FeatureRetrievalMetrics(
        feature_group="test_group",
        operation="get_entity_features",
        cache_hit=True,
        cache_latency_ms=5.2,
        db_latency_ms=15.3,
        total_latency_ms=20.5,
        feature_count=10,
    )

    assert metrics.feature_group == "test_group"
    assert metrics.cache_hit is True
    assert metrics.feature_count == 10
    assert metrics.error is None


# =============================================================================
# Test Context Managers
# =============================================================================


@pytest.mark.asyncio
async def test_track_retrieval_context_manager():
    """Test track_retrieval context manager."""
    with track_retrieval("test_group", "get_entity_features") as metrics:
        time.sleep(0.001)  # Simulate work
        metrics.cache_hit = True
        metrics.feature_count = 5

    assert metrics.total_latency_ms > 0
    assert metrics.cache_hit is True
    assert metrics.feature_count == 5


@pytest.mark.asyncio
async def test_track_retrieval_with_error():
    """Test track_retrieval handles errors."""
    with pytest.raises(ValueError):
        with track_retrieval("test_group") as metrics:
            raise ValueError("Test error")

    # Metrics should still be recorded even with error


@pytest.mark.asyncio
async def test_track_cache_operation():
    """Test track_cache_operation context manager."""
    with track_cache_operation("get") as context:
        time.sleep(0.001)
        context["hit"] = True

    assert context["latency_ms"] > 0
    assert context["hit"] is True


@pytest.mark.asyncio
async def test_track_db_operation():
    """Test track_db_operation context manager."""
    with track_db_operation("query", "test_group") as context:
        time.sleep(0.001)
        context["row_count"] = 100

    assert context["latency_ms"] > 0
    assert context["row_count"] == 100


# =============================================================================
# Test Metrics Recording Functions
# =============================================================================


@patch("src.feature_store.monitoring.PROMETHEUS_AVAILABLE", True)
@patch("src.feature_store.monitoring._fs_cache_hits")
def test_record_cache_hit(mock_cache_hits):
    """Test recording cache hit."""
    mock_metric = MagicMock()
    mock_cache_hits.labels.return_value = mock_metric

    record_cache_hit("test_group", "get_entity_features")

    mock_cache_hits.labels.assert_called_once_with(
        feature_group="test_group",
        operation="get_entity_features",
    )
    mock_metric.inc.assert_called_once()


@patch("src.feature_store.monitoring.PROMETHEUS_AVAILABLE", True)
@patch("src.feature_store.monitoring._fs_cache_misses")
def test_record_cache_miss(mock_cache_misses):
    """Test recording cache miss."""
    mock_metric = MagicMock()
    mock_cache_misses.labels.return_value = mock_metric

    record_cache_miss("test_group", "get_entity_features")

    mock_cache_misses.labels.assert_called_once_with(
        feature_group="test_group",
        operation="get_entity_features",
    )
    mock_metric.inc.assert_called_once()


@patch("src.feature_store.monitoring.PROMETHEUS_AVAILABLE", True)
@patch("src.feature_store.monitoring._fs_batch_size")
def test_record_batch_size(mock_batch_size):
    """Test recording batch size."""
    mock_metric = MagicMock()
    mock_batch_size.labels.return_value = mock_metric

    record_batch_size("write_batch", 100)

    mock_batch_size.labels.assert_called_once_with(operation="write_batch")
    mock_metric.observe.assert_called_once_with(100)


@patch("src.feature_store.monitoring.PROMETHEUS_AVAILABLE", True)
@patch("src.feature_store.monitoring._fs_errors")
def test_record_error(mock_errors):
    """Test recording error."""
    mock_metric = MagicMock()
    mock_errors.labels.return_value = mock_metric

    record_error("get_entity_features", "TimeoutError")

    mock_errors.labels.assert_called_once_with(
        operation="get_entity_features",
        error_type="TimeoutError",
    )
    mock_metric.inc.assert_called_once()


@patch("src.feature_store.monitoring.PROMETHEUS_AVAILABLE", False)
def test_record_functions_no_prometheus():
    """Test record functions when Prometheus is not available."""
    # Should not raise
    record_cache_hit("test_group")
    record_cache_miss("test_group")
    record_batch_size("write", 10)
    record_error("operation", "error")


# =============================================================================
# Test LatencyStats Dataclass
# =============================================================================


def test_latency_stats_creation():
    """Test LatencyStats creation."""
    stats = LatencyStats(
        count=100,
        total_ms=1000.0,
        min_ms=5.0,
        max_ms=50.0,
        avg_ms=10.0,
        p50_ms=9.0,
        p95_ms=25.0,
        p99_ms=45.0,
        cache_hit_rate=0.75,
    )

    assert stats.count == 100
    assert stats.avg_ms == 10.0
    assert stats.cache_hit_rate == 0.75


# =============================================================================
# Test LatencyTracker Class
# =============================================================================


def test_latency_tracker_creation():
    """Test LatencyTracker initialization."""
    tracker = LatencyTracker(max_samples=500)

    assert tracker.max_samples == 500
    assert len(tracker.samples) == 0
    assert tracker._total_cache_hits == 0
    assert tracker._total_cache_misses == 0


def test_latency_tracker_record():
    """Test recording metrics in tracker."""
    tracker = LatencyTracker()

    metrics = FeatureRetrievalMetrics(
        feature_group="test_group",
        operation="get_entity_features",
        cache_hit=True,
        total_latency_ms=10.5,
        feature_count=5,
    )

    tracker.record(metrics)

    assert len(tracker.samples) == 1
    assert tracker._total_cache_hits == 1
    assert tracker._total_cache_misses == 0


def test_latency_tracker_circular_buffer():
    """Test tracker maintains max samples."""
    tracker = LatencyTracker(max_samples=100)

    # Add 150 samples
    for i in range(150):
        metrics = FeatureRetrievalMetrics(
            feature_group="test",
            total_latency_ms=float(i),
        )
        tracker.record(metrics)

    # Should only keep last 100
    assert len(tracker.samples) == 100
    assert tracker.samples[0].total_latency_ms == 50.0  # First of last 100


def test_latency_tracker_get_stats():
    """Test getting statistics from tracker."""
    tracker = LatencyTracker()

    # Add samples
    for i in range(10):
        metrics = FeatureRetrievalMetrics(
            feature_group="test_group",
            total_latency_ms=float(i * 10),
            cache_hit=(i % 2 == 0),  # 50% hit rate
        )
        tracker.record(metrics)

    stats = tracker.get_stats()

    assert stats.count == 10
    assert stats.min_ms == 0.0
    assert stats.max_ms == 90.0
    assert stats.avg_ms == 45.0
    assert stats.cache_hit_rate == 0.5


def test_latency_tracker_get_stats_filtered():
    """Test getting statistics with filter."""
    tracker = LatencyTracker()

    # Add samples for different groups
    for i in range(5):
        tracker.record(
            FeatureRetrievalMetrics(
                feature_group="group1",
                total_latency_ms=10.0,
            )
        )
        tracker.record(
            FeatureRetrievalMetrics(
                feature_group="group2",
                total_latency_ms=20.0,
            )
        )

    stats = tracker.get_stats(feature_group="group1")

    assert stats.count == 5
    assert stats.avg_ms == 10.0


def test_latency_tracker_get_stats_empty():
    """Test getting statistics when no samples."""
    tracker = LatencyTracker()

    stats = tracker.get_stats()

    assert stats.count == 0


def test_latency_tracker_get_recent():
    """Test getting recent samples."""
    tracker = LatencyTracker()

    for i in range(10):
        tracker.record(
            FeatureRetrievalMetrics(
                feature_group="test",
                total_latency_ms=float(i),
            )
        )

    recent = tracker.get_recent(count=3)

    assert len(recent) == 3
    assert recent[-1].total_latency_ms == 9.0  # Most recent


def test_latency_tracker_clear():
    """Test clearing tracker."""
    tracker = LatencyTracker()

    tracker.record(FeatureRetrievalMetrics(feature_group="test"))
    tracker.record(FeatureRetrievalMetrics(feature_group="test", cache_hit=True))

    assert len(tracker.samples) == 2
    assert tracker._total_cache_hits == 1

    tracker.clear()

    assert len(tracker.samples) == 0
    assert tracker._total_cache_hits == 0
    assert tracker._total_cache_misses == 0


# =============================================================================
# Test Singleton
# =============================================================================


def test_get_latency_tracker_singleton():
    """Test get_latency_tracker returns singleton."""
    tracker1 = get_latency_tracker()
    tracker2 = get_latency_tracker()

    assert tracker1 is tracker2


# =============================================================================
# Test Decorator
# =============================================================================


@patch("src.feature_store.monitoring.track_retrieval")
def test_track_feature_retrieval_decorator(mock_track):
    """Test track_feature_retrieval decorator."""
    from src.feature_store.monitoring import track_feature_retrieval

    # Mock the context manager
    mock_context = MagicMock()
    mock_context.__enter__ = MagicMock()
    mock_context.__exit__ = MagicMock()
    mock_track.return_value = mock_context

    @track_feature_retrieval(feature_group_arg="feature_group")
    def test_function(entity_values, feature_group=None):
        return {"features": [1, 2, 3]}

    result = test_function({"id": "123"}, feature_group="test_group")

    assert result == {"features": [1, 2, 3]}


# =============================================================================
# Test Percentile Calculation
# =============================================================================


def test_latency_stats_percentiles():
    """Test percentile calculations in LatencyStats."""
    tracker = LatencyTracker()

    # Add 100 samples with known distribution
    for i in range(100):
        tracker.record(
            FeatureRetrievalMetrics(
                feature_group="test",
                total_latency_ms=float(i),
            )
        )

    stats = tracker.get_stats()

    assert stats.count == 100
    assert stats.p50_ms == 50.0
    assert stats.p95_ms == 95.0
    assert stats.p99_ms == 99.0


def test_latency_stats_small_sample():
    """Test percentiles with small sample size."""
    tracker = LatencyTracker()

    # Add only 10 samples
    for i in range(10):
        tracker.record(
            FeatureRetrievalMetrics(
                feature_group="test",
                total_latency_ms=float(i),
            )
        )

    stats = tracker.get_stats()

    # Should still calculate what it can
    assert stats.count == 10
    assert stats.p50_ms >= 0
    assert stats.p95_ms >= stats.p50_ms
