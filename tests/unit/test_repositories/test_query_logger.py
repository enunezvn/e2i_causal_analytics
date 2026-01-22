"""Unit tests for Database Query Logger module.

Tests cover:
- QueryMetrics dataclass initialization
- SlowQueryConfig and SlowQueryDetector
- QueryLogger execution and tracking
- Context managers for query tracking
- Decorators for repository methods

G13 from observability audit remediation plan.
"""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.repositories.query_logger import (
    QueryMetrics,
    SlowQueryRecord,
    SlowQueryConfig,
    SlowQueryDetector,
    QueryLogger,
    query_metrics,
    query_logger,
    slow_query_detector,
    logged_query,
    logged_query_async,
    get_query_stats,
    configure_slow_query_thresholds,
    PROMETHEUS_AVAILABLE,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_prometheus_registry():
    """Mock Prometheus registry."""
    return MagicMock()


@pytest.fixture
def slow_config():
    """Create SlowQueryConfig for testing."""
    return SlowQueryConfig(
        default_threshold_sec=0.5,
        log_slow_queries=False,  # Quiet for tests
    )


@pytest.fixture
def detector(slow_config):
    """Create SlowQueryDetector for testing."""
    return SlowQueryDetector(config=slow_config)


@pytest.fixture
def logger_instance(detector, mock_prometheus_registry):
    """Create QueryLogger instance."""
    with patch("src.repositories.query_logger.PROMETHEUS_AVAILABLE", False):
        return QueryLogger(slow_query_detector=detector, registry=mock_prometheus_registry)


# =============================================================================
# QUERY METRICS TESTS
# =============================================================================


class TestQueryMetrics:
    """Tests for QueryMetrics dataclass."""

    def test_metrics_default_not_initialized(self):
        """Test metrics are not initialized by default."""
        metrics = QueryMetrics()
        assert metrics._initialized is False
        assert metrics.query_duration is None

    def test_metrics_initialize_without_prometheus(self):
        """Test metrics initialization when Prometheus unavailable."""
        metrics = QueryMetrics()
        with patch("src.repositories.query_logger.PROMETHEUS_AVAILABLE", False):
            metrics.initialize()
            assert metrics._initialized is False

    def test_metrics_double_initialization_is_idempotent(self):
        """Test that double initialization does nothing."""
        metrics = QueryMetrics()
        metrics._initialized = True
        metrics.initialize()
        # Should not raise


# =============================================================================
# SLOW QUERY RECORD TESTS
# =============================================================================


class TestSlowQueryRecord:
    """Tests for SlowQueryRecord dataclass."""

    def test_record_creation(self):
        """Test SlowQueryRecord creation."""
        record = SlowQueryRecord(
            operation="select",
            table="kpi_values",
            duration_ms=1500.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert record.operation == "select"
        assert record.table == "kpi_values"
        assert record.duration_ms == 1500.0

    def test_record_with_params(self):
        """Test SlowQueryRecord with query params."""
        record = SlowQueryRecord(
            operation="select",
            table="kpi_values",
            duration_ms=1500.0,
            timestamp=datetime.now(timezone.utc),
            query_params={"kpi_id": "abc123"},
        )
        assert record.query_params == {"kpi_id": "abc123"}


# =============================================================================
# SLOW QUERY CONFIG TESTS
# =============================================================================


class TestSlowQueryConfig:
    """Tests for SlowQueryConfig dataclass."""

    def test_config_default_values(self):
        """Test SlowQueryConfig default values."""
        config = SlowQueryConfig()
        assert config.default_threshold_sec == 1.0
        assert config.log_slow_queries is True
        assert config.include_params_in_logs is False
        assert config.max_retained_queries == 100

    def test_config_operation_thresholds(self):
        """Test operation-specific thresholds."""
        config = SlowQueryConfig()
        assert "select" in config.operation_thresholds
        assert "insert" in config.operation_thresholds
        assert config.operation_thresholds["rpc"] == 2.0

    def test_get_threshold_default(self):
        """Test get_threshold returns default."""
        config = SlowQueryConfig()
        threshold = config.get_threshold("unknown_op", "unknown_table")
        assert threshold == config.default_threshold_sec

    def test_get_threshold_operation_specific(self):
        """Test get_threshold returns operation-specific threshold."""
        config = SlowQueryConfig()
        threshold = config.get_threshold("select", "any_table")
        assert threshold == config.operation_thresholds["select"]

    def test_get_threshold_table_specific(self):
        """Test get_threshold returns table-specific threshold."""
        config = SlowQueryConfig()
        config.table_thresholds["large_table"] = 5.0
        threshold = config.get_threshold("select", "large_table")
        assert threshold == 5.0


# =============================================================================
# SLOW QUERY DETECTOR TESTS
# =============================================================================


class TestSlowQueryDetector:
    """Tests for SlowQueryDetector class."""

    def test_detector_creation(self, detector):
        """Test detector can be created."""
        assert detector is not None
        assert isinstance(detector, SlowQueryDetector)

    def test_check_query_fast(self, detector):
        """Test check_query for fast query."""
        record = detector.check_query(
            operation="select",
            table="test_table",
            duration_sec=0.01,  # 10ms - very fast
        )
        assert record is None

    def test_check_query_slow(self, detector):
        """Test check_query for slow query."""
        record = detector.check_query(
            operation="select",
            table="test_table",
            duration_sec=2.0,  # 2s - definitely slow
        )
        assert record is not None
        assert isinstance(record, SlowQueryRecord)
        assert record.duration_ms == 2000.0

    def test_check_query_stores_record(self, detector):
        """Test check_query stores slow query record."""
        detector.check_query(
            operation="select",
            table="test_table",
            duration_sec=2.0,
        )
        assert len(detector._slow_queries) == 1

    def test_check_query_limit_retained(self, slow_config):
        """Test slow queries are limited to max_retained_queries."""
        slow_config.max_retained_queries = 5
        detector = SlowQueryDetector(config=slow_config)

        for i in range(10):
            detector.check_query(
                operation="select",
                table=f"table_{i}",
                duration_sec=2.0,
            )

        assert len(detector._slow_queries) <= 5

    def test_check_query_fires_callback(self, slow_config):
        """Test check_query fires alert callback."""
        callback = MagicMock()
        detector = SlowQueryDetector(config=slow_config, alert_callback=callback)

        detector.check_query(
            operation="select",
            table="test_table",
            duration_sec=2.0,
        )

        callback.assert_called_once()


class TestGetRecentSlowQueries:
    """Tests for get_recent_slow_queries method."""

    def test_get_recent_empty(self, detector):
        """Test get_recent_slow_queries with no queries."""
        queries = detector.get_recent_slow_queries()
        assert queries == []

    def test_get_recent_with_limit(self, detector):
        """Test get_recent_slow_queries with limit."""
        for i in range(5):
            detector.check_query(
                operation="select",
                table=f"table_{i}",
                duration_sec=2.0,
            )

        queries = detector.get_recent_slow_queries(limit=3)
        assert len(queries) == 3

    def test_get_recent_filter_by_operation(self, detector):
        """Test get_recent_slow_queries filter by operation."""
        detector.check_query(operation="select", table="t1", duration_sec=2.0)
        detector.check_query(operation="insert", table="t2", duration_sec=2.0)
        detector.check_query(operation="select", table="t3", duration_sec=2.0)

        queries = detector.get_recent_slow_queries(operation="select")
        assert len(queries) == 2

    def test_get_recent_filter_by_table(self, detector):
        """Test get_recent_slow_queries filter by table."""
        detector.check_query(operation="select", table="t1", duration_sec=2.0)
        detector.check_query(operation="insert", table="t1", duration_sec=2.0)
        detector.check_query(operation="select", table="t2", duration_sec=2.0)

        queries = detector.get_recent_slow_queries(table="t1")
        assert len(queries) == 2


class TestGetSlowQueryStats:
    """Tests for get_slow_query_stats method."""

    def test_get_stats_empty(self, detector):
        """Test get_slow_query_stats with no queries."""
        stats = detector.get_slow_query_stats()
        assert stats["total_slow_queries"] == 0
        assert stats["avg_duration_ms"] == 0

    def test_get_stats_with_queries(self, detector):
        """Test get_slow_query_stats with queries."""
        detector.check_query(operation="select", table="t1", duration_sec=1.0)
        detector.check_query(operation="select", table="t1", duration_sec=2.0)
        detector.check_query(operation="insert", table="t2", duration_sec=1.5)

        stats = detector.get_slow_query_stats()
        assert stats["total_slow_queries"] == 3
        assert stats["by_operation"]["select"] == 2
        assert stats["by_operation"]["insert"] == 1
        assert stats["by_table"]["t1"] == 2
        assert stats["max_duration_ms"] == 2000.0


class TestClear:
    """Tests for clear method."""

    def test_clear_queries(self, detector):
        """Test clear method removes all queries."""
        detector.check_query(operation="select", table="t1", duration_sec=2.0)
        assert len(detector._slow_queries) == 1

        detector.clear()
        assert len(detector._slow_queries) == 0


# =============================================================================
# QUERY LOGGER TESTS
# =============================================================================


class TestQueryLogger:
    """Tests for QueryLogger class."""

    def test_logger_creation(self, logger_instance):
        """Test logger can be created."""
        assert logger_instance is not None
        assert isinstance(logger_instance, QueryLogger)


class TestExecute:
    """Tests for execute method."""

    def test_execute_success(self, logger_instance):
        """Test execute with successful query."""
        result = logger_instance.execute(
            operation="select",
            table="test_table",
            func=lambda: "result",
        )
        assert result == "result"

    def test_execute_error(self, logger_instance):
        """Test execute with failing query."""
        def failing_func():
            raise ValueError("Query failed")

        with pytest.raises(ValueError, match="Query failed"):
            logger_instance.execute(
                operation="select",
                table="test_table",
                func=failing_func,
            )


class TestExecuteAsync:
    """Tests for execute_async method."""

    @pytest.mark.asyncio
    async def test_execute_async_success(self, logger_instance):
        """Test execute_async with successful query."""
        async def async_query():
            return "async_result"

        result = await logger_instance.execute_async(
            operation="select",
            table="test_table",
            func=async_query,
        )
        assert result == "async_result"

    @pytest.mark.asyncio
    async def test_execute_async_error(self, logger_instance):
        """Test execute_async with failing query."""
        async def failing_async_query():
            raise ValueError("Async query failed")

        with pytest.raises(ValueError, match="Async query failed"):
            await logger_instance.execute_async(
                operation="select",
                table="test_table",
                func=failing_async_query,
            )


class TestTrackQuery:
    """Tests for track_query context manager."""

    def test_track_query_success(self, logger_instance):
        """Test track_query context manager on success."""
        with logger_instance.track_query("select", "test_table") as tracker:
            tracker.set_result("result")

        assert tracker.result == "result"
        assert tracker.error is None

    def test_track_query_error(self, logger_instance):
        """Test track_query context manager on error."""
        with pytest.raises(ValueError):
            with logger_instance.track_query("select", "test_table") as tracker:
                raise ValueError("Error")

        assert tracker.error is not None


class TestTrackQueryAsync:
    """Tests for track_query_async context manager."""

    @pytest.mark.asyncio
    async def test_track_query_async_success(self, logger_instance):
        """Test track_query_async context manager on success."""
        async with logger_instance.track_query_async("select", "test_table") as tracker:
            tracker.set_result("async_result")

        assert tracker.result == "async_result"

    @pytest.mark.asyncio
    async def test_track_query_async_error(self, logger_instance):
        """Test track_query_async context manager on error."""
        with pytest.raises(ValueError):
            async with logger_instance.track_query_async("select", "test_table") as tracker:
                raise ValueError("Async error")

        assert tracker.error is not None


# =============================================================================
# DECORATOR TESTS
# =============================================================================


class TestLoggedQueryDecorator:
    """Tests for logged_query decorator."""

    def test_logged_query_wraps_function(self):
        """Test logged_query decorator wraps function."""
        @logged_query("select", "test_table")
        def test_func():
            return "result"

        result = test_func()
        assert result == "result"

    def test_logged_query_preserves_name(self):
        """Test logged_query preserves function name."""
        @logged_query("select", "test_table")
        def test_func():
            return "result"

        assert test_func.__name__ == "test_func"


class TestLoggedQueryAsyncDecorator:
    """Tests for logged_query_async decorator."""

    @pytest.mark.asyncio
    async def test_logged_query_async_wraps_function(self):
        """Test logged_query_async decorator wraps function."""
        @logged_query_async("select", "test_table")
        async def test_async_func():
            return "async_result"

        result = await test_async_func()
        assert result == "async_result"

    def test_logged_query_async_preserves_name(self):
        """Test logged_query_async preserves function name."""
        @logged_query_async("select", "test_table")
        async def test_async_func():
            return "async_result"

        assert test_async_func.__name__ == "test_async_func"


# =============================================================================
# CONVENIENCE FUNCTIONS TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_query_stats_returns_dict(self):
        """Test get_query_stats returns dictionary."""
        stats = get_query_stats()
        assert isinstance(stats, dict)
        assert "slow_queries" in stats
        assert "metrics_initialized" in stats

    def test_configure_slow_query_thresholds(self):
        """Test configure_slow_query_thresholds function."""
        original_threshold = slow_query_detector.config.default_threshold_sec
        try:
            configure_slow_query_thresholds(default_threshold_sec=2.5)
            assert slow_query_detector.config.default_threshold_sec == 2.5
        finally:
            # Restore original
            slow_query_detector.config.default_threshold_sec = original_threshold

    def test_configure_operation_thresholds(self):
        """Test configure operation-specific thresholds."""
        original = slow_query_detector.config.operation_thresholds.get("select")
        try:
            configure_slow_query_thresholds(operation_thresholds={"select": 0.1})
            assert slow_query_detector.config.operation_thresholds["select"] == 0.1
        finally:
            if original:
                slow_query_detector.config.operation_thresholds["select"] = original


# =============================================================================
# GLOBAL INSTANCES TESTS
# =============================================================================


class TestGlobalInstances:
    """Tests for global instances."""

    def test_global_query_metrics_exists(self):
        """Test global query_metrics instance exists."""
        assert query_metrics is not None
        assert isinstance(query_metrics, QueryMetrics)

    def test_global_slow_query_detector_exists(self):
        """Test global slow_query_detector instance exists."""
        assert slow_query_detector is not None
        assert isinstance(slow_query_detector, SlowQueryDetector)

    def test_global_query_logger_exists(self):
        """Test global query_logger instance exists."""
        assert query_logger is not None
        assert isinstance(query_logger, QueryLogger)
