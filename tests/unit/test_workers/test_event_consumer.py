"""Unit tests for Celery Event Consumer module.

Tests cover:
- CeleryMetrics dataclass initialization
- TaskTiming dataclass
- CeleryEventConsumer event handlers
- Trace ID propagation functions
- Traced task context manager

G12 from observability audit remediation plan.
"""

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.workers.event_consumer import (
    CeleryMetrics,
    TaskTiming,
    CeleryEventConsumer,
    inject_trace_context,
    extract_trace_context,
    traced_task,
    celery_metrics,
    PROMETHEUS_AVAILABLE,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_celery_app():
    """Mock Celery application."""
    mock = MagicMock()
    mock.connection.return_value.__enter__ = MagicMock()
    mock.connection.return_value.__exit__ = MagicMock()
    return mock


@pytest.fixture
def mock_prometheus_registry():
    """Mock Prometheus registry."""
    return MagicMock()


@pytest.fixture
def consumer(mock_celery_app, mock_prometheus_registry):
    """Create CeleryEventConsumer instance."""
    with patch("src.workers.event_consumer.PROMETHEUS_AVAILABLE", False):
        return CeleryEventConsumer(mock_celery_app, registry=mock_prometheus_registry)


@pytest.fixture
def sample_task_event():
    """Create sample task event."""
    return {
        "uuid": "task-123",
        "name": "src.tasks.process_data",
        "timestamp": time.time(),
        "routing_key": "analytics",
        "queue": "analytics",
    }


# =============================================================================
# CELERY METRICS TESTS
# =============================================================================


class TestCeleryMetrics:
    """Tests for CeleryMetrics dataclass."""

    def test_metrics_default_not_initialized(self):
        """Test metrics are not initialized by default."""
        metrics = CeleryMetrics()
        assert metrics._initialized is False
        assert metrics.task_started is None

    def test_metrics_initialize_without_prometheus(self):
        """Test metrics initialization when Prometheus unavailable."""
        metrics = CeleryMetrics()
        with patch("src.workers.event_consumer.PROMETHEUS_AVAILABLE", False):
            metrics.initialize()
            assert metrics._initialized is False

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="Prometheus not installed")
    def test_metrics_initialize_with_prometheus(self, mock_prometheus_registry):
        """Test metrics initialization with Prometheus available."""
        metrics = CeleryMetrics()
        metrics.initialize(registry=mock_prometheus_registry)
        assert metrics._initialized is True or not PROMETHEUS_AVAILABLE

    def test_metrics_double_initialization_is_idempotent(self):
        """Test that double initialization does nothing."""
        metrics = CeleryMetrics()
        metrics._initialized = True
        metrics.initialize()
        # Should not raise


# =============================================================================
# TASK TIMING TESTS
# =============================================================================


class TestTaskTiming:
    """Tests for TaskTiming dataclass."""

    def test_timing_creation(self):
        """Test TaskTiming creation."""
        timing = TaskTiming(
            task_id="task-123",
            task_name="test_task",
            queue="default",
        )
        assert timing.task_id == "task-123"
        assert timing.task_name == "test_task"
        assert timing.queue == "default"
        assert timing.sent_at is None
        assert timing.started_at is None

    def test_timing_with_timestamps(self):
        """Test TaskTiming with all timestamps."""
        now = time.time()
        timing = TaskTiming(
            task_id="task-123",
            task_name="test_task",
            queue="default",
            sent_at=now,
            started_at=now + 0.1,
            succeeded_at=now + 1.0,
        )
        assert timing.sent_at == now
        assert timing.started_at == now + 0.1
        assert timing.succeeded_at == now + 1.0


# =============================================================================
# EVENT CONSUMER TESTS
# =============================================================================


class TestCeleryEventConsumer:
    """Tests for CeleryEventConsumer class."""

    def test_consumer_creation(self, consumer):
        """Test consumer can be created."""
        assert consumer is not None
        assert isinstance(consumer, CeleryEventConsumer)
        assert consumer._running is False

    def test_get_task_queue_from_routing_key(self, consumer, sample_task_event):
        """Test queue extraction from routing_key."""
        queue = consumer._get_task_queue(sample_task_event)
        assert queue == "analytics"

    def test_get_task_queue_default(self, consumer):
        """Test queue defaults to 'default' when not specified."""
        event = {"uuid": "task-123", "name": "test"}
        queue = consumer._get_task_queue(event)
        assert queue == "default"

    def test_get_or_create_timing_new(self, consumer, sample_task_event):
        """Test creating new task timing."""
        timing = consumer._get_or_create_timing(sample_task_event)
        assert timing.task_id == "task-123"
        assert timing.task_name == "src.tasks.process_data"
        assert timing.queue == "analytics"

    def test_get_or_create_timing_existing(self, consumer, sample_task_event):
        """Test retrieving existing task timing."""
        timing1 = consumer._get_or_create_timing(sample_task_event)
        timing2 = consumer._get_or_create_timing(sample_task_event)
        assert timing1 is timing2

    def test_cleanup_timing(self, consumer, sample_task_event):
        """Test cleaning up task timing."""
        consumer._get_or_create_timing(sample_task_event)
        timing = consumer._cleanup_timing("task-123")
        assert timing is not None
        assert "task-123" not in consumer._task_timings

    def test_cleanup_timing_nonexistent(self, consumer):
        """Test cleaning up nonexistent task timing."""
        timing = consumer._cleanup_timing("nonexistent")
        assert timing is None


class TestEventHandlers:
    """Tests for event handler methods."""

    def test_on_task_sent(self, consumer, sample_task_event):
        """Test task-sent event handler."""
        consumer.on_task_sent(sample_task_event)
        timing = consumer._task_timings.get("task-123")
        assert timing is not None
        assert timing.sent_at is not None

    def test_on_task_received(self, consumer, sample_task_event):
        """Test task-received event handler."""
        consumer.on_task_received(sample_task_event)
        timing = consumer._task_timings.get("task-123")
        assert timing is not None

    def test_on_task_started(self, consumer, sample_task_event):
        """Test task-started event handler."""
        # First send the task
        consumer.on_task_sent(sample_task_event)
        # Then start it
        consumer.on_task_started(sample_task_event)
        timing = consumer._task_timings.get("task-123")
        assert timing.started_at is not None

    def test_on_task_succeeded(self, consumer, sample_task_event):
        """Test task-succeeded event handler."""
        consumer.on_task_sent(sample_task_event)
        consumer.on_task_started(sample_task_event)
        consumer.on_task_succeeded(sample_task_event)
        # Timing should be cleaned up
        assert "task-123" not in consumer._task_timings

    def test_on_task_failed(self, consumer, sample_task_event):
        """Test task-failed event handler."""
        sample_task_event["exception"] = "ValueError(test error)"
        consumer.on_task_sent(sample_task_event)
        consumer.on_task_started(sample_task_event)
        consumer.on_task_failed(sample_task_event)
        # Timing should be cleaned up
        assert "task-123" not in consumer._task_timings

    def test_on_task_retried(self, consumer, sample_task_event):
        """Test task-retried event handler."""
        # Should not raise
        consumer.on_task_retried(sample_task_event)

    def test_on_task_rejected(self, consumer, sample_task_event):
        """Test task-rejected event handler."""
        # Should not raise
        consumer.on_task_rejected(sample_task_event)

    def test_on_task_revoked(self, consumer, sample_task_event):
        """Test task-revoked event handler."""
        consumer.on_task_sent(sample_task_event)
        consumer.on_task_revoked(sample_task_event)
        # Timing should be cleaned up
        assert "task-123" not in consumer._task_timings

    def test_on_worker_online(self, consumer):
        """Test worker-online event handler."""
        event = {"hostname": "worker-light-1"}
        # Should not raise
        consumer.on_worker_online(event)

    def test_on_worker_offline(self, consumer):
        """Test worker-offline event handler."""
        event = {"hostname": "worker-medium-1"}
        # Should not raise
        consumer.on_worker_offline(event)

    def test_on_worker_heartbeat(self, consumer):
        """Test worker-heartbeat event handler."""
        # Celery's state.event() requires 'type' field for event processing
        event = {
            "type": "worker-heartbeat",
            "hostname": "worker-1",
            "timestamp": time.time(),
        }
        # Should not raise
        consumer.on_worker_heartbeat(event)


class TestWorkerTypeInference:
    """Tests for worker type inference."""

    def test_infer_light_worker(self, consumer):
        """Test inferring light worker type."""
        assert consumer._infer_worker_type("celery-light-1") == "light"

    def test_infer_medium_worker(self, consumer):
        """Test inferring medium worker type."""
        assert consumer._infer_worker_type("celery-medium-1") == "medium"

    def test_infer_heavy_worker(self, consumer):
        """Test inferring heavy worker type."""
        assert consumer._infer_worker_type("celery-heavy-1") == "heavy"

    def test_infer_unknown_worker(self, consumer):
        """Test inferring unknown worker type."""
        assert consumer._infer_worker_type("celery-worker-1") == "unknown"


class TestGetHandlers:
    """Tests for get_handlers method."""

    def test_get_handlers_returns_dict(self, consumer):
        """Test get_handlers returns dictionary."""
        handlers = consumer.get_handlers()
        assert isinstance(handlers, dict)

    def test_get_handlers_has_expected_events(self, consumer):
        """Test get_handlers includes expected event types."""
        handlers = consumer.get_handlers()
        expected_events = [
            "task-sent",
            "task-received",
            "task-started",
            "task-succeeded",
            "task-failed",
            "task-retried",
            "task-rejected",
            "task-revoked",
            "worker-online",
            "worker-offline",
            "worker-heartbeat",
        ]
        for event in expected_events:
            assert event in handlers
            assert callable(handlers[event])


class TestConsumerLifecycle:
    """Tests for consumer lifecycle methods."""

    def test_stop_sets_running_false(self, consumer):
        """Test stop method sets _running to False."""
        consumer._running = True
        consumer.stop()
        assert consumer._running is False


# =============================================================================
# TRACE CONTEXT TESTS
# =============================================================================


class TestTraceContext:
    """Tests for trace context functions."""

    def test_inject_trace_context_with_trace_id(self):
        """Test injecting trace context with trace ID."""
        headers = {}
        result = inject_trace_context(headers, trace_id="abc123")
        assert result["X-Trace-ID"] == "abc123"
        assert result["X-Request-ID"] == "abc123"
        assert "X-Task-Sent-At" in result

    def test_inject_trace_context_without_trace_id(self):
        """Test injecting trace context without trace ID."""
        headers = {}
        result = inject_trace_context(headers)
        assert "X-Trace-ID" not in result
        assert "X-Task-Sent-At" in result

    def test_inject_trace_context_preserves_existing(self):
        """Test injecting trace context preserves existing headers."""
        headers = {"existing": "value"}
        result = inject_trace_context(headers, trace_id="abc123")
        assert result["existing"] == "value"

    def test_extract_trace_context_from_trace_id(self):
        """Test extracting trace context from X-Trace-ID."""
        headers = {"X-Trace-ID": "trace-123"}
        trace_id = extract_trace_context(headers)
        assert trace_id == "trace-123"

    def test_extract_trace_context_from_request_id(self):
        """Test extracting trace context from X-Request-ID."""
        headers = {"X-Request-ID": "request-456"}
        trace_id = extract_trace_context(headers)
        assert trace_id == "request-456"

    def test_extract_trace_context_prefers_trace_id(self):
        """Test X-Trace-ID is preferred over X-Request-ID."""
        headers = {"X-Trace-ID": "trace-123", "X-Request-ID": "request-456"}
        trace_id = extract_trace_context(headers)
        assert trace_id == "trace-123"

    def test_extract_trace_context_empty(self):
        """Test extracting trace context from empty headers."""
        headers = {}
        trace_id = extract_trace_context(headers)
        assert trace_id is None


# =============================================================================
# TRACED TASK CONTEXT MANAGER TESTS
# =============================================================================


class TestTracedTask:
    """Tests for traced_task context manager."""

    def test_traced_task_success(self):
        """Test traced_task context manager on success."""
        with traced_task("test_task", trace_id="trace-123") as tid:
            assert tid == "trace-123"

    def test_traced_task_generates_trace_id(self):
        """Test traced_task generates trace ID if not provided."""
        with traced_task("test_task") as tid:
            assert tid is not None
            assert "test_task" in tid

    def test_traced_task_exception_propagates(self):
        """Test traced_task propagates exceptions."""
        with pytest.raises(ValueError, match="test error"):
            with traced_task("test_task"):
                raise ValueError("test error")

    def test_traced_task_logs_completion(self):
        """Test traced_task logs completion."""
        with patch("src.workers.event_consumer.logger") as mock_logger:
            with traced_task("test_task"):
                pass
            # Should have logged debug messages
            assert mock_logger.debug.called


# =============================================================================
# GLOBAL METRICS TESTS
# =============================================================================


class TestGlobalMetrics:
    """Tests for global metrics instance."""

    def test_global_metrics_exists(self):
        """Test global celery_metrics instance exists."""
        assert celery_metrics is not None
        assert isinstance(celery_metrics, CeleryMetrics)
