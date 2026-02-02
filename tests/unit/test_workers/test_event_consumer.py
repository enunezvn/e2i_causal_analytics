"""
Unit tests for workers/event_consumer.py

Tests cover:
- CeleryMetrics initialization
- CeleryEventConsumer event handlers
- Task lifecycle tracking
- Prometheus metrics recording
- Trace ID propagation
- Event handler routing
"""

# Mock prometheus_client before importing event_consumer
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

mock_prom = MagicMock()
sys.modules["prometheus_client"] = mock_prom

from src.workers.event_consumer import (
    CeleryEventConsumer,
    CeleryMetrics,
    TaskTiming,
    extract_trace_context,
    inject_trace_context,
    traced_task,
)


@pytest.fixture
def mock_celery_app():
    """Mock Celery application."""
    app = MagicMock()
    app.connection = MagicMock()
    return app


@pytest.fixture
def mock_prometheus_registry():
    """Mock Prometheus registry."""
    return MagicMock()


@pytest.fixture
def celery_metrics():
    """CeleryMetrics instance."""
    metrics = CeleryMetrics()
    metrics._initialized = False
    return metrics


@pytest.fixture
def consumer(mock_celery_app):
    """CeleryEventConsumer instance."""
    return CeleryEventConsumer(mock_celery_app)


class TestCeleryMetrics:
    """Tests for CeleryMetrics dataclass and initialization."""

    def test_metrics_default_state(self, celery_metrics):
        """Test default metric state."""
        assert celery_metrics.registry is None
        assert celery_metrics.task_started is None
        assert celery_metrics.task_succeeded is None
        assert celery_metrics.task_failed is None
        assert celery_metrics.task_retried is None
        assert celery_metrics.task_rejected is None
        assert celery_metrics.task_revoked is None
        assert celery_metrics.task_latency is None
        assert celery_metrics.task_runtime is None
        assert celery_metrics.active_tasks is None
        assert celery_metrics.queue_length is None
        assert celery_metrics.worker_count is None
        assert celery_metrics._initialized is False

    @patch("src.workers.event_consumer.PROMETHEUS_AVAILABLE", True)
    def test_initialize_prometheus_metrics(self, celery_metrics):
        """Test Prometheus metrics initialization."""
        with (
            patch("src.workers.event_consumer.Counter") as MockCounter,
            patch("src.workers.event_consumer.Histogram") as MockHistogram,
            patch("src.workers.event_consumer.Gauge") as MockGauge,
        ):
            celery_metrics.initialize()

            assert celery_metrics._initialized is True
            assert MockCounter.call_count > 0
            assert MockHistogram.call_count > 0
            assert MockGauge.call_count > 0

    @patch("src.workers.event_consumer.PROMETHEUS_AVAILABLE", False)
    def test_initialize_no_prometheus(self, celery_metrics):
        """Test initialization without Prometheus available."""
        celery_metrics.initialize()

        assert celery_metrics._initialized is False

    def test_initialize_only_once(self, celery_metrics):
        """Test metrics only initialize once."""
        with patch("src.workers.event_consumer.Counter") as MockCounter:
            celery_metrics.initialize()
            celery_metrics.initialize()  # Second call should be no-op

            # If initialized, Counter should only be called once
            if celery_metrics._initialized:
                first_call_count = MockCounter.call_count
                celery_metrics.initialize()
                assert MockCounter.call_count == first_call_count


class TestTaskTiming:
    """Tests for TaskTiming dataclass."""

    def test_timing_defaults(self):
        """Test TaskTiming default values."""
        timing = TaskTiming(
            task_id="test-123",
            task_name="test_task",
            queue="default",
        )

        assert timing.task_id == "test-123"
        assert timing.task_name == "test_task"
        assert timing.queue == "default"
        assert timing.sent_at is None
        assert timing.started_at is None
        assert timing.succeeded_at is None
        assert timing.failed_at is None

    def test_timing_with_timestamps(self):
        """Test TaskTiming with timestamp values."""
        now = time.time()
        timing = TaskTiming(
            task_id="test-456",
            task_name="my_task",
            queue="analytics",
            sent_at=now,
            started_at=now + 1.0,
            succeeded_at=now + 10.0,
        )

        assert timing.sent_at == now
        assert timing.started_at == now + 1.0
        assert timing.succeeded_at == now + 10.0


class TestCeleryEventConsumer:
    """Tests for CeleryEventConsumer class."""

    def test_initialization(self, mock_celery_app):
        """Test consumer initialization."""
        consumer = CeleryEventConsumer(mock_celery_app)

        assert consumer.app == mock_celery_app
        assert consumer._task_timings == {}
        assert consumer._running is False

    def test_get_task_queue_from_routing_key(self, consumer):
        """Test extracting queue from routing_key."""
        event = {"routing_key": "analytics"}

        queue = consumer._get_task_queue(event)

        assert queue == "analytics"

    def test_get_task_queue_fallback_to_queue(self, consumer):
        """Test extracting queue from queue field."""
        event = {"queue": "shap"}

        queue = consumer._get_task_queue(event)

        assert queue == "shap"

    def test_get_task_queue_default(self, consumer):
        """Test default queue when no routing_key or queue."""
        event = {}

        queue = consumer._get_task_queue(event)

        assert queue == "default"

    def test_get_or_create_timing_new(self, consumer):
        """Test creating new task timing."""
        event = {
            "uuid": "task-123",
            "name": "test_task",
            "routing_key": "analytics",
        }

        timing = consumer._get_or_create_timing(event)

        assert timing.task_id == "task-123"
        assert timing.task_name == "test_task"
        assert timing.queue == "analytics"
        assert "task-123" in consumer._task_timings

    def test_get_or_create_timing_existing(self, consumer):
        """Test getting existing task timing."""
        event = {
            "uuid": "task-456",
            "name": "test_task",
            "routing_key": "default",
        }

        timing1 = consumer._get_or_create_timing(event)
        timing2 = consumer._get_or_create_timing(event)

        assert timing1 is timing2

    def test_cleanup_timing(self, consumer):
        """Test cleaning up task timing."""
        event = {"uuid": "task-789", "name": "test", "routing_key": "default"}
        timing = consumer._get_or_create_timing(event)

        cleaned = consumer._cleanup_timing("task-789")

        assert cleaned is timing
        assert "task-789" not in consumer._task_timings

    def test_cleanup_timing_not_found(self, consumer):
        """Test cleaning up non-existent timing."""
        result = consumer._cleanup_timing("nonexistent")

        assert result is None

    def test_on_task_sent(self, consumer):
        """Test task-sent event handler."""
        event = {
            "uuid": "task-sent-1",
            "name": "my_task",
            "routing_key": "default",
            "timestamp": 1234567890.0,
        }

        consumer.on_task_sent(event)

        timing = consumer._task_timings["task-sent-1"]
        assert timing.sent_at == 1234567890.0

    def test_on_task_received(self, consumer):
        """Test task-received event handler."""
        event = {
            "uuid": "task-recv-1",
            "name": "my_task",
            "routing_key": "default",
            "timestamp": 1234567891.0,
        }

        consumer.on_task_received(event)

        timing = consumer._task_timings["task-recv-1"]
        assert timing.sent_at == 1234567891.0  # Set if not already set

    def test_on_task_started(self, consumer):
        """Test task-started event handler."""
        # Setup task timing
        event = {
            "uuid": "task-start-1",
            "name": "my_task",
            "routing_key": "analytics",
            "timestamp": 1234567892.0,
        }
        consumer.on_task_sent(
            {
                "uuid": "task-start-1",
                "name": "my_task",
                "routing_key": "analytics",
                "timestamp": 1234567890.0,
            }
        )

        # Mock metrics
        with patch("src.workers.event_consumer.celery_metrics") as mock_metrics:
            mock_metrics.task_started = MagicMock()
            mock_metrics.task_started.labels = MagicMock(return_value=MagicMock())
            mock_metrics.task_latency = MagicMock()
            mock_metrics.task_latency.labels = MagicMock(return_value=MagicMock())
            mock_metrics.active_tasks = MagicMock()
            mock_metrics.active_tasks.labels = MagicMock(return_value=MagicMock())

            consumer.on_task_started(event)

        timing = consumer._task_timings["task-start-1"]
        assert timing.started_at == 1234567892.0

    def test_on_task_succeeded(self, consumer):
        """Test task-succeeded event handler."""
        # Setup timing
        consumer.on_task_sent(
            {
                "uuid": "task-success-1",
                "name": "my_task",
                "routing_key": "default",
                "timestamp": 1234567890.0,
            }
        )
        consumer.on_task_started(
            {
                "uuid": "task-success-1",
                "name": "my_task",
                "routing_key": "default",
                "timestamp": 1234567891.0,
            }
        )

        event = {
            "uuid": "task-success-1",
            "name": "my_task",
            "routing_key": "default",
            "timestamp": 1234567900.0,
        }

        with patch("src.workers.event_consumer.celery_metrics") as mock_metrics:
            mock_metrics.task_succeeded = MagicMock()
            mock_metrics.task_succeeded.labels = MagicMock(return_value=MagicMock())
            mock_metrics.task_runtime = MagicMock()
            mock_metrics.task_runtime.labels = MagicMock(return_value=MagicMock())
            mock_metrics.active_tasks = MagicMock()
            mock_metrics.active_tasks.labels = MagicMock(return_value=MagicMock())

            consumer.on_task_succeeded(event)

        # Task should be cleaned up
        assert "task-success-1" not in consumer._task_timings

    def test_on_task_failed(self, consumer):
        """Test task-failed event handler."""
        # Setup timing
        consumer.on_task_sent(
            {
                "uuid": "task-fail-1",
                "name": "my_task",
                "routing_key": "default",
                "timestamp": 1234567890.0,
            }
        )
        consumer.on_task_started(
            {
                "uuid": "task-fail-1",
                "name": "my_task",
                "routing_key": "default",
                "timestamp": 1234567891.0,
            }
        )

        event = {
            "uuid": "task-fail-1",
            "name": "my_task",
            "routing_key": "default",
            "timestamp": 1234567895.0,
            "exception": "ValueError('test error')",
        }

        with patch("src.workers.event_consumer.celery_metrics") as mock_metrics:
            mock_metrics.task_failed = MagicMock()
            mock_metrics.task_failed.labels = MagicMock(return_value=MagicMock())
            mock_metrics.task_runtime = MagicMock()
            mock_metrics.task_runtime.labels = MagicMock(return_value=MagicMock())
            mock_metrics.active_tasks = MagicMock()
            mock_metrics.active_tasks.labels = MagicMock(return_value=MagicMock())

            consumer.on_task_failed(event)

        # Task should be cleaned up
        assert "task-fail-1" not in consumer._task_timings

    def test_on_task_retried(self, consumer):
        """Test task-retried event handler."""
        event = {
            "uuid": "task-retry-1",
            "name": "my_task",
            "routing_key": "default",
        }

        with patch("src.workers.event_consumer.celery_metrics") as mock_metrics:
            mock_metrics.task_retried = MagicMock()
            mock_metrics.task_retried.labels = MagicMock(return_value=MagicMock())

            consumer.on_task_retried(event)

    def test_on_task_rejected(self, consumer):
        """Test task-rejected event handler."""
        event = {
            "uuid": "task-reject-1",
            "name": "my_task",
            "routing_key": "default",
        }

        with patch("src.workers.event_consumer.celery_metrics") as mock_metrics:
            mock_metrics.task_rejected = MagicMock()
            mock_metrics.task_rejected.labels = MagicMock(return_value=MagicMock())

            consumer.on_task_rejected(event)

    def test_on_task_revoked(self, consumer):
        """Test task-revoked event handler."""
        # Setup timing first
        consumer.on_task_sent(
            {"uuid": "task-revoke-1", "name": "my_task", "routing_key": "default"}
        )

        event = {
            "uuid": "task-revoke-1",
            "name": "my_task",
        }

        with patch("src.workers.event_consumer.celery_metrics") as mock_metrics:
            mock_metrics.task_revoked = MagicMock()
            mock_metrics.task_revoked.labels = MagicMock(return_value=MagicMock())

            consumer.on_task_revoked(event)

        # Task should be cleaned up
        assert "task-revoke-1" not in consumer._task_timings

    def test_on_worker_online(self, consumer):
        """Test worker-online event handler."""
        event = {"hostname": "worker-light-01"}

        with patch("src.workers.event_consumer.celery_metrics") as mock_metrics:
            mock_metrics.worker_count = MagicMock()
            mock_metrics.worker_count.labels = MagicMock(return_value=MagicMock())

            consumer.on_worker_online(event)

    def test_on_worker_offline(self, consumer):
        """Test worker-offline event handler."""
        event = {"hostname": "worker-heavy-02"}

        with patch("src.workers.event_consumer.celery_metrics") as mock_metrics:
            mock_metrics.worker_count = MagicMock()
            mock_metrics.worker_count.labels = MagicMock(return_value=MagicMock())

            consumer.on_worker_offline(event)

    def test_on_worker_heartbeat(self, consumer):
        """Test worker-heartbeat event handler."""
        event = {"hostname": "worker-medium-01", "type": "worker-heartbeat"}

        # Should call state.event() without raising an exception
        consumer.on_worker_heartbeat(event)

        # Verify no exception was raised

    def test_infer_worker_type_light(self, consumer):
        """Test inferring light worker type."""
        assert consumer._infer_worker_type("worker-light-01") == "light"

    def test_infer_worker_type_medium(self, consumer):
        """Test inferring medium worker type."""
        assert consumer._infer_worker_type("worker-medium-02") == "medium"

    def test_infer_worker_type_heavy(self, consumer):
        """Test inferring heavy worker type."""
        assert consumer._infer_worker_type("worker-heavy-03") == "heavy"

    def test_infer_worker_type_unknown(self, consumer):
        """Test inferring unknown worker type."""
        assert consumer._infer_worker_type("random-worker") == "unknown"

    def test_get_handlers(self, consumer):
        """Test getting event handler mapping."""
        handlers = consumer.get_handlers()

        assert "task-sent" in handlers
        assert "task-received" in handlers
        assert "task-started" in handlers
        assert "task-succeeded" in handlers
        assert "task-failed" in handlers
        assert "task-retried" in handlers
        assert "task-rejected" in handlers
        assert "task-revoked" in handlers
        assert "worker-online" in handlers
        assert "worker-offline" in handlers
        assert "worker-heartbeat" in handlers

        # Verify handlers are callable
        assert callable(handlers["task-started"])
        assert callable(handlers["task-succeeded"])

    def test_stop(self, consumer):
        """Test stopping consumer."""
        consumer._running = True

        consumer.stop()

        assert consumer._running is False


class TestTraceIDPropagation:
    """Tests for trace ID propagation helpers."""

    def test_inject_trace_context(self):
        """Test injecting trace context into headers."""
        headers = {}
        trace_id = "test-trace-123"

        result = inject_trace_context(headers, trace_id)

        assert result["X-Trace-ID"] == trace_id
        assert result["X-Request-ID"] == trace_id
        assert "X-Task-Sent-At" in result
        assert isinstance(result["X-Task-Sent-At"], float)

    def test_inject_trace_context_no_trace_id(self):
        """Test injecting without trace ID."""
        headers = {}

        result = inject_trace_context(headers)

        assert "X-Trace-ID" not in result
        assert "X-Task-Sent-At" in result

    def test_extract_trace_context(self):
        """Test extracting trace context from headers."""
        headers = {"X-Trace-ID": "trace-456"}

        trace_id = extract_trace_context(headers)

        assert trace_id == "trace-456"

    def test_extract_trace_context_from_request_id(self):
        """Test extracting from X-Request-ID."""
        headers = {"X-Request-ID": "request-789"}

        trace_id = extract_trace_context(headers)

        assert trace_id == "request-789"

    def test_extract_trace_context_not_found(self):
        """Test extracting when not present."""
        headers = {}

        trace_id = extract_trace_context(headers)

        assert trace_id is None


class TestTracedTask:
    """Tests for traced_task context manager."""

    def test_traced_task_success(self):
        """Test traced task successful execution."""
        with traced_task("my_task", trace_id="test-123") as trace_id:
            assert trace_id == "test-123"
            # Task logic here
            result = 42

        assert result == 42

    def test_traced_task_auto_trace_id(self):
        """Test traced task with auto-generated trace ID."""
        with traced_task("my_task") as trace_id:
            assert trace_id.startswith("task-my_task-")

    def test_traced_task_exception(self):
        """Test traced task with exception."""
        with pytest.raises(ValueError):
            with traced_task("failing_task"):
                raise ValueError("Test error")


class TestEventHandlerIntegration:
    """Integration tests for event handler flow."""

    def test_complete_task_lifecycle(self, consumer):
        """Test complete task lifecycle from sent to success."""
        task_id = "lifecycle-1"

        # Task sent
        consumer.on_task_sent(
            {
                "uuid": task_id,
                "name": "integration_task",
                "routing_key": "analytics",
                "timestamp": 1000.0,
            }
        )

        # Task received
        consumer.on_task_received(
            {
                "uuid": task_id,
                "name": "integration_task",
                "routing_key": "analytics",
                "timestamp": 1001.0,
            }
        )

        # Task started
        with patch("src.workers.event_consumer.celery_metrics"):
            consumer.on_task_started(
                {
                    "uuid": task_id,
                    "name": "integration_task",
                    "routing_key": "analytics",
                    "timestamp": 1002.0,
                }
            )

        # Task succeeded
        with patch("src.workers.event_consumer.celery_metrics"):
            consumer.on_task_succeeded(
                {
                    "uuid": task_id,
                    "name": "integration_task",
                    "routing_key": "analytics",
                    "timestamp": 1010.0,
                }
            )

        # Task should be cleaned up
        assert task_id not in consumer._task_timings

    def test_task_lifecycle_with_failure(self, consumer):
        """Test task lifecycle ending in failure."""
        task_id = "failure-1"

        # Task sent and started
        consumer.on_task_sent(
            {
                "uuid": task_id,
                "name": "failing_task",
                "routing_key": "default",
                "timestamp": 2000.0,
            }
        )

        with patch("src.workers.event_consumer.celery_metrics"):
            consumer.on_task_started(
                {
                    "uuid": task_id,
                    "name": "failing_task",
                    "routing_key": "default",
                    "timestamp": 2001.0,
                }
            )

        # Task failed
        with patch("src.workers.event_consumer.celery_metrics"):
            consumer.on_task_failed(
                {
                    "uuid": task_id,
                    "name": "failing_task",
                    "routing_key": "default",
                    "timestamp": 2005.0,
                    "exception": "RuntimeError('fail')",
                }
            )

        # Task should be cleaned up
        assert task_id not in consumer._task_timings
