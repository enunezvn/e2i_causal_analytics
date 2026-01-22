"""
Celery Event Consumer for Task Observability.

G12 from observability audit remediation plan:
- Consumes Celery task events for monitoring
- Records task latency, success/failure rates
- Emits Prometheus metrics for task operations
- Propagates trace IDs through task execution

Version: 1.0.0
"""

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from celery import Celery
from celery.events import EventReceiver
from celery.events.state import State

logger = logging.getLogger(__name__)

# =============================================================================
# Prometheus Metrics Integration
# =============================================================================

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed, task metrics disabled")


@dataclass
class CeleryMetrics:
    """Container for Celery task metrics."""

    registry: Optional[Any] = None
    task_started: Optional[Counter] = None
    task_succeeded: Optional[Counter] = None
    task_failed: Optional[Counter] = None
    task_retried: Optional[Counter] = None
    task_rejected: Optional[Counter] = None
    task_revoked: Optional[Counter] = None
    task_latency: Optional[Histogram] = None
    task_runtime: Optional[Histogram] = None
    active_tasks: Optional[Gauge] = None
    queue_length: Optional[Gauge] = None
    worker_count: Optional[Gauge] = None

    _initialized: bool = field(default=False, repr=False)

    def initialize(self, registry: Optional[Any] = None) -> None:
        """Initialize Prometheus metrics for Celery."""
        if self._initialized or not PROMETHEUS_AVAILABLE:
            return

        self.registry = registry or CollectorRegistry()

        # Task lifecycle counters
        self.task_started = Counter(
            "celery_task_started_total",
            "Total number of tasks started",
            ["task_name", "queue"],
            registry=self.registry,
        )

        self.task_succeeded = Counter(
            "celery_task_succeeded_total",
            "Total number of tasks that succeeded",
            ["task_name", "queue"],
            registry=self.registry,
        )

        self.task_failed = Counter(
            "celery_task_failed_total",
            "Total number of tasks that failed",
            ["task_name", "queue", "exception"],
            registry=self.registry,
        )

        self.task_retried = Counter(
            "celery_task_retried_total",
            "Total number of task retries",
            ["task_name", "queue"],
            registry=self.registry,
        )

        self.task_rejected = Counter(
            "celery_task_rejected_total",
            "Total number of tasks rejected",
            ["task_name", "queue"],
            registry=self.registry,
        )

        self.task_revoked = Counter(
            "celery_task_revoked_total",
            "Total number of tasks revoked",
            ["task_name"],
            registry=self.registry,
        )

        # Task timing histograms
        self.task_latency = Histogram(
            "celery_task_latency_seconds",
            "Time from task sent to task started",
            ["task_name", "queue"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            registry=self.registry,
        )

        self.task_runtime = Histogram(
            "celery_task_runtime_seconds",
            "Task execution runtime",
            ["task_name", "queue"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
            registry=self.registry,
        )

        # Current state gauges
        self.active_tasks = Gauge(
            "celery_active_tasks",
            "Number of currently active tasks",
            ["queue"],
            registry=self.registry,
        )

        self.queue_length = Gauge(
            "celery_queue_length",
            "Number of tasks waiting in queue",
            ["queue"],
            registry=self.registry,
        )

        self.worker_count = Gauge(
            "celery_workers_online",
            "Number of online workers",
            ["worker_type"],
            registry=self.registry,
        )

        self._initialized = True
        logger.info("Celery Prometheus metrics initialized")


# Global metrics instance
celery_metrics = CeleryMetrics()


# =============================================================================
# Event Handlers
# =============================================================================


@dataclass
class TaskTiming:
    """Track task timing for latency calculation."""

    task_id: str
    task_name: str
    queue: str
    sent_at: Optional[float] = None
    started_at: Optional[float] = None
    succeeded_at: Optional[float] = None
    failed_at: Optional[float] = None


class CeleryEventConsumer:
    """
    Consumes and processes Celery task events.

    Features:
    - Records task lifecycle metrics (started, succeeded, failed, retried)
    - Tracks task latency (sent → started) and runtime (started → finished)
    - Monitors worker availability
    - Supports trace ID propagation via task headers

    Usage:
        consumer = CeleryEventConsumer(celery_app)
        consumer.start()  # Blocks and processes events
    """

    def __init__(
        self,
        app: Celery,
        registry: Optional[Any] = None,
    ):
        """
        Initialize the event consumer.

        Args:
            app: Celery application instance
            registry: Optional Prometheus registry to use
        """
        self.app = app
        self.state = State()
        self._task_timings: Dict[str, TaskTiming] = {}
        self._running = False

        # Initialize metrics
        celery_metrics.initialize(registry)

    def _get_task_queue(self, event: Dict[str, Any]) -> str:
        """Extract queue name from event."""
        # Try to get from routing_key or default to 'default'
        return event.get("routing_key", event.get("queue", "default"))

    def _get_or_create_timing(self, event: Dict[str, Any]) -> TaskTiming:
        """Get or create task timing tracker."""
        task_id = event.get("uuid", "unknown")
        if task_id not in self._task_timings:
            self._task_timings[task_id] = TaskTiming(
                task_id=task_id,
                task_name=event.get("name", "unknown"),
                queue=self._get_task_queue(event),
            )
        return self._task_timings[task_id]

    def _cleanup_timing(self, task_id: str) -> Optional[TaskTiming]:
        """Remove and return task timing."""
        return self._task_timings.pop(task_id, None)

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def on_task_sent(self, event: Dict[str, Any]) -> None:
        """Handle task-sent event."""
        timing = self._get_or_create_timing(event)
        timing.sent_at = event.get("timestamp", time.time())

        logger.debug(
            f"Task sent: {event.get('name')} [{event.get('uuid')}] "
            f"to queue {timing.queue}"
        )

    def on_task_received(self, event: Dict[str, Any]) -> None:
        """Handle task-received event (worker received task)."""
        timing = self._get_or_create_timing(event)

        # If we missed the sent event, record received as sent
        if timing.sent_at is None:
            timing.sent_at = event.get("timestamp", time.time())

        logger.debug(
            f"Task received: {event.get('name')} [{event.get('uuid')}]"
        )

    def on_task_started(self, event: Dict[str, Any]) -> None:
        """Handle task-started event."""
        timing = self._get_or_create_timing(event)
        timing.started_at = event.get("timestamp", time.time())

        # Record latency (time from sent to started)
        if celery_metrics.task_latency and timing.sent_at:
            latency = timing.started_at - timing.sent_at
            celery_metrics.task_latency.labels(
                task_name=timing.task_name,
                queue=timing.queue,
            ).observe(latency)

        # Record started counter
        if celery_metrics.task_started:
            celery_metrics.task_started.labels(
                task_name=timing.task_name,
                queue=timing.queue,
            ).inc()

        # Update active tasks gauge
        if celery_metrics.active_tasks:
            celery_metrics.active_tasks.labels(queue=timing.queue).inc()

        logger.debug(
            f"Task started: {timing.task_name} [{timing.task_id}] "
            f"latency={timing.started_at - timing.sent_at:.3f}s"
            if timing.sent_at else ""
        )

    def on_task_succeeded(self, event: Dict[str, Any]) -> None:
        """Handle task-succeeded event."""
        task_id = event.get("uuid", "unknown")
        timing = self._cleanup_timing(task_id)

        if timing is None:
            timing = TaskTiming(
                task_id=task_id,
                task_name=event.get("name", "unknown"),
                queue=self._get_task_queue(event),
            )

        timing.succeeded_at = event.get("timestamp", time.time())

        # Record runtime
        if celery_metrics.task_runtime and timing.started_at:
            runtime = timing.succeeded_at - timing.started_at
            celery_metrics.task_runtime.labels(
                task_name=timing.task_name,
                queue=timing.queue,
            ).observe(runtime)

            logger.debug(
                f"Task succeeded: {timing.task_name} [{timing.task_id}] "
                f"runtime={runtime:.3f}s"
            )

        # Record success counter
        if celery_metrics.task_succeeded:
            celery_metrics.task_succeeded.labels(
                task_name=timing.task_name,
                queue=timing.queue,
            ).inc()

        # Update active tasks gauge
        if celery_metrics.active_tasks:
            celery_metrics.active_tasks.labels(queue=timing.queue).dec()

    def on_task_failed(self, event: Dict[str, Any]) -> None:
        """Handle task-failed event."""
        task_id = event.get("uuid", "unknown")
        timing = self._cleanup_timing(task_id)

        if timing is None:
            timing = TaskTiming(
                task_id=task_id,
                task_name=event.get("name", "unknown"),
                queue=self._get_task_queue(event),
            )

        timing.failed_at = event.get("timestamp", time.time())
        exception = event.get("exception", "unknown")

        # Extract exception class name
        exc_class = exception.split("(")[0] if exception else "unknown"

        # Record runtime even for failed tasks
        if celery_metrics.task_runtime and timing.started_at:
            runtime = timing.failed_at - timing.started_at
            celery_metrics.task_runtime.labels(
                task_name=timing.task_name,
                queue=timing.queue,
            ).observe(runtime)

        # Record failure counter
        if celery_metrics.task_failed:
            celery_metrics.task_failed.labels(
                task_name=timing.task_name,
                queue=timing.queue,
                exception=exc_class[:50],  # Truncate long exception names
            ).inc()

        # Update active tasks gauge
        if celery_metrics.active_tasks:
            celery_metrics.active_tasks.labels(queue=timing.queue).dec()

        logger.warning(
            f"Task failed: {timing.task_name} [{timing.task_id}] "
            f"exception={exc_class}"
        )

    def on_task_retried(self, event: Dict[str, Any]) -> None:
        """Handle task-retried event."""
        task_name = event.get("name", "unknown")
        queue = self._get_task_queue(event)

        if celery_metrics.task_retried:
            celery_metrics.task_retried.labels(
                task_name=task_name,
                queue=queue,
            ).inc()

        logger.debug(f"Task retried: {task_name} [{event.get('uuid')}]")

    def on_task_rejected(self, event: Dict[str, Any]) -> None:
        """Handle task-rejected event."""
        task_name = event.get("name", "unknown")
        queue = self._get_task_queue(event)

        if celery_metrics.task_rejected:
            celery_metrics.task_rejected.labels(
                task_name=task_name,
                queue=queue,
            ).inc()

        logger.warning(f"Task rejected: {task_name} [{event.get('uuid')}]")

    def on_task_revoked(self, event: Dict[str, Any]) -> None:
        """Handle task-revoked event."""
        task_id = event.get("uuid", "unknown")
        task_name = event.get("name", "unknown")

        # Cleanup timing if exists
        self._cleanup_timing(task_id)

        if celery_metrics.task_revoked:
            celery_metrics.task_revoked.labels(task_name=task_name).inc()

        logger.info(f"Task revoked: {task_name} [{task_id}]")

    def on_worker_online(self, event: Dict[str, Any]) -> None:
        """Handle worker-online event."""
        hostname = event.get("hostname", "unknown")
        worker_type = self._infer_worker_type(hostname)

        if celery_metrics.worker_count:
            celery_metrics.worker_count.labels(worker_type=worker_type).inc()

        logger.info(f"Worker online: {hostname} (type={worker_type})")

    def on_worker_offline(self, event: Dict[str, Any]) -> None:
        """Handle worker-offline event."""
        hostname = event.get("hostname", "unknown")
        worker_type = self._infer_worker_type(hostname)

        if celery_metrics.worker_count:
            celery_metrics.worker_count.labels(worker_type=worker_type).dec()

        logger.info(f"Worker offline: {hostname} (type={worker_type})")

    def on_worker_heartbeat(self, event: Dict[str, Any]) -> None:
        """Handle worker-heartbeat event (periodic)."""
        # Update state to track active workers
        self.state.event(event)

    def _infer_worker_type(self, hostname: str) -> str:
        """Infer worker type from hostname."""
        hostname_lower = hostname.lower()
        if "light" in hostname_lower:
            return "light"
        elif "medium" in hostname_lower:
            return "medium"
        elif "heavy" in hostname_lower:
            return "heavy"
        return "unknown"

    # -------------------------------------------------------------------------
    # Event Consumer
    # -------------------------------------------------------------------------

    def get_handlers(self) -> Dict[str, Callable]:
        """Get mapping of event types to handler methods."""
        return {
            "task-sent": self.on_task_sent,
            "task-received": self.on_task_received,
            "task-started": self.on_task_started,
            "task-succeeded": self.on_task_succeeded,
            "task-failed": self.on_task_failed,
            "task-retried": self.on_task_retried,
            "task-rejected": self.on_task_rejected,
            "task-revoked": self.on_task_revoked,
            "worker-online": self.on_worker_online,
            "worker-offline": self.on_worker_offline,
            "worker-heartbeat": self.on_worker_heartbeat,
        }

    def start(self) -> None:
        """
        Start consuming Celery events.

        This method blocks and continuously processes events.
        Use in a dedicated process or thread.
        """
        self._running = True
        logger.info("Starting Celery event consumer...")

        handlers = self.get_handlers()

        with self.app.connection() as connection:
            recv = EventReceiver(
                connection,
                handlers=handlers,
                app=self.app,
            )
            recv.capture(limit=None, timeout=None, wakeup=True)

    def stop(self) -> None:
        """Signal the consumer to stop."""
        self._running = False
        logger.info("Stopping Celery event consumer...")


# =============================================================================
# Trace ID Propagation
# =============================================================================


def inject_trace_context(headers: Dict[str, Any], trace_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Inject trace context into task headers for propagation.

    Args:
        headers: Existing task headers
        trace_id: Optional trace ID to inject

    Returns:
        Headers with trace context added
    """
    if trace_id:
        headers["X-Trace-ID"] = trace_id
        headers["X-Request-ID"] = trace_id

    # Add timestamp for latency calculation
    headers["X-Task-Sent-At"] = time.time()

    return headers


def extract_trace_context(headers: Dict[str, Any]) -> Optional[str]:
    """
    Extract trace context from task headers.

    Args:
        headers: Task headers

    Returns:
        Trace ID if present, None otherwise
    """
    return headers.get("X-Trace-ID") or headers.get("X-Request-ID")


# =============================================================================
# Task Decorators
# =============================================================================


@contextmanager
def traced_task(task_name: str, trace_id: Optional[str] = None):
    """
    Context manager for tracing task execution.

    Usage:
        with traced_task("my_task", trace_id="abc123"):
            # Task logic here
            pass
    """
    start_time = time.time()
    trace_id = trace_id or f"task-{task_name}-{int(start_time)}"

    logger.debug(f"Starting traced task: {task_name} [{trace_id}]")

    try:
        yield trace_id
        duration = time.time() - start_time
        logger.debug(f"Completed traced task: {task_name} [{trace_id}] in {duration:.3f}s")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed traced task: {task_name} [{trace_id}] after {duration:.3f}s: {e}")
        raise


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """Run the event consumer from command line."""
    from .celery_app import celery_app

    consumer = CeleryEventConsumer(celery_app)

    try:
        consumer.start()
    except KeyboardInterrupt:
        consumer.stop()
        logger.info("Event consumer stopped by user")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
