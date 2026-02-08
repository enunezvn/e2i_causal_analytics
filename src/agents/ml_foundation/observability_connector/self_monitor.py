"""Self-Monitoring for Observability Connector.

Provides internal monitoring and health tracking for the observability system:
- Latency tracking for span emission, Opik API, and database writes
- Periodic health span emission (default: every 60 seconds)
- Alert thresholds for degraded performance detection
- Metrics aggregation for dashboard reporting

Version: 1.0.0 (Phase 3.5)

Usage:
    from src.agents.ml_foundation.observability_connector.self_monitor import (
        get_self_monitor,
        SelfMonitor,
    )

    # Get singleton instance
    monitor = get_self_monitor()

    # Record latencies
    monitor.record_span_emission_latency(15.5)
    monitor.record_opik_latency(45.2)
    monitor.record_database_latency(8.3)

    # Check health status
    health = monitor.get_health_status()
    if health.status == HealthStatus.DEGRADED:
        logger.warning("Observability system is degraded")

    # Start background health span emission
    await monitor.start_health_emission()

Author: E2I Causal Analytics Team
"""

import asyncio
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================


class HealthStatus(str, Enum):
    """Overall health status of the observability system."""

    HEALTHY = "healthy"  # All metrics within thresholds
    DEGRADED = "degraded"  # Some metrics exceeding thresholds
    UNHEALTHY = "unhealthy"  # Critical metrics failing
    UNKNOWN = "unknown"  # Insufficient data


class MetricType(str, Enum):
    """Types of metrics being tracked."""

    SPAN_EMISSION = "span_emission"
    OPIK_API = "opik_api"
    DATABASE_WRITE = "database_write"
    BATCH_FLUSH = "batch_flush"
    CACHE_OPERATION = "cache_operation"


# Default thresholds (in milliseconds)
DEFAULT_THRESHOLDS = {
    MetricType.SPAN_EMISSION: {"warning": 100.0, "critical": 500.0},
    MetricType.OPIK_API: {"warning": 200.0, "critical": 1000.0},
    MetricType.DATABASE_WRITE: {"warning": 50.0, "critical": 200.0},
    MetricType.BATCH_FLUSH: {"warning": 500.0, "critical": 2000.0},
    MetricType.CACHE_OPERATION: {"warning": 10.0, "critical": 50.0},
}

# Default configuration
DEFAULT_HEALTH_EMISSION_INTERVAL = 60.0  # seconds
DEFAULT_WINDOW_SIZE = 100  # samples to keep for rolling statistics
DEFAULT_MIN_SAMPLES_FOR_STATS = 5  # minimum samples before calculating stats


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class LatencyThresholds:
    """Thresholds for latency alerts."""

    warning_ms: float = 100.0  # Yellow alert
    critical_ms: float = 500.0  # Red alert

    def check(self, latency_ms: float) -> HealthStatus:
        """Check latency against thresholds."""
        if latency_ms >= self.critical_ms:
            return HealthStatus.UNHEALTHY
        elif latency_ms >= self.warning_ms:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY


@dataclass
class LatencyStats:
    """Statistics for a latency metric."""

    count: int = 0
    min_ms: float = 0.0
    max_ms: float = 0.0
    avg_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    last_ms: float = 0.0
    last_recorded: Optional[datetime] = None
    status: HealthStatus = HealthStatus.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "avg_ms": round(self.avg_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "last_ms": round(self.last_ms, 2),
            "last_recorded": (self.last_recorded.isoformat() if self.last_recorded else None),
            "status": self.status.value,
        }


@dataclass
class ComponentHealth:
    """Health status for a single component."""

    component: str
    status: HealthStatus
    latency_stats: LatencyStats
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "status": self.status.value,
            "latency": self.latency_stats.to_dict(),
            "error_count": self.error_count,
            "last_error": self.last_error,
            "last_error_time": (self.last_error_time.isoformat() if self.last_error_time else None),
        }


@dataclass
class OverallHealth:
    """Overall health status of the observability system."""

    status: HealthStatus
    components: Dict[str, ComponentHealth]
    uptime_seconds: float
    last_health_check: datetime
    health_span_count: int = 0
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "uptime_seconds": round(self.uptime_seconds, 2),
            "last_health_check": self.last_health_check.isoformat(),
            "health_span_count": self.health_span_count,
            "alerts": self.alerts,
        }


@dataclass
class SelfMonitorConfig:
    """Configuration for self-monitoring."""

    health_emission_interval_seconds: float = DEFAULT_HEALTH_EMISSION_INTERVAL
    window_size: int = DEFAULT_WINDOW_SIZE
    min_samples_for_stats: int = DEFAULT_MIN_SAMPLES_FOR_STATS
    thresholds: Dict[MetricType, LatencyThresholds] = field(default_factory=dict)
    emit_health_spans: bool = True
    log_degraded_status: bool = True

    def __post_init__(self) -> None:
        """Initialize default thresholds if not provided."""
        if not self.thresholds:
            self.thresholds = {
                metric_type: LatencyThresholds(
                    warning_ms=values["warning"],
                    critical_ms=values["critical"],
                )
                for metric_type, values in DEFAULT_THRESHOLDS.items()
            }


# ============================================================================
# LATENCY TRACKER
# ============================================================================


class LatencyTracker:
    """Tracks latency samples with rolling window statistics."""

    def __init__(
        self,
        metric_type: MetricType,
        thresholds: LatencyThresholds,
        window_size: int = DEFAULT_WINDOW_SIZE,
        min_samples: int = DEFAULT_MIN_SAMPLES_FOR_STATS,
    ):
        """Initialize the latency tracker.

        Args:
            metric_type: Type of metric being tracked
            thresholds: Alert thresholds for this metric
            window_size: Number of samples to keep for rolling stats
            min_samples: Minimum samples before calculating percentiles
        """
        self.metric_type = metric_type
        self.thresholds = thresholds
        self.window_size = window_size
        self.min_samples = min_samples

        self._samples: Deque[float] = deque(maxlen=window_size)
        self._total_count: int = 0
        self._error_count: int = 0
        self._last_error: Optional[str] = None
        self._last_error_time: Optional[datetime] = None
        self._last_recorded: Optional[datetime] = None

    def record(self, latency_ms: float) -> None:
        """Record a latency sample.

        Args:
            latency_ms: Latency in milliseconds
        """
        self._samples.append(latency_ms)
        self._total_count += 1
        self._last_recorded = datetime.now(timezone.utc)

    def record_error(self, error_message: str) -> None:
        """Record an error occurrence.

        Args:
            error_message: Description of the error
        """
        self._error_count += 1
        self._last_error = error_message
        self._last_error_time = datetime.now(timezone.utc)

    def get_stats(self) -> LatencyStats:
        """Calculate current statistics.

        Returns:
            LatencyStats with current metrics
        """
        samples = list(self._samples)

        if not samples:
            return LatencyStats(status=HealthStatus.UNKNOWN)

        sorted_samples = sorted(samples)
        count = len(samples)

        # Calculate percentiles
        p50_idx = int(count * 0.50)
        p95_idx = int(count * 0.95)
        p99_idx = int(count * 0.99)

        # Ensure indices are within bounds
        p50_idx = min(p50_idx, count - 1)
        p95_idx = min(p95_idx, count - 1)
        p99_idx = min(p99_idx, count - 1)

        avg_ms = statistics.mean(samples)
        status = self.thresholds.check(avg_ms)

        return LatencyStats(
            count=self._total_count,
            min_ms=min(samples),
            max_ms=max(samples),
            avg_ms=avg_ms,
            p50_ms=sorted_samples[p50_idx],
            p95_ms=sorted_samples[p95_idx] if count >= self.min_samples else avg_ms,
            p99_ms=sorted_samples[p99_idx] if count >= self.min_samples else avg_ms,
            last_ms=samples[-1],
            last_recorded=self._last_recorded,
            status=status,
        )

    def get_component_health(self) -> ComponentHealth:
        """Get health status for this component.

        Returns:
            ComponentHealth with current status
        """
        stats = self.get_stats()

        return ComponentHealth(
            component=self.metric_type.value,
            status=stats.status,
            latency_stats=stats,
            error_count=self._error_count,
            last_error=self._last_error,
            last_error_time=self._last_error_time,
        )

    def reset(self) -> None:
        """Reset all tracked data."""
        self._samples.clear()
        self._total_count = 0
        self._error_count = 0
        self._last_error = None
        self._last_error_time = None
        self._last_recorded = None


# ============================================================================
# SELF MONITOR
# ============================================================================


class SelfMonitor:
    """Self-monitoring for the observability connector.

    Tracks internal metrics and emits health spans periodically.
    """

    def __init__(self, config: Optional[SelfMonitorConfig] = None):
        """Initialize the self monitor.

        Args:
            config: Configuration options
        """
        self.config = config or SelfMonitorConfig()
        self._start_time = time.time()
        self._health_span_count = 0
        self._health_task: Optional[asyncio.Task] = None
        self._running = False

        # Initialize latency trackers for each metric type
        self._trackers: Dict[MetricType, LatencyTracker] = {}
        for metric_type in MetricType:
            thresholds = self.config.thresholds.get(
                metric_type,
                LatencyThresholds(),
            )
            self._trackers[metric_type] = LatencyTracker(
                metric_type=metric_type,
                thresholds=thresholds,
                window_size=self.config.window_size,
                min_samples=self.config.min_samples_for_stats,
            )

        # Callbacks for span emission
        self._span_emitter: Optional[Callable] = None

    def set_span_emitter(self, emitter: Callable) -> None:
        """Set the callback function for emitting health spans.

        Args:
            emitter: Async function that accepts span data dict
        """
        self._span_emitter = emitter

    # -------------------------------------------------------------------------
    # Latency Recording Methods
    # -------------------------------------------------------------------------

    def record_span_emission_latency(self, latency_ms: float) -> None:
        """Record span emission latency.

        Args:
            latency_ms: Time taken to emit span in milliseconds
        """
        self._trackers[MetricType.SPAN_EMISSION].record(latency_ms)

    def record_opik_latency(self, latency_ms: float) -> None:
        """Record Opik API call latency.

        Args:
            latency_ms: Time taken for Opik API call in milliseconds
        """
        self._trackers[MetricType.OPIK_API].record(latency_ms)

    def record_database_latency(self, latency_ms: float) -> None:
        """Record database write latency.

        Args:
            latency_ms: Time taken for database write in milliseconds
        """
        self._trackers[MetricType.DATABASE_WRITE].record(latency_ms)

    def record_batch_flush_latency(self, latency_ms: float) -> None:
        """Record batch flush latency.

        Args:
            latency_ms: Time taken to flush batch in milliseconds
        """
        self._trackers[MetricType.BATCH_FLUSH].record(latency_ms)

    def record_cache_latency(self, latency_ms: float) -> None:
        """Record cache operation latency.

        Args:
            latency_ms: Time taken for cache operation in milliseconds
        """
        self._trackers[MetricType.CACHE_OPERATION].record(latency_ms)

    # -------------------------------------------------------------------------
    # Error Recording Methods
    # -------------------------------------------------------------------------

    def record_span_emission_error(self, error: str) -> None:
        """Record a span emission error."""
        self._trackers[MetricType.SPAN_EMISSION].record_error(error)

    def record_opik_error(self, error: str) -> None:
        """Record an Opik API error."""
        self._trackers[MetricType.OPIK_API].record_error(error)

    def record_database_error(self, error: str) -> None:
        """Record a database write error."""
        self._trackers[MetricType.DATABASE_WRITE].record_error(error)

    def record_batch_flush_error(self, error: str) -> None:
        """Record a batch flush error."""
        self._trackers[MetricType.BATCH_FLUSH].record_error(error)

    def record_cache_error(self, error: str) -> None:
        """Record a cache operation error."""
        self._trackers[MetricType.CACHE_OPERATION].record_error(error)

    # -------------------------------------------------------------------------
    # Health Status Methods
    # -------------------------------------------------------------------------

    def get_health_status(self) -> OverallHealth:
        """Get current overall health status.

        Returns:
            OverallHealth with all component statuses
        """
        components: Dict[str, ComponentHealth] = {}
        alerts: List[str] = []
        worst_status = HealthStatus.HEALTHY

        for metric_type, tracker in self._trackers.items():
            health = tracker.get_component_health()
            components[metric_type.value] = health

            # Collect alerts for degraded/unhealthy components
            if health.status == HealthStatus.DEGRADED:
                alerts.append(
                    f"{metric_type.value}: Latency degraded "
                    f"(avg: {health.latency_stats.avg_ms:.1f}ms)"
                )
                if worst_status == HealthStatus.HEALTHY:
                    worst_status = HealthStatus.DEGRADED

            elif health.status == HealthStatus.UNHEALTHY:
                alerts.append(
                    f"{metric_type.value}: Latency critical "
                    f"(avg: {health.latency_stats.avg_ms:.1f}ms)"
                )
                worst_status = HealthStatus.UNHEALTHY

            # Check for high error rates
            if health.error_count > 0:
                total = health.latency_stats.count + health.error_count
                if total > 0:
                    error_rate = health.error_count / total
                    if error_rate > 0.1:  # More than 10% errors
                        alerts.append(f"{metric_type.value}: High error rate ({error_rate:.1%})")
                        if worst_status != HealthStatus.UNHEALTHY:
                            worst_status = HealthStatus.DEGRADED

        # Check if we have enough data
        total_samples = sum(tracker.get_stats().count for tracker in self._trackers.values())
        if total_samples < self.config.min_samples_for_stats:
            worst_status = HealthStatus.UNKNOWN

        uptime = time.time() - self._start_time

        return OverallHealth(
            status=worst_status,
            components=components,
            uptime_seconds=uptime,
            last_health_check=datetime.now(timezone.utc),
            health_span_count=self._health_span_count,
            alerts=alerts,
        )

    def get_latency_stats(self, metric_type: MetricType) -> LatencyStats:
        """Get latency statistics for a specific metric type.

        Args:
            metric_type: Type of metric to get stats for

        Returns:
            LatencyStats for the specified metric
        """
        return self._trackers[metric_type].get_stats()

    def get_all_latency_stats(self) -> Dict[str, LatencyStats]:
        """Get latency statistics for all metric types.

        Returns:
            Dictionary of metric type to LatencyStats
        """
        return {
            metric_type.value: tracker.get_stats()
            for metric_type, tracker in self._trackers.items()
        }

    # -------------------------------------------------------------------------
    # Health Span Emission
    # -------------------------------------------------------------------------

    async def start_health_emission(self) -> None:
        """Start background health span emission."""
        if self._running:
            logger.warning("Health emission already running")
            return

        self._running = True
        self._health_task = asyncio.create_task(self._health_emission_loop())
        logger.info(
            f"Started health emission (interval: {self.config.health_emission_interval_seconds}s)"
        )

    async def stop_health_emission(self) -> None:
        """Stop background health span emission."""
        self._running = False
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None
        logger.info("Stopped health emission")

    async def _health_emission_loop(self) -> None:
        """Background loop for emitting health spans."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_emission_interval_seconds)

                if not self._running:
                    break

                await self._emit_health_span()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health emission loop: {e}")

    async def _emit_health_span(self) -> None:
        """Emit a single health span."""
        if not self.config.emit_health_spans:
            return

        health = self.get_health_status()
        self._health_span_count += 1

        # Log if degraded
        if self.config.log_degraded_status and health.status != HealthStatus.HEALTHY:
            logger.warning(f"Observability health: {health.status.value}, alerts: {health.alerts}")

        # Emit span if emitter is configured
        if self._span_emitter:
            try:
                span_data = {
                    "agent_name": "observability_connector",
                    "operation": "health_check",
                    "span_type": "general",
                    "status": "completed",
                    "metadata": {
                        "health_status": health.status.value,
                        "uptime_seconds": health.uptime_seconds,
                        "health_span_count": self._health_span_count,
                        "alerts": health.alerts,
                        "components": {k: v.to_dict() for k, v in health.components.items()},
                    },
                }
                await self._span_emitter(span_data)
            except Exception as e:
                logger.error(f"Failed to emit health span: {e}")

    async def emit_health_span_now(self) -> OverallHealth:
        """Emit a health span immediately and return health status.

        Returns:
            Current overall health status
        """
        await self._emit_health_span()
        return self.get_health_status()

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all tracked metrics."""
        for tracker in self._trackers.values():
            tracker.reset()
        self._health_span_count = 0
        self._start_time = time.time()
        logger.info("Self-monitor metrics reset")

    @property
    def is_running(self) -> bool:
        """Check if health emission is running."""
        return self._running

    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self._start_time


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_self_monitor: Optional[SelfMonitor] = None


def get_self_monitor(
    config: Optional[SelfMonitorConfig] = None,
    force_new: bool = False,
) -> SelfMonitor:
    """Get singleton SelfMonitor instance.

    Args:
        config: Configuration (only used on first call or force_new)
        force_new: Force creation of new instance

    Returns:
        SelfMonitor singleton instance
    """
    global _self_monitor

    if _self_monitor is None or force_new:
        _self_monitor = SelfMonitor(config)

    return _self_monitor


def reset_self_monitor() -> None:
    """Reset singleton instance (for testing)."""
    global _self_monitor
    if _self_monitor and _self_monitor.is_running:
        # Note: Can't await here, so just mark as not running
        _self_monitor._running = False
    _self_monitor = None


# ============================================================================
# CONTEXT MANAGER FOR LATENCY TRACKING
# ============================================================================


class LatencyContext:
    """Context manager for tracking operation latency.

    Usage:
        monitor = get_self_monitor()
        with LatencyContext(monitor, MetricType.OPIK_API) as ctx:
            result = await opik_call()
        # Latency automatically recorded
    """

    def __init__(
        self,
        monitor: SelfMonitor,
        metric_type: MetricType,
        record_errors: bool = True,
    ):
        """Initialize the latency context.

        Args:
            monitor: SelfMonitor instance to record to
            metric_type: Type of metric being tracked
            record_errors: Whether to record errors on exception
        """
        self.monitor = monitor
        self.metric_type = metric_type
        self.record_errors = record_errors
        self._start_time: float = 0.0
        self._latency_ms: float = 0.0

    def __enter__(self) -> "LatencyContext":
        """Start timing."""
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and record latency."""
        self._latency_ms = (time.perf_counter() - self._start_time) * 1000

        # Record latency
        tracker = self.monitor._trackers.get(self.metric_type)
        if tracker:
            tracker.record(self._latency_ms)

            # Record error if exception occurred
            if exc_type is not None and self.record_errors:
                error_msg = f"{exc_type.__name__}: {exc_val}"
                tracker.record_error(error_msg)

        # Don't suppress exceptions (implicitly returns None)

    @property
    def latency_ms(self) -> float:
        """Get the recorded latency in milliseconds."""
        return self._latency_ms


class AsyncLatencyContext:
    """Async context manager for tracking operation latency.

    Usage:
        monitor = get_self_monitor()
        async with AsyncLatencyContext(monitor, MetricType.OPIK_API) as ctx:
            result = await opik_call()
        # Latency automatically recorded
    """

    def __init__(
        self,
        monitor: SelfMonitor,
        metric_type: MetricType,
        record_errors: bool = True,
    ):
        """Initialize the async latency context.

        Args:
            monitor: SelfMonitor instance to record to
            metric_type: Type of metric being tracked
            record_errors: Whether to record errors on exception
        """
        self.monitor = monitor
        self.metric_type = metric_type
        self.record_errors = record_errors
        self._start_time: float = 0.0
        self._latency_ms: float = 0.0

    async def __aenter__(self) -> "AsyncLatencyContext":
        """Start timing."""
        self._start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Stop timing and record latency."""
        self._latency_ms = (time.perf_counter() - self._start_time) * 1000

        # Record latency
        tracker = self.monitor._trackers.get(self.metric_type)
        if tracker:
            tracker.record(self._latency_ms)

            # Record error if exception occurred
            if exc_type is not None and self.record_errors:
                error_msg = f"{exc_type.__name__}: {exc_val}"
                tracker.record_error(error_msg)

        return False  # Don't suppress exceptions

    @property
    def latency_ms(self) -> float:
        """Get the recorded latency in milliseconds."""
        return self._latency_ms
