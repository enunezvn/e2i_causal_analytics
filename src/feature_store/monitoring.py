"""
Feature Store Monitoring and Metrics.

Provides latency tracking, cache hit metrics, and observability
for feature retrieval operations.

Phase 4 - G17: Feature retrieval latency tracking
Version: 1.0.0
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# Type hints for prometheus_client types (only for type checking)
if TYPE_CHECKING:
    from prometheus_client import Counter, Histogram, Summary

# Try to import prometheus_client
try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Summary,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed, feature store metrics disabled")


# =============================================================================
# Metrics Registry
# =============================================================================

_fs_metrics_initialized = False
_fs_cache_hits: "Optional[Counter]" = None
_fs_cache_misses: "Optional[Counter]" = None
_fs_retrieval_latency: "Optional[Histogram]" = None
_fs_cache_latency: "Optional[Histogram]" = None
_fs_db_latency: "Optional[Histogram]" = None
_fs_batch_size: "Optional[Histogram]" = None
_fs_feature_count: "Optional[Summary]" = None
_fs_stale_features: "Optional[Counter]" = None
_fs_errors: "Optional[Counter]" = None


def init_feature_store_metrics(registry: Optional[Any] = None) -> None:
    """
    Initialize feature store metrics.

    Args:
        registry: Optional CollectorRegistry. If None, uses default REGISTRY.
    """
    global _fs_metrics_initialized
    global _fs_cache_hits, _fs_cache_misses
    global _fs_retrieval_latency, _fs_cache_latency, _fs_db_latency
    global _fs_batch_size, _fs_feature_count, _fs_stale_features, _fs_errors

    if _fs_metrics_initialized or not PROMETHEUS_AVAILABLE:
        return

    kwargs = {"registry": registry} if registry else {}

    # Cache metrics
    _fs_cache_hits = Counter(
        "e2i_feature_store_cache_hits_total",
        "Total number of feature store cache hits",
        ["feature_group", "operation"],
        **kwargs,
    )

    _fs_cache_misses = Counter(
        "e2i_feature_store_cache_misses_total",
        "Total number of feature store cache misses",
        ["feature_group", "operation"],
        **kwargs,
    )

    # Latency metrics
    _fs_retrieval_latency = Histogram(
        "e2i_feature_store_retrieval_latency_seconds",
        "End-to-end feature retrieval latency in seconds",
        ["feature_group", "operation", "cache_hit"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        **kwargs,
    )

    _fs_cache_latency = Histogram(
        "e2i_feature_store_cache_latency_seconds",
        "Redis cache operation latency in seconds",
        ["operation"],
        buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
        **kwargs,
    )

    _fs_db_latency = Histogram(
        "e2i_feature_store_db_latency_seconds",
        "Database (Supabase) query latency in seconds",
        ["operation", "feature_group"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        **kwargs,
    )

    # Batch and count metrics
    _fs_batch_size = Histogram(
        "e2i_feature_store_batch_size",
        "Number of features in batch operations",
        ["operation"],
        buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
        **kwargs,
    )

    _fs_feature_count = Summary(
        "e2i_feature_store_features_retrieved",
        "Number of features retrieved per request",
        ["feature_group"],
        **kwargs,
    )

    # Quality metrics
    _fs_stale_features = Counter(
        "e2i_feature_store_stale_features_total",
        "Total number of stale features returned",
        ["feature_group"],
        **kwargs,
    )

    _fs_errors = Counter(
        "e2i_feature_store_errors_total",
        "Total number of feature store errors",
        ["operation", "error_type"],
        **kwargs,
    )

    _fs_metrics_initialized = True
    logger.info("Feature store metrics initialized")


# =============================================================================
# Latency Tracking Context Managers
# =============================================================================


@dataclass
class FeatureRetrievalMetrics:
    """Metrics collected during a feature retrieval operation."""

    feature_group: str = "unknown"
    operation: str = "get_entity_features"
    cache_hit: bool = False
    cache_latency_ms: float = 0.0
    db_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    feature_count: int = 0
    stale_count: int = 0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@contextmanager
def track_retrieval(
    feature_group: str = "unknown",
    operation: str = "get_entity_features",
) -> Generator[FeatureRetrievalMetrics, None, None]:
    """
    Context manager to track feature retrieval metrics.

    Usage:
        with track_retrieval("hcp_demographics") as metrics:
            # ... perform retrieval ...
            metrics.cache_hit = True
            metrics.feature_count = 10

    Args:
        feature_group: Name of the feature group being accessed
        operation: Type of operation (get_entity_features, get_historical, etc.)

    Yields:
        FeatureRetrievalMetrics object to populate during retrieval
    """
    metrics = FeatureRetrievalMetrics(
        feature_group=feature_group,
        operation=operation,
    )
    start_time = time.perf_counter()

    try:
        yield metrics
    except Exception as e:
        metrics.error = type(e).__name__
        raise
    finally:
        # Calculate total latency
        end_time = time.perf_counter()
        metrics.total_latency_ms = (end_time - start_time) * 1000

        # Record metrics
        _record_retrieval_metrics(metrics)


@contextmanager
def track_cache_operation(operation: str = "get") -> Generator[Dict[str, Any], None, None]:
    """
    Context manager to track cache operation latency.

    Args:
        operation: Cache operation type (get, set, delete)

    Yields:
        Dict to store cache operation metadata
    """
    context: Dict[str, Any] = {"hit": False, "latency_ms": 0.0}
    start_time = time.perf_counter()

    try:
        yield context
    finally:
        end_time = time.perf_counter()
        latency_seconds = end_time - start_time
        context["latency_ms"] = latency_seconds * 1000

        if PROMETHEUS_AVAILABLE and _fs_cache_latency:
            try:
                _fs_cache_latency.labels(operation=operation).observe(latency_seconds)
            except Exception as e:
                logger.debug(f"Failed to record cache latency: {e}")


@contextmanager
def track_db_operation(
    operation: str = "query",
    feature_group: str = "unknown",
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager to track database operation latency.

    Args:
        operation: Database operation type (query, rpc, etc.)
        feature_group: Feature group being accessed

    Yields:
        Dict to store operation metadata
    """
    context: Dict[str, Any] = {"latency_ms": 0.0, "row_count": 0}
    start_time = time.perf_counter()

    try:
        yield context
    finally:
        end_time = time.perf_counter()
        latency_seconds = end_time - start_time
        context["latency_ms"] = latency_seconds * 1000

        if PROMETHEUS_AVAILABLE and _fs_db_latency:
            try:
                _fs_db_latency.labels(
                    operation=operation,
                    feature_group=feature_group,
                ).observe(latency_seconds)
            except Exception as e:
                logger.debug(f"Failed to record db latency: {e}")


# =============================================================================
# Metrics Recording Functions
# =============================================================================


def _record_retrieval_metrics(metrics: FeatureRetrievalMetrics) -> None:
    """Record retrieval metrics to Prometheus."""
    if not PROMETHEUS_AVAILABLE or not _fs_metrics_initialized:
        return

    try:
        # Record cache hit/miss
        if metrics.cache_hit and _fs_cache_hits:
            _fs_cache_hits.labels(
                feature_group=metrics.feature_group,
                operation=metrics.operation,
            ).inc()
        elif _fs_cache_misses:
            _fs_cache_misses.labels(
                feature_group=metrics.feature_group,
                operation=metrics.operation,
            ).inc()

        # Record total retrieval latency
        if _fs_retrieval_latency:
            _fs_retrieval_latency.labels(
                feature_group=metrics.feature_group,
                operation=metrics.operation,
                cache_hit=str(metrics.cache_hit).lower(),
            ).observe(metrics.total_latency_ms / 1000)

        # Record feature count
        if _fs_feature_count and metrics.feature_count > 0:
            _fs_feature_count.labels(
                feature_group=metrics.feature_group,
            ).observe(metrics.feature_count)

        # Record stale features
        if _fs_stale_features and metrics.stale_count > 0:
            _fs_stale_features.labels(
                feature_group=metrics.feature_group,
            ).inc(metrics.stale_count)

        # Record errors
        if _fs_errors and metrics.error:
            _fs_errors.labels(
                operation=metrics.operation,
                error_type=metrics.error,
            ).inc()

    except Exception as e:
        logger.debug(f"Failed to record retrieval metrics: {e}")


def record_cache_hit(feature_group: str, operation: str = "get_entity_features") -> None:
    """Record a cache hit."""
    if PROMETHEUS_AVAILABLE and _fs_cache_hits:
        try:
            _fs_cache_hits.labels(
                feature_group=feature_group,
                operation=operation,
            ).inc()
        except Exception as e:
            logger.debug(f"Failed to record cache hit: {e}")


def record_cache_miss(feature_group: str, operation: str = "get_entity_features") -> None:
    """Record a cache miss."""
    if PROMETHEUS_AVAILABLE and _fs_cache_misses:
        try:
            _fs_cache_misses.labels(
                feature_group=feature_group,
                operation=operation,
            ).inc()
        except Exception as e:
            logger.debug(f"Failed to record cache miss: {e}")


def record_batch_size(operation: str, size: int) -> None:
    """Record batch operation size."""
    if PROMETHEUS_AVAILABLE and _fs_batch_size:
        try:
            _fs_batch_size.labels(operation=operation).observe(size)
        except Exception as e:
            logger.debug(f"Failed to record batch size: {e}")


def record_error(operation: str, error_type: str) -> None:
    """Record a feature store error."""
    if PROMETHEUS_AVAILABLE and _fs_errors:
        try:
            _fs_errors.labels(
                operation=operation,
                error_type=error_type,
            ).inc()
        except Exception as e:
            logger.debug(f"Failed to record error: {e}")


# =============================================================================
# Latency Statistics
# =============================================================================


@dataclass
class LatencyStats:
    """Latency statistics for feature retrieval."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    avg_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    cache_hit_rate: float = 0.0


class LatencyTracker:
    """
    In-memory latency tracker for feature store operations.

    Useful for debugging and real-time dashboards when Prometheus
    isn't available or for more detailed local analysis.
    """

    def __init__(self, max_samples: int = 1000):
        """
        Initialize latency tracker.

        Args:
            max_samples: Maximum number of samples to keep (circular buffer)
        """
        self.max_samples = max_samples
        self.samples: List[FeatureRetrievalMetrics] = []
        self._total_cache_hits = 0
        self._total_cache_misses = 0

    def record(self, metrics: FeatureRetrievalMetrics) -> None:
        """Record a retrieval operation."""
        self.samples.append(metrics)
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)

        if metrics.cache_hit:
            self._total_cache_hits += 1
        else:
            self._total_cache_misses += 1

    def get_stats(self, feature_group: Optional[str] = None) -> LatencyStats:
        """
        Get latency statistics.

        Args:
            feature_group: Optional filter by feature group

        Returns:
            LatencyStats with computed statistics
        """
        samples = self.samples
        if feature_group:
            samples = [s for s in samples if s.feature_group == feature_group]

        if not samples:
            return LatencyStats()

        latencies = [s.total_latency_ms for s in samples]
        latencies.sort()

        n = len(latencies)
        cache_hits = sum(1 for s in samples if s.cache_hit)

        return LatencyStats(
            count=n,
            total_ms=sum(latencies),
            min_ms=min(latencies),
            max_ms=max(latencies),
            avg_ms=sum(latencies) / n,
            p50_ms=latencies[int(n * 0.50)],
            p95_ms=latencies[int(n * 0.95)] if n >= 20 else latencies[-1],
            p99_ms=latencies[int(n * 0.99)] if n >= 100 else latencies[-1],
            cache_hit_rate=cache_hits / n if n > 0 else 0.0,
        )

    def get_recent(self, count: int = 10) -> List[FeatureRetrievalMetrics]:
        """Get most recent samples."""
        return self.samples[-count:]

    def clear(self) -> None:
        """Clear all samples."""
        self.samples.clear()
        self._total_cache_hits = 0
        self._total_cache_misses = 0


# Global latency tracker instance
_latency_tracker: Optional[LatencyTracker] = None


def get_latency_tracker() -> LatencyTracker:
    """Get or create the global latency tracker."""
    global _latency_tracker
    if _latency_tracker is None:
        _latency_tracker = LatencyTracker()
    return _latency_tracker


# =============================================================================
# Decorator for Method Instrumentation
# =============================================================================


def track_feature_retrieval(
    feature_group_arg: str = "feature_group",
    operation: str = "get_entity_features",
):
    """
    Decorator to track feature retrieval latency.

    Args:
        feature_group_arg: Name of the argument containing feature group
        operation: Operation name for metrics

    Example:
        @track_feature_retrieval(feature_group_arg="feature_group")
        def get_entity_features(self, entity_values, feature_group=None):
            ...
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Try to extract feature group from kwargs or args
            fg = kwargs.get(feature_group_arg, "unknown")
            if fg is None:
                fg = "unknown"

            with track_retrieval(feature_group=fg, operation=operation) as metrics:
                result = func(*args, **kwargs)

                # Try to extract feature count from result
                if hasattr(result, "features"):
                    metrics.feature_count = len(result.features)
                elif isinstance(result, list):
                    metrics.feature_count = len(result)

                return result

        return wrapper

    return decorator
