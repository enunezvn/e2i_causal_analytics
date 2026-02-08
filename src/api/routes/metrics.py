"""Prometheus Metrics Endpoint.

Provides a /metrics endpoint for Prometheus scraping with application metrics.

Quick Win QW1 from observability audit remediation plan:
- Exposes request counts, latencies, and error rates
- Provides model serving metrics from BentoML
- Includes system health gauges

Version: 1.0.0
"""

import logging
from typing import TYPE_CHECKING, Any, Dict

from fastapi import APIRouter
from fastapi.responses import Response

logger = logging.getLogger(__name__)

# Type hints for prometheus_client types (only for type checking)
if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# Try to import prometheus_client
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        multiprocess,  # noqa: F401
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"
    REGISTRY = None  # type: ignore[assignment]
    logger.warning("prometheus_client not installed, /metrics endpoint will return empty")

# Create router - no prefix since /metrics is typically at root level
router = APIRouter(tags=["Metrics"])

# =============================================================================
# Application Metrics Registry
# =============================================================================

# Use a separate registry to avoid conflicts with default registry
_metrics_registry: "CollectorRegistry | None" = None
_metrics_initialized = False

# Metric instances
_request_counter: "Counter | None" = None
_request_latency: "Histogram | None" = None
_active_requests: "Gauge | None" = None
_error_counter: "Counter | None" = None
_agent_invocations: "Counter | None" = None
_health_gauge: "Gauge | None" = None


def _init_metrics() -> None:
    """Initialize application metrics (lazy initialization)."""
    global _metrics_registry, _metrics_initialized
    global _request_counter, _request_latency, _active_requests
    global _error_counter, _agent_invocations, _health_gauge

    if _metrics_initialized or not PROMETHEUS_AVAILABLE:
        return

    _metrics_registry = CollectorRegistry()

    # Request metrics
    _request_counter = Counter(
        "e2i_api_requests_total",
        "Total number of API requests",
        ["method", "endpoint", "status_code"],
        registry=_metrics_registry,
    )

    _request_latency = Histogram(
        "e2i_api_request_latency_seconds",
        "Request latency in seconds",
        ["method", "endpoint"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        registry=_metrics_registry,
    )

    _active_requests = Gauge(
        "e2i_api_active_requests",
        "Number of active requests being processed",
        ["method"],
        registry=_metrics_registry,
    )

    # Error metrics
    _error_counter = Counter(
        "e2i_api_errors_total",
        "Total number of API errors",
        ["method", "endpoint", "error_type"],
        registry=_metrics_registry,
    )

    # Agent metrics
    _agent_invocations = Counter(
        "e2i_agent_invocations_total",
        "Total number of agent invocations",
        ["agent_name", "tier", "status"],
        registry=_metrics_registry,
    )

    # Health gauges
    _health_gauge = Gauge(
        "e2i_component_health",
        "Health status of system components (1=healthy, 0=unhealthy)",
        ["component"],
        registry=_metrics_registry,
    )

    _metrics_initialized = True
    logger.info("Prometheus metrics initialized")


def get_metrics_registry() -> "CollectorRegistry | None":
    """Get the metrics registry, initializing if needed."""
    if not _metrics_initialized:
        _init_metrics()
    return _metrics_registry


# =============================================================================
# Metrics Recording Functions (for use by middleware/handlers)
# =============================================================================


def record_request(method: str, endpoint: str, status_code: int, latency: float) -> None:
    """Record an API request metric."""
    if not PROMETHEUS_AVAILABLE or not _metrics_initialized:
        return
    if _request_counter is None or _request_latency is None:
        return

    try:
        _request_counter.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()

        _request_latency.labels(
            method=method,
            endpoint=endpoint,
        ).observe(latency)
    except Exception as e:
        logger.debug(f"Failed to record request metric: {e}")


def record_error(method: str, endpoint: str, error_type: str) -> None:
    """Record an API error metric."""
    if not PROMETHEUS_AVAILABLE or not _metrics_initialized:
        return
    if _error_counter is None:
        return

    try:
        _error_counter.labels(
            method=method,
            endpoint=endpoint,
            error_type=error_type,
        ).inc()
    except Exception as e:
        logger.debug(f"Failed to record error metric: {e}")


def record_agent_invocation(agent_name: str, tier: str, status: str) -> None:
    """Record an agent invocation metric."""
    if not PROMETHEUS_AVAILABLE or not _metrics_initialized:
        return
    if _agent_invocations is None:
        return

    try:
        _agent_invocations.labels(
            agent_name=agent_name,
            tier=tier,
            status=status,
        ).inc()
    except Exception as e:
        logger.debug(f"Failed to record agent metric: {e}")


def set_component_health(component: str, healthy: bool) -> None:
    """Set health status for a component."""
    if not PROMETHEUS_AVAILABLE or not _metrics_initialized:
        return
    if _health_gauge is None:
        return

    try:
        _health_gauge.labels(component=component).set(1 if healthy else 0)
    except Exception as e:
        logger.debug(f"Failed to set health metric: {e}")


# =============================================================================
# Metrics Endpoint
# =============================================================================


@router.get("/metrics", response_class=Response)
async def metrics_endpoint() -> Response:
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping.
    Includes:
    - API request counts and latencies
    - Error rates by type
    - Agent invocation counts
    - Component health status

    Returns:
        Response with Prometheus-formatted metrics
    """
    if not PROMETHEUS_AVAILABLE:
        return Response(
            content="# prometheus_client not installed\n",
            media_type="text/plain",
        )

    # Ensure metrics are initialized
    registry = get_metrics_registry()
    if registry is None:
        return Response(
            content="# metrics not initialized\n",
            media_type="text/plain",
        )

    try:
        # Generate metrics output
        metrics_output = generate_latest(registry)

        return Response(
            content=metrics_output,
            media_type=CONTENT_TYPE_LATEST,
        )
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return Response(
            content=f"# error generating metrics: {e}\n",
            media_type="text/plain",
            status_code=500,
        )


@router.get("/metrics/health")
async def metrics_health() -> Dict[str, Any]:
    """
    Check if metrics collection is operational.

    Returns:
        Health status of the metrics subsystem
    """
    return {
        "status": "healthy" if PROMETHEUS_AVAILABLE and _metrics_initialized else "degraded",
        "prometheus_available": PROMETHEUS_AVAILABLE,
        "metrics_initialized": _metrics_initialized,
        "registry_collectors": len(list(_metrics_registry._names_to_collectors.keys()))
        if _metrics_registry
        else 0,
    }
