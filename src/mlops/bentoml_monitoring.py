"""BentoML Health Monitoring and Metrics Collection.

This module provides health monitoring, metrics collection, and alerting
for BentoML model serving services.

Features:
- Service health checks
- Performance metrics tracking
- Latency monitoring
- Error rate tracking
- Prometheus metrics export
- Integration with Opik for observability

Version: 1.0.0
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CollectorRegistry = None

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================


class ServiceStatus(str, Enum):
    """Service health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    service_name: str
    status: ServiceStatus
    timestamp: datetime
    latency_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class MetricSnapshot:
    """Snapshot of service metrics."""

    service_name: str
    timestamp: datetime
    prediction_count: int
    error_count: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    memory_mb: float
    cpu_percent: float


@dataclass
class Alert:
    """Alert for service issues."""

    service_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


# ============================================================================
# Prometheus Metrics
# ============================================================================


class PrometheusMetrics:
    """Prometheus metrics for BentoML services."""

    def __init__(self, namespace: str = "e2i_model", registry: Optional[Any] = None):
        """Initialize Prometheus metrics.

        Args:
            namespace: Metric namespace prefix
            registry: Optional custom CollectorRegistry (for testing)
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus_client not installed, metrics disabled")
            self._enabled = False
            return

        self._enabled = True
        self.namespace = namespace
        self._registry = registry or CollectorRegistry()

        # Request metrics
        self.request_counter = Counter(
            f"{namespace}_requests_total",
            "Total number of prediction requests",
            ["service", "endpoint", "status"],
            registry=self._registry,
        )

        self.request_latency = Histogram(
            f"{namespace}_request_latency_seconds",
            "Request latency in seconds",
            ["service", "endpoint"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self._registry,
        )

        # Prediction metrics
        self.prediction_counter = Counter(
            f"{namespace}_predictions_total",
            "Total number of predictions made",
            ["service", "model_type"],
            registry=self._registry,
        )

        self.batch_size = Histogram(
            f"{namespace}_batch_size",
            "Batch size of prediction requests",
            ["service"],
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
            registry=self._registry,
        )

        # Error metrics
        self.error_counter = Counter(
            f"{namespace}_errors_total",
            "Total number of errors",
            ["service", "error_type"],
            registry=self._registry,
        )

        # Resource metrics
        self.memory_gauge = Gauge(
            f"{namespace}_memory_usage_bytes",
            "Memory usage in bytes",
            ["service"],
            registry=self._registry,
        )

        self.model_load_time = Gauge(
            f"{namespace}_model_load_seconds",
            "Time taken to load model",
            ["service", "model_tag"],
            registry=self._registry,
        )

        # Health metrics
        self.health_status = Gauge(
            f"{namespace}_health_status",
            "Health status (1=healthy, 0=unhealthy)",
            ["service"],
            registry=self._registry,
        )

    def record_request(
        self,
        service: str,
        endpoint: str,
        status: str,
        latency_seconds: float,
    ) -> None:
        """Record a request metric."""
        if not getattr(self, "_enabled", False):
            return

        self.request_counter.labels(service=service, endpoint=endpoint, status=status).inc()
        self.request_latency.labels(service=service, endpoint=endpoint).observe(latency_seconds)

    def record_prediction(
        self,
        service: str,
        model_type: str,
        batch_size: int,
    ) -> None:
        """Record prediction metrics."""
        if not getattr(self, "_enabled", False):
            return

        self.prediction_counter.labels(service=service, model_type=model_type).inc(batch_size)
        self.batch_size.labels(service=service).observe(batch_size)

    def record_error(
        self,
        service: str,
        error_type: str,
    ) -> None:
        """Record an error."""
        if not getattr(self, "_enabled", False):
            return

        self.error_counter.labels(service=service, error_type=error_type).inc()

    def set_health_status(
        self,
        service: str,
        healthy: bool,
    ) -> None:
        """Set health status gauge."""
        if not getattr(self, "_enabled", False):
            return

        self.health_status.labels(service=service).set(1 if healthy else 0)

    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format."""
        if not getattr(self, "_enabled", False):
            return b""

        return generate_latest(self._registry)


# ============================================================================
# Health Monitor
# ============================================================================


class BentoMLHealthMonitor:
    """Health monitor for BentoML services.

    This class provides:
    - Periodic health checks
    - Metric collection
    - Alerting on threshold breaches
    - Integration with Opik observability
    """

    def __init__(
        self,
        check_interval_seconds: int = 30,
        alert_handlers: Optional[List[Callable[[Alert], None]]] = None,
    ):
        """Initialize health monitor.

        Args:
            check_interval_seconds: Interval between health checks
            alert_handlers: Callbacks for alerts
        """
        self.check_interval = check_interval_seconds
        self.alert_handlers = alert_handlers or []
        self.metrics = PrometheusMetrics()

        # Service registry
        self._services: Dict[str, Dict[str, Any]] = {}

        # Health history
        self._health_history: Dict[str, List[HealthCheckResult]] = {}

        # Thresholds
        self._thresholds = {
            "latency_warning_ms": 500,
            "latency_error_ms": 2000,
            "error_rate_warning": 0.05,
            "error_rate_error": 0.10,
            "health_check_timeout_ms": 5000,
        }

        # Running state
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

    def register_service(
        self,
        name: str,
        url: str,
        model_type: str = "classification",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a service for monitoring.

        Args:
            name: Service name
            url: Service URL
            model_type: Type of model
            metadata: Additional metadata
        """
        self._services[name] = {
            "url": url.rstrip("/"),
            "model_type": model_type,
            "metadata": metadata or {},
            "registered_at": datetime.now(timezone.utc),
        }
        self._health_history[name] = []
        logger.info(f"Registered service: {name} at {url}")

    def unregister_service(self, name: str) -> None:
        """Unregister a service.

        Args:
            name: Service name
        """
        if name in self._services:
            del self._services[name]
            del self._health_history[name]
            logger.info(f"Unregistered service: {name}")

    def set_threshold(self, name: str, value: float) -> None:
        """Set an alert threshold.

        Args:
            name: Threshold name
            value: Threshold value
        """
        if name in self._thresholds:
            self._thresholds[name] = value
            logger.info(f"Set threshold {name} = {value}")

    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler.

        Args:
            handler: Callback function for alerts
        """
        self.alert_handlers.append(handler)

    async def check_health(self, service_name: str) -> HealthCheckResult:
        """Check health of a service.

        Args:
            service_name: Name of the service

        Returns:
            Health check result
        """
        import aiohttp

        if service_name not in self._services:
            return HealthCheckResult(
                service_name=service_name,
                status=ServiceStatus.UNKNOWN,
                timestamp=datetime.now(timezone.utc),
                latency_ms=0,
                error="Service not registered",
            )

        service = self._services[service_name]
        health_url = f"{service['url']}/health"

        start_time = time.time()
        try:
            timeout = aiohttp.ClientTimeout(
                total=self._thresholds["health_check_timeout_ms"] / 1000
            )
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_url) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    data = await response.json()

                    if response.status == 200:
                        status = ServiceStatus.HEALTHY
                    elif response.status < 500:
                        status = ServiceStatus.DEGRADED
                    else:
                        status = ServiceStatus.UNHEALTHY

                    result = HealthCheckResult(
                        service_name=service_name,
                        status=status,
                        timestamp=datetime.now(timezone.utc),
                        latency_ms=latency_ms,
                        details=data,
                    )

        except asyncio.TimeoutError:
            result = HealthCheckResult(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                timestamp=datetime.now(timezone.utc),
                latency_ms=self._thresholds["health_check_timeout_ms"],
                error="Health check timed out",
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                timestamp=datetime.now(timezone.utc),
                latency_ms=latency_ms,
                error=str(e),
            )

        # Store history
        self._health_history[service_name].append(result)
        if len(self._health_history[service_name]) > 100:
            self._health_history[service_name] = self._health_history[service_name][-100:]

        # Update metrics
        self.metrics.set_health_status(
            service_name,
            result.status == ServiceStatus.HEALTHY,
        )

        # Check for alerts
        await self._check_alerts(service_name, result)

        return result

    async def get_metrics(self, service_name: str) -> Optional[MetricSnapshot]:
        """Get metrics from a service.

        Args:
            service_name: Name of the service

        Returns:
            Metric snapshot or None if unavailable
        """
        import aiohttp

        if service_name not in self._services:
            return None

        service = self._services[service_name]
        metrics_url = f"{service['url']}/metrics"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(metrics_url) as response:
                    if response.status != 200:
                        return None

                    data = await response.json()

                    return MetricSnapshot(
                        service_name=service_name,
                        timestamp=datetime.now(timezone.utc),
                        prediction_count=data.get("prediction_count", 0),
                        error_count=data.get("error_count", 0),
                        avg_latency_ms=data.get("average_latency_ms", 0),
                        p50_latency_ms=data.get("p50_latency_ms", 0),
                        p95_latency_ms=data.get("p95_latency_ms", 0),
                        p99_latency_ms=data.get("p99_latency_ms", 0),
                        memory_mb=data.get("memory_mb", 0),
                        cpu_percent=data.get("cpu_percent", 0),
                    )

        except Exception as e:
            logger.error(f"Failed to get metrics for {service_name}: {e}")
            return None

    async def _check_alerts(
        self,
        service_name: str,
        health_result: HealthCheckResult,
    ) -> None:
        """Check and emit alerts based on health results.

        Args:
            service_name: Service name
            health_result: Health check result
        """
        alerts = []

        # Check health status
        if health_result.status == ServiceStatus.UNHEALTHY:
            alerts.append(
                Alert(
                    service_name=service_name,
                    severity=AlertSeverity.ERROR,
                    message=f"Service {service_name} is unhealthy: {health_result.error}",
                    timestamp=health_result.timestamp,
                )
            )

        elif health_result.status == ServiceStatus.DEGRADED:
            alerts.append(
                Alert(
                    service_name=service_name,
                    severity=AlertSeverity.WARNING,
                    message=f"Service {service_name} is degraded",
                    timestamp=health_result.timestamp,
                )
            )

        # Check latency
        if health_result.latency_ms > self._thresholds["latency_error_ms"]:
            alerts.append(
                Alert(
                    service_name=service_name,
                    severity=AlertSeverity.ERROR,
                    message=f"High latency for {service_name}",
                    timestamp=health_result.timestamp,
                    metric_name="latency_ms",
                    metric_value=health_result.latency_ms,
                    threshold=self._thresholds["latency_error_ms"],
                )
            )

        elif health_result.latency_ms > self._thresholds["latency_warning_ms"]:
            alerts.append(
                Alert(
                    service_name=service_name,
                    severity=AlertSeverity.WARNING,
                    message=f"Elevated latency for {service_name}",
                    timestamp=health_result.timestamp,
                    metric_name="latency_ms",
                    metric_value=health_result.latency_ms,
                    threshold=self._thresholds["latency_warning_ms"],
                )
            )

        # Emit alerts
        for alert in alerts:
            await self._emit_alert(alert)

    async def _emit_alert(self, alert: Alert) -> None:
        """Emit an alert to all handlers.

        Args:
            alert: Alert to emit
        """
        logger.warning(f"Alert: {alert.severity.value} - {alert.message}")

        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            for service_name in list(self._services.keys()):
                try:
                    await self.check_health(service_name)
                except Exception as e:
                    logger.error(f"Health check failed for {service_name}: {e}")

            await asyncio.sleep(self.check_interval)

    def start(self) -> None:
        """Start the health monitor."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started")

    def stop(self) -> None:
        """Stop the health monitor."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None
        logger.info("Health monitor stopped")

    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of all service health.

        Returns:
            Health summary for all services
        """
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {},
            "overall_status": ServiceStatus.HEALTHY.value,
        }

        unhealthy_count = 0
        degraded_count = 0

        for service_name, history in self._health_history.items():
            if history:
                latest = history[-1]
                summary["services"][service_name] = {
                    "status": latest.status.value,
                    "latency_ms": latest.latency_ms,
                    "last_check": latest.timestamp.isoformat(),
                    "error": latest.error,
                }

                if latest.status == ServiceStatus.UNHEALTHY:
                    unhealthy_count += 1
                elif latest.status == ServiceStatus.DEGRADED:
                    degraded_count += 1

        if unhealthy_count > 0:
            summary["overall_status"] = ServiceStatus.UNHEALTHY.value
        elif degraded_count > 0:
            summary["overall_status"] = ServiceStatus.DEGRADED.value

        return summary


# ============================================================================
# Integration with Opik
# ============================================================================


async def log_prediction_to_opik(
    service_name: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    latency_ms: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a prediction to Opik for observability.

    Args:
        service_name: Name of the service
        input_data: Prediction input
        output_data: Prediction output
        latency_ms: Prediction latency
        metadata: Additional metadata
    """
    try:
        from src.mlops.opik_connector import OpikConnector

        connector = OpikConnector()

        await connector.log_model_prediction(
            model_name=service_name,
            input_data=input_data,
            output_data=output_data,
            metadata={
                "latency_ms": latency_ms,
                "service_type": "bentoml",
                **(metadata or {}),
            },
        )

    except ImportError:
        logger.debug("Opik connector not available")
    except Exception as e:
        logger.error(f"Failed to log to Opik: {e}")


# ============================================================================
# Convenience Functions
# ============================================================================


def create_health_monitor(
    services: Optional[List[Dict[str, str]]] = None,
    check_interval: int = 30,
) -> BentoMLHealthMonitor:
    """Create and configure a health monitor.

    Args:
        services: List of services to register [{"name": ..., "url": ...}]
        check_interval: Check interval in seconds

    Returns:
        Configured health monitor
    """
    monitor = BentoMLHealthMonitor(check_interval_seconds=check_interval)

    if services:
        for svc in services:
            monitor.register_service(
                name=svc["name"],
                url=svc["url"],
                model_type=svc.get("model_type", "classification"),
            )

    return monitor


async def quick_health_check(url: str) -> Dict[str, Any]:
    """Perform a quick health check on a service.

    Args:
        url: Service URL

    Returns:
        Health check response
    """
    import aiohttp

    health_url = f"{url.rstrip('/')}/health"

    try:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                latency_ms = (time.time() - start_time) * 1000
                data = await response.json()
                return {
                    "status": "healthy" if response.status == 200 else "unhealthy",
                    "latency_ms": latency_ms,
                    "details": data,
                }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
