"""
BentoML Service repository for service deployments and metrics.

Provides database access for BentoML service records, enabling:
- Service creation and lifecycle tracking
- Health monitoring
- Serving metrics time-series data
- Performance analysis

Tables: ml_bentoml_services, ml_bentoml_serving_metrics
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from .base import BaseRepository

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class ServiceHealthStatus(str, Enum):
    """Service health status values."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
    STARTING = "starting"
    STOPPED = "stopped"


class ServiceStatus(str, Enum):
    """Service deployment status matching deployment_status_enum."""

    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    DRAINING = "draining"
    ROLLED_BACK = "rolled_back"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class BentoMLService:
    """BentoML Service entity matching ml_bentoml_services table."""

    id: Optional[UUID] = None
    service_name: str = ""
    bento_tag: str = ""
    bento_version: Optional[str] = None

    # Relations
    model_registry_id: Optional[UUID] = None
    deployment_id: Optional[UUID] = None

    # Container info
    container_image: Optional[str] = None
    container_tag: Optional[str] = None

    # Configuration
    replicas: int = 1
    resources: Dict[str, Any] = field(default_factory=lambda: {"cpu": "1", "memory": "2Gi"})
    environment: str = "staging"

    # Health tracking
    health_status: str = "unknown"
    last_health_check: Optional[datetime] = None
    health_check_failures: int = 0

    # Endpoint info
    serving_endpoint: Optional[str] = None
    internal_endpoint: Optional[str] = None

    # Lifecycle
    status: str = "pending"
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Metadata
    created_by: Optional[str] = None
    labels: Dict[str, Any] = field(default_factory=dict)
    annotations: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": str(self.id) if self.id else None,
            "service_name": self.service_name,
            "bento_tag": self.bento_tag,
            "bento_version": self.bento_version,
            "model_registry_id": str(self.model_registry_id) if self.model_registry_id else None,
            "deployment_id": str(self.deployment_id) if self.deployment_id else None,
            "container_image": self.container_image,
            "container_tag": self.container_tag,
            "replicas": self.replicas,
            "resources": self.resources,
            "environment": self.environment,
            "health_status": self.health_status,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "health_check_failures": self.health_check_failures,
            "serving_endpoint": self.serving_endpoint,
            "internal_endpoint": self.internal_endpoint,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "labels": self.labels,
            "annotations": self.annotations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BentoMLService":
        """Create instance from dictionary."""
        return cls(
            id=UUID(data["id"]) if data.get("id") else None,
            service_name=data.get("service_name", ""),
            bento_tag=data.get("bento_tag", ""),
            bento_version=data.get("bento_version"),
            model_registry_id=UUID(data["model_registry_id"]) if data.get("model_registry_id") else None,
            deployment_id=UUID(data["deployment_id"]) if data.get("deployment_id") else None,
            container_image=data.get("container_image"),
            container_tag=data.get("container_tag"),
            replicas=data.get("replicas", 1),
            resources=data.get("resources") or {"cpu": "1", "memory": "2Gi"},
            environment=data.get("environment", "staging"),
            health_status=data.get("health_status", "unknown"),
            last_health_check=datetime.fromisoformat(data["last_health_check"]) if data.get("last_health_check") else None,
            health_check_failures=data.get("health_check_failures", 0),
            serving_endpoint=data.get("serving_endpoint"),
            internal_endpoint=data.get("internal_endpoint"),
            status=data.get("status", "pending"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            stopped_at=datetime.fromisoformat(data["stopped_at"]) if data.get("stopped_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            created_by=data.get("created_by"),
            labels=data.get("labels") or {},
            annotations=data.get("annotations") or {},
        )


@dataclass
class BentoMLServingMetrics:
    """Serving metrics entity matching ml_bentoml_serving_metrics table."""

    id: Optional[UUID] = None
    service_id: Optional[UUID] = None
    recorded_at: Optional[datetime] = None

    # Throughput metrics
    requests_total: int = 0
    requests_per_second: Optional[float] = None
    successful_requests: int = 0
    failed_requests: int = 0

    # Latency metrics (milliseconds)
    avg_latency_ms: Optional[float] = None
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None

    # Error metrics
    error_rate: Optional[float] = None
    error_types: Dict[str, int] = field(default_factory=dict)

    # Resource utilization
    memory_mb: Optional[float] = None
    memory_percent: Optional[float] = None
    cpu_percent: Optional[float] = None

    # Prediction metrics
    predictions_count: int = 0
    batch_size_avg: Optional[float] = None

    # Model-specific
    model_load_time_ms: Optional[float] = None
    inference_time_avg_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": str(self.id) if self.id else None,
            "service_id": str(self.service_id) if self.service_id else None,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "requests_total": self.requests_total,
            "requests_per_second": self.requests_per_second,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "error_rate": self.error_rate,
            "error_types": self.error_types,
            "memory_mb": self.memory_mb,
            "memory_percent": self.memory_percent,
            "cpu_percent": self.cpu_percent,
            "predictions_count": self.predictions_count,
            "batch_size_avg": self.batch_size_avg,
            "model_load_time_ms": self.model_load_time_ms,
            "inference_time_avg_ms": self.inference_time_avg_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BentoMLServingMetrics":
        """Create instance from dictionary."""
        return cls(
            id=UUID(data["id"]) if data.get("id") else None,
            service_id=UUID(data["service_id"]) if data.get("service_id") else None,
            recorded_at=datetime.fromisoformat(data["recorded_at"]) if data.get("recorded_at") else None,
            requests_total=data.get("requests_total", 0),
            requests_per_second=data.get("requests_per_second"),
            successful_requests=data.get("successful_requests", 0),
            failed_requests=data.get("failed_requests", 0),
            avg_latency_ms=data.get("avg_latency_ms"),
            p50_latency_ms=data.get("p50_latency_ms"),
            p95_latency_ms=data.get("p95_latency_ms"),
            p99_latency_ms=data.get("p99_latency_ms"),
            max_latency_ms=data.get("max_latency_ms"),
            error_rate=data.get("error_rate"),
            error_types=data.get("error_types") or {},
            memory_mb=data.get("memory_mb"),
            memory_percent=data.get("memory_percent"),
            cpu_percent=data.get("cpu_percent"),
            predictions_count=data.get("predictions_count", 0),
            batch_size_avg=data.get("batch_size_avg"),
            model_load_time_ms=data.get("model_load_time_ms"),
            inference_time_avg_ms=data.get("inference_time_avg_ms"),
        )


# ============================================================================
# REPOSITORY CLASSES
# ============================================================================


class BentoMLServiceRepository(BaseRepository[BentoMLService]):
    """Repository for BentoML service records."""

    table_name = "ml_bentoml_services"
    model_class = BentoMLService

    def _to_model(self, data: Dict[str, Any]) -> BentoMLService:
        """Convert database row to model."""
        return BentoMLService.from_dict(data)

    async def create_service(
        self,
        service_name: str,
        bento_tag: str,
        bento_version: Optional[str] = None,
        model_registry_id: Optional[UUID] = None,
        deployment_id: Optional[UUID] = None,
        container_image: Optional[str] = None,
        container_tag: Optional[str] = None,
        replicas: int = 1,
        resources: Optional[Dict[str, Any]] = None,
        environment: str = "staging",
        serving_endpoint: Optional[str] = None,
        created_by: Optional[str] = None,
        labels: Optional[Dict[str, Any]] = None,
    ) -> Optional[BentoMLService]:
        """Create a new BentoML service record.

        Args:
            service_name: Name of the service
            bento_tag: BentoML tag
            bento_version: Version string
            model_registry_id: Link to model registry
            deployment_id: Link to ml_deployments
            container_image: Docker image
            container_tag: Docker tag
            replicas: Number of replicas
            resources: Resource configuration
            environment: Deployment environment
            serving_endpoint: External endpoint URL
            created_by: Username of creator
            labels: Service labels

        Returns:
            Created BentoMLService or None
        """
        if not self.client:
            logger.warning("No Supabase client available for service creation")
            return None

        try:
            service = BentoMLService(
                id=uuid4(),
                service_name=service_name,
                bento_tag=bento_tag,
                bento_version=bento_version,
                model_registry_id=model_registry_id,
                deployment_id=deployment_id,
                container_image=container_image,
                container_tag=container_tag,
                replicas=replicas,
                resources=resources or {"cpu": "1", "memory": "2Gi"},
                environment=environment,
                health_status="starting",
                serving_endpoint=serving_endpoint,
                status="pending",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                created_by=created_by,
                labels=labels or {},
            )

            data = service.to_dict()
            # Remove None id for database auto-generation
            data = {k: v for k, v in data.items() if v is not None}

            result = self.client.table(self.table_name).insert(data).execute()

            if result.data:
                logger.info(f"Created BentoML service: {service_name} (tag: {bento_tag})")
                return self._to_model(result.data[0])

            return None

        except Exception as e:
            logger.error(f"Failed to create BentoML service: {e}")
            return None

    async def update_status(
        self,
        service_id: UUID,
        status: ServiceStatus,
        started_at: Optional[datetime] = None,
        stopped_at: Optional[datetime] = None,
    ) -> Optional[BentoMLService]:
        """Update service deployment status.

        Args:
            service_id: Service UUID
            status: New status
            started_at: When service started (for ACTIVE)
            stopped_at: When service stopped (for ROLLED_BACK/DRAINING)

        Returns:
            Updated service or None
        """
        if not self.client:
            return None

        try:
            update_data = {
                "status": status.value,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            if started_at:
                update_data["started_at"] = started_at.isoformat()
            if stopped_at:
                update_data["stopped_at"] = stopped_at.isoformat()

            result = (
                self.client.table(self.table_name)
                .update(update_data)
                .eq("id", str(service_id))
                .execute()
            )

            if result.data:
                logger.info(f"Updated service {service_id} status to {status.value}")
                return self._to_model(result.data[0])

            return None

        except Exception as e:
            logger.error(f"Failed to update service status: {e}")
            return None

    async def update_health(
        self,
        service_id: UUID,
        health_status: ServiceHealthStatus,
        increment_failures: bool = False,
    ) -> Optional[BentoMLService]:
        """Update service health status.

        Args:
            service_id: Service UUID
            health_status: New health status
            increment_failures: Whether to increment failure count

        Returns:
            Updated service or None
        """
        if not self.client:
            return None

        try:
            update_data = {
                "health_status": health_status.value,
                "last_health_check": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            # If healthy, reset failure count
            if health_status == ServiceHealthStatus.HEALTHY:
                update_data["health_check_failures"] = 0

            result = (
                self.client.table(self.table_name)
                .update(update_data)
                .eq("id", str(service_id))
                .execute()
            )

            # Increment failures separately if needed
            if increment_failures and result.data:
                current = result.data[0].get("health_check_failures", 0)
                self.client.table(self.table_name).update(
                    {"health_check_failures": current + 1}
                ).eq("id", str(service_id)).execute()

            if result.data:
                return self._to_model(result.data[0])

            return None

        except Exception as e:
            logger.error(f"Failed to update service health: {e}")
            return None

    async def get_active_services(
        self,
        environment: Optional[str] = None,
    ) -> List[BentoMLService]:
        """Get all active services.

        Args:
            environment: Optional environment filter

        Returns:
            List of active services
        """
        if not self.client:
            return []

        try:
            query = (
                self.client.table(self.table_name)
                .select("*")
                .eq("status", "active")
            )

            if environment:
                query = query.eq("environment", environment)

            result = query.execute()
            return [self._to_model(row) for row in result.data]

        except Exception as e:
            logger.error(f"Failed to get active services: {e}")
            return []

    async def get_by_bento_tag(self, bento_tag: str) -> Optional[BentoMLService]:
        """Get service by BentoML tag.

        Args:
            bento_tag: BentoML tag

        Returns:
            Service or None
        """
        if not self.client:
            return None

        try:
            result = (
                self.client.table(self.table_name)
                .select("*")
                .eq("bento_tag", bento_tag)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )

            if result.data:
                return self._to_model(result.data[0])

            return None

        except Exception as e:
            logger.error(f"Failed to get service by tag: {e}")
            return None

    async def get_unhealthy_services(
        self,
        failure_threshold: int = 3,
    ) -> List[BentoMLService]:
        """Get services with health issues.

        Args:
            failure_threshold: Minimum failures to include

        Returns:
            List of unhealthy services
        """
        if not self.client:
            return []

        try:
            result = (
                self.client.table(self.table_name)
                .select("*")
                .eq("status", "active")
                .neq("health_status", "healthy")
                .gte("health_check_failures", failure_threshold)
                .execute()
            )

            return [self._to_model(row) for row in result.data]

        except Exception as e:
            logger.error(f"Failed to get unhealthy services: {e}")
            return []


class BentoMLMetricsRepository(BaseRepository[BentoMLServingMetrics]):
    """Repository for BentoML serving metrics."""

    table_name = "ml_bentoml_serving_metrics"
    model_class = BentoMLServingMetrics

    def _to_model(self, data: Dict[str, Any]) -> BentoMLServingMetrics:
        """Convert database row to model."""
        return BentoMLServingMetrics.from_dict(data)

    async def record_metrics(
        self,
        service_id: UUID,
        requests_total: int = 0,
        requests_per_second: Optional[float] = None,
        successful_requests: int = 0,
        failed_requests: int = 0,
        avg_latency_ms: Optional[float] = None,
        p50_latency_ms: Optional[float] = None,
        p95_latency_ms: Optional[float] = None,
        p99_latency_ms: Optional[float] = None,
        error_rate: Optional[float] = None,
        memory_mb: Optional[float] = None,
        cpu_percent: Optional[float] = None,
        predictions_count: int = 0,
    ) -> Optional[BentoMLServingMetrics]:
        """Record serving metrics for a service.

        Args:
            service_id: Service UUID
            requests_total: Total requests in period
            requests_per_second: Request rate
            successful_requests: Successful request count
            failed_requests: Failed request count
            avg_latency_ms: Average latency
            p50_latency_ms: Median latency
            p95_latency_ms: 95th percentile latency
            p99_latency_ms: 99th percentile latency
            error_rate: Error percentage
            memory_mb: Memory usage in MB
            cpu_percent: CPU utilization percentage
            predictions_count: Number of predictions

        Returns:
            Created metrics record or None
        """
        if not self.client:
            return None

        try:
            metrics = BentoMLServingMetrics(
                id=uuid4(),
                service_id=service_id,
                recorded_at=datetime.now(timezone.utc),
                requests_total=requests_total,
                requests_per_second=requests_per_second,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                avg_latency_ms=avg_latency_ms,
                p50_latency_ms=p50_latency_ms,
                p95_latency_ms=p95_latency_ms,
                p99_latency_ms=p99_latency_ms,
                error_rate=error_rate,
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                predictions_count=predictions_count,
            )

            data = metrics.to_dict()
            data = {k: v for k, v in data.items() if v is not None}

            result = self.client.table(self.table_name).insert(data).execute()

            if result.data:
                return self._to_model(result.data[0])

            return None

        except Exception as e:
            logger.error(f"Failed to record metrics: {e}")
            return None

    async def get_latest_metrics(
        self,
        service_id: UUID,
    ) -> Optional[BentoMLServingMetrics]:
        """Get latest metrics for a service.

        Args:
            service_id: Service UUID

        Returns:
            Latest metrics or None
        """
        if not self.client:
            return None

        try:
            result = (
                self.client.table(self.table_name)
                .select("*")
                .eq("service_id", str(service_id))
                .order("recorded_at", desc=True)
                .limit(1)
                .execute()
            )

            if result.data:
                return self._to_model(result.data[0])

            return None

        except Exception as e:
            logger.error(f"Failed to get latest metrics: {e}")
            return None

    async def get_metrics_range(
        self,
        service_id: UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[BentoMLServingMetrics]:
        """Get metrics within a time range.

        Args:
            service_id: Service UUID
            start_time: Start of range (default: 1 hour ago)
            end_time: End of range (default: now)
            limit: Maximum records

        Returns:
            List of metrics records
        """
        if not self.client:
            return []

        try:
            if not start_time:
                start_time = datetime.now(timezone.utc) - timedelta(hours=1)
            if not end_time:
                end_time = datetime.now(timezone.utc)

            result = (
                self.client.table(self.table_name)
                .select("*")
                .eq("service_id", str(service_id))
                .gte("recorded_at", start_time.isoformat())
                .lte("recorded_at", end_time.isoformat())
                .order("recorded_at", desc=True)
                .limit(limit)
                .execute()
            )

            return [self._to_model(row) for row in result.data]

        except Exception as e:
            logger.error(f"Failed to get metrics range: {e}")
            return []

    async def get_metrics_summary(
        self,
        service_id: UUID,
        hours: int = 1,
    ) -> Dict[str, Any]:
        """Get aggregated metrics summary.

        Args:
            service_id: Service UUID
            hours: Hours to aggregate

        Returns:
            Summary dictionary with aggregated metrics
        """
        if not self.client:
            return {}

        try:
            # Use database function if available
            result = self.client.rpc(
                "get_bentoml_metrics_summary",
                {
                    "p_service_id": str(service_id),
                    "p_start_time": (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(),
                    "p_end_time": datetime.now(timezone.utc).isoformat(),
                },
            ).execute()

            if result.data:
                return result.data[0] if isinstance(result.data, list) else result.data

            # Fallback to manual aggregation
            metrics = await self.get_metrics_range(
                service_id=service_id,
                start_time=datetime.now(timezone.utc) - timedelta(hours=hours),
            )

            if not metrics:
                return {}

            return {
                "total_requests": sum(m.requests_total for m in metrics),
                "avg_rps": sum(m.requests_per_second or 0 for m in metrics) / len(metrics) if metrics else 0,
                "avg_latency_ms": sum(m.avg_latency_ms or 0 for m in metrics) / len(metrics) if metrics else 0,
                "avg_error_rate": sum(m.error_rate or 0 for m in metrics) / len(metrics) if metrics else 0,
                "avg_memory_mb": sum(m.memory_mb or 0 for m in metrics) / len(metrics) if metrics else 0,
                "avg_cpu_percent": sum(m.cpu_percent or 0 for m in metrics) / len(metrics) if metrics else 0,
                "data_points": len(metrics),
            }

        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {}
