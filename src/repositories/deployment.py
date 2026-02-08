"""
ML Deployment repository for model deployments.

Provides database access for deployment records, enabling:
- Deployment creation and tracking
- Status management (pending, deploying, active, draining, rolled_back)
- Rollback history and recovery
- Performance metrics tracking

Table: ml_deployments
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from .base import BaseRepository

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class DeploymentStatus(str, Enum):
    """Deployment status matching deployment_status_enum in database."""

    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    DRAINING = "draining"
    ROLLED_BACK = "rolled_back"


class DeploymentEnvironment(str, Enum):
    """Deployment environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    SHADOW = "shadow"
    PRODUCTION = "production"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class MLDeployment:
    """ML Deployment entity matching ml_deployments table."""

    id: Optional[UUID] = None
    model_registry_id: Optional[UUID] = None

    # Deployment identification
    deployment_name: str = ""
    environment: str = "staging"

    # Endpoint info
    endpoint_name: Optional[str] = None
    endpoint_url: Optional[str] = None

    # Status
    status: str = "pending"

    # Deployment metadata
    deployed_by: Optional[str] = None
    deployment_config: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    shadow_metrics: Dict[str, Any] = field(default_factory=dict)
    production_metrics: Dict[str, Any] = field(default_factory=dict)

    # Rollback info
    previous_deployment_id: Optional[UUID] = None
    rollback_reason: Optional[str] = None
    rolled_back_at: Optional[datetime] = None

    # Timestamps
    created_at: Optional[datetime] = None
    deployed_at: Optional[datetime] = None
    deactivated_at: Optional[datetime] = None

    # SLA tracking
    latency_p50_ms: Optional[int] = None
    latency_p95_ms: Optional[int] = None
    latency_p99_ms: Optional[int] = None
    error_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": str(self.id) if self.id else None,
            "model_registry_id": str(self.model_registry_id) if self.model_registry_id else None,
            "deployment_name": self.deployment_name,
            "environment": self.environment,
            "endpoint_name": self.endpoint_name,
            "endpoint_url": self.endpoint_url,
            "status": self.status,
            "deployed_by": self.deployed_by,
            "deployment_config": self.deployment_config,
            "shadow_metrics": self.shadow_metrics,
            "production_metrics": self.production_metrics,
            "previous_deployment_id": (
                str(self.previous_deployment_id) if self.previous_deployment_id else None
            ),
            "rollback_reason": self.rollback_reason,
            "rolled_back_at": self.rolled_back_at.isoformat() if self.rolled_back_at else None,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "deactivated_at": self.deactivated_at.isoformat() if self.deactivated_at else None,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "error_rate": self.error_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLDeployment":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if data.get("id") else None,
            model_registry_id=(
                UUID(data["model_registry_id"]) if data.get("model_registry_id") else None
            ),
            deployment_name=data.get("deployment_name", ""),
            environment=data.get("environment", "staging"),
            endpoint_name=data.get("endpoint_name"),
            endpoint_url=data.get("endpoint_url"),
            status=data.get("status", "pending"),
            deployed_by=data.get("deployed_by"),
            deployment_config=data.get("deployment_config", {}),
            shadow_metrics=data.get("shadow_metrics", {}),
            production_metrics=data.get("production_metrics", {}),
            previous_deployment_id=(
                UUID(data["previous_deployment_id"]) if data.get("previous_deployment_id") else None
            ),
            rollback_reason=data.get("rollback_reason"),
            rolled_back_at=data.get("rolled_back_at"),
            created_at=data.get("created_at"),
            deployed_at=data.get("deployed_at"),
            deactivated_at=data.get("deactivated_at"),
            latency_p50_ms=data.get("latency_p50_ms"),
            latency_p95_ms=data.get("latency_p95_ms"),
            latency_p99_ms=data.get("latency_p99_ms"),
            error_rate=data.get("error_rate"),
        )

    @property
    def is_active(self) -> bool:
        """Check if deployment is currently active."""
        return self.status == DeploymentStatus.ACTIVE.value

    @property
    def can_rollback(self) -> bool:
        """Check if deployment can be rolled back (has previous deployment)."""
        return self.previous_deployment_id is not None


# ============================================================================
# REPOSITORY
# ============================================================================


class MLDeploymentRepository(BaseRepository[MLDeployment]):
    """Repository for ML Deployments."""

    table_name = "ml_deployments"
    model_class = MLDeployment

    def _to_model(self, data: Dict[str, Any]) -> MLDeployment:
        """Convert database row to model."""
        return MLDeployment.from_dict(data)

    # -------------------------------------------------------------------------
    # CREATE OPERATIONS
    # -------------------------------------------------------------------------

    async def create_deployment(
        self,
        model_registry_id: UUID,
        deployment_name: str,
        environment: str,
        endpoint_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        deployed_by: Optional[str] = None,
        deployment_config: Optional[Dict[str, Any]] = None,
        previous_deployment_id: Optional[UUID] = None,
    ) -> MLDeployment:
        """Create a new deployment record.

        Args:
            model_registry_id: ID of the model being deployed
            deployment_name: Unique deployment name
            environment: Target environment (development, staging, production)
            endpoint_name: Name of the endpoint
            endpoint_url: URL of the deployed endpoint
            deployed_by: Username of deployer
            deployment_config: Configuration for deployment (strategy, resources, etc.)
            previous_deployment_id: ID of deployment being replaced (for rollback chain)

        Returns:
            Created MLDeployment
        """
        deployment = MLDeployment(
            id=uuid4(),
            model_registry_id=model_registry_id,
            deployment_name=deployment_name,
            environment=environment,
            endpoint_name=endpoint_name,
            endpoint_url=endpoint_url,
            status=DeploymentStatus.PENDING.value,
            deployed_by=deployed_by,
            deployment_config=deployment_config or {},
            previous_deployment_id=previous_deployment_id,
            created_at=datetime.now(timezone.utc),
        )

        if self.client:
            data = deployment.to_dict()
            data.pop("id", None)  # Let DB generate ID
            result = await self.client.table(self.table_name).insert(data).execute()
            if result.data:
                return self._to_model(result.data[0])

        return deployment

    # -------------------------------------------------------------------------
    # READ OPERATIONS
    # -------------------------------------------------------------------------

    async def get_active_deployment(
        self,
        environment: str,
        model_name: Optional[str] = None,
    ) -> Optional[MLDeployment]:
        """Get currently active deployment for an environment.

        Args:
            environment: Target environment
            model_name: Optional model name filter (requires join)

        Returns:
            Active MLDeployment or None
        """
        if not self.client:
            return None

        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("environment", environment)
            .eq("status", DeploymentStatus.ACTIVE.value)
            .order("deployed_at", desc=True)
            .limit(1)
        )

        result = await query.execute()
        return self._to_model(result.data[0]) if result.data else None

    async def get_deployments_for_model(
        self,
        model_registry_id: UUID,
        environment: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[MLDeployment]:
        """Get deployment history for a model.

        Args:
            model_registry_id: Model registry ID
            environment: Optional environment filter
            status: Optional status filter
            limit: Maximum results

        Returns:
            List of MLDeployment records
        """
        if not self.client:
            return []

        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("model_registry_id", str(model_registry_id))
        )

        if environment:
            query = query.eq("environment", environment)
        if status:
            query = query.eq("status", status)

        query = query.order("created_at", desc=True).limit(limit)
        result = await query.execute()

        return [self._to_model(row) for row in result.data]

    async def get_deployments_by_status(
        self,
        status: str,
        environment: Optional[str] = None,
        limit: int = 100,
    ) -> List[MLDeployment]:
        """Get deployments by status.

        Args:
            status: Deployment status
            environment: Optional environment filter
            limit: Maximum results

        Returns:
            List of MLDeployment records
        """
        filters = {"status": status}
        if environment:
            filters["environment"] = environment

        return await self.get_many(filters=filters, limit=limit)

    async def get_previous_deployment(
        self,
        deployment_id: UUID,
    ) -> Optional[MLDeployment]:
        """Get the previous deployment in the rollback chain.

        Args:
            deployment_id: Current deployment ID

        Returns:
            Previous MLDeployment or None
        """
        current = await self.get_by_id(str(deployment_id))
        if not current or not current.previous_deployment_id:
            return None

        return await self.get_by_id(str(current.previous_deployment_id))

    async def get_rollback_chain(
        self,
        deployment_id: UUID,
        max_depth: int = 10,
    ) -> List[MLDeployment]:
        """Get the chain of previous deployments for rollback.

        Args:
            deployment_id: Starting deployment ID
            max_depth: Maximum chain length

        Returns:
            List of MLDeployment in rollback order (most recent first)
        """
        chain = []
        current_id = deployment_id

        for _ in range(max_depth):
            deployment = await self.get_by_id(str(current_id))
            if not deployment:
                break

            chain.append(deployment)

            if not deployment.previous_deployment_id:
                break

            current_id = deployment.previous_deployment_id

        return chain

    async def get_deployment_by_name(
        self,
        deployment_name: str,
        environment: Optional[str] = None,
    ) -> Optional[MLDeployment]:
        """Get deployment by name.

        Args:
            deployment_name: Deployment name
            environment: Optional environment filter

        Returns:
            MLDeployment or None
        """
        if not self.client:
            return None

        query = (
            self.client.table(self.table_name).select("*").eq("deployment_name", deployment_name)
        )

        if environment:
            query = query.eq("environment", environment)

        result = await query.limit(1).execute()
        return self._to_model(result.data[0]) if result.data else None

    # -------------------------------------------------------------------------
    # UPDATE OPERATIONS
    # -------------------------------------------------------------------------

    async def update_status(
        self,
        deployment_id: UUID,
        new_status: str,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update deployment status.

        Args:
            deployment_id: Deployment ID
            new_status: New status value
            error_message: Optional error message

        Returns:
            True if successful
        """
        if not self.client:
            return False

        updates: Dict[str, Any] = {"status": new_status}

        # Set timestamps based on status
        now = datetime.now(timezone.utc).isoformat()
        if new_status == DeploymentStatus.ACTIVE.value:
            updates["deployed_at"] = now
        elif new_status in [DeploymentStatus.DRAINING.value, DeploymentStatus.ROLLED_BACK.value]:
            updates["deactivated_at"] = now

        if error_message:
            updates["rollback_reason"] = error_message

        await (
            self.client.table(self.table_name)
            .update(updates)
            .eq("id", str(deployment_id))
            .execute()
        )

        return True

    async def update_endpoint_info(
        self,
        deployment_id: UUID,
        endpoint_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ) -> bool:
        """Update endpoint information.

        Args:
            deployment_id: Deployment ID
            endpoint_name: Endpoint name
            endpoint_url: Endpoint URL

        Returns:
            True if successful
        """
        if not self.client:
            return False

        updates = {}
        if endpoint_name:
            updates["endpoint_name"] = endpoint_name
        if endpoint_url:
            updates["endpoint_url"] = endpoint_url

        if updates:
            await (
                self.client.table(self.table_name)
                .update(updates)
                .eq("id", str(deployment_id))
                .execute()
            )
            return True

        return False

    async def update_metrics(
        self,
        deployment_id: UUID,
        shadow_metrics: Optional[Dict[str, Any]] = None,
        production_metrics: Optional[Dict[str, Any]] = None,
        latency_p50_ms: Optional[int] = None,
        latency_p95_ms: Optional[int] = None,
        latency_p99_ms: Optional[int] = None,
        error_rate: Optional[float] = None,
    ) -> bool:
        """Update deployment performance metrics.

        Args:
            deployment_id: Deployment ID
            shadow_metrics: Metrics from shadow mode
            production_metrics: Production metrics
            latency_p50_ms: 50th percentile latency
            latency_p95_ms: 95th percentile latency
            latency_p99_ms: 99th percentile latency
            error_rate: Error rate (0.0 to 1.0)

        Returns:
            True if successful
        """
        if not self.client:
            return False

        updates: Dict[str, Any] = {}
        if shadow_metrics is not None:
            updates["shadow_metrics"] = shadow_metrics
        if production_metrics is not None:
            updates["production_metrics"] = production_metrics
        if latency_p50_ms is not None:
            updates["latency_p50_ms"] = latency_p50_ms
        if latency_p95_ms is not None:
            updates["latency_p95_ms"] = latency_p95_ms
        if latency_p99_ms is not None:
            updates["latency_p99_ms"] = latency_p99_ms
        if error_rate is not None:
            updates["error_rate"] = error_rate

        if updates:
            await (
                self.client.table(self.table_name)
                .update(updates)
                .eq("id", str(deployment_id))
                .execute()
            )
            return True

        return False

    # -------------------------------------------------------------------------
    # ROLLBACK OPERATIONS
    # -------------------------------------------------------------------------

    async def mark_rolled_back(
        self,
        deployment_id: UUID,
        reason: str,
    ) -> bool:
        """Mark a deployment as rolled back.

        Args:
            deployment_id: Deployment ID to roll back
            reason: Rollback reason

        Returns:
            True if successful
        """
        if not self.client:
            return False

        now = datetime.now(timezone.utc).isoformat()
        updates = {
            "status": DeploymentStatus.ROLLED_BACK.value,
            "rollback_reason": reason,
            "rolled_back_at": now,
            "deactivated_at": now,
        }

        await (
            self.client.table(self.table_name)
            .update(updates)
            .eq("id", str(deployment_id))
            .execute()
        )

        return True

    async def deactivate_other_deployments(
        self,
        current_deployment_id: UUID,
        environment: str,
    ) -> int:
        """Deactivate all other active deployments in an environment.

        Used when promoting a new deployment to active status.

        Args:
            current_deployment_id: ID of deployment being made active
            environment: Environment to deactivate in

        Returns:
            Number of deployments deactivated
        """
        if not self.client:
            return 0

        now = datetime.now(timezone.utc).isoformat()

        result = await (
            self.client.table(self.table_name)
            .update(
                {
                    "status": DeploymentStatus.DRAINING.value,
                    "deactivated_at": now,
                }
            )
            .eq("environment", environment)
            .eq("status", DeploymentStatus.ACTIVE.value)
            .neq("id", str(current_deployment_id))
            .execute()
        )

        return len(result.data) if result.data else 0

    # -------------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------------

    async def get_deployment_count_by_status(
        self,
        environment: Optional[str] = None,
    ) -> Dict[str, int]:
        """Get count of deployments by status.

        Args:
            environment: Optional environment filter

        Returns:
            Dict mapping status to count
        """
        counts = {}
        for status in DeploymentStatus:
            deployments = await self.get_deployments_by_status(
                status=status.value,
                environment=environment,
                limit=1000,
            )
            counts[status.value] = len(deployments)

        return counts

    async def cleanup_stale_deployments(
        self,
        max_age_hours: int = 24,
    ) -> int:
        """Clean up stale pending/deploying deployments.

        Deployments stuck in pending/deploying for too long are marked as failed.

        Args:
            max_age_hours: Maximum age in hours before marking as failed

        Returns:
            Number of deployments cleaned up
        """
        if not self.client:
            return 0

        # This would ideally use a proper date comparison, but for now
        # we'll implement it in application code
        stale_statuses = [DeploymentStatus.PENDING.value, DeploymentStatus.DEPLOYING.value]
        count = 0

        for status in stale_statuses:
            deployments = await self.get_deployments_by_status(status=status, limit=100)
            cutoff = datetime.now(timezone.utc).replace(
                hour=datetime.now(timezone.utc).hour - max_age_hours
            )

            for deployment in deployments:
                if deployment.created_at and deployment.created_at < cutoff:
                    if deployment.id is not None:
                        await self.mark_rolled_back(
                            deployment_id=deployment.id,
                            reason=f"Stale deployment (pending > {max_age_hours}h)",
                        )
                        count += 1

        return count
