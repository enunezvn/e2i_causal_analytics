"""Tests for MLDeploymentRepository.

Tests deployment repository operations:
- CRUD operations
- Status management
- Rollback chain handling
- Metrics updates
"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from src.repositories.deployment import (
    DeploymentEnvironment,
    DeploymentStatus,
    MLDeployment,
    MLDeploymentRepository,
)


class TestDeploymentStatus:
    """Test DeploymentStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert DeploymentStatus.PENDING.value == "pending"
        assert DeploymentStatus.DEPLOYING.value == "deploying"
        assert DeploymentStatus.ACTIVE.value == "active"
        assert DeploymentStatus.DRAINING.value == "draining"
        assert DeploymentStatus.ROLLED_BACK.value == "rolled_back"

    def test_all_statuses_defined(self):
        """Test all expected statuses are defined."""
        expected_statuses = {"pending", "deploying", "active", "draining", "rolled_back"}
        actual_statuses = {s.value for s in DeploymentStatus}
        assert actual_statuses == expected_statuses


class TestDeploymentEnvironment:
    """Test DeploymentEnvironment enum."""

    def test_environment_values(self):
        """Test environment enum values."""
        assert DeploymentEnvironment.DEVELOPMENT.value == "development"
        assert DeploymentEnvironment.STAGING.value == "staging"
        assert DeploymentEnvironment.SHADOW.value == "shadow"
        assert DeploymentEnvironment.PRODUCTION.value == "production"

    def test_all_environments_defined(self):
        """Test all expected environments are defined."""
        expected_environments = {"development", "staging", "shadow", "production"}
        actual_environments = {e.value for e in DeploymentEnvironment}
        assert actual_environments == expected_environments


class TestMLDeployment:
    """Test MLDeployment dataclass."""

    def test_default_deployment(self):
        """Test default deployment values."""
        deployment = MLDeployment()

        assert deployment.id is None
        assert deployment.model_registry_id is None
        assert deployment.deployment_name == ""
        assert deployment.environment == "staging"
        assert deployment.endpoint_name is None
        assert deployment.endpoint_url is None
        assert deployment.status == "pending"
        assert deployment.deployment_config == {}
        assert deployment.shadow_metrics == {}
        assert deployment.production_metrics == {}

    def test_deployment_with_values(self):
        """Test deployment with custom values."""
        deployment_id = uuid4()
        model_id = uuid4()
        prev_deployment_id = uuid4()

        deployment = MLDeployment(
            id=deployment_id,
            model_registry_id=model_id,
            deployment_name="test-deployment",
            environment="production",
            endpoint_name="test-endpoint-prod",
            endpoint_url="https://api.e2i.com/v1/test/predict",
            status="active",
            deployed_by="model_deployer",
            deployment_config={"strategy": "blue_green", "replicas": 3},
            previous_deployment_id=prev_deployment_id,
            latency_p50_ms=50,
            latency_p95_ms=150,
            latency_p99_ms=300,
            error_rate=0.001,
        )

        assert deployment.id == deployment_id
        assert deployment.model_registry_id == model_id
        assert deployment.deployment_name == "test-deployment"
        assert deployment.environment == "production"
        assert deployment.status == "active"
        assert deployment.deployment_config["strategy"] == "blue_green"
        assert deployment.previous_deployment_id == prev_deployment_id
        assert deployment.latency_p99_ms == 300
        assert deployment.error_rate == 0.001

    def test_is_active_property(self):
        """Test is_active property."""
        active_deployment = MLDeployment(status="active")
        pending_deployment = MLDeployment(status="pending")
        rolled_back_deployment = MLDeployment(status="rolled_back")

        assert active_deployment.is_active is True
        assert pending_deployment.is_active is False
        assert rolled_back_deployment.is_active is False

    def test_can_rollback_property(self):
        """Test can_rollback property."""
        deployment_with_prev = MLDeployment(previous_deployment_id=uuid4())
        deployment_without_prev = MLDeployment()

        assert deployment_with_prev.can_rollback is True
        assert deployment_without_prev.can_rollback is False

    def test_to_dict(self):
        """Test to_dict conversion."""
        deployment_id = uuid4()
        model_id = uuid4()
        deployment = MLDeployment(
            id=deployment_id,
            model_registry_id=model_id,
            deployment_name="test-deployment",
            environment="staging",
            status="pending",
            deployment_config={"cpu": "2", "memory": "4Gi"},
        )

        result = deployment.to_dict()

        assert result["id"] == str(deployment_id)
        assert result["model_registry_id"] == str(model_id)
        assert result["deployment_name"] == "test-deployment"
        assert result["environment"] == "staging"
        assert result["status"] == "pending"
        assert result["deployment_config"]["cpu"] == "2"

    def test_to_dict_none_values(self):
        """Test to_dict with None values."""
        deployment = MLDeployment()
        result = deployment.to_dict()

        assert result["id"] is None
        assert result["model_registry_id"] is None
        assert result["previous_deployment_id"] is None
        assert result["rolled_back_at"] is None

    def test_from_dict(self):
        """Test from_dict conversion."""
        deployment_id = str(uuid4())
        model_id = str(uuid4())
        data = {
            "id": deployment_id,
            "model_registry_id": model_id,
            "deployment_name": "test-deployment",
            "environment": "production",
            "endpoint_name": "test-endpoint",
            "endpoint_url": "https://api.e2i.com/v1/test/predict",
            "status": "active",
            "deployed_by": "model_deployer",
            "deployment_config": {"strategy": "canary"},
            "shadow_metrics": {"accuracy": 0.95},
            "production_metrics": {"latency_avg_ms": 100},
        }

        deployment = MLDeployment.from_dict(data)

        assert str(deployment.id) == deployment_id
        assert str(deployment.model_registry_id) == model_id
        assert deployment.deployment_name == "test-deployment"
        assert deployment.environment == "production"
        assert deployment.status == "active"
        assert deployment.deployment_config["strategy"] == "canary"
        assert deployment.shadow_metrics["accuracy"] == 0.95

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        data = {"deployment_name": "minimal-deployment"}

        deployment = MLDeployment.from_dict(data)

        assert deployment.deployment_name == "minimal-deployment"
        assert deployment.environment == "staging"  # default
        assert deployment.status == "pending"  # default
        assert deployment.deployment_config == {}

    def test_roundtrip_dict_conversion(self):
        """Test that to_dict and from_dict are inverse operations."""
        original = MLDeployment(
            id=uuid4(),
            model_registry_id=uuid4(),
            deployment_name="roundtrip-test",
            environment="shadow",
            status="deploying",
            deployment_config={"replicas": 2},
            error_rate=0.005,
        )

        converted = MLDeployment.from_dict(original.to_dict())

        assert converted.id == original.id
        assert converted.model_registry_id == original.model_registry_id
        assert converted.deployment_name == original.deployment_name
        assert converted.environment == original.environment
        assert converted.status == original.status
        assert converted.deployment_config == original.deployment_config
        assert converted.error_rate == original.error_rate


class TestMLDeploymentRepository:
    """Test MLDeploymentRepository methods."""

    def test_repository_table_name(self):
        """Test repository table name."""
        assert MLDeploymentRepository.table_name == "ml_deployments"

    def test_repository_model_class(self):
        """Test repository model class."""
        assert MLDeploymentRepository.model_class == MLDeployment

    def test_to_model(self):
        """Test _to_model conversion."""
        repo = MLDeploymentRepository(supabase_client=None)
        data = {
            "id": str(uuid4()),
            "deployment_name": "test-deployment",
            "environment": "staging",
            "status": "pending",
        }

        result = repo._to_model(data)

        assert isinstance(result, MLDeployment)
        assert result.deployment_name == "test-deployment"
        assert result.environment == "staging"

    @pytest.mark.asyncio
    async def test_create_deployment_without_client(self):
        """Test create_deployment without database client returns local object."""
        repo = MLDeploymentRepository(supabase_client=None)
        model_registry_id = uuid4()

        result = await repo.create_deployment(
            model_registry_id=model_registry_id,
            deployment_name="test-deployment",
            environment="staging",
            endpoint_name="test-endpoint",
            endpoint_url="https://api.e2i.com/test",
            deployed_by="test_user",
            deployment_config={"strategy": "direct"},
        )

        assert result.model_registry_id == model_registry_id
        assert result.deployment_name == "test-deployment"
        assert result.environment == "staging"
        assert result.status == DeploymentStatus.PENDING.value
        assert result.deployed_by == "test_user"
        assert result.id is not None  # ID is generated

    @pytest.mark.asyncio
    async def test_get_active_deployment_without_client(self):
        """Test get_active_deployment without client returns None."""
        repo = MLDeploymentRepository(supabase_client=None)

        result = await repo.get_active_deployment(environment="production")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_deployments_for_model_without_client(self):
        """Test get_deployments_for_model without client returns empty list."""
        repo = MLDeploymentRepository(supabase_client=None)

        result = await repo.get_deployments_for_model(model_registry_id=uuid4())

        assert result == []

    @pytest.mark.asyncio
    async def test_update_status_without_client(self):
        """Test update_status without client returns False."""
        repo = MLDeploymentRepository(supabase_client=None)

        result = await repo.update_status(
            deployment_id=uuid4(),
            new_status=DeploymentStatus.ACTIVE.value,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_update_endpoint_info_without_client(self):
        """Test update_endpoint_info without client returns False."""
        repo = MLDeploymentRepository(supabase_client=None)

        result = await repo.update_endpoint_info(
            deployment_id=uuid4(),
            endpoint_name="new-endpoint",
            endpoint_url="https://api.e2i.com/new",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_update_endpoint_info_no_updates(self):
        """Test update_endpoint_info with no updates returns False."""
        repo = MLDeploymentRepository(supabase_client=None)

        result = await repo.update_endpoint_info(deployment_id=uuid4())

        assert result is False

    @pytest.mark.asyncio
    async def test_update_metrics_without_client(self):
        """Test update_metrics without client returns False."""
        repo = MLDeploymentRepository(supabase_client=None)

        result = await repo.update_metrics(
            deployment_id=uuid4(),
            latency_p50_ms=50,
            latency_p95_ms=150,
            error_rate=0.001,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_update_metrics_no_updates(self):
        """Test update_metrics with no updates returns False."""
        repo = MLDeploymentRepository(supabase_client=None)

        result = await repo.update_metrics(deployment_id=uuid4())

        assert result is False

    @pytest.mark.asyncio
    async def test_mark_rolled_back_without_client(self):
        """Test mark_rolled_back without client returns False."""
        repo = MLDeploymentRepository(supabase_client=None)

        result = await repo.mark_rolled_back(
            deployment_id=uuid4(),
            reason="Performance degradation",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_deactivate_other_deployments_without_client(self):
        """Test deactivate_other_deployments without client returns 0."""
        repo = MLDeploymentRepository(supabase_client=None)

        result = await repo.deactivate_other_deployments(
            current_deployment_id=uuid4(),
            environment="production",
        )

        assert result == 0

    @pytest.mark.asyncio
    async def test_get_deployment_by_name_without_client(self):
        """Test get_deployment_by_name without client returns None."""
        repo = MLDeploymentRepository(supabase_client=None)

        result = await repo.get_deployment_by_name(
            deployment_name="test-deployment",
            environment="staging",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_previous_deployment_no_client(self):
        """Test get_previous_deployment without client returns None."""
        repo = MLDeploymentRepository(supabase_client=None)

        result = await repo.get_previous_deployment(deployment_id=uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_get_rollback_chain_without_client(self):
        """Test get_rollback_chain without client returns empty list."""
        repo = MLDeploymentRepository(supabase_client=None)

        result = await repo.get_rollback_chain(deployment_id=uuid4())

        # Without client, get_by_id returns None, so chain is empty
        assert result == []

    @pytest.mark.asyncio
    async def test_cleanup_stale_deployments_without_client(self):
        """Test cleanup_stale_deployments without client returns 0."""
        repo = MLDeploymentRepository(supabase_client=None)

        result = await repo.cleanup_stale_deployments(max_age_hours=24)

        assert result == 0


class TestDeploymentStatusTransitions:
    """Test deployment status transition logic."""

    def test_valid_status_transitions(self):
        """Test valid status transitions are supported."""
        # These are the valid status values that can be set
        valid_statuses = [s.value for s in DeploymentStatus]

        assert "pending" in valid_statuses
        assert "deploying" in valid_statuses
        assert "active" in valid_statuses
        assert "draining" in valid_statuses
        assert "rolled_back" in valid_statuses

    def test_deployment_lifecycle_states(self):
        """Test deployment lifecycle state properties."""
        # Pending: Initial state
        pending = MLDeployment(status="pending")
        assert not pending.is_active

        # Deploying: In-progress state
        deploying = MLDeployment(status="deploying")
        assert not deploying.is_active

        # Active: Running state
        active = MLDeployment(status="active")
        assert active.is_active

        # Draining: Shutting down state
        draining = MLDeployment(status="draining")
        assert not draining.is_active

        # Rolled back: Reverted state
        rolled_back = MLDeployment(status="rolled_back")
        assert not rolled_back.is_active
