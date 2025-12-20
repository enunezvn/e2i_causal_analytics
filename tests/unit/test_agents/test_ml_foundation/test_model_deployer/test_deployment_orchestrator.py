"""Tests for deployment_orchestrator nodes (package_model, deploy_to_endpoint, check_rollback_availability)."""

import pytest
from src.agents.ml_foundation.model_deployer.nodes.deployment_orchestrator import (
    package_model,
    deploy_to_endpoint,
    check_rollback_availability,
)


class TestPackageModel:
    """Test package_model node."""

    @pytest.mark.asyncio
    async def test_package_model_success(self):
        """Test successful BentoML packaging."""
        state = {
            "model_uri": "mlflow://models/test_model/1",
            "experiment_id": "exp_123",
            "model_version": 1,
        }

        result = await package_model(state)

        assert result["bento_packaging_successful"] is True
        assert result["bento_tag"] == "e2i_exp_123_model:v1"
        assert result["final_bento_tag"] == "e2i_exp_123_model:v1"

    @pytest.mark.asyncio
    async def test_package_model_missing_model_uri(self):
        """Test packaging with missing model_uri."""
        state = {
            "experiment_id": "exp_123",
            "model_version": 1,
        }

        result = await package_model(state)

        assert result["bento_packaging_successful"] is False
        assert result["error"] == "Missing model_uri for packaging"
        assert result["error_type"] == "missing_model_uri"

    @pytest.mark.asyncio
    async def test_package_model_with_version_2(self):
        """Test packaging with version 2."""
        state = {
            "model_uri": "mlflow://models/test_model/2",
            "experiment_id": "exp_456",
            "model_version": 2,
        }

        result = await package_model(state)

        assert result["bento_packaging_successful"] is True
        assert result["bento_tag"] == "e2i_exp_456_model:v2"


class TestDeployToEndpoint:
    """Test deploy_to_endpoint node."""

    @pytest.mark.asyncio
    async def test_deploy_to_staging(self):
        """Test deployment to staging environment."""
        state = {
            "bento_tag": "e2i_exp_123_model:v1",
            "deployment_name": "test_deployment",
            "target_environment": "staging",
            "resources": {"cpu": "2", "memory": "4Gi"},
        }

        result = await deploy_to_endpoint(state)

        assert result["deployment_successful"] is True
        assert result["deployment_status"] == "healthy"
        assert result["replicas"] == 1
        assert result["autoscaling"]["min"] == 1
        assert result["autoscaling"]["max"] == 3
        assert result["autoscaling"]["target_cpu"] == 80
        assert result["endpoint_name"] == "test_deployment-staging"
        assert "https://api.e2i.com/v1/test_deployment-staging/predict" in result["endpoint_url"]
        assert "deployment_id" in result
        assert "deployed_at" in result
        assert result["deployed_by"] == "model_deployer"

    @pytest.mark.asyncio
    async def test_deploy_to_shadow(self):
        """Test deployment to shadow environment."""
        state = {
            "bento_tag": "e2i_exp_123_model:v1",
            "deployment_name": "test_deployment",
            "target_environment": "shadow",
            "resources": {"cpu": "2", "memory": "4Gi"},
        }

        result = await deploy_to_endpoint(state)

        assert result["deployment_successful"] is True
        assert result["deployment_status"] == "healthy"
        assert result["replicas"] == 2
        assert result["autoscaling"]["min"] == 1
        assert result["autoscaling"]["max"] == 5
        assert result["autoscaling"]["target_cpu"] == 80
        assert result["endpoint_name"] == "test_deployment-shadow"

    @pytest.mark.asyncio
    async def test_deploy_to_production(self):
        """Test deployment to production environment."""
        state = {
            "bento_tag": "e2i_exp_123_model:v1",
            "deployment_name": "test_deployment",
            "target_environment": "production",
            "resources": {"cpu": "4", "memory": "8Gi"},
        }

        result = await deploy_to_endpoint(state)

        assert result["deployment_successful"] is True
        assert result["deployment_status"] == "healthy"
        assert result["replicas"] == 3
        assert result["autoscaling"]["min"] == 2
        assert result["autoscaling"]["max"] == 10
        assert result["autoscaling"]["target_cpu"] == 70
        assert result["endpoint_name"] == "test_deployment-production"
        assert result["cpu_limit"] == "4"
        assert result["memory_limit"] == "8Gi"

    @pytest.mark.asyncio
    async def test_deploy_missing_bento_tag(self):
        """Test deployment with missing bento_tag."""
        state = {
            "deployment_name": "test_deployment",
            "target_environment": "staging",
            "resources": {"cpu": "2", "memory": "4Gi"},
        }

        result = await deploy_to_endpoint(state)

        assert result["deployment_successful"] is False
        assert result["error"] == "Missing bento_tag for deployment"
        assert result["error_type"] == "missing_bento_tag"
        assert result["deployment_status"] == "failed"

    @pytest.mark.asyncio
    async def test_deploy_with_default_resources(self):
        """Test deployment with default resources."""
        state = {
            "bento_tag": "e2i_exp_123_model:v1",
            "deployment_name": "test_deployment",
            "target_environment": "staging",
        }

        result = await deploy_to_endpoint(state)

        assert result["deployment_successful"] is True
        assert result["cpu_limit"] == "2"  # default
        assert result["memory_limit"] == "4Gi"  # default

    @pytest.mark.asyncio
    async def test_deploy_custom_resources(self):
        """Test deployment with custom resources."""
        state = {
            "bento_tag": "e2i_exp_123_model:v1",
            "deployment_name": "test_deployment",
            "target_environment": "production",
            "resources": {"cpu": "8", "memory": "16Gi"},
        }

        result = await deploy_to_endpoint(state)

        assert result["deployment_successful"] is True
        assert result["cpu_limit"] == "8"
        assert result["memory_limit"] == "16Gi"

    @pytest.mark.asyncio
    async def test_deploy_generates_unique_deployment_id(self):
        """Test that each deployment generates a unique deployment_id."""
        state = {
            "bento_tag": "e2i_exp_123_model:v1",
            "deployment_name": "test_deployment",
            "target_environment": "staging",
            "resources": {"cpu": "2", "memory": "4Gi"},
        }

        result1 = await deploy_to_endpoint(state)
        result2 = await deploy_to_endpoint(state)

        assert result1["deployment_id"] != result2["deployment_id"]
        assert result1["deployment_id"].startswith("deploy_")
        assert result2["deployment_id"].startswith("deploy_")


class TestCheckRollbackAvailability:
    """Test check_rollback_availability node."""

    @pytest.mark.asyncio
    async def test_rollback_available_production_stage(self):
        """Test rollback availability for Production stage."""
        state = {
            "experiment_id": "exp_123",
            "current_stage": "Production",
        }

        result = await check_rollback_availability(state)

        assert result["rollback_available"] is True
        assert result["previous_deployment_id"] is not None
        assert result["previous_deployment_url"] is not None
        assert "deploy_prev_" in result["previous_deployment_id"]

    @pytest.mark.asyncio
    async def test_rollback_available_shadow_stage(self):
        """Test rollback availability for Shadow stage."""
        state = {
            "experiment_id": "exp_123",
            "current_stage": "Shadow",
        }

        result = await check_rollback_availability(state)

        assert result["rollback_available"] is True
        assert result["previous_deployment_id"] is not None
        assert result["previous_deployment_url"] is not None

    @pytest.mark.asyncio
    async def test_rollback_not_available_none_stage(self):
        """Test rollback not available for None stage."""
        state = {
            "experiment_id": "exp_123",
            "current_stage": "None",
        }

        result = await check_rollback_availability(state)

        assert result["rollback_available"] is False
        assert result["previous_deployment_id"] is None
        assert result["previous_deployment_url"] is None

    @pytest.mark.asyncio
    async def test_rollback_not_available_staging_stage(self):
        """Test rollback not available for Staging stage."""
        state = {
            "experiment_id": "exp_123",
            "current_stage": "Staging",
        }

        result = await check_rollback_availability(state)

        assert result["rollback_available"] is False
        assert result["previous_deployment_id"] is None
        assert result["previous_deployment_url"] is None

    @pytest.mark.asyncio
    async def test_rollback_availability_with_default_stage(self):
        """Test rollback availability with default (None) stage."""
        state = {
            "experiment_id": "exp_456",
        }

        result = await check_rollback_availability(state)

        assert result["rollback_available"] is False
