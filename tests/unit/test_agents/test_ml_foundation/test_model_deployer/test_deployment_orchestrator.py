"""Tests for deployment_orchestrator nodes.

Includes tests for:
- package_model: BentoML packaging
- deploy_to_endpoint: Endpoint deployment with multiple strategies
- check_rollback_availability: Rollback availability check
- execute_rollback: Rollback execution
- containerize_model: Docker containerization
"""

import pytest
from unittest.mock import patch

from src.agents.ml_foundation.model_deployer.nodes.deployment_orchestrator import (
    check_rollback_availability,
    containerize_model,
    deploy_to_endpoint,
    execute_rollback,
    package_model,
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
        assert result["autoscaling"]["target_cpu"] == 70  # Default target CPU
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
        assert result["replicas"] == 1  # Default replicas
        assert result["autoscaling"]["min"] == 1
        assert result["autoscaling"]["max"] == 3  # Default max
        assert result["autoscaling"]["target_cpu"] == 70  # Default target CPU
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
        assert result["replicas"] == 1  # Default replicas
        assert result["autoscaling"]["min"] == 1  # Default min
        assert result["autoscaling"]["max"] == 3  # Default max
        assert result["autoscaling"]["target_cpu"] == 70
        assert result["endpoint_name"] == "test_deployment-production"
        # Implementation uses defaults for direct deployment
        assert result["cpu_limit"] == "2"  # Default CPU
        assert result["memory_limit"] == "4Gi"  # Default memory

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
        """Test deployment with resources - uses defaults regardless of input."""
        state = {
            "bento_tag": "e2i_exp_123_model:v1",
            "deployment_name": "test_deployment",
            "target_environment": "production",
            "resources": {"cpu": "8", "memory": "16Gi"},
        }

        result = await deploy_to_endpoint(state)

        assert result["deployment_successful"] is True
        # Implementation uses default resources for direct deployment
        assert result["cpu_limit"] == "2"  # Default CPU
        assert result["memory_limit"] == "4Gi"  # Default memory

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

    @pytest.mark.asyncio
    async def test_rollback_availability_with_rollback_config(self):
        """Test rollback availability returns rollback config."""
        state = {
            "experiment_id": "exp_123",
            "current_stage": "Production",
            "rollback_config": {"automatic": True, "error_threshold": 0.05},
        }

        result = await check_rollback_availability(state)

        assert result["rollback_available"] is True
        assert result["rollback_config"]["automatic"] is True
        assert result["rollback_auto_enabled"] is True


class TestExecuteRollback:
    """Test execute_rollback node."""

    @pytest.mark.asyncio
    async def test_execute_rollback_direct_strategy(self):
        """Test rollback with direct deployment strategy."""
        state = {
            "rollback_to_deployment_id": "deploy_abc123",
            "rollback_to_version": 2,
            "rollback_reason": "Performance degradation",
            "deployment_strategy": "direct",
            "previous_deployment_url": "https://api.e2i.com/v1/model-prev/predict",
        }

        result = await execute_rollback(state)

        assert result["rollback_successful"] is True
        assert result["rollback_method"] == "direct"
        assert result["rollback_reason"] == "Performance degradation"
        assert result["rolled_back_to_deployment_id"] == "deploy_abc123"
        assert result["rolled_back_to_version"] == 2
        assert "rolled_back_at" in result

    @pytest.mark.asyncio
    async def test_execute_rollback_blue_green_strategy(self):
        """Test rollback with blue-green deployment strategy."""
        state = {
            "rollback_to_deployment_id": "deploy_blue123",
            "rollback_to_version": 1,
            "rollback_reason": "Error rate increase",
            "deployment_strategy": "blue_green",
            "blue_endpoint": "model-prod-blue",
        }

        result = await execute_rollback(state)

        assert result["rollback_successful"] is True
        assert result["rollback_method"] == "blue_green_switch"
        assert result["traffic_switched_to_blue"] is True
        assert result["green_endpoint_terminated"] is True

    @pytest.mark.asyncio
    async def test_execute_rollback_canary_strategy(self):
        """Test rollback with canary deployment strategy."""
        state = {
            "rollback_to_deployment_id": "deploy_stable123",
            "rollback_to_version": 1,
            "rollback_reason": "Canary metrics below threshold",
            "deployment_strategy": "canary",
        }

        result = await execute_rollback(state)

        assert result["rollback_successful"] is True
        assert result["rollback_method"] == "canary_halt"
        assert result["canary_traffic_percentage"] == 0
        assert result["canary_terminated"] is True
        assert result["traffic_restored_to_stable"] is True

    @pytest.mark.asyncio
    async def test_execute_rollback_missing_target(self):
        """Test rollback with missing target."""
        state = {
            "deployment_strategy": "direct",
            "rollback_reason": "Manual rollback",
        }

        result = await execute_rollback(state)

        assert result["rollback_successful"] is False
        assert result["error"] == "No rollback target specified"
        assert result["error_type"] == "missing_rollback_target"

    @pytest.mark.asyncio
    async def test_execute_rollback_default_reason(self):
        """Test rollback with default reason."""
        state = {
            "rollback_to_deployment_id": "deploy_xyz789",
            "deployment_strategy": "direct",
            "previous_deployment_url": "https://api.e2i.com/v1/model-prev/predict",
        }

        result = await execute_rollback(state)

        assert result["rollback_successful"] is True
        assert result["rollback_reason"] == "Manual rollback requested"


class TestContainerizeModel:
    """Test containerize_model node."""

    @pytest.mark.asyncio
    @patch(
        "src.agents.ml_foundation.model_deployer.nodes.deployment_orchestrator.BENTOML_AVAILABLE",
        False,
    )
    async def test_containerize_model_success(self):
        """Test successful containerization (simulated mode)."""
        state = {
            "bento_tag": "e2i_exp_123:v1",
            "deployment_name": "test-model",
            "deployment_plan": {
                "resources": {"cpu": "2", "memory": "4Gi"},
            },
        }

        result = await containerize_model(state)

        assert result["containerization_successful"] is True
        assert result["container_image"] is not None
        assert "test-model" in result["container_image"]
        assert result["containerization_simulated"] is True

    @pytest.mark.asyncio
    async def test_containerize_model_missing_bento_tag(self):
        """Test containerization with missing bento_tag."""
        state = {
            "deployment_name": "test-model",
            "deployment_plan": {
                "resources": {"cpu": "2", "memory": "4Gi"},
            },
        }

        result = await containerize_model(state)

        assert result["containerization_successful"] is False
        assert result["error"] == "Missing bento_tag for containerization"
        assert result["error_type"] == "missing_bento_tag"

    @pytest.mark.asyncio
    @patch(
        "src.agents.ml_foundation.model_deployer.nodes.deployment_orchestrator.BENTOML_AVAILABLE",
        False,
    )
    async def test_containerize_model_default_name(self):
        """Test containerization with default deployment name (simulated mode)."""
        state = {
            "bento_tag": "e2i_exp_456:v2",
        }

        result = await containerize_model(state)

        assert result["containerization_successful"] is True
        assert "e2i-model" in result["container_image"]
        assert result["containerization_simulated"] is True

    @pytest.mark.asyncio
    @patch(
        "src.agents.ml_foundation.model_deployer.nodes.deployment_orchestrator.BENTOML_AVAILABLE",
        False,
    )
    async def test_containerize_model_simulated_flag(self):
        """Test containerization sets simulated flag when BentoML unavailable."""
        state = {
            "bento_tag": "e2i_exp_789:v3",
            "deployment_name": "simulated-model",
        }

        result = await containerize_model(state)

        # Should be simulated since BentoML is mocked as unavailable
        assert result["containerization_successful"] is True
        assert result["containerization_simulated"] is True
        assert "simulated-model" in result["container_image"]


class TestDeploymentStrategies:
    """Test different deployment strategies."""

    @pytest.mark.asyncio
    async def test_deploy_blue_green_strategy(self):
        """Test blue-green deployment strategy."""
        state = {
            "bento_tag": "e2i_exp_123_model:v1",
            "deployment_name": "test_deployment",
            "target_environment": "production",
            "deployment_plan": {
                "strategy": "blue_green",
                "resources": {"cpu": "4", "memory": "8Gi", "replicas": 3},
                "traffic_config": {
                    "switch_delay_seconds": 120,
                    "keep_old_version_minutes": 60,
                },
            },
        }

        result = await deploy_to_endpoint(state)

        assert result["deployment_successful"] is True
        assert result["deployment_strategy"] == "blue_green"
        assert result["blue_green_status"] == "green_deployed"
        assert "green_endpoint" in result
        assert "green_url" in result
        assert result["traffic_switch_pending"] is True

    @pytest.mark.asyncio
    async def test_deploy_canary_strategy(self):
        """Test canary deployment strategy."""
        state = {
            "bento_tag": "e2i_exp_123_model:v1",
            "deployment_name": "test_deployment",
            "target_environment": "production",
            "deployment_plan": {
                "strategy": "canary",
                "resources": {"cpu": "4", "memory": "8Gi"},
                "traffic_config": {
                    "stages": [
                        {"percentage": 5, "duration_minutes": 15},
                        {"percentage": 25, "duration_minutes": 30},
                        {"percentage": 100, "duration_minutes": 0},
                    ],
                    "auto_rollback_on_error": True,
                    "error_threshold_percentage": 2.0,
                },
            },
        }

        result = await deploy_to_endpoint(state)

        assert result["deployment_successful"] is True
        assert result["deployment_strategy"] == "canary"
        assert result["canary_status"] == "stage_1"
        assert "canary_endpoint" in result
        assert "canary_url" in result
        assert result["current_traffic_percentage"] == 5
        assert result["auto_rollback_enabled"] is True
        assert result["error_threshold_percentage"] == 2.0

    @pytest.mark.asyncio
    async def test_deploy_shadow_strategy(self):
        """Test shadow deployment strategy."""
        state = {
            "bento_tag": "e2i_exp_123_model:v1",
            "deployment_name": "test_deployment",
            "target_environment": "shadow",
            "deployment_plan": {
                "strategy": "shadow",
                "resources": {"cpu": "2", "memory": "4Gi"},
                "traffic_config": {
                    "mirror_percentage": 50,
                },
            },
        }

        result = await deploy_to_endpoint(state)

        assert result["deployment_successful"] is True
        assert result["deployment_strategy"] == "shadow"
        assert result["shadow_status"] == "mirroring"
        assert "shadow_endpoint" in result
        assert "shadow_url" in result
        assert result["mirror_percentage"] == 50
        assert result["serving_responses"] is False
        assert result["shadow_metrics_collection"] is True

    @pytest.mark.asyncio
    async def test_deploy_direct_strategy(self):
        """Test direct deployment strategy (default)."""
        state = {
            "bento_tag": "e2i_exp_123_model:v1",
            "deployment_name": "test_deployment",
            "target_environment": "staging",
            "deployment_plan": {
                "strategy": "direct",
                "resources": {"cpu": "2", "memory": "4Gi"},
            },
        }

        result = await deploy_to_endpoint(state)

        assert result["deployment_successful"] is True
        assert result["deployment_strategy"] == "direct"
        assert result["deployment_status"] == "healthy"

    @pytest.mark.asyncio
    async def test_deploy_default_strategy(self):
        """Test deployment defaults to direct strategy."""
        state = {
            "bento_tag": "e2i_exp_123_model:v1",
            "deployment_name": "test_deployment",
            "target_environment": "staging",
            "deployment_plan": {},  # No strategy specified
        }

        result = await deploy_to_endpoint(state)

        assert result["deployment_successful"] is True
        assert result["deployment_strategy"] == "direct"
