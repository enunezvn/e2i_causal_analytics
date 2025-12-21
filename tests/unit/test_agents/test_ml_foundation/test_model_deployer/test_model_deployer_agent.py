"""Integration tests for ModelDeployerAgent."""

import pytest

from src.agents.ml_foundation.model_deployer.agent import ModelDeployerAgent


class TestModelDeployerAgent:
    """Integration tests for complete model_deployer workflow."""

    @pytest.mark.asyncio
    async def test_full_deployment_to_staging(self):
        """Test complete deployment workflow to staging."""
        agent = ModelDeployerAgent()

        input_data = {
            "model_uri": "mlflow://models/test_model/1",
            "experiment_id": "exp_123",
            "validation_metrics": {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.91,
            },
            "success_criteria_met": True,
            "deployment_name": "test_deployment",
            "target_environment": "staging",
        }

        result = await agent.run(input_data)

        # Check overall status
        assert result["status"] == "completed"
        assert result["deployment_successful"] is True
        assert result["health_check_passed"] is True

        # Check deployment manifest
        manifest = result["deployment_manifest"]
        assert manifest["environment"] == "staging"
        assert manifest["experiment_id"] == "exp_123"
        assert "endpoint_url" in manifest
        assert "staging" in manifest["endpoint_url"]

        # Check version record
        version_record = result["version_record"]
        assert version_record["stage"] == "Staging"
        assert version_record["version"] == 1

        # Check BentoML tag
        assert result["bentoml_tag"] == "e2i_exp_123_model:v1"

    @pytest.mark.asyncio
    async def test_full_deployment_to_production_with_valid_shadow(self):
        """Test complete deployment to production with valid shadow mode metrics."""
        agent = ModelDeployerAgent()

        input_data = {
            "model_uri": "mlflow://models/test_model/2",
            "experiment_id": "exp_456",
            "validation_metrics": {
                "accuracy": 0.95,
                "precision": 0.93,
                "recall": 0.94,
            },
            "success_criteria_met": True,
            "deployment_name": "prod_deployment",
            "target_environment": "production",
            # Valid shadow mode metrics
            "shadow_mode_duration_hours": 48,
            "shadow_mode_requests": 5000,
            "shadow_mode_error_rate": 0.005,
            "shadow_mode_latency_p99_ms": 100,
        }

        result = await agent.run(input_data)

        assert result["status"] == "completed"
        assert result["deployment_successful"] is True

        manifest = result["deployment_manifest"]
        assert manifest["environment"] == "production"

        version_record = result["version_record"]
        assert version_record["stage"] == "Production"

    @pytest.mark.asyncio
    async def test_deployment_fails_without_shadow_validation(self):
        """Test that production deployment fails without valid shadow metrics."""
        agent = ModelDeployerAgent()

        input_data = {
            "model_uri": "mlflow://models/test_model/3",
            "experiment_id": "exp_789",
            "validation_metrics": {"accuracy": 0.90},
            "success_criteria_met": True,
            "deployment_name": "failed_prod_deployment",
            "target_environment": "production",
            # Invalid shadow mode metrics (insufficient duration)
            "shadow_mode_duration_hours": 10,
            "shadow_mode_requests": 1500,
            "shadow_mode_error_rate": 0.005,
            "shadow_mode_latency_p99_ms": 100,
        }

        with pytest.raises(RuntimeError):
            await agent.run(input_data)

    @pytest.mark.asyncio
    async def test_promotion_only_no_deployment(self):
        """Test promotion-only action (no deployment)."""
        agent = ModelDeployerAgent()

        input_data = {
            "model_uri": "mlflow://models/test_model/4",
            "experiment_id": "exp_promote",
            "validation_metrics": {"accuracy": 0.88},
            "success_criteria_met": True,
            "deployment_name": "promote_only",
            "target_environment": "staging",
            "deployment_action": "promote",
        }

        result = await agent.run(input_data)

        # Promotion successful but no deployment
        assert result["status"] == "completed"
        assert result["deployment_successful"] is False  # No deployment happened
        assert result["version_record"]["stage"] == "Staging"

    @pytest.mark.asyncio
    async def test_custom_resources(self):
        """Test deployment with custom resource configuration."""
        agent = ModelDeployerAgent()

        input_data = {
            "model_uri": "mlflow://models/test_model/5",
            "experiment_id": "exp_custom",
            "validation_metrics": {"accuracy": 0.91},
            "success_criteria_met": True,
            "deployment_name": "custom_resources",
            "target_environment": "staging",
            "resources": {"cpu": "4", "memory": "8Gi"},
            "max_batch_size": 200,
            "max_latency_ms": 50,
        }

        result = await agent.run(input_data)

        assert result["status"] == "completed"
        manifest = result["deployment_manifest"]
        assert manifest["resources"]["cpu"] == "4"
        assert manifest["resources"]["memory"] == "8Gi"

    @pytest.mark.asyncio
    async def test_missing_required_field(self):
        """Test that missing required fields raise ValueError."""
        agent = ModelDeployerAgent()

        input_data = {
            "model_uri": "mlflow://models/test_model/6",
            "experiment_id": "exp_missing",
            # Missing validation_metrics
            "success_criteria_met": True,
            "deployment_name": "missing_field",
        }

        with pytest.raises(ValueError) as exc_info:
            await agent.run(input_data)

        assert "Missing required field: validation_metrics" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_shadow_deployment(self):
        """Test deployment to shadow environment."""
        agent = ModelDeployerAgent()

        input_data = {
            "model_uri": "mlflow://models/test_model/7",
            "experiment_id": "exp_shadow",
            "validation_metrics": {"accuracy": 0.93},
            "success_criteria_met": True,
            "deployment_name": "shadow_test",
            "target_environment": "shadow",
        }

        result = await agent.run(input_data)

        assert result["status"] == "completed"
        manifest = result["deployment_manifest"]
        assert manifest["environment"] == "shadow"
        assert "shadow" in manifest["endpoint_url"]

    @pytest.mark.asyncio
    async def test_rollback_availability_production(self):
        """Test rollback availability for production deployment."""
        agent = ModelDeployerAgent()

        input_data = {
            "model_uri": "mlflow://models/test_model/8",
            "experiment_id": "exp_rollback",
            "validation_metrics": {"accuracy": 0.94},
            "success_criteria_met": True,
            "deployment_name": "rollback_test",
            "target_environment": "production",
            "shadow_mode_duration_hours": 30,
            "shadow_mode_requests": 2000,
            "shadow_mode_error_rate": 0.003,
            "shadow_mode_latency_p99_ms": 80,
        }

        result = await agent.run(input_data)

        assert result["status"] == "completed"
        assert result["rollback_available"] is True

    @pytest.mark.asyncio
    async def test_rollback_not_available_staging(self):
        """Test rollback not available for staging deployment."""
        agent = ModelDeployerAgent()

        input_data = {
            "model_uri": "mlflow://models/test_model/9",
            "experiment_id": "exp_no_rollback",
            "validation_metrics": {"accuracy": 0.89},
            "success_criteria_met": True,
            "deployment_name": "no_rollback_test",
            "target_environment": "staging",
        }

        result = await agent.run(input_data)

        assert result["status"] == "completed"
        assert result["rollback_available"] is False

    @pytest.mark.asyncio
    async def test_default_values(self):
        """Test that default values are applied correctly."""
        agent = ModelDeployerAgent()

        input_data = {
            "model_uri": "mlflow://models/test_model/10",
            "experiment_id": "exp_defaults",
            "validation_metrics": {"accuracy": 0.87},
            "success_criteria_met": True,
            "deployment_name": "defaults_test",
            # No target_environment (should default to staging)
            # No resources (should use defaults)
            # No max_batch_size, max_latency_ms (should use defaults)
        }

        result = await agent.run(input_data)

        assert result["status"] == "completed"
        manifest = result["deployment_manifest"]
        assert manifest["environment"] == "staging"  # default
        assert manifest["resources"]["cpu"] == "2"  # default
        assert manifest["resources"]["memory"] == "4Gi"  # default
