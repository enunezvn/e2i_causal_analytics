"""Tests for deployment_planner nodes (plan_deployment, validate_deployment_plan)."""

import pytest

from src.agents.ml_foundation.model_deployer.nodes.deployment_planner import (
    DeploymentPlan,
    DeploymentStrategy,
    ModelType,
    ResourceProfile,
    plan_deployment,
    validate_deployment_plan,
)


class TestDeploymentStrategy:
    """Test DeploymentStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert DeploymentStrategy.DIRECT.value == "direct"
        assert DeploymentStrategy.BLUE_GREEN.value == "blue_green"
        assert DeploymentStrategy.CANARY.value == "canary"
        assert DeploymentStrategy.SHADOW.value == "shadow"

    def test_strategy_string_conversion(self):
        """Test strategy enum string and value access."""
        # Use .value to get the string representation
        assert DeploymentStrategy.DIRECT.value == "direct"
        assert DeploymentStrategy.BLUE_GREEN.value == "blue_green"
        # Verify the enum can be compared directly with strings
        assert DeploymentStrategy.DIRECT == "direct"
        assert DeploymentStrategy.BLUE_GREEN == "blue_green"


class TestModelType:
    """Test ModelType enum."""

    def test_model_type_values(self):
        """Test model type enum values."""
        assert ModelType.CLASSIFICATION.value == "classification"
        assert ModelType.REGRESSION.value == "regression"
        assert ModelType.CAUSAL.value == "causal"
        assert ModelType.ENSEMBLE.value == "ensemble"


class TestResourceProfile:
    """Test ResourceProfile dataclass."""

    def test_default_profile(self):
        """Test default resource profile."""
        profile = ResourceProfile(cpu="2", memory="4Gi")
        assert profile.cpu == "2"
        assert profile.memory == "4Gi"
        assert profile.gpu is None
        assert profile.replicas == 1
        assert profile.min_replicas == 1
        assert profile.max_replicas == 3
        assert profile.target_cpu_utilization == 70

    def test_custom_profile(self):
        """Test custom resource profile."""
        profile = ResourceProfile(
            cpu="4",
            memory="8Gi",
            gpu="1",
            replicas=2,
            min_replicas=2,
            max_replicas=10,
            target_cpu_utilization=60,
        )
        assert profile.cpu == "4"
        assert profile.memory == "8Gi"
        assert profile.gpu == "1"
        assert profile.replicas == 2
        assert profile.min_replicas == 2
        assert profile.max_replicas == 10
        assert profile.target_cpu_utilization == 60


class TestDeploymentPlan:
    """Test DeploymentPlan dataclass."""

    def test_deployment_plan_creation(self):
        """Test deployment plan creation."""
        resources = ResourceProfile(cpu="2", memory="4Gi")
        plan = DeploymentPlan(
            strategy=DeploymentStrategy.BLUE_GREEN,
            model_type=ModelType.CLASSIFICATION,
            resources=resources,
            service_template="classification",
            health_check_config={"interval": 30, "timeout": 10},
            traffic_config={"initial_traffic": 0.1},
            rollback_config={"error_threshold": 0.05},
        )

        assert plan.strategy == DeploymentStrategy.BLUE_GREEN
        assert plan.model_type == ModelType.CLASSIFICATION
        assert plan.resources.cpu == "2"
        assert plan.service_template == "classification"
        assert plan.health_check_config["interval"] == 30
        assert plan.traffic_config["initial_traffic"] == 0.1
        assert plan.rollback_config["error_threshold"] == 0.05


class TestPlanDeployment:
    """Test plan_deployment node."""

    @pytest.mark.asyncio
    async def test_plan_deployment_production(self):
        """Test deployment planning for production environment."""
        state = {
            "target_environment": "production",
            "model_type": "classification",
            "experiment_id": "exp_123",
        }

        result = await plan_deployment(state)

        assert result["deployment_strategy"] == "blue_green"
        assert result["deployment_plan"] is not None
        assert result["service_template"] == "ClassificationServiceTemplate"
        assert result["health_check_config"] is not None
        assert result["traffic_config"] is not None
        assert result["rollback_config"] is not None
        assert result["deployment_plan_created"] is True

    @pytest.mark.asyncio
    async def test_plan_deployment_staging(self):
        """Test deployment planning for staging environment."""
        state = {
            "target_environment": "staging",
            "model_type": "regression",
            "experiment_id": "exp_456",
        }

        result = await plan_deployment(state)

        assert result["deployment_strategy"] == "direct"
        assert result["service_template"] == "RegressionServiceTemplate"
        assert result["deployment_plan_created"] is True

    @pytest.mark.asyncio
    async def test_plan_deployment_shadow(self):
        """Test deployment planning for shadow environment."""
        state = {
            "target_environment": "shadow",
            "model_type": "causal",
            "experiment_id": "exp_789",
        }

        result = await plan_deployment(state)

        assert result["deployment_strategy"] == "shadow"
        assert result["service_template"] == "CausalInferenceServiceTemplate"
        assert result["deployment_plan_created"] is True

    @pytest.mark.asyncio
    async def test_plan_deployment_causal_model(self):
        """Test deployment planning for causal model type."""
        state = {
            "target_environment": "production",
            "model_type": "causal",
            "experiment_id": "exp_causal",
        }

        result = await plan_deployment(state)

        assert result["service_template"] == "CausalInferenceServiceTemplate"
        # Causal models in production use canary strategy
        assert result["deployment_strategy"] == "canary"

    @pytest.mark.asyncio
    async def test_plan_deployment_ensemble_model(self):
        """Test deployment planning for ensemble model type."""
        state = {
            "target_environment": "production",
            "model_type": "ensemble",
            "experiment_id": "exp_ensemble",
        }

        result = await plan_deployment(state)

        # Ensemble uses ClassificationServiceTemplate as default
        assert result["service_template"] == "ClassificationServiceTemplate"

    @pytest.mark.asyncio
    async def test_plan_deployment_auto_detect_model_type(self):
        """Test auto-detection of model type from model_uri."""
        state = {
            "target_environment": "staging",
            "model_uri": "mlflow://models/xgboost_classifier_v1",
            "experiment_id": "exp_auto",
        }

        result = await plan_deployment(state)

        assert result["deployment_plan"] is not None

    @pytest.mark.asyncio
    async def test_plan_deployment_default_environment(self):
        """Test deployment planning with default environment."""
        state = {
            "model_type": "classification",
            "experiment_id": "exp_default",
        }

        result = await plan_deployment(state)

        # Should default to staging
        assert result["deployment_strategy"] == "direct"

    @pytest.mark.asyncio
    async def test_plan_deployment_resources_included(self):
        """Test that resource profile is included in plan."""
        state = {
            "target_environment": "production",
            "model_type": "classification",
            "experiment_id": "exp_resources",
        }

        result = await plan_deployment(state)

        plan = result["deployment_plan"]
        assert "resources" in plan
        resources = plan["resources"]
        assert "cpu" in resources
        assert "memory" in resources

    @pytest.mark.asyncio
    async def test_plan_deployment_rollback_action(self):
        """Test deployment planning with rollback action."""
        state = {
            "deployment_action": "rollback",
            "rollback_to_deployment_id": "deploy_prev_123",
            "experiment_id": "exp_rollback",
        }

        result = await plan_deployment(state)

        # Rollback action should skip normal planning
        assert result.get("deployment_strategy") is not None


class TestValidateDeploymentPlan:
    """Test validate_deployment_plan node."""

    @pytest.mark.asyncio
    async def test_validate_plan_success(self):
        """Test successful plan validation."""
        state = {
            "deployment_plan": {
                "strategy": "blue_green",
                "model_type": "classification",
                "resources": {"cpu": "2", "memory": "4Gi"},
                "service_template": "ClassificationServiceTemplate",
                "health_check_config": {"interval": 30},
                "traffic_config": {"initial_traffic": 0.1},
                "rollback_config": {"error_threshold": 0.05},
            },
            "target_environment": "production",
        }

        result = await validate_deployment_plan(state)

        assert result["deployment_plan_valid"] is True
        assert result["validation_errors"] == []

    @pytest.mark.asyncio
    async def test_validate_plan_missing_plan(self):
        """Test validation with missing plan."""
        state = {
            "target_environment": "production",
        }

        result = await validate_deployment_plan(state)

        assert result["deployment_plan_valid"] is False
        assert len(result["validation_errors"]) > 0

    @pytest.mark.asyncio
    async def test_validate_plan_missing_resources(self):
        """Test validation with missing resources."""
        state = {
            "deployment_plan": {
                "strategy": "blue_green",
                "model_type": "classification",
                "service_template": "ClassificationServiceTemplate",
                "health_check_config": {},
                "traffic_config": {},
                "rollback_config": {},
            },
            "target_environment": "production",
        }

        result = await validate_deployment_plan(state)

        # Should still validate - missing resources use defaults
        # The validator parses cpu="0" and memory="0" which are valid
        assert result["deployment_plan_valid"] is True

    @pytest.mark.asyncio
    async def test_validate_plan_canary_production(self):
        """Test validation of canary strategy in production."""
        state = {
            "deployment_plan": {
                "strategy": "canary",
                "model_type": "classification",
                "resources": {"cpu": "4", "memory": "8Gi"},
                "service_template": "ClassificationServiceTemplate",
                "health_check_config": {"interval": 30},
                "traffic_config": {"initial_traffic": 0.05, "stages": [0.05, 0.25, 0.5, 1.0]},
                "rollback_config": {"error_threshold": 0.01},
            },
            "target_environment": "production",
        }

        result = await validate_deployment_plan(state)

        assert result["deployment_plan_valid"] is True

    @pytest.mark.asyncio
    async def test_validate_plan_direct_production_rejected(self):
        """Test that direct deployment to production is rejected."""
        state = {
            "deployment_plan": {
                "strategy": "direct",
                "model_type": "classification",
                "resources": {"cpu": "2", "memory": "4Gi"},
            },
            "target_environment": "production",
        }

        result = await validate_deployment_plan(state)

        assert result["deployment_plan_valid"] is False
        assert any(
            "Direct deployment to production is not allowed" in err
            for err in result["validation_errors"]
        )

    @pytest.mark.asyncio
    async def test_validate_plan_empty_plan(self):
        """Test validation with empty plan."""
        state = {
            "deployment_plan": {},
            "target_environment": "staging",
        }

        result = await validate_deployment_plan(state)

        # Empty dict is falsy in Python, so it's treated as "no plan"
        assert result["deployment_plan_valid"] is False
        assert "No deployment plan found" in result["validation_errors"]
