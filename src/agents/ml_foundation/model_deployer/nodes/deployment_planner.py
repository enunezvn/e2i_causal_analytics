"""Deployment Planner Node - Analyzes requirements and plans deployment strategy.

Handles:
1. Model type analysis (classification, regression, causal)
2. Deployment strategy selection (direct, blue-green, canary)
3. Resource allocation planning
4. Deployment configuration generation
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class DeploymentStrategy(str, Enum):
    """Deployment strategy options."""

    DIRECT = "direct"  # Replace existing deployment immediately
    BLUE_GREEN = "blue_green"  # Deploy new version alongside old, then switch
    CANARY = "canary"  # Gradual traffic shift
    SHADOW = "shadow"  # Mirror traffic to new version without serving


class ModelType(str, Enum):
    """Supported model types for deployment."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CAUSAL = "causal"  # Causal inference models (e.g., EconML)
    ENSEMBLE = "ensemble"


@dataclass
class ResourceProfile:
    """Resource allocation profile for deployment."""

    cpu: str  # CPU request (e.g., "2", "500m")
    memory: str  # Memory request (e.g., "4Gi", "512Mi")
    gpu: Optional[str] = None  # GPU request if needed
    replicas: int = 1
    min_replicas: int = 1
    max_replicas: int = 3
    target_cpu_utilization: int = 70

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu": self.cpu,
            "memory": self.memory,
            "gpu": self.gpu,
            "replicas": self.replicas,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "target_cpu_utilization": self.target_cpu_utilization,
        }


@dataclass
class DeploymentPlan:
    """Complete deployment plan."""

    strategy: DeploymentStrategy
    model_type: ModelType
    resources: ResourceProfile
    service_template: str  # BentoML template to use
    health_check_config: Dict[str, Any]
    traffic_config: Dict[str, Any]
    rollback_config: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "strategy": self.strategy.value,
            "model_type": self.model_type.value,
            "resources": self.resources.to_dict(),
            "service_template": self.service_template,
            "health_check_config": self.health_check_config,
            "traffic_config": self.traffic_config,
            "rollback_config": self.rollback_config,
            "metadata": self.metadata,
        }


# Resource profiles by environment and model type
RESOURCE_PROFILES: Dict[str, Dict[str, ResourceProfile]] = {
    "staging": {
        "classification": ResourceProfile(
            cpu="1", memory="2Gi", replicas=1, min_replicas=1, max_replicas=2
        ),
        "regression": ResourceProfile(
            cpu="1", memory="2Gi", replicas=1, min_replicas=1, max_replicas=2
        ),
        "causal": ResourceProfile(
            cpu="2", memory="4Gi", replicas=1, min_replicas=1, max_replicas=2
        ),
        "ensemble": ResourceProfile(
            cpu="2", memory="4Gi", replicas=1, min_replicas=1, max_replicas=3
        ),
    },
    "shadow": {
        "classification": ResourceProfile(
            cpu="2", memory="4Gi", replicas=2, min_replicas=1, max_replicas=4
        ),
        "regression": ResourceProfile(
            cpu="2", memory="4Gi", replicas=2, min_replicas=1, max_replicas=4
        ),
        "causal": ResourceProfile(
            cpu="4", memory="8Gi", replicas=2, min_replicas=2, max_replicas=6
        ),
        "ensemble": ResourceProfile(
            cpu="4", memory="8Gi", replicas=2, min_replicas=2, max_replicas=6
        ),
    },
    "production": {
        "classification": ResourceProfile(
            cpu="4",
            memory="8Gi",
            replicas=3,
            min_replicas=2,
            max_replicas=10,
            target_cpu_utilization=70,
        ),
        "regression": ResourceProfile(
            cpu="4",
            memory="8Gi",
            replicas=3,
            min_replicas=2,
            max_replicas=10,
            target_cpu_utilization=70,
        ),
        "causal": ResourceProfile(
            cpu="8",
            memory="16Gi",
            replicas=3,
            min_replicas=3,
            max_replicas=15,
            target_cpu_utilization=60,
        ),
        "ensemble": ResourceProfile(
            cpu="8",
            memory="16Gi",
            replicas=4,
            min_replicas=3,
            max_replicas=20,
            target_cpu_utilization=60,
        ),
    },
}

# BentoML service templates by model type
SERVICE_TEMPLATES = {
    "classification": "ClassificationServiceTemplate",
    "regression": "RegressionServiceTemplate",
    "causal": "CausalInferenceServiceTemplate",
    "ensemble": "ClassificationServiceTemplate",  # Default for ensembles
}


def _detect_model_type(state: Dict[str, Any]) -> ModelType:
    """Detect model type from state information.

    Args:
        state: Agent state with model metadata

    Returns:
        Detected model type
    """
    # Check explicit model type if provided
    explicit_type = state.get("model_type")
    if explicit_type:
        try:
            return ModelType(explicit_type.lower())
        except ValueError:
            pass

    # Check validation metrics for clues
    validation_metrics = state.get("validation_metrics", {})

    # Causal models typically have CATE or ATE metrics
    if any(key in validation_metrics for key in ["ate", "cate", "treatment_effect"]):
        return ModelType.CAUSAL

    # Classification models have accuracy, precision, recall, f1
    classification_metrics = {"accuracy", "precision", "recall", "f1", "auc", "roc_auc"}
    if any(key in validation_metrics for key in classification_metrics):
        return ModelType.CLASSIFICATION

    # Regression models have mse, rmse, mae, r2
    regression_metrics = {"mse", "rmse", "mae", "r2", "mape"}
    if any(key in validation_metrics for key in regression_metrics):
        return ModelType.REGRESSION

    # Default to classification
    return ModelType.CLASSIFICATION


def _select_strategy(
    state: Dict[str, Any],
    target_environment: str,
    model_type: ModelType,
) -> DeploymentStrategy:
    """Select deployment strategy based on environment and requirements.

    Args:
        state: Agent state
        target_environment: Target deployment environment
        model_type: Detected model type

    Returns:
        Selected deployment strategy
    """
    # Check explicit strategy if provided
    explicit_strategy = state.get("deployment_strategy")
    if explicit_strategy:
        try:
            return DeploymentStrategy(explicit_strategy.lower())
        except ValueError:
            pass

    # Shadow environment always uses shadow strategy
    if target_environment == "shadow":
        return DeploymentStrategy.SHADOW

    # Staging uses direct deployment (fast iteration)
    if target_environment == "staging":
        return DeploymentStrategy.DIRECT

    # Production strategy selection
    if target_environment == "production":
        # Causal models need careful rollout - use canary
        if model_type == ModelType.CAUSAL:
            return DeploymentStrategy.CANARY

        # Check if this is a critical model
        is_critical = state.get("is_critical_model", False)
        if is_critical:
            return DeploymentStrategy.BLUE_GREEN

        # Default to blue-green for production safety
        return DeploymentStrategy.BLUE_GREEN

    # Default fallback
    return DeploymentStrategy.DIRECT


def _create_health_check_config(
    model_type: ModelType,
    target_environment: str,
    max_latency_ms: int,
) -> Dict[str, Any]:
    """Create health check configuration.

    Args:
        model_type: Model type
        target_environment: Target environment
        max_latency_ms: Maximum acceptable latency

    Returns:
        Health check configuration
    """
    # Base configuration
    config = {
        "enabled": True,
        "endpoint": "/health",
        "interval_seconds": 30,
        "timeout_seconds": 10,
        "success_threshold": 1,
        "failure_threshold": 3,
    }

    # Adjust for environment
    if target_environment == "production":
        config["interval_seconds"] = 15
        config["failure_threshold"] = 2
    elif target_environment == "staging":
        config["interval_seconds"] = 60
        config["failure_threshold"] = 5

    # Add latency checks
    config["latency_threshold_ms"] = max_latency_ms
    config["latency_p99_threshold_ms"] = int(max_latency_ms * 1.5)

    # Add model-specific checks
    if model_type == ModelType.CAUSAL:
        config["additional_checks"] = [
            {"name": "cate_sanity", "endpoint": "/health/cate"},
        ]

    return config


def _create_traffic_config(
    strategy: DeploymentStrategy,
    target_environment: str,
) -> Dict[str, Any]:
    """Create traffic management configuration.

    Args:
        strategy: Deployment strategy
        target_environment: Target environment

    Returns:
        Traffic configuration
    """
    if strategy == DeploymentStrategy.DIRECT:
        return {
            "type": "direct",
            "cutover_immediate": True,
        }

    if strategy == DeploymentStrategy.SHADOW:
        return {
            "type": "shadow",
            "mirror_percentage": 100,
            "serve_responses": False,  # Don't serve shadow responses
        }

    if strategy == DeploymentStrategy.BLUE_GREEN:
        return {
            "type": "blue_green",
            "switch_delay_seconds": 60,  # Wait before switching
            "keep_old_version_minutes": 30,  # Keep old version for rollback
            "health_check_before_switch": True,
        }

    if strategy == DeploymentStrategy.CANARY:
        return {
            "type": "canary",
            "stages": [
                {"percentage": 5, "duration_minutes": 15},
                {"percentage": 25, "duration_minutes": 30},
                {"percentage": 50, "duration_minutes": 30},
                {"percentage": 100, "duration_minutes": 0},
            ],
            "auto_rollback_on_error": True,
            "error_threshold_percentage": 5.0,
        }

    return {"type": "unknown"}


def _create_rollback_config(
    strategy: DeploymentStrategy,
    target_environment: str,
) -> Dict[str, Any]:
    """Create rollback configuration.

    Args:
        strategy: Deployment strategy
        target_environment: Target environment

    Returns:
        Rollback configuration
    """
    config = {
        "enabled": True,
        "automatic": target_environment == "production",
        "health_check_failures_threshold": 3,
        "error_rate_threshold": 0.05,  # 5% error rate
        "latency_p99_threshold_multiplier": 2.0,  # 2x normal latency
    }

    # Blue-green specific
    if strategy == DeploymentStrategy.BLUE_GREEN:
        config["rollback_strategy"] = "switch_back"
        config["old_version_retention_minutes"] = 60

    # Canary specific
    elif strategy == DeploymentStrategy.CANARY:
        config["rollback_strategy"] = "halt_and_switch"
        config["auto_pause_on_error"] = True

    return config


async def plan_deployment(state: Dict[str, Any]) -> Dict[str, Any]:
    """Plan deployment based on model requirements and target environment.

    This node analyzes the model and environment to create a deployment plan
    that includes:
    - Deployment strategy (direct, blue-green, canary, shadow)
    - Resource allocation
    - Health check configuration
    - Traffic management
    - Rollback configuration

    Args:
        state: Current agent state with model information

    Returns:
        State updates with deployment plan
    """
    try:
        # Extract required fields
        target_environment = state.get("target_environment", "staging")
        max_latency_ms = state.get("max_latency_ms", 500)
        deployment_name = state.get("deployment_name", "e2i-model")

        # Validate target environment
        if target_environment not in ["staging", "shadow", "production"]:
            return {
                "error": f"Invalid target environment: {target_environment}",
                "error_type": "invalid_environment",
                "deployment_plan_created": False,
            }

        # Detect model type
        model_type = _detect_model_type(state)

        # Select deployment strategy
        strategy = _select_strategy(state, target_environment, model_type)

        # Get resource profile
        resource_profile = RESOURCE_PROFILES.get(target_environment, {}).get(
            model_type.value,
            ResourceProfile(cpu="2", memory="4Gi"),
        )

        # Override with explicit resources if provided
        explicit_resources = state.get("resources")
        if explicit_resources:
            resource_profile = ResourceProfile(
                cpu=explicit_resources.get("cpu", resource_profile.cpu),
                memory=explicit_resources.get("memory", resource_profile.memory),
                gpu=explicit_resources.get("gpu"),
                replicas=resource_profile.replicas,
                min_replicas=resource_profile.min_replicas,
                max_replicas=resource_profile.max_replicas,
                target_cpu_utilization=resource_profile.target_cpu_utilization,
            )

        # Create configurations
        health_check_config = _create_health_check_config(
            model_type, target_environment, max_latency_ms
        )
        traffic_config = _create_traffic_config(strategy, target_environment)
        rollback_config = _create_rollback_config(strategy, target_environment)

        # Get service template
        service_template = SERVICE_TEMPLATES.get(model_type.value, "ClassificationServiceTemplate")

        # Create deployment plan
        plan = DeploymentPlan(
            strategy=strategy,
            model_type=model_type,
            resources=resource_profile,
            service_template=service_template,
            health_check_config=health_check_config,
            traffic_config=traffic_config,
            rollback_config=rollback_config,
            metadata={
                "planned_at": datetime.now(timezone.utc).isoformat(),
                "deployment_name": deployment_name,
                "target_environment": target_environment,
            },
        )

        # Create deployment manifest
        deployment_manifest = {
            "apiVersion": "e2i.ml/v1",
            "kind": "ModelDeployment",
            "metadata": {
                "name": deployment_name,
                "environment": target_environment,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "spec": {
                "model_type": model_type.value,
                "strategy": strategy.value,
                "resources": resource_profile.to_dict(),
                "service_template": service_template,
                "health_check": health_check_config,
                "traffic": traffic_config,
                "rollback": rollback_config,
            },
        }

        return {
            "deployment_plan": plan.to_dict(),
            "deployment_manifest": deployment_manifest,
            "deployment_strategy": strategy.value,
            "model_type": model_type.value,
            "service_template": service_template,
            "resources": resource_profile.to_dict(),
            "replicas": resource_profile.replicas,
            "cpu_limit": resource_profile.cpu,
            "memory_limit": resource_profile.memory,
            "autoscaling": {
                "min": resource_profile.min_replicas,
                "max": resource_profile.max_replicas,
                "target_cpu": resource_profile.target_cpu_utilization,
            },
            "health_check_config": health_check_config,
            "traffic_config": traffic_config,
            "rollback_config": rollback_config,
            "deployment_plan_created": True,
        }

    except Exception as e:
        return {
            "error": f"Deployment planning failed: {str(e)}",
            "error_type": "planning_error",
            "error_details": {"exception": str(e)},
            "deployment_plan_created": False,
        }


async def validate_deployment_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that deployment plan is feasible.

    Checks:
    - Required resources are available
    - Target environment is accessible
    - Model is compatible with selected template

    Args:
        state: Current agent state with deployment plan

    Returns:
        State updates with validation results
    """
    try:
        deployment_plan = state.get("deployment_plan")
        if not deployment_plan:
            return {
                "plan_validated": False,
                "deployment_plan_valid": False,
                "validation_errors": ["No deployment plan found"],
                "plan_validation_errors": ["No deployment plan found"],
            }

        validation_errors: List[str] = []
        validation_warnings: List[str] = []

        # Check strategy compatibility with environment
        strategy = deployment_plan.get("strategy")
        target_env = state.get("target_environment")

        if target_env == "staging" and strategy == "canary":
            validation_warnings.append(
                "Canary deployment in staging may not provide representative traffic"
            )

        if target_env == "production" and strategy == "direct":
            validation_errors.append(
                "Direct deployment to production is not allowed. Use blue-green or canary."
            )

        # Check resource limits
        resources = deployment_plan.get("resources", {})
        cpu = resources.get("cpu", "0")
        memory = resources.get("memory", "0")

        # Parse CPU (handle both numeric and millicores format)
        try:
            cpu_value = float(cpu.replace("m", "")) / 1000 if "m" in cpu else float(cpu)
            if cpu_value > 16:
                validation_errors.append(f"CPU request {cpu} exceeds maximum (16 cores)")
        except ValueError:
            validation_errors.append(f"Invalid CPU format: {cpu}")

        # Parse memory
        try:
            if "Gi" in memory:
                memory_gb = float(memory.replace("Gi", ""))
            elif "Mi" in memory:
                memory_gb = float(memory.replace("Mi", "")) / 1024
            else:
                memory_gb = float(memory)
            if memory_gb > 64:
                validation_errors.append(f"Memory request {memory} exceeds maximum (64Gi)")
        except ValueError:
            validation_errors.append(f"Invalid memory format: {memory}")

        # Check model type compatibility
        model_type = deployment_plan.get("model_type")
        service_template = deployment_plan.get("service_template")

        if model_type == "causal" and service_template != "CausalInferenceServiceTemplate":
            validation_warnings.append(
                f"Model type is causal but using {service_template}. "
                "Consider using CausalInferenceServiceTemplate."
            )

        # Determine validity
        is_valid = len(validation_errors) == 0

        return {
            "plan_validated": is_valid,
            "deployment_plan_valid": is_valid,  # Legacy compatibility
            "validation_errors": validation_errors,
            "plan_validation_errors": validation_errors,  # Match state key
            "validation_warnings": validation_warnings,
        }

    except Exception as e:
        return {
            "plan_validated": False,
            "deployment_plan_valid": False,
            "validation_errors": [f"Validation failed: {str(e)}"],
            "plan_validation_errors": [f"Validation failed: {str(e)}"],
            "error": str(e),
            "error_type": "validation_error",
        }
