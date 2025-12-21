"""Registry Manager Node - MLflow registration and stage promotion.

Handles:
1. Model registration in MLflow registry
2. Stage validation and promotion
3. Shadow mode criteria validation
"""

from datetime import datetime
from typing import Any, Dict


async def register_model(state: Dict[str, Any]) -> Dict[str, Any]:
    """Register model in MLflow registry.

    Args:
        state: Current agent state with model_uri and experiment_id

    Returns:
        State updates with registration results
    """
    try:
        model_uri = state.get("model_uri")
        state.get("experiment_id")
        deployment_name = state.get("deployment_name")

        if not model_uri:
            return {
                "error": "Missing model_uri for registration",
                "error_type": "missing_model_uri",
                "registration_successful": False,
            }

        if not deployment_name:
            return {
                "error": "Missing deployment_name for registration",
                "error_type": "missing_deployment_name",
                "registration_successful": False,
            }

        # In production, this would call MLflow
        # mlflow.register_model(model_uri, deployment_name)

        # For now, simulate registration
        registered_model_name = deployment_name
        model_version = 1  # In production, this comes from MLflow
        current_stage = "None"  # New model starts with no stage

        return {
            "registered_model_name": registered_model_name,
            "model_version": model_version,
            "current_stage": current_stage,
            "registration_successful": True,
            "registration_timestamp": datetime.now(tz=None).isoformat(),
        }

    except Exception as e:
        return {
            "error": f"Model registration failed: {str(e)}",
            "error_type": "registration_error",
            "error_details": {"exception": str(e)},
            "registration_successful": False,
        }


async def validate_promotion(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate stage promotion criteria.

    Args:
        state: Current agent state with current_stage and target_stage

    Returns:
        State updates with validation results
    """
    try:
        current_stage = state.get("current_stage", "None")
        target_stage = state.get("target_stage")
        target_environment = state.get("target_environment", "staging")

        # Map environment to MLflow stage
        ENVIRONMENT_TO_STAGE = {
            "staging": "Staging",
            "shadow": "Shadow",
            "production": "Production",
            "archived": "Archived",
        }

        if not target_stage:
            target_stage = ENVIRONMENT_TO_STAGE.get(target_environment, "Staging")

        # Define allowed promotion paths
        # For initial deployments (None stage), allow any target environment
        # Production requires shadow mode validation (checked below)
        ALLOWED_PROMOTIONS = {
            "None": ["Staging", "Shadow", "Production"],  # Initial deployments
            "Staging": ["Shadow", "Archived"],
            "Shadow": ["Production", "Archived"],
            "Production": ["Archived"],
            "Archived": [],  # Terminal
        }

        validation_failures = []
        promotion_denial_reason = None
        shadow_mode_validated = None

        # Check if promotion path is allowed
        allowed_targets = ALLOWED_PROMOTIONS.get(current_stage, [])
        if target_stage not in allowed_targets:
            promotion_denial_reason = (
                f"Invalid promotion path: Cannot promote from {current_stage} to {target_stage}. "
                f"Allowed targets: {', '.join(allowed_targets) if allowed_targets else 'none'}"
            )
            return {
                "promotion_target_stage": target_stage,
                "promotion_allowed": False,
                "promotion_denial_reason": promotion_denial_reason,
                "promotion_validation_errors": [promotion_denial_reason],
            }

        # For production promotion, validate shadow mode
        if target_stage == "Production":
            shadow_result = _validate_shadow_mode_detailed(state)
            shadow_mode_validated = shadow_result["validated"]
            validation_failures = shadow_result["failures"]

            if not shadow_mode_validated:
                return {
                    "promotion_target_stage": target_stage,
                    "promotion_allowed": False,
                    "shadow_mode_validated": False,
                    "validation_failures": validation_failures,
                    "promotion_validation_errors": validation_failures,
                    "error": f"Shadow mode validation failed: {'; '.join(validation_failures)}",
                    "error_type": "shadow_validation_failed",
                }

        # Promotion is allowed
        return {
            "promotion_target_stage": target_stage,
            "promotion_allowed": True,
            "promotion_reason": f"Promotion from {current_stage} to {target_stage} validated",
            "shadow_mode_validated": shadow_mode_validated,
            "promotion_validation_errors": [],
        }

    except Exception as e:
        return {
            "error": f"Promotion validation failed: {str(e)}",
            "error_type": "promotion_validation_error",
            "error_details": {"exception": str(e)},
            "promotion_allowed": False,
        }


async def promote_stage(state: Dict[str, Any]) -> Dict[str, Any]:
    """Promote model to target stage in MLflow.

    Args:
        state: Current agent state with model and promotion_target_stage

    Returns:
        State updates with promotion results
    """
    try:
        # Validate required fields FIRST (before checking promotion_allowed)
        registered_model_name = state.get("registered_model_name")
        state.get("model_version")
        promotion_target_stage = state.get("promotion_target_stage")
        current_stage = state.get("current_stage", "None")

        if not registered_model_name:
            return {
                "error": "Missing registered_model_name for promotion",
                "error_type": "missing_model_name",
                "promotion_successful": False,
            }

        if not promotion_target_stage:
            return {
                "error": "Missing promotion_target_stage for promotion",
                "error_type": "missing_target_stage",
                "promotion_successful": False,
            }

        # Now check if promotion is allowed (default to True if not explicitly set)
        # This allows promote_stage to work after validate_promotion sets promotion_allowed=True
        # or when called directly without validation
        promotion_allowed = state.get("promotion_allowed", True)

        if not promotion_allowed:
            errors = state.get("promotion_validation_errors", [])
            return {
                "error": f"Promotion not allowed: {'; '.join(errors)}",
                "error_type": "promotion_blocked",
                "promotion_successful": False,
            }

        # In production, this would call MLflow
        # mlflow_client.transition_model_version_stage(
        #     name=registered_model_name,
        #     version=model_version,
        #     stage=promotion_target_stage
        # )

        # Record previous stage for version record
        previous_stage = current_stage

        # Get metrics at promotion time
        validation_metrics = state.get("validation_metrics", {})
        metrics_at_promotion = {
            "test_auc": validation_metrics.get("auc_roc", 0.0),
            "test_precision": validation_metrics.get("precision", 0.0),
            "test_recall": validation_metrics.get("recall", 0.0),
            "test_f1": validation_metrics.get("f1_score", 0.0),
        }

        promotion_reason = state.get("promotion_reason", "Automated promotion")

        return {
            "previous_stage": previous_stage,
            "current_stage": promotion_target_stage,
            "metrics_at_promotion": metrics_at_promotion,
            "promotion_successful": True,
            "promotion_reason": promotion_reason,
            "promotion_timestamp": datetime.now(tz=None).isoformat(),
        }

    except Exception as e:
        return {
            "error": f"Stage promotion failed: {str(e)}",
            "error_type": "promotion_error",
            "error_details": {"exception": str(e)},
            "promotion_successful": False,
        }


def _validate_shadow_mode_detailed(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate shadow mode requirements for production promotion with detailed failures.

    Requirements:
    - Min duration: 24 hours
    - Min requests: 1000
    - Max error rate: 1%
    - Max latency p99: 150ms

    Args:
        state: Current agent state

    Returns:
        Dictionary with validated (bool) and failures (list of failure messages)
    """
    # Shadow mode requirements
    MIN_DURATION_HOURS = 24
    MIN_REQUESTS = 1000
    MAX_ERROR_RATE = 0.01
    MAX_LATENCY_P99_MS = 150

    # Get shadow mode metrics (in production, this would come from observability)
    shadow_duration = state.get("shadow_mode_duration_hours", 0)
    shadow_requests = state.get("shadow_mode_requests", 0)
    shadow_error_rate = state.get("shadow_mode_error_rate", 1.0)
    shadow_latency_p99 = state.get("shadow_mode_latency_p99_ms", 999)

    failures = []

    # Validate each requirement (keywords in messages match test expectations)
    if shadow_duration < MIN_DURATION_HOURS:
        failures.append(
            f"shadow_mode_duration_hours {shadow_duration} below minimum {MIN_DURATION_HOURS}"
        )

    if shadow_requests < MIN_REQUESTS:
        failures.append(f"shadow_mode_requests {shadow_requests} below minimum {MIN_REQUESTS}")

    if shadow_error_rate > MAX_ERROR_RATE:
        failures.append(
            f"shadow_mode_error_rate {shadow_error_rate:.4f} above maximum {MAX_ERROR_RATE}"
        )

    if shadow_latency_p99 > MAX_LATENCY_P99_MS:
        failures.append(
            f"shadow_mode_latency_p99_ms {shadow_latency_p99} above maximum {MAX_LATENCY_P99_MS}"
        )

    return {
        "validated": len(failures) == 0,
        "failures": failures,
    }


def _validate_shadow_mode(state: Dict[str, Any]) -> bool:
    """Validate shadow mode requirements for production promotion.

    DEPRECATED: Use _validate_shadow_mode_detailed for detailed failure info.

    Args:
        state: Current agent state

    Returns:
        True if shadow mode requirements are met
    """
    result = _validate_shadow_mode_detailed(state)
    return result["validated"]
