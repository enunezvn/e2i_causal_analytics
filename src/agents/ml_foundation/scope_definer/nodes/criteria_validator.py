"""Success criteria definition and validation for scope_definer.

This module defines measurable success criteria and validates constraints.
"""

from typing import Dict, Any, List, Optional


async def define_success_criteria(state: Dict[str, Any]) -> Dict[str, Any]:
    """Define success criteria based on problem type and requirements.

    Creates performance thresholds that model_trainer must meet to pass validation.

    Args:
        state: ScopeDefinerState with problem_type, performance_requirements

    Returns:
        Dictionary with success_criteria, validation_passed, validation_warnings,
        validation_errors
    """
    problem_type = state.get("inferred_problem_type", "binary_classification")
    performance_reqs = state.get("performance_requirements", {})

    # Define criteria based on problem type
    if problem_type in ["binary_classification", "multiclass_classification"]:
        success_criteria = _define_classification_criteria(performance_reqs)
    elif problem_type == "regression":
        success_criteria = _define_regression_criteria(performance_reqs)
    elif problem_type == "causal_inference":
        success_criteria = _define_causal_criteria(performance_reqs)
    elif problem_type == "time_series":
        success_criteria = _define_timeseries_criteria(performance_reqs)
    else:
        success_criteria = _define_classification_criteria(performance_reqs)

    # Add common criteria
    success_criteria["experiment_id"] = state.get("experiment_id", "")
    success_criteria["baseline_model"] = _define_baseline_model(problem_type)
    success_criteria["minimum_lift_over_baseline"] = performance_reqs.get(
        "min_lift", 0.10
    )  # 10% improvement over baseline

    # Validate criteria
    validation_result = _validate_criteria(success_criteria, state)

    return {
        "success_criteria": success_criteria,
        "validation_passed": validation_result["passed"],
        "validation_warnings": validation_result["warnings"],
        "validation_errors": validation_result["errors"],
    }


def _define_classification_criteria(
    performance_reqs: Dict[str, float]
) -> Dict[str, Any]:
    """Define success criteria for classification problems."""
    return {
        "minimum_auc": performance_reqs.get("min_auc", 0.75),
        "minimum_precision": performance_reqs.get("min_precision", 0.70),
        "minimum_recall": performance_reqs.get("min_recall", 0.65),
        "minimum_f1": performance_reqs.get("min_f1", 0.70),
        "minimum_rmse": None,  # Not applicable
        "minimum_r2": None,  # Not applicable
        "minimum_mape": None,  # Not applicable
    }


def _define_regression_criteria(
    performance_reqs: Dict[str, float]
) -> Dict[str, Any]:
    """Define success criteria for regression problems."""
    return {
        "minimum_auc": None,  # Not applicable
        "minimum_precision": None,  # Not applicable
        "minimum_recall": None,  # Not applicable
        "minimum_f1": None,  # Not applicable
        "minimum_rmse": performance_reqs.get("max_rmse", 10.0),  # Lower is better
        "minimum_r2": performance_reqs.get("min_r2", 0.60),
        "minimum_mape": performance_reqs.get("max_mape", 0.20),  # 20% error
    }


def _define_causal_criteria(
    performance_reqs: Dict[str, float]
) -> Dict[str, Any]:
    """Define success criteria for causal inference problems."""
    return {
        "minimum_auc": None,
        "minimum_precision": None,
        "minimum_recall": None,
        "minimum_f1": None,
        "minimum_rmse": performance_reqs.get("max_ate_se", 0.5),  # ATE std error
        "minimum_r2": performance_reqs.get("min_r2", 0.50),
        "minimum_mape": None,
    }


def _define_timeseries_criteria(
    performance_reqs: Dict[str, float]
) -> Dict[str, Any]:
    """Define success criteria for time series problems."""
    return {
        "minimum_auc": None,
        "minimum_precision": None,
        "minimum_recall": None,
        "minimum_f1": None,
        "minimum_rmse": performance_reqs.get("max_rmse", 15.0),
        "minimum_r2": performance_reqs.get("min_r2", 0.55),
        "minimum_mape": performance_reqs.get("max_mape", 0.25),
    }


def _define_baseline_model(problem_type: str) -> str:
    """Define baseline model to beat."""
    baselines = {
        "binary_classification": "random_forest_baseline",
        "multiclass_classification": "random_forest_baseline",
        "regression": "linear_regression_baseline",
        "causal_inference": "ols_baseline",
        "time_series": "arima_baseline",
    }
    return baselines.get(problem_type, "random_baseline")


def _validate_criteria(
    criteria: Dict[str, Any], state: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate success criteria for consistency and feasibility.

    Returns:
        Dictionary with 'passed' (bool), 'warnings' (List[str]), 'errors' (List[str])
    """
    warnings: List[str] = []
    errors: List[str] = []

    # Check if minimum samples requirement is feasible
    minimum_samples = state.get("scope_spec", {}).get("minimum_samples", 0)
    if minimum_samples < 100:
        warnings.append(
            f"Minimum sample size ({minimum_samples}) is very low. "
            "Consider requiring at least 500 samples for robust training."
        )

    # Check if performance thresholds are realistic
    min_auc = criteria.get("minimum_auc")
    if min_auc and min_auc > 0.95:
        warnings.append(
            f"Minimum AUC ({min_auc}) is very high. "
            "May be difficult to achieve in production."
        )

    min_r2 = criteria.get("minimum_r2")
    if min_r2 and min_r2 > 0.90:
        warnings.append(
            f"Minimum R² ({min_r2}) is very high. "
            "May be difficult to achieve in real-world data."
        )

    # Check for conflicting constraints
    time_budget = state.get("time_budget_hours")
    if time_budget and time_budget < 1.0:
        warnings.append(
            f"Time budget ({time_budget}h) is very low. "
            "May not be sufficient for proper hyperparameter tuning."
        )

    # Check for required fields
    if not criteria.get("experiment_id"):
        errors.append("experiment_id is required in success criteria")

    if not criteria.get("baseline_model"):
        errors.append("baseline_model must be specified")

    # Passed if no errors
    passed = len(errors) == 0

    return {
        "passed": passed,
        "warnings": warnings,
        "errors": errors,
    }
