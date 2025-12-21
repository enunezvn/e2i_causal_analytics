"""Problem type classification logic for scope_definer.

This module infers ML problem type from business objectives.
"""

import re
from typing import Any, Dict, Literal


async def classify_problem(state: Dict[str, Any]) -> Dict[str, Any]:
    """Classify ML problem type from business objective.

    Infers problem type, target variable, and prediction horizon based on
    business objective and target outcome description.

    Args:
        state: ScopeDefinerState with problem_description, business_objective,
               target_outcome fields

    Returns:
        Dictionary with inferred_problem_type, inferred_target_variable,
        prediction_horizon_days
    """
    business_objective = state.get("business_objective", "")
    target_outcome = state.get("target_outcome", "")
    problem_type_hint = state.get("problem_type_hint")

    # If hint provided, trust it
    if problem_type_hint:
        inferred_type = problem_type_hint
    else:
        # Classify from keywords
        inferred_type = _infer_problem_type(business_objective, target_outcome)

    # Infer target variable name
    inferred_target = _infer_target_variable(target_outcome, inferred_type)

    # Infer prediction horizon
    prediction_horizon = _infer_prediction_horizon(target_outcome)

    return {
        "inferred_problem_type": inferred_type,
        "inferred_target_variable": inferred_target,
        "prediction_horizon_days": prediction_horizon,
    }


def _infer_problem_type(business_objective: str, target_outcome: str) -> Literal[
    "binary_classification",
    "multiclass_classification",
    "regression",
    "causal_inference",
    "time_series",
]:
    """Infer problem type from objective keywords."""
    combined = f"{business_objective} {target_outcome}".lower()

    # Regression indicators
    regression_keywords = [
        "volume",
        "count",
        "number of",
        "quantity",
        "amount",
        "increase by",
        "reduce by",
        "time to",
        "duration",
        "prescription volume",
        "trx",
        "nrx",
    ]

    # Binary classification indicators
    binary_keywords = [
        "will prescribe",
        "will churn",
        "will convert",
        "will abandon",
        "yes/no",
        "true/false",
        "prescriber or not",
        "adoption",
        "will adopt",
        "likely to",
        "predict whether",
    ]

    # Causal inference indicators
    causal_keywords = [
        "impact of",
        "effect of",
        "caused by",
        "due to",
        "influence of",
        "causal",
        "attribution",
        "uplift",
        "incremental",
        "counterfactual",
    ]

    # Time series indicators
    timeseries_keywords = [
        "forecast",
        "predict future",
        "trend",
        "seasonal",
        "over time",
        "time series",
        "next month",
        "next quarter",
    ]

    # Check in order of specificity
    if any(kw in combined for kw in causal_keywords):
        return "causal_inference"

    # Check for regression indicators with specific volume/count targets
    # These take precedence even if time_series keywords are present
    has_regression = any(kw in combined for kw in regression_keywords)
    has_timeseries = any(kw in combined for kw in timeseries_keywords)

    if has_regression and has_timeseries:
        # When both match, check if it's a volume/count prediction
        # Volume/count prediction is regression, not time_series
        volume_indicators = ["count", "volume", "trx", "nrx", "quantity", "amount"]
        if any(vi in combined for vi in volume_indicators):
            return "regression"

    if has_timeseries:
        return "time_series"
    if any(kw in combined for kw in binary_keywords):
        return "binary_classification"
    if has_regression:
        return "regression"

    # Default to binary classification (most common in pharmaceutical targeting)
    return "binary_classification"


def _infer_target_variable(target_outcome: str, problem_type: str) -> str:
    """Infer target variable name from outcome description."""
    target_lower = target_outcome.lower()

    # Common patterns
    if "prescribe" in target_lower or "prescription" in target_lower:
        if problem_type == "regression":
            return "prescription_volume"
        else:
            return "will_prescribe"

    if "churn" in target_lower:
        return "will_churn"

    if "convert" in target_lower or "conversion" in target_lower:
        return "will_convert"

    if "adopt" in target_lower or "adoption" in target_lower:
        return "will_adopt"

    if "abandon" in target_lower:
        return "will_abandon"

    if "trx" in target_lower or "nrx" in target_lower:
        return "prescription_count"

    if "time to" in target_lower:
        return "time_to_event_days"

    # Default: sanitize outcome string to variable name
    sanitized = re.sub(r"[^a-z0-9]+", "_", target_lower)
    sanitized = sanitized.strip("_")
    return sanitized or "target_outcome"


def _infer_prediction_horizon(target_outcome: str) -> int:
    """Infer prediction horizon in days from outcome description."""
    target_lower = target_outcome.lower()

    # Check for explicit time periods
    if "90 day" in target_lower or "3 month" in target_lower:
        return 90
    if "60 day" in target_lower or "2 month" in target_lower:
        return 60
    if "30 day" in target_lower or "1 month" in target_lower or "next month" in target_lower:
        return 30
    if "7 day" in target_lower or "week" in target_lower:
        return 7
    if "quarter" in target_lower:
        return 90
    if "year" in target_lower:
        return 365

    # Default to 30 days (industry standard for pharmaceutical targeting)
    return 30
