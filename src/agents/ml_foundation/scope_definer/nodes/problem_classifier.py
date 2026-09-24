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
               target_outcome fields. Optional ``problem_type_hint`` pins the
               problem type; optional ``target_variable_hint`` (alias:
               ``target_variable``) pins a physical target column and bypasses
               the name rewrite.

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

    # Same contract for the target variable (#2284). ``_infer_target_variable``
    # invents a canonical name from a natural-language objective ("likely to
    # adopt" -> ``will_adopt``), which is what this node is for and is still
    # test-pinned. But a caller that already knows the *physical column* — the
    # retrain path reads the registry's ``cohort_target_outcome``, the physical
    # label — had no way to say so, and its ``adopted`` came back out as
    # ``will_adopt``: a name no table defines, which the data_preparer's target
    # guard then refuses. A hint bypasses the rewrite wholesale, so it covers
    # every family (prescribe / churn / convert / adopt / abandon / trx|nrx /
    # time-to), not just the one that surfaced the bug.
    target_variable_hint = resolve_target_variable_hint(state)
    if target_variable_hint:
        inferred_target = target_variable_hint
    else:
        # Infer target variable name
        inferred_target = _infer_target_variable(target_outcome, inferred_type)

    # Infer prediction horizon
    prediction_horizon = _infer_prediction_horizon(target_outcome)

    return {
        "inferred_problem_type": inferred_type,
        "inferred_target_variable": inferred_target,
        "prediction_horizon_days": prediction_horizon,
    }


def resolve_target_variable_hint(state: Dict[str, Any]) -> str:
    """The physical target column a caller pinned, or ``""`` (#2284).

    Public because the pipeline's deployer leg must resolve the hint the SAME way
    this node does — it records the result as ``cohort_target_outcome``, which the
    drift sweep feeds back as the next retrain's target. Resolving it twice by hand
    let the alias and the whitespace rules drift apart (codex r3 HIGH).

    ``target_variable_hint`` is the name that mirrors ``problem_type_hint``.
    ``target_variable`` is accepted as its alias: it has been on the state and
    documented in ``ScopeDefinerAgent.run`` as "Target variable name if known"
    since the initial commit — ``scripts/sample_ml_pipeline.py`` pairs a
    physical ``target_variable`` with a descriptive ``target_outcome`` — but no
    node ever read it. Honouring it here resolves the omission rather than
    leaving a decoy beside a near-identical live field.

    Returned verbatim apart from surrounding whitespace: a physical column name
    is case- and punctuation-sensitive, so sanitising it would recreate the very
    bug this closes. Blank / absent -> ``""`` (fall back to inference; never
    blank the target).
    """
    for key in ("target_variable_hint", "target_variable"):
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _infer_problem_type(
    business_objective: str, target_outcome: str
) -> Literal[
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
