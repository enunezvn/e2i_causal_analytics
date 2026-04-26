"""Scope specification builder for scope_definer.

This module builds the complete ScopeSpec from business requirements.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, cast


def _normalise_prediction_timestamp(value: Any) -> Optional[str]:
    """Coerce a prediction_timestamp input to an ISO 8601 string.

    Accepts ``datetime``, ``pd.Timestamp``, or ISO-format ``str`` and returns
    the normalised string form for stable storage in ``scope_spec``. Returns
    ``None`` when the input is missing or empty so downstream agents can
    distinguish "not provided" from "explicit timestamp".
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    # Late import to keep scope_builder dependency-light.
    try:
        import pandas as pd

        if isinstance(value, pd.Timestamp):
            # pd.Timestamp.isoformat is typed as Any in pandas-stubs, so cast.
            return cast(str, value.isoformat())
    except ImportError:  # pragma: no cover - pandas is a hard dep elsewhere
        pass
    if isinstance(value, str):
        return value
    return str(value)


_COST_MATRIX_KEYS = ("tp", "fp", "fn", "tn")


def _validate_cost_matrix(value: Any) -> Optional[Dict[str, float]]:
    """Coerce and validate a cost_matrix input.

    Accepts a dict with all four confusion-matrix keys (``tp``/``fp``/
    ``fn``/``tn``) mapped to numeric values. Returns ``None`` when the
    input is missing or empty so downstream code can treat "no cost
    matrix configured" the same as "skip business_utility".

    Raises ``ValueError`` on malformed input (missing keys or non-numeric
    values) — fail-loud at the scope_definer boundary rather than
    silently dropping a misconfigured matrix that would defeat the
    purpose of the business_utility metric.
    """
    if value is None or value == {}:
        return None
    if not isinstance(value, dict):
        raise ValueError(
            f"cost_matrix must be a dict with keys {_COST_MATRIX_KEYS}, got "
            f"{type(value).__name__}"
        )
    missing = [k for k in _COST_MATRIX_KEYS if k not in value]
    if missing:
        raise ValueError(
            f"cost_matrix missing required keys: {missing}. All four of "
            f"{_COST_MATRIX_KEYS} are required."
        )
    coerced: Dict[str, float] = {}
    for k in _COST_MATRIX_KEYS:
        v = value[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(
                f"cost_matrix['{k}'] must be int or float, got {type(v).__name__}"
            )
        coerced[k] = float(v)
    return coerced


async def build_scope_spec(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build complete ScopeSpec from inferred problem details.

    Creates a complete ML experiment specification including:
    - Experiment identification
    - Problem type and target
    - Population criteria
    - Feature requirements
    - Constraints

    Args:
        state: ScopeDefinerState with inferred problem type, target variable,
               and business context

    Returns:
        Dictionary with scope_spec, experiment_id, experiment_name
    """
    # Generate unique experiment ID
    brand = state.get("brand", "unknown")
    region = state.get("region", "all")
    timestamp = datetime.now(tz=None).strftime("%Y%m%d%H%M%S")
    # Add UUID suffix for uniqueness even within same second
    uuid_suffix = uuid.uuid4().hex[:6]
    experiment_id = f"exp_{brand.lower()[:4]}_{region.lower()[:2]}_{timestamp}_{uuid_suffix}"

    # Generate experiment name
    target_outcome = state.get("target_outcome", "ML Model")
    experiment_name = f"{brand} - {target_outcome}"

    # Get inferred problem details
    problem_type = state.get("inferred_problem_type", "binary_classification")
    prediction_target = state.get("inferred_target_variable", "target")
    prediction_horizon = state.get("prediction_horizon_days", 30)

    # Define population criteria
    target_population = _define_target_population(state)
    inclusion_criteria = _define_inclusion_criteria(state)
    exclusion_criteria = _define_exclusion_criteria(state)

    # Define feature requirements
    required_features = _define_required_features(state)
    excluded_features = _define_excluded_features(state)
    feature_categories = _define_feature_categories(state)

    # Define constraints
    regulatory_constraints = ["HIPAA", "GDPR"]
    ethical_constraints = ["no_protected_attributes", "no_race_features", "no_direct_pii"]
    technical_constraints = ["inference_latency_<100ms", "model_size_<1GB"]

    # Determine minimum samples
    minimum_samples = _calculate_minimum_samples(problem_type)

    # Block 1B: forward the optional inference cutoff timestamp from the
    # business spec so downstream agents (Tier 1-5) share a single anchor.
    prediction_timestamp = _normalise_prediction_timestamp(state.get("prediction_timestamp"))

    # Block 5 (finding #10): forward the optional cost matrix so the
    # evaluator can compute a business_utility metric driven by the chosen
    # decision threshold and the per-outcome dollar value the business
    # assigns to each confusion-matrix cell. None means "skip the metric".
    cost_matrix = _validate_cost_matrix(state.get("cost_matrix"))

    # Build complete ScopeSpec.
    # Naming guard (Block 1B-M6): the two ``prediction_*`` fields below are
    # NOT redundant — ``prediction_horizon_days`` is a *duration* (e.g. "30
    # days lookahead from the cutoff"), while ``prediction_timestamp`` is the
    # *anchor* (the inference cutoff itself, e.g. ``2026-04-26T00:00``).
    # Together they define a half-open prediction window
    # ``[prediction_timestamp, prediction_timestamp + horizon)``.
    scope_spec = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "problem_type": problem_type,
        "prediction_target": prediction_target,
        "prediction_horizon_days": prediction_horizon,
        "prediction_timestamp": prediction_timestamp,
        "cost_matrix": cost_matrix,
        "target_population": target_population,
        "inclusion_criteria": inclusion_criteria,
        "exclusion_criteria": exclusion_criteria,
        "required_features": required_features,
        "excluded_features": excluded_features,
        "feature_categories": feature_categories,
        "regulatory_constraints": regulatory_constraints,
        "ethical_constraints": ethical_constraints,
        "technical_constraints": technical_constraints,
        "minimum_samples": minimum_samples,
        "brand": brand,
        "region": region,
        "use_case": state.get("use_case", "commercial_targeting"),
        "created_by": "scope_definer",
        "created_at": datetime.now(tz=None).isoformat(),
    }

    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "scope_spec": scope_spec,
    }


def _define_target_population(state: Dict[str, Any]) -> str:
    """Define target population description."""
    brand = state.get("brand", "")
    state.get("inferred_problem_type", "")

    # Brand-specific populations
    if "remibrutinib" in brand.lower() or "csu" in brand.lower():
        return "HCPs treating Chronic Spontaneous Urticaria patients"
    elif "fabhalta" in brand.lower() or "pnh" in brand.lower():
        return "HCPs treating Paroxysmal Nocturnal Hemoglobinuria patients"
    elif "kisqali" in brand.lower() or "breast" in brand.lower():
        return "Oncologists treating HR+/HER2- breast cancer patients"

    # Generic
    return "HCPs with relevant patient population"


def _define_inclusion_criteria(state: Dict[str, Any]) -> List[str]:
    """Define data inclusion criteria."""
    brand = state.get("brand", "")

    criteria = ["hcp_is_active", "has_patient_data", "recent_activity_90days"]

    # Brand-specific criteria
    if "remibrutinib" in brand.lower():
        criteria.append("specialty_in_dermatology_or_allergy")
    elif "fabhalta" in brand.lower():
        criteria.append("specialty_in_hematology")
    elif "kisqali" in brand.lower():
        criteria.append("specialty_in_oncology")

    return criteria


def _define_exclusion_criteria(state: Dict[str, Any]) -> List[str]:
    """Define data exclusion criteria."""
    return ["test_accounts", "invalid_data", "duplicate_records", "missing_required_fields"]


def _define_required_features(state: Dict[str, Any]) -> List[str]:
    """Define required features based on problem type."""
    problem_type = state.get("inferred_problem_type", "")
    candidate_features = state.get("candidate_features", [])

    if candidate_features:
        return cast(List[str], candidate_features)

    # Default feature sets by problem type
    base_features = [
        "hcp_specialty",
        "patient_count",
        "prescription_history",
        "brand_affinity_score",
    ]

    if problem_type == "regression":
        base_features.extend(["historical_prescription_volume", "market_share"])
    elif problem_type == "binary_classification":
        base_features.extend(["engagement_score", "channel_response_rate"])

    return base_features


def _define_excluded_features(state: Dict[str, Any]) -> List[str]:
    """Define features to exclude (PII, leakage risks, pipeline metadata)."""
    return [
        # PII
        "hcp_name",
        "hcp_npi",
        "patient_name",
        "patient_ssn",
        "exact_address",
        "phone_number",
        "email_address",
        # Temporal leakage
        "future_prescription_data",
        # Pipeline construction metadata (never clinical predictors)
        "index_date",
        "lookback_start_date",
        "prediction_end_date",
        "cohort_entry_date",
        "observation_end_date",
    ]


def _define_feature_categories(state: Dict[str, Any]) -> List[str]:
    """Define feature categories for this problem."""
    return [
        "demographics",
        "prescription_history",
        "engagement",
        "market_dynamics",
        "brand_affinity",
    ]


def _calculate_minimum_samples(problem_type: str) -> int:
    """Calculate minimum required samples based on problem type."""
    # Rule of thumb: 10 samples per feature minimum
    # Adjust by problem complexity

    if problem_type == "binary_classification":
        return 500  # Need balanced classes
    elif problem_type == "regression":
        return 300
    elif problem_type == "multiclass_classification":
        return 1000  # Need samples per class
    elif problem_type == "causal_inference":
        return 1000  # Need treatment and control groups
    elif problem_type == "time_series":
        return 500

    return 500  # Default
