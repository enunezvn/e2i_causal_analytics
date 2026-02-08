"""Scope specification builder for scope_definer.

This module builds the complete ScopeSpec from business requirements.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, cast


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

    # Build complete ScopeSpec
    scope_spec = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "problem_type": problem_type,
        "prediction_target": prediction_target,
        "prediction_horizon_days": prediction_horizon,
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
    """Define features to exclude (PII, leakage risks)."""
    return [
        "hcp_name",
        "hcp_npi",
        "patient_name",
        "patient_ssn",
        "exact_address",
        "phone_number",
        "email_address",
        "future_prescription_data",  # Temporal leakage
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
