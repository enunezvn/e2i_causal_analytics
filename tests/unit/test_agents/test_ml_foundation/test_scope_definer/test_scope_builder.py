"""Tests for scope specification builder."""

from datetime import datetime

import pandas as pd
import pytest

from src.agents.ml_foundation.scope_definer.nodes.scope_builder import (
    _calculate_minimum_samples,
    _define_excluded_features,
    _define_inclusion_criteria,
    _define_target_population,
    _normalise_prediction_timestamp,
    _validate_cost_matrix,
    build_scope_spec,
)


@pytest.mark.asyncio
async def test_build_scope_spec_creates_complete_spec():
    """Test that build_scope_spec creates a complete ScopeSpec."""
    state = {
        "business_objective": "Increase prescriptions",
        "target_outcome": "Predict HCP prescribing",
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "prediction_horizon_days": 30,
        "brand": "Remibrutinib",
        "region": "US",
        "use_case": "hcp_targeting",
    }

    result = await build_scope_spec(state)

    # Check required output fields
    assert "experiment_id" in result
    assert "experiment_name" in result
    assert "scope_spec" in result

    scope_spec = result["scope_spec"]

    # Check required ScopeSpec fields
    assert scope_spec["experiment_id"] == result["experiment_id"]
    assert scope_spec["problem_type"] == "binary_classification"
    assert scope_spec["prediction_target"] == "will_prescribe"
    assert scope_spec["prediction_horizon_days"] == 30
    assert "target_population" in scope_spec
    assert "inclusion_criteria" in scope_spec
    assert "exclusion_criteria" in scope_spec
    assert "required_features" in scope_spec
    assert "excluded_features" in scope_spec
    assert "feature_categories" in scope_spec
    assert "regulatory_constraints" in scope_spec
    assert "ethical_constraints" in scope_spec
    assert "technical_constraints" in scope_spec
    assert "minimum_samples" in scope_spec
    assert scope_spec["brand"] == "Remibrutinib"
    assert scope_spec["region"] == "US"
    assert scope_spec["created_by"] == "scope_definer"


@pytest.mark.asyncio
async def test_experiment_id_format():
    """Test that experiment_id follows expected format."""
    state = {
        "brand": "Remibrutinib",
        "region": "US",
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "target_outcome": "Test",
    }

    result = await build_scope_spec(state)

    experiment_id = result["experiment_id"]

    # Should start with "exp_"
    assert experiment_id.startswith("exp_")

    # Should contain brand code
    assert "remi" in experiment_id.lower()

    # Should contain region code
    assert "us" in experiment_id.lower()

    # Should contain timestamp (numeric) and UUID suffix
    parts = experiment_id.split("_")
    # Format: exp_{brand}_{region}_{timestamp}_{uuid}
    assert parts[-2].isdigit()  # Timestamp part
    assert len(parts[-1]) == 6  # UUID suffix (6 hex chars)


@pytest.mark.asyncio
async def test_experiment_name_includes_brand_and_outcome():
    """Test that experiment_name is human-readable."""
    state = {
        "brand": "Kisqali",
        "target_outcome": "Increase prescriptions",
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
    }

    result = await build_scope_spec(state)

    experiment_name = result["experiment_name"]

    assert "Kisqali" in experiment_name
    assert "Increase prescriptions" in experiment_name


def test_define_target_population_remibrutinib():
    """Test target population for Remibrutinib brand."""
    state = {"brand": "Remibrutinib"}

    population = _define_target_population(state)

    assert "CSU" in population or "Chronic Spontaneous Urticaria" in population


def test_define_target_population_fabhalta():
    """Test target population for Fabhalta brand."""
    state = {"brand": "Fabhalta"}

    population = _define_target_population(state)

    assert "PNH" in population or "Paroxysmal Nocturnal Hemoglobinuria" in population


def test_define_target_population_kisqali():
    """Test target population for Kisqali brand."""
    state = {"brand": "Kisqali"}

    population = _define_target_population(state)

    assert "breast cancer" in population.lower()
    assert "HR+" in population or "HER2-" in population


def test_define_target_population_generic():
    """Test target population for unknown brand."""
    state = {"brand": "UnknownBrand"}

    population = _define_target_population(state)

    # Should return generic population
    assert "HCP" in population


def test_define_inclusion_criteria_has_base_criteria():
    """Test that inclusion criteria always include base requirements."""
    state = {"brand": "test"}

    criteria = _define_inclusion_criteria(state)

    # Should always include base criteria
    assert "hcp_is_active" in criteria
    assert "has_patient_data" in criteria
    assert any("activity" in c.lower() for c in criteria)


def test_define_inclusion_criteria_brand_specific():
    """Test brand-specific inclusion criteria."""
    # Remibrutinib
    state_remi = {"brand": "Remibrutinib"}
    criteria_remi = _define_inclusion_criteria(state_remi)
    assert any("dermatology" in c.lower() or "allergy" in c.lower() for c in criteria_remi)

    # Fabhalta
    state_fab = {"brand": "Fabhalta"}
    criteria_fab = _define_inclusion_criteria(state_fab)
    assert any("hematology" in c.lower() for c in criteria_fab)

    # Kisqali
    state_kis = {"brand": "Kisqali"}
    criteria_kis = _define_inclusion_criteria(state_kis)
    assert any("oncology" in c.lower() for c in criteria_kis)


def test_define_excluded_features_prevents_pii():
    """Test that excluded features list prevents PII leakage."""
    state = {}

    excluded = _define_excluded_features(state)

    # Should exclude common PII fields
    pii_keywords = ["name", "npi", "ssn", "address", "phone", "email"]
    for keyword in pii_keywords:
        assert any(keyword in feat.lower() for feat in excluded)


def test_define_excluded_features_prevents_temporal_leakage():
    """Test that excluded features prevent temporal leakage."""
    state = {}

    excluded = _define_excluded_features(state)

    # Should exclude future data
    assert any("future" in feat.lower() for feat in excluded)


def test_calculate_minimum_samples_binary_classification():
    """Test minimum samples for binary classification."""
    min_samples = _calculate_minimum_samples("binary_classification")

    # Should require at least 500 samples for balanced classes
    assert min_samples >= 500


def test_calculate_minimum_samples_regression():
    """Test minimum samples for regression."""
    min_samples = _calculate_minimum_samples("regression")

    # Should require at least 300 samples
    assert min_samples >= 300


def test_calculate_minimum_samples_causal():
    """Test minimum samples for causal inference."""
    min_samples = _calculate_minimum_samples("causal_inference")

    # Should require more samples for treatment/control groups
    assert min_samples >= 1000


@pytest.mark.asyncio
async def test_build_scope_includes_regulatory_constraints():
    """Test that scope includes required regulatory constraints."""
    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "target_outcome": "Test",
        "brand": "Test",
    }

    result = await build_scope_spec(state)

    regulatory = result["scope_spec"]["regulatory_constraints"]

    # Should include HIPAA and GDPR
    assert "HIPAA" in regulatory
    assert "GDPR" in regulatory


@pytest.mark.asyncio
async def test_build_scope_includes_ethical_constraints():
    """Test that scope includes ethical constraints."""
    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "target_outcome": "Test",
        "brand": "Test",
    }

    result = await build_scope_spec(state)

    ethical = result["scope_spec"]["ethical_constraints"]

    # Should exclude protected attributes
    assert any("protected" in c.lower() or "race" in c.lower() for c in ethical)
    assert any("pii" in c.lower() for c in ethical)


@pytest.mark.asyncio
async def test_build_scope_uses_candidate_features_if_provided():
    """Test that provided candidate_features override defaults."""
    custom_features = ["custom_feature1", "custom_feature2"]

    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "target_outcome": "Test",
        "brand": "Test",
        "candidate_features": custom_features,
    }

    result = await build_scope_spec(state)

    required_features = result["scope_spec"]["required_features"]

    # Should use custom features
    assert required_features == custom_features


# === Block 1B: prediction_timestamp scaffolding =============================


def test_normalise_prediction_timestamp_handles_datetime():
    """``datetime`` inputs are normalised to ISO 8601 strings."""
    dt = datetime(2026, 4, 26, 12, 30, 45)
    assert _normalise_prediction_timestamp(dt) == dt.isoformat()


def test_normalise_prediction_timestamp_handles_pandas_timestamp():
    """``pd.Timestamp`` inputs are normalised to ISO 8601 strings."""
    ts = pd.Timestamp("2026-04-26T12:30:45")
    assert _normalise_prediction_timestamp(ts) == ts.isoformat()


def test_normalise_prediction_timestamp_passes_through_strings():
    """ISO-format string inputs are preserved verbatim."""
    s = "2026-04-26T12:30:45"
    assert _normalise_prediction_timestamp(s) == s


@pytest.mark.parametrize("value", [None, ""])
def test_normalise_prediction_timestamp_returns_none_for_empty(value):
    """Missing / empty inputs map to ``None`` (not "" or current time)."""
    assert _normalise_prediction_timestamp(value) is None


def test_normalise_prediction_timestamp_falls_back_to_str_for_unknown_types():
    """Unknown types route through ``str(value)``.

    The helper is permissive on purpose — anything not recognised as
    ``datetime`` / ``pd.Timestamp`` / ``str`` / empty is round-tripped as
    its string representation. Locking this in so future input shapes
    (numpy datetimes, custom datetime-like objects) don't regress to
    silently dropping the value.
    """

    class _StringableTimestamp:
        def __str__(self) -> str:
            return "2026-04-26T00:00:00+00:00"

    assert (
        _normalise_prediction_timestamp(_StringableTimestamp())
        == "2026-04-26T00:00:00+00:00"
    )
    # Numeric inputs are an obvious abuse — but the contract says coerce,
    # not raise. Document the permissiveness so callers know to validate
    # upstream when they care.
    assert _normalise_prediction_timestamp(1714089600) == "1714089600"


@pytest.mark.asyncio
async def test_build_scope_propagates_prediction_timestamp_when_provided():
    """``prediction_timestamp`` from state lands on ``scope_spec``."""
    ts = pd.Timestamp("2026-04-26T00:00:00")
    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "target_outcome": "Test",
        "brand": "Test",
        "prediction_timestamp": ts,
    }

    result = await build_scope_spec(state)
    scope_spec = result["scope_spec"]

    assert "prediction_timestamp" in scope_spec
    assert scope_spec["prediction_timestamp"] == ts.isoformat()


@pytest.mark.asyncio
async def test_build_scope_prediction_timestamp_absent_when_unset():
    """Without ``prediction_timestamp`` in state, ``scope_spec`` has None."""
    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "target_outcome": "Test",
        "brand": "Test",
    }

    result = await build_scope_spec(state)
    scope_spec = result["scope_spec"]

    # Block 1B threading rule: the field is always present in the spec for a
    # stable schema, but its value is None when no timestamp was provided.
    assert "prediction_timestamp" in scope_spec
    assert scope_spec["prediction_timestamp"] is None


# === Block 5 (#10): cost_matrix scaffolding ================================


def test_validate_cost_matrix_accepts_full_dict():
    """All four required keys with numeric values → coerced to float dict."""
    cm = {"tp": 100, "fp": -10.5, "fn": -50, "tn": 0}
    result = _validate_cost_matrix(cm)
    assert result == {"tp": 100.0, "fp": -10.5, "fn": -50.0, "tn": 0.0}
    assert all(isinstance(v, float) for v in result.values())


@pytest.mark.parametrize("value", [None, {}])
def test_validate_cost_matrix_returns_none_for_empty(value):
    """None and {} both signal 'no cost matrix configured'."""
    assert _validate_cost_matrix(value) is None


def test_validate_cost_matrix_rejects_missing_keys():
    """A partial cost matrix is misconfiguration — fail loud at the boundary."""
    with pytest.raises(ValueError, match="missing required keys"):
        _validate_cost_matrix({"tp": 1.0, "fp": 0.0})


def test_validate_cost_matrix_rejects_non_numeric_values():
    """Strings/None/bools as values are rejected before the evaluator multiplies."""
    with pytest.raises(ValueError, match="must be int or float"):
        _validate_cost_matrix({"tp": "100", "fp": -10.0, "fn": -50.0, "tn": 0.0})


def test_validate_cost_matrix_rejects_non_dict():
    """The matrix must be a dict — lists/tuples are rejected."""
    with pytest.raises(ValueError, match="must be a dict"):
        _validate_cost_matrix([100, -10, -50, 0])


@pytest.mark.asyncio
async def test_build_scope_propagates_cost_matrix_when_provided():
    """A valid cost_matrix on state is forwarded onto scope_spec verbatim."""
    cm = {"tp": 100.0, "fp": -10.0, "fn": -50.0, "tn": 0.0}
    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "target_outcome": "Test",
        "brand": "Test",
        "cost_matrix": cm,
    }

    result = await build_scope_spec(state)
    scope_spec = result["scope_spec"]

    assert "cost_matrix" in scope_spec
    assert scope_spec["cost_matrix"] == cm


@pytest.mark.asyncio
async def test_build_scope_cost_matrix_absent_when_unset():
    """Without ``cost_matrix`` in state, ``scope_spec`` has None.

    The field is always present so the schema is stable; downstream code
    uses ``None`` as the "skip business_utility" signal.
    """
    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "target_outcome": "Test",
        "brand": "Test",
    }

    result = await build_scope_spec(state)
    scope_spec = result["scope_spec"]
    assert "cost_matrix" in scope_spec
    assert scope_spec["cost_matrix"] is None
