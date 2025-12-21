"""Tests for scope specification builder."""

import pytest

from src.agents.ml_foundation.scope_definer.nodes.scope_builder import (
    _calculate_minimum_samples,
    _define_excluded_features,
    _define_inclusion_criteria,
    _define_target_population,
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
