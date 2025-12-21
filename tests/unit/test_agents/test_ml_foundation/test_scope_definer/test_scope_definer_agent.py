"""Integration tests for ScopeDefinerAgent."""

import pytest

from src.agents.ml_foundation.scope_definer import ScopeDefinerAgent


@pytest.mark.asyncio
async def test_scope_definer_agent_initialization():
    """Test that ScopeDefinerAgent initializes correctly."""
    agent = ScopeDefinerAgent()

    assert agent.tier == 0
    assert agent.tier_name == "ml_foundation"
    assert agent.agent_type == "standard"
    assert agent.sla_seconds == 5
    assert agent.graph is not None


@pytest.mark.asyncio
async def test_scope_definer_missing_required_fields():
    """Test that agent raises error when required fields are missing."""
    agent = ScopeDefinerAgent()

    # Missing problem_description
    with pytest.raises(ValueError, match="problem_description"):
        await agent.run(
            {
                "business_objective": "Test",
                "target_outcome": "Test",
            }
        )

    # Missing business_objective
    with pytest.raises(ValueError, match="business_objective"):
        await agent.run(
            {
                "problem_description": "Test",
                "target_outcome": "Test",
            }
        )

    # Missing target_outcome
    with pytest.raises(ValueError, match="target_outcome"):
        await agent.run(
            {
                "problem_description": "Test",
                "business_objective": "Test",
            }
        )


@pytest.mark.asyncio
async def test_scope_definer_full_pipeline_binary_classification():
    """Test full pipeline for binary classification problem."""
    agent = ScopeDefinerAgent()

    input_data = {
        "problem_description": "We want to identify HCPs who are likely to prescribe Kisqali",
        "business_objective": "Increase Kisqali market share in oncology",
        "target_outcome": "Predict which HCPs will prescribe in next 30 days",
        "brand": "Kisqali",
        "region": "US",
    }

    output = await agent.run(input_data)

    # Check output structure
    assert "scope_spec" in output
    assert "success_criteria" in output
    assert "experiment_id" in output
    assert "experiment_name" in output
    assert "validation_passed" in output

    # Check scope_spec
    scope_spec = output["scope_spec"]
    assert scope_spec["problem_type"] == "binary_classification"
    assert scope_spec["prediction_target"] == "will_prescribe"
    assert scope_spec["prediction_horizon_days"] == 30
    assert scope_spec["brand"] == "Kisqali"
    assert scope_spec["region"] == "US"
    assert len(scope_spec["required_features"]) > 0
    assert len(scope_spec["excluded_features"]) > 0

    # Check success_criteria
    success_criteria = output["success_criteria"]
    assert success_criteria["experiment_id"] == output["experiment_id"]
    assert success_criteria["minimum_auc"] is not None
    assert success_criteria["minimum_precision"] is not None
    assert success_criteria["baseline_model"] == "random_forest_baseline"


@pytest.mark.asyncio
async def test_scope_definer_full_pipeline_regression():
    """Test full pipeline for regression problem."""
    agent = ScopeDefinerAgent()

    input_data = {
        "problem_description": "Forecast prescription volume per HCP",
        "business_objective": "Optimize resource allocation based on predicted volume",
        "target_outcome": "Predict monthly TRx count for each HCP",
        "brand": "Fabhalta",
        "region": "EU",
    }

    output = await agent.run(input_data)

    # Check problem type
    scope_spec = output["scope_spec"]
    assert scope_spec["problem_type"] == "regression"
    assert (
        "prescription" in scope_spec["prediction_target"].lower()
        or "trx" in scope_spec["prediction_target"].lower()
    )

    # Check success criteria
    success_criteria = output["success_criteria"]
    assert success_criteria["minimum_rmse"] is not None
    assert success_criteria["minimum_r2"] is not None
    assert success_criteria["baseline_model"] == "linear_regression_baseline"


@pytest.mark.asyncio
async def test_scope_definer_full_pipeline_causal():
    """Test full pipeline for causal inference problem."""
    agent = ScopeDefinerAgent()

    input_data = {
        "problem_description": "Measure effectiveness of email campaigns",
        "business_objective": "Optimize marketing spend allocation",
        "target_outcome": "Determine causal impact of email campaigns on prescriptions",
        "brand": "Remibrutinib",
    }

    output = await agent.run(input_data)

    # Check problem type
    scope_spec = output["scope_spec"]
    assert scope_spec["problem_type"] == "causal_inference"

    # Check success criteria
    success_criteria = output["success_criteria"]
    assert success_criteria["baseline_model"] == "ols_baseline"


@pytest.mark.asyncio
async def test_scope_definer_respects_problem_type_hint():
    """Test that problem_type_hint overrides automatic classification."""
    agent = ScopeDefinerAgent()

    input_data = {
        "problem_description": "Predict prescriptions",
        "business_objective": "Increase prescriptions",
        "target_outcome": "Predict prescriber behavior",
        "problem_type_hint": "regression",  # Override
        "brand": "Kisqali",
    }

    output = await agent.run(input_data)

    # Should use hint
    scope_spec = output["scope_spec"]
    assert scope_spec["problem_type"] == "regression"


@pytest.mark.asyncio
async def test_scope_definer_uses_custom_performance_requirements():
    """Test that custom performance requirements are applied."""
    agent = ScopeDefinerAgent()

    custom_reqs = {
        "min_auc": 0.85,
        "min_precision": 0.80,
        "min_recall": 0.75,
        "min_f1": 0.78,
        "min_lift": 0.15,
    }

    input_data = {
        "problem_description": "Predict HCP behavior",
        "business_objective": "Increase prescriptions",
        "target_outcome": "Predict prescribing",
        "performance_requirements": custom_reqs,
        "brand": "Fabhalta",
    }

    output = await agent.run(input_data)

    success_criteria = output["success_criteria"]
    assert success_criteria["minimum_auc"] == 0.85
    assert success_criteria["minimum_precision"] == 0.80
    assert success_criteria["minimum_recall"] == 0.75
    assert success_criteria["minimum_f1"] == 0.78
    assert success_criteria["minimum_lift_over_baseline"] == 0.15


@pytest.mark.asyncio
async def test_scope_definer_uses_candidate_features():
    """Test that provided candidate_features are used."""
    agent = ScopeDefinerAgent()

    custom_features = ["hcp_specialty", "patient_count", "brand_affinity"]

    input_data = {
        "problem_description": "Predict HCP behavior",
        "business_objective": "Increase prescriptions",
        "target_outcome": "Predict prescribing",
        "candidate_features": custom_features,
        "brand": "Kisqali",
    }

    output = await agent.run(input_data)

    scope_spec = output["scope_spec"]
    assert scope_spec["required_features"] == custom_features


@pytest.mark.asyncio
async def test_scope_definer_includes_regulatory_constraints():
    """Test that regulatory constraints are included."""
    agent = ScopeDefinerAgent()

    input_data = {
        "problem_description": "Predict HCP behavior",
        "business_objective": "Increase prescriptions",
        "target_outcome": "Predict prescribing",
        "brand": "Kisqali",
    }

    output = await agent.run(input_data)

    scope_spec = output["scope_spec"]
    regulatory = scope_spec["regulatory_constraints"]

    assert "HIPAA" in regulatory
    assert "GDPR" in regulatory


@pytest.mark.asyncio
async def test_scope_definer_includes_ethical_constraints():
    """Test that ethical constraints prevent protected attributes."""
    agent = ScopeDefinerAgent()

    input_data = {
        "problem_description": "Predict HCP behavior",
        "business_objective": "Increase prescriptions",
        "target_outcome": "Predict prescribing",
        "brand": "Kisqali",
    }

    output = await agent.run(input_data)

    scope_spec = output["scope_spec"]
    ethical = scope_spec["ethical_constraints"]

    # Should exclude protected attributes
    assert any("protected" in c.lower() or "race" in c.lower() for c in ethical)


@pytest.mark.asyncio
async def test_scope_definer_excludes_pii_features():
    """Test that PII features are excluded."""
    agent = ScopeDefinerAgent()

    input_data = {
        "problem_description": "Predict HCP behavior",
        "business_objective": "Increase prescriptions",
        "target_outcome": "Predict prescribing",
        "brand": "Kisqali",
    }

    output = await agent.run(input_data)

    scope_spec = output["scope_spec"]
    excluded = scope_spec["excluded_features"]

    # Should exclude PII
    pii_keywords = ["name", "npi", "ssn", "address", "phone", "email"]
    for keyword in pii_keywords:
        assert any(keyword in feat.lower() for feat in excluded)


@pytest.mark.asyncio
async def test_scope_definer_validation_warnings():
    """Test that validation warnings are populated."""
    agent = ScopeDefinerAgent()

    input_data = {
        "problem_description": "Predict HCP behavior",
        "business_objective": "Increase prescriptions",
        "target_outcome": "Predict prescribing",
        "performance_requirements": {"min_auc": 0.98},  # Very high - should warn
        "brand": "Kisqali",
    }

    output = await agent.run(input_data)

    # Should have warnings
    assert len(output["validation_warnings"]) > 0


@pytest.mark.asyncio
async def test_scope_definer_validation_passes():
    """Test that validation passes with reasonable inputs."""
    agent = ScopeDefinerAgent()

    input_data = {
        "problem_description": "Predict HCP behavior",
        "business_objective": "Increase prescriptions",
        "target_outcome": "Predict prescribing in next 30 days",
        "brand": "Kisqali",
        "region": "US",
    }

    output = await agent.run(input_data)

    # Should pass validation
    assert output["validation_passed"] is True
    assert len(output["validation_errors"]) == 0


@pytest.mark.asyncio
async def test_scope_definer_experiment_id_is_unique():
    """Test that each run generates a unique experiment_id."""
    agent = ScopeDefinerAgent()

    input_data = {
        "problem_description": "Predict HCP behavior",
        "business_objective": "Increase prescriptions",
        "target_outcome": "Predict prescribing",
        "brand": "Kisqali",
    }

    output1 = await agent.run(input_data)
    output2 = await agent.run(input_data)

    # Should have different experiment IDs
    assert output1["experiment_id"] != output2["experiment_id"]


@pytest.mark.asyncio
async def test_scope_definer_handles_minimal_input():
    """Test agent works with only required fields."""
    agent = ScopeDefinerAgent()

    input_data = {
        "problem_description": "Simple problem",
        "business_objective": "Simple objective",
        "target_outcome": "Simple outcome",
    }

    output = await agent.run(input_data)

    # Should still produce valid output
    assert "scope_spec" in output
    assert "success_criteria" in output
    assert output["validation_passed"] is True
