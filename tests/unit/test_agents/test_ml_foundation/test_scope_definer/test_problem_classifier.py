"""Tests for problem classification logic."""

import pytest
from src.agents.ml_foundation.scope_definer.nodes.problem_classifier import (
    classify_problem,
    _infer_problem_type,
    _infer_target_variable,
    _infer_prediction_horizon,
)


@pytest.mark.asyncio
async def test_classify_binary_classification_from_keywords():
    """Test classification of binary classification problems."""
    state = {
        "business_objective": "Identify which HCPs will prescribe our drug",
        "target_outcome": "Predict whether HCP will prescribe in next 30 days",
    }

    result = await classify_problem(state)

    assert result["inferred_problem_type"] == "binary_classification"
    assert result["inferred_target_variable"] == "will_prescribe"
    assert result["prediction_horizon_days"] == 30


@pytest.mark.asyncio
async def test_classify_regression_from_keywords():
    """Test classification of regression problems."""
    state = {
        "business_objective": "Predict prescription volume for our brand",
        "target_outcome": "Forecast monthly TRx count per HCP",
    }

    result = await classify_problem(state)

    assert result["inferred_problem_type"] == "regression"
    assert result["inferred_target_variable"] == "prescription_count"


@pytest.mark.asyncio
async def test_classify_causal_inference_from_keywords():
    """Test classification of causal inference problems."""
    state = {
        "business_objective": "Measure impact of email campaigns",
        "target_outcome": "Determine causal effect of email on prescriptions",
    }

    result = await classify_problem(state)

    assert result["inferred_problem_type"] == "causal_inference"


@pytest.mark.asyncio
async def test_classify_time_series_from_keywords():
    """Test classification of time series problems."""
    state = {
        "business_objective": "Forecast future market trends",
        "target_outcome": "Predict next quarter prescription trends",
    }

    result = await classify_problem(state)

    assert result["inferred_problem_type"] == "time_series"
    assert result["prediction_horizon_days"] == 90  # Quarter


@pytest.mark.asyncio
async def test_problem_type_hint_overrides_inference():
    """Test that explicit problem_type_hint overrides automatic inference."""
    state = {
        "business_objective": "Identify prescribers",
        "target_outcome": "Predict prescriptions",
        "problem_type_hint": "regression",  # Override
    }

    result = await classify_problem(state)

    # Should use hint instead of inferring binary classification
    assert result["inferred_problem_type"] == "regression"


def test_infer_binary_classification():
    """Test binary classification inference from various phrasings."""
    test_cases = [
        ("will prescribe", "will_prescribe"),
        ("will churn", "will_churn"),
        ("will convert", "will_convert"),
        ("likely to adopt", "will_adopt"),
    ]

    for objective, expected_target in test_cases:
        problem_type = _infer_problem_type(objective, "")
        assert problem_type == "binary_classification"

        target = _infer_target_variable(objective, "binary_classification")
        assert target == expected_target


def test_infer_regression():
    """Test regression inference from various phrasings."""
    test_cases = [
        "prescription volume",
        "number of TRx",
        "amount of prescriptions",
        "prescription count",
    ]

    for objective in test_cases:
        problem_type = _infer_problem_type(objective, "")
        assert problem_type == "regression"


def test_infer_causal():
    """Test causal inference detection."""
    test_cases = [
        "impact of email",
        "effect of campaign",
        "caused by intervention",
        "causal relationship",
        "attribution of sales",
    ]

    for objective in test_cases:
        problem_type = _infer_problem_type(objective, "")
        assert problem_type == "causal_inference"


def test_infer_target_variable_sanitization():
    """Test that complex outcome strings are sanitized to valid variable names."""
    outcome = "Predict HCP's likelihood to prescribe our drug"
    target = _infer_target_variable(outcome, "binary_classification")

    # Should sanitize to valid Python variable name
    assert "_" in target or target.isidentifier()
    assert not target.startswith("_")
    assert not target.endswith("_")


def test_infer_prediction_horizon_90days():
    """Test 90-day prediction horizon inference."""
    test_cases = [
        "in next 90 days",
        "over 3 months",
        "next quarter",
    ]

    for outcome in test_cases:
        horizon = _infer_prediction_horizon(outcome)
        assert horizon == 90


def test_infer_prediction_horizon_30days():
    """Test 30-day prediction horizon inference (default)."""
    test_cases = [
        "in next 30 days",
        "next month",
        "over 1 month",
        "unknown time period",  # Should default to 30
    ]

    for outcome in test_cases:
        horizon = _infer_prediction_horizon(outcome)
        assert horizon == 30


def test_infer_prediction_horizon_7days():
    """Test 7-day prediction horizon inference."""
    test_cases = [
        "in next 7 days",
        "next week",
    ]

    for outcome in test_cases:
        horizon = _infer_prediction_horizon(outcome)
        assert horizon == 7


@pytest.mark.asyncio
async def test_classify_handles_empty_fields():
    """Test classification handles missing or empty optional fields."""
    state = {
        "business_objective": "Identify prescribers",
        "target_outcome": "Predict prescriptions",
        # No problem_type_hint
    }

    result = await classify_problem(state)

    # Should still classify successfully
    assert "inferred_problem_type" in result
    assert "inferred_target_variable" in result
    assert "prediction_horizon_days" in result
