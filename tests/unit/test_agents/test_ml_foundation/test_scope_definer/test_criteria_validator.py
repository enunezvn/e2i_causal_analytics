"""Tests for success criteria validation."""

import pytest
from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
    define_success_criteria,
    _define_classification_criteria,
    _define_regression_criteria,
    _define_causal_criteria,
    _define_baseline_model,
    _validate_criteria,
)


@pytest.mark.asyncio
async def test_define_classification_criteria():
    """Test success criteria for classification problems."""
    state = {
        "inferred_problem_type": "binary_classification",
        "experiment_id": "exp_test_123",
    }

    result = await define_success_criteria(state)

    criteria = result["success_criteria"]

    # Should have classification metrics
    assert "minimum_auc" in criteria
    assert "minimum_precision" in criteria
    assert "minimum_recall" in criteria
    assert "minimum_f1" in criteria

    # Should NOT have regression metrics
    assert criteria["minimum_rmse"] is None
    assert criteria["minimum_r2"] is None

    # Should have baseline
    assert criteria["baseline_model"] == "random_forest_baseline"


@pytest.mark.asyncio
async def test_define_regression_criteria():
    """Test success criteria for regression problems."""
    state = {
        "inferred_problem_type": "regression",
        "experiment_id": "exp_test_123",
    }

    result = await define_success_criteria(state)

    criteria = result["success_criteria"]

    # Should have regression metrics
    assert "minimum_rmse" in criteria
    assert "minimum_r2" in criteria
    assert "minimum_mape" in criteria

    # Should NOT have classification metrics
    assert criteria["minimum_auc"] is None
    assert criteria["minimum_precision"] is None

    # Should have baseline
    assert criteria["baseline_model"] == "linear_regression_baseline"


@pytest.mark.asyncio
async def test_define_causal_criteria():
    """Test success criteria for causal inference problems."""
    state = {
        "inferred_problem_type": "causal_inference",
        "experiment_id": "exp_test_123",
    }

    result = await define_success_criteria(state)

    criteria = result["success_criteria"]

    # Should use RMSE for ATE standard error
    assert "minimum_rmse" in criteria
    assert "minimum_r2" in criteria

    # Should have causal baseline
    assert criteria["baseline_model"] == "ols_baseline"


@pytest.mark.asyncio
async def test_performance_requirements_override_defaults():
    """Test that explicit performance requirements override defaults."""
    state = {
        "inferred_problem_type": "binary_classification",
        "experiment_id": "exp_test_123",
        "performance_requirements": {
            "min_auc": 0.85,
            "min_precision": 0.80,
            "min_recall": 0.75,
            "min_f1": 0.78,
        },
    }

    result = await define_success_criteria(state)

    criteria = result["success_criteria"]

    # Should use custom thresholds
    assert criteria["minimum_auc"] == 0.85
    assert criteria["minimum_precision"] == 0.80
    assert criteria["minimum_recall"] == 0.75
    assert criteria["minimum_f1"] == 0.78


@pytest.mark.asyncio
async def test_minimum_lift_over_baseline():
    """Test that minimum lift over baseline is set."""
    state = {
        "inferred_problem_type": "binary_classification",
        "experiment_id": "exp_test_123",
        "performance_requirements": {"min_lift": 0.15},
    }

    result = await define_success_criteria(state)

    criteria = result["success_criteria"]

    # Should have 15% lift requirement
    assert criteria["minimum_lift_over_baseline"] == 0.15


@pytest.mark.asyncio
async def test_default_minimum_lift():
    """Test default minimum lift is 10%."""
    state = {
        "inferred_problem_type": "binary_classification",
        "experiment_id": "exp_test_123",
    }

    result = await define_success_criteria(state)

    criteria = result["success_criteria"]

    # Should default to 10% lift
    assert criteria["minimum_lift_over_baseline"] == 0.10


def test_classification_criteria_defaults():
    """Test default thresholds for classification."""
    criteria = _define_classification_criteria({})

    # Check reasonable defaults
    assert 0.5 <= criteria["minimum_auc"] <= 0.9
    assert 0.5 <= criteria["minimum_precision"] <= 0.9
    assert 0.5 <= criteria["minimum_recall"] <= 0.9
    assert 0.5 <= criteria["minimum_f1"] <= 0.9


def test_regression_criteria_defaults():
    """Test default thresholds for regression."""
    criteria = _define_regression_criteria({})

    # Check reasonable defaults
    assert criteria["minimum_r2"] >= 0.5
    assert criteria["minimum_rmse"] is not None
    assert criteria["minimum_mape"] is not None


def test_baseline_model_selection():
    """Test baseline model selection for different problem types."""
    # Classification
    assert _define_baseline_model("binary_classification") == "random_forest_baseline"
    assert _define_baseline_model("multiclass_classification") == "random_forest_baseline"

    # Regression
    assert _define_baseline_model("regression") == "linear_regression_baseline"

    # Causal
    assert _define_baseline_model("causal_inference") == "ols_baseline"

    # Time series
    assert _define_baseline_model("time_series") == "arima_baseline"


def test_validate_criteria_warns_on_low_samples():
    """Test validation warns when minimum samples is too low."""
    criteria = {"experiment_id": "exp_test_123", "baseline_model": "test"}

    state = {
        "scope_spec": {"minimum_samples": 50}  # Very low
    }

    result = _validate_criteria(criteria, state)

    # Should pass but warn
    assert result["passed"] is True
    assert len(result["warnings"]) > 0
    assert any("sample" in w.lower() for w in result["warnings"])


def test_validate_criteria_warns_on_high_auc():
    """Test validation warns when AUC threshold is unrealistic."""
    criteria = {
        "experiment_id": "exp_test_123",
        "baseline_model": "test",
        "minimum_auc": 0.98,  # Very high
    }

    state = {"scope_spec": {"minimum_samples": 1000}}

    result = _validate_criteria(criteria, state)

    # Should pass but warn
    assert result["passed"] is True
    assert len(result["warnings"]) > 0
    assert any("auc" in w.lower() for w in result["warnings"])


def test_validate_criteria_warns_on_high_r2():
    """Test validation warns when R² threshold is unrealistic."""
    criteria = {
        "experiment_id": "exp_test_123",
        "baseline_model": "test",
        "minimum_r2": 0.95,  # Very high
    }

    state = {"scope_spec": {"minimum_samples": 1000}}

    result = _validate_criteria(criteria, state)

    # Should pass but warn
    assert result["passed"] is True
    assert len(result["warnings"]) > 0
    assert any("r²" in w.lower() or "r2" in w.lower() for w in result["warnings"])


def test_validate_criteria_warns_on_low_time_budget():
    """Test validation warns when time budget is too low."""
    criteria = {
        "experiment_id": "exp_test_123",
        "baseline_model": "test",
    }

    state = {
        "scope_spec": {"minimum_samples": 1000},
        "time_budget_hours": 0.5,  # 30 minutes - very low
    }

    result = _validate_criteria(criteria, state)

    # Should pass but warn
    assert result["passed"] is True
    assert len(result["warnings"]) > 0
    assert any("time" in w.lower() for w in result["warnings"])


def test_validate_criteria_fails_on_missing_experiment_id():
    """Test validation fails when experiment_id is missing."""
    criteria = {
        "baseline_model": "test",
        # Missing experiment_id
    }

    state = {"scope_spec": {"minimum_samples": 1000}}

    result = _validate_criteria(criteria, state)

    # Should fail
    assert result["passed"] is False
    assert len(result["errors"]) > 0
    assert any("experiment_id" in e.lower() for e in result["errors"])


def test_validate_criteria_fails_on_missing_baseline():
    """Test validation fails when baseline_model is missing."""
    criteria = {
        "experiment_id": "exp_test_123",
        # Missing baseline_model
    }

    state = {"scope_spec": {"minimum_samples": 1000}}

    result = _validate_criteria(criteria, state)

    # Should fail
    assert result["passed"] is False
    assert len(result["errors"]) > 0
    assert any("baseline" in e.lower() for e in result["errors"])


@pytest.mark.asyncio
async def test_validation_passed_flag_set():
    """Test that validation_passed flag is correctly set."""
    # Valid state
    state = {
        "inferred_problem_type": "binary_classification",
        "experiment_id": "exp_test_123",
        "scope_spec": {"minimum_samples": 1000},
    }

    result = await define_success_criteria(state)

    # Should pass validation
    assert result["validation_passed"] is True
    assert len(result["validation_errors"]) == 0


@pytest.mark.asyncio
async def test_validation_errors_populated_on_failure():
    """Test that validation_errors are populated when validation fails."""
    # Missing experiment_id
    state = {
        "inferred_problem_type": "binary_classification",
        # No experiment_id
        "scope_spec": {"minimum_samples": 1000},
    }

    result = await define_success_criteria(state)

    # Should fail validation
    assert result["validation_passed"] is False
    assert len(result["validation_errors"]) > 0
