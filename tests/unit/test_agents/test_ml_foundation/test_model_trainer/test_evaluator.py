"""Smoke + integration tests for ``evaluate_model``.

End-to-end coverage of the evaluator node: invokes ``evaluate_model``
with a populated state and asserts the high-level shape of the result
(metric blocks present, error branches reachable, sklearn classifiers
and regressors round-trip cleanly).

The focused helper-level tests live in sibling files (split out in
1A-M-6 to keep this file tractable):

* ``test_threshold_selection.py`` - threshold tuning, freezing,
  provenance, ``_select_threshold``.
* ``test_metrics_computation.py`` - ``_compute_precision_at_k``,
  ``_positive_class_proba``, business_utility from cost_matrix.
* ``test_provenance.py`` - ``_check_success_criteria`` audit fields.

Shared mocks and the minimal-state fixtures
(``binary_classification_state`` / ``regression_state``) live in
``conftest.py`` so all four files share a single source of truth.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    evaluate_model,
)
from tests.unit.test_agents.test_ml_foundation.test_model_trainer.conftest import (
    N_FEATURES,
    N_TEST_SAMPLES,
    N_TRAIN_SAMPLES,
    N_VAL_SAMPLES,
    RANDOM_STATE,
    RF_N_ESTIMATORS,
    MockClassifierNoProba,
)


@pytest.fixture
def real_classifier_state():
    """Create state with real trained classifier for accurate testing."""
    np.random.seed(RANDOM_STATE)
    X_train = np.random.rand(N_TRAIN_SAMPLES, N_FEATURES)
    y_train = np.random.randint(0, 2, N_TRAIN_SAMPLES)
    X_val = np.random.rand(N_VAL_SAMPLES, N_FEATURES)
    y_val = np.random.randint(0, 2, N_VAL_SAMPLES)
    X_test = np.random.rand(N_TEST_SAMPLES, N_FEATURES)
    y_test = np.random.randint(0, 2, N_TEST_SAMPLES)

    model = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return {
        "trained_model": model,
        "problem_type": "binary_classification",
        "X_train_preprocessed": X_train,
        "X_validation_preprocessed": X_val,
        "X_test_preprocessed": X_test,
        "train_data": {"y": y_train},
        "validation_data": {"y": y_val},
        "test_data": {"y": y_test},
        "success_criteria": {},
    }


@pytest.fixture
def real_regressor_state():
    """Create state with real trained regressor."""
    np.random.seed(RANDOM_STATE)
    X_train = np.random.rand(N_TRAIN_SAMPLES, N_FEATURES)
    y_train = np.random.rand(N_TRAIN_SAMPLES)
    X_val = np.random.rand(N_VAL_SAMPLES, N_FEATURES)
    y_val = np.random.rand(N_VAL_SAMPLES)
    X_test = np.random.rand(N_TEST_SAMPLES, N_FEATURES)
    y_test = np.random.rand(N_TEST_SAMPLES)

    model = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return {
        "trained_model": model,
        "problem_type": "regression",
        "X_train_preprocessed": X_train,
        "X_validation_preprocessed": X_val,
        "X_test_preprocessed": X_test,
        "train_data": {"y": y_train},
        "validation_data": {"y": y_val},
        "test_data": {"y": y_test},
        "success_criteria": {},
    }


# ============================================================================
# Test evaluate_model function
# ============================================================================


@pytest.mark.asyncio
class TestEvaluateModel:
    """Test core model evaluation."""

    async def test_evaluates_on_all_splits(self, binary_classification_state):
        """Should evaluate on train, validation, and test splits."""
        result = await evaluate_model(binary_classification_state)

        assert "error" not in result
        assert "train_metrics" in result
        assert "validation_metrics" in result
        assert "test_metrics" in result

    async def test_returns_classification_metrics(self, binary_classification_state):
        """Should return classification metrics for classification problems."""
        result = await evaluate_model(binary_classification_state)

        assert "error" not in result
        assert result["auc_roc"] is not None
        assert result["precision"] is not None
        assert result["recall"] is not None
        assert result["f1_score"] is not None

    async def test_returns_regression_metrics(self, regression_state):
        """Should return regression metrics for regression problems."""
        result = await evaluate_model(regression_state)

        assert "error" not in result
        assert result["rmse"] is not None
        assert result["mae"] is not None
        assert result["r2"] is not None
        # Classification metrics should be None
        assert result["auc_roc"] is None
        assert result["precision"] is None

    async def test_checks_success_criteria(self, binary_classification_state):
        """Should check if model meets success criteria.

        Section B (pre_phase2_unblockers): the lift criterion participates
        in the aggregation alongside the other thresholds — when both
        train and test have enough samples (binary_classification_state
        uses N_TRAIN=100, N_TEST=20 — both ≥ 10), the baseline AUC is
        produced and the criterion is evaluated rather than soft-skipped.
        """
        binary_classification_state["success_criteria"] = {
            "accuracy": 0.90,
            "minimum_lift_over_baseline": 0.10,
        }

        result = await evaluate_model(binary_classification_state)

        assert "success_criteria_met" in result
        assert "success_criteria_results" in result
        assert "accuracy" in result["success_criteria_results"]
        # Lift criterion must be present in the results dict — the
        # MockBinaryClassifier's noisy predictions may or may not pass the
        # 0.10 threshold (deterministic since random_state is pinned), but
        # the criterion must have actually been checked (not soft-skipped).
        assert "minimum_lift_over_baseline" in result["success_criteria_results"]
        assert result["success_criteria_results"]["minimum_lift_over_baseline"] in (
            True,
            False,
        )
        # And the underlying metrics must be in test_metrics.
        assert "baseline_test_auc" in result["test_metrics"]
        assert "minimum_lift_over_baseline" in result["test_metrics"]

    async def test_success_criteria_met_when_threshold_passed(self, real_classifier_state):
        """Should set success_criteria_met=True when threshold is passed."""
        # Set very low threshold that should always be met
        real_classifier_state["success_criteria"] = {"accuracy": 0.1}

        result = await evaluate_model(real_classifier_state)

        assert result["success_criteria_met"] is True

    async def test_returns_confusion_matrix(self, binary_classification_state):
        """Should return confusion matrix for classification."""
        result = await evaluate_model(binary_classification_state)

        assert "confusion_matrix" in result
        assert result["confusion_matrix"] is not None

    async def test_returns_optimal_threshold(self, real_classifier_state):
        """Should compute optimal threshold for binary classification."""
        result = await evaluate_model(real_classifier_state)

        assert "optimal_threshold" in result
        # Threshold should be valid (0-1) or default (0.5)
        threshold = result["optimal_threshold"]
        assert isinstance(threshold, (int, float))
        assert 0.0 <= threshold <= 1.0 or threshold == 0.5

    async def test_returns_confidence_intervals(self, binary_classification_state):
        """Should compute bootstrap confidence intervals."""
        result = await evaluate_model(binary_classification_state)

        assert "confidence_interval" in result
        assert "bootstrap_samples" in result
        assert result["bootstrap_samples"] == 1000

    async def test_error_when_no_trained_model(self):
        """Should return error when trained_model is None."""
        state = {
            "problem_type": "binary_classification",
            "X_test_preprocessed": np.random.rand(N_TEST_SAMPLES, N_FEATURES),
            "test_data": {"y": np.random.randint(0, 2, N_TEST_SAMPLES)},
        }

        result = await evaluate_model(state)

        assert "error" in result
        assert result["error_type"] == "missing_trained_model"

    async def test_error_when_no_test_data(self, binary_classification_state):
        """Should return error when test data is missing."""
        del binary_classification_state["X_test_preprocessed"]
        del binary_classification_state["test_data"]

        result = await evaluate_model(binary_classification_state)

        assert "error" in result
        assert result["error_type"] == "missing_test_data"

    async def test_error_for_unsupported_problem_type(self, binary_classification_state):
        """Should return error for unsupported problem type."""
        binary_classification_state["problem_type"] = "unsupported_type"

        result = await evaluate_model(binary_classification_state)

        assert "error" in result
        assert result["error_type"] == "unsupported_problem_type"

    async def test_handles_model_without_predict_proba(self):
        """Should handle classifiers without predict_proba."""
        np.random.seed(RANDOM_STATE)
        state = {
            "trained_model": MockClassifierNoProba(),
            "problem_type": "binary_classification",
            "X_test_preprocessed": np.random.rand(N_TEST_SAMPLES, N_FEATURES),
            "test_data": {"y": np.random.randint(0, 2, N_TEST_SAMPLES)},
            "success_criteria": {},
        }

        result = await evaluate_model(state)

        # Should still succeed but without probability-based metrics
        assert "error" not in result
        assert result.get("auc_roc") is None  # No proba available

    async def test_evaluates_with_real_classifier(self, real_classifier_state):
        """Should evaluate real sklearn classifier correctly."""
        result = await evaluate_model(real_classifier_state)

        assert "error" not in result
        assert result["auc_roc"] is not None
        assert 0.0 <= result["auc_roc"] <= 1.0
        assert result["test_metrics"]["accuracy"] is not None

    async def test_evaluates_with_real_regressor(self, real_regressor_state):
        """Should evaluate real sklearn regressor correctly."""
        result = await evaluate_model(real_regressor_state)

        assert "error" not in result
        assert result["rmse"] is not None
        assert result["rmse"] >= 0
        assert result["mae"] is not None
        assert result["mae"] >= 0

    async def test_handles_missing_validation_data(self, binary_classification_state):
        """Should handle missing validation data gracefully."""
        del binary_classification_state["X_validation_preprocessed"]
        del binary_classification_state["validation_data"]

        result = await evaluate_model(binary_classification_state)

        assert "error" not in result
        assert result["validation_metrics"] == {}

    async def test_handles_continuous_problem_type(self, regression_state):
        """Should treat 'continuous' as regression."""
        regression_state["problem_type"] = "continuous"

        result = await evaluate_model(regression_state)

        assert "error" not in result
        assert result["rmse"] is not None


def test_check_success_criteria_skips_criteria_source_field() -> None:
    """The new ``criteria_source`` audit field must not be treated as a metric.

    Regression guard for the ADAPTIVE_CRITERIA flag plumbing in task 02 of
    .claude/plans/adaptive_success_criteria/. The validator now tags every
    ``success_criteria`` dict with a string ``criteria_source`` value;
    ``_check_success_criteria`` must route it through the existing
    skip-non-numeric branch (alongside ``experiment_id`` /
    ``baseline_model``) without recording a False result.
    """
    from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
        _check_success_criteria,
    )

    test_metrics = {"roc_auc": 0.80}
    success_criteria = {
        "minimum_auc": 0.75,
        "criteria_source": "fixed",  # str, not numeric — must be skipped
        "experiment_id": "abc",  # existing precedent for skip-non-numeric
        "baseline_model": "stratified_dummy",
    }
    result = _check_success_criteria(
        test_metrics, success_criteria, "binary_classification"
    )
    assert result["success_criteria_met"] is True
    assert "criteria_source" not in result["success_criteria_results"]
    assert "experiment_id" not in result["success_criteria_results"]
    assert result["success_criteria_results"]["minimum_auc"] is True
