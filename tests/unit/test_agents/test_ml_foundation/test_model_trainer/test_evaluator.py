"""Tests for model evaluator node.

Tests the evaluate_model function with various problem types and edge cases.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    _check_success_criteria,
    _compute_optimal_threshold,
    _compute_precision_at_k,
    evaluate_model,
)

# ============================================================================
# Test fixtures
# ============================================================================


class MockBinaryClassifier:
    """Mock trained binary classifier."""

    def predict(self, X):
        return np.random.randint(0, 2, len(X))

    def predict_proba(self, X):
        proba = np.random.rand(len(X))
        return np.column_stack([1 - proba, proba])


class MockRegressor:
    """Mock trained regressor."""

    def predict(self, X):
        return np.random.rand(len(X))


class MockClassifierNoProba:
    """Mock classifier without predict_proba."""

    def predict(self, X):
        return np.random.randint(0, 2, len(X))


@pytest.fixture
def binary_classification_state():
    """Create state for binary classification evaluation."""
    np.random.seed(42)
    model = MockBinaryClassifier()
    return {
        "trained_model": model,
        "problem_type": "binary_classification",
        "X_train_preprocessed": np.random.rand(100, 5),
        "X_validation_preprocessed": np.random.rand(30, 5),
        "X_test_preprocessed": np.random.rand(20, 5),
        "train_data": {"y": np.random.randint(0, 2, 100)},
        "validation_data": {"y": np.random.randint(0, 2, 30)},
        "test_data": {"y": np.random.randint(0, 2, 20)},
        "success_criteria": {},
    }


@pytest.fixture
def regression_state():
    """Create state for regression evaluation."""
    np.random.seed(42)
    model = MockRegressor()
    return {
        "trained_model": model,
        "problem_type": "regression",
        "X_train_preprocessed": np.random.rand(100, 5),
        "X_validation_preprocessed": np.random.rand(30, 5),
        "X_test_preprocessed": np.random.rand(20, 5),
        "train_data": {"y": np.random.rand(100)},
        "validation_data": {"y": np.random.rand(30)},
        "test_data": {"y": np.random.rand(20)},
        "success_criteria": {},
    }


@pytest.fixture
def real_classifier_state():
    """Create state with real trained classifier for accurate testing."""
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.rand(30, 5)
    y_val = np.random.randint(0, 2, 30)
    X_test = np.random.rand(20, 5)
    y_test = np.random.randint(0, 2, 20)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
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
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)
    X_val = np.random.rand(30, 5)
    y_val = np.random.rand(30)
    X_test = np.random.rand(20, 5)
    y_test = np.random.rand(20)

    model = RandomForestRegressor(n_estimators=10, random_state=42)
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
        """Should check if model meets success criteria."""
        binary_classification_state["success_criteria"] = {"accuracy": 0.90}

        result = await evaluate_model(binary_classification_state)

        assert "success_criteria_met" in result
        assert "success_criteria_results" in result
        assert "accuracy" in result["success_criteria_results"]

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
            "X_test_preprocessed": np.random.rand(20, 5),
            "test_data": {"y": np.random.randint(0, 2, 20)},
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
        np.random.seed(42)
        state = {
            "trained_model": MockClassifierNoProba(),
            "problem_type": "binary_classification",
            "X_test_preprocessed": np.random.rand(20, 5),
            "test_data": {"y": np.random.randint(0, 2, 20)},
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


# ============================================================================
# Test helper functions
# ============================================================================


class TestComputeOptimalThreshold:
    """Test optimal threshold computation."""

    def test_returns_threshold_with_proba(self):
        """Should compute optimal threshold with probabilities."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_proba = np.column_stack(
            [
                1 - np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4]),
                np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4]),
            ]
        )

        threshold = _compute_optimal_threshold(y_true, y_proba)

        assert 0.0 <= threshold <= 1.0

    def test_returns_default_without_proba(self):
        """Should return 0.5 when no probabilities provided."""
        y_true = np.array([0, 0, 1, 1])

        threshold = _compute_optimal_threshold(y_true, None)

        assert threshold == 0.5


class TestComputePrecisionAtK:
    """Test precision@k computation."""

    def test_computes_precision_at_k(self):
        """Should compute precision at various k values."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_proba = np.column_stack(
            [
                1 - np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4]),
                np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4]),
            ]
        )

        result = _compute_precision_at_k(y_true, y_proba, k_values=[2, 4])

        assert 2 in result
        assert 4 in result
        assert 0.0 <= result[2] <= 1.0
        assert 0.0 <= result[4] <= 1.0

    def test_returns_empty_without_proba(self):
        """Should return empty dict without probabilities."""
        y_true = np.array([0, 0, 1, 1])

        result = _compute_precision_at_k(y_true, None, k_values=[2])

        assert result == {}

    def test_skips_k_larger_than_samples(self):
        """Should skip k values larger than sample size."""
        y_true = np.array([0, 1, 1])
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])

        result = _compute_precision_at_k(y_true, y_proba, k_values=[2, 100])

        assert 2 in result
        assert 100 not in result


class TestCheckSuccessCriteria:
    """Test success criteria checking."""

    def test_all_criteria_met(self):
        """Should return True when all criteria met."""
        test_metrics = {"accuracy": 0.85, "roc_auc": 0.90}
        success_criteria = {"accuracy": 0.80, "auc": 0.85}

        result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")

        assert result["success_criteria_met"] is True

    def test_criteria_not_met(self):
        """Should return False when criteria not met."""
        test_metrics = {"accuracy": 0.75, "roc_auc": 0.80}
        success_criteria = {"accuracy": 0.90}

        result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")

        assert result["success_criteria_met"] is False
        assert result["success_criteria_results"]["accuracy"] is False

    def test_lower_is_better_metrics(self):
        """Should correctly handle metrics where lower is better."""
        test_metrics = {"rmse": 0.1, "mae": 0.05}
        success_criteria = {"rmse": 0.2, "mae": 0.1}

        result = _check_success_criteria(test_metrics, success_criteria, "regression")

        assert result["success_criteria_met"] is True

    def test_empty_criteria_returns_true(self):
        """Should return True when no criteria specified."""
        result = _check_success_criteria({}, {}, "binary_classification")

        assert result["success_criteria_met"] is True

    def test_handles_missing_metrics(self):
        """Should handle missing metrics gracefully."""
        test_metrics = {"accuracy": 0.85}
        success_criteria = {"accuracy": 0.80, "nonexistent_metric": 0.5}

        result = _check_success_criteria(test_metrics, success_criteria, "binary_classification")

        assert result["success_criteria_met"] is False
        assert result["success_criteria_results"]["nonexistent_metric"] is False
