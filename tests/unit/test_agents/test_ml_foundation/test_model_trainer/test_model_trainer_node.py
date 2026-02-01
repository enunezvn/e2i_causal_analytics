"""Tests for model training node.

Tests the train_model function with various algorithms and configurations.
"""

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.model_trainer_node import (
    train_model,
    _get_model_class_dynamic,
    _filter_hyperparameters,
    _get_framework,
)


# ============================================================================
# Test fixtures
# ============================================================================

@pytest.fixture
def binary_classification_state():
    """Create state for binary classification."""
    np.random.seed(42)
    return {
        "algorithm_name": "RandomForest",
        "problem_type": "binary_classification",
        "best_hyperparameters": {"n_estimators": 10, "max_depth": 3},
        "X_train_preprocessed": np.random.rand(100, 5),
        "X_validation_preprocessed": np.random.rand(30, 5),
        "train_data": {"y": np.random.randint(0, 2, 100)},
        "validation_data": {"y": np.random.randint(0, 2, 30)},
        "early_stopping": False,
    }


@pytest.fixture
def regression_state():
    """Create state for regression."""
    np.random.seed(42)
    return {
        "algorithm_name": "RandomForest",
        "problem_type": "regression",
        "best_hyperparameters": {"n_estimators": 10, "max_depth": 3},
        "X_train_preprocessed": np.random.rand(100, 5),
        "train_data": {"y": np.random.rand(100)},
        "early_stopping": False,
    }


# ============================================================================
# Test train_model function
# ============================================================================

@pytest.mark.asyncio
class TestTrainModel:
    """Test core model training."""

    async def test_trains_random_forest_classifier(self, binary_classification_state):
        """Should train RandomForest classifier successfully."""
        result = await train_model(binary_classification_state)

        assert "error" not in result
        assert result["trained_model"] is not None
        assert result["training_status"] == "completed"
        assert result["algorithm_name"] == "RandomForest"
        assert result["framework"] == "sklearn"

    async def test_trains_random_forest_regressor(self, regression_state):
        """Should train RandomForest regressor successfully."""
        result = await train_model(regression_state)

        assert "error" not in result
        assert result["trained_model"] is not None
        assert result["training_status"] == "completed"

    async def test_trains_logistic_regression(self, binary_classification_state):
        """Should train LogisticRegression successfully."""
        binary_classification_state["algorithm_name"] = "LogisticRegression"
        binary_classification_state["best_hyperparameters"] = {"C": 1.0}

        result = await train_model(binary_classification_state)

        assert "error" not in result
        assert result["trained_model"] is not None
        assert result["framework"] == "sklearn"

    async def test_trains_gradient_boosting(self, binary_classification_state):
        """Should train GradientBoosting successfully."""
        binary_classification_state["algorithm_name"] = "GradientBoosting"
        binary_classification_state["best_hyperparameters"] = {
            "n_estimators": 10,
            "max_depth": 3,
        }

        result = await train_model(binary_classification_state)

        assert "error" not in result
        assert result["trained_model"] is not None

    async def test_trains_extra_trees(self, binary_classification_state):
        """Should train ExtraTrees successfully."""
        binary_classification_state["algorithm_name"] = "ExtraTrees"
        binary_classification_state["best_hyperparameters"] = {
            "n_estimators": 10,
            "max_depth": 3,
        }

        result = await train_model(binary_classification_state)

        assert "error" not in result
        assert result["trained_model"] is not None

    async def test_trains_ridge_regressor(self, regression_state):
        """Should train Ridge regressor successfully."""
        regression_state["algorithm_name"] = "Ridge"
        regression_state["best_hyperparameters"] = {"alpha": 1.0}

        result = await train_model(regression_state)

        assert "error" not in result
        assert result["trained_model"] is not None

    async def test_trains_lasso_regressor(self, regression_state):
        """Should train Lasso regressor successfully."""
        regression_state["algorithm_name"] = "Lasso"
        regression_state["best_hyperparameters"] = {"alpha": 0.1}

        result = await train_model(regression_state)

        assert "error" not in result
        assert result["trained_model"] is not None

    async def test_records_training_duration(self, binary_classification_state):
        """Should record training duration."""
        result = await train_model(binary_classification_state)

        assert result["training_duration_seconds"] >= 0
        assert result["training_started_at"] is not None
        assert result["training_completed_at"] is not None

    async def test_error_when_missing_training_data(self):
        """Should return error when training data missing."""
        state = {
            "algorithm_name": "RandomForest",
            "best_hyperparameters": {},
        }

        result = await train_model(state)

        assert "error" in result
        assert result["error_type"] == "missing_training_data"
        assert result["training_status"] == "failed"

    async def test_error_when_missing_algorithm_name(self, binary_classification_state):
        """Should return error when algorithm_name missing."""
        del binary_classification_state["algorithm_name"]

        result = await train_model(binary_classification_state)

        assert "error" in result
        assert result["error_type"] == "missing_algorithm_name"

    async def test_error_for_unsupported_algorithm(self, binary_classification_state):
        """Should return error for unsupported algorithm."""
        binary_classification_state["algorithm_name"] = "UnsupportedAlgorithm"

        result = await train_model(binary_classification_state)

        assert "error" in result
        assert result["error_type"] == "unsupported_algorithm"

    async def test_filters_incompatible_hyperparameters(self, binary_classification_state):
        """Should filter out incompatible hyperparameters."""
        binary_classification_state["best_hyperparameters"] = {
            "n_estimators": 10,
            "max_depth": 3,
            "invalid_param": "should_be_filtered",
            "learning_rate": 0.1,  # Not valid for RandomForest
        }

        result = await train_model(binary_classification_state)

        # Should still train successfully despite invalid params
        assert "error" not in result
        assert result["trained_model"] is not None


# ============================================================================
# Test helper functions
# ============================================================================

class TestGetModelClassDynamic:
    """Test model class lookup."""

    def test_gets_random_forest_classifier(self):
        """Should get RandomForestClassifier for binary classification."""
        model_class = _get_model_class_dynamic("RandomForest", "binary_classification")
        assert model_class is not None
        assert "RandomForest" in model_class.__name__

    def test_gets_random_forest_regressor(self):
        """Should get RandomForestRegressor for regression."""
        model_class = _get_model_class_dynamic("RandomForest", "regression")
        assert model_class is not None
        assert "RandomForest" in model_class.__name__
        assert "Regressor" in model_class.__name__

    def test_gets_logistic_regression(self):
        """Should get LogisticRegression."""
        model_class = _get_model_class_dynamic(
            "LogisticRegression", "binary_classification"
        )
        assert model_class is not None

    def test_gets_ridge(self):
        """Should get Ridge regressor."""
        model_class = _get_model_class_dynamic("Ridge", "regression")
        assert model_class is not None

    def test_gets_gradient_boosting_classifier(self):
        """Should get GradientBoostingClassifier."""
        model_class = _get_model_class_dynamic(
            "GradientBoosting", "binary_classification"
        )
        assert model_class is not None

    def test_returns_none_for_unknown_algorithm(self):
        """Should return None for unknown algorithm."""
        model_class = _get_model_class_dynamic("UnknownAlgorithm", "binary_classification")
        assert model_class is None


class TestFilterHyperparameters:
    """Test hyperparameter filtering."""

    def test_filters_random_forest_params(self):
        """Should keep valid RandomForest params and filter invalid."""
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,  # Invalid for RF
            "invalid_param": "value",
        }

        filtered = _filter_hyperparameters("RandomForest", params)

        assert "n_estimators" in filtered
        assert "max_depth" in filtered
        assert "learning_rate" not in filtered
        assert "invalid_param" not in filtered

    def test_adds_common_params(self):
        """Should add common params like random_state."""
        params = {"n_estimators": 100}

        filtered = _filter_hyperparameters("RandomForest", params)

        assert "random_state" in filtered
        assert "n_jobs" in filtered

    def test_filters_logistic_regression_params(self):
        """Should filter LogisticRegression params correctly."""
        params = {
            "C": 1.0,
            "penalty": "l2",
            "n_estimators": 100,  # Invalid for LR
        }

        filtered = _filter_hyperparameters("LogisticRegression", params)

        assert "C" in filtered
        assert "penalty" in filtered
        assert "n_estimators" not in filtered


class TestGetFramework:
    """Test framework identification."""

    def test_identifies_sklearn_algorithms(self):
        """Should identify sklearn algorithms."""
        assert _get_framework("RandomForest") == "sklearn"
        assert _get_framework("LogisticRegression") == "sklearn"
        assert _get_framework("Ridge") == "sklearn"
        assert _get_framework("GradientBoosting") == "sklearn"

    def test_identifies_xgboost(self):
        """Should identify XGBoost."""
        assert _get_framework("XGBoost") == "xgboost"

    def test_identifies_lightgbm(self):
        """Should identify LightGBM."""
        assert _get_framework("LightGBM") == "lightgbm"

    def test_identifies_econml_algorithms(self):
        """Should identify econml algorithms."""
        assert _get_framework("CausalForest") == "econml"
        assert _get_framework("LinearDML") == "econml"
        assert _get_framework("SLearner") == "econml"
        assert _get_framework("DRLearner") == "econml"
        assert _get_framework("TLearner") == "econml"
        assert _get_framework("XLearner") == "econml"

    def test_returns_unknown_for_unrecognized(self):
        """Should return 'unknown' for unrecognized algorithms."""
        assert _get_framework("UnknownAlgorithm") == "unknown"
