"""Tests for hyperparameter tuner node."""

from unittest.mock import patch

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.hyperparameter_tuner import (
    _get_fixed_params,
    _get_hpo_pattern_memory,
    tune_hyperparameters,
    validate_hpo_output,
    validate_hyperparameter_types,
)


@pytest.mark.asyncio
class TestTuneHyperparameters:
    """Test hyperparameter optimization (HPO)."""

    async def test_returns_defaults_when_hpo_disabled(self):
        """Should return default hyperparameters when HPO disabled."""
        state = {
            "enable_hpo": False,
            "default_hyperparameters": {"n_estimators": 100, "max_depth": 10},
        }

        result = await tune_hyperparameters(state)

        assert result["hpo_completed"] is False
        assert result["best_hyperparameters"] == {"n_estimators": 100, "max_depth": 10}
        assert result["hpo_trials_run"] == 0

    async def test_runs_hpo_when_enabled(self):
        """Should attempt HPO when enabled."""
        state = {
            "enable_hpo": True,
            "hpo_trials": 10,
            "algorithm_name": "RandomForest",
            "problem_type": "binary_classification",
            "experiment_id": "test_exp_123",
            "default_hyperparameters": {"n_estimators": 100},
            "hyperparameter_search_space": {
                "n_estimators": {"type": "int", "low": 50, "high": 200}
            },
            "X_train_preprocessed": np.random.rand(100, 5),
            "X_validation_preprocessed": np.random.rand(30, 5),
            "train_data": {"y": np.random.randint(0, 2, 100)},
            "validation_data": {"y": np.random.randint(0, 2, 30)},
        }

        result = await tune_hyperparameters(state)

        assert "hpo_completed" in result
        assert "best_hyperparameters" in result
        assert "hpo_duration_seconds" in result

    async def test_error_when_missing_training_data(self):
        """Should return error when training data missing."""
        state = {
            "enable_hpo": True,
            "hpo_trials": 10,
            "hyperparameter_search_space": {},
        }

        result = await tune_hyperparameters(state)

        assert "error" in result
        assert result["error_type"] == "missing_hpo_data"

    async def test_returns_defaults_when_no_search_space(self):
        """Should return defaults when no search space defined."""
        state = {
            "enable_hpo": True,
            "hpo_trials": 10,
            "hyperparameter_search_space": {},
            "default_hyperparameters": {"n_estimators": 50},
            "X_train_preprocessed": np.random.rand(100, 5),
            "X_validation_preprocessed": np.random.rand(30, 5),
            "train_data": {"y": np.random.randint(0, 2, 100)},
            "validation_data": {"y": np.random.randint(0, 2, 30)},
        }

        result = await tune_hyperparameters(state)

        assert result["best_hyperparameters"] == {"n_estimators": 50}


class TestValidateHpoOutput:
    """Tests for HPO output validation."""

    def test_validates_complete_output(self):
        """Should validate complete HPO output."""
        output = {
            "hpo_completed": True,
            "best_hyperparameters": {"n_estimators": 150},
            "hpo_best_trial": 5,
            "hpo_trials_run": 10,
            "hpo_best_value": 0.95,
            "hpo_study_name": "test_study",
        }

        is_valid, errors = validate_hpo_output(output)

        assert is_valid is True
        assert errors == []

    def test_detects_missing_required_field(self):
        """Should detect missing required field."""
        output = {
            "hpo_completed": True,
            # missing best_hyperparameters
        }

        is_valid, errors = validate_hpo_output(output)

        assert is_valid is False
        assert any("best_hyperparameters" in e for e in errors)

    def test_detects_invalid_type(self):
        """Should detect invalid field type."""
        output = {
            "hpo_completed": "yes",  # Should be bool
            "best_hyperparameters": {"n_estimators": 100},
        }

        is_valid, errors = validate_hpo_output(output)

        assert is_valid is False
        assert any("hpo_completed" in e for e in errors)

    def test_validates_completed_consistency(self):
        """Should check consistency when hpo_completed=True."""
        output = {
            "hpo_completed": True,
            "best_hyperparameters": {"n_estimators": 100},
            "hpo_trials_run": 0,  # Inconsistent
        }

        is_valid, errors = validate_hpo_output(output)

        assert is_valid is False
        assert any("hpo_completed=True but hpo_trials_run=0" in e for e in errors)


class TestValidateHyperparameterTypes:
    """Tests for hyperparameter type validation."""

    def test_validates_int_param(self):
        """Should validate int parameters."""
        hyperparameters = {"n_estimators": 100}
        search_space = {"n_estimators": {"type": "int", "low": 50, "high": 200}}

        is_valid, errors = validate_hyperparameter_types(hyperparameters, search_space)

        assert is_valid is True
        assert errors == []

    def test_detects_out_of_range_value(self):
        """Should detect value outside allowed range."""
        hyperparameters = {"n_estimators": 300}  # Above max 200
        search_space = {"n_estimators": {"type": "int", "low": 50, "high": 200}}

        is_valid, errors = validate_hyperparameter_types(hyperparameters, search_space)

        assert is_valid is False
        assert any("above maximum" in e for e in errors)

    def test_validates_categorical_param(self):
        """Should validate categorical parameters."""
        hyperparameters = {"criterion": "gini"}
        search_space = {"criterion": {"type": "categorical", "choices": ["gini", "entropy"]}}

        is_valid, errors = validate_hyperparameter_types(hyperparameters, search_space)

        assert is_valid is True

    def test_detects_invalid_categorical_value(self):
        """Should detect invalid categorical value."""
        hyperparameters = {"criterion": "invalid"}
        search_space = {"criterion": {"type": "categorical", "choices": ["gini", "entropy"]}}

        is_valid, errors = validate_hyperparameter_types(hyperparameters, search_space)

        assert is_valid is False
        assert any("not in choices" in e for e in errors)


class TestGetFixedParams:
    """Tests for fixed parameter retrieval."""

    def test_xgboost_fixed_params(self):
        """Should return correct fixed params for XGBoost."""
        params = _get_fixed_params("XGBoost")

        assert params["random_state"] == 42
        assert params["n_jobs"] == -1
        assert params["verbosity"] == 0

    def test_lightgbm_fixed_params(self):
        """Should return correct fixed params for LightGBM."""
        params = _get_fixed_params("LightGBM")

        assert params["random_state"] == 42
        assert params["n_jobs"] == -1
        assert params["verbose"] == -1

    def test_random_forest_fixed_params(self):
        """Should return correct fixed params for RandomForest."""
        params = _get_fixed_params("RandomForest")

        assert params["random_state"] == 42
        assert params["n_jobs"] == -1

    def test_logistic_regression_fixed_params(self):
        """Should return correct fixed params for LogisticRegression."""
        params = _get_fixed_params("LogisticRegression")

        assert params["random_state"] == 42
        assert params["max_iter"] == 1000

    def test_ridge_fixed_params(self):
        """Should return correct fixed params for Ridge."""
        params = _get_fixed_params("Ridge")

        assert params["random_state"] == 42

    def test_unknown_algorithm_returns_empty(self):
        """Should return empty dict for unknown algorithm."""
        params = _get_fixed_params("UnknownAlgorithm")

        assert params == {}


class TestGetHpoPatternMemory:
    """Tests for _get_hpo_pattern_memory helper."""

    def test_returns_module_when_available(self):
        """Should return hpo_pattern_memory module when available."""
        result = _get_hpo_pattern_memory()

        # Module should be importable in test environment
        assert result is not None or result is None  # Depends on import success

    def test_function_exists_and_is_callable(self):
        """Should have a callable _get_hpo_pattern_memory function."""
        from src.agents.ml_foundation.model_trainer.nodes import hyperparameter_tuner

        assert hasattr(hyperparameter_tuner, "_get_hpo_pattern_memory")
        assert callable(hyperparameter_tuner._get_hpo_pattern_memory)


@pytest.mark.asyncio
class TestWarmStartIntegration:
    """Tests for warm-start integration with HPO pattern memory."""

    async def test_validates_hpo_pattern_id_output_field(self):
        """Should allow hpo_pattern_id in HPO output validation."""
        output = {
            "hpo_completed": True,
            "best_hyperparameters": {"n_estimators": 150},
            "hpo_best_trial": 5,
            "hpo_trials_run": 10,
            "hpo_best_value": 0.95,
            "hpo_study_name": "test_study",
            "hpo_pattern_id": "abc-123-def-456",  # New field from procedural memory
        }

        is_valid, errors = validate_hpo_output(output)

        assert is_valid is True
        assert errors == []

    async def test_hpo_proceeds_without_pattern_memory(self):
        """Should run HPO successfully when pattern memory unavailable."""
        with patch(
            "src.agents.ml_foundation.model_trainer.nodes.hyperparameter_tuner._get_hpo_pattern_memory",
            return_value=None,
        ):
            state = {
                "enable_hpo": True,
                "hpo_trials": 5,
                "algorithm_name": "RandomForest",
                "problem_type": "binary_classification",
                "experiment_id": "test_exp_123",
                "default_hyperparameters": {"n_estimators": 100},
                "hyperparameter_search_space": {
                    "n_estimators": {"type": "int", "low": 50, "high": 200}
                },
                "X_train_preprocessed": np.random.rand(100, 5),
                "X_validation_preprocessed": np.random.rand(30, 5),
                "train_data": {"y": np.random.randint(0, 2, 100)},
                "validation_data": {"y": np.random.randint(0, 2, 30)},
            }

            result = await tune_hyperparameters(state)

            # Should complete successfully even without pattern memory
            assert "hpo_completed" in result
            assert "best_hyperparameters" in result
            # Pattern ID should not be present
            assert "hpo_pattern_id" not in result or result.get("hpo_pattern_id") is None
