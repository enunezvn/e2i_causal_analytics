"""Tests for hyperparameter tuner node."""

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.hyperparameter_tuner import tune_hyperparameters


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
