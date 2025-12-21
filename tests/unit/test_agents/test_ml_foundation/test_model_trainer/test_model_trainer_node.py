"""Tests for model training node."""

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.model_trainer_node import train_model


@pytest.mark.asyncio
class TestTrainModel:
    """Test core model training."""

    async def test_trains_model_successfully(self):
        """Should train model and return trained model object."""
        state = {
            "algorithm_class": "sklearn.ensemble.RandomForestClassifier",
            "algorithm_name": "RandomForest",
            "best_hyperparameters": {"n_estimators": 10, "max_depth": 3},
            "X_train_preprocessed": np.random.rand(100, 5),
            "train_data": {"y": np.random.randint(0, 2, 100)},
            "early_stopping": False,
        }

        result = await train_model(state)

        assert "error" not in result
        assert result["trained_model"] is not None
        assert result["training_status"] == "completed"

    async def test_records_training_duration(self):
        """Should record training duration."""
        state = {
            "algorithm_class": "sklearn.ensemble.RandomForestClassifier",
            "algorithm_name": "RandomForest",
            "best_hyperparameters": {},
            "X_train_preprocessed": np.random.rand(100, 5),
            "train_data": {"y": np.random.randint(0, 2, 100)},
            "early_stopping": False,
        }

        result = await train_model(state)

        assert result["training_duration_seconds"] >= 0

    async def test_error_when_missing_training_data(self):
        """Should return error when training data missing."""
        state = {
            "algorithm_class": "sklearn.ensemble.RandomForestClassifier",
            "best_hyperparameters": {},
        }

        result = await train_model(state)

        assert "error" in result
        assert result["error_type"] == "missing_training_data"
