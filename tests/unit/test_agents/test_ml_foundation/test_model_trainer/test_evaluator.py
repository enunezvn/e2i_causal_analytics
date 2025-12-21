"""Tests for model evaluator node."""

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.evaluator import evaluate_model


class MockModel:
    """Mock trained model."""

    def predict(self, X):
        return np.random.randint(0, 2, len(X))

    def predict_proba(self, X):
        proba = np.random.rand(len(X))
        return np.column_stack([1 - proba, proba])


@pytest.mark.asyncio
class TestEvaluateModel:
    """Test model evaluation."""

    async def test_evaluates_on_all_splits(self):
        """Should evaluate on train, validation, and test splits."""
        model = MockModel()
        state = {
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

        result = await evaluate_model(state)

        assert "error" not in result
        assert "train_metrics" in result
        assert "validation_metrics" in result
        assert "test_metrics" in result

    async def test_returns_classification_metrics(self):
        """Should return classification metrics for classification problems."""
        model = MockModel()
        state = {
            "trained_model": model,
            "problem_type": "binary_classification",
            "X_test_preprocessed": np.random.rand(20, 5),
            "test_data": {"y": np.random.randint(0, 2, 20)},
            "success_criteria": {},
        }

        result = await evaluate_model(state)

        assert result["auc_roc"] is not None
        assert result["precision"] is not None
        assert result["recall"] is not None
        assert result["f1_score"] is not None

    async def test_checks_success_criteria(self):
        """Should check if model meets success criteria."""
        model = MockModel()
        state = {
            "trained_model": model,
            "problem_type": "binary_classification",
            "X_test_preprocessed": np.random.rand(20, 5),
            "test_data": {"y": np.random.randint(0, 2, 20)},
            "success_criteria": {"accuracy": 0.90},  # Very high threshold
        }

        result = await evaluate_model(state)

        assert "success_criteria_met" in result
        assert "success_criteria_results" in result

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
