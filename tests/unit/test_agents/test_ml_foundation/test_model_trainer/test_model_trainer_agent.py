"""Integration tests for model_trainer agent."""

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.model_trainer.agent import ModelTrainerAgent


@pytest.mark.asyncio
class TestModelTrainerAgent:
    """Integration tests for complete model training workflow."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return ModelTrainerAgent()

    @pytest.fixture
    def valid_input_data(self):
        """Create valid input data for training."""
        return {
            "model_candidate": {
                "algorithm_name": "RandomForest",
                "algorithm_class": "sklearn.ensemble.RandomForestClassifier",
                "hyperparameter_search_space": {
                    "n_estimators": {"type": "int", "low": 10, "high": 50},
                },
                "default_hyperparameters": {"n_estimators": 10, "max_depth": 5},
            },
            "qc_report": {
                "qc_passed": True,
                "overall_score": 0.92,
                "qc_errors": [],
                "qc_warnings": [],
            },
            "experiment_id": "exp_test_123",
            "success_criteria": {"accuracy": 0.70},
            "problem_type": "binary_classification",
            "enable_hpo": False,
            "early_stopping": False,
            # Pre-loaded splits
            "train_data": {
                "X": pd.DataFrame(np.random.rand(600, 10)),
                "y": np.random.randint(0, 2, 600),
                "row_count": 600,
            },
            "validation_data": {
                "X": pd.DataFrame(np.random.rand(200, 10)),
                "y": np.random.randint(0, 2, 200),
                "row_count": 200,
            },
            "test_data": {
                "X": pd.DataFrame(np.random.rand(150, 10)),
                "y": np.random.randint(0, 2, 150),
                "row_count": 150,
            },
            "holdout_data": {
                "X": pd.DataFrame(np.random.rand(50, 10)),
                "y": np.random.randint(0, 2, 50),
                "row_count": 50,
            },
            # Disable MLflow for unit tests
            "enable_mlflow": False,
            "enable_checkpointing": False,
        }

    async def test_complete_training_workflow(self, agent, valid_input_data):
        """Should complete full training workflow successfully."""
        result = await agent.run(valid_input_data)

        assert result["training_status"] == "completed"
        assert result["trained_model"] is not None
        assert "test_metrics" in result
        assert "train_metrics" in result
        assert "validation_metrics" in result

    async def test_validates_required_fields(self, agent):
        """Should validate required input fields."""
        with pytest.raises(ValueError, match="Missing required field"):
            await agent.run({})

    async def test_validates_model_candidate_structure(self, agent, valid_input_data):
        """Should validate model_candidate has required fields."""
        invalid_input = {
            **valid_input_data,
            "model_candidate": {"algorithm_name": "RandomForest"},  # Missing other fields
        }

        with pytest.raises(ValueError, match="model_candidate missing required field"):
            await agent.run(invalid_input)

    async def test_qc_gate_blocks_training(self, agent, valid_input_data):
        """Should block training when QC failed."""
        invalid_input = {
            **valid_input_data,
            "qc_report": {
                "qc_passed": False,
                "overall_score": 0.45,
                "qc_errors": ["Data quality too low"],
            },
        }

        with pytest.raises(RuntimeError, match="qc_gate_blocked"):
            await agent.run(invalid_input)

    async def test_returns_classification_metrics(self, agent, valid_input_data):
        """Should return classification metrics."""
        result = await agent.run(valid_input_data)

        assert "auc_roc" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result

    async def test_returns_training_metadata(self, agent, valid_input_data):
        """Should return training metadata."""
        result = await agent.run(valid_input_data)

        assert "training_run_id" in result
        assert "model_id" in result
        assert "algorithm_name" in result
        assert "best_hyperparameters" in result
        assert "training_duration_seconds" in result

    async def test_checks_success_criteria(self, agent, valid_input_data):
        """Should check if model meets success criteria."""
        result = await agent.run(valid_input_data)

        assert "success_criteria_met" in result
        assert "success_criteria_results" in result

    async def test_split_validation(self, agent, valid_input_data):
        """Should validate split ratios."""
        # Modify splits to invalid ratios
        invalid_input = {
            **valid_input_data,
            "train_data": {
                "X": pd.DataFrame(np.random.rand(500, 10)),
                "y": np.random.randint(0, 2, 500),
                "row_count": 500,  # 50% instead of 60%
            },
        }

        with pytest.raises(RuntimeError):
            await agent.run(invalid_input)

    async def test_agent_properties(self, agent):
        """Should have correct agent properties."""
        assert agent.tier == 0
        assert agent.tier_name == "ml_foundation"
        assert agent.agent_type == "standard"

    async def test_returns_split_information(self, agent, valid_input_data):
        """Should return split sample counts."""
        result = await agent.run(valid_input_data)

        assert result["train_samples"] == 600
        assert result["validation_samples"] == 200
        assert result["test_samples"] == 150
        assert result["total_samples"] == 1000
