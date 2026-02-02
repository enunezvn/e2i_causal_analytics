"""Tests for SHAP computation node."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.agents.ml_foundation.feature_analyzer.nodes.shap_computer import (
    _select_explainer_type,
    compute_shap,
)


@pytest.mark.asyncio
class TestComputeSHAP:
    """Test SHAP computation node."""

    @pytest.fixture
    def mock_random_forest_model(self):
        """Create mock RandomForest model."""
        model = Mock()
        model.__class__.__name__ = "RandomForestClassifier"
        model.feature_names_in_ = ["feat_1", "feat_2", "feat_3", "feat_4", "feat_5"]
        model.n_features_in_ = 5
        model.predict = Mock(return_value=np.random.randint(0, 2, 100))
        model.predict_proba = Mock(return_value=np.random.rand(100, 2))
        return model

    @pytest.fixture
    def mock_linear_model(self):
        """Create mock Linear model."""
        model = Mock()
        model.__class__.__name__ = "LogisticRegression"
        model.feature_names_in_ = ["feat_1", "feat_2", "feat_3"]
        model.n_features_in_ = 3
        model.coef_ = np.random.rand(1, 3)
        model.intercept_ = np.array([0.5])
        return model

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.mlflow")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.shap")
    async def test_computes_shap_for_tree_model(
        self, mock_shap, mock_mlflow, mock_random_forest_model
    ):
        """Should compute SHAP values for tree-based models using TreeExplainer."""
        # Setup
        state = {
            "model_uri": "runs:/abc123/model",
            "experiment_id": "exp_001",
            "max_samples": 100,
        }

        mock_mlflow.sklearn.load_model.return_value = mock_random_forest_model
        mock_mlflow.get_run.return_value = Mock(
            info=Mock(run_id="abc123"), data=Mock(params={"model_version": "v1"})
        )

        # Mock TreeExplainer
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.random.rand(100, 5)
        mock_explainer.expected_value = 0.5
        mock_shap.TreeExplainer.return_value = mock_explainer

        # Execute
        result = await compute_shap(state)

        # Assert
        assert "error" not in result
        assert result["explainer_type"] == "TreeExplainer"
        assert "shap_values" in result
        assert "global_importance" in result
        assert "global_importance_ranked" in result
        assert "feature_directions" in result
        assert "top_features" in result
        assert len(result["top_features"]) == 5
        assert result["samples_analyzed"] == 100

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.mlflow")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.shap")
    async def test_computes_shap_for_linear_model(self, mock_shap, mock_mlflow, mock_linear_model):
        """Should compute SHAP values for linear models using LinearExplainer."""
        # Setup
        state = {
            "model_uri": "runs:/def456/model",
            "experiment_id": "exp_002",
            "max_samples": 50,
        }

        mock_mlflow.sklearn.load_model.return_value = mock_linear_model
        mock_mlflow.get_run.return_value = Mock(
            info=Mock(run_id="def456"), data=Mock(params={"model_version": "v2"})
        )

        # Mock LinearExplainer
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.random.rand(50, 3)
        mock_explainer.expected_value = 0.3
        mock_shap.LinearExplainer.return_value = mock_explainer

        # Execute
        result = await compute_shap(state)

        # Assert
        assert "error" not in result
        assert result["explainer_type"] == "LinearExplainer"
        assert "shap_values" in result
        assert len(result["top_features"]) == 3

    async def test_skips_when_missing_model_uri(self):
        """Should skip SHAP computation when model_uri is missing."""
        state = {
            "experiment_id": "exp_003",
        }

        result = await compute_shap(state)

        assert result["shap_skipped"] is True
        assert result["status"] == "skipped"

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.mlflow")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.shap")
    async def test_limits_sample_size(self, mock_shap, mock_mlflow, mock_random_forest_model):
        """Should limit sample size to max_samples."""
        # Setup with large dataset
        state = {
            "model_uri": "runs:/ghi789/model",
            "experiment_id": "exp_004",
            "max_samples": 50,
            "X_sample": np.random.rand(1000, 5),  # 1000 samples
        }

        mock_mlflow.sklearn.load_model.return_value = mock_random_forest_model
        mock_mlflow.get_run.return_value = Mock(info=Mock(run_id="ghi789"), data=Mock(params={}))

        # Mock TreeExplainer
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.random.rand(50, 5)
        mock_explainer.expected_value = 0.5
        mock_shap.TreeExplainer.return_value = mock_explainer

        # Execute
        result = await compute_shap(state)

        # Assert
        assert result["samples_analyzed"] == 50  # Limited to max_samples

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.mlflow")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.shap")
    async def test_determines_feature_directions(
        self, mock_shap, mock_mlflow, mock_random_forest_model
    ):
        """Should determine feature directions (positive/negative/mixed)."""
        # Setup
        state = {
            "model_uri": "runs:/jkl012/model",
            "experiment_id": "exp_005",
            "max_samples": 100,
        }

        mock_mlflow.sklearn.load_model.return_value = mock_random_forest_model
        mock_mlflow.get_run.return_value = Mock(info=Mock(run_id="jkl012"), data=Mock(params={}))

        # Mock TreeExplainer with controlled SHAP values
        shap_values = np.array(
            [
                [0.5, -0.3, 0.1, -0.1, 0.0],  # Positive, Negative, Mixed, Negative, Neutral
                [0.6, -0.4, -0.2, -0.05, 0.0],
                [0.4, -0.2, 0.3, -0.15, 0.0],
            ]
        )
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = shap_values
        mock_explainer.expected_value = 0.5
        mock_shap.TreeExplainer.return_value = mock_explainer

        # Execute
        result = await compute_shap(state)

        # Assert
        assert "feature_directions" in result
        assert result["feature_directions"]["feat_1"] == "positive"
        assert result["feature_directions"]["feat_2"] == "negative"

    def test_select_explainer_type_for_tree_models(self):
        """Should select TreeExplainer for tree-based models."""
        model = Mock()
        model.__class__.__name__ = "RandomForestClassifier"
        assert _select_explainer_type(model) == "TreeExplainer"

        model.__class__.__name__ = "XGBRegressor"
        assert _select_explainer_type(model) == "TreeExplainer"

        model.__class__.__name__ = "LGBMClassifier"
        assert _select_explainer_type(model) == "TreeExplainer"

    def test_select_explainer_type_for_linear_models(self):
        """Should select LinearExplainer for linear models."""
        model = Mock()
        model.__class__.__name__ = "LogisticRegression"
        assert _select_explainer_type(model) == "LinearExplainer"

        model.__class__.__name__ = "LinearRegression"
        assert _select_explainer_type(model) == "LinearExplainer"

        model.__class__.__name__ = "Ridge"
        assert _select_explainer_type(model) == "LinearExplainer"

    def test_select_explainer_type_fallback_to_kernel(self):
        """Should fallback to KernelExplainer for unknown model types."""
        model = Mock()
        model.__class__.__name__ = "CustomModel"
        assert _select_explainer_type(model) == "KernelExplainer"

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.mlflow")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.shap")
    async def test_generates_analysis_id(self, mock_shap, mock_mlflow, mock_random_forest_model):
        """Should generate unique analysis ID."""
        # Setup
        state = {
            "model_uri": "runs:/mno345/model",
            "experiment_id": "exp_006",
            "max_samples": 100,
        }

        mock_mlflow.sklearn.load_model.return_value = mock_random_forest_model
        mock_mlflow.get_run.return_value = Mock(info=Mock(run_id="mno345"), data=Mock(params={}))

        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.random.rand(100, 5)
        mock_explainer.expected_value = 0.5
        mock_shap.TreeExplainer.return_value = mock_explainer

        # Execute
        result = await compute_shap(state)

        # Assert
        assert "shap_analysis_id" in result
        assert result["shap_analysis_id"].startswith("shap_exp_006_")
