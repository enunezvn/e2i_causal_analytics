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
        """Should compute SHAP values for tree-based models using TreeExplainer.

        Uses SHAP >=0.42 return contract: 3-D ndarray (n, f, n_classes) for binary
        classifiers and length-2 expected_value array.
        """
        # Setup. With the C1 wiring, callers pre-populate ``loaded_model`` to
        # bypass the MLflow loader; we exercise that path here so the test
        # is independent of the MLflow flavor resolution code.
        state = {
            "model_uri": "runs:/abc123/model",
            "allow_synthetic_background": True,  # F8: synthetic-bg opt-in (mechanics test)
            "experiment_id": "exp_001",
            "max_samples": 100,
            "loaded_model": mock_random_forest_model,
        }

        mock_mlflow.get_run.return_value = Mock(
            info=Mock(run_id="abc123"), data=Mock(params={"model_version": "v1"})
        )

        # Mock TreeExplainer with SHAP 0.48 return shape:
        # shap_values -> (n_samples, n_features, n_classes); expected_value -> len-2 array
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.random.rand(100, 5, 2)
        mock_explainer.expected_value = np.array([0.4, 0.6])
        mock_shap.TreeExplainer.return_value = mock_explainer

        # Execute
        result = await compute_shap(state)

        # Assert
        assert "error" not in result
        assert result["explainer_type"] == "TreeExplainer"
        assert "shap_values" in result
        # Normalization must collapse the class axis to a 2-D (n, f) array
        assert result["shap_values"].ndim == 2
        assert result["shap_values"].shape == (100, 5)
        # base_value must be a Python float (not ndarray / np.number)
        assert isinstance(result["base_value"], float)
        assert "global_importance" in result
        # Every feature's importance should be a scalar float
        for _, value in result["global_importance"].items():
            assert isinstance(value, float)
        assert "global_importance_ranked" in result
        assert "feature_directions" in result
        assert "top_features" in result
        assert len(result["top_features"]) == 5
        assert result["samples_analyzed"] == 100

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.mlflow")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.shap")
    async def test_tree_model_legacy_list_return(
        self, mock_shap, mock_mlflow, mock_random_forest_model
    ):
        """Should handle legacy SHAP list return: [class_0_vals, class_1_vals]."""
        state = {
            "model_uri": "runs:/legacy123/model",
            "allow_synthetic_background": True,  # F8: synthetic-bg opt-in (mechanics test)
            "experiment_id": "exp_legacy",
            "max_samples": 100,
            "loaded_model": mock_random_forest_model,
        }

        mock_mlflow.get_run.return_value = Mock(info=Mock(run_id="legacy123"), data=Mock(params={}))

        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = [
            np.random.rand(100, 5),  # class 0
            np.random.rand(100, 5),  # class 1
        ]
        mock_explainer.expected_value = [0.4, 0.6]
        mock_shap.TreeExplainer.return_value = mock_explainer

        result = await compute_shap(state)

        assert "error" not in result
        assert result["shap_values"].ndim == 2
        assert result["shap_values"].shape == (100, 5)
        assert isinstance(result["base_value"], float)

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.mlflow")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.shap")
    async def test_linear_model_array_base_value(self, mock_shap, mock_mlflow, mock_linear_model):
        """Should handle LinearExplainer returning 2-D values and length-1 base_value array."""
        state = {
            "model_uri": "runs:/lin_arr/model",
            "allow_synthetic_background": True,  # F8: synthetic-bg opt-in (mechanics test)
            "experiment_id": "exp_lin_arr",
            "max_samples": 50,
            "loaded_model": mock_linear_model,
        }

        mock_mlflow.get_run.return_value = Mock(info=Mock(run_id="lin_arr"), data=Mock(params={}))

        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.random.rand(50, 3)
        mock_explainer.expected_value = np.array([0.3])  # length-1 array
        mock_shap.LinearExplainer.return_value = mock_explainer

        result = await compute_shap(state)

        assert "error" not in result
        assert result["explainer_type"] == "LinearExplainer"
        assert result["shap_values"].ndim == 2
        assert result["shap_values"].shape == (50, 3)
        assert isinstance(result["base_value"], float)
        assert result["base_value"] == pytest.approx(0.3)

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.mlflow")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.shap")
    async def test_computes_shap_for_linear_model(self, mock_shap, mock_mlflow, mock_linear_model):
        """Should compute SHAP values for linear models using LinearExplainer."""
        # Setup
        state = {
            "model_uri": "runs:/def456/model",
            "allow_synthetic_background": True,  # F8: synthetic-bg opt-in (mechanics test)
            "experiment_id": "exp_002",
            "max_samples": 50,
            "loaded_model": mock_linear_model,
        }

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
            "loaded_model": mock_random_forest_model,
        }

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
            "allow_synthetic_background": True,  # F8: synthetic-bg opt-in (mechanics test)
            "experiment_id": "exp_005",
            "max_samples": 100,
            "loaded_model": mock_random_forest_model,
        }

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
            "allow_synthetic_background": True,  # F8: synthetic-bg opt-in (mechanics test)
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

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.mlflow")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.shap")
    async def test_f8_fail_closed_when_no_real_sample(
        self, mock_shap, mock_mlflow, mock_random_forest_model
    ):
        """F8: with no real sample data and no synthetic opt-in, SHAP must fail CLOSED
        (skip) rather than compute importances over an np.random background."""
        state = {
            "model_uri": "runs:/f8nodata/model",
            "experiment_id": "exp_f8",
            "max_samples": 100,
            "loaded_model": mock_random_forest_model,
            # no X_sample / X_train_selected / X_train; allow_synthetic_background unset
        }
        mock_mlflow.get_run.return_value = Mock(info=Mock(run_id="f8nodata"), data=Mock(params={}))

        result = await compute_shap(state)

        assert result.get("shap_skipped") is True
        assert result.get("status") == "skipped"
        assert result.get("data_provenance") == "unavailable"
        assert "shap_analysis_id" not in result
        # No SHAP explainer should be constructed over fabricated data.
        mock_shap.TreeExplainer.assert_not_called()

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.mlflow")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.shap")
    async def test_f8_bridges_x_train_selected_real_data(
        self, mock_shap, mock_mlflow, mock_random_forest_model
    ):
        """F8: when X_sample is absent, real X_train_selected is bridged + used;
        provenance is 'real'."""
        state = {
            "model_uri": "runs:/f8real/model",
            "experiment_id": "exp_f8real",
            "max_samples": 100,
            "loaded_model": mock_random_forest_model,
            "X_train_selected": np.random.rand(80, 5),
        }
        mock_mlflow.get_run.return_value = Mock(info=Mock(run_id="f8real"), data=Mock(params={}))
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.random.rand(80, 5, 2)
        mock_explainer.expected_value = np.array([0.4, 0.6])
        mock_shap.TreeExplainer.return_value = mock_explainer

        result = await compute_shap(state)

        assert "error" not in result
        assert result.get("shap_skipped") is not True
        assert result["data_provenance"] == "real"
        assert result["samples_analyzed"] == 80

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.mlflow")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.shap")
    async def test_f8_synthetic_opt_in_is_labeled(
        self, mock_shap, mock_mlflow, mock_random_forest_model
    ):
        """F8: synthetic background, when explicitly opted into, is stamped
        data_provenance='synthetic' (never silently presented as real)."""
        state = {
            "model_uri": "runs:/f8syn/model",
            "experiment_id": "exp_f8syn",
            "max_samples": 100,
            "loaded_model": mock_random_forest_model,
            "allow_synthetic_background": True,
        }
        mock_mlflow.get_run.return_value = Mock(info=Mock(run_id="f8syn"), data=Mock(params={}))
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.random.rand(100, 5, 2)
        mock_explainer.expected_value = np.array([0.4, 0.6])
        mock_shap.TreeExplainer.return_value = mock_explainer

        result = await compute_shap(state)

        assert "error" not in result
        assert result["data_provenance"] == "synthetic"
