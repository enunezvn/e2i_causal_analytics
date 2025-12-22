"""Tests for MLflow logging node.

Tests the log_to_mlflow function with mock MLflow connector.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sklearn.ensemble import RandomForestClassifier

import numpy as np

from src.agents.ml_foundation.model_trainer.nodes.mlflow_logger import (
    log_to_mlflow,
    _get_framework,
    _get_mlflow_flavor,
    _get_primary_metric,
)


# ============================================================================
# Test fixtures
# ============================================================================


class MockModel:
    """Mock trained model."""

    def predict(self, X):
        return np.random.randint(0, 2, len(X))


@pytest.fixture
def training_state():
    """Create state after model training."""
    np.random.seed(42)
    return {
        "trained_model": MockModel(),
        "experiment_id": "exp_001",
        "experiment_name": "test_experiment",
        "algorithm_name": "RandomForest",
        "problem_type": "binary_classification",
        "framework": "sklearn",
        "best_hyperparameters": {"n_estimators": 100, "max_depth": 5},
        "training_duration_seconds": 10.5,
        "early_stopped": False,
        "hpo_completed": True,
        "hpo_best_value": 0.85,
        "hpo_trials_run": 20,
        "evaluation_metrics": {
            "train_metrics": {"accuracy": 0.95, "roc_auc": 0.98},
            "validation_metrics": {"accuracy": 0.88, "roc_auc": 0.92},
            "test_metrics": {"accuracy": 0.85, "roc_auc": 0.90},
        },
        "enable_mlflow": True,
        "register_model": False,
    }


@pytest.fixture
def mock_mlflow_connector():
    """Create mock MLflow connector."""
    mock_conn = AsyncMock()
    mock_conn.get_or_create_experiment = AsyncMock(return_value="mlflow_exp_123")

    # Mock run context manager
    mock_run = AsyncMock()
    mock_run.run_id = "run_abc123"
    mock_run.log_params = AsyncMock()
    mock_run.log_metrics = AsyncMock()
    mock_run.log_model = AsyncMock(return_value="runs:/run_abc123/model")
    mock_run.log_artifact = AsyncMock()

    # Make the context manager async
    mock_run.__aenter__ = AsyncMock(return_value=mock_run)
    mock_run.__aexit__ = AsyncMock(return_value=None)

    mock_conn.start_run = MagicMock(return_value=mock_run)
    mock_conn.register_model = AsyncMock(return_value=None)

    return mock_conn


# ============================================================================
# Test log_to_mlflow function
# ============================================================================


@pytest.mark.asyncio
class TestLogToMlflow:
    """Test MLflow logging functionality."""

    async def test_logs_when_enabled(self, training_state, mock_mlflow_connector):
        """Should log to MLflow when enabled."""
        with patch(
            "src.mlops.mlflow_connector.get_mlflow_connector",
            return_value=mock_mlflow_connector,
        ):
            result = await log_to_mlflow(training_state)

        assert result["mlflow_status"] == "success"
        assert result["mlflow_run_id"] == "run_abc123"
        assert result["mlflow_experiment_id"] == "mlflow_exp_123"

    async def test_skips_when_disabled(self, training_state):
        """Should skip logging when mlflow disabled."""
        training_state["enable_mlflow"] = False

        result = await log_to_mlflow(training_state)

        assert result["mlflow_status"] == "disabled"
        assert result["mlflow_run_id"] is None

    async def test_skips_when_no_model(self, training_state):
        """Should skip logging when no trained model."""
        training_state["trained_model"] = None

        result = await log_to_mlflow(training_state)

        assert result["mlflow_status"] == "skipped"
        assert "error" in result

    async def test_logs_hyperparameters(self, training_state, mock_mlflow_connector):
        """Should log hyperparameters to MLflow."""
        with patch(
            "src.mlops.mlflow_connector.get_mlflow_connector",
            return_value=mock_mlflow_connector,
        ):
            await log_to_mlflow(training_state)

        # Verify log_params was called
        mock_run = mock_mlflow_connector.start_run.return_value
        mock_run.log_params.assert_called()

    async def test_logs_metrics(self, training_state, mock_mlflow_connector):
        """Should log metrics to MLflow."""
        with patch(
            "src.mlops.mlflow_connector.get_mlflow_connector",
            return_value=mock_mlflow_connector,
        ):
            await log_to_mlflow(training_state)

        # Verify log_metrics was called
        mock_run = mock_mlflow_connector.start_run.return_value
        mock_run.log_metrics.assert_called()

    async def test_logs_model_artifact(self, training_state, mock_mlflow_connector):
        """Should log model artifact to MLflow."""
        with patch(
            "src.mlops.mlflow_connector.get_mlflow_connector",
            return_value=mock_mlflow_connector,
        ):
            result = await log_to_mlflow(training_state)

        assert result["mlflow_model_uri"] == "runs:/run_abc123/model"

    async def test_registers_model_when_requested(
        self, training_state, mock_mlflow_connector
    ):
        """Should register model when requested."""
        training_state["register_model"] = True
        training_state["model_name"] = "my_model"

        mock_version = MagicMock()
        mock_version.version = "1"
        mock_mlflow_connector.register_model = AsyncMock(return_value=mock_version)

        with patch(
            "src.mlops.mlflow_connector.get_mlflow_connector",
            return_value=mock_mlflow_connector,
        ):
            result = await log_to_mlflow(training_state)

        assert result["mlflow_registered"] is True
        assert result["mlflow_model_version"] == "1"

    async def test_handles_import_error(self, training_state):
        """Should handle MLflow import error gracefully."""
        with patch(
            "src.mlops.mlflow_connector.get_mlflow_connector",
            side_effect=ImportError("MLflow not installed"),
        ):
            result = await log_to_mlflow(training_state)

        assert result["mlflow_status"] == "unavailable"
        assert "error" in result

    async def test_handles_logging_error(self, training_state, mock_mlflow_connector):
        """Should handle logging errors gracefully."""
        mock_mlflow_connector.get_or_create_experiment = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        with patch(
            "src.mlops.mlflow_connector.get_mlflow_connector",
            return_value=mock_mlflow_connector,
        ):
            result = await log_to_mlflow(training_state)

        assert result["mlflow_status"] == "failed"
        assert "error" in result

    async def test_uses_default_experiment_name(
        self, training_state, mock_mlflow_connector
    ):
        """Should use default experiment name if not provided."""
        del training_state["experiment_name"]

        with patch(
            "src.mlops.mlflow_connector.get_mlflow_connector",
            return_value=mock_mlflow_connector,
        ):
            await log_to_mlflow(training_state)

        mock_mlflow_connector.get_or_create_experiment.assert_called()


# ============================================================================
# Test helper functions
# ============================================================================


class TestGetFramework:
    """Test framework identification."""

    def test_identifies_sklearn_algorithms(self):
        """Should identify sklearn algorithms."""
        assert _get_framework("RandomForest") == "sklearn"
        assert _get_framework("LogisticRegression") == "sklearn"
        assert _get_framework("Ridge") == "sklearn"
        assert _get_framework("GradientBoosting") == "sklearn"
        assert _get_framework("ExtraTrees") == "sklearn"

    def test_identifies_xgboost(self):
        """Should identify XGBoost."""
        assert _get_framework("XGBoost") == "xgboost"

    def test_identifies_lightgbm(self):
        """Should identify LightGBM."""
        assert _get_framework("LightGBM") == "lightgbm"

    def test_identifies_econml(self):
        """Should identify EconML algorithms."""
        assert _get_framework("CausalForest") == "econml"
        assert _get_framework("LinearDML") == "econml"
        assert _get_framework("SLearner") == "econml"

    def test_returns_sklearn_for_unknown(self):
        """Should default to sklearn for unknown algorithms."""
        assert _get_framework("UnknownAlgorithm") == "sklearn"


class TestGetMlflowFlavor:
    """Test MLflow flavor selection."""

    def test_returns_xgboost_flavor(self):
        """Should return xgboost flavor for XGBoost."""
        assert _get_mlflow_flavor("XGBoost", "xgboost") == "xgboost"

    def test_returns_lightgbm_flavor(self):
        """Should return lightgbm flavor for LightGBM."""
        assert _get_mlflow_flavor("LightGBM", "lightgbm") == "lightgbm"

    def test_returns_sklearn_flavor_for_others(self):
        """Should return sklearn flavor for other algorithms."""
        assert _get_mlflow_flavor("RandomForest", "sklearn") == "sklearn"
        assert _get_mlflow_flavor("LogisticRegression", "sklearn") == "sklearn"


class TestGetPrimaryMetric:
    """Test primary metric selection."""

    def test_selects_roc_auc_for_binary_classification(self):
        """Should prefer roc_auc for binary classification."""
        metrics = {"roc_auc": 0.85, "f1": 0.80, "accuracy": 0.82}

        result = _get_primary_metric(metrics, "binary_classification")

        assert result == 0.85

    def test_selects_f1_weighted_for_multiclass(self):
        """Should prefer f1_weighted for multiclass."""
        metrics = {"f1_weighted": 0.78, "f1": 0.75, "accuracy": 0.80}

        result = _get_primary_metric(metrics, "multiclass_classification")

        assert result == 0.78

    def test_selects_r2_for_regression(self):
        """Should prefer r2 for regression."""
        metrics = {"r2": 0.92, "rmse": 0.15, "mae": 0.10}

        result = _get_primary_metric(metrics, "regression")

        assert result == 0.92

    def test_returns_none_for_empty_metrics(self):
        """Should return None for empty metrics."""
        result = _get_primary_metric({}, "binary_classification")

        assert result is None

    def test_falls_back_to_next_metric(self):
        """Should fall back if primary metric not available."""
        metrics = {"f1": 0.80, "accuracy": 0.82}  # No roc_auc

        result = _get_primary_metric(metrics, "binary_classification")

        assert result == 0.80  # Falls back to f1
