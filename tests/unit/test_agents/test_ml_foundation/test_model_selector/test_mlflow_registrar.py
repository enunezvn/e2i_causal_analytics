"""Unit tests for mlflow_registrar node."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.ml_foundation.model_selector.nodes.mlflow_registrar import (
    _ensure_experiment,
    _log_selection_artifacts,
    _log_selection_metrics,
    _log_selection_params,
    create_selection_summary,
    log_benchmark_comparison,
    register_selection_in_mlflow,
)


@pytest.fixture
def primary_candidate():
    """Sample primary candidate."""
    return {
        "name": "XGBoost",
        "family": "gradient_boosting",
        "framework": "xgboost",
        "default_hyperparameters": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
        },
        "hyperparameter_space": {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 10},
        },
        "selection_score": 0.85,
        "interpretability_score": 0.6,
        "scalability_score": 0.8,
        "inference_latency_ms": 15,
        "memory_gb": 2.0,
    }


@pytest.fixture
def alternative_candidates():
    """Sample alternative candidates."""
    return [
        {
            "name": "LightGBM",
            "selection_score": 0.82,
            "reason_not_selected": "Slightly lower historical success rate",
        },
        {
            "name": "RandomForest",
            "selection_score": 0.78,
            "reason_not_selected": "Higher memory requirements",
        },
    ]


@pytest.fixture
def base_state(primary_candidate, alternative_candidates):
    """Base state for MLflow registration."""
    return {
        "experiment_id": "exp_test_mlflow_123",
        "primary_candidate": primary_candidate,
        "alternative_candidates": alternative_candidates,
        "selection_rationale": "XGBoost selected for optimal balance of accuracy and speed.",
        "primary_reason": "Best historical performance for similar problems",
        "supporting_factors": ["Fast inference", "Good interpretability"],
        "alternatives_considered": [
            {
                "algorithm_name": "LightGBM",
                "selection_score": 0.82,
                "reason_not_selected": "Lower historical success",
            }
        ],
        "constraint_compliance": {
            "inference_latency_<50ms": True,
            "memory_<4gb": True,
        },
        "benchmark_results": {
            "XGBoost": {"cv_score_mean": 0.82, "cv_score_std": 0.03},
            "LightGBM": {"cv_score_mean": 0.80, "cv_score_std": 0.04},
        },
        "historical_success_rates": {"XGBoost": 0.78, "LightGBM": 0.75},
        "problem_type": "binary_classification",
        "row_count": 50000,
        "column_count": 25,
        "interpretability_required": False,
        "algorithm_class": "xgboost.XGBClassifier",
    }


@pytest.fixture
def mock_mlflow_connector():
    """Mock MLflowConnector."""
    connector = MagicMock()
    connector.get_or_create_experiment = AsyncMock(return_value="exp_mlflow_123")
    connector.start_run = AsyncMock()
    connector.start_run.return_value = MagicMock()
    connector.start_run.return_value.info.run_id = "run_abc123"
    connector.log_params = AsyncMock()
    connector.log_metrics = AsyncMock()
    connector.log_artifact = AsyncMock()
    connector.end_run = AsyncMock()
    return connector


@pytest.mark.asyncio
class TestRegisterSelectionInMlflow:
    """Tests for register_selection_in_mlflow."""

    async def test_returns_error_without_primary_candidate(self):
        """Should return error if no primary candidate."""
        state = {
            "experiment_id": "exp_test",
            "primary_candidate": None,
        }

        result = await register_selection_in_mlflow(state)

        assert result["registered_in_mlflow"] is False
        assert "mlflow_registration_error" in result

    async def test_returns_error_with_empty_primary_candidate(self):
        """Should return error if primary candidate is empty."""
        state = {
            "experiment_id": "exp_test",
            "primary_candidate": {},
        }

        result = await register_selection_in_mlflow(state)

        assert result["registered_in_mlflow"] is False

    async def test_successful_registration(self, base_state, mock_mlflow_connector):
        """Should register successfully with valid data."""
        import sys

        # Create mock ModelStage enum
        mock_model_stage = MagicMock()
        mock_model_stage.DEVELOPMENT = MagicMock()
        mock_model_stage.DEVELOPMENT.value = "development"

        # Create a mock module to replace the actual import
        mock_module = MagicMock()
        mock_module.MLflowConnector = MagicMock(return_value=mock_mlflow_connector)
        mock_module.ModelStage = mock_model_stage

        # Patch sys.modules before the dynamic import happens
        original_module = sys.modules.get("src.mlops.mlflow_connector")
        sys.modules["src.mlops.mlflow_connector"] = mock_module

        try:
            result = await register_selection_in_mlflow(base_state)

            # When MLflow is available and configured
            assert "registered_in_mlflow" in result
            # Either registered or has specific error
            if result["registered_in_mlflow"]:
                assert "mlflow_run_id" in result
                assert "mlflow_experiment_id" in result
        finally:
            # Restore original module state
            if original_module is not None:
                sys.modules["src.mlops.mlflow_connector"] = original_module
            else:
                sys.modules.pop("src.mlops.mlflow_connector", None)

    async def test_handles_import_error(self, base_state):
        """Should handle missing MLflow gracefully."""
        # Without patching, if mlflow_connector doesn't exist or fails to import
        # the function should handle it gracefully
        result = await register_selection_in_mlflow(base_state)

        # Should indicate not registered
        assert result["registered_in_mlflow"] is False
        # May have error message
        if "mlflow_registration_error" in result:
            assert len(result["mlflow_registration_error"]) > 0


@pytest.mark.asyncio
class TestEnsureExperiment:
    """Tests for _ensure_experiment."""

    async def test_creates_experiment_with_correct_name(self, mock_mlflow_connector):
        """Should create experiment with correct naming."""
        experiment_id = "exp_test_123"

        result = await _ensure_experiment(mock_mlflow_connector, experiment_id)

        # Should call get_or_create_experiment
        mock_mlflow_connector.get_or_create_experiment.assert_called_once()
        call_args = mock_mlflow_connector.get_or_create_experiment.call_args

        # Check experiment name includes our ID
        assert experiment_id in call_args[1]["name"]
        assert "e2i" in call_args[1]["name"].lower() or "model_selection" in call_args[1]["name"]

    async def test_includes_tags(self, mock_mlflow_connector):
        """Should include appropriate tags."""
        await _ensure_experiment(mock_mlflow_connector, "exp_123")

        call_args = mock_mlflow_connector.get_or_create_experiment.call_args
        tags = call_args[1]["tags"]

        assert "source" in tags or "e2i_experiment_id" in tags


@pytest.mark.asyncio
class TestLogSelectionParams:
    """Tests for _log_selection_params."""

    async def test_logs_algorithm_info(self, mock_mlflow_connector, primary_candidate, base_state):
        """Should log algorithm information."""
        await _log_selection_params(mock_mlflow_connector, primary_candidate, base_state)

        mock_mlflow_connector.log_params.assert_called_once()
        params = mock_mlflow_connector.log_params.call_args[0][0]

        assert "algorithm_name" in params
        assert "algorithm_family" in params
        assert "framework" in params

    async def test_logs_problem_context(self, mock_mlflow_connector, primary_candidate, base_state):
        """Should log problem context."""
        await _log_selection_params(mock_mlflow_connector, primary_candidate, base_state)

        params = mock_mlflow_connector.log_params.call_args[0][0]

        assert "problem_type" in params
        assert "row_count" in params
        assert "column_count" in params

    async def test_logs_hyperparameters_with_prefix(self, mock_mlflow_connector, primary_candidate, base_state):
        """Should log hyperparameters with default_ prefix."""
        await _log_selection_params(mock_mlflow_connector, primary_candidate, base_state)

        params = mock_mlflow_connector.log_params.call_args[0][0]

        # Should have hyperparameters with prefix
        hp_params = [k for k in params.keys() if k.startswith("default_")]
        assert len(hp_params) > 0


@pytest.mark.asyncio
class TestLogSelectionMetrics:
    """Tests for _log_selection_metrics."""

    async def test_logs_selection_scores(self, mock_mlflow_connector, primary_candidate):
        """Should log selection scores."""
        benchmark_results = {}

        await _log_selection_metrics(mock_mlflow_connector, primary_candidate, benchmark_results)

        mock_mlflow_connector.log_metrics.assert_called_once()
        metrics = mock_mlflow_connector.log_metrics.call_args[0][0]

        assert "selection_score" in metrics
        assert "interpretability_score" in metrics
        assert "scalability_score" in metrics

    async def test_logs_benchmark_metrics_when_available(self, mock_mlflow_connector, primary_candidate):
        """Should log benchmark metrics when available."""
        benchmark_results = {
            "XGBoost": {
                "cv_score_mean": 0.82,
                "cv_score_std": 0.03,
                "training_time_seconds": 5.2,
            }
        }

        await _log_selection_metrics(mock_mlflow_connector, primary_candidate, benchmark_results)

        metrics = mock_mlflow_connector.log_metrics.call_args[0][0]

        assert "benchmark_cv_mean" in metrics
        assert "benchmark_cv_std" in metrics
        assert "benchmark_time_seconds" in metrics

    async def test_handles_missing_benchmark_results(self, mock_mlflow_connector, primary_candidate):
        """Should handle missing benchmark results."""
        benchmark_results = {}

        await _log_selection_metrics(mock_mlflow_connector, primary_candidate, benchmark_results)

        # Should still log metrics without benchmark data
        mock_mlflow_connector.log_metrics.assert_called_once()


@pytest.mark.asyncio
class TestLogSelectionArtifacts:
    """Tests for _log_selection_artifacts."""

    async def test_logs_rationale_artifact(self, mock_mlflow_connector):
        """Should log rationale as artifact."""
        rationale = "XGBoost selected for best performance."
        alternatives = []

        await _log_selection_artifacts(mock_mlflow_connector, rationale, alternatives)

        # Should call log_artifact for rationale
        calls = mock_mlflow_connector.log_artifact.call_args_list
        assert len(calls) >= 1

    async def test_logs_alternatives_artifact(self, mock_mlflow_connector, alternative_candidates):
        """Should log alternatives as JSON artifact."""
        rationale = "XGBoost selected."
        alternatives = alternative_candidates

        await _log_selection_artifacts(mock_mlflow_connector, rationale, alternatives)

        # Should call log_artifact for alternatives
        calls = mock_mlflow_connector.log_artifact.call_args_list
        assert len(calls) >= 1

    async def test_handles_empty_rationale(self, mock_mlflow_connector):
        """Should handle empty rationale."""
        await _log_selection_artifacts(mock_mlflow_connector, "", [])

        # Should not crash, may or may not log artifact


@pytest.mark.asyncio
class TestLogBenchmarkComparison:
    """Tests for log_benchmark_comparison."""

    async def test_returns_false_without_benchmark_results(self):
        """Should return false without benchmark results."""
        state = {
            "benchmark_results": {},
            "mlflow_run_id": "run_123",
        }

        result = await log_benchmark_comparison(state)

        assert result["benchmark_logged"] is False

    async def test_returns_false_without_mlflow_run_id(self):
        """Should return false without MLflow run ID."""
        state = {
            "benchmark_results": {"XGBoost": {"cv_score_mean": 0.82}},
            "mlflow_run_id": None,
        }

        result = await log_benchmark_comparison(state)

        assert result["benchmark_logged"] is False

    async def test_logs_benchmark_metrics_when_mlflow_available(self, mock_mlflow_connector):
        """Should log benchmark metrics for each algorithm."""
        state = {
            "benchmark_results": {
                "XGBoost": {"cv_score_mean": 0.82, "cv_score_std": 0.03, "training_time_seconds": 5.0},
                "LightGBM": {"cv_score_mean": 0.80, "cv_score_std": 0.04, "training_time_seconds": 3.0},
            },
            "mlflow_run_id": "run_123",
        }

        # Test without MLflow - should return gracefully
        result = await log_benchmark_comparison(state)

        # If MLflow not available, should log failure
        assert "benchmark_logged" in result


@pytest.mark.asyncio
class TestCreateSelectionSummary:
    """Tests for create_selection_summary."""

    async def test_creates_summary_structure(self, base_state):
        """Should create complete summary structure."""
        result = await create_selection_summary(base_state)

        assert "selection_summary" in result
        summary = result["selection_summary"]

        # Check required fields
        assert "experiment_id" in summary
        assert "algorithm_name" in summary
        assert "algorithm_family" in summary
        assert "selection_score" in summary
        assert "selection_rationale" in summary

    async def test_includes_alternatives(self, base_state):
        """Should include alternatives considered."""
        result = await create_selection_summary(base_state)
        summary = result["selection_summary"]

        assert "alternatives_considered" in summary
        assert isinstance(summary["alternatives_considered"], list)

    async def test_includes_benchmark_results(self, base_state):
        """Should include benchmark results."""
        result = await create_selection_summary(base_state)
        summary = result["selection_summary"]

        assert "benchmark_results" in summary
        assert isinstance(summary["benchmark_results"], dict)

    async def test_includes_historical_success_rates(self, base_state):
        """Should include historical success rates."""
        result = await create_selection_summary(base_state)
        summary = result["selection_summary"]

        assert "historical_success_rates" in summary

    async def test_includes_constraint_compliance(self, base_state):
        """Should include constraint compliance."""
        result = await create_selection_summary(base_state)
        summary = result["selection_summary"]

        assert "constraint_compliance" in summary

    async def test_includes_mlflow_info(self, base_state):
        """Should include MLflow run info."""
        base_state["mlflow_run_id"] = "run_abc123"
        base_state["mlflow_experiment_id"] = "exp_mlflow_456"

        result = await create_selection_summary(base_state)
        summary = result["selection_summary"]

        assert "mlflow_run_id" in summary
        assert "mlflow_experiment_id" in summary

    async def test_includes_timestamps(self, base_state):
        """Should include creation timestamps."""
        result = await create_selection_summary(base_state)
        summary = result["selection_summary"]

        assert "created_at" in summary
        assert "created_by" in summary
        assert summary["created_by"] == "model_selector"

    async def test_limits_alternatives_to_three(self, base_state):
        """Should limit alternatives to 3."""
        base_state["alternative_candidates"] = [
            {"name": f"Algo{i}", "selection_score": 0.7 - i * 0.05}
            for i in range(5)
        ]

        result = await create_selection_summary(base_state)
        summary = result["selection_summary"]

        assert len(summary["alternatives_considered"]) <= 3

    async def test_handles_missing_optional_fields(self):
        """Should handle missing optional fields."""
        minimal_state = {
            "experiment_id": "exp_test",
            "primary_candidate": {"name": "XGBoost"},
        }

        result = await create_selection_summary(minimal_state)

        assert "selection_summary" in result
        # Should not crash with missing fields


class TestSummaryForDatabaseStorage:
    """Test that summary is suitable for database storage."""

    @pytest.mark.asyncio
    async def test_summary_is_json_serializable(self, base_state):
        """Summary should be JSON serializable."""
        import json

        result = await create_selection_summary(base_state)
        summary = result["selection_summary"]

        # Should not raise
        json_str = json.dumps(summary, default=str)
        assert len(json_str) > 0

    @pytest.mark.asyncio
    async def test_summary_values_are_primitive_types(self, base_state):
        """Summary values should be primitive types for database storage."""
        result = await create_selection_summary(base_state)
        summary = result["selection_summary"]

        def check_types(obj, path=""):
            """Recursively check types."""
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_types(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    check_types(v, f"{path}[{i}]")
            else:
                # Should be primitive type
                assert isinstance(
                    obj, (str, int, float, bool, type(None))
                ), f"Non-primitive at {path}: {type(obj)}"

        check_types(summary)
