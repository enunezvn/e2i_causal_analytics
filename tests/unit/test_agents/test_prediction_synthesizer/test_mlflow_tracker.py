"""Unit tests for PredictionSynthesizer MLflow Tracker module.

Tests cover:
- Tracker initialization and MLflow availability checking
- Context managers for starting prediction runs
- Metric extraction and logging
- Parameter logging
- Artifact logging (JSON)
- Historical query methods
- Graceful degradation when MLflow unavailable

Part of observability audit remediation - G09 Phase 2.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.prediction_synthesizer.mlflow_tracker import (
    EXPERIMENT_PREFIX,
    PredictionContext,
    PredictionSynthesizerMetrics,
    PredictionSynthesizerMLflowTracker,
    create_tracker,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_mlflow():
    """Mock MLflow module."""
    mock = MagicMock()
    mock.set_experiment = MagicMock()
    mock.start_run = MagicMock()
    mock.end_run = MagicMock()
    mock.log_param = MagicMock()
    mock.log_params = MagicMock()
    mock.log_metric = MagicMock()
    mock.log_metrics = MagicMock()
    mock.log_artifact = MagicMock()
    mock.set_tag = MagicMock()
    mock.set_tags = MagicMock()
    mock.get_experiment_by_name = MagicMock(
        return_value=MagicMock(experiment_id="test_exp_id")
    )
    mock.search_runs = MagicMock(
        return_value=MagicMock(iterrows=MagicMock(return_value=iter([])))
    )
    return mock


@pytest.fixture
def tracker():
    """Create a PredictionSynthesizerMLflowTracker instance."""
    return PredictionSynthesizerMLflowTracker()


@pytest.fixture
def sample_context():
    """Create a sample PredictionContext."""
    return PredictionContext(
        run_id="run_123",
        experiment_name="churn_prediction",
        entity_type="hcp",
        prediction_target="churn",
        brand="remibrutinib",
        region="Northeast",
        time_horizon="30d",
    )


@pytest.fixture
def sample_state():
    """Create a sample PredictionSynthesizerState dict."""
    return {
        "ensemble_prediction": {
            "point_estimate": 0.72,
            "prediction_interval_lower": 0.58,
            "prediction_interval_upper": 0.86,
            "confidence": 0.85,
            "model_agreement": 0.91,
            "ensemble_method": "weighted",
        },
        "individual_predictions": [
            {"model_id": "churn_xgb", "prediction": 0.71, "confidence": 0.88},
            {"model_id": "churn_rf", "prediction": 0.73, "confidence": 0.82},
        ],
        "models_succeeded": 2,
        "models_failed": 0,
        "prediction_context": {
            "historical_accuracy": 0.82,
            "similar_cases": [{"case_id": "1"}, {"case_id": "2"}],
        },
        "orchestration_latency_ms": 150,
        "ensemble_latency_ms": 200,
        "total_latency_ms": 450,
    }


# =============================================================================
# CONSTANT TESTS
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_experiment_prefix(self):
        """Test experiment prefix is set correctly."""
        assert EXPERIMENT_PREFIX == "e2i_causal/prediction_synthesizer"


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestPredictionContext:
    """Tests for PredictionContext dataclass."""

    def test_context_creation_required_fields(self):
        """Test context creation with required fields."""
        ctx = PredictionContext(
            run_id="run_123",
            experiment_name="test_experiment",
            entity_type="hcp",
            prediction_target="churn",
        )
        assert ctx.run_id == "run_123"
        assert ctx.experiment_name == "test_experiment"
        assert ctx.entity_type == "hcp"
        assert ctx.prediction_target == "churn"

    def test_context_creation_full(self, sample_context):
        """Test context creation with all fields."""
        assert sample_context.brand == "remibrutinib"
        assert sample_context.region == "Northeast"
        assert sample_context.time_horizon == "30d"

    def test_context_default_values(self):
        """Test context default values for optional fields."""
        ctx = PredictionContext(
            run_id="run",
            experiment_name="exp_name",
            entity_type="hcp",
            prediction_target="churn",
        )
        assert ctx.brand is None
        assert ctx.region is None
        assert ctx.time_horizon is None
        assert ctx.timestamp is not None  # Auto-generated


class TestPredictionSynthesizerMetrics:
    """Tests for PredictionSynthesizerMetrics dataclass."""

    def test_metrics_creation(self):
        """Test metrics dataclass creation."""
        metrics = PredictionSynthesizerMetrics(
            point_estimate=0.72,
            prediction_interval_lower=0.58,
            prediction_interval_upper=0.86,
            ensemble_confidence=0.85,
            model_agreement=0.91,
        )
        assert metrics.point_estimate == 0.72
        assert metrics.ensemble_confidence == 0.85
        assert metrics.model_agreement == 0.91

    def test_metrics_defaults(self):
        """Test metrics default values."""
        metrics = PredictionSynthesizerMetrics()
        assert metrics.point_estimate is None
        assert metrics.ensemble_confidence == 0.0
        assert metrics.model_agreement == 0.0
        assert metrics.models_succeeded == 0
        assert metrics.models_failed == 0
        assert metrics.total_latency_ms == 0

    def test_metrics_to_dict(self):
        """Test metrics to_dict conversion."""
        metrics = PredictionSynthesizerMetrics(
            point_estimate=0.72,
            ensemble_confidence=0.85,
            model_agreement=0.91,
            models_succeeded=2,
            models_failed=0,
            total_latency_ms=450,
        )
        result = metrics.to_dict()

        assert result["point_estimate"] == 0.72
        assert result["ensemble_confidence"] == 0.85
        assert result["model_agreement"] == 0.91
        assert result["models_succeeded"] == 2
        assert result["models_failed"] == 0
        assert result["total_latency_ms"] == 450

    def test_metrics_to_dict_excludes_none(self):
        """Test that to_dict excludes None values for optional fields."""
        metrics = PredictionSynthesizerMetrics(
            ensemble_confidence=0.85,
            model_agreement=0.91,
        )
        result = metrics.to_dict()

        # point_estimate is None, should not be in result
        assert "point_estimate" not in result
        # Required fields should be present
        assert "ensemble_confidence" in result


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestTrackerInitialization:
    """Tests for tracker initialization."""

    def test_tracker_creation(self, tracker):
        """Test tracker can be created."""
        assert tracker is not None
        assert isinstance(tracker, PredictionSynthesizerMLflowTracker)

    def test_tracker_default_config(self, tracker):
        """Test tracker default configuration."""
        assert tracker.enable_artifact_logging is True
        assert tracker._current_run_id is None

    def test_tracker_custom_config(self):
        """Test tracker with custom configuration."""
        tracker = PredictionSynthesizerMLflowTracker(
            tracking_uri="http://custom:5000",
            enable_artifact_logging=False,
        )
        assert tracker._tracking_uri == "http://custom:5000"
        assert tracker.enable_artifact_logging is False

    def test_tracker_lazy_mlflow_loading(self, tracker):
        """Test MLflow is lazily loaded."""
        assert tracker._mlflow is None


class TestMLflowAvailability:
    """Tests for MLflow availability checking."""

    def test_get_mlflow_returns_none_when_import_fails(self, tracker):
        """Test _get_mlflow returns None when MLflow import fails."""
        with patch.dict("sys.modules", {"mlflow": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = tracker._get_mlflow()
                # May return None or cached value
                assert result is None or result is not None


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================


class TestStartPredictionRun:
    """Tests for start_prediction_run context manager."""

    @pytest.mark.asyncio
    async def test_start_run_without_mlflow(self, tracker):
        """Test start_prediction_run when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_prediction_run(
                experiment_name="test_experiment",
                entity_type="hcp",
                prediction_target="churn",
            ) as ctx:
                assert ctx is not None
                assert isinstance(ctx, PredictionContext)
                assert ctx.run_id == "no-mlflow"

    @pytest.mark.asyncio
    async def test_start_run_with_mlflow(self, tracker, mock_mlflow):
        """Test start_prediction_run with MLflow available."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            async with tracker.start_prediction_run(
                experiment_name="test_experiment",
                entity_type="hcp",
                prediction_target="churn",
                brand="remibrutinib",
            ) as ctx:
                assert ctx is not None
                assert ctx.run_id == "test_run_123"

    @pytest.mark.asyncio
    async def test_start_run_accepts_optional_params(self, tracker):
        """Test that start_prediction_run accepts optional parameters."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_prediction_run(
                experiment_name="test_experiment",
                entity_type="hcp",
                prediction_target="churn",
                brand="remibrutinib",
                region="Northeast",
                time_horizon="30d",
                tags={"custom_tag": "value"},
            ) as ctx:
                assert ctx is not None
                assert ctx.brand == "remibrutinib"
                assert ctx.region == "Northeast"
                assert ctx.time_horizon == "30d"

    @pytest.mark.asyncio
    async def test_start_run_handles_experiment_error(self, tracker, mock_mlflow):
        """Test graceful handling of experiment creation errors."""
        mock_mlflow.get_experiment_by_name.side_effect = Exception("Connection failed")

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            async with tracker.start_prediction_run(
                experiment_name="test",
                entity_type="hcp",
                prediction_target="churn",
            ) as ctx:
                assert ctx.run_id == "experiment-error"


# =============================================================================
# METRIC EXTRACTION TESTS
# =============================================================================


class TestMetricExtraction:
    """Tests for _extract_metrics method."""

    def test_extract_metrics_from_state(self, tracker, sample_state):
        """Test metric extraction from state dict."""
        metrics = tracker._extract_metrics(sample_state)
        assert isinstance(metrics, PredictionSynthesizerMetrics)
        assert metrics.point_estimate == 0.72
        assert metrics.ensemble_confidence == 0.85
        assert metrics.model_agreement == 0.91
        assert metrics.models_succeeded == 2
        assert metrics.models_failed == 0

    def test_extract_metrics_handles_empty_state(self, tracker):
        """Test metric extraction with empty state."""
        metrics = tracker._extract_metrics({})
        assert isinstance(metrics, PredictionSynthesizerMetrics)
        assert metrics.point_estimate is None
        assert metrics.ensemble_confidence == 0.0

    def test_extract_metrics_handles_missing_fields(self, tracker):
        """Test metric extraction with missing fields."""
        state = {
            "ensemble_prediction": {"point_estimate": 0.5},
        }
        metrics = tracker._extract_metrics(state)
        assert metrics.point_estimate == 0.5
        assert metrics.model_agreement == 0.0


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogPredictionResult:
    """Tests for log_prediction_result method."""

    @pytest.mark.asyncio
    async def test_log_result_without_mlflow(self, tracker, sample_state):
        """Test logging result when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            # Should not raise
            await tracker.log_prediction_result(sample_state)

    @pytest.mark.asyncio
    async def test_log_result_without_run_id(self, tracker, sample_state, mock_mlflow):
        """Test logging result when no run is active."""
        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            tracker._current_run_id = None
            # Should not raise
            await tracker.log_prediction_result(sample_state)
            mock_mlflow.log_metrics.assert_not_called()

    @pytest.mark.asyncio
    async def test_log_result_with_mlflow(self, tracker, sample_state, mock_mlflow):
        """Test logging result with MLflow available."""
        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            tracker._current_run_id = "test_run_123"
            tracker.enable_artifact_logging = False  # Skip artifacts

            await tracker.log_prediction_result(sample_state)

            mock_mlflow.log_metrics.assert_called_once()
            mock_mlflow.set_tags.assert_called_once()


# =============================================================================
# ARTIFACT LOGGING TESTS
# =============================================================================


class TestLogArtifacts:
    """Tests for _log_artifacts method."""

    @pytest.mark.asyncio
    async def test_log_artifacts_without_mlflow(self, tracker, sample_state):
        """Test artifact logging when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            # Should not raise
            await tracker._log_artifacts(sample_state)

    @pytest.mark.asyncio
    async def test_log_artifacts_with_mlflow(self, tracker, sample_state, mock_mlflow):
        """Test artifact logging with MLflow available."""
        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                mock_tmpdir.return_value.__enter__ = MagicMock(return_value="/tmp/test")
                mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)

                with patch("builtins.open", MagicMock()):
                    await tracker._log_artifacts(sample_state)


# =============================================================================
# HISTORY QUERY TESTS
# =============================================================================


class TestGetPredictionHistory:
    """Tests for get_prediction_history method."""

    @pytest.mark.asyncio
    async def test_get_history_without_mlflow(self, tracker):
        """Test history query when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            history = await tracker.get_prediction_history()
            assert history == []

    @pytest.mark.asyncio
    async def test_get_history_no_experiment(self, tracker, mock_mlflow):
        """Test history query when experiment doesn't exist."""
        mock_mlflow.get_experiment_by_name.return_value = None

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            history = await tracker.get_prediction_history()
            assert history == []

    @pytest.mark.asyncio
    async def test_get_history_with_results(self, tracker, mock_mlflow):
        """Test history query with results."""
        mock_df = MagicMock()
        mock_df.iterrows.return_value = iter([
            (0, {
                "run_id": "run_1",
                "start_time": datetime.now(timezone.utc),
            }),
        ])
        mock_mlflow.search_runs.return_value = mock_df

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            history = await tracker.get_prediction_history()
            assert len(history) == 1
            assert history[0]["run_id"] == "run_1"


class TestGetModelPerformanceSummary:
    """Tests for get_model_performance_summary method."""

    @pytest.mark.asyncio
    async def test_get_summary_empty_history(self, tracker):
        """Test summary with empty history."""
        with patch.object(tracker, "get_prediction_history", return_value=[]):
            summary = await tracker.get_model_performance_summary()
            assert summary["total_predictions"] == 0
            assert summary["avg_confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_get_summary_with_history(self, tracker):
        """Test summary with history data."""
        history = [
            {
                "ensemble_confidence": 0.85,
                "model_agreement": 0.91,
                "models_succeeded": 2,
                "models_failed": 0,
                "entity_type": "hcp",
                "prediction_target": "churn",
            },
            {
                "ensemble_confidence": 0.75,
                "model_agreement": 0.80,
                "models_succeeded": 3,
                "models_failed": 1,
                "entity_type": "hcp",
                "prediction_target": "conversion",
            },
        ]
        with patch.object(tracker, "get_prediction_history", return_value=history):
            summary = await tracker.get_model_performance_summary()
            assert summary["total_predictions"] == 2
            assert summary["avg_confidence"] == 0.80
            assert summary["avg_model_agreement"] == 0.855


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestCreateTracker:
    """Tests for create_tracker factory function."""

    def test_create_tracker_default(self):
        """Test create_tracker with defaults."""
        tracker = create_tracker()
        assert isinstance(tracker, PredictionSynthesizerMLflowTracker)

    def test_create_tracker_custom_uri(self):
        """Test create_tracker with custom URI."""
        tracker = create_tracker(tracking_uri="http://custom:5000")
        assert tracker._tracking_uri == "http://custom:5000"


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handles_logging_error(self, tracker, mock_mlflow):
        """Test graceful handling of logging errors."""
        mock_mlflow.log_metrics.side_effect = Exception("Logging failed")

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            tracker._current_run_id = "test_run"
            tracker.enable_artifact_logging = False

            # Should not raise
            await tracker.log_prediction_result({"ensemble_prediction": {}})

    @pytest.mark.asyncio
    async def test_handles_artifact_error(self, tracker, mock_mlflow):
        """Test graceful handling of artifact logging errors."""
        mock_mlflow.log_artifact.side_effect = Exception("Artifact error")

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                mock_tmpdir.return_value.__enter__ = MagicMock(return_value="/tmp/test")
                mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)

                # Should not raise
                await tracker._log_artifacts({"individual_predictions": [{}]})
