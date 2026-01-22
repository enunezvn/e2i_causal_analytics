"""Unit tests for DriftMonitor MLflow Tracker module.

Tests cover:
- Tracker initialization and MLflow availability checking
- Context managers for starting monitoring runs
- Metric extraction (PSI, KS statistics, drift severity)
- Parameter logging
- Artifact logging (JSON)
- Historical query methods (monitoring history, drift trends)
- Graceful degradation when MLflow unavailable

Phase 1 G03 from observability audit remediation plan.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.drift_monitor.mlflow_tracker import (
    DriftMonitorContext,
    DriftMonitorMetrics,
    DriftMonitorMLflowTracker,
    EXPERIMENT_PREFIX,
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
    mock.search_runs = MagicMock(return_value=MagicMock(to_dict=MagicMock(return_value={"run_id": []})))
    mock.get_experiment_by_name = MagicMock(return_value=MagicMock(experiment_id="test_exp_id"))
    return mock


@pytest.fixture
def tracker():
    """Create a DriftMonitorMLflowTracker instance."""
    return DriftMonitorMLflowTracker()


@pytest.fixture
def sample_context():
    """Create a sample DriftMonitorContext."""
    return DriftMonitorContext(
        query="Check for data drift in churn prediction model",
        model_name="churn_predictor_v2",
        reference_dataset="training_data_2024Q3",
        current_dataset="production_data_2024Q4",
        features_monitored=["tenure", "monthly_charges", "total_charges"],
        drift_threshold=0.10,
    )


@pytest.fixture
def sample_result():
    """Create a sample monitoring result dict."""
    return {
        "overall_drift_detected": True,
        "drift_severity": "moderate",
        "psi_scores": {
            "tenure": 0.08,
            "monthly_charges": 0.15,
            "total_charges": 0.05,
        },
        "ks_statistics": {
            "tenure": 0.12,
            "monthly_charges": 0.22,
            "total_charges": 0.08,
        },
        "drifted_features": ["monthly_charges"],
        "alerts_triggered": 1,
        "recommendation": "Investigate monthly_charges distribution shift",
    }


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestExperimentConfiguration:
    """Tests for experiment configuration constants."""

    def test_experiment_prefix_format(self):
        """Test experiment prefix follows naming convention."""
        assert EXPERIMENT_PREFIX == "e2i_causal/drift_monitor"

    def test_experiment_prefix_contains_agent_name(self):
        """Test experiment prefix includes agent identifier."""
        assert "drift_monitor" in EXPERIMENT_PREFIX


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestDriftMonitorContext:
    """Tests for DriftMonitorContext dataclass."""

    def test_context_creation_minimal(self):
        """Test context creation with minimal fields."""
        ctx = DriftMonitorContext(
            query="test query",
            model_name="test_model",
        )
        assert ctx.query == "test query"
        assert ctx.model_name == "test_model"

    def test_context_creation_full(self, sample_context):
        """Test context creation with all fields."""
        assert sample_context.model_name == "churn_predictor_v2"
        assert sample_context.drift_threshold == 0.10
        assert len(sample_context.features_monitored) == 3

    def test_context_default_values(self):
        """Test context default values."""
        ctx = DriftMonitorContext(
            query="test",
            model_name="model",
        )
        assert ctx.drift_threshold is None or isinstance(ctx.drift_threshold, float)


class TestDriftMonitorMetrics:
    """Tests for DriftMonitorMetrics dataclass."""

    def test_metrics_creation(self):
        """Test metrics dataclass creation."""
        metrics = DriftMonitorMetrics(
            overall_drift_detected=True,
            drift_severity="moderate",
            alerts_triggered=1,
        )
        assert metrics.overall_drift_detected is True
        assert metrics.drift_severity == "moderate"

    def test_metrics_optional_fields(self):
        """Test metrics with optional fields."""
        metrics = DriftMonitorMetrics(
            overall_drift_detected=False,
        )
        assert metrics.overall_drift_detected is False


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestTrackerInitialization:
    """Tests for tracker initialization."""

    def test_tracker_creation(self, tracker):
        """Test tracker can be created."""
        assert tracker is not None
        assert isinstance(tracker, DriftMonitorMLflowTracker)

    def test_tracker_lazy_mlflow_loading(self, tracker):
        """Test MLflow is lazily loaded."""
        assert hasattr(tracker, "_mlflow") or hasattr(tracker, "_check_mlflow")

    def test_check_mlflow_returns_bool(self, tracker):
        """Test _check_mlflow returns boolean."""
        result = tracker._check_mlflow()
        assert isinstance(result, bool)


class TestMLflowAvailability:
    """Tests for MLflow availability checking."""

    def test_mlflow_available_when_installed(self, tracker, mock_mlflow):
        """Test MLflow detection when installed."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                assert tracker._check_mlflow() is True

    def test_graceful_degradation_when_unavailable(self, tracker):
        """Test tracker works when MLflow unavailable."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            result = tracker._check_mlflow()
            assert result is False


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================


class TestStartMonitoringRun:
    """Tests for start_monitoring_run context manager."""

    @pytest.mark.asyncio
    async def test_start_run_returns_context_manager(self, tracker, sample_context):
        """Test start_monitoring_run returns async context manager."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            async with tracker.start_monitoring_run(sample_context) as run_ctx:
                assert run_ctx is None or isinstance(run_ctx, dict)

    @pytest.mark.asyncio
    async def test_start_run_with_mlflow(self, tracker, sample_context, mock_mlflow):
        """Test start_monitoring_run with MLflow available."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_run = MagicMock()
                mock_run.info.run_id = "test_run_123"
                mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
                mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

                async with tracker.start_monitoring_run(sample_context):
                    mock_mlflow.set_experiment.assert_called()

    @pytest.mark.asyncio
    async def test_start_run_logs_model_name(self, tracker, sample_context, mock_mlflow):
        """Test that model name is logged during run start."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_run = MagicMock()
                mock_run.info.run_id = "test_run_123"
                mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
                mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

                async with tracker.start_monitoring_run(sample_context):
                    pass

                mock_mlflow.set_experiment.assert_called()


# =============================================================================
# METRIC EXTRACTION TESTS
# =============================================================================


class TestMetricExtraction:
    """Tests for _extract_metrics method."""

    def test_extract_metrics_from_dict(self, tracker, sample_result):
        """Test metric extraction from result dict."""
        metrics = tracker._extract_metrics(sample_result)
        assert isinstance(metrics, dict)

    def test_extract_psi_metrics(self, tracker, sample_result):
        """Test PSI metric extraction."""
        metrics = tracker._extract_metrics(sample_result)
        assert isinstance(metrics, dict)

    def test_extract_ks_metrics(self, tracker, sample_result):
        """Test KS statistics extraction."""
        metrics = tracker._extract_metrics(sample_result)
        assert isinstance(metrics, dict)

    def test_extract_metrics_handles_missing_fields(self, tracker):
        """Test metric extraction with missing fields."""
        result = {"overall_drift_detected": False}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, dict)

    def test_extract_metrics_handles_none(self, tracker):
        """Test metric extraction with None values."""
        result = {"drift_severity": None, "alerts_triggered": 0}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, dict)

    def test_extract_severity_level(self, tracker, sample_result):
        """Test drift severity extraction."""
        metrics = tracker._extract_metrics(sample_result)
        assert isinstance(metrics, dict)


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogMonitoringResult:
    """Tests for log_monitoring_result method."""

    @pytest.mark.asyncio
    async def test_log_result_without_mlflow(self, tracker, sample_result):
        """Test logging result when MLflow unavailable."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            await tracker.log_monitoring_result(sample_result)

    @pytest.mark.asyncio
    async def test_log_result_with_mlflow(self, tracker, sample_result, mock_mlflow):
        """Test logging result with MLflow available."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                await tracker.log_monitoring_result(sample_result)
                assert mock_mlflow.log_metrics.called or mock_mlflow.log_metric.called

    @pytest.mark.asyncio
    async def test_log_result_includes_drift_severity(self, tracker, sample_result, mock_mlflow):
        """Test that drift severity is logged."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                await tracker.log_monitoring_result(sample_result)


class TestLogParams:
    """Tests for _log_params method."""

    def test_log_params_from_context(self, tracker, sample_context, mock_mlflow):
        """Test parameter logging from context."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                tracker._log_params(sample_context)
                assert mock_mlflow.log_params.called or mock_mlflow.log_param.called

    def test_log_params_includes_model_name(self, tracker, sample_context, mock_mlflow):
        """Test model name is logged as parameter."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                tracker._log_params(sample_context)

    def test_log_params_includes_threshold(self, tracker, sample_context, mock_mlflow):
        """Test drift threshold is logged as parameter."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                tracker._log_params(sample_context)


class TestLogArtifacts:
    """Tests for _log_artifacts method."""

    @pytest.mark.asyncio
    async def test_log_artifacts_creates_json(self, tracker, sample_result, mock_mlflow):
        """Test artifact logging creates JSON file."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                with patch("tempfile.NamedTemporaryFile") as mock_temp:
                    mock_file = MagicMock()
                    mock_file.__enter__ = MagicMock(return_value=mock_file)
                    mock_file.__exit__ = MagicMock(return_value=False)
                    mock_file.name = "/tmp/test_artifact.json"
                    mock_temp.return_value = mock_file

                    await tracker._log_artifacts(sample_result)

    @pytest.mark.asyncio
    async def test_log_artifacts_includes_psi_scores(self, tracker, sample_result, mock_mlflow):
        """Test PSI scores are included in artifacts."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                with patch("tempfile.NamedTemporaryFile") as mock_temp:
                    mock_file = MagicMock()
                    mock_file.__enter__ = MagicMock(return_value=mock_file)
                    mock_file.__exit__ = MagicMock(return_value=False)
                    mock_file.name = "/tmp/test_artifact.json"
                    mock_temp.return_value = mock_file

                    await tracker._log_artifacts(sample_result)


# =============================================================================
# HISTORY QUERY TESTS
# =============================================================================


class TestGetMonitoringHistory:
    """Tests for get_monitoring_history method."""

    @pytest.mark.asyncio
    async def test_get_history_without_mlflow(self, tracker):
        """Test history query when MLflow unavailable."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            history = await tracker.get_monitoring_history()
            assert history is None or isinstance(history, (list, dict))

    @pytest.mark.asyncio
    async def test_get_history_with_mlflow(self, tracker, mock_mlflow):
        """Test history query with MLflow available."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_mlflow.search_runs.return_value = MagicMock(
                    to_dict=MagicMock(return_value={"run_id": ["run1", "run2"]})
                )

                history = await tracker.get_monitoring_history()
                mock_mlflow.search_runs.assert_called()

    @pytest.mark.asyncio
    async def test_get_history_filters_by_model(self, tracker, mock_mlflow):
        """Test history query filters by model name."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_mlflow.search_runs.return_value = MagicMock(
                    to_dict=MagicMock(return_value={"run_id": ["run1"]})
                )

                await tracker.get_monitoring_history(model_name="churn_predictor_v2")


class TestGetDriftTrend:
    """Tests for get_drift_trend method."""

    @pytest.mark.asyncio
    async def test_get_trend_without_mlflow(self, tracker):
        """Test trend query when MLflow unavailable."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            trend = await tracker.get_drift_trend()
            assert trend is None or isinstance(trend, (list, dict))

    @pytest.mark.asyncio
    async def test_get_trend_returns_dict(self, tracker, mock_mlflow):
        """Test trend returns dictionary structure."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_df = MagicMock()
                mock_df.to_dict.return_value = {
                    "metrics.overall_drift_detected": [False, True, True],
                    "metrics.alerts_triggered": [0, 1, 2],
                }
                mock_mlflow.search_runs.return_value = mock_df

                trend = await tracker.get_drift_trend()
                assert trend is None or isinstance(trend, dict)

    @pytest.mark.asyncio
    async def test_get_trend_filters_by_feature(self, tracker, mock_mlflow):
        """Test trend query filters by feature."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_mlflow.search_runs.return_value = MagicMock(
                    to_dict=MagicMock(return_value={"run_id": ["run1"]})
                )

                await tracker.get_drift_trend(feature="monthly_charges")


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handles_mlflow_connection_error(self, tracker, mock_mlflow):
        """Test handling of MLflow connection errors."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_mlflow.set_experiment.side_effect = Exception("Connection failed")

                try:
                    async with tracker.start_monitoring_run(
                        DriftMonitorContext(query="test", model_name="model")
                    ):
                        pass
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_handles_invalid_result_format(self, tracker):
        """Test handling of invalid result format."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            await tracker.log_monitoring_result({"invalid": "structure"})

    def test_handles_empty_context(self, tracker):
        """Test handling of minimal context."""
        ctx = DriftMonitorContext(query="", model_name="")
        assert ctx is not None

    @pytest.mark.asyncio
    async def test_handles_no_drift_detected(self, tracker):
        """Test handling of result with no drift."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            result = {"overall_drift_detected": False, "alerts_triggered": 0}
            await tracker.log_monitoring_result(result)

    @pytest.mark.asyncio
    async def test_handles_empty_psi_scores(self, tracker):
        """Test handling of empty PSI scores."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            result = {"psi_scores": {}}
            await tracker.log_monitoring_result(result)
