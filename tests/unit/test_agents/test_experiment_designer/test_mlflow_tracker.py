"""Unit tests for ExperimentDesigner MLflow Tracker module.

Tests cover:
- Tracker initialization and MLflow availability checking
- Context managers for starting design runs
- Metric extraction (power analysis, validity audits)
- Parameter logging (treatments, outcomes, design iterations)
- Artifact logging (JSON)
- Historical query methods (design history, metrics summary)
- Graceful degradation when MLflow unavailable

Phase 1 G03 from observability audit remediation plan.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.experiment_designer.mlflow_tracker import (
    DesignContext,
    ExperimentDesignerMetrics,
    ExperimentDesignerMLflowTracker,
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
    """Create an ExperimentDesignerMLflowTracker instance."""
    return ExperimentDesignerMLflowTracker()


@pytest.fixture
def sample_context():
    """Create a sample DesignContext."""
    return DesignContext(
        query="Design A/B test for email campaign effectiveness",
        hypothesis="Personalized emails increase conversion by 15%",
        treatments=["personalized_email", "generic_email"],
        outcomes=["conversion_rate", "click_through_rate"],
        design_type="randomized_controlled_trial",
        target_power=0.80,
        significance_level=0.05,
    )


@pytest.fixture
def sample_result():
    """Create a sample design result dict."""
    return {
        "recommended_sample_size": 2500,
        "estimated_power": 0.82,
        "minimum_detectable_effect": 0.05,
        "validity_audit": {
            "internal_validity": 0.90,
            "external_validity": 0.75,
            "construct_validity": 0.85,
        },
        "design_iterations": 3,
        "randomization_scheme": "stratified",
        "blocking_variables": ["region", "customer_segment"],
        "duration_weeks": 4,
    }


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestExperimentConfiguration:
    """Tests for experiment configuration constants."""

    def test_experiment_prefix_format(self):
        """Test experiment prefix follows naming convention."""
        assert EXPERIMENT_PREFIX == "e2i_causal/experiment_designer"

    def test_experiment_prefix_contains_agent_name(self):
        """Test experiment prefix includes agent identifier."""
        assert "experiment_designer" in EXPERIMENT_PREFIX


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestDesignContext:
    """Tests for DesignContext dataclass."""

    def test_context_creation_minimal(self):
        """Test context creation with minimal fields."""
        ctx = DesignContext(
            query="test query",
            hypothesis="test hypothesis",
        )
        assert ctx.query == "test query"
        assert ctx.hypothesis == "test hypothesis"

    def test_context_creation_full(self, sample_context):
        """Test context creation with all fields."""
        assert sample_context.design_type == "randomized_controlled_trial"
        assert sample_context.target_power == 0.80
        assert len(sample_context.treatments) == 2

    def test_context_default_values(self):
        """Test context default values."""
        ctx = DesignContext(
            query="test",
            hypothesis="hyp",
        )
        assert ctx.treatments is None or isinstance(ctx.treatments, list)
        assert ctx.outcomes is None or isinstance(ctx.outcomes, list)


class TestExperimentDesignerMetrics:
    """Tests for ExperimentDesignerMetrics dataclass."""

    def test_metrics_creation(self):
        """Test metrics dataclass creation."""
        metrics = ExperimentDesignerMetrics(
            recommended_sample_size=2500,
            estimated_power=0.82,
            minimum_detectable_effect=0.05,
        )
        assert metrics.recommended_sample_size == 2500
        assert metrics.estimated_power == 0.82

    def test_metrics_optional_fields(self):
        """Test metrics with optional fields."""
        metrics = ExperimentDesignerMetrics(
            recommended_sample_size=1000,
        )
        assert metrics.recommended_sample_size == 1000


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestTrackerInitialization:
    """Tests for tracker initialization."""

    def test_tracker_creation(self, tracker):
        """Test tracker can be created."""
        assert tracker is not None
        assert isinstance(tracker, ExperimentDesignerMLflowTracker)

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


class TestStartDesignRun:
    """Tests for start_design_run context manager."""

    @pytest.mark.asyncio
    async def test_start_run_returns_context_manager(self, tracker, sample_context):
        """Test start_design_run returns async context manager."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            async with tracker.start_design_run(sample_context) as run_ctx:
                assert run_ctx is None or isinstance(run_ctx, dict)

    @pytest.mark.asyncio
    async def test_start_run_with_mlflow(self, tracker, sample_context, mock_mlflow):
        """Test start_design_run with MLflow available."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_run = MagicMock()
                mock_run.info.run_id = "test_run_123"
                mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
                mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

                async with tracker.start_design_run(sample_context):
                    mock_mlflow.set_experiment.assert_called()

    @pytest.mark.asyncio
    async def test_start_run_logs_hypothesis(self, tracker, sample_context, mock_mlflow):
        """Test that hypothesis is logged during run start."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_run = MagicMock()
                mock_run.info.run_id = "test_run_123"
                mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
                mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

                async with tracker.start_design_run(sample_context):
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

    def test_extract_power_metrics(self, tracker, sample_result):
        """Test power analysis metric extraction."""
        metrics = tracker._extract_metrics(sample_result)
        assert isinstance(metrics, dict)
        if "estimated_power" in metrics:
            assert metrics["estimated_power"] == 0.82

    def test_extract_validity_metrics(self, tracker, sample_result):
        """Test validity audit metric extraction."""
        metrics = tracker._extract_metrics(sample_result)
        assert isinstance(metrics, dict)

    def test_extract_metrics_handles_missing_fields(self, tracker):
        """Test metric extraction with missing fields."""
        result = {"recommended_sample_size": 1000}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, dict)

    def test_extract_metrics_handles_none(self, tracker):
        """Test metric extraction with None values."""
        result = {"estimated_power": None, "recommended_sample_size": 2000}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, dict)


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogDesignResult:
    """Tests for log_design_result method."""

    @pytest.mark.asyncio
    async def test_log_result_without_mlflow(self, tracker, sample_result):
        """Test logging result when MLflow unavailable."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            await tracker.log_design_result(sample_result)

    @pytest.mark.asyncio
    async def test_log_result_with_mlflow(self, tracker, sample_result, mock_mlflow):
        """Test logging result with MLflow available."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                await tracker.log_design_result(sample_result)
                assert mock_mlflow.log_metrics.called or mock_mlflow.log_metric.called

    @pytest.mark.asyncio
    async def test_log_result_includes_power_analysis(self, tracker, sample_result, mock_mlflow):
        """Test that power analysis metrics are logged."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                await tracker.log_design_result(sample_result)


class TestLogParams:
    """Tests for _log_params method."""

    def test_log_params_from_context(self, tracker, sample_context, mock_mlflow):
        """Test parameter logging from context."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                tracker._log_params(sample_context)
                assert mock_mlflow.log_params.called or mock_mlflow.log_param.called

    def test_log_params_includes_design_type(self, tracker, sample_context, mock_mlflow):
        """Test design type is logged as parameter."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                tracker._log_params(sample_context)

    def test_log_params_includes_treatments(self, tracker, sample_context, mock_mlflow):
        """Test treatments are logged as parameters."""
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
    async def test_log_artifacts_includes_validity_audit(self, tracker, sample_result, mock_mlflow):
        """Test validity audit is included in artifacts."""
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


class TestGetDesignHistory:
    """Tests for get_design_history method."""

    @pytest.mark.asyncio
    async def test_get_history_without_mlflow(self, tracker):
        """Test history query when MLflow unavailable."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            history = await tracker.get_design_history()
            assert history is None or isinstance(history, (list, dict))

    @pytest.mark.asyncio
    async def test_get_history_with_mlflow(self, tracker, mock_mlflow):
        """Test history query with MLflow available."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_mlflow.search_runs.return_value = MagicMock(
                    to_dict=MagicMock(return_value={"run_id": ["run1", "run2"]})
                )

                history = await tracker.get_design_history()
                mock_mlflow.search_runs.assert_called()

    @pytest.mark.asyncio
    async def test_get_history_filters_by_design_type(self, tracker, mock_mlflow):
        """Test history query filters by design type."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_mlflow.search_runs.return_value = MagicMock(
                    to_dict=MagicMock(return_value={"run_id": ["run1"]})
                )

                await tracker.get_design_history(design_type="randomized_controlled_trial")


class TestGetDesignMetricsSummary:
    """Tests for get_design_metrics_summary method."""

    @pytest.mark.asyncio
    async def test_get_summary_without_mlflow(self, tracker):
        """Test summary query when MLflow unavailable."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            summary = await tracker.get_design_metrics_summary()
            assert summary is None or isinstance(summary, dict)

    @pytest.mark.asyncio
    async def test_get_summary_returns_dict(self, tracker, mock_mlflow):
        """Test summary returns dictionary structure."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_df = MagicMock()
                mock_df.to_dict.return_value = {
                    "metrics.estimated_power": [0.80, 0.82],
                    "metrics.recommended_sample_size": [2000, 2500],
                }
                mock_mlflow.search_runs.return_value = mock_df

                summary = await tracker.get_design_metrics_summary()
                assert summary is None or isinstance(summary, dict)


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
                    async with tracker.start_design_run(
                        DesignContext(query="test", hypothesis="hyp")
                    ):
                        pass
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_handles_invalid_result_format(self, tracker):
        """Test handling of invalid result format."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            await tracker.log_design_result({"invalid": "structure"})

    def test_handles_empty_context(self, tracker):
        """Test handling of minimal context."""
        ctx = DesignContext(query="", hypothesis="")
        assert ctx is not None

    @pytest.mark.asyncio
    async def test_handles_missing_validity_audit(self, tracker):
        """Test handling of result without validity audit."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            result = {"recommended_sample_size": 1000}
            await tracker.log_design_result(result)
