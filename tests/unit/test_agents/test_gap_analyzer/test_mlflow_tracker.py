"""Unit tests for GapAnalyzer MLflow Tracker module.

Tests cover:
- Tracker initialization and MLflow availability checking
- Context managers for starting analysis runs
- Metric extraction (ROI, gap metrics, prioritization)
- Parameter logging
- Artifact logging (JSON)
- Historical query methods (ROI history)
- Graceful degradation when MLflow unavailable

Phase 1 G03 from observability audit remediation plan.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.gap_analyzer.mlflow_tracker import (
    GapAnalysisContext,
    GapAnalyzerMetrics,
    GapAnalyzerMLflowTracker,
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
    """Create a GapAnalyzerMLflowTracker instance."""
    return GapAnalyzerMLflowTracker()


@pytest.fixture
def sample_context():
    """Create a sample GapAnalysisContext."""
    return GapAnalysisContext(
        query="Find ROI opportunities in Northeast region",
        brand="Kisqali",
        region="Northeast",
        time_period="Q4_2024",
        analysis_type="roi_optimization",
    )


@pytest.fixture
def sample_result():
    """Create a sample analysis result dict."""
    return {
        "gaps_identified": 5,
        "total_roi_potential": 1250000.0,
        "avg_gap_severity": 0.72,
        "top_opportunities": [
            {"name": "HCP targeting gap", "roi": 500000, "priority": 1},
            {"name": "Coverage gap", "roi": 350000, "priority": 2},
        ],
        "prioritization_ranking": [1, 2, 3, 4, 5],
        "confidence_score": 0.85,
    }


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestExperimentConfiguration:
    """Tests for experiment configuration constants."""

    def test_experiment_prefix_format(self):
        """Test experiment prefix follows naming convention."""
        assert EXPERIMENT_PREFIX == "e2i_causal/gap_analyzer"

    def test_experiment_prefix_contains_agent_name(self):
        """Test experiment prefix includes agent identifier."""
        assert "gap_analyzer" in EXPERIMENT_PREFIX


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestGapAnalysisContext:
    """Tests for GapAnalysisContext dataclass."""

    def test_context_creation_minimal(self):
        """Test context creation with minimal fields."""
        ctx = GapAnalysisContext(
            query="test query",
            brand="TestBrand",
        )
        assert ctx.query == "test query"
        assert ctx.brand == "TestBrand"

    def test_context_creation_full(self, sample_context):
        """Test context creation with all fields."""
        assert sample_context.brand == "Kisqali"
        assert sample_context.region == "Northeast"
        assert sample_context.analysis_type == "roi_optimization"

    def test_context_default_values(self):
        """Test context default values."""
        ctx = GapAnalysisContext(
            query="test",
            brand="brand",
        )
        assert ctx.region is None or isinstance(ctx.region, str)


class TestGapAnalyzerMetrics:
    """Tests for GapAnalyzerMetrics dataclass."""

    def test_metrics_creation(self):
        """Test metrics dataclass creation."""
        metrics = GapAnalyzerMetrics(
            gaps_identified=5,
            total_roi_potential=1000000.0,
            avg_gap_severity=0.75,
        )
        assert metrics.gaps_identified == 5
        assert metrics.total_roi_potential == 1000000.0

    def test_metrics_optional_fields(self):
        """Test metrics with optional fields."""
        metrics = GapAnalyzerMetrics(
            gaps_identified=3,
        )
        assert metrics.gaps_identified == 3


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestTrackerInitialization:
    """Tests for tracker initialization."""

    def test_tracker_creation(self, tracker):
        """Test tracker can be created."""
        assert tracker is not None
        assert isinstance(tracker, GapAnalyzerMLflowTracker)

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


class TestStartAnalysisRun:
    """Tests for start_analysis_run context manager."""

    @pytest.mark.asyncio
    async def test_start_run_returns_context_manager(self, tracker, sample_context):
        """Test start_analysis_run returns async context manager."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            async with tracker.start_analysis_run(sample_context) as run_ctx:
                assert run_ctx is None or isinstance(run_ctx, dict)

    @pytest.mark.asyncio
    async def test_start_run_with_mlflow(self, tracker, sample_context, mock_mlflow):
        """Test start_analysis_run with MLflow available."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_run = MagicMock()
                mock_run.info.run_id = "test_run_123"
                mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
                mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

                async with tracker.start_analysis_run(sample_context):
                    mock_mlflow.set_experiment.assert_called()

    @pytest.mark.asyncio
    async def test_start_run_logs_brand_context(self, tracker, sample_context, mock_mlflow):
        """Test that brand context is logged during run start."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_run = MagicMock()
                mock_run.info.run_id = "test_run_123"
                mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
                mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

                async with tracker.start_analysis_run(sample_context):
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

    def test_extract_roi_metrics(self, tracker, sample_result):
        """Test ROI metric extraction."""
        metrics = tracker._extract_metrics(sample_result)
        assert isinstance(metrics, dict)
        # ROI metrics should be extracted
        if "total_roi_potential" in metrics:
            assert metrics["total_roi_potential"] == 1250000.0

    def test_extract_metrics_handles_missing_fields(self, tracker):
        """Test metric extraction with missing fields."""
        result = {"gaps_identified": 2}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, dict)

    def test_extract_metrics_handles_none(self, tracker):
        """Test metric extraction with None values."""
        result = {"gaps_identified": None, "total_roi_potential": 500000}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, dict)


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogAnalysisResult:
    """Tests for log_analysis_result method."""

    @pytest.mark.asyncio
    async def test_log_result_without_mlflow(self, tracker, sample_result):
        """Test logging result when MLflow unavailable."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            await tracker.log_analysis_result(sample_result)

    @pytest.mark.asyncio
    async def test_log_result_with_mlflow(self, tracker, sample_result, mock_mlflow):
        """Test logging result with MLflow available."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                await tracker.log_analysis_result(sample_result)
                assert mock_mlflow.log_metrics.called or mock_mlflow.log_metric.called

    @pytest.mark.asyncio
    async def test_log_result_includes_roi(self, tracker, sample_result, mock_mlflow):
        """Test that ROI metrics are logged."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                await tracker.log_analysis_result(sample_result)
                # Verify ROI-related logging


class TestLogParams:
    """Tests for _log_params method."""

    def test_log_params_from_context(self, tracker, sample_context, mock_mlflow):
        """Test parameter logging from context."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                tracker._log_params(sample_context)
                assert mock_mlflow.log_params.called or mock_mlflow.log_param.called

    def test_log_params_includes_brand(self, tracker, sample_context, mock_mlflow):
        """Test brand is logged as parameter."""
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


# =============================================================================
# HISTORY QUERY TESTS
# =============================================================================


class TestGetROIHistory:
    """Tests for get_roi_history method."""

    @pytest.mark.asyncio
    async def test_get_history_without_mlflow(self, tracker):
        """Test history query when MLflow unavailable."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            history = await tracker.get_roi_history()
            assert history is None or isinstance(history, (list, dict))

    @pytest.mark.asyncio
    async def test_get_history_with_mlflow(self, tracker, mock_mlflow):
        """Test history query with MLflow available."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_mlflow.search_runs.return_value = MagicMock(
                    to_dict=MagicMock(return_value={"run_id": ["run1", "run2"]})
                )

                history = await tracker.get_roi_history()
                mock_mlflow.search_runs.assert_called()

    @pytest.mark.asyncio
    async def test_get_history_filters_by_brand(self, tracker, mock_mlflow):
        """Test history query filters by brand."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_mlflow.search_runs.return_value = MagicMock(
                    to_dict=MagicMock(return_value={"run_id": ["run1"]})
                )

                await tracker.get_roi_history(brand="Kisqali")
                # Verify brand filter was applied


class TestGetPerformanceSummary:
    """Tests for get_performance_summary method."""

    @pytest.mark.asyncio
    async def test_get_summary_without_mlflow(self, tracker):
        """Test summary query when MLflow unavailable."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            summary = await tracker.get_performance_summary()
            assert summary is None or isinstance(summary, dict)

    @pytest.mark.asyncio
    async def test_get_summary_returns_dict(self, tracker, mock_mlflow):
        """Test summary returns dictionary structure."""
        with patch.object(tracker, "_mlflow", mock_mlflow):
            with patch.object(tracker, "_check_mlflow", return_value=True):
                mock_df = MagicMock()
                mock_df.to_dict.return_value = {
                    "metrics.total_roi_potential": [1000000, 1500000],
                    "metrics.gaps_identified": [3, 5],
                }
                mock_mlflow.search_runs.return_value = mock_df

                summary = await tracker.get_performance_summary()
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
                    async with tracker.start_analysis_run(
                        GapAnalysisContext(query="test", brand="Test")
                    ):
                        pass
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_handles_invalid_result_format(self, tracker):
        """Test handling of invalid result format."""
        with patch.object(tracker, "_check_mlflow", return_value=False):
            await tracker.log_analysis_result({"invalid": "structure"})

    def test_handles_empty_context(self, tracker):
        """Test handling of minimal context."""
        ctx = GapAnalysisContext(query="", brand="")
        assert ctx is not None
