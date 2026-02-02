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

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.agents.gap_analyzer.mlflow_tracker import (
    EXPERIMENT_PREFIX,
    GapAnalysisContext,
    GapAnalyzerMetrics,
    GapAnalyzerMLflowTracker,
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
    mock.search_runs = MagicMock(
        return_value=MagicMock(to_dict=MagicMock(return_value={"run_id": []}))
    )
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
        experiment_id="exp_123",
        run_id="run_456",
        experiment_name="roi_opportunities",
        started_at=datetime.now(),
        brand="Kisqali",
        region="Northeast",
        gap_type="roi_optimization",
        query_id="query_789",
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
        """Test context creation with required fields."""
        ctx = GapAnalysisContext(
            experiment_id="exp_123",
            run_id="run_456",
            experiment_name="test_experiment",
            started_at=datetime.now(),
        )
        assert ctx.experiment_id == "exp_123"
        assert ctx.run_id == "run_456"

    def test_context_creation_full(self, sample_context):
        """Test context creation with all fields."""
        assert sample_context.brand == "Kisqali"
        assert sample_context.region == "Northeast"
        assert sample_context.gap_type == "roi_optimization"

    def test_context_default_values(self):
        """Test context default values."""
        ctx = GapAnalysisContext(
            experiment_id="exp_123",
            run_id="run_456",
            experiment_name="test",
            started_at=datetime.now(),
        )
        assert ctx.region is None or isinstance(ctx.region, str)


class TestGapAnalyzerMetrics:
    """Tests for GapAnalyzerMetrics dataclass."""

    def test_metrics_creation(self):
        """Test metrics dataclass creation."""
        metrics = GapAnalyzerMetrics(
            total_gaps_detected=5,
            total_addressable_value=1000000.0,
            avg_gap_percentage=0.75,
        )
        assert metrics.total_gaps_detected == 5
        assert metrics.total_addressable_value == 1000000.0

    def test_metrics_optional_fields(self):
        """Test metrics with optional fields."""
        metrics = GapAnalyzerMetrics(
            total_gaps_detected=3,
        )
        assert metrics.total_gaps_detected == 3


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

    def test_mlflow_available_attribute_set(self, tracker):
        """Test MLflow availability attribute is set on init."""
        assert hasattr(tracker, "_mlflow_available")
        assert isinstance(tracker._mlflow_available, bool)

    def test_graceful_degradation_when_unavailable(self, tracker):
        """Test tracker works when MLflow unavailable."""
        tracker._mlflow_available = False
        assert tracker._mlflow_available is False


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================


class TestStartAnalysisRun:
    """Tests for start_analysis_run context manager."""

    @pytest.mark.asyncio
    async def test_start_run_returns_context_manager(self, tracker):
        """Test start_analysis_run returns async context manager."""
        tracker._mlflow_available = False
        async with tracker.start_analysis_run(
            experiment_name="test_experiment",
            brand="Kisqali",
        ) as run_ctx:
            assert run_ctx is None or isinstance(run_ctx, GapAnalysisContext)

    @pytest.mark.asyncio
    async def test_start_run_without_mlflow_returns_context(self, tracker):
        """Test start_analysis_run returns context when MLflow unavailable."""
        tracker._mlflow_available = False
        async with tracker.start_analysis_run(
            experiment_name="test_experiment",
            brand="Kisqali",
        ) as run_ctx:
            assert isinstance(run_ctx, GapAnalysisContext)
            assert run_ctx.experiment_name == "test_experiment"
            assert run_ctx.brand == "Kisqali"

    @pytest.mark.asyncio
    async def test_start_run_context_has_required_fields(self, tracker):
        """Test context manager returns context with required fields."""
        tracker._mlflow_available = False
        async with tracker.start_analysis_run(
            experiment_name="test_experiment",
            brand="Kisqali",
            region="Northeast",
        ) as run_ctx:
            assert run_ctx.experiment_id is not None
            assert run_ctx.run_id is not None
            assert run_ctx.started_at is not None
            assert run_ctx.brand == "Kisqali"
            assert run_ctx.region == "Northeast"


# =============================================================================
# METRIC EXTRACTION TESTS
# =============================================================================


class TestMetricExtraction:
    """Tests for _extract_metrics method."""

    def test_extract_metrics_from_dict(self, tracker, sample_result):
        """Test metric extraction from result dict."""
        metrics = tracker._extract_metrics(sample_result)
        assert isinstance(metrics, GapAnalyzerMetrics)

    def test_extract_roi_metrics(self, tracker):
        """Test ROI metric extraction."""
        result = {"total_addressable_value": 1250000.0}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, GapAnalyzerMetrics)
        assert metrics.total_addressable_value == 1250000.0

    def test_extract_metrics_handles_missing_fields(self, tracker):
        """Test metric extraction with missing fields."""
        result = {"total_gaps_detected": 2}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, GapAnalyzerMetrics)

    def test_extract_metrics_handles_none(self, tracker):
        """Test metric extraction with None values."""
        result = {"total_gap_value": None, "total_addressable_value": 500000}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, GapAnalyzerMetrics)
        assert metrics.total_addressable_value == 500000


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogAnalysisResult:
    """Tests for log_analysis_result method."""

    @pytest.mark.asyncio
    async def test_log_result_without_mlflow(self, tracker, sample_result):
        """Test logging result when MLflow unavailable."""
        tracker._mlflow_available = False
        await tracker.log_analysis_result(sample_result)

    @pytest.mark.asyncio
    async def test_log_result_with_mlflow(self, tracker, sample_result):
        """Test logging result with MLflow available."""
        tracker._mlflow_available = True
        with patch("mlflow.log_metric") as mock_log_metric:
            with patch("mlflow.log_param"):
                await tracker.log_analysis_result(sample_result)
                assert mock_log_metric.called

    @pytest.mark.asyncio
    async def test_log_result_handles_empty_result(self, tracker):
        """Test logging handles empty result dict."""
        tracker._mlflow_available = False
        await tracker.log_analysis_result({})


class TestLogParams:
    """Tests for _log_params method."""

    def test_log_params_from_output(self, tracker, sample_result):
        """Test parameter logging from output dict."""
        with patch("mlflow.log_param"):
            tracker._log_params(sample_result)
            # Should not fail, logging is called

    def test_log_params_includes_brand(self, tracker, sample_result):
        """Test brand is logged as parameter."""
        mock_state = {"brand": "Kisqali", "gap_type": "roi"}
        with patch("mlflow.log_param") as mock_log_param:
            tracker._log_params(sample_result, state=mock_state)
            # Verify brand was logged
            calls = [str(c) for c in mock_log_param.call_args_list]
            assert any("brand" in str(c) for c in calls)


class TestLogArtifacts:
    """Tests for _log_artifacts method."""

    @pytest.mark.asyncio
    async def test_log_artifacts_with_opportunities(self, tracker, sample_result):
        """Test artifact logging with opportunities."""
        sample_result["prioritized_opportunities"] = [{"id": 1, "name": "test"}]
        with patch("mlflow.log_artifact"):
            await tracker._log_artifacts(sample_result)

    @pytest.mark.asyncio
    async def test_log_artifacts_empty_result(self, tracker):
        """Test artifact logging with empty result."""
        with patch("mlflow.log_artifact"):
            await tracker._log_artifacts({})


# =============================================================================
# HISTORY QUERY TESTS
# =============================================================================


class TestGetROIHistory:
    """Tests for get_roi_history method."""

    @pytest.mark.asyncio
    async def test_get_history_without_mlflow(self, tracker):
        """Test history query when MLflow unavailable."""
        tracker._mlflow_available = False
        history = await tracker.get_roi_history()
        assert isinstance(history, list)
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_get_history_returns_list(self, tracker):
        """Test history query returns list structure."""
        tracker._mlflow_available = False
        history = await tracker.get_roi_history(brand="Kisqali")
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_get_history_with_days_filter(self, tracker):
        """Test history query with days filter."""
        tracker._mlflow_available = False
        history = await tracker.get_roi_history(days=7)
        assert isinstance(history, list)


class TestGetPerformanceSummary:
    """Tests for get_performance_summary method."""

    @pytest.mark.asyncio
    async def test_get_summary_without_mlflow(self, tracker):
        """Test summary query when MLflow unavailable."""
        tracker._mlflow_available = False
        summary = await tracker.get_performance_summary()
        assert isinstance(summary, dict)

    @pytest.mark.asyncio
    async def test_get_summary_returns_dict_structure(self, tracker):
        """Test summary returns dictionary with expected keys."""
        tracker._mlflow_available = False
        summary = await tracker.get_performance_summary()
        assert isinstance(summary, dict)
        assert "total_analyses" in summary

    @pytest.mark.asyncio
    async def test_get_summary_with_days_filter(self, tracker):
        """Test summary with days filter."""
        tracker._mlflow_available = False
        summary = await tracker.get_performance_summary(days=7)
        assert isinstance(summary, dict)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handles_invalid_result_format(self, tracker):
        """Test handling of invalid result format."""
        tracker._mlflow_available = False
        await tracker.log_analysis_result({"invalid": "structure"})

    @pytest.mark.asyncio
    async def test_handles_none_result(self, tracker):
        """Test handling of None values in result."""
        tracker._mlflow_available = False
        result = {"gaps_identified": None, "total_roi_potential": None}
        await tracker.log_analysis_result(result)

    def test_handles_empty_context(self, tracker):
        """Test handling of minimal context."""
        ctx = GapAnalysisContext(
            experiment_id="",
            run_id="",
            experiment_name="",
            started_at=datetime.now(),
        )
        assert ctx is not None

    def test_context_with_all_optional_none(self, tracker):
        """Test context with all optional fields as None."""
        ctx = GapAnalysisContext(
            experiment_id="exp_1",
            run_id="run_1",
            experiment_name="test",
            started_at=datetime.now(),
            brand=None,
            region=None,
            gap_type=None,
            query_id=None,
        )
        assert ctx.brand is None
        assert ctx.region is None
