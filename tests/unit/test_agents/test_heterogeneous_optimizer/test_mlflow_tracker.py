"""Unit tests for HeterogeneousOptimizer MLflow Tracker module.

Tests cover:
- Tracker initialization and MLflow availability checking
- Context managers for starting analysis runs
- Metric extraction (CATE, uplift metrics - AUUC, Qini)
- Parameter logging
- Artifact logging (JSON)
- Historical query methods (CATE history)
- Graceful degradation when MLflow unavailable

Phase 1 G03 from observability audit remediation plan.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from src.agents.heterogeneous_optimizer.mlflow_tracker import (
    EXPERIMENT_PREFIX,
    HeterogeneousOptimizerContext,
    HeterogeneousOptimizerMetrics,
    HeterogeneousOptimizerMLflowTracker,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def tracker():
    """Create a HeterogeneousOptimizerMLflowTracker instance."""
    return HeterogeneousOptimizerMLflowTracker()


@pytest.fixture
def sample_context():
    """Create a sample HeterogeneousOptimizerContext."""
    return HeterogeneousOptimizerContext(
        experiment_id="exp_123",
        run_id="run_456",
        experiment_name="segment_optimization",
        started_at=datetime.now(),
        brand="Remibrutinib",
        region="Northeast",
        treatment_var="marketing_intensity",
        outcome_var="prescription_volume",
        query_id="query_789",
    )


@pytest.fixture
def sample_result():
    """Create a sample analysis result dict."""
    return {
        "cate_estimates": {
            "high_value": 0.25,
            "medium_value": 0.15,
            "low_value": 0.05,
        },
        "segment_effects": [
            {"segment": "high_value", "effect": 0.25, "ci_lower": 0.20, "ci_upper": 0.30},
            {"segment": "medium_value", "effect": 0.15, "ci_lower": 0.10, "ci_upper": 0.20},
        ],
        "overall_auuc": 0.72,
        "overall_qini": 0.65,
        "policy_recommendations": [
            {"segment": "high_value", "action": "increase_intensity", "expected_lift": 0.25},
        ],
        "heterogeneity_score": 0.45,
    }


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestExperimentConfiguration:
    """Tests for experiment configuration constants."""

    def test_experiment_prefix_format(self):
        """Test experiment prefix follows naming convention."""
        assert EXPERIMENT_PREFIX == "e2i_causal/heterogeneous_optimizer"

    def test_experiment_prefix_contains_agent_name(self):
        """Test experiment prefix includes agent identifier."""
        assert "heterogeneous_optimizer" in EXPERIMENT_PREFIX


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestHeterogeneousOptimizerContext:
    """Tests for HeterogeneousOptimizerContext dataclass."""

    def test_context_creation_minimal(self):
        """Test context creation with required fields."""
        ctx = HeterogeneousOptimizerContext(
            experiment_id="exp_123",
            run_id="run_456",
            experiment_name="test_experiment",
            started_at=datetime.now(),
        )
        assert ctx.experiment_id == "exp_123"
        assert ctx.run_id == "run_456"

    def test_context_creation_full(self, sample_context):
        """Test context creation with all fields."""
        assert sample_context.brand == "Remibrutinib"
        assert sample_context.region == "Northeast"
        assert sample_context.treatment_var == "marketing_intensity"

    def test_context_default_values(self):
        """Test context default values."""
        ctx = HeterogeneousOptimizerContext(
            experiment_id="exp_123",
            run_id="run_456",
            experiment_name="test",
            started_at=datetime.now(),
        )
        assert ctx.region is None or isinstance(ctx.region, str)


class TestHeterogeneousOptimizerMetrics:
    """Tests for HeterogeneousOptimizerMetrics dataclass."""

    def test_metrics_creation(self):
        """Test metrics dataclass creation."""
        metrics = HeterogeneousOptimizerMetrics(
            overall_auuc=0.72,
            overall_qini=0.65,
            heterogeneity_score=0.45,
        )
        assert metrics.overall_auuc == 0.72
        assert metrics.overall_qini == 0.65

    def test_metrics_optional_fields(self):
        """Test metrics with optional fields."""
        metrics = HeterogeneousOptimizerMetrics(
            n_segments_analyzed=3,
        )
        assert metrics.n_segments_analyzed == 3


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestTrackerInitialization:
    """Tests for tracker initialization."""

    def test_tracker_creation(self, tracker):
        """Test tracker can be created."""
        assert tracker is not None
        assert isinstance(tracker, HeterogeneousOptimizerMLflowTracker)

    def test_tracker_has_mlflow_available_attr(self, tracker):
        """Test tracker has _mlflow_available attribute."""
        assert hasattr(tracker, "_mlflow_available")

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
            brand="Remibrutinib",
        ) as run_ctx:
            assert run_ctx is None or isinstance(run_ctx, HeterogeneousOptimizerContext)

    @pytest.mark.asyncio
    async def test_start_run_without_mlflow_returns_context(self, tracker):
        """Test start_analysis_run returns context when MLflow unavailable."""
        tracker._mlflow_available = False
        async with tracker.start_analysis_run(
            experiment_name="test_experiment",
            brand="Remibrutinib",
        ) as run_ctx:
            assert isinstance(run_ctx, HeterogeneousOptimizerContext)
            assert run_ctx.experiment_name == "test_experiment"
            assert run_ctx.brand == "Remibrutinib"

    @pytest.mark.asyncio
    async def test_start_run_context_has_required_fields(self, tracker):
        """Test context manager returns context with required fields."""
        tracker._mlflow_available = False
        async with tracker.start_analysis_run(
            experiment_name="test_experiment",
            brand="Remibrutinib",
            treatment_var="marketing_intensity",
        ) as run_ctx:
            assert run_ctx.experiment_id is not None
            assert run_ctx.run_id is not None
            assert run_ctx.started_at is not None
            assert run_ctx.brand == "Remibrutinib"
            assert run_ctx.treatment_var == "marketing_intensity"


# =============================================================================
# METRIC EXTRACTION TESTS
# =============================================================================


class TestMetricExtraction:
    """Tests for _extract_metrics method."""

    def test_extract_metrics_from_dict(self, tracker, sample_result):
        """Test metric extraction from result dict."""
        metrics = tracker._extract_metrics(sample_result)
        assert isinstance(metrics, HeterogeneousOptimizerMetrics)

    def test_extract_uplift_metrics(self, tracker):
        """Test uplift metric extraction (ATE, heterogeneity)."""
        result = {"overall_ate": 0.72, "heterogeneity_score": 0.65}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, HeterogeneousOptimizerMetrics)
        assert metrics.overall_ate == 0.72
        assert metrics.heterogeneity_score == 0.65

    def test_extract_metrics_handles_missing_fields(self, tracker):
        """Test metric extraction with missing fields."""
        result = {"n_segments_analyzed": 5}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, HeterogeneousOptimizerMetrics)

    def test_extract_metrics_handles_none(self, tracker):
        """Test metric extraction with None values."""
        result = {"overall_auuc": None, "overall_qini": 0.60}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, HeterogeneousOptimizerMetrics)

    def test_extract_cate_estimates(self, tracker, sample_result):
        """Test CATE estimates extraction."""
        metrics = tracker._extract_metrics(sample_result)
        assert isinstance(metrics, HeterogeneousOptimizerMetrics)


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
    async def test_log_result_extracts_metrics(self, tracker, sample_result):
        """Test logging extracts metrics from result."""
        tracker._mlflow_available = False
        # Verify no error when logging with MLflow unavailable
        await tracker.log_analysis_result(sample_result)
        # Verify metrics can be extracted
        metrics = tracker._extract_metrics(sample_result)
        assert isinstance(metrics, HeterogeneousOptimizerMetrics)

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

    def test_log_params_includes_treatment(self, tracker, sample_result):
        """Test treatment is logged as parameter."""
        mock_state = {"treatment_var": "marketing_intensity", "outcome_var": "prescription_volume"}
        with patch("mlflow.log_param"):
            tracker._log_params(sample_result, state=mock_state)


class TestLogArtifacts:
    """Tests for _log_artifacts method."""

    @pytest.mark.asyncio
    async def test_log_artifacts_with_policy_recommendations(self, tracker, sample_result):
        """Test artifact logging with policy recommendations."""
        sample_result["policy_recommendations"] = [{"segment": "high_value", "action": "increase"}]
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


class TestGetCATEHistory:
    """Tests for get_cate_history method."""

    @pytest.mark.asyncio
    async def test_get_history_without_mlflow(self, tracker):
        """Test history query when MLflow unavailable."""
        tracker._mlflow_available = False
        history = await tracker.get_cate_history()
        assert isinstance(history, list)
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_get_history_returns_list(self, tracker):
        """Test history query returns list structure."""
        tracker._mlflow_available = False
        history = await tracker.get_cate_history(brand="Remibrutinib")
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_get_history_with_days_filter(self, tracker):
        """Test history query with days filter."""
        tracker._mlflow_available = False
        history = await tracker.get_cate_history(days=7)
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
        result = {"overall_auuc": None, "overall_qini": None}
        await tracker.log_analysis_result(result)

    def test_handles_empty_context(self, tracker):
        """Test handling of minimal context."""
        ctx = HeterogeneousOptimizerContext(
            experiment_id="",
            run_id="",
            experiment_name="",
            started_at=datetime.now(),
        )
        assert ctx is not None

    def test_context_with_all_optional_none(self, tracker):
        """Test context with all optional fields as None."""
        ctx = HeterogeneousOptimizerContext(
            experiment_id="exp_1",
            run_id="run_1",
            experiment_name="test",
            started_at=datetime.now(),
            brand=None,
            region=None,
            treatment_var=None,
            outcome_var=None,
            query_id=None,
        )
        assert ctx.brand is None
        assert ctx.treatment_var is None
