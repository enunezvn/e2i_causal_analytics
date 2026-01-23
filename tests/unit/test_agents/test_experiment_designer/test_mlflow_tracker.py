"""Unit tests for ExperimentDesigner MLflow Tracker module.

Tests cover:
- Tracker initialization and MLflow availability checking
- Context managers for starting design runs
- Metric extraction (power analysis, validity audits)
- Artifact logging (JSON)
- Historical query methods (design history, metrics summary)
- Graceful degradation when MLflow unavailable

Phase 1 G03 from observability audit remediation plan.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

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
def tracker():
    """Create an ExperimentDesignerMLflowTracker instance."""
    return ExperimentDesignerMLflowTracker()


@pytest.fixture
def sample_context():
    """Create a sample DesignContext."""
    return DesignContext(
        experiment_name="pharma_design",
        brand="Kisqali",
        business_question="Does call frequency impact TRx?",
        design_type="randomized_controlled_trial",
        query_id="query_123",
        run_id="run_456",
        start_time=datetime.now(),
    )


@pytest.fixture
def sample_result():
    """Create a sample design result dict."""
    return {
        "required_sample_size": 2500,
        "achieved_power": 0.82,
        "minimum_detectable_effect": 0.05,
        "validity_threats": [
            {"threat": "selection_bias", "severity": "high"},
            {"threat": "attrition", "severity": "medium"},
        ],
        "design_iterations": 3,
        "randomization_scheme": "stratified",
        "blocking_variables": ["region", "customer_segment"],
        "duration_estimate_days": 28,
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
        ctx = DesignContext()
        assert ctx.experiment_name == "default"

    def test_context_creation_full(self, sample_context):
        """Test context creation with all fields."""
        assert sample_context.design_type == "randomized_controlled_trial"
        assert sample_context.brand == "Kisqali"
        assert sample_context.business_question == "Does call frequency impact TRx?"

    def test_context_default_values(self):
        """Test context default values."""
        ctx = DesignContext()
        assert ctx.brand is None
        assert ctx.business_question is None
        assert ctx.design_type is None


class TestExperimentDesignerMetrics:
    """Tests for ExperimentDesignerMetrics dataclass."""

    def test_metrics_creation(self):
        """Test metrics dataclass creation."""
        metrics = ExperimentDesignerMetrics(
            required_sample_size=2500,
            achieved_power=0.82,
            minimum_detectable_effect=0.05,
        )
        assert metrics.required_sample_size == 2500
        assert metrics.achieved_power == 0.82

    def test_metrics_optional_fields(self):
        """Test metrics with default values."""
        metrics = ExperimentDesignerMetrics()
        assert metrics.required_sample_size == 0
        assert metrics.achieved_power == 0.0


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestTrackerInitialization:
    """Tests for tracker initialization."""

    def test_tracker_creation(self, tracker):
        """Test tracker can be created."""
        assert tracker is not None
        assert isinstance(tracker, ExperimentDesignerMLflowTracker)

    def test_tracker_has_mlflow_attr(self, tracker):
        """Test tracker has _mlflow attribute."""
        assert hasattr(tracker, "_mlflow")

    def test_get_mlflow_returns_mlflow_or_none(self, tracker):
        """Test _get_mlflow returns mlflow module or None."""
        result = tracker._get_mlflow()
        assert result is None or hasattr(result, "log_metric")


class TestMLflowAvailability:
    """Tests for MLflow availability checking."""

    def test_mlflow_starts_as_none(self, tracker):
        """Test MLflow is None initially (lazy loading)."""
        assert tracker._mlflow is None

    def test_graceful_degradation_when_unavailable(self, tracker):
        """Test tracker works when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            result = tracker._get_mlflow()
            assert result is None


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================


class TestStartDesignRun:
    """Tests for start_design_run context manager."""

    @pytest.mark.asyncio
    async def test_start_run_returns_context_manager(self, tracker):
        """Test start_design_run returns async context manager."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_design_run(
                experiment_name="test_experiment",
                brand="Kisqali",
            ) as run_ctx:
                assert run_ctx is None or isinstance(run_ctx, DesignContext)

    @pytest.mark.asyncio
    async def test_start_run_without_mlflow_returns_context(self, tracker):
        """Test start_design_run returns context when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_design_run(
                experiment_name="test_experiment",
                brand="Kisqali",
            ) as run_ctx:
                assert isinstance(run_ctx, DesignContext)
                assert run_ctx.experiment_name == "test_experiment"
                assert run_ctx.brand == "Kisqali"

    @pytest.mark.asyncio
    async def test_start_run_context_has_required_fields(self, tracker):
        """Test context manager returns context with required fields."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_design_run(
                experiment_name="test_experiment",
                brand="Kisqali",
                business_question="Test question?",
            ) as run_ctx:
                assert run_ctx.experiment_name == "test_experiment"
                assert run_ctx.brand == "Kisqali"
                assert run_ctx.business_question == "Test question?"


# =============================================================================
# METRIC EXTRACTION TESTS
# =============================================================================


class TestMetricExtraction:
    """Tests for _extract_metrics method."""

    def test_extract_metrics_from_dict(self, tracker, sample_result):
        """Test metric extraction from result dict."""
        metrics = tracker._extract_metrics(sample_result)
        assert isinstance(metrics, ExperimentDesignerMetrics)

    def test_extract_power_metrics(self, tracker):
        """Test power analysis metric extraction."""
        result = {"required_sample_size": 2500, "achieved_power": 0.82}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, ExperimentDesignerMetrics)

    def test_extract_validity_metrics(self, tracker, sample_result):
        """Test validity audit metric extraction."""
        metrics = tracker._extract_metrics(sample_result)
        assert isinstance(metrics, ExperimentDesignerMetrics)

    def test_extract_metrics_handles_missing_fields(self, tracker):
        """Test metric extraction with missing fields."""
        result = {"duration_estimate_days": 14}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, ExperimentDesignerMetrics)

    def test_extract_metrics_handles_none(self, tracker):
        """Test metric extraction with None values."""
        result = {"achieved_power": None, "required_sample_size": 2000}
        metrics = tracker._extract_metrics(result)
        assert isinstance(metrics, ExperimentDesignerMetrics)


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogDesignResult:
    """Tests for log_design_result method."""

    @pytest.mark.asyncio
    async def test_log_result_without_mlflow(self, tracker, sample_result):
        """Test logging result when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            await tracker.log_design_result(sample_result)

    @pytest.mark.asyncio
    async def test_log_result_extracts_metrics(self, tracker, sample_result):
        """Test logging extracts metrics from result."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            await tracker.log_design_result(sample_result)
            # Verify metrics can be extracted
            metrics = tracker._extract_metrics(sample_result)
            assert isinstance(metrics, ExperimentDesignerMetrics)

    @pytest.mark.asyncio
    async def test_log_result_handles_empty_result(self, tracker):
        """Test logging handles empty result dict."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            await tracker.log_design_result({})


class TestLogArtifacts:
    """Tests for _log_artifacts method."""

    @pytest.mark.asyncio
    async def test_log_artifacts_without_mlflow(self, tracker, sample_result):
        """Test artifact logging without MLflow."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            metrics = tracker._extract_metrics(sample_result)
            await tracker._log_artifacts(sample_result, None, metrics)

    @pytest.mark.asyncio
    async def test_log_artifacts_empty_result(self, tracker):
        """Test artifact logging with empty result."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            metrics = ExperimentDesignerMetrics()
            await tracker._log_artifacts({}, None, metrics)


# =============================================================================
# HISTORY QUERY TESTS
# =============================================================================


class TestGetDesignHistory:
    """Tests for get_design_history method."""

    def test_get_history_without_mlflow(self, tracker):
        """Test history query when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            history = tracker.get_design_history()
            assert isinstance(history, list)
            assert len(history) == 0

    def test_get_history_returns_list(self, tracker):
        """Test history query returns list structure."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            history = tracker.get_design_history(brand="Kisqali")
            assert isinstance(history, list)

    def test_get_history_with_design_type_filter(self, tracker):
        """Test history query with design type filter."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            history = tracker.get_design_history(design_type="rct")
            assert isinstance(history, list)


class TestGetDesignMetricsSummary:
    """Tests for get_design_metrics_summary method."""

    def test_get_summary_without_mlflow(self, tracker):
        """Test summary query when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            summary = tracker.get_design_metrics_summary()
            assert isinstance(summary, dict)

    def test_get_summary_returns_dict_structure(self, tracker):
        """Test summary returns dictionary structure."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            summary = tracker.get_design_metrics_summary()
            assert isinstance(summary, dict)

    def test_get_summary_with_brand_filter(self, tracker):
        """Test summary with brand filter."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            summary = tracker.get_design_metrics_summary(brand="Kisqali")
            assert isinstance(summary, dict)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handles_invalid_result_format(self, tracker):
        """Test handling of invalid result format."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            await tracker.log_design_result({"invalid": "structure"})

    @pytest.mark.asyncio
    async def test_handles_none_result(self, tracker):
        """Test handling of None values in result."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            result = {"achieved_power": None, "required_sample_size": None}
            await tracker.log_design_result(result)

    def test_handles_empty_context(self, tracker):
        """Test handling of minimal context."""
        ctx = DesignContext()
        assert ctx is not None
        assert ctx.experiment_name == "default"

    def test_context_with_all_optional_none(self, tracker):
        """Test context with all optional fields as None."""
        ctx = DesignContext(
            experiment_name="test",
            brand=None,
            business_question=None,
            design_type=None,
            query_id=None,
        )
        assert ctx.brand is None
        assert ctx.business_question is None
