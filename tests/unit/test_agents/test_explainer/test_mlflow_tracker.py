"""Unit tests for Explainer MLflow Tracker module.

Tests cover:
- Tracker initialization and MLflow availability checking
- Context managers for starting explanation runs
- Metric extraction and logging
- Parameter logging
- Artifact logging (JSON)
- Historical query methods
- Insight summary analysis
- Graceful degradation when MLflow unavailable

Part of observability audit remediation - G09 Phase 2.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.explainer.mlflow_tracker import (
    EXPERIMENT_PREFIX,
    ExplanationContext,
    ExplainerMetrics,
    ExplainerMLflowTracker,
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
    """Create an ExplainerMLflowTracker instance."""
    return ExplainerMLflowTracker()


@pytest.fixture
def sample_context():
    """Create a sample ExplanationContext."""
    return ExplanationContext(
        run_id="run_123",
        experiment_name="quarterly_review",
        user_expertise="executive",
        output_format="narrative",
        brand="remibrutinib",
        region="Northeast",
        source_agents=["causal_impact", "gap_analyzer"],
    )


@pytest.fixture
def sample_state():
    """Create a sample ExplainerState dict."""
    return {
        "extracted_insights": [
            {
                "insight_id": "1",
                "category": "finding",
                "statement": "Northeast shows highest response",
                "confidence": 0.89,
                "priority": 1,
                "actionability": "immediate",
            },
            {
                "insight_id": "2",
                "category": "recommendation",
                "statement": "Increase investment in Northeast",
                "confidence": 0.85,
                "priority": 2,
                "actionability": "immediate",
            },
            {
                "insight_id": "3",
                "category": "warning",
                "statement": "Southeast shows declining trend",
                "confidence": 0.72,
                "priority": 3,
                "actionability": "monitor",
            },
            {
                "insight_id": "4",
                "category": "opportunity",
                "statement": "Untapped potential in West",
                "confidence": 0.65,
                "priority": 4,
                "actionability": "evaluate",
            },
        ],
        "narrative_sections": [
            {"title": "Overview", "content": "..."},
            {"title": "Key Findings", "content": "..."},
            {"title": "Recommendations", "content": "..."},
        ],
        "executive_summary": "Analysis shows 23% improvement opportunity in Northeast territory with high confidence (89%).",
        "detailed_explanation": "## Key Findings\n\nThe causal analysis identified...",
        "analysis_context": [
            {"source_agent": "causal_impact", "data": "..."},
            {"source_agent": "gap_analyzer", "data": "..."},
        ],
        "analysis_results": [{"agent": "causal_impact"}, {"agent": "gap_analyzer"}],
        "visual_suggestions": [
            {"type": "effect_plot", "title": "Causal Effect Estimate"},
            {"type": "opportunity_matrix", "title": "Priority Matrix"},
        ],
        "follow_up_questions": [
            "What's driving the Northeast performance?",
            "How do we prioritize these recommendations?",
        ],
        "assembly_latency_ms": 100,
        "reasoning_latency_ms": 200,
        "generation_latency_ms": 150,
        "total_latency_ms": 450,
    }


# =============================================================================
# CONSTANT TESTS
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_experiment_prefix(self):
        """Test experiment prefix is set correctly."""
        assert EXPERIMENT_PREFIX == "e2i_causal/explainer"


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestExplanationContext:
    """Tests for ExplanationContext dataclass."""

    def test_context_creation_required_fields(self):
        """Test context creation with required fields."""
        ctx = ExplanationContext(
            run_id="run_123",
            experiment_name="quarterly_review",
            user_expertise="executive",
            output_format="narrative",
        )
        assert ctx.run_id == "run_123"
        assert ctx.experiment_name == "quarterly_review"
        assert ctx.user_expertise == "executive"
        assert ctx.output_format == "narrative"

    def test_context_creation_full(self, sample_context):
        """Test context creation with all fields."""
        assert sample_context.brand == "remibrutinib"
        assert sample_context.region == "Northeast"
        assert sample_context.source_agents == ["causal_impact", "gap_analyzer"]

    def test_context_default_values(self):
        """Test context default values for optional fields."""
        ctx = ExplanationContext(
            run_id="run",
            experiment_name="exp_name",
            user_expertise="analyst",
            output_format="structured",
        )
        assert ctx.brand is None
        assert ctx.region is None
        assert ctx.source_agents is None
        assert ctx.timestamp is not None  # Auto-generated


class TestExplainerMetrics:
    """Tests for ExplainerMetrics dataclass."""

    def test_metrics_creation(self):
        """Test metrics dataclass creation."""
        metrics = ExplainerMetrics(
            insight_count=5,
            findings_count=2,
            recommendations_count=2,
            warnings_count=1,
            avg_insight_confidence=0.78,
        )
        assert metrics.insight_count == 5
        assert metrics.findings_count == 2
        assert metrics.avg_insight_confidence == 0.78

    def test_metrics_defaults(self):
        """Test metrics default values."""
        metrics = ExplainerMetrics()
        assert metrics.insight_count == 0
        assert metrics.findings_count == 0
        assert metrics.recommendations_count == 0
        assert metrics.avg_insight_confidence == 0.0
        assert metrics.total_latency_ms == 0

    def test_metrics_to_dict(self):
        """Test metrics to_dict conversion."""
        metrics = ExplainerMetrics(
            insight_count=5,
            findings_count=2,
            recommendations_count=2,
            warnings_count=1,
            opportunities_count=0,
            avg_insight_confidence=0.78,
            high_priority_count=2,
            immediate_actionable_count=2,
            narrative_section_count=3,
            executive_summary_length=150,
            total_latency_ms=450,
        )
        result = metrics.to_dict()

        assert result["insight_count"] == 5
        assert result["findings_count"] == 2
        assert result["recommendations_count"] == 2
        assert result["avg_insight_confidence"] == 0.78
        assert result["narrative_section_count"] == 3


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestTrackerInitialization:
    """Tests for tracker initialization."""

    def test_tracker_creation(self, tracker):
        """Test tracker can be created."""
        assert tracker is not None
        assert isinstance(tracker, ExplainerMLflowTracker)

    def test_tracker_default_config(self, tracker):
        """Test tracker default configuration."""
        assert tracker.enable_artifact_logging is True
        assert tracker._current_run_id is None

    def test_tracker_custom_config(self):
        """Test tracker with custom configuration."""
        tracker = ExplainerMLflowTracker(
            tracking_uri="http://custom:5000",
            enable_artifact_logging=False,
        )
        assert tracker._tracking_uri == "http://custom:5000"
        assert tracker.enable_artifact_logging is False

    def test_tracker_lazy_mlflow_loading(self, tracker):
        """Test MLflow is lazily loaded."""
        assert tracker._mlflow is None


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================


class TestStartExplanationRun:
    """Tests for start_explanation_run context manager."""

    @pytest.mark.asyncio
    async def test_start_run_without_mlflow(self, tracker):
        """Test start_explanation_run when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_explanation_run(
                experiment_name="quarterly_review",
                user_expertise="executive",
                output_format="narrative",
            ) as ctx:
                assert ctx is not None
                assert isinstance(ctx, ExplanationContext)
                assert ctx.run_id == "no-mlflow"

    @pytest.mark.asyncio
    async def test_start_run_with_mlflow(self, tracker, mock_mlflow):
        """Test start_explanation_run with MLflow available."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            async with tracker.start_explanation_run(
                experiment_name="quarterly_review",
                user_expertise="executive",
                output_format="narrative",
                brand="remibrutinib",
            ) as ctx:
                assert ctx is not None
                assert ctx.run_id == "test_run_123"

    @pytest.mark.asyncio
    async def test_start_run_accepts_optional_params(self, tracker):
        """Test that start_explanation_run accepts optional parameters."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_explanation_run(
                experiment_name="quarterly_review",
                user_expertise="data_scientist",
                output_format="structured",
                source_agents=["causal_impact", "gap_analyzer"],
                brand="remibrutinib",
                region="Northeast",
                tags={"custom_tag": "value"},
            ) as ctx:
                assert ctx is not None
                assert ctx.user_expertise == "data_scientist"
                assert ctx.output_format == "structured"
                assert ctx.source_agents == ["causal_impact", "gap_analyzer"]

    @pytest.mark.asyncio
    async def test_start_run_handles_experiment_error(self, tracker, mock_mlflow):
        """Test graceful handling of experiment creation errors."""
        mock_mlflow.get_experiment_by_name.side_effect = Exception("Connection failed")

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            async with tracker.start_explanation_run(
                experiment_name="test",
                user_expertise="analyst",
                output_format="brief",
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
        assert isinstance(metrics, ExplainerMetrics)
        assert metrics.insight_count == 4
        assert metrics.findings_count == 1
        assert metrics.recommendations_count == 1
        assert metrics.warnings_count == 1
        assert metrics.opportunities_count == 1
        assert metrics.high_priority_count == 2  # priority <= 2
        assert metrics.immediate_actionable_count == 2

    def test_extract_metrics_handles_empty_state(self, tracker):
        """Test metric extraction with empty state."""
        metrics = tracker._extract_metrics({})
        assert isinstance(metrics, ExplainerMetrics)
        assert metrics.insight_count == 0
        assert metrics.findings_count == 0

    def test_extract_metrics_handles_missing_fields(self, tracker):
        """Test metric extraction with missing fields."""
        state = {
            "extracted_insights": [
                {"category": "finding", "confidence": 0.9}
            ],
        }
        metrics = tracker._extract_metrics(state)
        assert metrics.insight_count == 1
        assert metrics.findings_count == 1

    def test_extract_metrics_calculates_avg_confidence(self, tracker, sample_state):
        """Test average confidence calculation."""
        metrics = tracker._extract_metrics(sample_state)
        expected_avg = (0.89 + 0.85 + 0.72 + 0.65) / 4
        assert abs(metrics.avg_insight_confidence - expected_avg) < 0.01

    def test_extract_metrics_counts_sources(self, tracker, sample_state):
        """Test source agent counting."""
        metrics = tracker._extract_metrics(sample_state)
        assert metrics.source_agents_count == 2
        assert metrics.analysis_results_count == 2


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogExplanationResult:
    """Tests for log_explanation_result method."""

    @pytest.mark.asyncio
    async def test_log_result_without_mlflow(self, tracker, sample_state):
        """Test logging result when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            # Should not raise
            await tracker.log_explanation_result(sample_state)

    @pytest.mark.asyncio
    async def test_log_result_without_run_id(self, tracker, sample_state, mock_mlflow):
        """Test logging result when no run is active."""
        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            tracker._current_run_id = None
            # Should not raise
            await tracker.log_explanation_result(sample_state)
            mock_mlflow.log_metrics.assert_not_called()

    @pytest.mark.asyncio
    async def test_log_result_with_mlflow(self, tracker, sample_state, mock_mlflow):
        """Test logging result with MLflow available."""
        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            tracker._current_run_id = "test_run_123"
            tracker.enable_artifact_logging = False  # Skip artifacts

            await tracker.log_explanation_result(sample_state)

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


class TestGetExplanationHistory:
    """Tests for get_explanation_history method."""

    @pytest.mark.asyncio
    async def test_get_history_without_mlflow(self, tracker):
        """Test history query when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            history = await tracker.get_explanation_history()
            assert history == []

    @pytest.mark.asyncio
    async def test_get_history_no_experiment(self, tracker, mock_mlflow):
        """Test history query when experiment doesn't exist."""
        mock_mlflow.get_experiment_by_name.return_value = None

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            history = await tracker.get_explanation_history()
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
            history = await tracker.get_explanation_history()
            assert len(history) == 1
            assert history[0]["run_id"] == "run_1"


class TestGetInsightSummary:
    """Tests for get_insight_summary method."""

    @pytest.mark.asyncio
    async def test_get_summary_empty_history(self, tracker):
        """Test insight summary with empty history."""
        with patch.object(tracker, "get_explanation_history", return_value=[]):
            summary = await tracker.get_insight_summary()
            assert summary["total_explanations"] == 0
            assert summary["total_insights"] == 0
            assert summary["avg_insights_per_run"] == 0.0

    @pytest.mark.asyncio
    async def test_get_summary_with_history(self, tracker):
        """Test insight summary with history data."""
        history = [
            {
                "insight_count": 5,
                "avg_insight_confidence": 0.85,
                "user_expertise": "executive",
                "output_format": "narrative",
            },
            {
                "insight_count": 3,
                "avg_insight_confidence": 0.75,
                "user_expertise": "analyst",
                "output_format": "structured",
            },
        ]
        with patch.object(tracker, "get_explanation_history", return_value=history):
            summary = await tracker.get_insight_summary()
            assert summary["total_explanations"] == 2
            assert summary["total_insights"] == 8
            assert summary["avg_insights_per_run"] == 4.0
            assert summary["avg_confidence"] == 0.80


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestCreateTracker:
    """Tests for create_tracker factory function."""

    def test_create_tracker_default(self):
        """Test create_tracker with defaults."""
        tracker = create_tracker()
        assert isinstance(tracker, ExplainerMLflowTracker)

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
            await tracker.log_explanation_result({"extracted_insights": []})

    @pytest.mark.asyncio
    async def test_handles_artifact_error(self, tracker, mock_mlflow):
        """Test graceful handling of artifact logging errors."""
        mock_mlflow.log_artifact.side_effect = Exception("Artifact error")

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                mock_tmpdir.return_value.__enter__ = MagicMock(return_value="/tmp/test")
                mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)

                # Should not raise
                await tracker._log_artifacts({"extracted_insights": [{}]})

    @pytest.mark.asyncio
    async def test_handles_history_query_error(self, tracker, mock_mlflow):
        """Test graceful handling of history query errors."""
        mock_mlflow.search_runs.side_effect = Exception("Query failed")

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            # Should not raise, returns empty list
            history = await tracker.get_explanation_history()
            assert history == []
