"""Unit tests for FeedbackLearner MLflow Tracker module.

Tests cover:
- Tracker initialization and MLflow lazy loading
- LearningContext dataclass creation and defaults
- FeedbackLearnerMetrics dataclass and to_dict conversion
- Context managers for starting learning runs
- Metric extraction from state
- Artifact logging (patterns, recommendations, rubric evaluation)
- Historical query methods
- Learning summary and pattern trends
- Graceful degradation when MLflow unavailable

From observability audit remediation plan.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.feedback_learner.mlflow_tracker import (
    EXPERIMENT_PREFIX,
    FeedbackLearnerMetrics,
    FeedbackLearnerMLflowTracker,
    LearningContext,
    create_tracker,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_mlflow():
    """Mock MLflow module."""
    mock = MagicMock()
    mock.set_tracking_uri = MagicMock()
    mock.set_experiment = MagicMock()
    mock.get_experiment_by_name = MagicMock(
        return_value=MagicMock(experiment_id="test_exp_id")
    )
    mock.create_experiment = MagicMock(return_value="new_exp_id")
    mock.start_run = MagicMock()
    mock.end_run = MagicMock()
    mock.log_param = MagicMock()
    mock.log_params = MagicMock()
    mock.log_metric = MagicMock()
    mock.log_metrics = MagicMock()
    mock.log_artifact = MagicMock()
    mock.set_tag = MagicMock()
    mock.set_tags = MagicMock()
    mock.search_runs = MagicMock()
    return mock


@pytest.fixture
def tracker():
    """Create a FeedbackLearnerMLflowTracker instance."""
    return FeedbackLearnerMLflowTracker()


@pytest.fixture
def sample_context():
    """Create a sample LearningContext."""
    return LearningContext(
        run_id="run_123",
        experiment_name="weekly_feedback",
        batch_id="batch_2024_01",
        time_range_start="2024-01-01T00:00:00Z",
        time_range_end="2024-01-07T23:59:59Z",
        focus_agents=["causal_impact", "gap_analyzer"],
    )


@pytest.fixture
def sample_state():
    """Create a sample FeedbackLearnerState dict."""
    return {
        "status": "completed",
        "feedback_items": [
            {"feedback_type": "rating", "value": 4},
            {"feedback_type": "correction", "value": "fix typo"},
            {"feedback_type": "outcome", "value": True},
            {"feedback_type": "rating", "value": 5},
            {"feedback_type": "explicit", "value": "Great analysis"},
        ],
        "feedback_summary": {"average_rating": 4.5, "total_count": 5},
        "detected_patterns": [
            {
                "pattern_id": "p1",
                "severity": "high",
                "affected_agents": ["causal_impact"],
            },
            {
                "pattern_id": "p2",
                "severity": "critical",
                "affected_agents": ["gap_analyzer", "explainer"],
            },
            {
                "pattern_id": "p3",
                "severity": "low",
                "affected_agents": ["causal_impact"],
            },
        ],
        "learning_recommendations": [
            {"category": "prompt_update", "priority": "high"},
            {"category": "model_retrain", "priority": "medium"},
            {"category": "config_change", "priority": "low"},
            {"category": "prompt_update", "priority": "high"},
        ],
        "priority_improvements": [
            {"improvement_id": "i1", "impact": "high"},
            {"improvement_id": "i2", "impact": "medium"},
        ],
        "proposed_updates": [
            {"update_id": "u1", "type": "prompt"},
            {"update_id": "u2", "type": "config"},
        ],
        "applied_updates": [{"update_id": "u1", "status": "applied"}],
        "rubric_weighted_score": 0.85,
        "rubric_evaluation": {"dimension_scores": {"accuracy": 0.9}},
        "rubric_decision": "approve",
        "training_signal": {"signal_type": "positive", "strength": 0.8},
        "learning_summary": "Detected 3 patterns and proposed 2 updates.",
        "collection_latency_ms": 100,
        "analysis_latency_ms": 200,
        "extraction_latency_ms": 50,
        "update_latency_ms": 75,
        "total_latency_ms": 425,
    }


# =============================================================================
# CONSTANT TESTS
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_experiment_prefix(self):
        """Test experiment prefix is set correctly."""
        assert EXPERIMENT_PREFIX == "e2i_causal/feedback_learner"


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestLearningContext:
    """Tests for LearningContext dataclass."""

    def test_context_creation_required_fields(self):
        """Test context creation with required fields."""
        ctx = LearningContext(
            run_id="run_123",
            experiment_name="test_experiment",
            batch_id="batch_001",
        )
        assert ctx.run_id == "run_123"
        assert ctx.experiment_name == "test_experiment"
        assert ctx.batch_id == "batch_001"

    def test_context_creation_full(self, sample_context):
        """Test context creation with all fields."""
        assert sample_context.run_id == "run_123"
        assert sample_context.experiment_name == "weekly_feedback"
        assert sample_context.batch_id == "batch_2024_01"
        assert sample_context.time_range_start == "2024-01-01T00:00:00Z"
        assert sample_context.time_range_end == "2024-01-07T23:59:59Z"
        assert sample_context.focus_agents == ["causal_impact", "gap_analyzer"]

    def test_context_default_values(self):
        """Test context default values for optional fields."""
        ctx = LearningContext(
            run_id="run",
            experiment_name="exp",
            batch_id="batch",
        )
        assert ctx.time_range_start is None
        assert ctx.time_range_end is None
        assert ctx.focus_agents is None
        assert ctx.timestamp is not None  # Auto-generated

    def test_context_timestamp_auto_generated(self):
        """Test that timestamp is automatically generated."""
        ctx = LearningContext(
            run_id="run",
            experiment_name="exp",
            batch_id="batch",
        )
        assert isinstance(ctx.timestamp, str)
        # Should be ISO format
        datetime.fromisoformat(ctx.timestamp.replace("Z", "+00:00"))


class TestFeedbackLearnerMetrics:
    """Tests for FeedbackLearnerMetrics dataclass."""

    def test_metrics_creation_defaults(self):
        """Test metrics dataclass creation with defaults."""
        metrics = FeedbackLearnerMetrics()
        assert metrics.total_feedback_items == 0
        assert metrics.patterns_detected == 0
        assert metrics.recommendations_count == 0
        assert metrics.average_rating is None
        assert metrics.rubric_weighted_score is None
        assert metrics.has_rubric_evaluation is False
        assert metrics.has_training_signal is False

    def test_metrics_creation_with_values(self):
        """Test metrics dataclass creation with values."""
        metrics = FeedbackLearnerMetrics(
            total_feedback_items=10,
            patterns_detected=3,
            high_severity_patterns=1,
            critical_patterns=1,
            recommendations_count=5,
            average_rating=4.2,
            rubric_weighted_score=0.85,
            has_rubric_evaluation=True,
        )
        assert metrics.total_feedback_items == 10
        assert metrics.patterns_detected == 3
        assert metrics.high_severity_patterns == 1
        assert metrics.critical_patterns == 1
        assert metrics.recommendations_count == 5
        assert metrics.average_rating == 4.2
        assert metrics.rubric_weighted_score == 0.85
        assert metrics.has_rubric_evaluation is True

    def test_metrics_to_dict(self):
        """Test metrics to_dict conversion."""
        metrics = FeedbackLearnerMetrics(
            total_feedback_items=5,
            rating_feedback_count=2,
            patterns_detected=3,
            recommendations_count=4,
            average_rating=4.5,
            rubric_weighted_score=0.8,
            has_rubric_evaluation=True,
            has_training_signal=True,
            total_latency_ms=500,
        )
        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["total_feedback_items"] == 5
        assert result["rating_feedback_count"] == 2
        assert result["patterns_detected"] == 3
        assert result["recommendations_count"] == 4
        assert result["average_rating"] == 4.5
        assert result["rubric_weighted_score"] == 0.8
        assert result["has_rubric_evaluation"] == 1  # Converted to int
        assert result["has_training_signal"] == 1  # Converted to int
        assert result["total_latency_ms"] == 500

    def test_metrics_to_dict_without_optional(self):
        """Test to_dict excludes None optional values."""
        metrics = FeedbackLearnerMetrics(
            total_feedback_items=5,
            average_rating=None,
            rubric_weighted_score=None,
        )
        result = metrics.to_dict()

        assert "average_rating" not in result
        assert "rubric_weighted_score" not in result
        assert "total_feedback_items" in result


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestTrackerInitialization:
    """Tests for tracker initialization."""

    def test_tracker_creation(self, tracker):
        """Test tracker can be created."""
        assert tracker is not None
        assert isinstance(tracker, FeedbackLearnerMLflowTracker)

    def test_tracker_custom_uri(self):
        """Test tracker with custom tracking URI."""
        tracker = FeedbackLearnerMLflowTracker(
            tracking_uri="http://custom:5000"
        )
        assert tracker._tracking_uri == "http://custom:5000"

    def test_tracker_artifact_logging_enabled_by_default(self, tracker):
        """Test artifact logging is enabled by default."""
        assert tracker.enable_artifact_logging is True

    def test_tracker_artifact_logging_disabled(self):
        """Test tracker with artifact logging disabled."""
        tracker = FeedbackLearnerMLflowTracker(enable_artifact_logging=False)
        assert tracker.enable_artifact_logging is False

    def test_tracker_lazy_mlflow_loading(self, tracker):
        """Test MLflow is lazily loaded."""
        assert tracker._mlflow is None


class TestLazyLoading:
    """Tests for lazy MLflow loading."""

    def test_get_mlflow_lazy_loads(self, tracker, mock_mlflow):
        """Test _get_mlflow lazy loads MLflow."""
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            with patch("mlflow.set_tracking_uri"):
                result = tracker._get_mlflow()
                assert result is not None

    def test_get_mlflow_handles_import_error(self, tracker):
        """Test _get_mlflow handles ImportError gracefully."""
        with patch.dict("sys.modules", {"mlflow": None}):
            with patch(
                "builtins.__import__", side_effect=ImportError("No MLflow")
            ):
                result = tracker._get_mlflow()
                assert result is None


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================


class TestStartLearningRun:
    """Tests for start_learning_run context manager."""

    @pytest.mark.asyncio
    async def test_start_run_without_mlflow(self, tracker):
        """Test start_learning_run when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            async with tracker.start_learning_run(
                experiment_name="test",
                batch_id="batch_001",
            ) as ctx:
                assert ctx is not None
                assert isinstance(ctx, LearningContext)
                assert ctx.run_id == "no-mlflow"

    @pytest.mark.asyncio
    async def test_start_run_with_mlflow(self, tracker, mock_mlflow):
        """Test start_learning_run with MLflow available."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=mock_run
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            async with tracker.start_learning_run(
                experiment_name="test",
                batch_id="batch_001",
            ) as ctx:
                assert ctx is not None
                assert isinstance(ctx, LearningContext)
                assert ctx.run_id == "test_run_123"

    @pytest.mark.asyncio
    async def test_start_run_logs_parameters(self, tracker, mock_mlflow):
        """Test that parameters are logged on run start."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=mock_run
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            async with tracker.start_learning_run(
                experiment_name="test",
                batch_id="batch_001",
                focus_agents=["agent1", "agent2"],
            ) as ctx:
                pass

            mock_mlflow.log_params.assert_called()
            mock_mlflow.set_tags.assert_called()

    @pytest.mark.asyncio
    async def test_start_run_creates_experiment_if_missing(
        self, tracker, mock_mlflow
    ):
        """Test experiment creation when not exists."""
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=mock_run
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            async with tracker.start_learning_run(
                experiment_name="new_experiment",
                batch_id="batch_001",
            ) as ctx:
                pass

            mock_mlflow.create_experiment.assert_called_once()


# =============================================================================
# METRIC EXTRACTION TESTS
# =============================================================================


class TestMetricExtraction:
    """Tests for _extract_metrics method."""

    def test_extract_metrics_from_state(self, tracker, sample_state):
        """Test metric extraction from full state."""
        metrics = tracker._extract_metrics(sample_state)

        assert isinstance(metrics, FeedbackLearnerMetrics)
        assert metrics.total_feedback_items == 5
        assert metrics.rating_feedback_count == 2
        assert metrics.correction_feedback_count == 1
        assert metrics.outcome_feedback_count == 1
        assert metrics.explicit_feedback_count == 1
        assert metrics.average_rating == 4.5

    def test_extract_pattern_metrics(self, tracker, sample_state):
        """Test pattern metric extraction."""
        metrics = tracker._extract_metrics(sample_state)

        assert metrics.patterns_detected == 3
        assert metrics.high_severity_patterns == 1
        assert metrics.critical_patterns == 1
        assert metrics.affected_agents_count == 3  # causal_impact, gap_analyzer, explainer

    def test_extract_recommendation_metrics(self, tracker, sample_state):
        """Test recommendation metric extraction."""
        metrics = tracker._extract_metrics(sample_state)

        assert metrics.recommendations_count == 4
        assert metrics.prompt_updates_recommended == 2
        assert metrics.model_retrains_recommended == 1
        assert metrics.config_changes_recommended == 1
        assert metrics.priority_improvements_count == 2

    def test_extract_update_metrics(self, tracker, sample_state):
        """Test knowledge update metric extraction."""
        metrics = tracker._extract_metrics(sample_state)

        assert metrics.proposed_updates_count == 2
        assert metrics.applied_updates_count == 1

    def test_extract_rubric_metrics(self, tracker, sample_state):
        """Test rubric evaluation metric extraction."""
        metrics = tracker._extract_metrics(sample_state)

        assert metrics.rubric_weighted_score == 0.85
        assert metrics.has_rubric_evaluation is True
        assert metrics.has_training_signal is True

    def test_extract_latency_metrics(self, tracker, sample_state):
        """Test latency metric extraction."""
        metrics = tracker._extract_metrics(sample_state)

        assert metrics.collection_latency_ms == 100
        assert metrics.analysis_latency_ms == 200
        assert metrics.extraction_latency_ms == 50
        assert metrics.update_latency_ms == 75
        assert metrics.total_latency_ms == 425

    def test_extract_metrics_handles_empty_state(self, tracker):
        """Test metric extraction with empty state."""
        metrics = tracker._extract_metrics({})

        assert metrics.total_feedback_items == 0
        assert metrics.patterns_detected == 0
        assert metrics.recommendations_count == 0
        assert metrics.average_rating is None

    def test_extract_metrics_handles_none_lists(self, tracker):
        """Test metric extraction with None lists."""
        state = {
            "feedback_items": None,
            "detected_patterns": None,
            "learning_recommendations": None,
        }
        metrics = tracker._extract_metrics(state)

        assert metrics.total_feedback_items == 0
        assert metrics.patterns_detected == 0
        assert metrics.recommendations_count == 0


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogLearningResult:
    """Tests for log_learning_result method."""

    @pytest.mark.asyncio
    async def test_log_result_without_mlflow(self, tracker, sample_state):
        """Test logging when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            # Should not raise
            await tracker.log_learning_result(sample_state)

    @pytest.mark.asyncio
    async def test_log_result_without_active_run(self, tracker, sample_state):
        """Test logging without active run."""
        tracker._current_run_id = None
        with patch.object(tracker, "_get_mlflow", return_value=MagicMock()):
            # Should not raise
            await tracker.log_learning_result(sample_state)

    @pytest.mark.asyncio
    async def test_log_result_logs_metrics(self, tracker, mock_mlflow, sample_state):
        """Test that metrics are logged."""
        tracker._current_run_id = "active_run"

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            with patch.object(tracker, "_log_artifacts", new_callable=AsyncMock):
                await tracker.log_learning_result(sample_state)

                mock_mlflow.log_metrics.assert_called_once()
                mock_mlflow.set_tags.assert_called()

    @pytest.mark.asyncio
    async def test_log_result_sets_quality_tags(
        self, tracker, mock_mlflow, sample_state
    ):
        """Test that quality tags are set."""
        tracker._current_run_id = "active_run"

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            with patch.object(tracker, "_log_artifacts", new_callable=AsyncMock):
                await tracker.log_learning_result(sample_state)

                # Check that set_tags was called with quality indicators
                calls = mock_mlflow.set_tags.call_args_list
                assert len(calls) > 0


class TestLogArtifacts:
    """Tests for _log_artifacts method."""

    @pytest.mark.asyncio
    async def test_log_artifacts_without_mlflow(self, tracker, sample_state):
        """Test artifact logging when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            # Should not raise
            await tracker._log_artifacts(sample_state)

    @pytest.mark.asyncio
    async def test_log_artifacts_logs_patterns(
        self, tracker, mock_mlflow, sample_state
    ):
        """Test that detected patterns are logged as artifacts."""
        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
                mock_tmpdir.return_value.__enter__ = MagicMock(
                    return_value="/tmp/test"
                )
                mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)

                with patch("builtins.open", MagicMock()):
                    await tracker._log_artifacts(sample_state)

                    mock_mlflow.log_artifact.assert_called()

    @pytest.mark.asyncio
    async def test_log_artifacts_handles_empty_state(self, tracker, mock_mlflow):
        """Test artifact logging with empty state."""
        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            # Should not raise, just skip logging
            await tracker._log_artifacts({})


# =============================================================================
# HISTORY QUERY TESTS
# =============================================================================


class TestGetLearningHistory:
    """Tests for get_learning_history method."""

    @pytest.mark.asyncio
    async def test_get_history_without_mlflow(self, tracker):
        """Test history query when MLflow unavailable."""
        with patch.object(tracker, "_get_mlflow", return_value=None):
            history = await tracker.get_learning_history()
            assert history == []

    @pytest.mark.asyncio
    async def test_get_history_experiment_not_found(self, tracker, mock_mlflow):
        """Test history query when experiment not found."""
        mock_mlflow.get_experiment_by_name.return_value = None

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            history = await tracker.get_learning_history()
            assert history == []

    @pytest.mark.asyncio
    async def test_get_history_with_results(self, tracker, mock_mlflow):
        """Test history query with results."""
        mock_experiment = MagicMock(experiment_id="exp_123")
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        # Mock search_runs returning DataFrame-like with iterrows
        mock_df = MagicMock()
        mock_df.iterrows.return_value = iter(
            [
                (
                    0,
                    {
                        "run_id": "run_1",
                        "start_time": datetime.now(timezone.utc),
                        "params.batch_id": "batch_001",
                        "metrics.total_feedback_items": 10,
                        "metrics.patterns_detected": 3,
                        "metrics.recommendations_count": 5,
                        "tags.completion_status": "completed",
                    },
                ),
            ]
        )
        mock_mlflow.search_runs.return_value = mock_df

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            history = await tracker.get_learning_history()

            assert len(history) == 1
            assert history[0]["run_id"] == "run_1"

    @pytest.mark.asyncio
    async def test_get_history_with_batch_filter(self, tracker, mock_mlflow):
        """Test history query with batch_id filter."""
        mock_experiment = MagicMock(experiment_id="exp_123")
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_df = MagicMock()
        mock_df.iterrows.return_value = iter([])
        mock_mlflow.search_runs.return_value = mock_df

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            await tracker.get_learning_history(batch_id="batch_001")

            call_args = mock_mlflow.search_runs.call_args
            assert "batch_id" in str(call_args)


class TestGetLearningSummary:
    """Tests for get_learning_summary method."""

    @pytest.mark.asyncio
    async def test_get_summary_empty_history(self, tracker):
        """Test summary with no history."""
        with patch.object(
            tracker, "get_learning_history", new_callable=AsyncMock
        ) as mock_history:
            mock_history.return_value = []

            summary = await tracker.get_learning_summary()

            assert summary["total_learning_runs"] == 0
            assert summary["total_feedback_processed"] == 0

    @pytest.mark.asyncio
    async def test_get_summary_with_history(self, tracker):
        """Test summary calculation with history."""
        mock_history = [
            {
                "total_feedback_items": 10,
                "patterns_detected": 3,
                "recommendations_count": 5,
                "rubric_weighted_score": 0.8,
                "completion_status": "completed",
            },
            {
                "total_feedback_items": 15,
                "patterns_detected": 2,
                "recommendations_count": 3,
                "rubric_weighted_score": 0.9,
                "completion_status": "completed",
            },
        ]

        with patch.object(
            tracker, "get_learning_history", new_callable=AsyncMock
        ) as mock_get_history:
            mock_get_history.return_value = mock_history

            summary = await tracker.get_learning_summary()

            assert summary["total_learning_runs"] == 2
            assert summary["total_feedback_processed"] == 25
            assert summary["total_patterns_detected"] == 5
            assert summary["total_recommendations"] == 8
            assert summary["avg_rubric_score"] == pytest.approx(0.85)
            assert summary["successful_runs"] == 2


class TestGetPatternTrends:
    """Tests for get_pattern_trends method."""

    @pytest.mark.asyncio
    async def test_get_trends_empty_history(self, tracker):
        """Test pattern trends with no history."""
        with patch.object(
            tracker, "get_learning_history", new_callable=AsyncMock
        ) as mock_history:
            mock_history.return_value = []

            trends = await tracker.get_pattern_trends()

            assert trends["total_runs"] == 0
            assert trends["pattern_detection_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_get_trends_with_history(self, tracker):
        """Test pattern trends calculation."""
        mock_history = [
            {"patterns_detected": 3, "critical_patterns": 1},
            {"patterns_detected": 0, "critical_patterns": 0},
            {"patterns_detected": 2, "critical_patterns": 2},
            {"patterns_detected": 1, "critical_patterns": 0},
        ]

        with patch.object(
            tracker, "get_learning_history", new_callable=AsyncMock
        ) as mock_get_history:
            mock_get_history.return_value = mock_history

            trends = await tracker.get_pattern_trends()

            assert trends["total_runs"] == 4
            assert trends["runs_with_patterns"] == 3
            assert trends["runs_with_critical_patterns"] == 2
            assert trends["pattern_detection_rate"] == 0.75
            assert trends["critical_pattern_rate"] == 0.5


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestCreateTracker:
    """Tests for create_tracker factory function."""

    def test_create_tracker_default(self):
        """Test creating tracker with defaults."""
        tracker = create_tracker()
        assert isinstance(tracker, FeedbackLearnerMLflowTracker)

    def test_create_tracker_custom_uri(self):
        """Test creating tracker with custom URI."""
        tracker = create_tracker(tracking_uri="http://custom:5000")
        assert tracker._tracking_uri == "http://custom:5000"


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handles_mlflow_connection_error(self, tracker, mock_mlflow):
        """Test handling of MLflow connection errors."""
        mock_mlflow.get_experiment_by_name.side_effect = Exception(
            "Connection failed"
        )

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            async with tracker.start_learning_run(
                experiment_name="test",
                batch_id="batch_001",
            ) as ctx:
                # Should gracefully degrade
                assert ctx.run_id == "experiment-error"

    @pytest.mark.asyncio
    async def test_handles_logging_error(self, tracker, mock_mlflow, sample_state):
        """Test handling of logging errors."""
        tracker._current_run_id = "active_run"
        mock_mlflow.log_metrics.side_effect = Exception("Logging failed")

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            # Should not raise
            await tracker.log_learning_result(sample_state)

    @pytest.mark.asyncio
    async def test_handles_artifact_logging_error(
        self, tracker, mock_mlflow, sample_state
    ):
        """Test handling of artifact logging errors."""
        mock_mlflow.log_artifact.side_effect = Exception("Artifact failed")

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            # Should not raise
            await tracker._log_artifacts(sample_state)

    @pytest.mark.asyncio
    async def test_handles_history_query_error(self, tracker, mock_mlflow):
        """Test handling of history query errors."""
        mock_mlflow.search_runs.side_effect = Exception("Query failed")
        mock_experiment = MagicMock(experiment_id="exp_123")
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        with patch.object(tracker, "_get_mlflow", return_value=mock_mlflow):
            history = await tracker.get_learning_history()
            assert history == []
