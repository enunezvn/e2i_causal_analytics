"""
MLflow Integration for Feedback Learner Agent.

Provides comprehensive MLflow tracking for feedback learning cycles,
logging pattern detection, learning recommendations, knowledge updates,
and rubric evaluation metrics.

Integration Points:
    - MLflow experiment tracking (via MLflowConnector)
    - Opik tracing (via existing agent integration)
    - DSPy/GEPA optimization metrics
    - Dashboard metrics queries

Usage:
    tracker = FeedbackLearnerMLflowTracker()
    async with tracker.start_learning_run(
        experiment_name="weekly_feedback",
        batch_id="batch_2024_01"
    ):
        output = await agent.run(input_data)
        await tracker.log_learning_result(output, state)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional, Union

if TYPE_CHECKING:
    from .state import FeedbackLearnerState

logger = logging.getLogger(__name__)

# Experiment prefix for Feedback Learner Agent
EXPERIMENT_PREFIX = "e2i_causal/feedback_learner"


@dataclass
class LearningContext:
    """Context for an MLflow learning run."""

    run_id: str
    experiment_name: str
    batch_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Optional context
    time_range_start: Optional[str] = None
    time_range_end: Optional[str] = None
    focus_agents: Optional[list[str]] = None


@dataclass
class FeedbackLearnerMetrics:
    """Structured metrics for feedback learner tracking."""

    # Feedback metrics
    total_feedback_items: int = 0
    rating_feedback_count: int = 0
    correction_feedback_count: int = 0
    outcome_feedback_count: int = 0
    explicit_feedback_count: int = 0
    average_rating: Optional[float] = None

    # Pattern metrics
    patterns_detected: int = 0
    high_severity_patterns: int = 0
    critical_patterns: int = 0
    affected_agents_count: int = 0

    # Learning output metrics
    recommendations_count: int = 0
    priority_improvements_count: int = 0
    prompt_updates_recommended: int = 0
    model_retrains_recommended: int = 0
    config_changes_recommended: int = 0

    # Knowledge update metrics
    proposed_updates_count: int = 0
    applied_updates_count: int = 0

    # Rubric evaluation metrics
    rubric_weighted_score: Optional[float] = None
    has_rubric_evaluation: bool = False

    # DSPy/GEPA training signal
    has_training_signal: bool = False

    # Latency metrics
    collection_latency_ms: int = 0
    analysis_latency_ms: int = 0
    extraction_latency_ms: int = 0
    update_latency_ms: int = 0
    total_latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for MLflow logging."""
        metrics: Dict[str, Union[int, float]] = {
            "total_feedback_items": self.total_feedback_items,
            "rating_feedback_count": self.rating_feedback_count,
            "correction_feedback_count": self.correction_feedback_count,
            "outcome_feedback_count": self.outcome_feedback_count,
            "explicit_feedback_count": self.explicit_feedback_count,
            "patterns_detected": self.patterns_detected,
            "high_severity_patterns": self.high_severity_patterns,
            "critical_patterns": self.critical_patterns,
            "affected_agents_count": self.affected_agents_count,
            "recommendations_count": self.recommendations_count,
            "priority_improvements_count": self.priority_improvements_count,
            "prompt_updates_recommended": self.prompt_updates_recommended,
            "model_retrains_recommended": self.model_retrains_recommended,
            "config_changes_recommended": self.config_changes_recommended,
            "proposed_updates_count": self.proposed_updates_count,
            "applied_updates_count": self.applied_updates_count,
            "has_rubric_evaluation": int(self.has_rubric_evaluation),
            "has_training_signal": int(self.has_training_signal),
            "collection_latency_ms": self.collection_latency_ms,
            "analysis_latency_ms": self.analysis_latency_ms,
            "extraction_latency_ms": self.extraction_latency_ms,
            "update_latency_ms": self.update_latency_ms,
            "total_latency_ms": self.total_latency_ms,
        }

        # Add optional metrics
        if self.average_rating is not None:
            metrics["average_rating"] = self.average_rating
        if self.rubric_weighted_score is not None:
            metrics["rubric_weighted_score"] = self.rubric_weighted_score

        return metrics


class FeedbackLearnerMLflowTracker:
    """
    Tracks Feedback Learner Agent metrics in MLflow.

    Integrates with MLflow to log:
    - Feedback collection statistics
    - Pattern detection results
    - Learning recommendations
    - Knowledge update tracking
    - Rubric evaluation scores
    - DSPy/GEPA training signals

    Example:
        tracker = FeedbackLearnerMLflowTracker()

        async with tracker.start_learning_run("weekly", batch_id="batch_001"):
            output = await agent.run(input_data)
            await tracker.log_learning_result(output, final_state)

        # Query historical results
        history = await tracker.get_learning_history(days=30)
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        enable_artifact_logging: bool = True,
    ):
        """
        Initialize the tracker.

        Args:
            tracking_uri: MLflow tracking server URI (default: from env)
            enable_artifact_logging: Whether to log artifacts
        """
        self._mlflow = None
        self._tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.enable_artifact_logging = enable_artifact_logging
        self._current_run_id: Optional[str] = None

    def _get_mlflow(self):
        """Lazy load MLflow to avoid import errors when not installed."""
        if self._mlflow is None:
            try:
                import mlflow

                mlflow.set_tracking_uri(self._tracking_uri)
                self._mlflow = mlflow
            except ImportError:
                logger.warning("MLflow not installed, tracking disabled")
                return None
        return self._mlflow

    @asynccontextmanager
    async def start_learning_run(
        self,
        experiment_name: str = "default",
        batch_id: str = "unknown",
        time_range_start: Optional[str] = None,
        time_range_end: Optional[str] = None,
        focus_agents: Optional[list[str]] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> AsyncIterator[LearningContext]:
        """
        Start an MLflow run for learning tracking.

        Args:
            experiment_name: Name of the experiment
            batch_id: Feedback batch identifier
            time_range_start: Start of feedback time range
            time_range_end: End of feedback time range
            focus_agents: List of agents to focus on
            tags: Additional MLflow tags

        Yields:
            LearningContext with run information
        """
        mlflow = self._get_mlflow()

        if mlflow is None:
            yield LearningContext(
                run_id="no-mlflow",
                experiment_name=experiment_name,
                batch_id=batch_id,
                time_range_start=time_range_start,
                time_range_end=time_range_end,
                focus_agents=focus_agents,
            )
            return

        full_experiment_name = f"{EXPERIMENT_PREFIX}/{experiment_name}"

        try:
            experiment = mlflow.get_experiment_by_name(full_experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    full_experiment_name,
                    artifact_location="mlflow-artifacts:/",
                    tags={
                        "framework": "e2i_causal",
                        "agent": "feedback_learner",
                        "tier": "5",
                    },
                )
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            logger.warning(f"Could not create/get experiment: {e}")
            yield LearningContext(
                run_id="experiment-error",
                experiment_name=experiment_name,
                batch_id=batch_id,
            )
            return

        try:
            with mlflow.start_run(experiment_id=experiment_id) as run:
                self._current_run_id = run.info.run_id

                # Log run parameters
                mlflow.log_params(
                    {
                        "agent": "feedback_learner",
                        "tier": 5,
                        "batch_id": batch_id,
                    }
                )

                # Log time range if provided
                if time_range_start:
                    mlflow.log_param("time_range_start", time_range_start)
                if time_range_end:
                    mlflow.log_param("time_range_end", time_range_end)

                # Log focus agents if provided
                if focus_agents:
                    mlflow.log_param("focus_agents", ",".join(focus_agents))
                    mlflow.log_param("focus_agents_count", len(focus_agents))

                # Log context tags
                mlflow.set_tags(
                    {
                        "agent_type": "self_improvement",
                        "framework_version": "4.3",
                    }
                )

                # Custom tags
                for key, value in (tags or {}).items():
                    mlflow.set_tag(key, value)

                ctx = LearningContext(
                    run_id=run.info.run_id,
                    experiment_name=experiment_name,
                    batch_id=batch_id,
                    time_range_start=time_range_start,
                    time_range_end=time_range_end,
                    focus_agents=focus_agents,
                )

                yield ctx

                self._current_run_id = None

        except Exception as e:
            logger.error(f"MLflow run failed: {e}")
            self._current_run_id = None
            raise

    async def log_learning_result(
        self,
        state: "FeedbackLearnerState",
    ) -> None:
        """
        Log learning results to MLflow.

        Args:
            state: Final FeedbackLearnerState from agent execution
        """
        mlflow = self._get_mlflow()
        if mlflow is None or self._current_run_id is None:
            return

        try:
            # Extract metrics from state
            metrics = self._extract_metrics(state)
            mlflow.log_metrics(metrics.to_dict())

            # Log quality tags
            mlflow.set_tags(
                {
                    "has_patterns": str(metrics.patterns_detected > 0).lower(),
                    "has_critical_patterns": str(metrics.critical_patterns > 0).lower(),
                    "has_recommendations": str(metrics.recommendations_count > 0).lower(),
                    "has_applied_updates": str(metrics.applied_updates_count > 0).lower(),
                    "rubric_evaluated": str(metrics.has_rubric_evaluation).lower(),
                }
            )

            # Log rubric decision if present
            rubric_decision = state.get("rubric_decision")
            if rubric_decision:
                mlflow.set_tag("rubric_decision", rubric_decision)

            # Log status
            status = state.get("status", "unknown")
            mlflow.set_tag("completion_status", status)

            # Log artifacts
            if self.enable_artifact_logging:
                await self._log_artifacts(state)

            logger.debug(
                f"Logged learning metrics to MLflow run {self._current_run_id}: "
                f"patterns={metrics.patterns_detected}, recommendations={metrics.recommendations_count}"
            )

        except Exception as e:
            logger.warning(f"Failed to log learning metrics to MLflow: {e}")

    def _extract_metrics(
        self,
        state: "FeedbackLearnerState",
    ) -> FeedbackLearnerMetrics:
        """Extract metrics from state."""
        metrics = FeedbackLearnerMetrics()

        # Feedback metrics
        feedback_items = state.get("feedback_items", []) or []
        metrics.total_feedback_items = len(feedback_items)

        for item in feedback_items:
            feedback_type = item.get("feedback_type", "")
            if feedback_type == "rating":
                metrics.rating_feedback_count += 1
            elif feedback_type == "correction":
                metrics.correction_feedback_count += 1
            elif feedback_type == "outcome":
                metrics.outcome_feedback_count += 1
            elif feedback_type == "explicit":
                metrics.explicit_feedback_count += 1

        # Feedback summary metrics
        feedback_summary = state.get("feedback_summary", {}) or {}
        metrics.average_rating = feedback_summary.get("average_rating")

        # Pattern metrics
        patterns = state.get("detected_patterns", []) or []
        metrics.patterns_detected = len(patterns)

        affected_agents = set()
        for pattern in patterns:
            severity = pattern.get("severity", "")
            if severity == "high":
                metrics.high_severity_patterns += 1
            elif severity == "critical":
                metrics.critical_patterns += 1

            for agent in pattern.get("affected_agents", []):
                affected_agents.add(agent)

        metrics.affected_agents_count = len(affected_agents)

        # Learning recommendation metrics
        recommendations = state.get("learning_recommendations", []) or []
        metrics.recommendations_count = len(recommendations)

        for rec in recommendations:
            category = rec.get("category", "")
            if category == "prompt_update":
                metrics.prompt_updates_recommended += 1
            elif category == "model_retrain":
                metrics.model_retrains_recommended += 1
            elif category == "config_change":
                metrics.config_changes_recommended += 1

        priority_improvements = state.get("priority_improvements", []) or []
        metrics.priority_improvements_count = len(priority_improvements)

        # Knowledge update metrics
        proposed_updates = state.get("proposed_updates", []) or []
        metrics.proposed_updates_count = len(proposed_updates)

        applied_updates = state.get("applied_updates", []) or []
        metrics.applied_updates_count = len(applied_updates)

        # Rubric evaluation metrics
        metrics.rubric_weighted_score = state.get("rubric_weighted_score")
        metrics.has_rubric_evaluation = state.get("rubric_evaluation") is not None

        # DSPy/GEPA training signal
        metrics.has_training_signal = state.get("training_signal") is not None

        # Latency metrics
        metrics.collection_latency_ms = state.get("collection_latency_ms", 0)
        metrics.analysis_latency_ms = state.get("analysis_latency_ms", 0)
        metrics.extraction_latency_ms = state.get("extraction_latency_ms", 0)
        metrics.update_latency_ms = state.get("update_latency_ms", 0)
        metrics.total_latency_ms = state.get("total_latency_ms", 0)

        return metrics

    async def _log_artifacts(
        self,
        state: "FeedbackLearnerState",
    ) -> None:
        """Log artifacts to MLflow."""
        mlflow = self._get_mlflow()
        if mlflow is None:
            return

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Log detected patterns
                patterns = state.get("detected_patterns", [])
                if patterns:
                    patterns_path = os.path.join(tmpdir, "detected_patterns.json")
                    with open(patterns_path, "w") as f:
                        json.dump(patterns, f, indent=2, default=str)
                    mlflow.log_artifact(patterns_path, "patterns")

                # Log learning recommendations
                recommendations = state.get("learning_recommendations", [])
                if recommendations:
                    recs_path = os.path.join(tmpdir, "learning_recommendations.json")
                    with open(recs_path, "w") as f:
                        json.dump(recommendations, f, indent=2, default=str)
                    mlflow.log_artifact(recs_path, "recommendations")

                # Log proposed updates
                proposed_updates = state.get("proposed_updates", [])
                if proposed_updates:
                    updates_path = os.path.join(tmpdir, "proposed_updates.json")
                    with open(updates_path, "w") as f:
                        json.dump(proposed_updates, f, indent=2, default=str)
                    mlflow.log_artifact(updates_path, "updates")

                # Log rubric evaluation
                rubric_eval = state.get("rubric_evaluation")
                if rubric_eval:
                    rubric_path = os.path.join(tmpdir, "rubric_evaluation.json")
                    with open(rubric_path, "w") as f:
                        json.dump(rubric_eval, f, indent=2, default=str)
                    mlflow.log_artifact(rubric_path, "evaluation")

                # Log learning summary
                learning_summary = state.get("learning_summary")
                if learning_summary:
                    summary_path = os.path.join(tmpdir, "learning_summary.txt")
                    with open(summary_path, "w") as f:
                        f.write(learning_summary)
                    mlflow.log_artifact(summary_path, "summary")

                # Log feedback summary
                feedback_summary = state.get("feedback_summary")
                if feedback_summary:
                    fb_summary_path = os.path.join(tmpdir, "feedback_summary.json")
                    with open(fb_summary_path, "w") as f:
                        json.dump(feedback_summary, f, indent=2, default=str)
                    mlflow.log_artifact(fb_summary_path, "feedback")

                # Log training signal if present (for GEPA integration)
                training_signal = state.get("training_signal")
                if training_signal:
                    signal_path = os.path.join(tmpdir, "training_signal.json")
                    # Convert to dict if it's a dataclass
                    signal_data = (
                        training_signal.__dict__
                        if hasattr(training_signal, "__dict__")
                        else training_signal
                    )
                    with open(signal_path, "w") as f:
                        json.dump(signal_data, f, indent=2, default=str)
                    mlflow.log_artifact(signal_path, "training")

        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")

    async def get_learning_history(
        self,
        experiment_name: str = "default",
        batch_id: Optional[str] = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query historical learning runs.

        Args:
            experiment_name: Name of the experiment to query
            batch_id: Filter by batch ID
            max_results: Maximum number of results to return

        Returns:
            List of historical learning results
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            return []

        try:
            full_experiment_name = f"{EXPERIMENT_PREFIX}/{experiment_name}"
            experiment = mlflow.get_experiment_by_name(full_experiment_name)
            if experiment is None:
                return []

            # Build filter string
            filters = []
            if batch_id:
                filters.append(f"params.batch_id = '{batch_id}'")

            filter_string = " AND ".join(filters) if filters else None

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                order_by=["start_time DESC"],
            )

            history = []
            for _, row in runs.iterrows():
                history.append(
                    {
                        "run_id": row["run_id"],
                        "timestamp": row["start_time"],
                        "batch_id": row.get("params.batch_id"),
                        "total_feedback_items": row.get("metrics.total_feedback_items"),
                        "patterns_detected": row.get("metrics.patterns_detected"),
                        "critical_patterns": row.get("metrics.critical_patterns"),
                        "recommendations_count": row.get("metrics.recommendations_count"),
                        "applied_updates_count": row.get("metrics.applied_updates_count"),
                        "rubric_weighted_score": row.get("metrics.rubric_weighted_score"),
                        "total_latency_ms": row.get("metrics.total_latency_ms"),
                        "completion_status": row.get("tags.completion_status"),
                    }
                )

            return history

        except Exception as e:
            logger.warning(f"Failed to query learning history: {e}")
            return []

    async def get_learning_summary(
        self,
        experiment_name: str = "default",
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Get summary of learning performance over time.

        Args:
            experiment_name: Experiment to analyze
            days: Number of days to look back

        Returns:
            Dictionary with learning summary
        """
        history = await self.get_learning_history(experiment_name, max_results=1000)

        if not history:
            return {
                "total_learning_runs": 0,
                "total_feedback_processed": 0,
                "total_patterns_detected": 0,
                "total_recommendations": 0,
                "avg_rubric_score": 0.0,
            }

        feedback_counts = [
            h.get("total_feedback_items", 0) for h in history if h.get("total_feedback_items")
        ]
        pattern_counts = [
            h.get("patterns_detected", 0) for h in history if h.get("patterns_detected")
        ]
        rec_counts = [
            h.get("recommendations_count", 0) for h in history if h.get("recommendations_count")
        ]
        rubric_scores = [
            h.get("rubric_weighted_score", 0)
            for h in history
            if h.get("rubric_weighted_score") is not None
        ]

        return {
            "total_learning_runs": len(history),
            "total_feedback_processed": sum(feedback_counts),
            "total_patterns_detected": sum(pattern_counts),
            "total_recommendations": sum(rec_counts),
            "avg_rubric_score": sum(rubric_scores) / len(rubric_scores) if rubric_scores else 0.0,
            "successful_runs": sum(1 for h in history if h.get("completion_status") == "completed"),
            "failed_runs": sum(1 for h in history if h.get("completion_status") == "failed"),
        }

    async def get_pattern_trends(
        self,
        experiment_name: str = "default",
        max_results: int = 100,
    ) -> dict[str, Any]:
        """
        Get trends in pattern detection over time.

        Args:
            experiment_name: Experiment to analyze
            max_results: Maximum runs to analyze

        Returns:
            Dictionary with pattern trends
        """
        history = await self.get_learning_history(experiment_name, max_results=max_results)

        if not history:
            return {
                "total_runs": 0,
                "runs_with_patterns": 0,
                "runs_with_critical_patterns": 0,
                "pattern_detection_rate": 0.0,
                "critical_pattern_rate": 0.0,
            }

        runs_with_patterns = sum(1 for h in history if (h.get("patterns_detected") or 0) > 0)
        runs_with_critical = sum(1 for h in history if (h.get("critical_patterns") or 0) > 0)

        return {
            "total_runs": len(history),
            "runs_with_patterns": runs_with_patterns,
            "runs_with_critical_patterns": runs_with_critical,
            "pattern_detection_rate": runs_with_patterns / len(history) if history else 0.0,
            "critical_pattern_rate": runs_with_critical / len(history) if history else 0.0,
        }


def create_tracker(tracking_uri: Optional[str] = None) -> FeedbackLearnerMLflowTracker:
    """Factory function to create a Feedback Learner MLflow tracker."""
    return FeedbackLearnerMLflowTracker(tracking_uri=tracking_uri)
