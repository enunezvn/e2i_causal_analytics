"""Health Score Agent MLflow Tracker.

This module provides MLflow experiment tracking for the Health Score Agent,
enabling monitoring of health check metrics across different scopes.

Tracked metrics:
- Overall health score and grade
- Component health scores (component, model, pipeline, agent)
- Check latency and scope
- Issue and warning counts
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import HealthScoreOutput
    from .state import HealthScoreState

logger = logging.getLogger(__name__)

# Experiment prefix for Health Score Agent
EXPERIMENT_PREFIX = "e2i_causal/health_score"


@dataclass
class HealthScoreContext:
    """Context for a health score tracking run."""

    run_id: str
    experiment_name: str
    check_scope: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class HealthScoreMetrics:
    """Structured metrics for health score tracking."""

    # Overall metrics
    overall_health_score: float = 0.0
    health_grade: str = "F"

    # Component scores (0-1 scale)
    component_health_score: float = 0.0
    model_health_score: float = 0.0
    pipeline_health_score: float = 0.0
    agent_health_score: float = 0.0

    # Issue counts
    critical_issues_count: int = 0
    warnings_count: int = 0

    # Execution metadata
    check_scope: str = "full"
    check_latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for MLflow logging."""
        return {
            "overall_health_score": self.overall_health_score,
            "health_grade_numeric": self._grade_to_numeric(self.health_grade),
            "component_health_score": self.component_health_score,
            "model_health_score": self.model_health_score,
            "pipeline_health_score": self.pipeline_health_score,
            "agent_health_score": self.agent_health_score,
            "critical_issues_count": self.critical_issues_count,
            "warnings_count": self.warnings_count,
            "check_latency_ms": self.check_latency_ms,
        }

    @staticmethod
    def _grade_to_numeric(grade: str) -> int:
        """Convert letter grade to numeric for trending."""
        grade_map = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
        return grade_map.get(grade, 0)


class HealthScoreMLflowTracker:
    """MLflow tracker for Health Score Agent.

    Provides experiment tracking for health check runs including:
    - Overall and component health scores
    - Issue and warning tracking
    - Latency monitoring
    - Historical trend analysis

    Usage:
        tracker = HealthScoreMLflowTracker()
        async with tracker.start_health_run(
            experiment_name="production",
            check_scope="full"
        ) as ctx:
            # Run health check
            output = await agent.check_health(scope="full")
            # Log results
            await tracker.log_health_result(output, state)
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize MLflow tracker.

        Args:
            tracking_uri: Optional MLflow tracking URI. If not provided,
                uses the default from environment or local storage.
        """
        self._mlflow = None
        self._tracking_uri = tracking_uri
        self._current_run_id: Optional[str] = None

    def _get_mlflow(self):
        """Lazy load MLflow to avoid import errors when not installed."""
        if self._mlflow is None:
            try:
                import mlflow

                if self._tracking_uri:
                    mlflow.set_tracking_uri(self._tracking_uri)
                self._mlflow = mlflow
            except ImportError:
                logger.warning("MLflow not installed, tracking disabled")
                return None
        return self._mlflow

    @asynccontextmanager
    async def start_health_run(
        self,
        experiment_name: str = "default",
        check_scope: str = "full",
    ) -> AsyncIterator[HealthScoreContext]:
        """Start an MLflow run for health check tracking.

        Args:
            experiment_name: Name of the experiment (e.g., "production", "staging")
            check_scope: Scope of health check ("full", "quick", "models", etc.)

        Yields:
            HealthScoreContext with run information
        """
        mlflow = self._get_mlflow()

        if mlflow is None:
            # Yield a dummy context if MLflow is not available
            yield HealthScoreContext(
                run_id="no-mlflow",
                experiment_name=experiment_name,
                check_scope=check_scope,
            )
            return

        # Create experiment if it doesn't exist
        full_experiment_name = f"{EXPERIMENT_PREFIX}/{experiment_name}"
        try:
            experiment = mlflow.get_experiment_by_name(full_experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(full_experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            logger.warning(f"Could not create/get experiment: {e}")
            yield HealthScoreContext(
                run_id="experiment-error",
                experiment_name=experiment_name,
                check_scope=check_scope,
            )
            return

        # Start MLflow run
        try:
            with mlflow.start_run(experiment_id=experiment_id) as run:
                self._current_run_id = run.info.run_id

                # Log run parameters
                mlflow.log_params(
                    {
                        "agent": "health_score",
                        "tier": 3,
                        "check_scope": check_scope,
                        "agent_type": "standard",
                    }
                )

                ctx = HealthScoreContext(
                    run_id=run.info.run_id,
                    experiment_name=experiment_name,
                    check_scope=check_scope,
                )

                yield ctx

                self._current_run_id = None

        except Exception as e:
            logger.error(f"MLflow run failed: {e}")
            self._current_run_id = None
            raise

    async def log_health_result(
        self,
        output: "HealthScoreOutput",
        state: Optional["HealthScoreState"] = None,
    ) -> None:
        """Log health check results to MLflow.

        Args:
            output: HealthScoreOutput from agent execution
            state: Optional final state for detailed logging
        """
        mlflow = self._get_mlflow()
        if mlflow is None or self._current_run_id is None:
            return

        try:
            # Create structured metrics
            metrics = HealthScoreMetrics(
                overall_health_score=output.overall_health_score,
                health_grade=output.health_grade,
                component_health_score=output.component_health_score,
                model_health_score=output.model_health_score,
                pipeline_health_score=output.pipeline_health_score,
                agent_health_score=output.agent_health_score,
                critical_issues_count=len(output.critical_issues),
                warnings_count=len(output.warnings),
                check_latency_ms=output.check_latency_ms,
            )

            # Log metrics
            mlflow.log_metrics(metrics.to_dict())

            # Log tags for filtering
            mlflow.set_tags(
                {
                    "health_grade": output.health_grade,
                    "has_critical_issues": str(len(output.critical_issues) > 0).lower(),
                    "has_warnings": str(len(output.warnings) > 0).lower(),
                }
            )

            # Log detailed results as artifact
            if state:
                artifact_data = {
                    "timestamp": output.timestamp,
                    "overall_health_score": output.overall_health_score,
                    "health_grade": output.health_grade,
                    "health_summary": output.health_summary,
                    "component_scores": {
                        "component": output.component_health_score,
                        "model": output.model_health_score,
                        "pipeline": output.pipeline_health_score,
                        "agent": output.agent_health_score,
                    },
                    "critical_issues": output.critical_issues,
                    "warnings": output.warnings,
                    "component_statuses": state.get("component_statuses", []),
                    "model_metrics": state.get("model_metrics", []),
                    "pipeline_statuses": state.get("pipeline_statuses", []),
                    "agent_statuses": state.get("agent_statuses", []),
                }

                # Write artifact
                import tempfile
                import os

                with tempfile.TemporaryDirectory() as tmpdir:
                    artifact_path = os.path.join(tmpdir, "health_check_results.json")
                    with open(artifact_path, "w") as f:
                        json.dump(artifact_data, f, indent=2, default=str)
                    mlflow.log_artifact(artifact_path)

            logger.debug(
                f"Logged health metrics to MLflow run {self._current_run_id}: "
                f"score={output.overall_health_score}, grade={output.health_grade}"
            )

        except Exception as e:
            logger.warning(f"Failed to log health metrics to MLflow: {e}")

    async def get_health_history(
        self,
        experiment_name: str = "default",
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query historical health check runs.

        Args:
            experiment_name: Name of the experiment to query
            max_results: Maximum number of results to return

        Returns:
            List of historical health check results
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            return []

        try:
            full_experiment_name = f"{EXPERIMENT_PREFIX}/{experiment_name}"
            experiment = mlflow.get_experiment_by_name(full_experiment_name)
            if experiment is None:
                return []

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"],
            )

            history = []
            for _, row in runs.iterrows():
                history.append(
                    {
                        "run_id": row["run_id"],
                        "timestamp": row["start_time"],
                        "overall_health_score": row.get("metrics.overall_health_score"),
                        "health_grade_numeric": row.get("metrics.health_grade_numeric"),
                        "component_health_score": row.get("metrics.component_health_score"),
                        "model_health_score": row.get("metrics.model_health_score"),
                        "pipeline_health_score": row.get("metrics.pipeline_health_score"),
                        "agent_health_score": row.get("metrics.agent_health_score"),
                        "critical_issues_count": row.get("metrics.critical_issues_count"),
                        "warnings_count": row.get("metrics.warnings_count"),
                        "check_latency_ms": row.get("metrics.check_latency_ms"),
                        "check_scope": row.get("params.check_scope"),
                    }
                )

            return history

        except Exception as e:
            logger.warning(f"Failed to query health history: {e}")
            return []

    async def get_health_trend(
        self,
        experiment_name: str = "default",
        hours: int = 24,
    ) -> Dict[str, Any]:
        """Get health score trend over time.

        Args:
            experiment_name: Name of the experiment to query
            hours: Number of hours to look back

        Returns:
            Dictionary with trend analysis
        """
        history = await self.get_health_history(experiment_name, max_results=1000)

        if not history:
            return {"trend": "unknown", "data_points": 0}

        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Filter to time window
        recent = [
            h
            for h in history
            if h.get("timestamp")
            and (
                isinstance(h["timestamp"], datetime)
                and h["timestamp"].replace(tzinfo=timezone.utc) > cutoff
            )
        ]

        if len(recent) < 2:
            return {
                "trend": "insufficient_data",
                "data_points": len(recent),
            }

        # Calculate trend
        scores = [
            h["overall_health_score"]
            for h in recent
            if h.get("overall_health_score") is not None
        ]

        if not scores:
            return {"trend": "no_scores", "data_points": 0}

        avg_score = sum(scores) / len(scores)
        first_half = scores[: len(scores) // 2]
        second_half = scores[len(scores) // 2 :]

        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0

        if second_avg > first_avg + 5:
            trend = "improving"
        elif second_avg < first_avg - 5:
            trend = "degrading"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "data_points": len(recent),
            "avg_score": avg_score,
            "min_score": min(scores),
            "max_score": max(scores),
            "latest_score": scores[0] if scores else None,
        }


def create_tracker(tracking_uri: Optional[str] = None) -> HealthScoreMLflowTracker:
    """Factory function to create a Health Score MLflow tracker.

    Args:
        tracking_uri: Optional MLflow tracking URI

    Returns:
        Configured HealthScoreMLflowTracker instance
    """
    return HealthScoreMLflowTracker(tracking_uri=tracking_uri)
