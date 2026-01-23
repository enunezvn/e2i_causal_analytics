"""
MLflow Integration for Explainer Agent.

Provides comprehensive MLflow tracking for natural language explanation generation,
logging insight extraction metrics, narrative quality indicators, and audience
adaptation statistics.

Integration Points:
    - MLflow experiment tracking (via MLflowConnector)
    - Opik tracing (via existing agent integration)
    - Dashboard metrics queries

Usage:
    tracker = ExplainerMLflowTracker()
    async with tracker.start_explanation_run(
        experiment_name="quarterly_review",
        user_expertise="executive"
    ):
        output = await agent.run(input_data)
        await tracker.log_explanation_result(output, state)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .state import ExplainerState

logger = logging.getLogger(__name__)

# Experiment prefix for Explainer Agent
EXPERIMENT_PREFIX = "e2i_causal/explainer"


@dataclass
class ExplanationContext:
    """Context for an MLflow explanation run."""

    run_id: str
    experiment_name: str
    user_expertise: str
    output_format: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Optional context
    brand: Optional[str] = None
    region: Optional[str] = None
    source_agents: Optional[list[str]] = None


@dataclass
class ExplainerMetrics:
    """Structured metrics for explainer tracking."""

    # Insight metrics
    insight_count: int = 0
    findings_count: int = 0
    recommendations_count: int = 0
    warnings_count: int = 0
    opportunities_count: int = 0

    # Insight quality
    avg_insight_confidence: float = 0.0
    high_priority_count: int = 0
    immediate_actionable_count: int = 0

    # Narrative metrics
    narrative_section_count: int = 0
    executive_summary_length: int = 0
    detailed_explanation_length: int = 0

    # Source metrics
    source_agents_count: int = 0
    analysis_results_count: int = 0

    # Supplementary metrics
    visual_suggestions_count: int = 0
    follow_up_questions_count: int = 0

    # Latency metrics
    assembly_latency_ms: int = 0
    reasoning_latency_ms: int = 0
    generation_latency_ms: int = 0
    total_latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for MLflow logging."""
        return {
            "insight_count": self.insight_count,
            "findings_count": self.findings_count,
            "recommendations_count": self.recommendations_count,
            "warnings_count": self.warnings_count,
            "opportunities_count": self.opportunities_count,
            "avg_insight_confidence": self.avg_insight_confidence,
            "high_priority_count": self.high_priority_count,
            "immediate_actionable_count": self.immediate_actionable_count,
            "narrative_section_count": self.narrative_section_count,
            "executive_summary_length": self.executive_summary_length,
            "detailed_explanation_length": self.detailed_explanation_length,
            "source_agents_count": self.source_agents_count,
            "analysis_results_count": self.analysis_results_count,
            "visual_suggestions_count": self.visual_suggestions_count,
            "follow_up_questions_count": self.follow_up_questions_count,
            "assembly_latency_ms": self.assembly_latency_ms,
            "reasoning_latency_ms": self.reasoning_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "total_latency_ms": self.total_latency_ms,
        }


class ExplainerMLflowTracker:
    """
    Tracks Explainer Agent metrics in MLflow.

    Integrates with MLflow to log:
    - Insight extraction metrics
    - Narrative generation statistics
    - Audience adaptation parameters
    - Source agent coverage
    - Performance metrics

    Example:
        tracker = ExplainerMLflowTracker()

        async with tracker.start_explanation_run("quarterly", user_expertise="executive"):
            output = await agent.run(input_data)
            await tracker.log_explanation_result(output, final_state)

        # Query historical results
        history = await tracker.get_explanation_history(days=30)
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
    async def start_explanation_run(
        self,
        experiment_name: str = "default",
        user_expertise: str = "analyst",
        output_format: str = "narrative",
        source_agents: Optional[list[str]] = None,
        brand: Optional[str] = None,
        region: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> AsyncIterator[ExplanationContext]:
        """
        Start an MLflow run for explanation tracking.

        Args:
            experiment_name: Name of the experiment
            user_expertise: Target audience expertise level
            output_format: Desired output format
            source_agents: List of source agent names
            brand: E2I brand context
            region: E2I region context
            tags: Additional MLflow tags

        Yields:
            ExplanationContext with run information
        """
        mlflow = self._get_mlflow()

        if mlflow is None:
            yield ExplanationContext(
                run_id="no-mlflow",
                experiment_name=experiment_name,
                user_expertise=user_expertise,
                output_format=output_format,
                brand=brand,
                region=region,
                source_agents=source_agents,
            )
            return

        full_experiment_name = f"{EXPERIMENT_PREFIX}/{experiment_name}"

        try:
            experiment = mlflow.get_experiment_by_name(full_experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    full_experiment_name,
                    tags={
                        "framework": "e2i_causal",
                        "agent": "explainer",
                        "tier": "5",
                    },
                )
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            logger.warning(f"Could not create/get experiment: {e}")
            yield ExplanationContext(
                run_id="experiment-error",
                experiment_name=experiment_name,
                user_expertise=user_expertise,
                output_format=output_format,
            )
            return

        try:
            with mlflow.start_run(experiment_id=experiment_id) as run:
                self._current_run_id = run.info.run_id

                # Log run parameters
                mlflow.log_params({
                    "agent": "explainer",
                    "tier": 5,
                    "user_expertise": user_expertise,
                    "output_format": output_format,
                })

                # Log source agents if provided
                if source_agents:
                    mlflow.log_param("source_agents", ",".join(source_agents))
                    mlflow.log_param("source_agents_count", len(source_agents))

                # Log context tags
                mlflow.set_tags({
                    "agent_type": "self_improvement",
                    "framework_version": "4.3",
                })
                if brand:
                    mlflow.set_tag("brand", brand)
                if region:
                    mlflow.set_tag("region", region)

                # Custom tags
                for key, value in (tags or {}).items():
                    mlflow.set_tag(key, value)

                ctx = ExplanationContext(
                    run_id=run.info.run_id,
                    experiment_name=experiment_name,
                    user_expertise=user_expertise,
                    output_format=output_format,
                    brand=brand,
                    region=region,
                    source_agents=source_agents,
                )

                yield ctx

                self._current_run_id = None

        except Exception as e:
            logger.error(f"MLflow run failed: {e}")
            self._current_run_id = None
            raise

    async def log_explanation_result(
        self,
        state: "ExplainerState",
    ) -> None:
        """
        Log explanation results to MLflow.

        Args:
            state: Final ExplainerState from agent execution
        """
        mlflow = self._get_mlflow()
        if mlflow is None or self._current_run_id is None:
            return

        try:
            # Extract metrics from state
            metrics = self._extract_metrics(state)
            mlflow.log_metrics(metrics.to_dict())

            # Log quality tags
            mlflow.set_tags({
                "high_insight_count": str(metrics.insight_count >= 5).lower(),
                "has_recommendations": str(metrics.recommendations_count > 0).lower(),
                "has_actionables": str(metrics.immediate_actionable_count > 0).lower(),
                "multi_source": str(metrics.source_agents_count > 1).lower(),
            })

            # Log artifacts
            if self.enable_artifact_logging:
                await self._log_artifacts(state)

            logger.debug(
                f"Logged explanation metrics to MLflow run {self._current_run_id}: "
                f"insights={metrics.insight_count}, sections={metrics.narrative_section_count}"
            )

        except Exception as e:
            logger.warning(f"Failed to log explanation metrics to MLflow: {e}")

    def _extract_metrics(
        self,
        state: "ExplainerState",
    ) -> ExplainerMetrics:
        """Extract metrics from state."""
        metrics = ExplainerMetrics()

        # Insight metrics
        insights = state.get("extracted_insights", []) or []
        metrics.insight_count = len(insights)

        # Count by category
        for insight in insights:
            category = insight.get("category", "")
            if category == "finding":
                metrics.findings_count += 1
            elif category == "recommendation":
                metrics.recommendations_count += 1
            elif category == "warning":
                metrics.warnings_count += 1
            elif category == "opportunity":
                metrics.opportunities_count += 1

            # Priority and actionability
            if insight.get("priority", 999) <= 2:
                metrics.high_priority_count += 1
            if insight.get("actionability") == "immediate":
                metrics.immediate_actionable_count += 1

        # Average confidence
        confidences = [i.get("confidence", 0.0) for i in insights if i.get("confidence")]
        if confidences:
            metrics.avg_insight_confidence = sum(confidences) / len(confidences)

        # Narrative metrics
        sections = state.get("narrative_sections", []) or []
        metrics.narrative_section_count = len(sections)

        executive_summary = state.get("executive_summary", "") or ""
        metrics.executive_summary_length = len(executive_summary)

        detailed = state.get("detailed_explanation", "") or ""
        metrics.detailed_explanation_length = len(detailed)

        # Source metrics
        analysis_context = state.get("analysis_context", []) or []
        source_agents = set()
        for ctx in analysis_context:
            if ctx.get("source_agent"):
                source_agents.add(ctx["source_agent"])
        metrics.source_agents_count = len(source_agents)

        analysis_results = state.get("analysis_results", []) or []
        metrics.analysis_results_count = len(analysis_results)

        # Supplementary metrics
        visuals = state.get("visual_suggestions", []) or []
        metrics.visual_suggestions_count = len(visuals)

        follow_ups = state.get("follow_up_questions", []) or []
        metrics.follow_up_questions_count = len(follow_ups)

        # Latency metrics
        metrics.assembly_latency_ms = state.get("assembly_latency_ms", 0)
        metrics.reasoning_latency_ms = state.get("reasoning_latency_ms", 0)
        metrics.generation_latency_ms = state.get("generation_latency_ms", 0)
        metrics.total_latency_ms = state.get("total_latency_ms", 0)

        return metrics

    async def _log_artifacts(
        self,
        state: "ExplainerState",
    ) -> None:
        """Log artifacts to MLflow."""
        mlflow = self._get_mlflow()
        if mlflow is None:
            return

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Log extracted insights
                insights = state.get("extracted_insights", [])
                if insights:
                    insights_path = os.path.join(tmpdir, "extracted_insights.json")
                    with open(insights_path, "w") as f:
                        json.dump(insights, f, indent=2, default=str)
                    mlflow.log_artifact(insights_path, "insights")

                # Log narrative sections
                sections = state.get("narrative_sections", [])
                if sections:
                    sections_path = os.path.join(tmpdir, "narrative_sections.json")
                    with open(sections_path, "w") as f:
                        json.dump(sections, f, indent=2, default=str)
                    mlflow.log_artifact(sections_path, "narrative")

                # Log executive summary
                executive_summary = state.get("executive_summary")
                if executive_summary:
                    summary_path = os.path.join(tmpdir, "executive_summary.txt")
                    with open(summary_path, "w") as f:
                        f.write(executive_summary)
                    mlflow.log_artifact(summary_path, "narrative")

                # Log visual suggestions
                visuals = state.get("visual_suggestions", [])
                if visuals:
                    visuals_path = os.path.join(tmpdir, "visual_suggestions.json")
                    with open(visuals_path, "w") as f:
                        json.dump(visuals, f, indent=2, default=str)
                    mlflow.log_artifact(visuals_path, "supplementary")

                # Log follow-up questions
                follow_ups = state.get("follow_up_questions", [])
                if follow_ups:
                    followups_path = os.path.join(tmpdir, "follow_up_questions.json")
                    with open(followups_path, "w") as f:
                        json.dump(follow_ups, f, indent=2, default=str)
                    mlflow.log_artifact(followups_path, "supplementary")

        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")

    async def get_explanation_history(
        self,
        experiment_name: str = "default",
        user_expertise: Optional[str] = None,
        output_format: Optional[str] = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query historical explanation runs.

        Args:
            experiment_name: Name of the experiment to query
            user_expertise: Filter by user expertise level
            output_format: Filter by output format
            max_results: Maximum number of results to return

        Returns:
            List of historical explanation results
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
            if user_expertise:
                filters.append(f"params.user_expertise = '{user_expertise}'")
            if output_format:
                filters.append(f"params.output_format = '{output_format}'")

            filter_string = " AND ".join(filters) if filters else None

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                order_by=["start_time DESC"],
            )

            history = []
            for _, row in runs.iterrows():
                history.append({
                    "run_id": row["run_id"],
                    "timestamp": row["start_time"],
                    "insight_count": row.get("metrics.insight_count"),
                    "recommendations_count": row.get("metrics.recommendations_count"),
                    "narrative_section_count": row.get("metrics.narrative_section_count"),
                    "avg_insight_confidence": row.get("metrics.avg_insight_confidence"),
                    "total_latency_ms": row.get("metrics.total_latency_ms"),
                    "user_expertise": row.get("params.user_expertise"),
                    "output_format": row.get("params.output_format"),
                    "source_agents_count": row.get("metrics.source_agents_count"),
                })

            return history

        except Exception as e:
            logger.warning(f"Failed to query explanation history: {e}")
            return []

    async def get_insight_summary(
        self,
        experiment_name: str = "default",
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Get summary of insight generation over time.

        Args:
            experiment_name: Experiment to analyze
            days: Number of days to look back

        Returns:
            Dictionary with insight summary
        """
        history = await self.get_explanation_history(experiment_name, max_results=1000)

        if not history:
            return {
                "total_explanations": 0,
                "total_insights": 0,
                "avg_insights_per_run": 0.0,
                "avg_confidence": 0.0,
            }

        insight_counts = [h.get("insight_count", 0) for h in history if h.get("insight_count")]
        confidences = [
            h.get("avg_insight_confidence", 0)
            for h in history
            if h.get("avg_insight_confidence")
        ]

        return {
            "total_explanations": len(history),
            "total_insights": sum(insight_counts),
            "avg_insights_per_run": sum(insight_counts) / len(insight_counts) if insight_counts else 0.0,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "by_user_expertise": self._count_by_field(history, "user_expertise"),
            "by_output_format": self._count_by_field(history, "output_format"),
        }

    def _count_by_field(
        self, history: list[dict[str, Any]], field: str
    ) -> dict[str, int]:
        """Count records by a specific field."""
        counts: dict[str, int] = {}
        for h in history:
            value = h.get(field) or "unknown"
            counts[value] = counts.get(value, 0) + 1
        return counts


def create_tracker(tracking_uri: Optional[str] = None) -> ExplainerMLflowTracker:
    """Factory function to create an Explainer MLflow tracker."""
    return ExplainerMLflowTracker(tracking_uri=tracking_uri)
