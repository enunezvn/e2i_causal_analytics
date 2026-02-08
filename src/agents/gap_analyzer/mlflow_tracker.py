"""
MLflow Integration for Gap Analyzer Agent.

Provides comprehensive MLflow tracking for gap analysis and ROI calculations,
logging gap metrics, ROI estimates, prioritization rankings, and opportunity artifacts.

Integration Points:
    - MLflow experiment tracking (via MLflowConnector)
    - Opik tracing (via existing agent integration)
    - Dashboard metrics queries

Usage:
    tracker = GapAnalyzerMLflowTracker()
    async with tracker.start_analysis_run(
        experiment_name="roi_opportunities",
        brand="kisqali"
    ):
        output = await agent.run(input_data)
        await tracker.log_analysis_result(output, state)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.agents.gap_analyzer.state import GapAnalyzerState

logger = logging.getLogger(__name__)

# MLflow experiment prefix for this agent
EXPERIMENT_PREFIX = "e2i_causal/gap_analyzer"


@dataclass
class GapAnalysisContext:
    """Context for an MLflow gap analysis run."""

    experiment_id: str
    run_id: str
    experiment_name: str
    started_at: datetime

    # E2I specific context
    brand: Optional[str] = None
    region: Optional[str] = None
    gap_type: Optional[str] = None
    query_id: Optional[str] = None


@dataclass
class GapAnalyzerMetrics:
    """Metrics collected from gap analysis."""

    # Gap detection metrics
    total_gaps_detected: int = 0
    total_gap_value: float = 0.0
    avg_gap_percentage: float = 0.0
    max_gap_percentage: float = 0.0

    # ROI metrics
    total_addressable_value: float = 0.0
    avg_expected_roi: float = 0.0
    avg_risk_adjusted_roi: float = 0.0
    max_expected_roi: float = 0.0

    # Prioritization metrics
    n_quick_wins: int = 0
    n_strategic_bets: int = 0
    n_prioritized_opportunities: int = 0

    # Performance metrics
    detection_latency_ms: int = 0
    roi_latency_ms: int = 0
    total_latency_ms: int = 0
    segments_analyzed: int = 0

    # Quality metrics
    confidence: float = 0.0


class GapAnalyzerMLflowTracker:
    """
    Tracks Gap Analyzer Agent metrics in MLflow.

    Integrates with the MLflow infrastructure to log:
    - Gap detection results (count, values, percentages)
    - ROI estimates and calculations
    - Prioritization rankings
    - Quick wins and strategic bets

    Example:
        tracker = GapAnalyzerMLflowTracker()

        async with tracker.start_analysis_run("q4_opportunities", brand="kisqali"):
            output = await agent.run(input_data)
            await tracker.log_analysis_result(output, final_state)

        # Query historical results
        history = await tracker.get_roi_history(days=30)
    """

    EXPERIMENT_PREFIX = "e2i_causal/gap_analyzer"

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        enable_artifact_logging: bool = True,
    ):
        """
        Initialize the tracker.

        Args:
            tracking_uri: MLflow tracking server URI (default: from env)
            enable_artifact_logging: Whether to log artifacts (opportunities JSON)
        """
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.enable_artifact_logging = enable_artifact_logging

        self._current_context: Optional[GapAnalysisContext] = None
        self._mlflow_available = self._check_mlflow()

    def _check_mlflow(self) -> bool:
        """Check if MLflow is available."""
        try:
            import mlflow

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            return True
        except ImportError:
            logger.warning("MLflow not installed, metrics will be logged locally only")
            return False

    @asynccontextmanager
    async def start_analysis_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        brand: Optional[str] = None,
        region: Optional[str] = None,
        gap_type: Optional[str] = None,
        query_id: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ):
        """
        Start an MLflow run for gap analysis.

        Args:
            experiment_name: Name of the experiment (will be prefixed)
            run_name: Optional run name
            brand: E2I brand context
            region: E2I region context
            gap_type: Type of gap analysis
            query_id: Query identifier
            tags: Additional MLflow tags

        Yields:
            GapAnalysisContext for the run
        """
        full_experiment_name = f"{self.EXPERIMENT_PREFIX}/{experiment_name}"

        if self._mlflow_available:
            import mlflow

            # Set or create experiment - handle connection failures gracefully
            try:
                experiment = mlflow.get_experiment_by_name(full_experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(
                        full_experiment_name,
                        artifact_location="mlflow-artifacts:/",
                        tags={
                            "framework": "e2i_causal",
                            "agent": "gap_analyzer",
                            "tier": "2",
                        },
                    )
                else:
                    experiment_id = experiment.experiment_id
            except Exception as e:
                logger.warning(f"MLflow connection failed, continuing without tracking: {e}")
                self._mlflow_available = False
                # Fall through to else block below for dummy context

        if self._mlflow_available:
            import mlflow

            # Generate run name
            run_name = (
                run_name or f"gap_analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            )

            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name,
            ) as run:
                # Log standard tags
                mlflow.set_tag("agent_type", "gap_analyzer")
                mlflow.set_tag("agent_tier", "2")
                mlflow.set_tag("framework_version", "4.3")

                # Log context tags
                if brand:
                    mlflow.set_tag("brand", brand)
                if region:
                    mlflow.set_tag("region", region)
                if gap_type:
                    mlflow.set_tag("gap_type", gap_type)
                if query_id:
                    mlflow.set_tag("query_id", query_id)

                # Log custom tags
                for key, value in (tags or {}).items():
                    mlflow.set_tag(key, value)

                self._current_context = GapAnalysisContext(
                    experiment_id=experiment_id,
                    run_id=run.info.run_id,
                    experiment_name=experiment_name,
                    started_at=datetime.now(timezone.utc),
                    brand=brand,
                    region=region,
                    gap_type=gap_type,
                    query_id=query_id,
                )

                try:
                    yield self._current_context
                finally:
                    self._current_context = None
        else:
            # Fallback: create dummy context
            from uuid import uuid4

            self._current_context = GapAnalysisContext(
                experiment_id=str(uuid4()),
                run_id=str(uuid4()),
                experiment_name=experiment_name,
                started_at=datetime.now(timezone.utc),
                brand=brand,
                region=region,
                gap_type=gap_type,
                query_id=query_id,
            )
            try:
                yield self._current_context
            finally:
                self._current_context = None

    async def log_analysis_result(
        self,
        output: dict[str, Any],
        state: Optional["GapAnalyzerState"] = None,
        additional_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log a complete gap analysis result to MLflow.

        Args:
            output: Gap analyzer output dictionary
            state: Optional final state with detailed results
            additional_params: Additional parameters to log
        """
        if not self._mlflow_available:
            logger.debug("MLflow not available, skipping metric logging")
            return

        try:
            # Extract and log metrics
            metrics = self._extract_metrics(output, state)
            self._log_metrics(metrics)

            # Log parameters
            self._log_params(output, state, additional_params)

            # Log artifacts
            if self.enable_artifact_logging:
                await self._log_artifacts(output, state)

            logger.info(
                f"Logged gap analysis to MLflow: "
                f"gaps={metrics.total_gaps_detected}, "
                f"addressable_value=${metrics.total_addressable_value:,.0f}"
            )

        except Exception as e:
            logger.error(f"Failed to log gap analysis result to MLflow: {e}")

    def _extract_metrics(
        self,
        output: dict[str, Any],
        state: Optional["GapAnalyzerState"] = None,
    ) -> GapAnalyzerMetrics:
        """Extract metrics from output and state."""
        metrics = GapAnalyzerMetrics()

        # Gap detection metrics
        metrics.total_gap_value = output.get("total_gap_value", 0.0)
        metrics.segments_analyzed = output.get("segments_analyzed", 0)

        # ROI metrics
        metrics.total_addressable_value = output.get("total_addressable_value", 0.0)

        # Prioritization metrics
        quick_wins = output.get("quick_wins", [])
        strategic_bets = output.get("strategic_bets", [])
        prioritized = output.get("prioritized_opportunities", [])

        metrics.n_quick_wins = len(quick_wins)
        metrics.n_strategic_bets = len(strategic_bets)
        metrics.n_prioritized_opportunities = len(prioritized)
        metrics.total_gaps_detected = len(prioritized)

        # Performance metrics
        metrics.detection_latency_ms = output.get("detection_latency_ms", 0)
        metrics.roi_latency_ms = output.get("roi_latency_ms", 0)
        metrics.total_latency_ms = output.get("total_latency_ms", 0)

        # Quality metrics
        metrics.confidence = output.get("confidence", 0.0)

        # Calculate aggregate ROI metrics from state
        if state:
            roi_estimates = state.get("roi_estimates", []) or []
            if roi_estimates:
                expected_rois = [r.get("expected_roi", 0) for r in roi_estimates]
                risk_adjusted_rois = [r.get("risk_adjusted_roi", 0) for r in roi_estimates]

                metrics.avg_expected_roi = (
                    sum(expected_rois) / len(expected_rois) if expected_rois else 0.0
                )
                metrics.avg_risk_adjusted_roi = (
                    sum(risk_adjusted_rois) / len(risk_adjusted_rois) if risk_adjusted_rois else 0.0
                )
                metrics.max_expected_roi = max(expected_rois) if expected_rois else 0.0

            # Gap percentage metrics
            gaps_detected = state.get("gaps_detected", []) or []
            if gaps_detected:
                gap_percentages = [g.get("gap_percentage", 0) for g in gaps_detected]
                metrics.avg_gap_percentage = (
                    sum(gap_percentages) / len(gap_percentages) if gap_percentages else 0.0
                )
                metrics.max_gap_percentage = max(gap_percentages) if gap_percentages else 0.0

        return metrics

    def _log_metrics(self, metrics: GapAnalyzerMetrics) -> None:
        """Log metrics to MLflow."""
        import mlflow

        # Gap detection metrics
        mlflow.log_metric("total_gaps_detected", metrics.total_gaps_detected)
        mlflow.log_metric("total_gap_value", metrics.total_gap_value)
        mlflow.log_metric("avg_gap_percentage", metrics.avg_gap_percentage)
        mlflow.log_metric("max_gap_percentage", metrics.max_gap_percentage)

        # ROI metrics
        mlflow.log_metric("total_addressable_value", metrics.total_addressable_value)
        mlflow.log_metric("avg_expected_roi", metrics.avg_expected_roi)
        mlflow.log_metric("avg_risk_adjusted_roi", metrics.avg_risk_adjusted_roi)
        mlflow.log_metric("max_expected_roi", metrics.max_expected_roi)

        # Prioritization metrics
        mlflow.log_metric("n_quick_wins", metrics.n_quick_wins)
        mlflow.log_metric("n_strategic_bets", metrics.n_strategic_bets)
        mlflow.log_metric("n_prioritized_opportunities", metrics.n_prioritized_opportunities)

        # Performance metrics
        mlflow.log_metric("detection_latency_ms", metrics.detection_latency_ms)
        mlflow.log_metric("roi_latency_ms", metrics.roi_latency_ms)
        mlflow.log_metric("total_latency_ms", metrics.total_latency_ms)
        mlflow.log_metric("segments_analyzed", metrics.segments_analyzed)

        # Quality metrics
        mlflow.log_metric("confidence", metrics.confidence)

    def _log_params(
        self,
        output: dict[str, Any],
        state: Optional["GapAnalyzerState"] = None,
        additional_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log parameters to MLflow."""
        import mlflow

        # Configuration parameters from state
        if state:
            mlflow.log_param("gap_type", state.get("gap_type", "unknown"))
            mlflow.log_param("min_gap_threshold", state.get("min_gap_threshold", 5.0))
            mlflow.log_param("max_opportunities", state.get("max_opportunities", 10))
            mlflow.log_param("brand", state.get("brand", "unknown"))
            mlflow.log_param("n_metrics", len(state.get("metrics", [])))
            mlflow.log_param("n_segments", len(state.get("segments", [])))

        # Result parameters
        mlflow.log_param(
            "requires_further_analysis", output.get("requires_further_analysis", False)
        )
        if output.get("suggested_next_agent"):
            mlflow.log_param("suggested_next_agent", output["suggested_next_agent"])

        # Additional parameters
        for key, value in (additional_params or {}).items():
            mlflow.log_param(key, value)

    async def _log_artifacts(
        self,
        output: dict[str, Any],
        state: Optional["GapAnalyzerState"] = None,
    ) -> None:
        """Log artifacts to MLflow."""

        # Log prioritized opportunities
        opportunities = output.get("prioritized_opportunities", [])
        if opportunities:
            await self._log_json_artifact(
                opportunities, "prioritized_opportunities.json", "opportunities"
            )

        # Log quick wins
        quick_wins = output.get("quick_wins", [])
        if quick_wins:
            await self._log_json_artifact(quick_wins, "quick_wins.json", "opportunities")

        # Log strategic bets
        strategic_bets = output.get("strategic_bets", [])
        if strategic_bets:
            await self._log_json_artifact(strategic_bets, "strategic_bets.json", "opportunities")

        # Log detailed state if available
        if state:
            # Log ROI estimates
            roi_estimates = state.get("roi_estimates", [])
            if roi_estimates:
                await self._log_json_artifact(roi_estimates, "roi_estimates.json", "analysis")

            # Log gaps by segment
            gaps_by_segment = state.get("gaps_by_segment", {})
            if gaps_by_segment:
                await self._log_json_artifact(gaps_by_segment, "gaps_by_segment.json", "analysis")

    async def _log_json_artifact(
        self,
        data: Any,
        filename: str,
        artifact_dir: str,
    ) -> None:
        """Log a JSON artifact to MLflow."""
        import mlflow

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(data, f, indent=2, default=str)
                temp_path = f.name

            mlflow.log_artifact(temp_path, artifact_dir)
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to log artifact {filename}: {e}")

    async def get_roi_history(
        self,
        experiment_name: Optional[str] = None,
        brand: Optional[str] = None,
        days: int = 30,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get historical ROI analysis results.

        Args:
            experiment_name: Filter by experiment name
            brand: Filter by brand
            days: Number of days to look back
            limit: Maximum results to return

        Returns:
            List of historical analysis results
        """
        if not self._mlflow_available:
            return []

        try:
            import mlflow

            # Build filter string
            filters = []
            if brand:
                filters.append(f"tags.brand = '{brand}'")

            filter_string = " AND ".join(filters) if filters else ""

            # Get experiment
            if experiment_name:
                full_name = f"{self.EXPERIMENT_PREFIX}/{experiment_name}"
                experiment = mlflow.get_experiment_by_name(full_name)
                if not experiment:
                    return []
                experiment_ids = [experiment.experiment_id]
            else:
                # Search all gap analyzer experiments
                experiments = mlflow.search_experiments(
                    filter_string=f"name LIKE '{self.EXPERIMENT_PREFIX}%'"
                )
                experiment_ids = [e.experiment_id for e in experiments]

            if not experiment_ids:
                return []

            # Search runs
            runs = mlflow.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                max_results=limit,
                order_by=["start_time DESC"],
            )

            # Convert to list of dicts (runs is a DataFrame by default)
            results = []
            for _, row in runs.iterrows():  # type: ignore[union-attr]
                result = {
                    "run_id": row.get("run_id"),
                    "experiment_id": row.get("experiment_id"),
                    "start_time": row.get("start_time"),
                    "total_addressable_value": row.get("metrics.total_addressable_value"),
                    "total_gaps_detected": row.get("metrics.total_gaps_detected"),
                    "n_quick_wins": row.get("metrics.n_quick_wins"),
                    "n_strategic_bets": row.get("metrics.n_strategic_bets"),
                    "avg_expected_roi": row.get("metrics.avg_expected_roi"),
                    "confidence": row.get("metrics.confidence"),
                    "latency_ms": row.get("metrics.total_latency_ms"),
                    "brand": row.get("tags.brand"),
                    "gap_type": row.get("tags.gap_type"),
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Failed to get ROI history: {e}")
            return []

    async def get_performance_summary(
        self,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Get performance summary for the Gap Analyzer Agent.

        Returns:
            Summary including avg ROI, total addressable value, latency stats
        """
        history = await self.get_roi_history(days=days, limit=1000)

        if not history:
            return {
                "total_analyses": 0,
                "total_addressable_value_sum": 0.0,
                "avg_expected_roi": 0.0,
                "avg_latency_ms": 0.0,
            }

        total = len(history)
        addressable_values = [
            h["total_addressable_value"] for h in history if h.get("total_addressable_value")
        ]
        expected_rois = [h["avg_expected_roi"] for h in history if h.get("avg_expected_roi")]
        latencies = [h["latency_ms"] for h in history if h.get("latency_ms")]

        return {
            "total_analyses": total,
            "total_addressable_value_sum": sum(addressable_values),
            "avg_addressable_value": sum(addressable_values) / len(addressable_values)
            if addressable_values
            else 0.0,
            "avg_expected_roi": sum(expected_rois) / len(expected_rois) if expected_rois else 0.0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "analyses_by_brand": self._count_by_field(history, "brand"),
            "analyses_by_gap_type": self._count_by_field(history, "gap_type"),
        }

    def _count_by_field(self, history: list[dict[str, Any]], field: str) -> dict[str, int]:
        """Count analyses by a specific field."""
        counts: dict[str, int] = {}
        for h in history:
            value = h.get(field) or "unknown"
            counts[value] = counts.get(value, 0) + 1
        return counts


# Convenience function
def create_tracker(**kwargs) -> GapAnalyzerMLflowTracker:
    """Create a tracker with environment-based configuration."""
    return GapAnalyzerMLflowTracker(**kwargs)
