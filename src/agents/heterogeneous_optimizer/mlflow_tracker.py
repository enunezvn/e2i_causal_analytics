"""
MLflow Integration for Heterogeneous Optimizer Agent.

Provides comprehensive MLflow tracking for CATE analysis and treatment
heterogeneity, logging segment-level effects, uplift metrics, and
policy recommendations.

Integration Points:
    - MLflow experiment tracking (via MLflowConnector)
    - Opik tracing (via existing agent integration)
    - Dashboard metrics queries

Usage:
    tracker = HeterogeneousOptimizerMLflowTracker()
    async with tracker.start_analysis_run(
        experiment_name="segment_optimization",
        brand="remibrutinib"
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
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.heterogeneous_optimizer.state import HeterogeneousOptimizerState

logger = logging.getLogger(__name__)

# MLflow experiment prefix for this agent
EXPERIMENT_PREFIX = "e2i_causal/heterogeneous_optimizer"


@dataclass
class HeterogeneousOptimizerContext:
    """Context for an MLflow CATE analysis run."""

    experiment_id: str
    run_id: str
    experiment_name: str
    started_at: datetime

    # E2I specific context
    brand: Optional[str] = None
    region: Optional[str] = None
    treatment_var: Optional[str] = None
    outcome_var: Optional[str] = None
    query_id: Optional[str] = None


@dataclass
class HeterogeneousOptimizerMetrics:
    """Metrics collected from heterogeneous treatment effect analysis."""

    # CATE metrics
    overall_ate: Optional[float] = None
    heterogeneity_score: Optional[float] = None
    avg_cate: Optional[float] = None
    max_cate: Optional[float] = None
    min_cate: Optional[float] = None

    # Segment metrics
    n_segments_analyzed: int = 0
    n_high_responders: int = 0
    n_low_responders: int = 0
    n_significant_segments: int = 0

    # Uplift metrics
    overall_auuc: Optional[float] = None
    overall_qini: Optional[float] = None
    targeting_efficiency: Optional[float] = None

    # Policy metrics
    n_policy_recommendations: int = 0
    expected_total_lift: Optional[float] = None

    # Performance metrics
    estimation_latency_ms: int = 0
    analysis_latency_ms: int = 0
    total_latency_ms: int = 0

    # Quality metrics
    confidence: float = 0.0


class HeterogeneousOptimizerMLflowTracker:
    """
    Tracks Heterogeneous Optimizer Agent metrics in MLflow.

    Integrates with the MLflow infrastructure to log:
    - CATE estimates across segments
    - Treatment heterogeneity scores
    - Uplift metrics (AUUC, Qini)
    - Policy recommendations

    Example:
        tracker = HeterogeneousOptimizerMLflowTracker()

        async with tracker.start_analysis_run("segment_cate", brand="kisqali"):
            output = await agent.run(input_data)
            await tracker.log_analysis_result(output, final_state)

        # Query historical results
        history = await tracker.get_cate_history(days=30)
    """

    EXPERIMENT_PREFIX = "e2i_causal/heterogeneous_optimizer"

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        enable_artifact_logging: bool = True,
    ):
        """
        Initialize the tracker.

        Args:
            tracking_uri: MLflow tracking server URI (default: from env)
            enable_artifact_logging: Whether to log artifacts (segments JSON)
        """
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.enable_artifact_logging = enable_artifact_logging

        self._current_context: Optional[HeterogeneousOptimizerContext] = None
        self._mlflow_available = self._check_mlflow()

    def _check_mlflow(self) -> bool:
        """Check if MLflow is available."""
        try:
            import mlflow

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
        treatment_var: Optional[str] = None,
        outcome_var: Optional[str] = None,
        query_id: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ):
        """
        Start an MLflow run for heterogeneous treatment effect analysis.

        Args:
            experiment_name: Name of the experiment (will be prefixed)
            run_name: Optional run name
            brand: E2I brand context
            region: E2I region context
            treatment_var: Treatment variable name
            outcome_var: Outcome variable name
            query_id: Query identifier
            tags: Additional MLflow tags

        Yields:
            HeterogeneousOptimizerContext for the run
        """
        full_experiment_name = f"{self.EXPERIMENT_PREFIX}/{experiment_name}"

        if self._mlflow_available:
            import mlflow

            # Set or create experiment
            experiment = mlflow.get_experiment_by_name(full_experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    full_experiment_name,
                    tags={
                        "framework": "e2i_causal",
                        "agent": "heterogeneous_optimizer",
                        "tier": "2",
                    },
                )
            else:
                experiment_id = experiment.experiment_id

            # Generate run name
            run_name = (
                run_name or f"cate_analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            )

            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name,
            ) as run:
                # Log standard tags
                mlflow.set_tag("agent_type", "heterogeneous_optimizer")
                mlflow.set_tag("agent_tier", "2")
                mlflow.set_tag("framework_version", "4.3")

                # Log context tags
                if brand:
                    mlflow.set_tag("brand", brand)
                if region:
                    mlflow.set_tag("region", region)
                if treatment_var:
                    mlflow.set_tag("treatment_var", treatment_var)
                if outcome_var:
                    mlflow.set_tag("outcome_var", outcome_var)
                if query_id:
                    mlflow.set_tag("query_id", query_id)

                # Log custom tags
                for key, value in (tags or {}).items():
                    mlflow.set_tag(key, value)

                self._current_context = HeterogeneousOptimizerContext(
                    experiment_id=experiment_id,
                    run_id=run.info.run_id,
                    experiment_name=experiment_name,
                    started_at=datetime.now(timezone.utc),
                    brand=brand,
                    region=region,
                    treatment_var=treatment_var,
                    outcome_var=outcome_var,
                    query_id=query_id,
                )

                try:
                    yield self._current_context
                finally:
                    self._current_context = None
        else:
            # Fallback: create dummy context
            from uuid import uuid4

            self._current_context = HeterogeneousOptimizerContext(
                experiment_id=str(uuid4()),
                run_id=str(uuid4()),
                experiment_name=experiment_name,
                started_at=datetime.now(timezone.utc),
                brand=brand,
                region=region,
                treatment_var=treatment_var,
                outcome_var=outcome_var,
                query_id=query_id,
            )
            try:
                yield self._current_context
            finally:
                self._current_context = None

    async def log_analysis_result(
        self,
        output: dict[str, Any],
        state: Optional["HeterogeneousOptimizerState"] = None,
        additional_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log a complete CATE analysis result to MLflow.

        Args:
            output: Heterogeneous optimizer output dictionary
            state: Optional final state with detailed results
            additional_params: Additional parameters to log
        """
        if not self._mlflow_available:
            logger.debug("MLflow not available, skipping metric logging")
            return

        import mlflow

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
                f"Logged heterogeneous optimizer analysis to MLflow: "
                f"heterogeneity_score={metrics.heterogeneity_score}, "
                f"n_segments={metrics.n_segments_analyzed}"
            )

        except Exception as e:
            logger.error(f"Failed to log CATE analysis result to MLflow: {e}")

    def _extract_metrics(
        self,
        output: dict[str, Any],
        state: Optional["HeterogeneousOptimizerState"] = None,
    ) -> HeterogeneousOptimizerMetrics:
        """Extract metrics from output and state."""
        metrics = HeterogeneousOptimizerMetrics()

        # Extract from output
        metrics.overall_ate = output.get("overall_ate")
        metrics.heterogeneity_score = output.get("heterogeneity_score")
        metrics.expected_total_lift = output.get("expected_total_lift")
        metrics.confidence = output.get("confidence", 0.0)

        # Segment counts from output
        high_responders = output.get("high_responders", [])
        low_responders = output.get("low_responders", [])
        policy_recommendations = output.get("policy_recommendations", [])

        metrics.n_high_responders = len(high_responders)
        metrics.n_low_responders = len(low_responders)
        metrics.n_policy_recommendations = len(policy_recommendations)

        # Performance metrics
        metrics.estimation_latency_ms = output.get("estimation_latency_ms", 0)
        metrics.analysis_latency_ms = output.get("analysis_latency_ms", 0)
        metrics.total_latency_ms = output.get("total_latency_ms", 0)

        # Extract detailed metrics from state
        if state:
            # CATE aggregates
            cate_by_segment = state.get("cate_by_segment", {}) or {}
            all_cates = []
            for segment_results in cate_by_segment.values():
                for result in segment_results:
                    if result.get("cate_estimate") is not None:
                        all_cates.append(result["cate_estimate"])
                        if result.get("statistical_significance"):
                            metrics.n_significant_segments += 1

            if all_cates:
                metrics.avg_cate = sum(all_cates) / len(all_cates)
                metrics.max_cate = max(all_cates)
                metrics.min_cate = min(all_cates)

            metrics.n_segments_analyzed = state.get("n_segments_analyzed") or len(all_cates)

            # Uplift metrics
            metrics.overall_auuc = state.get("overall_auuc")
            metrics.overall_qini = state.get("overall_qini")
            metrics.targeting_efficiency = state.get("targeting_efficiency")

        return metrics

    def _log_metrics(self, metrics: HeterogeneousOptimizerMetrics) -> None:
        """Log metrics to MLflow."""
        import mlflow

        # CATE metrics
        if metrics.overall_ate is not None:
            mlflow.log_metric("overall_ate", metrics.overall_ate)
        if metrics.heterogeneity_score is not None:
            mlflow.log_metric("heterogeneity_score", metrics.heterogeneity_score)
        if metrics.avg_cate is not None:
            mlflow.log_metric("avg_cate", metrics.avg_cate)
        if metrics.max_cate is not None:
            mlflow.log_metric("max_cate", metrics.max_cate)
        if metrics.min_cate is not None:
            mlflow.log_metric("min_cate", metrics.min_cate)

        # Segment metrics
        mlflow.log_metric("n_segments_analyzed", metrics.n_segments_analyzed)
        mlflow.log_metric("n_high_responders", metrics.n_high_responders)
        mlflow.log_metric("n_low_responders", metrics.n_low_responders)
        mlflow.log_metric("n_significant_segments", metrics.n_significant_segments)

        # Uplift metrics
        if metrics.overall_auuc is not None:
            mlflow.log_metric("overall_auuc", metrics.overall_auuc)
        if metrics.overall_qini is not None:
            mlflow.log_metric("overall_qini", metrics.overall_qini)
        if metrics.targeting_efficiency is not None:
            mlflow.log_metric("targeting_efficiency", metrics.targeting_efficiency)

        # Policy metrics
        mlflow.log_metric("n_policy_recommendations", metrics.n_policy_recommendations)
        if metrics.expected_total_lift is not None:
            mlflow.log_metric("expected_total_lift", metrics.expected_total_lift)

        # Performance metrics
        mlflow.log_metric("estimation_latency_ms", metrics.estimation_latency_ms)
        mlflow.log_metric("analysis_latency_ms", metrics.analysis_latency_ms)
        mlflow.log_metric("total_latency_ms", metrics.total_latency_ms)

        # Quality metrics
        mlflow.log_metric("confidence", metrics.confidence)

    def _log_params(
        self,
        output: dict[str, Any],
        state: Optional["HeterogeneousOptimizerState"] = None,
        additional_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log parameters to MLflow."""
        import mlflow

        # Configuration parameters from state
        if state:
            mlflow.log_param("treatment_var", state.get("treatment_var", "unknown"))
            mlflow.log_param("outcome_var", state.get("outcome_var", "unknown"))
            mlflow.log_param("n_segment_vars", len(state.get("segment_vars", [])))
            mlflow.log_param("n_effect_modifiers", len(state.get("effect_modifiers", [])))
            mlflow.log_param("n_estimators", state.get("n_estimators", 100))
            mlflow.log_param("min_samples_leaf", state.get("min_samples_leaf", 10))
            mlflow.log_param("significance_level", state.get("significance_level", 0.05))

            if state.get("brand"):
                mlflow.log_param("brand", state["brand"])

            # Model parameters
            if state.get("model_type_used"):
                mlflow.log_param("model_type_used", state["model_type_used"])
            if state.get("primary_library"):
                mlflow.log_param("primary_library", state["primary_library"])

        # Result parameters
        mlflow.log_param("requires_further_analysis", output.get("requires_further_analysis", False))
        if output.get("suggested_next_agent"):
            mlflow.log_param("suggested_next_agent", output["suggested_next_agent"])

        # Additional parameters
        for key, value in (additional_params or {}).items():
            mlflow.log_param(key, value)

    async def _log_artifacts(
        self,
        output: dict[str, Any],
        state: Optional["HeterogeneousOptimizerState"] = None,
    ) -> None:
        """Log artifacts to MLflow."""
        import mlflow

        # Log high responders
        high_responders = output.get("high_responders", [])
        if high_responders:
            await self._log_json_artifact(
                high_responders, "high_responders.json", "segments"
            )

        # Log low responders
        low_responders = output.get("low_responders", [])
        if low_responders:
            await self._log_json_artifact(
                low_responders, "low_responders.json", "segments"
            )

        # Log policy recommendations
        policy = output.get("policy_recommendations", [])
        if policy:
            await self._log_json_artifact(
                policy, "policy_recommendations.json", "policy"
            )

        # Log detailed state if available
        if state:
            # Log CATE by segment
            cate_by_segment = state.get("cate_by_segment", {})
            if cate_by_segment:
                await self._log_json_artifact(
                    cate_by_segment, "cate_by_segment.json", "analysis"
                )

            # Log feature importance
            feature_importance = state.get("feature_importance", {})
            if feature_importance:
                await self._log_json_artifact(
                    feature_importance, "feature_importance.json", "analysis"
                )

            # Log uplift by segment
            uplift_by_segment = state.get("uplift_by_segment", {})
            if uplift_by_segment:
                await self._log_json_artifact(
                    uplift_by_segment, "uplift_by_segment.json", "uplift"
                )

    async def _log_json_artifact(
        self,
        data: Any,
        filename: str,
        artifact_dir: str,
    ) -> None:
        """Log a JSON artifact to MLflow."""
        import mlflow

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(data, f, indent=2, default=str)
                temp_path = f.name

            mlflow.log_artifact(temp_path, artifact_dir)
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to log artifact {filename}: {e}")

    async def get_cate_history(
        self,
        experiment_name: Optional[str] = None,
        brand: Optional[str] = None,
        days: int = 30,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get historical CATE analysis results.

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

            filter_string = " AND ".join(filters) if filters else None

            # Get experiment
            if experiment_name:
                full_name = f"{self.EXPERIMENT_PREFIX}/{experiment_name}"
                experiment = mlflow.get_experiment_by_name(full_name)
                if not experiment:
                    return []
                experiment_ids = [experiment.experiment_id]
            else:
                # Search all heterogeneous optimizer experiments
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

            # Convert to list of dicts
            results = []
            for _, row in runs.iterrows():
                result = {
                    "run_id": row.get("run_id"),
                    "experiment_id": row.get("experiment_id"),
                    "start_time": row.get("start_time"),
                    "overall_ate": row.get("metrics.overall_ate"),
                    "heterogeneity_score": row.get("metrics.heterogeneity_score"),
                    "n_segments_analyzed": row.get("metrics.n_segments_analyzed"),
                    "n_high_responders": row.get("metrics.n_high_responders"),
                    "n_low_responders": row.get("metrics.n_low_responders"),
                    "overall_auuc": row.get("metrics.overall_auuc"),
                    "targeting_efficiency": row.get("metrics.targeting_efficiency"),
                    "confidence": row.get("metrics.confidence"),
                    "latency_ms": row.get("metrics.total_latency_ms"),
                    "brand": row.get("tags.brand"),
                    "treatment_var": row.get("tags.treatment_var"),
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Failed to get CATE history: {e}")
            return []

    async def get_performance_summary(
        self,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Get performance summary for the Heterogeneous Optimizer Agent.

        Returns:
            Summary including avg heterogeneity, segment counts, latency stats
        """
        history = await self.get_cate_history(days=days, limit=1000)

        if not history:
            return {
                "total_analyses": 0,
                "avg_heterogeneity_score": 0.0,
                "avg_latency_ms": 0.0,
            }

        total = len(history)
        heterogeneity_scores = [h["heterogeneity_score"] for h in history if h.get("heterogeneity_score")]
        targeting_efficiencies = [h["targeting_efficiency"] for h in history if h.get("targeting_efficiency")]
        latencies = [h["latency_ms"] for h in history if h.get("latency_ms")]
        segments = [h["n_segments_analyzed"] for h in history if h.get("n_segments_analyzed")]

        return {
            "total_analyses": total,
            "avg_heterogeneity_score": sum(heterogeneity_scores) / len(heterogeneity_scores) if heterogeneity_scores else 0.0,
            "avg_targeting_efficiency": sum(targeting_efficiencies) / len(targeting_efficiencies) if targeting_efficiencies else 0.0,
            "avg_segments_analyzed": sum(segments) / len(segments) if segments else 0.0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "analyses_by_brand": self._count_by_field(history, "brand"),
        }

    def _count_by_field(
        self, history: list[dict[str, Any]], field: str
    ) -> dict[str, int]:
        """Count analyses by a specific field."""
        counts: dict[str, int] = {}
        for h in history:
            value = h.get(field) or "unknown"
            counts[value] = counts.get(value, 0) + 1
        return counts


# Convenience function
def create_tracker(**kwargs) -> HeterogeneousOptimizerMLflowTracker:
    """Create a tracker with environment-based configuration."""
    return HeterogeneousOptimizerMLflowTracker(**kwargs)
