"""
MLflow Integration for Resource Optimizer Agent.

Provides comprehensive MLflow tracking for resource allocation optimization,
logging objective values, allocation changes, ROI projections, and solver
performance metrics.

Integration Points:
    - MLflow experiment tracking (via MLflowConnector)
    - Opik tracing (via existing agent integration)
    - Dashboard metrics queries

Usage:
    tracker = ResourceOptimizerMLflowTracker()
    async with tracker.start_optimization_run(
        experiment_name="budget_allocation",
        resource_type="budget"
    ):
        output = await agent.run(input_data)
        await tracker.log_optimization_result(output, state)
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
    from .state import ResourceOptimizerState

logger = logging.getLogger(__name__)

# Experiment prefix for Resource Optimizer Agent
EXPERIMENT_PREFIX = "e2i_causal/resource_optimizer"


@dataclass
class OptimizationContext:
    """Context for an MLflow optimization run."""

    run_id: str
    experiment_name: str
    resource_type: str
    objective: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Optional context
    brand: Optional[str] = None
    region: Optional[str] = None


@dataclass
class ResourceOptimizerMetrics:
    """Structured metrics for resource optimizer tracking."""

    # Optimization results
    objective_value: Optional[float] = None
    projected_outcome: Optional[float] = None
    projected_roi: Optional[float] = None

    # Allocation changes
    entities_optimized: int = 0
    entities_increased: int = 0
    entities_decreased: int = 0
    entities_unchanged: int = 0
    total_allocation_change: float = 0.0
    avg_change_percentage: float = 0.0

    # Solver metrics
    solver_status: str = "unknown"
    solve_time_ms: int = 0
    constraint_violations: int = 0

    # Scenario metrics (if applicable)
    scenarios_analyzed: int = 0
    best_scenario_roi: Optional[float] = None

    # Latency metrics
    formulation_latency_ms: int = 0
    optimization_latency_ms: int = 0
    total_latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for MLflow logging."""
        metrics = {
            "entities_optimized": self.entities_optimized,
            "entities_increased": self.entities_increased,
            "entities_decreased": self.entities_decreased,
            "entities_unchanged": self.entities_unchanged,
            "total_allocation_change": self.total_allocation_change,
            "avg_change_percentage": self.avg_change_percentage,
            "solve_time_ms": self.solve_time_ms,
            "constraint_violations": self.constraint_violations,
            "scenarios_analyzed": self.scenarios_analyzed,
            "formulation_latency_ms": self.formulation_latency_ms,
            "optimization_latency_ms": self.optimization_latency_ms,
            "total_latency_ms": self.total_latency_ms,
        }

        # Add optional metrics
        if self.objective_value is not None:
            metrics["objective_value"] = self.objective_value
        if self.projected_outcome is not None:
            metrics["projected_outcome"] = self.projected_outcome
        if self.projected_roi is not None:
            metrics["projected_roi"] = self.projected_roi
        if self.best_scenario_roi is not None:
            metrics["best_scenario_roi"] = self.best_scenario_roi

        return metrics


class ResourceOptimizerMLflowTracker:
    """
    Tracks Resource Optimizer Agent metrics in MLflow.

    Integrates with MLflow to log:
    - Objective function values
    - Allocation changes by entity
    - ROI projections
    - Solver performance metrics
    - Scenario analysis results

    Example:
        tracker = ResourceOptimizerMLflowTracker()

        async with tracker.start_optimization_run("budget", resource_type="budget"):
            output = await agent.run(input_data)
            await tracker.log_optimization_result(output, final_state)

        # Query historical results
        history = await tracker.get_optimization_history(days=30)
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
    async def start_optimization_run(
        self,
        experiment_name: str = "default",
        resource_type: str = "budget",
        objective: str = "maximize_outcome",
        solver_type: str = "linear",
        brand: Optional[str] = None,
        region: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> AsyncIterator[OptimizationContext]:
        """
        Start an MLflow run for optimization tracking.

        Args:
            experiment_name: Name of the experiment
            resource_type: Type of resource being optimized
            objective: Optimization objective
            solver_type: Type of solver used
            brand: E2I brand context
            region: E2I region context
            tags: Additional MLflow tags

        Yields:
            OptimizationContext with run information
        """
        mlflow = self._get_mlflow()

        if mlflow is None:
            yield OptimizationContext(
                run_id="no-mlflow",
                experiment_name=experiment_name,
                resource_type=resource_type,
                objective=objective,
                brand=brand,
                region=region,
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
                        "agent": "resource_optimizer",
                        "tier": "4",
                    },
                )
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            logger.warning(f"Could not create/get experiment: {e}")
            yield OptimizationContext(
                run_id="experiment-error",
                experiment_name=experiment_name,
                resource_type=resource_type,
                objective=objective,
            )
            return

        try:
            with mlflow.start_run(experiment_id=experiment_id) as run:
                self._current_run_id = run.info.run_id

                # Log run parameters
                mlflow.log_params({
                    "agent": "resource_optimizer",
                    "tier": 4,
                    "resource_type": resource_type,
                    "objective": objective,
                    "solver_type": solver_type,
                })

                # Log context tags
                mlflow.set_tags({
                    "agent_type": "ml_predictions",
                    "framework_version": "4.3",
                })
                if brand:
                    mlflow.set_tag("brand", brand)
                if region:
                    mlflow.set_tag("region", region)

                # Custom tags
                for key, value in (tags or {}).items():
                    mlflow.set_tag(key, value)

                ctx = OptimizationContext(
                    run_id=run.info.run_id,
                    experiment_name=experiment_name,
                    resource_type=resource_type,
                    objective=objective,
                    brand=brand,
                    region=region,
                )

                yield ctx

                self._current_run_id = None

        except Exception as e:
            logger.error(f"MLflow run failed: {e}")
            self._current_run_id = None
            raise

    async def log_optimization_result(
        self,
        state: "ResourceOptimizerState",
    ) -> None:
        """
        Log optimization results to MLflow.

        Args:
            state: Final ResourceOptimizerState from agent execution
        """
        mlflow = self._get_mlflow()
        if mlflow is None or self._current_run_id is None:
            return

        try:
            # Extract metrics from state
            metrics = self._extract_metrics(state)
            mlflow.log_metrics(metrics.to_dict())

            # Log solver status tag
            mlflow.set_tags({
                "solver_status": metrics.solver_status,
                "has_violations": str(metrics.constraint_violations > 0).lower(),
                "positive_roi": str((metrics.projected_roi or 0) > 1.0).lower(),
            })

            # Log artifacts
            if self.enable_artifact_logging:
                await self._log_artifacts(state)

            logger.debug(
                f"Logged optimization metrics to MLflow run {self._current_run_id}: "
                f"objective={metrics.objective_value}, roi={metrics.projected_roi}"
            )

        except Exception as e:
            logger.warning(f"Failed to log optimization metrics to MLflow: {e}")

    def _extract_metrics(
        self,
        state: "ResourceOptimizerState",
    ) -> ResourceOptimizerMetrics:
        """Extract metrics from state."""
        metrics = ResourceOptimizerMetrics()

        # Core optimization results
        metrics.objective_value = state.get("objective_value")
        metrics.projected_outcome = state.get("projected_total_outcome")
        metrics.projected_roi = state.get("projected_roi")
        metrics.solver_status = state.get("solver_status", "unknown")
        metrics.solve_time_ms = state.get("solve_time_ms", 0)

        # Allocation changes
        allocations = state.get("optimal_allocations", [])
        if allocations:
            metrics.entities_optimized = len(allocations)
            metrics.entities_increased = sum(
                1 for a in allocations if a.get("change", 0) > 0
            )
            metrics.entities_decreased = sum(
                1 for a in allocations if a.get("change", 0) < 0
            )
            metrics.entities_unchanged = sum(
                1 for a in allocations if a.get("change", 0) == 0
            )
            metrics.total_allocation_change = sum(
                abs(a.get("change", 0)) for a in allocations
            )

            change_percentages = [
                abs(a.get("change_percentage", 0)) for a in allocations
            ]
            if change_percentages:
                metrics.avg_change_percentage = sum(change_percentages) / len(change_percentages)

        # Scenario analysis
        scenarios = state.get("scenarios", [])
        if scenarios:
            metrics.scenarios_analyzed = len(scenarios)
            rois = [s.get("roi", 0) for s in scenarios if s.get("roi") is not None]
            if rois:
                metrics.best_scenario_roi = max(rois)

        # Latency metrics
        metrics.formulation_latency_ms = state.get("formulation_latency_ms", 0)
        metrics.optimization_latency_ms = state.get("optimization_latency_ms", 0)
        metrics.total_latency_ms = state.get("total_latency_ms", 0)

        return metrics

    async def _log_artifacts(
        self,
        state: "ResourceOptimizerState",
    ) -> None:
        """Log artifacts to MLflow."""
        mlflow = self._get_mlflow()
        if mlflow is None:
            return

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Log optimal allocations
                allocations = state.get("optimal_allocations", [])
                if allocations:
                    alloc_path = os.path.join(tmpdir, "optimal_allocations.json")
                    with open(alloc_path, "w") as f:
                        json.dump(allocations, f, indent=2, default=str)
                    mlflow.log_artifact(alloc_path, "allocations")

                # Log scenarios
                scenarios = state.get("scenarios", [])
                if scenarios:
                    scenarios_path = os.path.join(tmpdir, "scenarios.json")
                    with open(scenarios_path, "w") as f:
                        json.dump(scenarios, f, indent=2, default=str)
                    mlflow.log_artifact(scenarios_path, "scenarios")

                # Log sensitivity analysis
                sensitivity = state.get("sensitivity_analysis", {})
                if sensitivity:
                    sens_path = os.path.join(tmpdir, "sensitivity_analysis.json")
                    with open(sens_path, "w") as f:
                        json.dump(sensitivity, f, indent=2, default=str)
                    mlflow.log_artifact(sens_path, "analysis")

                # Log impact by segment
                impact = state.get("impact_by_segment", {})
                if impact:
                    impact_path = os.path.join(tmpdir, "impact_by_segment.json")
                    with open(impact_path, "w") as f:
                        json.dump(impact, f, indent=2, default=str)
                    mlflow.log_artifact(impact_path, "analysis")

                # Log recommendations
                recommendations = state.get("recommendations", [])
                if recommendations:
                    rec_path = os.path.join(tmpdir, "recommendations.json")
                    with open(rec_path, "w") as f:
                        json.dump(recommendations, f, indent=2, default=str)
                    mlflow.log_artifact(rec_path, "recommendations")

        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")

    async def get_optimization_history(
        self,
        experiment_name: str = "default",
        resource_type: Optional[str] = None,
        objective: Optional[str] = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query historical optimization runs.

        Args:
            experiment_name: Name of the experiment to query
            resource_type: Filter by resource type
            objective: Filter by objective
            max_results: Maximum number of results to return

        Returns:
            List of historical optimization results
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
            if resource_type:
                filters.append(f"params.resource_type = '{resource_type}'")
            if objective:
                filters.append(f"params.objective = '{objective}'")

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
                    "objective_value": row.get("metrics.objective_value"),
                    "projected_roi": row.get("metrics.projected_roi"),
                    "entities_optimized": row.get("metrics.entities_optimized"),
                    "solve_time_ms": row.get("metrics.solve_time_ms"),
                    "total_latency_ms": row.get("metrics.total_latency_ms"),
                    "resource_type": row.get("params.resource_type"),
                    "objective": row.get("params.objective"),
                    "solver_type": row.get("params.solver_type"),
                    "solver_status": row.get("tags.solver_status"),
                })

            return history

        except Exception as e:
            logger.warning(f"Failed to query optimization history: {e}")
            return []

    async def get_roi_trends(
        self,
        experiment_name: str = "default",
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Get ROI trends over time.

        Args:
            experiment_name: Experiment to analyze
            days: Number of days to look back

        Returns:
            Dictionary with ROI trend analysis
        """
        history = await self.get_optimization_history(experiment_name, max_results=1000)

        if not history:
            return {
                "total_optimizations": 0,
                "avg_roi": 0.0,
                "avg_objective_value": 0.0,
                "trend": "insufficient_data",
            }

        rois = [h["projected_roi"] for h in history if h.get("projected_roi")]
        objectives = [h["objective_value"] for h in history if h.get("objective_value")]

        # Calculate trend
        trend = "stable"
        if len(rois) >= 4:
            first_half = rois[:len(rois)//2]
            second_half = rois[len(rois)//2:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)

            if second_avg > first_avg * 1.1:
                trend = "improving"
            elif second_avg < first_avg * 0.9:
                trend = "declining"

        return {
            "total_optimizations": len(history),
            "avg_roi": sum(rois) / len(rois) if rois else 0.0,
            "max_roi": max(rois) if rois else 0.0,
            "min_roi": min(rois) if rois else 0.0,
            "avg_objective_value": sum(objectives) / len(objectives) if objectives else 0.0,
            "trend": trend,
            "optimizations_by_resource": self._count_by_field(history, "resource_type"),
            "optimizations_by_objective": self._count_by_field(history, "objective"),
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


def create_tracker(tracking_uri: Optional[str] = None) -> ResourceOptimizerMLflowTracker:
    """Factory function to create a Resource Optimizer MLflow tracker."""
    return ResourceOptimizerMLflowTracker(tracking_uri=tracking_uri)
