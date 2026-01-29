"""
MLflow Integration for Causal Impact Agent.

Provides comprehensive MLflow tracking for causal impact analysis,
logging ATE/CATE estimates, refutation results, sensitivity analysis,
and causal DAG artifacts.

Integration Points:
    - MLflow experiment tracking (via MLflowConnector)
    - Opik tracing (via existing agent integration)
    - Dashboard metrics queries

Usage:
    tracker = CausalImpactMLflowTracker()
    async with tracker.start_analysis_run(
        experiment_name="trigger_effectiveness",
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.causal_impact.state import CausalImpactOutput, CausalImpactState

logger = logging.getLogger(__name__)


@dataclass
class AnalysisContext:
    """Context for an MLflow analysis run."""

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
class CausalImpactMetrics:
    """Metrics collected from causal impact analysis."""

    # Core estimates
    ate: Optional[float] = None
    ate_ci_lower: Optional[float] = None
    ate_ci_upper: Optional[float] = None
    standard_error: Optional[float] = None
    p_value: Optional[float] = None

    # CATE metrics (if available)
    cate_mean: Optional[float] = None
    cate_std: Optional[float] = None
    n_segments: int = 0

    # Refutation metrics
    refutation_passed: bool = False
    n_refutation_tests: int = 0
    tests_passed: int = 0
    confidence_adjustment: float = 1.0

    # Sensitivity analysis
    e_value: Optional[float] = None
    robust_to_confounding: bool = False

    # Energy score (if using estimator selector)
    energy_score: Optional[float] = None
    selection_strategy: Optional[str] = None

    # Performance
    computation_latency_ms: float = 0.0
    interpretation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Quality
    overall_confidence: float = 0.0
    statistical_significance: bool = False


class CausalImpactMLflowTracker:
    """
    Tracks Causal Impact Agent metrics in MLflow.

    Integrates with the MLflowConnector infrastructure to log:
    - ATE and CATE estimates
    - Refutation test results
    - E-value sensitivity analysis
    - Causal DAG artifacts
    - Performance metrics

    Example:
        tracker = CausalImpactMLflowTracker()

        async with tracker.start_analysis_run("customer_churn", brand="kisqali"):
            output = await agent.run(input_data)
            await tracker.log_analysis_result(output, final_state)

        # Query historical results
        history = await tracker.get_analysis_history(days=30)
    """

    EXPERIMENT_PREFIX = "e2i_causal/causal_impact"

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        enable_artifact_logging: bool = True,
    ):
        """
        Initialize the tracker.

        Args:
            tracking_uri: MLflow tracking server URI (default: from env)
            enable_artifact_logging: Whether to log artifacts (DAGs, JSON details)
        """
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.enable_artifact_logging = enable_artifact_logging

        self._current_context: Optional[AnalysisContext] = None
        self._mlflow_available = self._check_mlflow()
        self._mlflow_connector = None

    def _check_mlflow(self) -> bool:
        """Check if MLflow is available."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            return True
        except ImportError:
            logger.warning("MLflow not installed, metrics will be logged locally only")
            return False

    async def _get_connector(self):
        """Get or create MLflowConnector instance."""
        if self._mlflow_connector is None:
            try:
                from src.mlops.mlflow_connector import MLflowConnector

                self._mlflow_connector = MLflowConnector()
            except ImportError:
                logger.warning("MLflowConnector not available")
                return None
        return self._mlflow_connector

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
        Start an MLflow run for causal impact analysis.

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
            AnalysisContext for the run
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
                            "agent": "causal_impact",
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
                run_name or f"analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            )

            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name,
            ) as run:
                # Log standard tags
                mlflow.set_tag("agent_type", "causal_impact")
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

                self._current_context = AnalysisContext(
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
            # Fallback: create dummy context for non-MLflow environments
            from uuid import uuid4

            self._current_context = AnalysisContext(
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
        output: "CausalImpactOutput",
        state: Optional["CausalImpactState"] = None,
        additional_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log a complete analysis result to MLflow.

        Args:
            output: CausalImpactOutput from the agent
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
                f"Logged causal impact analysis to MLflow: "
                f"ATE={metrics.ate}, confidence={metrics.overall_confidence:.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to log analysis result to MLflow: {e}")

    def _extract_metrics(
        self,
        output: "CausalImpactOutput",
        state: Optional["CausalImpactState"] = None,
    ) -> CausalImpactMetrics:
        """Extract metrics from output and state."""
        metrics = CausalImpactMetrics()

        # Core estimates from output
        metrics.ate = output.get("ate_estimate")
        ci = output.get("confidence_interval")
        if ci and len(ci) == 2:
            metrics.ate_ci_lower = ci[0]
            metrics.ate_ci_upper = ci[1]
        metrics.standard_error = output.get("standard_error")
        metrics.p_value = output.get("p_value")
        metrics.statistical_significance = output.get("statistical_significance", False)
        metrics.overall_confidence = output.get("confidence", 0.0)
        metrics.refutation_passed = output.get("refutation_passed", False)

        # Latency metrics
        metrics.total_latency_ms = output.get("total_latency_ms", 0.0)
        metrics.computation_latency_ms = output.get("computation_latency_ms", 0.0)
        metrics.interpretation_latency_ms = output.get("interpretation_latency_ms", 0.0)

        # Extract detailed metrics from state if available
        if state:
            estimation_result = state.get("estimation_result", {})
            refutation_results = state.get("refutation_results", {})
            sensitivity_analysis = state.get("sensitivity_analysis", {})

            # CATE metrics
            cate_results = estimation_result.get("cate_results", {})
            if cate_results:
                metrics.cate_mean = cate_results.get("mean")
                metrics.cate_std = cate_results.get("std")
                metrics.n_segments = cate_results.get("n_segments", 0)

            # Energy score (if estimator selection was used)
            metrics.energy_score = estimation_result.get("energy_score")
            metrics.selection_strategy = estimation_result.get("selection_strategy")

            # Refutation details
            metrics.n_refutation_tests = refutation_results.get("n_tests", 0)
            metrics.tests_passed = refutation_results.get("tests_passed", 0)
            metrics.confidence_adjustment = refutation_results.get(
                "confidence_adjustment", 1.0
            )

            # Sensitivity analysis
            metrics.e_value = sensitivity_analysis.get("e_value")
            metrics.robust_to_confounding = sensitivity_analysis.get(
                "robust_to_confounding", False
            )

        return metrics

    def _log_metrics(self, metrics: CausalImpactMetrics) -> None:
        """Log metrics to MLflow."""
        import mlflow

        # Core estimates
        if metrics.ate is not None:
            mlflow.log_metric("ate", metrics.ate)
        if metrics.ate_ci_lower is not None:
            mlflow.log_metric("ate_ci_lower", metrics.ate_ci_lower)
        if metrics.ate_ci_upper is not None:
            mlflow.log_metric("ate_ci_upper", metrics.ate_ci_upper)
        if metrics.standard_error is not None:
            mlflow.log_metric("standard_error", metrics.standard_error)
        if metrics.p_value is not None:
            mlflow.log_metric("p_value", metrics.p_value)

        # CATE metrics
        if metrics.cate_mean is not None:
            mlflow.log_metric("cate_mean", metrics.cate_mean)
        if metrics.cate_std is not None:
            mlflow.log_metric("cate_std", metrics.cate_std)
        if metrics.n_segments > 0:
            mlflow.log_metric("n_segments", metrics.n_segments)

        # Refutation metrics
        mlflow.log_metric("refutation_passed", int(metrics.refutation_passed))
        mlflow.log_metric("n_refutation_tests", metrics.n_refutation_tests)
        mlflow.log_metric("tests_passed", metrics.tests_passed)
        mlflow.log_metric("confidence_adjustment", metrics.confidence_adjustment)

        # Sensitivity analysis
        if metrics.e_value is not None:
            mlflow.log_metric("e_value", metrics.e_value)
        mlflow.log_metric("robust_to_confounding", int(metrics.robust_to_confounding))

        # Energy score (if available)
        if metrics.energy_score is not None:
            mlflow.log_metric("energy_score", metrics.energy_score)

        # Performance metrics
        mlflow.log_metric("total_latency_ms", metrics.total_latency_ms)
        mlflow.log_metric("computation_latency_ms", metrics.computation_latency_ms)
        mlflow.log_metric("interpretation_latency_ms", metrics.interpretation_latency_ms)

        # Quality metrics
        mlflow.log_metric("overall_confidence", metrics.overall_confidence)
        mlflow.log_metric("statistical_significance", int(metrics.statistical_significance))

    def _log_params(
        self,
        output: "CausalImpactOutput",
        state: Optional["CausalImpactState"] = None,
        additional_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log parameters to MLflow."""
        import mlflow

        # Core parameters
        mlflow.log_param("estimation_method", output.get("estimation_method", "unknown"))
        mlflow.log_param("effect_type", output.get("effect_type", "ate"))
        mlflow.log_param("model_used", output.get("model_used", "unknown"))

        # State-derived parameters
        if state:
            mlflow.log_param("treatment_var", state.get("treatment_var", "unknown"))
            mlflow.log_param("outcome_var", state.get("outcome_var", "unknown"))
            mlflow.log_param("n_confounders", len(state.get("confounders", [])))
            mlflow.log_param("n_mediators", len(state.get("mediators", [])))
            mlflow.log_param("interpretation_depth", state.get("interpretation_depth", "standard"))

            if state.get("brand"):
                mlflow.log_param("brand", state["brand"])

        # Additional parameters
        for key, value in (additional_params or {}).items():
            mlflow.log_param(key, value)

    async def _log_artifacts(
        self,
        output: "CausalImpactOutput",
        state: Optional["CausalImpactState"] = None,
    ) -> None:
        """Log artifacts to MLflow."""
        import mlflow

        # Log causal DAG if available
        if state:
            causal_graph = state.get("causal_graph", {})
            if causal_graph:
                await self._log_json_artifact(
                    causal_graph, "causal_dag.json", "causal_graph"
                )

            # Log sensitivity analysis details
            sensitivity = state.get("sensitivity_analysis", {})
            if sensitivity:
                await self._log_json_artifact(
                    sensitivity, "sensitivity_analysis.json", "sensitivity"
                )

            # Log refutation details
            refutation = state.get("refutation_results", {})
            if refutation:
                await self._log_json_artifact(
                    refutation, "refutation_results.json", "refutation"
                )

        # Log full output as JSON
        output_dict = dict(output) if hasattr(output, "items") else output
        await self._log_json_artifact(
            output_dict, "analysis_output.json", "output"
        )

    async def _log_json_artifact(
        self,
        data: dict[str, Any],
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

    async def get_analysis_history(
        self,
        experiment_name: Optional[str] = None,
        brand: Optional[str] = None,
        days: int = 30,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get historical analysis results.

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
            from datetime import timedelta

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
                # Search all causal impact experiments
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
                    "ate": row.get("metrics.ate"),
                    "confidence": row.get("metrics.overall_confidence"),
                    "refutation_passed": bool(row.get("metrics.refutation_passed")),
                    "e_value": row.get("metrics.e_value"),
                    "latency_ms": row.get("metrics.total_latency_ms"),
                    "brand": row.get("tags.brand"),
                    "treatment_var": row.get("tags.treatment_var"),
                    "outcome_var": row.get("tags.outcome_var"),
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Failed to get analysis history: {e}")
            return []

    async def get_performance_summary(
        self,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Get performance summary for the Causal Impact Agent.

        Returns:
            Summary including success rate, avg latency, refutation pass rate
        """
        history = await self.get_analysis_history(days=days, limit=1000)

        if not history:
            return {
                "total_analyses": 0,
                "avg_latency_ms": 0.0,
                "refutation_pass_rate": 0.0,
                "avg_confidence": 0.0,
            }

        total = len(history)
        latencies = [h["latency_ms"] for h in history if h.get("latency_ms")]
        refutation_passed = sum(1 for h in history if h.get("refutation_passed"))
        confidences = [h["confidence"] for h in history if h.get("confidence")]

        return {
            "total_analyses": total,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "refutation_pass_rate": refutation_passed / total if total > 0 else 0.0,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
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
def create_tracker(**kwargs) -> CausalImpactMLflowTracker:
    """Create a tracker with environment-based configuration."""
    return CausalImpactMLflowTracker(**kwargs)
