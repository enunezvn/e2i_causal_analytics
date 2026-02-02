"""MLflow Tracker for Experiment Designer Agent.

Provides experiment tracking, metric logging, and artifact storage for
experiment design analyses. Tracks power analysis, validity audits,
and design iterations.

Tier: 3 (Monitoring & Design)
Pattern: Based on EnergyScoreMLflowTracker
"""

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# MLflow experiment prefix for this agent
EXPERIMENT_PREFIX = "e2i_causal/experiment_designer"


@dataclass
class DesignContext:
    """Context for experiment design MLflow tracking.

    Captures metadata about the design context for logging.
    """

    experiment_name: str = "default"
    brand: Optional[str] = None
    business_question: Optional[str] = None
    design_type: Optional[str] = None
    query_id: Optional[str] = None
    run_id: Optional[str] = None
    start_time: Optional[datetime] = None


@dataclass
class ExperimentDesignerMetrics:
    """Metrics collected during experiment design analysis.

    These metrics are logged to MLflow for tracking and comparison.
    """

    # Power analysis metrics
    required_sample_size: int = 0
    required_sample_size_per_arm: int = 0
    achieved_power: float = 0.0
    minimum_detectable_effect: float = 0.0
    alpha: float = 0.05
    duration_estimate_days: int = 0

    # Validity metrics
    validity_threats_count: int = 0
    critical_threats_count: int = 0
    high_threats_count: int = 0
    medium_threats_count: int = 0
    low_threats_count: int = 0
    overall_validity_score: float = 0.0
    mitigations_count: int = 0

    # Design metrics
    treatments_count: int = 0
    outcomes_count: int = 0
    primary_outcomes_count: int = 0
    stratification_vars_count: int = 0
    blocking_vars_count: int = 0

    # Iteration metrics
    redesign_iterations: int = 0

    # Latency metrics (milliseconds)
    context_latency_ms: int = 0
    reasoning_latency_ms: int = 0
    power_analysis_latency_ms: int = 0
    validity_audit_latency_ms: int = 0
    template_generation_latency_ms: int = 0
    total_latency_ms: int = 0

    # Token usage
    total_llm_tokens: int = 0

    # Warnings
    warnings: List[str] = field(default_factory=list)


class ExperimentDesignerMLflowTracker:
    """MLflow tracker for Experiment Designer Agent.

    Provides:
    - Experiment run management with async context manager
    - Metric logging for power analysis and validity audits
    - Artifact logging for design documents and analysis code
    - Historical query methods for dashboard integration

    Usage:
        tracker = ExperimentDesignerMLflowTracker()

        async with tracker.start_design_run(
            experiment_name="pharma_experiment_design",
            brand="Kisqali",
            business_question="Does call frequency impact TRx?",
        ) as ctx:
            # Run experiment design
            result = await agent.arun(input_data)

            # Log results
            await tracker.log_design_result(result, state)

        # Query historical designs
        history = tracker.get_design_history(brand="Kisqali", limit=10)
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize MLflow tracker.

        Args:
            tracking_uri: MLflow tracking server URI. If None, uses default.
        """
        self._mlflow = None
        self._tracking_uri = tracking_uri
        self._current_context: Optional[DesignContext] = None

    def _get_mlflow(self):
        """Lazy load MLflow to avoid import errors if not installed."""
        if self._mlflow is None:
            try:
                import mlflow

                self._mlflow = mlflow
                if self._tracking_uri:
                    mlflow.set_tracking_uri(self._tracking_uri)
            except (ImportError, OSError, PermissionError) as e:
                logger.warning(f"MLflow tracking unavailable ({type(e).__name__}): {e}")
                return None
        return self._mlflow

    def _get_or_create_experiment(self, experiment_name: str) -> str:
        """Get or create MLflow experiment.

        Args:
            experiment_name: Name suffix for experiment

        Returns:
            Experiment ID
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            return ""

        full_name = f"{EXPERIMENT_PREFIX}/{experiment_name}"

        experiment = mlflow.get_experiment_by_name(full_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                full_name,
                artifact_location="mlflow-artifacts:/",
                tags={
                    "agent": "experiment_designer",
                    "tier": "3",
                    "domain": "e2i_causal",
                },
            )
        else:
            experiment_id = experiment.experiment_id

        return experiment_id

    @asynccontextmanager
    async def start_design_run(
        self,
        experiment_name: str = "default",
        brand: Optional[str] = None,
        business_question: Optional[str] = None,
        design_type: Optional[str] = None,
        query_id: Optional[str] = None,
    ):
        """Start an MLflow run for experiment design analysis.

        Args:
            experiment_name: Name for the MLflow experiment
            brand: Brand being analyzed
            business_question: The business question being addressed
            design_type: Type of experimental design
            query_id: Unique identifier for this design query

        Yields:
            DesignContext with run information

        Example:
            async with tracker.start_design_run(
                experiment_name="pharma_design",
                brand="Kisqali",
                business_question="Does call frequency impact TRx?",
            ) as ctx:
                # Run design
                result = await agent.arun(input_data)
                await tracker.log_design_result(result, state)
        """
        mlflow = self._get_mlflow()
        context = DesignContext(
            experiment_name=experiment_name,
            brand=brand,
            business_question=business_question,
            design_type=design_type,
            query_id=query_id,
            start_time=datetime.utcnow(),
        )
        self._current_context = context

        if mlflow is None:
            yield context
            return

        experiment_id = self._get_or_create_experiment(experiment_name)

        # Start MLflow run
        with mlflow.start_run(experiment_id=experiment_id) as run:
            context.run_id = run.info.run_id

            # Log initial parameters
            mlflow.log_params(
                {
                    "brand": brand or "all",
                    "design_type": design_type or "unknown",
                    "query_id": query_id or "",
                }
            )

            # Log business question as tag (may be long)
            if business_question:
                mlflow.set_tag("business_question", business_question[:250])

            try:
                yield context
            except Exception as e:
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error", str(e)[:250])
                raise
            finally:
                self._current_context = None

    async def log_design_result(
        self,
        output: Any,
        state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log experiment design results to MLflow.

        Args:
            output: ExperimentDesignerOutput from agent
            state: Optional final state with additional metrics
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            return

        # Extract metrics from output
        metrics = self._extract_metrics(output, state)

        # Log numeric metrics
        metric_dict = {
            # Power analysis
            "required_sample_size": metrics.required_sample_size,
            "required_sample_size_per_arm": metrics.required_sample_size_per_arm,
            "achieved_power": metrics.achieved_power,
            "minimum_detectable_effect": metrics.minimum_detectable_effect,
            "alpha": metrics.alpha,
            "duration_estimate_days": metrics.duration_estimate_days,
            # Validity
            "validity_threats_count": metrics.validity_threats_count,
            "critical_threats_count": metrics.critical_threats_count,
            "high_threats_count": metrics.high_threats_count,
            "medium_threats_count": metrics.medium_threats_count,
            "low_threats_count": metrics.low_threats_count,
            "overall_validity_score": metrics.overall_validity_score,
            "mitigations_count": metrics.mitigations_count,
            # Design
            "treatments_count": metrics.treatments_count,
            "outcomes_count": metrics.outcomes_count,
            "primary_outcomes_count": metrics.primary_outcomes_count,
            "stratification_vars_count": metrics.stratification_vars_count,
            "blocking_vars_count": metrics.blocking_vars_count,
            # Iterations
            "redesign_iterations": metrics.redesign_iterations,
            # Latency
            "context_latency_ms": metrics.context_latency_ms,
            "reasoning_latency_ms": metrics.reasoning_latency_ms,
            "power_analysis_latency_ms": metrics.power_analysis_latency_ms,
            "validity_audit_latency_ms": metrics.validity_audit_latency_ms,
            "template_generation_latency_ms": metrics.template_generation_latency_ms,
            "total_latency_ms": metrics.total_latency_ms,
            # Tokens
            "total_llm_tokens": metrics.total_llm_tokens,
        }

        mlflow.log_metrics(metric_dict)

        # Log design type as tag
        if hasattr(output, "design_type"):
            mlflow.set_tag("design_type", output.design_type)
        elif isinstance(output, dict) and "design_type" in output:
            mlflow.set_tag("design_type", output["design_type"])

        # Log validity confidence as tag
        if hasattr(output, "validity_confidence"):
            mlflow.set_tag("validity_confidence", output.validity_confidence)
        elif isinstance(output, dict) and "validity_confidence" in output:
            mlflow.set_tag("validity_confidence", output["validity_confidence"])

        # Log warnings count
        if metrics.warnings:
            mlflow.set_tag("warnings_count", str(len(metrics.warnings)))

        # Log status
        mlflow.set_tag("status", "completed")

        # Log artifacts
        await self._log_artifacts(output, state, metrics)

    def _extract_metrics(
        self,
        output: Any,
        state: Optional[Dict[str, Any]] = None,
    ) -> ExperimentDesignerMetrics:
        """Extract metrics from output and state.

        Args:
            output: ExperimentDesignerOutput (Pydantic model or dict)
            state: Optional final state

        Returns:
            ExperimentDesignerMetrics dataclass
        """
        metrics = ExperimentDesignerMetrics()

        # Handle both Pydantic models and dicts
        if hasattr(output, "model_dump"):
            output_dict = output.model_dump()
        elif hasattr(output, "dict"):
            output_dict = output.dict()
        elif isinstance(output, dict):
            output_dict = output
        else:
            return metrics

        # Power analysis metrics
        power_analysis = output_dict.get("power_analysis")
        if power_analysis:
            if isinstance(power_analysis, dict):
                metrics.required_sample_size = power_analysis.get("required_sample_size", 0)
                metrics.required_sample_size_per_arm = power_analysis.get(
                    "required_sample_size_per_arm", 0
                )
                metrics.achieved_power = power_analysis.get("achieved_power", 0.0)
                metrics.minimum_detectable_effect = power_analysis.get(
                    "minimum_detectable_effect", 0.0
                )
                metrics.alpha = power_analysis.get("alpha", 0.05)
            else:
                # Pydantic model
                metrics.required_sample_size = getattr(power_analysis, "required_sample_size", 0)
                metrics.required_sample_size_per_arm = getattr(
                    power_analysis, "required_sample_size_per_arm", 0
                )
                metrics.achieved_power = getattr(power_analysis, "achieved_power", 0.0)
                metrics.minimum_detectable_effect = getattr(
                    power_analysis, "minimum_detectable_effect", 0.0
                )
                metrics.alpha = getattr(power_analysis, "alpha", 0.05)

        metrics.duration_estimate_days = output_dict.get("duration_estimate_days", 0)

        # Validity metrics
        validity_threats = output_dict.get("validity_threats", [])
        metrics.validity_threats_count = len(validity_threats)

        # Count by severity
        for threat in validity_threats:
            if isinstance(threat, dict):
                severity = threat.get("severity", "medium")
            else:
                severity = getattr(threat, "severity", "medium")

            if severity == "critical":
                metrics.critical_threats_count += 1
            elif severity == "high":
                metrics.high_threats_count += 1
            elif severity == "medium":
                metrics.medium_threats_count += 1
            else:
                metrics.low_threats_count += 1

        metrics.overall_validity_score = output_dict.get("overall_validity_score", 0.0)

        # Design metrics
        treatments = output_dict.get("treatments", [])
        outcomes = output_dict.get("outcomes", [])
        metrics.treatments_count = len(treatments)
        metrics.outcomes_count = len(outcomes)

        # Count primary outcomes
        for outcome in outcomes:
            if isinstance(outcome, dict):
                is_primary = outcome.get("is_primary", False)
            else:
                is_primary = getattr(outcome, "is_primary", False)
            if is_primary:
                metrics.primary_outcomes_count += 1

        metrics.stratification_vars_count = len(output_dict.get("stratification_variables", []))
        metrics.blocking_vars_count = len(output_dict.get("blocking_variables", []))

        # Iteration metrics
        metrics.redesign_iterations = output_dict.get("redesign_iterations", 0)

        # Latency metrics
        metrics.total_latency_ms = output_dict.get("total_latency_ms", 0)

        # State-based metrics
        if state:
            node_latencies = state.get("node_latencies_ms", {})
            metrics.context_latency_ms = node_latencies.get("context_loader", 0)
            metrics.reasoning_latency_ms = node_latencies.get("design_reasoning", 0)
            metrics.power_analysis_latency_ms = node_latencies.get("power_analysis", 0)
            metrics.validity_audit_latency_ms = node_latencies.get("validity_audit", 0)
            metrics.template_generation_latency_ms = node_latencies.get("template_generator", 0)
            metrics.total_llm_tokens = state.get("total_llm_tokens_used", 0)

            mitigations = state.get("mitigations", [])
            metrics.mitigations_count = len(mitigations)

        # Warnings
        metrics.warnings = output_dict.get("warnings", [])

        return metrics

    async def _log_artifacts(
        self,
        output: Any,
        state: Optional[Dict[str, Any]],
        metrics: ExperimentDesignerMetrics,
    ) -> None:
        """Log artifacts to MLflow.

        Args:
            output: ExperimentDesignerOutput
            state: Optional final state
            metrics: Extracted metrics
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            return

        import os
        import tempfile

        # Handle both Pydantic models and dicts
        if hasattr(output, "model_dump"):
            output_dict = output.model_dump()
        elif hasattr(output, "dict"):
            output_dict = output.dict()
        elif isinstance(output, dict):
            output_dict = output
        else:
            output_dict = {}

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Log design summary
                summary = {
                    "design_type": output_dict.get("design_type", "unknown"),
                    "design_rationale": output_dict.get("design_rationale", ""),
                    "randomization_unit": output_dict.get("randomization_unit", ""),
                    "randomization_method": output_dict.get("randomization_method", ""),
                    "power_analysis": output_dict.get("power_analysis"),
                    "validity_score": output_dict.get("overall_validity_score", 0.0),
                    "validity_confidence": output_dict.get("validity_confidence", "low"),
                    "redesign_iterations": output_dict.get("redesign_iterations", 0),
                    "context": {
                        "brand": self._current_context.brand if self._current_context else None,
                        "business_question": (
                            self._current_context.business_question
                            if self._current_context
                            else None
                        ),
                    },
                }
                summary_path = os.path.join(tmpdir, "design_summary.json")
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2, default=str)
                mlflow.log_artifact(summary_path, "design")

                # Log validity threats
                threats = output_dict.get("validity_threats", [])
                if threats:
                    # Convert to serializable format
                    threats_data = []
                    for t in threats:
                        if hasattr(t, "model_dump"):
                            threats_data.append(t.model_dump())
                        elif hasattr(t, "dict"):
                            threats_data.append(t.dict())
                        elif isinstance(t, dict):
                            threats_data.append(t)

                    threats_path = os.path.join(tmpdir, "validity_threats.json")
                    with open(threats_path, "w") as f:
                        json.dump(threats_data, f, indent=2, default=str)
                    mlflow.log_artifact(threats_path, "validity")

                # Log causal graph DOT
                causal_graph = output_dict.get("causal_graph_dot", "")
                if causal_graph:
                    dot_path = os.path.join(tmpdir, "causal_graph.dot")
                    with open(dot_path, "w") as f:
                        f.write(causal_graph)
                    mlflow.log_artifact(dot_path, "design")

                # Log analysis code template
                analysis_code = output_dict.get("analysis_code", "")
                if analysis_code:
                    code_path = os.path.join(tmpdir, "analysis_template.py")
                    with open(code_path, "w") as f:
                        f.write(analysis_code)
                    mlflow.log_artifact(code_path, "code")

                # Log pre-registration document
                prereg = output_dict.get("preregistration_document", "")
                if prereg:
                    prereg_path = os.path.join(tmpdir, "preregistration.md")
                    with open(prereg_path, "w") as f:
                        f.write(prereg)
                    mlflow.log_artifact(prereg_path, "documents")

                # Log treatments and outcomes
                treatments = output_dict.get("treatments", [])
                outcomes = output_dict.get("outcomes", [])
                if treatments or outcomes:
                    spec = {
                        "treatments": [
                            t.model_dump()
                            if hasattr(t, "model_dump")
                            else (t.dict() if hasattr(t, "dict") else t)
                            for t in treatments
                        ],
                        "outcomes": [
                            o.model_dump()
                            if hasattr(o, "model_dump")
                            else (o.dict() if hasattr(o, "dict") else o)
                            for o in outcomes
                        ],
                    }
                    spec_path = os.path.join(tmpdir, "experiment_spec.json")
                    with open(spec_path, "w") as f:
                        json.dump(spec, f, indent=2, default=str)
                    mlflow.log_artifact(spec_path, "design")

                # Log iteration history if available
                if state:
                    iteration_history = state.get("iteration_history", [])
                    if iteration_history:
                        history_path = os.path.join(tmpdir, "iteration_history.json")
                        with open(history_path, "w") as f:
                            json.dump(iteration_history, f, indent=2, default=str)
                        mlflow.log_artifact(history_path, "iterations")
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to log MLflow artifacts: {e}")

    def get_design_history(
        self,
        experiment_name: str = "default",
        brand: Optional[str] = None,
        design_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query historical experiment designs.

        Args:
            experiment_name: Name of the MLflow experiment
            brand: Filter by brand
            design_type: Filter by design type
            limit: Maximum number of runs to return

        Returns:
            List of run data dictionaries
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            return []

        full_name = f"{EXPERIMENT_PREFIX}/{experiment_name}"
        experiment = mlflow.get_experiment_by_name(full_name)
        if experiment is None:
            return []

        # Build filter string
        filter_parts = []
        if brand:
            filter_parts.append(f"params.brand = '{brand}'")
        if design_type:
            filter_parts.append(f"tags.design_type = '{design_type}'")

        filter_string = " AND ".join(filter_parts) if filter_parts else ""

        # Search runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            max_results=limit,
            order_by=["start_time DESC"],
        )

        if runs.empty:
            return []

        # Convert to list of dicts
        result = []
        for _, row in runs.iterrows():
            run_data = {
                "run_id": row.get("run_id"),
                "start_time": row.get("start_time"),
                "brand": row.get("params.brand"),
                "design_type": row.get("tags.design_type"),
                "validity_confidence": row.get("tags.validity_confidence"),
                "status": row.get("tags.status"),
                # Key metrics
                "required_sample_size": row.get("metrics.required_sample_size"),
                "achieved_power": row.get("metrics.achieved_power"),
                "overall_validity_score": row.get("metrics.overall_validity_score"),
                "validity_threats_count": row.get("metrics.validity_threats_count"),
                "critical_threats_count": row.get("metrics.critical_threats_count"),
                "redesign_iterations": row.get("metrics.redesign_iterations"),
                "total_latency_ms": row.get("metrics.total_latency_ms"),
            }
            result.append(run_data)

        return result

    def get_design_metrics_summary(
        self,
        experiment_name: str = "default",
        brand: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get summary statistics for experiment designs.

        Args:
            experiment_name: Name of the MLflow experiment
            brand: Filter by brand
            days: Number of days to look back

        Returns:
            Summary statistics dictionary
        """
        mlflow = self._get_mlflow()
        if mlflow is None:
            return {}

        full_name = f"{EXPERIMENT_PREFIX}/{experiment_name}"
        experiment = mlflow.get_experiment_by_name(full_name)
        if experiment is None:
            return {}

        # Build filter
        from datetime import timedelta

        start_time = datetime.utcnow() - timedelta(days=days)
        filter_parts = [f"attributes.start_time >= {int(start_time.timestamp() * 1000)}"]
        if brand:
            filter_parts.append(f"params.brand = '{brand}'")

        filter_string = " AND ".join(filter_parts)

        # Search runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            max_results=1000,
        )

        if runs.empty:
            return {
                "total_designs": 0,
                "period_days": days,
                "brand": brand,
            }

        # Calculate summary stats
        return {
            "total_designs": len(runs),
            "period_days": days,
            "brand": brand,
            # Power analysis stats
            "avg_sample_size": runs["metrics.required_sample_size"].mean(),
            "avg_achieved_power": runs["metrics.achieved_power"].mean(),
            # Validity stats
            "avg_validity_score": runs["metrics.overall_validity_score"].mean(),
            "avg_threats_count": runs["metrics.validity_threats_count"].mean(),
            "avg_critical_threats": runs["metrics.critical_threats_count"].mean(),
            # Design type distribution
            "design_types": runs["tags.design_type"].value_counts().to_dict()
            if "tags.design_type" in runs.columns
            else {},
            # Iteration stats
            "avg_redesign_iterations": runs["metrics.redesign_iterations"].mean(),
            # Performance stats
            "avg_latency_ms": runs["metrics.total_latency_ms"].mean(),
            "p95_latency_ms": runs["metrics.total_latency_ms"].quantile(0.95),
        }


def create_tracker(tracking_uri: Optional[str] = None) -> ExperimentDesignerMLflowTracker:
    """Factory function to create MLflow tracker.

    Args:
        tracking_uri: Optional MLflow tracking URI

    Returns:
        ExperimentDesignerMLflowTracker instance
    """
    return ExperimentDesignerMLflowTracker(tracking_uri=tracking_uri)
