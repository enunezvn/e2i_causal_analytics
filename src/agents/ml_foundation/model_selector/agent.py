"""ModelSelectorAgent - Tier 0 ML Foundation Agent.

This agent selects optimal ML algorithms based on problem scope and constraints.

Outputs:
- ModelCandidate: Selected algorithm with configuration
- SelectionRationale: Explanation of selection decision

Integration:
- Upstream: data_preparer (requires QC gate passed)
- Downstream: model_trainer (consumes ModelCandidate)
- Database: ml_model_registry table
- Memory: Procedural memory (successful selection patterns)
- Observability: Opik tracing
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Literal

from .graph import (
    create_conditional_selector_graph,
    create_model_selector_graph,
    create_simple_selector_graph,
)
from .state import ModelSelectorState

logger = logging.getLogger(__name__)


def _get_model_registry_repository():
    """Get MLModelRegistryRepository (lazy import to avoid circular deps)."""
    try:
        from src.memory.services.factories import get_supabase_client
        from src.repositories.ml_experiment import MLModelRegistryRepository

        client = get_supabase_client()
        return MLModelRegistryRepository(supabase_client=client)
    except Exception as e:
        logger.warning(f"Could not get model registry repository: {e}")
        return None


def _get_opik_connector():
    """Get OpikConnector (lazy import to avoid circular deps)."""
    try:
        from src.mlops.opik_connector import get_opik_connector

        return get_opik_connector()
    except Exception as e:
        logger.warning(f"Could not get Opik connector: {e}")
        return None


def _get_procedural_memory():
    """Get procedural memory client (lazy import with graceful degradation)."""
    try:
        from src.memory.procedural_memory import get_procedural_memory_client

        return get_procedural_memory_client()
    except Exception as e:
        logger.debug(f"Procedural memory not available: {e}")
        return None


class ModelSelectorAgent:
    """Select optimal ML algorithm based on problem characteristics.

    Responsibilities:
    - Filter algorithms by problem type and constraints
    - Rank candidates by composite score (historical, speed, memory, interpretability)
    - Run cross-validation benchmarks (if sample data provided)
    - Compare against baseline models
    - Select primary candidate and alternatives
    - Generate selection rationale
    - Register selection in MLflow

    Inputs:
        - scope_spec: Complete ScopeSpec from scope_definer
        - qc_report: QC validation from data_preparer
        - baseline_metrics: Optional baseline performance metrics
        - algorithm_preferences: Optional user preferences (e.g., ["CausalForest"])
        - excluded_algorithms: Optional algorithms to exclude
        - interpretability_required: Whether model must be interpretable
        - X_sample: Optional sample data for benchmarking
        - y_sample: Optional sample labels for benchmarking

    Outputs:
        - ModelCandidate: Selected algorithm with configuration
        - SelectionRationale: Explanation of selection decision
        - benchmark_results: CV benchmark results (if data provided)
        - mlflow_run_id: MLflow run ID (if registered)
    """

    # Agent metadata
    tier = 0
    tier_name = "ml_foundation"
    agent_name = "model_selector"
    agent_type = "standard"
    sla_seconds = 120  # 2 minutes for algorithm selection
    tools = ["mlflow", "optuna"]  # MLflow for registration, Optuna for hyperparameter search

    def __init__(
        self,
        mode: Literal["full", "simple", "conditional"] = "conditional",
    ):
        """Initialize ModelSelectorAgent with LangGraph.

        Args:
            mode: Graph mode to use:
                - "full": All nodes always run
                - "simple": Basic filter → rank → select → rationale
                - "conditional": Run benchmarks/MLflow conditionally (default)
        """
        self.mode = mode
        self._graph = None  # Lazy load
        logger.info(f"ModelSelectorAgent initialized with mode={mode}")

    @property
    def graph(self):
        """Get the LangGraph workflow (lazy loaded)."""
        if self._graph is None:
            if self.mode == "simple":
                self._graph = create_simple_selector_graph()
            elif self.mode == "full":
                self._graph = create_model_selector_graph()
            else:  # conditional
                self._graph = create_conditional_selector_graph()
        return self._graph

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model selection workflow.

        Args:
            input_data: Dictionary with:
                - scope_spec: Required. Complete ScopeSpec
                - qc_report: Required. QC validation result
                - algorithm_preferences: Optional list of preferred algorithms
                - excluded_algorithms: Optional list to exclude
                - interpretability_required: Optional bool
                - X_sample: Optional sample features for benchmarking
                - y_sample: Optional sample labels for benchmarking
                - skip_benchmarks: Optional bool to skip benchmarks
                - skip_mlflow: Optional bool to skip MLflow registration

        Returns:
            Dictionary with ModelCandidate, SelectionRationale, and metadata
        """
        # Validate required inputs
        required_fields = ["scope_spec", "qc_report"]
        for field in required_fields:
            if field not in input_data:
                return {
                    "error": f"Missing required field: {field}",
                    "error_type": "validation_error",
                }

        # Validate scope_spec passed QC
        scope_spec = input_data["scope_spec"]
        qc_report = input_data["qc_report"]

        if not qc_report.get("qc_passed", False):
            return {
                "error": (
                    "QC validation failed. Cannot proceed with model selection. "
                    f"QC errors: {qc_report.get('qc_errors', [])}"
                ),
                "error_type": "qc_validation_error",
            }

        # Extract problem details from scope_spec
        problem_type = scope_spec.get("problem_type", "binary_classification")
        technical_constraints = scope_spec.get("technical_constraints", [])
        experiment_id = scope_spec.get("experiment_id", "unknown")
        kpi_category = scope_spec.get("kpi_category")

        # Build initial state
        initial_state: ModelSelectorState = {
            # Required inputs
            "scope_spec": scope_spec,
            "experiment_id": experiment_id,
            "qc_report": qc_report,
            "baseline_metrics": input_data.get("baseline_metrics", {}),
            # User preferences
            "algorithm_preferences": input_data.get("algorithm_preferences"),
            "excluded_algorithms": input_data.get("excluded_algorithms"),
            "interpretability_required": input_data.get("interpretability_required", False),
            # Problem details
            "problem_type": problem_type,
            "technical_constraints": technical_constraints,
            "kpi_category": kpi_category,
            "row_count": qc_report.get("row_count", 1000),
            "column_count": qc_report.get("column_count", 10),
            # Sample data for benchmarking
            "X_sample": input_data.get("X_sample"),
            "y_sample": input_data.get("y_sample"),
            # Control flags
            "skip_benchmarks": input_data.get("skip_benchmarks", False),
            "skip_mlflow": input_data.get("skip_mlflow", False),
            # Metadata
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": "model_selector",
            "stage": "development",
            "registered_in_mlflow": False,
        }

        start_time = datetime.now(timezone.utc)
        logger.info(
            f"Starting model selection for experiment {experiment_id}, "
            f"problem_type={problem_type}, mode={self.mode}"
        )

        # Execute with optional Opik tracing
        opik = _get_opik_connector()
        try:
            if opik and opik.is_enabled:
                async with opik.trace_agent(
                    agent_name=self.agent_name,
                    operation="select_model",
                    metadata={
                        "tier": self.tier,
                        "experiment_id": experiment_id,
                        "problem_type": problem_type,
                        "mode": self.mode,
                    },
                    tags=[self.agent_name, "tier_0", "model_selection"],
                    input_data={"experiment_id": experiment_id, "problem_type": problem_type},
                ) as span:
                    final_state = await self.graph.ainvoke(initial_state)
                    # Set output on span
                    if span and not final_state.get("error"):
                        span.set_output(
                            {
                                "algorithm_name": final_state.get("algorithm_name"),
                                "selection_score": final_state.get("selection_score"),
                                "registered_in_mlflow": final_state.get("registered_in_mlflow"),
                            }
                        )
            else:
                final_state = await self.graph.ainvoke(initial_state)

            # Check for errors
            if final_state.get("error"):
                logger.error(
                    f"Model selection failed: {final_state['error']} "
                    f"(type: {final_state.get('error_type', 'unknown')})"
                )
                return {
                    "error": final_state["error"],
                    "error_type": final_state.get("error_type", "unknown"),
                }

            # Build output
            output = self._build_output(final_state, experiment_id)

            # Persist model candidate to database
            await self._persist_model_candidate(output)

            # Update procedural memory with successful selection pattern
            await self._update_procedural_memory(output)

            # Log execution time
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(
                f"Model selection complete: {output['model_candidate']['algorithm_name']} "
                f"(score: {output['model_candidate']['selection_score']:.3f}) in {duration:.2f}s"
            )

            # Check SLA
            if duration > self.sla_seconds:
                logger.warning(f"SLA violation: {duration:.2f}s > {self.sla_seconds}s")

            return output

        except Exception as e:
            logger.exception(f"Model selection failed with exception: {e}")
            return {
                "error": str(e),
                "error_type": "execution_error",
            }

    def _build_output(
        self,
        final_state: Dict[str, Any],
        experiment_id: str,
    ) -> Dict[str, Any]:
        """Build structured output from final state.

        Args:
            final_state: Final LangGraph state
            experiment_id: Experiment ID

        Returns:
            Structured output dictionary
        """
        # Extract ModelCandidate
        model_candidate = {
            "algorithm_name": final_state.get("algorithm_name", ""),
            "algorithm_class": final_state.get("algorithm_class", ""),
            "algorithm_family": final_state.get("algorithm_family", ""),
            "default_hyperparameters": final_state.get("default_hyperparameters", {}),
            "hyperparameter_search_space": final_state.get("hyperparameter_search_space", {}),
            "expected_performance": final_state.get("expected_performance", {}),
            "training_time_estimate_hours": final_state.get("training_time_estimate_hours", 0.0),
            "estimated_inference_latency_ms": final_state.get("estimated_inference_latency_ms", 0),
            "memory_requirement_gb": final_state.get("memory_requirement_gb", 0.0),
            "interpretability_score": final_state.get("interpretability_score", 0.5),
            "scalability_score": final_state.get("scalability_score", 0.7),
            "selection_score": final_state.get("selection_score", 0.5),
            "combined_score": final_state.get("combined_score"),
            "benchmark_score": final_state.get("benchmark_score"),
        }

        # Extract SelectionRationale
        selection_rationale = {
            "selection_rationale": final_state.get("selection_rationale", ""),
            "primary_reason": final_state.get("primary_reason", ""),
            "supporting_factors": final_state.get("supporting_factors", []),
            "alternatives_considered": final_state.get("alternatives_considered", []),
            "constraint_compliance": final_state.get("constraint_compliance", {}),
        }

        # Extract benchmark results
        benchmark_info = {
            "benchmark_results": final_state.get("benchmark_results", {}),
            "benchmarks_skipped": final_state.get("benchmarks_skipped", True),
            "benchmark_time_seconds": final_state.get("benchmark_time_seconds", 0.0),
        }

        # Extract baseline comparison
        baseline_comparison = final_state.get("baseline_comparison", {})

        # Extract MLflow registration
        mlflow_info = {
            "registered_in_mlflow": final_state.get("registered_in_mlflow", False),
            "mlflow_run_id": final_state.get("mlflow_run_id"),
            "mlflow_experiment_id": final_state.get("mlflow_experiment_id"),
        }

        # Extract selection summary (for database storage)
        selection_summary = final_state.get("selection_summary", {})

        return {
            # Primary outputs
            "model_candidate": model_candidate,
            "selection_rationale": selection_rationale,
            # Candidates
            "primary_candidate": final_state.get("primary_candidate", {}),
            "alternative_candidates": final_state.get("alternative_candidates", []),
            # Benchmarking
            **benchmark_info,
            "baseline_comparison": baseline_comparison,
            # Historical
            "historical_success_rates": final_state.get("historical_success_rates", {}),
            "similar_experiments": final_state.get("similar_experiments", []),
            # MLflow
            **mlflow_info,
            # Summary
            "selection_summary": selection_summary,
            # Metadata
            "experiment_id": experiment_id,
            "status": "completed",
        }

    async def _persist_model_candidate(self, output: Dict[str, Any]) -> None:
        """Persist ModelCandidate to ml_model_registry table.

        Graceful degradation: If repository is unavailable,
        logs a debug message and continues without error.

        Args:
            output: Agent output containing model_candidate and metadata
        """
        try:
            repo = _get_model_registry_repository()
            if repo is None:
                logger.debug("Skipping model persistence (no repository)")
                return

            model_candidate = output.get("model_candidate", {})
            experiment_id = output.get("experiment_id", "")
            mlflow_info = output.get("mlflow_info", {})

            # Register model candidate in ml_model_registry table with MLflow audit trail
            result = await repo.register_model_candidate(
                experiment_id=experiment_id,
                model_name=model_candidate.get("algorithm_name", "unknown"),
                model_type=model_candidate.get("algorithm_family", "unknown"),
                model_class=model_candidate.get("algorithm_class", ""),
                hyperparameters=model_candidate.get("default_hyperparameters", {}),
                hyperparameter_search_space=model_candidate.get("hyperparameter_search_space", {}),
                selection_score=model_candidate.get("selection_score", 0.0),
                selection_rationale=output.get("selection_rationale", {}).get(
                    "selection_rationale", ""
                ),
                stage="candidate",
                created_by="model_selector",
                mlflow_run_id=mlflow_info.get("mlflow_run_id"),
                mlflow_experiment_id=mlflow_info.get("mlflow_experiment_id"),
            )

            if result:
                logger.info(
                    f"Persisted model candidate: {model_candidate.get('algorithm_name')} for {experiment_id}"
                )
            else:
                logger.debug("Model candidate not persisted (no result returned)")

        except Exception as e:
            logger.warning(f"Failed to persist model candidate: {e}")

    async def _update_procedural_memory(self, output: Dict[str, Any]) -> None:
        """Update procedural memory with successful selection pattern.

        Graceful degradation: If memory is unavailable,
        logs a debug message and continues without error.

        Args:
            output: Agent output containing model_candidate and selection_rationale
        """
        try:
            memory = _get_procedural_memory()
            if memory is None:
                logger.debug("Procedural memory not available, skipping update")
                return

            model_candidate = output.get("model_candidate", {})
            selection_rationale = output.get("selection_rationale", {})

            # Store successful selection pattern for future reference
            await memory.store_pattern(
                agent_name=self.agent_name,
                pattern_type="model_selection",
                pattern_data={
                    "algorithm_name": model_candidate.get("algorithm_name"),
                    "algorithm_family": model_candidate.get("algorithm_family"),
                    "problem_type": output.get("selection_summary", {}).get("problem_type"),
                    "selection_score": model_candidate.get("selection_score"),
                    "primary_reason": selection_rationale.get("primary_reason"),
                    "supporting_factors": selection_rationale.get("supporting_factors", []),
                    "experiment_id": output.get("experiment_id"),
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            logger.info(f"Updated procedural memory for experiment: {output.get('experiment_id')}")

        except Exception as e:
            logger.debug(f"Failed to update procedural memory: {e}")
