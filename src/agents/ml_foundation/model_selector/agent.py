"""ModelSelectorAgent - Tier 0 ML Foundation Agent.

This agent selects optimal ML algorithms based on problem scope and constraints.
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
    agent_type = "standard"
    sla_seconds = 120  # 2 minutes for algorithm selection

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

        logger.info(
            f"Starting model selection for experiment {experiment_id}, "
            f"problem_type={problem_type}, mode={self.mode}"
        )

        try:
            # Execute LangGraph workflow
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

            logger.info(
                f"Model selection complete: {output['model_candidate']['algorithm_name']} "
                f"(score: {output['model_candidate']['selection_score']:.3f})"
            )

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
            "training_time_estimate_hours": final_state.get(
                "training_time_estimate_hours", 0.0
            ),
            "estimated_inference_latency_ms": final_state.get(
                "estimated_inference_latency_ms", 0
            ),
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
