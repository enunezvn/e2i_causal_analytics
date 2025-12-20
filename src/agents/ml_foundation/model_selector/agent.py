"""ModelSelectorAgent - Tier 0 ML Foundation Agent.

This agent selects optimal ML algorithms based on problem scope and constraints.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from .graph import create_model_selector_graph
from .state import ModelSelectorState

logger = logging.getLogger(__name__)


class ModelSelectorAgent:
    """Select optimal ML algorithm based on problem characteristics.

    Responsibilities:
    - Filter algorithms by problem type and constraints
    - Rank candidates by composite score (historical, speed, memory, interpretability)
    - Select primary candidate and alternatives
    - Generate selection rationale
    - (TODO) Compare to baseline models
    - (TODO) Register in MLflow model registry

    Inputs:
        - scope_spec: Complete ScopeSpec from scope_definer
        - qc_report: QC validation from data_preparer
        - baseline_metrics: Optional baseline performance metrics
        - algorithm_preferences: Optional user preferences (e.g., ["CausalForest"])
        - excluded_algorithms: Optional algorithms to exclude
        - interpretability_required: Whether model must be interpretable

    Outputs:
        - ModelCandidate: Selected algorithm with configuration
        - SelectionRationale: Explanation of selection decision
    """

    # Agent metadata
    tier = 0
    tier_name = "ml_foundation"
    agent_type = "standard"
    sla_seconds = 120  # 2 minutes for algorithm selection

    def __init__(self):
        """Initialize ModelSelectorAgent with LangGraph."""
        self.graph = create_model_selector_graph()
        logger.info("ModelSelectorAgent initialized")

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model selection workflow.

        Args:
            input_data: Dictionary with scope_spec, qc_report, and optional preferences

        Returns:
            Dictionary with ModelCandidate and SelectionRationale, or error dict
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

        # Build initial state
        initial_state: ModelSelectorState = {
            "scope_spec": scope_spec,
            "experiment_id": experiment_id,
            "qc_report": qc_report,
            "baseline_metrics": input_data.get("baseline_metrics", {}),
            "algorithm_preferences": input_data.get("algorithm_preferences"),
            "excluded_algorithms": input_data.get("excluded_algorithms"),
            "interpretability_required": input_data.get(
                "interpretability_required", False
            ),
            "problem_type": problem_type,
            "technical_constraints": technical_constraints,
            "row_count": qc_report.get("row_count", 1000),
            "column_count": qc_report.get("column_count", 10),
            # TODO: Fetch from Supabase procedural memory
            "historical_success_rates": {},
            "similar_experiments": [],
            "created_at": datetime.now(tz=None).isoformat(),
            "created_by": "model_selector",
            "stage": "development",
            "registered_in_mlflow": False,
        }

        logger.info(
            f"Starting model selection for experiment {experiment_id}, "
            f"problem_type={problem_type}"
        )

        try:
            # Execute LangGraph workflow
            final_state = await self.graph.ainvoke(initial_state)

            # Check for errors
            if "error" in final_state:
                logger.error(
                    f"Model selection failed: {final_state['error']} "
                    f"(type: {final_state.get('error_type', 'unknown')})"
                )
                return {
                    "error": final_state["error"],
                    "error_type": final_state.get("error_type", "unknown"),
                }

            # Extract ModelCandidate
            model_candidate = {
                "algorithm_name": final_state.get("algorithm_name", ""),
                "algorithm_class": final_state.get("algorithm_class", ""),
                "algorithm_family": final_state.get("algorithm_family", ""),
                "default_hyperparameters": final_state.get(
                    "default_hyperparameters", {}
                ),
                "hyperparameter_search_space": final_state.get(
                    "hyperparameter_search_space", {}
                ),
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
            }

            # Extract SelectionRationale
            selection_rationale = {
                "selection_rationale": final_state.get("selection_rationale", ""),
                "primary_reason": final_state.get("primary_reason", ""),
                "supporting_factors": final_state.get("supporting_factors", []),
                "alternatives_considered": final_state.get(
                    "alternatives_considered", []
                ),
                "constraint_compliance": final_state.get("constraint_compliance", {}),
            }

            logger.info(
                f"Model selection complete: {model_candidate['algorithm_name']} "
                f"(score: {model_candidate['selection_score']:.3f})"
            )

            # TODO: Persist to Supabase `model_candidates` table
            # TODO: Register in MLflow model registry
            # TODO: Store selection rationale in procedural memory

            return {
                "model_candidate": model_candidate,
                "selection_rationale": selection_rationale,
                "experiment_id": experiment_id,
                "primary_candidate": final_state.get("primary_candidate", {}),
                "alternative_candidates": final_state.get("alternative_candidates", []),
            }

        except Exception as e:
            logger.exception(f"Model selection failed with exception: {e}")
            return {
                "error": str(e),
                "error_type": "execution_error",
            }
