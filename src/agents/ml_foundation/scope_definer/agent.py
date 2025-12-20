"""Scope Definer Agent - Tier 0: ML Foundation.

Transforms business objectives into formal ML problem specifications with
measurable success criteria.
"""

from typing import Dict, Any, Optional
from datetime import datetime

from .graph import create_scope_definer_graph
from .state import ScopeDefinerState


class ScopeDefinerAgent:
    """Scope Definer: Transform business requirements into ML specifications.

    This agent is the entry point to the ML Foundation tier. It takes natural
    language business objectives and produces formal ScopeSpec and SuccessCriteria
    that drive all downstream ML agents.

    Responsibilities:
    - Classify ML problem type (classification, regression, causal, etc.)
    - Infer target variable and prediction horizon
    - Define population criteria (inclusion/exclusion)
    - Specify feature requirements and exclusions
    - Set regulatory, ethical, and technical constraints
    - Define measurable success criteria
    - Establish baseline expectations

    Tier: 0 (ML Foundation)
    Type: Standard (no LLM, pure computation)
    SLA: <5 seconds
    """

    tier = 0
    tier_name = "ml_foundation"
    agent_type = "standard"
    sla_seconds = 5

    def __init__(self):
        """Initialize the ScopeDefinerAgent."""
        self.graph = create_scope_definer_graph()

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scope definition workflow.

        Args:
            input_data: Dictionary with required fields:
                - problem_description (str): Natural language problem description
                - business_objective (str): Business objective this model serves
                - target_outcome (str): Target outcome (e.g., "Increase prescriptions")
                Optional fields:
                - problem_type_hint (str): Hint for problem type
                - target_variable (str): Target variable name if known
                - candidate_features (List[str]): Candidate features
                - time_budget_hours (float): Max training time
                - performance_requirements (Dict): Performance thresholds
                - brand (str): Brand context
                - region (str): Region context
                - use_case (str): Use case category

        Returns:
            Dictionary with:
                - scope_spec (Dict): Complete ScopeSpec
                - success_criteria (Dict): Complete SuccessCriteria
                - experiment_id (str): Unique experiment identifier
                - experiment_name (str): Human-readable name
                - validation_warnings (List[str]): Non-blocking warnings
                - error (Optional[str]): Error message if failed
                - error_type (Optional[str]): Error type if failed

        Raises:
            ValueError: If required inputs are missing
        """
        # Validate required inputs
        required_fields = ["problem_description", "business_objective", "target_outcome"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(
                    f"{field} is required. Please provide: {', '.join(required_fields)}"
                )

        # Construct initial state
        initial_state: ScopeDefinerState = {
            # Required inputs
            "problem_description": input_data["problem_description"],
            "business_objective": input_data["business_objective"],
            "target_outcome": input_data["target_outcome"],
            # Optional inputs
            "problem_type_hint": input_data.get("problem_type_hint"),
            "target_variable": input_data.get("target_variable"),
            "candidate_features": input_data.get("candidate_features"),
            "time_budget_hours": input_data.get("time_budget_hours"),
            "performance_requirements": input_data.get("performance_requirements", {}),
            "brand": input_data.get("brand", "unknown"),
            "region": input_data.get("region", "all"),
            "use_case": input_data.get("use_case", "commercial_targeting"),
        }

        # Execute LangGraph
        try:
            final_state = await self.graph.ainvoke(initial_state)
        except Exception as e:
            return {
                "error": f"Graph execution failed: {str(e)}",
                "error_type": "graph_execution_error",
            }

        # Check for errors
        if final_state.get("error"):
            return {
                "error": final_state["error"],
                "error_type": final_state.get("error_type", "unknown_error"),
            }

        # Construct output
        output = {
            # Primary outputs
            "scope_spec": final_state.get("scope_spec", {}),
            "success_criteria": final_state.get("success_criteria", {}),
            # Identifiers
            "experiment_id": final_state.get("experiment_id", ""),
            "experiment_name": final_state.get("experiment_name", ""),
            # Validation results
            "validation_passed": final_state.get("validation_passed", False),
            "validation_warnings": final_state.get("validation_warnings", []),
            "validation_errors": final_state.get("validation_errors", []),
            # Metadata
            "created_at": final_state.get("created_at", ""),
            "created_by": final_state.get("created_by", "scope_definer"),
        }

        # TODO: Write to database
        # await self._persist_scope_spec(output)

        # TODO: Update procedural memory with successful pattern
        # await self._update_procedural_memory(output)

        return output

    async def _persist_scope_spec(self, output: Dict[str, Any]) -> None:
        """Persist ScopeSpec to ml_experiments table.

        TODO: Implement database write operation.
        """
        # experiment_id = output["experiment_id"]
        # scope_spec = output["scope_spec"]
        # success_criteria = output["success_criteria"]
        #
        # await self.experiment_repo.create_experiment({
        #     "experiment_id": experiment_id,
        #     "scope_spec": scope_spec,
        #     "success_criteria": success_criteria,
        #     "status": "scoped",
        #     "created_at": output["created_at"],
        # })
        pass

    async def _update_procedural_memory(self, output: Dict[str, Any]) -> None:
        """Update procedural memory with successful scope pattern.

        TODO: Implement procedural memory update.
        """
        # problem_type = output["scope_spec"]["problem_type"]
        # brand = output["scope_spec"]["brand"]
        #
        # await self.procedural_memory.store_pattern({
        #     "agent": "scope_definer",
        #     "problem_type": problem_type,
        #     "brand": brand,
        #     "success_criteria": output["success_criteria"],
        #     "timestamp": output["created_at"],
        # })
        pass
