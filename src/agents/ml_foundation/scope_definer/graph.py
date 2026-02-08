"""LangGraph assembly for scope_definer agent.

This module defines the LangGraph workflow for transforming business
requirements into formal ML problem specifications.
"""

from datetime import datetime, timezone
from typing import Any, Dict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    build_scope_spec,
    classify_problem,
    define_success_criteria,
)
from .state import ScopeDefinerState


async def finalize_scope(state: Dict[str, Any]) -> Dict[str, Any]:
    """Finalize scope definition with timestamps and status.

    This node completes the scope definition process by adding metadata
    and performing final validation.

    Args:
        state: ScopeDefinerState with all scope and criteria defined

    Returns:
        Updated state with created_at, created_by, and error handling
    """
    try:
        # Add timestamps
        created_at = datetime.now(tz=None).isoformat()
        created_by = "scope_definer"

        # Check validation status
        validation_passed = state.get("validation_passed", False)
        validation_errors = state.get("validation_errors", [])

        if not validation_passed:
            error_msg = "; ".join(validation_errors) if validation_errors else "Validation failed"
            return {
                "error": f"Scope validation failed: {error_msg}",
                "error_type": "validation_error",
                "created_at": created_at,
                "created_by": created_by,
            }

        # Success - return final state
        return {
            "created_at": created_at,
            "created_by": created_by,
        }

    except Exception as e:
        return {
            "error": f"Error finalizing scope: {str(e)}",
            "error_type": "finalize_scope_error",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": "scope_definer",
        }


def create_scope_definer_graph() -> CompiledStateGraph:
    """Create the scope_definer LangGraph workflow.

    The workflow consists of 4 nodes:
    1. classify_problem: Infer problem type and target variable
    2. build_scope_spec: Build complete ScopeSpec with constraints
    3. define_success_criteria: Define performance thresholds
    4. finalize_scope: Add metadata and validate

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create graph with ScopeDefinerState
    workflow = StateGraph(ScopeDefinerState)

    # Add nodes
    workflow.add_node("classify_problem", classify_problem)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("build_scope_spec", build_scope_spec)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("define_success_criteria", define_success_criteria)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("finalize_scope", finalize_scope)  # type: ignore[type-var,arg-type,call-overload]

    # Define edges (linear flow)
    workflow.set_entry_point("classify_problem")
    workflow.add_edge("classify_problem", "build_scope_spec")
    workflow.add_edge("build_scope_spec", "define_success_criteria")
    workflow.add_edge("define_success_criteria", "finalize_scope")
    workflow.add_edge("finalize_scope", END)

    # Compile graph
    return workflow.compile()
