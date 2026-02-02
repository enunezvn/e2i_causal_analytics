"""LangGraph assembly for CohortConstructor agent.

This module defines the LangGraph workflow for patient cohort construction
following FDA/EMA label criteria.

Workflow:
    validate_config → apply_criteria → validate_temporal → generate_metadata → END
"""

import logging
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from .nodes import (
    apply_criteria,
    generate_metadata,
    validate_config,
    validate_temporal,
)
from .state import CohortConstructorState

logger = logging.getLogger(__name__)


def _should_continue(state: Dict[str, Any]) -> str:
    """Determine if workflow should continue or handle error.

    Args:
        state: Current pipeline state

    Returns:
        Next node name or "error_handler"
    """
    if state.get("status") == "failed":
        return "error_handler"
    return "continue"


async def error_handler(state: CohortConstructorState) -> Dict[str, Any]:
    """Handle errors in the workflow.

    This node captures error state and prepares final output
    for failed workflows.

    Args:
        state: Current state with error information

    Returns:
        Updated state with error handling complete
    """
    error = state.get("error", "Unknown error")
    error_code = state.get("error_code", "CC_000")
    error_category = state.get("error_category", "UNKNOWN")

    logger.error(f"CohortConstructor workflow failed: [{error_code}] {error_category} - {error}")

    return {
        "current_phase": "complete",
        "status": "failed",
        "pipeline_blocked": True,
        "block_reason": error,
        "suggested_next_agent": None,
        "key_findings": [f"Cohort construction failed: {error}"],
        "confidence": 0.0,
    }


def create_cohort_constructor_graph() -> StateGraph:
    """Create the CohortConstructor LangGraph workflow.

    The workflow consists of 4 main nodes:
    1. validate_config: Validate configuration and input data
    2. apply_criteria: Apply inclusion/exclusion criteria
    3. validate_temporal: Validate temporal eligibility
    4. generate_metadata: Generate execution metadata and audit trail

    Flow:
        validate_config → apply_criteria → validate_temporal → generate_metadata → END

    With error handling:
        Any node failure → error_handler → END

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create graph with CohortConstructorState
    workflow = StateGraph(CohortConstructorState)

    # Add main nodes
    workflow.add_node("validate_config", validate_config)
    workflow.add_node("apply_criteria", apply_criteria)
    workflow.add_node("validate_temporal", validate_temporal)
    workflow.add_node("generate_metadata", generate_metadata)
    workflow.add_node("error_handler", error_handler)

    # Set entry point
    workflow.set_entry_point("validate_config")

    # Add conditional edges for error handling
    workflow.add_conditional_edges(
        "validate_config",
        _should_continue,
        {
            "continue": "apply_criteria",
            "error_handler": "error_handler",
        },
    )

    workflow.add_conditional_edges(
        "apply_criteria",
        _should_continue,
        {
            "continue": "validate_temporal",
            "error_handler": "error_handler",
        },
    )

    workflow.add_conditional_edges(
        "validate_temporal",
        _should_continue,
        {
            "continue": "generate_metadata",
            "error_handler": "error_handler",
        },
    )

    workflow.add_conditional_edges(
        "generate_metadata",
        _should_continue,
        {
            "continue": END,
            "error_handler": "error_handler",
        },
    )

    # Error handler always ends
    workflow.add_edge("error_handler", END)

    # Compile and return
    return workflow.compile()


def create_simple_cohort_constructor_graph() -> StateGraph:
    """Create a simple CohortConstructor graph without error handling.

    This version uses linear flow without conditional edges.
    Useful for testing and simple use cases.

    Returns:
        Compiled StateGraph with linear flow
    """
    workflow = StateGraph(CohortConstructorState)

    # Add nodes
    workflow.add_node("validate_config", validate_config)
    workflow.add_node("apply_criteria", apply_criteria)
    workflow.add_node("validate_temporal", validate_temporal)
    workflow.add_node("generate_metadata", generate_metadata)

    # Linear flow
    workflow.set_entry_point("validate_config")
    workflow.add_edge("validate_config", "apply_criteria")
    workflow.add_edge("apply_criteria", "validate_temporal")
    workflow.add_edge("validate_temporal", "generate_metadata")
    workflow.add_edge("generate_metadata", END)

    return workflow.compile()
