"""LangGraph workflow for Causal Impact Agent.

Contract-compliant workflow with conditional routing:
  graph_builder → estimation → [refutation|interpretation|error] → sensitivity → interpretation → END

Contract: .claude/contracts/tier2-contracts.md
"""

from typing import Literal

from langgraph.graph import END, StateGraph

from src.agents.causal_impact.nodes.estimation import estimate_causal_effect
from src.agents.causal_impact.nodes.graph_builder import build_causal_graph
from src.agents.causal_impact.nodes.interpretation import interpret_results
from src.agents.causal_impact.nodes.refutation import refute_causal_estimate
from src.agents.causal_impact.nodes.sensitivity import analyze_sensitivity
from src.agents.causal_impact.state import CausalImpactState


def should_continue_after_estimation(
    state: CausalImpactState,
) -> Literal["refutation", "interpretation", "error_handler"]:
    """Conditional routing after estimation node.

    Contract: Partial success path - if ate_estimate exists, skip to interpretation on error.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    if state.get("estimation_error"):
        # Partial success: if we have an ATE estimate, go to interpretation
        if state.get("estimation_result", {}).get("ate") is not None:
            return "interpretation"
        return "error_handler"
    return "refutation"


def should_continue_after_refutation(
    state: CausalImpactState,
) -> Literal["sensitivity", "error_handler"]:
    """Conditional routing after refutation node.

    Contract: gate_decision determines flow.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    gate = state.get("refutation_results", {}).get("gate_decision", "proceed")
    if gate == "block":
        return "error_handler"
    return "sensitivity"


def handle_workflow_error(state: CausalImpactState) -> CausalImpactState:
    """Handle workflow errors gracefully.

    Contract: Accumulate errors and mark workflow as failed.

    Args:
        state: Current workflow state

    Returns:
        Updated state with error status
    """
    error_msg = state.get("error_message") or "Unknown error occurred"

    # Accumulate error if not already present
    errors = list(state.get("errors", []))
    errors.append({"phase": state.get("current_phase", "unknown"), "message": error_msg})

    return {
        **state,
        "status": "failed",
        "errors": errors,
        "current_phase": "failed",
    }


def create_causal_impact_graph(enable_checkpointing: bool = False):
    """Create causal impact workflow graph with conditional routing.

    Contract-compliant pipeline with error handling:
    1. graph_builder: Construct causal DAG (Standard, <10s)
    2. estimation: Estimate causal effect (Standard, <30s)
       → conditional: refutation | interpretation (partial success) | error_handler
    3. refutation: Robustness tests (Standard, <15s)
       → conditional: sensitivity | error_handler (if blocked)
    4. sensitivity: E-value analysis (Standard, <5s)
    5. interpretation: Natural language output (Deep Reasoning, <30s)

    Total target: <120s (60s computation + 30s interpretation)

    Args:
        enable_checkpointing: Whether to enable state checkpointing

    Returns:
        Compiled LangGraph workflow
    """
    # Create workflow
    workflow = StateGraph(CausalImpactState)

    # Add nodes
    workflow.add_node("graph_builder", build_causal_graph)
    workflow.add_node("estimation", estimate_causal_effect)
    workflow.add_node("refutation", refute_causal_estimate)
    workflow.add_node("sensitivity", analyze_sensitivity)
    workflow.add_node("interpretation", interpret_results)
    workflow.add_node("error_handler", handle_workflow_error)

    # Set entry point
    workflow.set_entry_point("graph_builder")

    # Linear edge: graph_builder → estimation
    workflow.add_edge("graph_builder", "estimation")

    # Conditional edge after estimation (contract: partial success routing)
    workflow.add_conditional_edges(
        "estimation",
        should_continue_after_estimation,
        {
            "refutation": "refutation",
            "interpretation": "interpretation",
            "error_handler": "error_handler",
        },
    )

    # Conditional edge after refutation (contract: gate_decision routing)
    workflow.add_conditional_edges(
        "refutation",
        should_continue_after_refutation,
        {
            "sensitivity": "sensitivity",
            "error_handler": "error_handler",
        },
    )

    # Linear edges for remaining flow
    workflow.add_edge("sensitivity", "interpretation")
    workflow.add_edge("interpretation", END)
    workflow.add_edge("error_handler", END)

    # Compile
    if enable_checkpointing:
        # Would add memory/checkpointing here in production
        return workflow.compile()
    else:
        return workflow.compile()
