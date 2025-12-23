"""LangGraph workflow for Causal Impact Agent.

Contract-compliant workflow with conditional routing:
  graph_builder → estimation → [refutation|interpretation|error] → sensitivity → interpretation → END

Contract: .claude/contracts/tier2-contracts.md
Observability: Per-node Opik tracing (CONTRACT_VALIDATION.md #12)
"""

import functools
import logging
from typing import Any, Callable, Dict, Literal, TypeVar

from langgraph.graph import END, StateGraph

from src.agents.causal_impact.nodes.estimation import estimate_causal_effect
from src.agents.causal_impact.nodes.graph_builder import build_causal_graph
from src.agents.causal_impact.nodes.interpretation import interpret_results
from src.agents.causal_impact.nodes.refutation import refute_causal_estimate
from src.agents.causal_impact.nodes.sensitivity import analyze_sensitivity
from src.agents.causal_impact.state import CausalImpactState
from src.mlops.opik_connector import get_opik_connector

logger = logging.getLogger(__name__)

# Type variable for node functions
F = TypeVar("F", bound=Callable[..., Any])


def traced_node(node_name: str) -> Callable[[F], F]:
    """Decorator to add Opik tracing to workflow nodes.

    Creates a span for each node execution with:
    - Node name and operation tracking
    - Input/output data (sanitized for large fields)
    - Latency measurement
    - Error tracking
    - Parent span linking via state.span_id

    Args:
        node_name: Name of the node (e.g., "graph_builder", "estimation")

    Returns:
        Decorated async function with Opik tracing

    Example:
        @traced_node("graph_builder")
        async def build_causal_graph(state: CausalImpactState) -> Dict:
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(state: CausalImpactState) -> Dict[str, Any]:
            opik = get_opik_connector()

            # Extract tracing context from state
            trace_id = state.get("query_id")  # Use query_id as trace correlation
            parent_span_id = state.get("span_id")  # Parent span from dispatcher
            session_id = state.get("session_id")

            # Prepare sanitized input (exclude large data structures)
            sanitized_input = {
                "query": state.get("query"),
                "treatment_var": state.get("treatment_var"),
                "outcome_var": state.get("outcome_var"),
                "current_phase": state.get("current_phase"),
                "session_id": session_id,
            }

            # Metadata for the span
            metadata = {
                "node_name": node_name,
                "agent_name": "causal_impact",
                "session_id": session_id,
                "dispatch_id": state.get("dispatch_id"),
            }

            async with opik.trace_agent(
                agent_name="causal_impact",
                operation=node_name,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                metadata=metadata,
                tags=["causal_impact", node_name, "workflow_node"],
                input_data=sanitized_input,
            ) as span:
                try:
                    # Execute the actual node function
                    result = await func(state)

                    # Set output data (sanitized)
                    output_summary = {
                        "current_phase": result.get("current_phase"),
                        "status": result.get("status"),
                        "has_error": bool(result.get(f"{node_name}_error")),
                    }

                    # Add node-specific output fields
                    if node_name == "graph_builder":
                        output_summary["graph_confidence"] = result.get(
                            "causal_graph", {}
                        ).get("confidence")
                    elif node_name == "estimation":
                        est = result.get("estimation_result", {})
                        output_summary["ate"] = est.get("ate")
                        output_summary["p_value"] = est.get("p_value")
                        output_summary["statistical_significance"] = est.get(
                            "statistical_significance"
                        )
                    elif node_name == "refutation":
                        ref = result.get("refutation_results", {})
                        output_summary["tests_passed"] = ref.get("tests_passed")
                        output_summary["overall_robust"] = ref.get("overall_robust")
                        output_summary["gate_decision"] = ref.get("gate_decision")
                    elif node_name == "sensitivity":
                        sens = result.get("sensitivity_analysis", {})
                        output_summary["e_value"] = sens.get("e_value")
                        output_summary["robust_to_confounding"] = sens.get(
                            "robust_to_confounding"
                        )
                    elif node_name == "interpretation":
                        interp = result.get("interpretation", {})
                        output_summary["causal_confidence"] = interp.get(
                            "causal_confidence"
                        )
                        output_summary["depth_level"] = interp.get("depth_level")

                    span.set_output(output_summary)

                    # Set latency attribute from node result
                    latency_key = f"{node_name}_latency_ms"
                    if latency_key in result:
                        span.set_attribute("node_latency_ms", result[latency_key])

                    return result

                except Exception as e:
                    # Log error details to span
                    span.set_attribute("error", str(e))
                    span.set_attribute("error_type", type(e).__name__)
                    logger.error(f"Node {node_name} failed: {e}")
                    raise

        return wrapper  # type: ignore

    return decorator


# Create traced versions of node functions
traced_build_causal_graph = traced_node("graph_builder")(build_causal_graph)
traced_estimate_causal_effect = traced_node("estimation")(estimate_causal_effect)
traced_refute_causal_estimate = traced_node("refutation")(refute_causal_estimate)
traced_analyze_sensitivity = traced_node("sensitivity")(analyze_sensitivity)
traced_interpret_results = traced_node("interpretation")(interpret_results)


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

    # Add nodes with Opik tracing wrappers (CONTRACT_VALIDATION.md #12)
    workflow.add_node("graph_builder", traced_build_causal_graph)
    workflow.add_node("estimation", traced_estimate_causal_effect)
    workflow.add_node("refutation", traced_refute_causal_estimate)
    workflow.add_node("sensitivity", traced_analyze_sensitivity)
    workflow.add_node("interpretation", traced_interpret_results)
    workflow.add_node("error_handler", handle_workflow_error)  # Error handler not traced

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
