"""LangGraph workflow for Causal Impact Agent.

Assembles 5-node pipeline: graph_builder → estimation → refutation → sensitivity → interpretation
"""

from langgraph.graph import StateGraph, END
from src.agents.causal_impact.state import CausalImpactState
from src.agents.causal_impact.nodes.graph_builder import build_causal_graph
from src.agents.causal_impact.nodes.estimation import estimate_causal_effect
from src.agents.causal_impact.nodes.refutation import refute_causal_estimate
from src.agents.causal_impact.nodes.sensitivity import analyze_sensitivity
from src.agents.causal_impact.nodes.interpretation import interpret_results


def create_causal_impact_graph(enable_checkpointing: bool = False):
    """Create causal impact workflow graph.

    Pipeline:
    1. graph_builder: Construct causal DAG (Standard, <10s)
    2. estimation: Estimate causal effect (Standard, <30s)
    3. refutation: Robustness tests (Standard, <15s)
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

    # Linear flow (no conditional branching for simplicity)
    workflow.set_entry_point("graph_builder")
    workflow.add_edge("graph_builder", "estimation")
    workflow.add_edge("estimation", "refutation")
    workflow.add_edge("refutation", "sensitivity")
    workflow.add_edge("sensitivity", "interpretation")
    workflow.add_edge("interpretation", END)

    # Compile
    if enable_checkpointing:
        # Would add memory/checkpointing here in production
        return workflow.compile()
    else:
        return workflow.compile()
