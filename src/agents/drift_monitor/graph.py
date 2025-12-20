"""Drift Monitor Agent Graph Assembly.

This module assembles the LangGraph workflow for the drift monitor agent.

Workflow: Sequential execution through all detection nodes
    START → data_drift → model_drift → concept_drift → alert_aggregator → END

Graph Architecture: .claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md lines 659-713
Contract: .claude/contracts/tier3-contracts.md lines 349-562
"""

from langgraph.graph import StateGraph, END

from src.agents.drift_monitor.state import DriftMonitorState
from src.agents.drift_monitor.nodes import (
    DataDriftNode,
    ModelDriftNode,
    ConceptDriftNode,
    AlertAggregatorNode,
)


def create_drift_monitor_graph() -> StateGraph:
    """Create the drift monitor agent graph.

    Workflow:
        1. data_drift: Detect feature distribution drift (PSI + KS test)
        2. model_drift: Detect prediction distribution drift
        3. concept_drift: Detect feature-target relationship drift
        4. alert_aggregator: Aggregate results and generate alerts

    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize graph
    workflow = StateGraph(DriftMonitorState)

    # Initialize nodes
    data_drift_node = DataDriftNode()
    model_drift_node = ModelDriftNode()
    concept_drift_node = ConceptDriftNode()
    alert_aggregator_node = AlertAggregatorNode()

    # Add nodes to graph
    workflow.add_node("data_drift", data_drift_node.execute)
    workflow.add_node("model_drift", model_drift_node.execute)
    workflow.add_node("concept_drift", concept_drift_node.execute)
    workflow.add_node("alert_aggregator", alert_aggregator_node.execute)

    # Define sequential workflow
    workflow.set_entry_point("data_drift")
    workflow.add_edge("data_drift", "model_drift")
    workflow.add_edge("model_drift", "concept_drift")
    workflow.add_edge("concept_drift", "alert_aggregator")
    workflow.add_edge("alert_aggregator", END)

    # Compile graph
    return workflow.compile()


# Export compiled graph
drift_monitor_graph = create_drift_monitor_graph()
