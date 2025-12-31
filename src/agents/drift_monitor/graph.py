"""Drift Monitor Agent Graph Assembly.

This module assembles the LangGraph workflow for the drift monitor agent.

Workflow: Sequential execution through all detection nodes
    audit_init → data_drift → model_drift → concept_drift → structural_drift → alert_aggregator → END

V4.4: Added structural_drift node for causal DAG structure drift detection.

Observability:
- Audit chain recording for tamper-evident logging

Graph Architecture: .claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md lines 659-713
Contract: .claude/contracts/tier3-contracts.md lines 349-562
"""

from langgraph.graph import END, StateGraph

from src.agents.base.audit_chain_mixin import create_workflow_initializer
from src.agents.drift_monitor.nodes import (
    AlertAggregatorNode,
    ConceptDriftNode,
    DataDriftNode,
    ModelDriftNode,
    StructuralDriftNode,
)
from src.agents.drift_monitor.state import DriftMonitorState
from src.utils.audit_chain import AgentTier


def create_drift_monitor_graph() -> StateGraph:
    """Create the drift monitor agent graph.

    Workflow:
        0. audit_init: Initialize audit chain workflow (genesis block)
        1. data_drift: Detect feature distribution drift (PSI + KS test)
        2. model_drift: Detect prediction distribution drift
        3. concept_drift: Detect feature-target relationship drift
        4. structural_drift: Detect causal DAG structure drift (V4.4)
        5. alert_aggregator: Aggregate results and generate alerts

    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize graph
    workflow = StateGraph(DriftMonitorState)

    # Create audit workflow initializer
    audit_initializer = create_workflow_initializer("drift_monitor", AgentTier.MONITORING)

    # Initialize nodes
    data_drift_node = DataDriftNode()
    model_drift_node = ModelDriftNode()
    concept_drift_node = ConceptDriftNode()
    structural_drift_node = StructuralDriftNode()  # V4.4
    alert_aggregator_node = AlertAggregatorNode()

    # Add nodes to graph
    workflow.add_node("audit_init", audit_initializer)  # Initialize audit chain
    workflow.add_node("data_drift", data_drift_node.execute)
    workflow.add_node("model_drift", model_drift_node.execute)
    workflow.add_node("concept_drift", concept_drift_node.execute)
    workflow.add_node("structural_drift", structural_drift_node.execute)  # V4.4
    workflow.add_node("alert_aggregator", alert_aggregator_node.execute)

    # Define sequential workflow starting with audit initialization
    # V4.4: Added structural_drift between concept_drift and alert_aggregator
    workflow.set_entry_point("audit_init")
    workflow.add_edge("audit_init", "data_drift")
    workflow.add_edge("data_drift", "model_drift")
    workflow.add_edge("model_drift", "concept_drift")
    workflow.add_edge("concept_drift", "structural_drift")  # V4.4
    workflow.add_edge("structural_drift", "alert_aggregator")  # V4.4
    workflow.add_edge("alert_aggregator", END)

    # Compile graph
    return workflow.compile()


# Export compiled graph
drift_monitor_graph = create_drift_monitor_graph()
