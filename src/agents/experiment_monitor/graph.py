"""Experiment Monitor Agent Graph Assembly.

This module assembles the LangGraph workflow for the experiment monitor agent.

Workflow: Sequential execution through all monitoring nodes
    START → health_checker → srm_detector → interim_analyzer → fidelity_checker → alert_generator → END

Tier: 3 (Monitoring)
"""

from langgraph.graph import END, StateGraph

from src.agents.experiment_monitor.nodes import (
    AlertGeneratorNode,
    FidelityCheckerNode,
    HealthCheckerNode,
    InterimAnalyzerNode,
    SRMDetectorNode,
)
from src.agents.experiment_monitor.state import ExperimentMonitorState


def create_experiment_monitor_graph() -> StateGraph:
    """Create the experiment monitor agent graph.

    Workflow:
        1. health_checker: Check experiment health, enrollment rates, and stale data
        2. srm_detector: Detect sample ratio mismatch
        3. interim_analyzer: Check for interim analysis triggers
        4. fidelity_checker: Check Digital Twin prediction fidelity
        5. alert_generator: Generate alerts and recommendations

    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize graph
    workflow = StateGraph(ExperimentMonitorState)

    # Initialize nodes
    health_checker_node = HealthCheckerNode()
    srm_detector_node = SRMDetectorNode()
    interim_analyzer_node = InterimAnalyzerNode()
    fidelity_checker_node = FidelityCheckerNode()
    alert_generator_node = AlertGeneratorNode()

    # Add nodes to graph
    workflow.add_node("health_checker", health_checker_node.execute)
    workflow.add_node("srm_detector", srm_detector_node.execute)
    workflow.add_node("interim_analyzer", interim_analyzer_node.execute)
    workflow.add_node("fidelity_checker", fidelity_checker_node.execute)
    workflow.add_node("alert_generator", alert_generator_node.execute)

    # Define sequential workflow
    workflow.set_entry_point("health_checker")
    workflow.add_edge("health_checker", "srm_detector")
    workflow.add_edge("srm_detector", "interim_analyzer")
    workflow.add_edge("interim_analyzer", "fidelity_checker")
    workflow.add_edge("fidelity_checker", "alert_generator")
    workflow.add_edge("alert_generator", END)

    # Compile graph
    return workflow.compile()


# Export compiled graph
experiment_monitor_graph = create_experiment_monitor_graph()
