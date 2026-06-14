"""Experiment Monitor Agent Graph Assembly.

This module assembles the LangGraph workflow for the experiment monitor agent.

Workflow: Sequential execution through all monitoring nodes
    START → audit_init → health_checker → srm_detector → interim_analyzer → fidelity_checker → alert_generator → END

Tier: 3 (Monitoring)
"""

from functools import partial

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.base.audit_chain_mixin import (
    add_audited_node,
    create_workflow_initializer,
)
from src.agents.experiment_monitor.nodes import (
    AlertGeneratorNode,
    FidelityCheckerNode,
    HealthCheckerNode,
    InterimAnalyzerNode,
    SRMDetectorNode,
)
from src.agents.experiment_monitor.state import ExperimentMonitorState
from src.utils.audit_chain import AgentTier


def create_experiment_monitor_graph() -> CompiledStateGraph:
    """Create the experiment monitor agent graph.

    Workflow:
        0. audit_init: Initialize audit chain workflow (genesis block)
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

    # Create audit workflow initializer (genesis block of the tamper-evident chain)
    audit_initializer = create_workflow_initializer("experiment_monitor", AgentTier.MONITORING)

    # Initialize nodes
    health_checker_node = HealthCheckerNode()
    srm_detector_node = SRMDetectorNode()
    interim_analyzer_node = InterimAnalyzerNode()
    fidelity_checker_node = FidelityCheckerNode()
    alert_generator_node = AlertGeneratorNode()

    # Add nodes. Business nodes are wrapped via add_audited_node so each emits a
    # real timed audit entry (duration_ms) -> populates the analytics latency panel.
    timed = partial(
        add_audited_node, agent_name="experiment_monitor", agent_tier=AgentTier.MONITORING
    )
    workflow.add_node("audit_init", audit_initializer)  # type: ignore[type-var,arg-type,call-overload]  # Initialize audit chain (genesis)
    timed(workflow, "health_checker", health_checker_node.execute)
    timed(workflow, "srm_detector", srm_detector_node.execute)
    timed(workflow, "interim_analyzer", interim_analyzer_node.execute)
    timed(workflow, "fidelity_checker", fidelity_checker_node.execute)
    timed(workflow, "alert_generator", alert_generator_node.execute)

    # Define sequential workflow starting with audit initialization
    workflow.set_entry_point("audit_init")
    workflow.add_edge("audit_init", "health_checker")
    workflow.add_edge("health_checker", "srm_detector")
    workflow.add_edge("srm_detector", "interim_analyzer")
    workflow.add_edge("interim_analyzer", "fidelity_checker")
    workflow.add_edge("fidelity_checker", "alert_generator")
    workflow.add_edge("alert_generator", END)

    # Compile graph
    return workflow.compile()


# Export compiled graph
experiment_monitor_graph = create_experiment_monitor_graph()
