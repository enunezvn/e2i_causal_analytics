"""
E2I Resource Optimizer Agent - Graph Assembly
Version: 4.2
Purpose: LangGraph workflow for resource allocation optimization

Observability:
- Audit chain recording for tamper-evident logging
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.base.audit_chain_mixin import (
    add_audited_node,
    create_workflow_initializer,
)
from src.utils.audit_chain import AgentTier

from .nodes import (
    ImpactProjectorNode,
    OptimizerNode,
    ProblemFormulatorNode,
    ScenarioAnalyzerNode,
)
from .state import ResourceOptimizerState

logger = logging.getLogger(__name__)


async def error_handler_node(state: ResourceOptimizerState) -> ResourceOptimizerState:
    """Handle errors and finalize failed state."""
    return {
        **state,
        "optimization_summary": "Optimization could not be completed.",
        "status": "failed",
    }


def route_after_formulation(state: ResourceOptimizerState) -> str:
    """Route after problem formulation."""
    if state.get("status") == "failed":
        return "error"
    return "optimize"


def route_after_optimization(state: ResourceOptimizerState) -> str:
    """Route after optimization."""
    if state.get("status") == "failed":
        return "error"
    if state.get("run_scenarios"):
        return "scenario"
    return "project"


def build_resource_optimizer_graph() -> Any:
    """
    Build the full Resource Optimizer agent graph.

    Pipeline:
    audit_init -> formulate -> optimize -> [scenario] -> project -> END

    Returns:
        Compiled LangGraph workflow
    """
    # Create audit workflow initializer
    audit_initializer = create_workflow_initializer("resource_optimizer", AgentTier.ML_PREDICTIONS)

    # Initialize nodes
    formulator = ProblemFormulatorNode()
    optimizer = OptimizerNode()
    scenario = ScenarioAnalyzerNode()
    projector = ImpactProjectorNode()

    # Build graph
    workflow = StateGraph(ResourceOptimizerState)

    # Add nodes. Business nodes are wrapped via add_audited_node so each emits a
    # real timed audit entry (duration_ms) -> populates the analytics latency panel.
    timed = partial(
        add_audited_node, agent_name="resource_optimizer", agent_tier=AgentTier.ML_PREDICTIONS
    )
    workflow.add_node("audit_init", audit_initializer)  # type: ignore[type-var,arg-type,call-overload]  # Initialize audit chain (genesis)
    timed(workflow, "formulate", formulator.execute)
    timed(workflow, "optimize", optimizer.execute)
    timed(workflow, "scenario", scenario.execute)
    timed(workflow, "project", projector.execute)
    workflow.add_node("error_handler", error_handler_node)  # type: ignore[type-var,arg-type,call-overload]

    # Set entry point - start with audit initialization
    workflow.set_entry_point("audit_init")

    # Edge from audit_init to formulate
    workflow.add_edge("audit_init", "formulate")

    # Add edges
    workflow.add_conditional_edges(
        "formulate",
        route_after_formulation,
        {"optimize": "optimize", "error": "error_handler"},
    )

    workflow.add_conditional_edges(
        "optimize",
        route_after_optimization,
        {"scenario": "scenario", "project": "project", "error": "error_handler"},
    )

    workflow.add_edge("scenario", "project")
    workflow.add_edge("project", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile()


def build_simple_optimizer_graph() -> Any:
    """
    Build a simplified Resource Optimizer graph without scenario analysis.

    Pipeline:
    audit_init -> formulate -> optimize -> project -> END

    Returns:
        Compiled LangGraph workflow
    """
    # Create audit workflow initializer
    audit_initializer = create_workflow_initializer(
        "resource_optimizer_simple", AgentTier.ML_PREDICTIONS
    )

    # Initialize nodes
    formulator = ProblemFormulatorNode()
    optimizer = OptimizerNode()
    projector = ImpactProjectorNode()

    # Build graph
    workflow = StateGraph(ResourceOptimizerState)

    # Add nodes. Business nodes are wrapped via add_audited_node so each emits a
    # real timed audit entry (duration_ms) -> populates the analytics latency panel.
    timed = partial(
        add_audited_node,
        agent_name="resource_optimizer_simple",
        agent_tier=AgentTier.ML_PREDICTIONS,
    )
    workflow.add_node("audit_init", audit_initializer)  # type: ignore[type-var,arg-type,call-overload]  # Initialize audit chain (genesis)
    timed(workflow, "formulate", formulator.execute)
    timed(workflow, "optimize", optimizer.execute)
    timed(workflow, "project", projector.execute)
    workflow.add_node("error_handler", error_handler_node)  # type: ignore[type-var,arg-type,call-overload]

    # Set entry point - start with audit initialization
    workflow.set_entry_point("audit_init")

    workflow.add_edge("audit_init", "formulate")

    # Add edges
    workflow.add_conditional_edges(
        "formulate",
        lambda s: "error" if s.get("status") == "failed" else "optimize",
        {"optimize": "optimize", "error": "error_handler"},
    )

    workflow.add_conditional_edges(
        "optimize",
        lambda s: "error" if s.get("status") == "failed" else "project",
        {"project": "project", "error": "error_handler"},
    )

    workflow.add_edge("project", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile()
