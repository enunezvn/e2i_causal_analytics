"""
E2I Resource Optimizer Agent - Graph Assembly
Version: 4.2
Purpose: LangGraph workflow for resource allocation optimization
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from .state import ResourceOptimizerState
from .nodes import (
    ProblemFormulatorNode,
    OptimizerNode,
    ScenarioAnalyzerNode,
    ImpactProjectorNode,
)

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
    formulate -> optimize -> [scenario] -> project -> END

    Returns:
        Compiled LangGraph workflow
    """
    # Initialize nodes
    formulator = ProblemFormulatorNode()
    optimizer = OptimizerNode()
    scenario = ScenarioAnalyzerNode()
    projector = ImpactProjectorNode()

    # Build graph
    workflow = StateGraph(ResourceOptimizerState)

    # Add nodes
    workflow.add_node("formulate", formulator.execute)
    workflow.add_node("optimize", optimizer.execute)
    workflow.add_node("scenario", scenario.execute)
    workflow.add_node("project", projector.execute)
    workflow.add_node("error_handler", error_handler_node)

    # Set entry point
    workflow.set_entry_point("formulate")

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
    formulate -> optimize -> project -> END

    Returns:
        Compiled LangGraph workflow
    """
    # Initialize nodes
    formulator = ProblemFormulatorNode()
    optimizer = OptimizerNode()
    projector = ImpactProjectorNode()

    # Build graph
    workflow = StateGraph(ResourceOptimizerState)

    # Add nodes
    workflow.add_node("formulate", formulator.execute)
    workflow.add_node("optimize", optimizer.execute)
    workflow.add_node("project", projector.execute)
    workflow.add_node("error_handler", error_handler_node)

    # Set entry point
    workflow.set_entry_point("formulate")

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
