"""Experiment Designer Agent Graph Assembly.

This module assembles the LangGraph workflow for the experiment designer agent.

Workflow: Sequential execution with conditional redesign loop
    START → context_loader → design_reasoning → power_analysis → validity_audit →
    (conditional: redesign → power_analysis) → template_generator → END

Graph Architecture: .claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md lines 864-951
Contract: .claude/contracts/tier3-contracts.md lines 82-142
"""

import asyncio
from typing import Any, Callable

from langgraph.graph import END, StateGraph

from src.agents.experiment_designer.nodes import (
    ContextLoaderNode,
    DesignReasoningNode,
    PowerAnalysisNode,
    RedesignNode,
    TemplateGeneratorNode,
    ValidityAuditNode,
)
from src.agents.experiment_designer.state import ExperimentDesignState


def wrap_async_node(async_func: Callable) -> Callable:
    """Wrap an async node function to work in sync mode.

    LangGraph needs sync functions for graph.invoke().
    This wrapper allows async nodes to work with both invoke() and ainvoke().

    Args:
        async_func: Async function that processes state

    Returns:
        Sync function that runs the async function
    """
    import nest_asyncio

    try:
        nest_asyncio.apply()
    except RuntimeError:
        # Already applied or no event loop
        pass

    def sync_wrapper(state: ExperimentDesignState) -> ExperimentDesignState:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Use nest_asyncio for nested event loops
            return loop.run_until_complete(async_func(state))
        else:
            # No running loop, create a new one
            return asyncio.run(async_func(state))

    return sync_wrapper


def error_handler_node(state: ExperimentDesignState) -> ExperimentDesignState:
    """Handle errors and prepare final state for failed workflows.

    Args:
        state: Current agent state with errors

    Returns:
        Updated state with error summary
    """
    errors = state.get("errors", [])
    error_summary = "; ".join([e.get("error", "Unknown error") for e in errors])

    state["warnings"] = state.get("warnings", []) + [
        f"Workflow failed with {len(errors)} error(s): {error_summary}"
    ]

    # Ensure status is failed
    state["status"] = "failed"

    return state


def create_experiment_designer_graph(
    knowledge_store: Any = None,
    max_redesign_iterations: int = 2,
) -> StateGraph:
    """Create the experiment designer agent graph with redesign loop.

    Workflow:
        1. context_loader: Load organizational learning context
        2. design_reasoning: Deep reasoning for experiment design (LLM)
        3. power_analysis: Statistical power calculations
        4. validity_audit: Adversarial validity assessment (LLM)
        5. (conditional) redesign: Incorporate feedback if needed
        6. template_generator: Generate DoWhy code and pre-registration docs

    The redesign loop allows the agent to iterate on the design based on
    validity audit feedback, up to max_redesign_iterations times.

    Args:
        knowledge_store: Optional knowledge store for context loading
        max_redesign_iterations: Maximum number of redesign iterations (default: 2)

    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize nodes
    context_node = ContextLoaderNode(knowledge_store)
    design_node = DesignReasoningNode()
    power_node = PowerAnalysisNode()
    validity_node = ValidityAuditNode()
    redesign_node = RedesignNode()
    template_node = TemplateGeneratorNode()

    # Initialize graph
    workflow = StateGraph(ExperimentDesignState)

    # Add nodes to graph (wrapped for sync compatibility)
    workflow.add_node("context_loader", wrap_async_node(context_node.execute))
    workflow.add_node("design_reasoning", wrap_async_node(design_node.execute))
    workflow.add_node("power_analysis", wrap_async_node(power_node.execute))
    workflow.add_node("validity_audit", wrap_async_node(validity_node.execute))
    workflow.add_node("redesign", wrap_async_node(redesign_node.execute))
    workflow.add_node("template_generator", wrap_async_node(template_node.execute))
    workflow.add_node("error_handler", error_handler_node)

    # Set entry point
    workflow.set_entry_point("context_loader")

    # Define edges
    # context_loader → design_reasoning (always)
    workflow.add_edge("context_loader", "design_reasoning")

    # design_reasoning → power_analysis or error_handler
    def design_to_next(state: ExperimentDesignState) -> str:
        if state.get("status") == "failed":
            return "error_handler"
        return "power_analysis"

    workflow.add_conditional_edges(
        "design_reasoning",
        design_to_next,
        {"power_analysis": "power_analysis", "error_handler": "error_handler"},
    )

    # power_analysis → validity_audit (always, even on soft failures)
    workflow.add_edge("power_analysis", "validity_audit")

    # validity_audit → redesign, template_generator, or error_handler
    def validity_to_next(state: ExperimentDesignState) -> str:
        """Determine next step after validity audit.

        Returns:
            - "redesign" if redesign is needed and iterations available
            - "template_generator" if proceeding with current design
            - "error_handler" if workflow failed
        """
        if state.get("status") == "failed":
            return "error_handler"

        if state.get("redesign_needed", False):
            current_iteration = state.get("current_iteration", 0)
            if current_iteration < max_redesign_iterations:
                return "redesign"

        return "template_generator"

    workflow.add_conditional_edges(
        "validity_audit",
        validity_to_next,
        {
            "redesign": "redesign",
            "template_generator": "template_generator",
            "error_handler": "error_handler",
        },
    )

    # redesign → power_analysis (loop back)
    workflow.add_edge("redesign", "power_analysis")

    # template_generator → END
    workflow.add_edge("template_generator", END)

    # error_handler → END
    workflow.add_edge("error_handler", END)

    # Compile graph
    return workflow.compile()


# Export compiled graph with default settings
experiment_designer_graph = create_experiment_designer_graph()


def create_initial_state(
    business_question: str,
    constraints: dict[str, Any] | None = None,
    available_data: dict[str, Any] | None = None,
    preregistration_formality: str = "medium",
    max_redesign_iterations: int = 2,
    enable_validity_audit: bool = True,
) -> ExperimentDesignState:
    """Create initial state for experiment designer workflow.

    This is a convenience function for creating properly initialized state.

    Args:
        business_question: The business question to design an experiment for
        constraints: Experimental constraints (budget, timeline, etc.)
        available_data: Available data sources and variables
        preregistration_formality: Level of pre-registration detail ("light", "medium", "heavy")
        max_redesign_iterations: Maximum redesign iterations
        enable_validity_audit: Whether to run validity audit

    Returns:
        Initialized ExperimentDesignState
    """
    return ExperimentDesignState(
        # Input fields
        business_question=business_question,
        constraints=constraints or {},
        available_data=available_data or {},
        preregistration_formality=preregistration_formality,  # type: ignore
        max_redesign_iterations=max_redesign_iterations,
        enable_validity_audit=enable_validity_audit,
        # Error handling (initialized)
        errors=[],
        warnings=[],
        status="pending",
    )
