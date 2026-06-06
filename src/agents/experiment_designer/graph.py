"""Experiment Designer Agent Graph Assembly.

This module assembles the LangGraph workflow for the experiment designer agent.

Workflow: Sequential execution with conditional redesign loop
    audit_init → context_loader → twin_simulation (optional) → design_reasoning →
    power_analysis → validity_audit → (conditional: redesign → power_analysis) →
    template_generator → END

Phase 15 Integration:
    - Added twin_simulation node between context_loader and design_reasoning
    - If simulation recommends SKIP, workflow exits early
    - If simulation recommends DEPLOY, passes prior_estimate to power_analysis

Observability:
- Audit chain recording for tamper-evident logging

Graph Architecture: .claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md lines 864-951
Contract: .claude/contracts/tier3-contracts.md lines 82-142
"""

import asyncio
from typing import Any, Callable

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.base.audit_chain_mixin import create_workflow_initializer
from src.agents.experiment_designer.nodes import (
    ContextLoaderNode,
    DesignReasoningNode,
    PowerAnalysisNode,
    RedesignNode,
    TemplateGeneratorNode,
    TwinSimulationNode,
    ValidityAuditNode,
)
from src.agents.experiment_designer.state import ExperimentDesignState
from src.utils.audit_chain import AgentTier


def wrap_async_node(async_func: Callable) -> Callable:
    """Wrap an async node function to work in sync mode.

    LangGraph needs sync functions for graph.invoke().
    This wrapper allows async nodes to work with both invoke() and ainvoke().

    Args:
        async_func: Async function that processes state

    Returns:
        Sync function that runs the async function
    """
    # Issue #215: nest_asyncio.apply() is a PROCESS-WIDE monkey-patch of
    # asyncio.run; calling it at wrap_async_node construction time (i.e.
    # when create_experiment_designer_graph() runs) pollutes sibling pytest
    # workers and produces ``RuntimeError: Event loop is closed`` in
    # unrelated tests that use plain ``asyncio.run(...)`` later on the same
    # worker (e.g. tests/integration/test_adaptive_validity_check_ablation_layer3.py).
    # Defer the apply() to inside sync_wrapper's running-loop branch where
    # it is actually needed for ``loop.run_until_complete`` to nest. Import
    # nest_asyncio at construction time (cheap, doesn't pollute) so the
    # closure captures availability once.
    try:
        import nest_asyncio as _nest_asyncio

        _has_nest_asyncio = True
    except ImportError:
        _nest_asyncio = None  # type: ignore[assignment]
        _has_nest_asyncio = False

    def sync_wrapper(state: ExperimentDesignState) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            if _has_nest_asyncio:
                # We are actually nested — invoke nest_asyncio.apply() ONLY
                # here, not at wrap_async_node construction. apply() is
                # idempotent across repeated calls; confining it to the
                # actually-nested execution path means the import-time +
                # graph-construction-time paths leave asyncio.run untouched.
                try:
                    _nest_asyncio.apply()  # type: ignore[union-attr]
                except RuntimeError:
                    # Already applied or no event loop
                    pass
                return loop.run_until_complete(async_func(state))
            else:
                # Fallback: create new event loop (may cause issues in nested scenarios)
                import warnings

                warnings.warn(
                    "nest_asyncio not installed. Nested event loop handling may fail.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                new_loop = asyncio.new_event_loop()
                try:
                    return new_loop.run_until_complete(async_func(state))
                finally:
                    new_loop.close()
        else:
            # No running loop, create a new one
            return asyncio.run(async_func(state))

    return sync_wrapper


def error_handler_node(state: ExperimentDesignState) -> dict[str, Any]:
    """Handle errors and prepare final state for failed workflows.

    Args:
        state: Current agent state with errors

    Returns:
        Updated state with error summary
    """
    errors = state.get("errors", [])
    error_summary = "; ".join([e.get("error", "Unknown error") for e in errors])

    return {
        "warnings": state.get("warnings", [])
        + [f"Workflow failed with {len(errors)} error(s): {error_summary}"],
        "status": "failed",
    }


def create_experiment_designer_graph(
    knowledge_store: Any = None,
    max_redesign_iterations: int = 2,
    enable_twin_simulation: bool = True,
    auto_skip_on_low_effect: bool = True,
) -> CompiledStateGraph:  # type: ignore[type-arg]
    """Create the experiment designer agent graph with redesign loop.

    Workflow:
        0. audit_init: Initialize audit chain workflow (genesis block)
        1. context_loader: Load organizational learning context
        2. twin_simulation: Digital twin pre-screening (Phase 15)
        3. design_reasoning: Deep reasoning for experiment design (LLM)
        4. power_analysis: Statistical power calculations
        5. validity_audit: Adversarial validity assessment (LLM)
        6. (conditional) redesign: Incorporate feedback if needed
        7. template_generator: Generate DoWhy code and pre-registration docs

    The redesign loop allows the agent to iterate on the design based on
    validity audit feedback, up to max_redesign_iterations times.

    Phase 15 Integration:
        - Twin simulation pre-screens experiments before design
        - If simulation recommends SKIP, workflow exits early
        - If simulation recommends DEPLOY, prior_estimate passes to power_analysis

    Args:
        knowledge_store: Optional knowledge store for context loading
        max_redesign_iterations: Maximum number of redesign iterations (default: 2)
        enable_twin_simulation: Whether to WIRE the twin pre-screen node into the
            graph (default: True). When True the node is present but stays dark at
            runtime unless state["enable_twin_simulation"] is also set (default
            False, seeded from ExperimentDesignerInput). When False the node is
            excised entirely and context_loader flows straight to design_reasoning.
        auto_skip_on_low_effect: If True, skip workflow when twin recommends SKIP (default: True)

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create audit workflow initializer
    audit_initializer = create_workflow_initializer("experiment_designer", AgentTier.MONITORING)

    # Initialize nodes
    context_node = ContextLoaderNode(knowledge_store)
    twin_node = TwinSimulationNode(auto_skip_on_low_effect=auto_skip_on_low_effect)
    design_node = DesignReasoningNode()
    power_node = PowerAnalysisNode()
    validity_node = ValidityAuditNode()
    redesign_node = RedesignNode()
    template_node = TemplateGeneratorNode()

    # Initialize graph
    workflow = StateGraph(ExperimentDesignState)

    # Add nodes to graph (wrapped for sync compatibility)
    workflow.add_node("audit_init", audit_initializer)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("context_loader", wrap_async_node(context_node.execute))  # type: ignore[type-var,arg-type,call-overload]
    if enable_twin_simulation:
        # H8: wire the twin pre-screen node only when enabled at graph-build time.
        # The node ALSO self-guards at runtime on state["enable_twin_simulation"]
        # (default False), so even when wired it stays dark unless the caller opts
        # in. This flag controls whether the node is part of the graph AT ALL.
        workflow.add_node("twin_simulation", wrap_async_node(twin_node.execute))  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("design_reasoning", wrap_async_node(design_node.execute))  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("power_analysis", wrap_async_node(power_node.execute))  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("validity_audit", wrap_async_node(validity_node.execute))  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("redesign", wrap_async_node(redesign_node.execute))  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("template_generator", wrap_async_node(template_node.execute))  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("error_handler", error_handler_node)  # type: ignore[call-overload]

    # Set entry point - start with audit initialization
    workflow.set_entry_point("audit_init")

    # Define edges
    # audit_init → context_loader
    workflow.add_edge("audit_init", "context_loader")

    if enable_twin_simulation:
        # context_loader → twin_simulation (when the node is wired)
        workflow.add_edge("context_loader", "twin_simulation")

        # twin_simulation → design_reasoning, error_handler, or END (if skipped)
        def twin_to_next(state: ExperimentDesignState) -> str:
            """Determine next step after twin simulation.

            Returns:
                - "design_reasoning" if proceeding with experiment design
                - "end" if twin recommends skip (early exit)
                - "error_handler" if workflow failed
            """
            if state.get("status") == "failed":
                return "error_handler"
            if state.get("status") == "skipped":
                return "end"
            return "design_reasoning"

        workflow.add_conditional_edges(
            "twin_simulation",
            twin_to_next,
            {
                "design_reasoning": "design_reasoning",
                "end": END,
                "error_handler": "error_handler",
            },
        )
    else:
        # Twin pre-screen excised at build time (#705 H8) — go straight to design.
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


# Issue #215: the eager module-level singleton
# ``experiment_designer_graph = create_experiment_designer_graph()`` has been
# REMOVED. Zero internal callers reference it (verified via repo-wide grep
# 2026-05-14). External callers, if any, should call
# ``create_experiment_designer_graph()`` directly. The eager evaluation
# triggered wrap_async_node's nest_asyncio.apply() at module import (see
# wrap_async_node docstring above for the asyncio pollution chain that
# produced the xdist flake described in issue #215).


def create_initial_state(
    business_question: str,
    constraints: dict[str, Any] | None = None,
    available_data: dict[str, Any] | None = None,
    preregistration_formality: str = "medium",
    max_redesign_iterations: int = 2,
    enable_validity_audit: bool = True,
    # Phase 15: Twin simulation parameters. Default OFF to match
    # ExperimentDesignerInput (the v1 synthetic DGP is not decision-useful for an
    # auto pre-screen) — opt in explicitly (#705 H8).
    enable_twin_simulation: bool = False,
    intervention_type: str | None = None,
    brand: str | None = None,
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
        enable_twin_simulation: Whether to run twin simulation pre-screening (Phase 15)
        intervention_type: Type of intervention for twin simulation
        brand: Pharmaceutical brand for twin simulation

    Returns:
        Initialized ExperimentDesignState
    """
    state = ExperimentDesignState(
        # Input fields
        business_question=business_question,
        constraints=constraints or {},
        available_data=available_data or {},
        preregistration_formality=preregistration_formality,  # type: ignore
        max_redesign_iterations=max_redesign_iterations,
        enable_validity_audit=enable_validity_audit,
        # Phase 15: Twin simulation
        enable_twin_simulation=enable_twin_simulation,
        # Error handling (initialized)
        errors=[],
        warnings=[],
        status="pending",
    )

    # Add optional twin simulation fields if provided
    if intervention_type:
        state["intervention_type"] = intervention_type
    if brand:
        state["brand"] = brand

    return state
