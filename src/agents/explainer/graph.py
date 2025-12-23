"""
E2I Explainer Agent - Graph Assembly
Version: 4.2
Purpose: LangGraph workflow for natural language explanations

Memory Integration:
- LangGraph RedisSaver checkpointer for workflow state persistence
- Enables pause/resume and conversation threading
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from .nodes import ContextAssemblerNode, DeepReasonerNode, NarrativeGeneratorNode
from .state import ExplainerState

logger = logging.getLogger(__name__)


def _get_default_checkpointer() -> Optional[BaseCheckpointSaver]:
    """Get default LangGraph checkpointer with graceful fallback.

    Returns:
        RedisSaver checkpointer or None if unavailable
    """
    try:
        from src.memory.working_memory import get_langgraph_checkpointer

        checkpointer = get_langgraph_checkpointer()
        logger.debug("LangGraph checkpointer initialized")
        return checkpointer
    except Exception as e:
        logger.warning(f"Failed to initialize checkpointer (non-fatal): {e}")
        return None


async def error_handler_node(state: ExplainerState) -> ExplainerState:
    """Handle errors and finalize failed state."""
    return {
        **state,
        "executive_summary": "Unable to generate explanation due to errors.",
        "detailed_explanation": "Please review the errors and try again.",
        "narrative_sections": [],
        "status": "failed",
    }


def route_after_assembly(state: ExplainerState) -> str:
    """Route after context assembly."""
    if state.get("status") == "failed":
        return "error"
    return "reason"


def route_after_reasoning(state: ExplainerState) -> str:
    """Route after deep reasoning."""
    if state.get("status") == "failed":
        return "error"
    return "generate"


def build_explainer_graph(
    conversation_store: Optional[Any] = None,
    use_llm: bool = False,
    llm: Optional[Any] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    use_default_checkpointer: bool = True,
) -> Any:
    """
    Build the Explainer agent graph.

    Pipeline:
    assemble -> reason -> generate -> END

    Args:
        conversation_store: Optional store for conversation history
        use_llm: Whether to use LLM for reasoning/generation
        llm: Optional LLM instance to use
        checkpointer: Optional LangGraph checkpointer for state persistence
        use_default_checkpointer: Whether to use default Redis checkpointer

    Returns:
        Compiled LangGraph workflow
    """
    # Initialize nodes
    assembler = ContextAssemblerNode(conversation_store)
    reasoner = DeepReasonerNode(use_llm=use_llm, llm=llm)
    generator = NarrativeGeneratorNode(use_llm=use_llm, llm=llm)

    # Build graph
    workflow = StateGraph(ExplainerState)

    # Add nodes
    workflow.add_node("assemble", assembler.execute)
    workflow.add_node("reason", reasoner.execute)
    workflow.add_node("generate", generator.execute)
    workflow.add_node("error_handler", error_handler_node)

    # Set entry point
    workflow.set_entry_point("assemble")

    # Add edges
    workflow.add_conditional_edges(
        "assemble",
        route_after_assembly,
        {"reason": "reason", "error": "error_handler"},
    )

    workflow.add_conditional_edges(
        "reason",
        route_after_reasoning,
        {"generate": "generate", "error": "error_handler"},
    )

    workflow.add_edge("generate", END)
    workflow.add_edge("error_handler", END)

    # Get checkpointer - use provided, default, or None
    if checkpointer is None and use_default_checkpointer:
        checkpointer = _get_default_checkpointer()

    return workflow.compile(checkpointer=checkpointer)


def build_simple_explainer_graph(use_checkpointer: bool = True) -> Any:
    """
    Build a simplified Explainer graph without LLM.

    Pipeline:
    assemble -> reason -> generate -> END

    Args:
        use_checkpointer: Whether to use Redis checkpointer

    Returns:
        Compiled LangGraph workflow
    """
    return build_explainer_graph(
        use_llm=False,
        use_default_checkpointer=use_checkpointer,
    )


# =============================================================================
# MODULE-LEVEL GRAPH (with checkpointer)
# =============================================================================

# Pre-compiled graph for direct import
# Uses default checkpointer for state persistence
explainer_graph = build_simple_explainer_graph(use_checkpointer=True)
