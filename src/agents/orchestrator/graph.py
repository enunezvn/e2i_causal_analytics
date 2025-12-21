"""LangGraph workflow for orchestrator agent.

Linear flow optimized for speed:
    [classify] → [rag_context] → [route] → [dispatch] → [synthesize] → END

Total latency target: <2 seconds for orchestration overhead
(excluding agent execution time)
"""

from typing import Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import OrchestratorState
from .nodes import (
    classify_intent,
    retrieve_rag_context,
    route_to_agents,
    dispatch_to_agents,
    synthesize_response,
)


def create_orchestrator_graph(
    agent_registry: Optional[Dict[str, Any]] = None,
    enable_checkpointing: bool = False,
    enable_rag: bool = True,
) -> StateGraph:
    """Build the Orchestrator agent graph.

    Architecture (with RAG enabled):
        [classify] → [rag_context] → [route] → [dispatch] → [synthesize] → END

    Architecture (with RAG disabled):
        [classify] → [route] → [dispatch] → [synthesize] → END

    Total latency target: <2 seconds for classification + routing
    (Agent execution time is additional)

    Args:
        agent_registry: Optional dict mapping agent_name to agent instance
        enable_checkpointing: Whether to enable graph checkpointing
        enable_rag: Whether to enable RAG context retrieval (default: True)

    Returns:
        Compiled StateGraph
    """
    # Build graph
    workflow = StateGraph(OrchestratorState)

    # Add nodes
    workflow.add_node("classify", classify_intent)

    # Conditionally add RAG node
    if enable_rag:
        workflow.add_node("rag_context", retrieve_rag_context)

    workflow.add_node("route", route_to_agents)

    # Dispatcher node with agent registry
    if agent_registry:
        from .nodes import DispatcherNode

        dispatcher = DispatcherNode(agent_registry)

        async def dispatch_with_registry(state):
            return await dispatcher.execute(state)

        workflow.add_node("dispatch", dispatch_with_registry)
    else:
        workflow.add_node("dispatch", dispatch_to_agents)

    workflow.add_node("synthesize", synthesize_response)

    # Linear flow (no conditionals for speed)
    workflow.set_entry_point("classify")

    if enable_rag:
        workflow.add_edge("classify", "rag_context")
        workflow.add_edge("rag_context", "route")
    else:
        workflow.add_edge("classify", "route")

    workflow.add_edge("route", "dispatch")
    workflow.add_edge("dispatch", "synthesize")
    workflow.add_edge("synthesize", END)

    # Compile
    if enable_checkpointing:
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    return workflow.compile()
