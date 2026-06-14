"""LangGraph workflow for orchestrator agent.

Linear flow optimized for speed:
    [audit_init] → [classify] → [rag_context] → [route] → [dispatch] → [synthesize] → END

Total latency target: <2 seconds for orchestration overhead
(excluding agent execution time)

Observability:
- Audit chain recording for tamper-evident logging
"""

from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.base.audit_chain_mixin import (
    add_audited_node,
    create_workflow_initializer,
)
from src.utils.audit_chain import AgentTier

from .nodes import (
    DispatcherNode,
    classify_intent,
    retrieve_rag_context,
    route_to_agents,
    synthesize_response,
)
from .state import OrchestratorState


def create_orchestrator_graph(
    agent_registry: Optional[Dict[str, Any]] = None,
    enable_checkpointing: bool = False,
    enable_rag: bool = True,
    allow_mock: bool = False,
) -> CompiledStateGraph:
    """Build the Orchestrator agent graph.

    Architecture (with RAG enabled):
        [audit_init] → [classify] → [rag_context] → [route] → [dispatch] → [synthesize] → END

    Architecture (with RAG disabled):
        [audit_init] → [classify] → [route] → [dispatch] → [synthesize] → END

    Total latency target: <2 seconds for classification + routing
    (Agent execution time is additional)

    Args:
        agent_registry: Optional dict mapping agent_name to agent instance
        enable_checkpointing: Whether to enable graph checkpointing
        enable_rag: Whether to enable RAG context retrieval (default: True)
        allow_mock: TEST-ONLY. Forwarded to the DispatcherNode — when True a
            dispatch to an agent absent from the registry returns the canned mock
            scaffold; default False makes a missing/partial registry FAIL CLOSED
            (no fabricated values, #814). Production never sets this.

    Returns:
        Compiled StateGraph
    """
    # Create audit workflow initializer
    audit_initializer = create_workflow_initializer("orchestrator", AgentTier.COORDINATION)

    # Build graph
    workflow = StateGraph(OrchestratorState)

    # Add audit init node
    workflow.add_node("audit_init", audit_initializer)  # type: ignore[type-var,arg-type,call-overload]

    # Add nodes (wrapped so each emits a real timed audit entry -> latency telemetry)
    add_audited_node(
        workflow,
        "classify",
        classify_intent,
        agent_name="orchestrator",
        agent_tier=AgentTier.COORDINATION,
    )

    # Conditionally add RAG node
    if enable_rag:
        add_audited_node(
            workflow,
            "rag_context",
            retrieve_rag_context,
            agent_name="orchestrator",
            agent_tier=AgentTier.COORDINATION,
        )

    add_audited_node(
        workflow,
        "route",
        route_to_agents,
        agent_name="orchestrator",
        agent_tier=AgentTier.COORDINATION,
    )

    # Dispatcher node. Always a DispatcherNode so the test-only mock scaffold is
    # gated by allow_mock: with a (possibly partial) registry and allow_mock=False
    # a missing agent FAILS CLOSED rather than fabricating a result (#814). The
    # registry-less branch (agent_registry falsy) is non-production; tests opt into
    # the canned scaffold via allow_mock=True.
    dispatcher = DispatcherNode(agent_registry, allow_mock=allow_mock)

    async def dispatch_node(state):
        return await dispatcher.execute(state)

    add_audited_node(
        workflow,
        "dispatch",
        dispatch_node,
        agent_name="orchestrator",
        agent_tier=AgentTier.COORDINATION,
    )

    add_audited_node(
        workflow,
        "synthesize",
        synthesize_response,
        agent_name="orchestrator",
        agent_tier=AgentTier.COORDINATION,
    )

    # Linear flow (no conditionals for speed) - start with audit_init
    workflow.set_entry_point("audit_init")

    # Edge from audit_init to classify
    workflow.add_edge("audit_init", "classify")

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
