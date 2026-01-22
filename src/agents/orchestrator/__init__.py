"""Orchestrator Agent - Tier 1 Coordination.

The orchestrator is the entry point for all queries. It routes to specialized
agents and synthesizes their responses.

Key Features:
- Fast intent classification (<500ms)
- Pattern-based + LLM fallback classification
- Parallel agent dispatch
- Multi-agent response synthesis
- <2s orchestration overhead target

Usage:
    orchestrator = OrchestratorAgent()
    result = await orchestrator.run({
        "query": "What is driving conversion rate changes?",
        "user_id": "user_123",
        "user_context": {"expertise": "analyst"}
    })

Author: E2I Causal Analytics Team
Version: 4.1.0
"""

from .agent import OrchestratorAgent
from .graph import create_orchestrator_graph
from .opik_tracer import (
    OrchestratorOpikTracer,
    OrchestrationTraceContext,
    NodeSpanContext,
    get_orchestrator_tracer,
)
from .state import (
    AgentDispatch,
    AgentResult,
    Citation,
    IntentClassification,
    OrchestratorState,
    ParsedEntity,
    ParsedQuery,
)

__version__ = "4.1.0"

__all__ = [
    "OrchestratorAgent",
    "OrchestratorState",
    "IntentClassification",
    "AgentDispatch",
    "AgentResult",
    "ParsedQuery",
    "ParsedEntity",
    "Citation",
    "create_orchestrator_graph",
    # Opik tracing
    "OrchestratorOpikTracer",
    "OrchestrationTraceContext",
    "NodeSpanContext",
    "get_orchestrator_tracer",
]
