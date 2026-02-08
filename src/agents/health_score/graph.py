"""
E2I Health Score Agent - LangGraph Assembly
Version: 4.2
Purpose: Build the Health Score agent graph

Observability:
- Audit chain recording for tamper-evident logging
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from src.agents.base.audit_chain_mixin import create_workflow_initializer
from src.utils.audit_chain import AgentTier

from .nodes.agent_health import AgentHealthNode
from .nodes.component_health import ComponentHealthNode
from .nodes.model_health import ModelHealthNode
from .nodes.pipeline_health import PipelineHealthNode
from .nodes.score_composer import ScoreComposerNode
from .state import HealthScoreState

logger = logging.getLogger(__name__)


def build_health_score_graph(
    health_client: Optional[Any] = None,
    metrics_store: Optional[Any] = None,
    pipeline_store: Optional[Any] = None,
    agent_registry: Optional[Any] = None,
) -> Any:
    """
    Build the Health Score agent graph.

    Architecture:
        [audit_init] → [component] → [model] → [pipeline] → [agent] → [compose] → END

    All health checks run sequentially to maintain deterministic execution,
    but each node performs internal parallelism for its checks.

    Args:
        health_client: Client for component health checks
        metrics_store: Store for model metrics
        pipeline_store: Store for pipeline status
        agent_registry: Registry of system agents

    Returns:
        Compiled LangGraph workflow
    """
    logger.info("Building Health Score agent graph")

    # Create audit workflow initializer
    audit_initializer = create_workflow_initializer("health_score", AgentTier.MONITORING)

    # Initialize nodes
    component = ComponentHealthNode(health_client)
    model = ModelHealthNode(metrics_store)
    pipeline = PipelineHealthNode(pipeline_store)
    agent = AgentHealthNode(agent_registry)
    composer = ScoreComposerNode()

    # Build graph
    workflow = StateGraph(HealthScoreState)

    # Add nodes
    workflow.add_node("audit_init", audit_initializer)  # type: ignore[type-var,arg-type,call-overload]  # Initialize audit chain
    workflow.add_node("component", component.execute)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("model", model.execute)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("pipeline", pipeline.execute)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("agent", agent.execute)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("compose", composer.execute)  # type: ignore[type-var,arg-type,call-overload]

    # Sequential flow starting with audit initialization
    workflow.set_entry_point("audit_init")
    workflow.add_edge("audit_init", "component")
    workflow.add_edge("component", "model")
    workflow.add_edge("model", "pipeline")
    workflow.add_edge("pipeline", "agent")
    workflow.add_edge("agent", "compose")
    workflow.add_edge("compose", END)

    compiled = workflow.compile()
    logger.info("Health Score agent graph compiled successfully")

    return compiled


def build_quick_check_graph(
    health_client: Optional[Any] = None,
) -> Any:
    """
    Build a minimal quick-check graph.

    Architecture:
        [audit_init] → [component] → [compose] → END

    Only checks component health for fast dashboard updates.

    Args:
        health_client: Client for component health checks

    Returns:
        Compiled LangGraph workflow
    """
    logger.info("Building Quick Check graph")

    # Create audit workflow initializer
    audit_initializer = create_workflow_initializer("health_score_quick", AgentTier.MONITORING)

    # Initialize nodes
    component = ComponentHealthNode(health_client)
    composer = ScoreComposerNode()

    # Build graph
    workflow = StateGraph(HealthScoreState)

    # Add nodes
    workflow.add_node("audit_init", audit_initializer)  # type: ignore[type-var,arg-type,call-overload]  # Initialize audit chain
    workflow.add_node("component", component.execute)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("compose", composer.execute)  # type: ignore[type-var,arg-type,call-overload]

    # Minimal flow starting with audit initialization
    workflow.set_entry_point("audit_init")
    workflow.add_edge("audit_init", "component")
    workflow.add_edge("component", "compose")
    workflow.add_edge("compose", END)

    compiled = workflow.compile()
    logger.info("Quick Check graph compiled successfully")

    return compiled
