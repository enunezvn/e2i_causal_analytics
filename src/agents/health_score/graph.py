"""
E2I Health Score Agent - LangGraph Assembly
Version: 4.2
Purpose: Build the Health Score agent graph
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from .state import HealthScoreState
from .nodes.component_health import ComponentHealthNode
from .nodes.model_health import ModelHealthNode
from .nodes.pipeline_health import PipelineHealthNode
from .nodes.agent_health import AgentHealthNode
from .nodes.score_composer import ScoreComposerNode

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
        [component] → [model] → [pipeline] → [agent] → [compose] → END

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

    # Initialize nodes
    component = ComponentHealthNode(health_client)
    model = ModelHealthNode(metrics_store)
    pipeline = PipelineHealthNode(pipeline_store)
    agent = AgentHealthNode(agent_registry)
    composer = ScoreComposerNode()

    # Build graph
    workflow = StateGraph(HealthScoreState)

    # Add nodes
    workflow.add_node("component", component.execute)
    workflow.add_node("model", model.execute)
    workflow.add_node("pipeline", pipeline.execute)
    workflow.add_node("agent", agent.execute)
    workflow.add_node("compose", composer.execute)

    # Sequential flow for simplicity and predictability
    workflow.set_entry_point("component")
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
        [component] → [compose] → END

    Only checks component health for fast dashboard updates.

    Args:
        health_client: Client for component health checks

    Returns:
        Compiled LangGraph workflow
    """
    logger.info("Building Quick Check graph")

    # Initialize nodes
    component = ComponentHealthNode(health_client)
    composer = ScoreComposerNode()

    # Build graph
    workflow = StateGraph(HealthScoreState)

    # Add nodes
    workflow.add_node("component", component.execute)
    workflow.add_node("compose", composer.execute)

    # Minimal flow
    workflow.set_entry_point("component")
    workflow.add_edge("component", "compose")
    workflow.add_edge("compose", END)

    compiled = workflow.compile()
    logger.info("Quick Check graph compiled successfully")

    return compiled
