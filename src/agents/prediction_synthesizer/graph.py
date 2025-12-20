"""
E2I Prediction Synthesizer Agent - LangGraph Assembly
Version: 4.2
Purpose: Build the prediction synthesizer workflow graph
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from .state import PredictionSynthesizerState
from .nodes.model_orchestrator import ModelOrchestratorNode
from .nodes.ensemble_combiner import EnsembleCombinerNode
from .nodes.context_enricher import ContextEnricherNode

logger = logging.getLogger(__name__)


def build_prediction_synthesizer_graph(
    model_registry: Optional[Any] = None,
    model_clients: Optional[Dict[str, Any]] = None,
    context_store: Optional[Any] = None,
    feature_store: Optional[Any] = None,
) -> StateGraph:
    """
    Build the Prediction Synthesizer agent graph.

    Architecture:
        [orchestrate] → [combine] → [enrich] → END
                 ↓           ↓
             [error]    [error]

    Args:
        model_registry: Registry of available models
        model_clients: Dict mapping model_id to prediction client
        context_store: Store for historical context
        feature_store: Store for feature metadata

    Returns:
        Compiled LangGraph workflow
    """
    # Initialize nodes
    orchestrator = ModelOrchestratorNode(
        model_registry=model_registry,
        model_clients=model_clients,
    )
    combiner = EnsembleCombinerNode()
    enricher = ContextEnricherNode(
        context_store=context_store,
        feature_store=feature_store,
    )

    # Build graph
    workflow = StateGraph(PredictionSynthesizerState)

    # Add nodes
    workflow.add_node("orchestrate", orchestrator.execute)
    workflow.add_node("combine", combiner.execute)
    workflow.add_node("enrich", enricher.execute)
    workflow.add_node("error_handler", _error_handler_node)

    # Entry point
    workflow.set_entry_point("orchestrate")

    # Conditional edges from orchestrator
    workflow.add_conditional_edges(
        "orchestrate",
        lambda s: "error" if s.get("status") == "failed" else "combine",
        {"combine": "combine", "error": "error_handler"},
    )

    # Conditional edges from combiner
    workflow.add_conditional_edges(
        "combine",
        lambda s: "error" if s.get("status") == "failed" else "enrich",
        {"enrich": "enrich", "error": "error_handler"},
    )

    # Terminal edges
    workflow.add_edge("enrich", END)
    workflow.add_edge("error_handler", END)

    logger.debug("Prediction synthesizer graph built successfully")

    return workflow.compile()


async def _error_handler_node(
    state: PredictionSynthesizerState,
) -> PredictionSynthesizerState:
    """Handle errors in the prediction pipeline."""
    errors = state.get("errors", [])
    error_messages = [e.get("error", "Unknown error") for e in errors]

    logger.error(f"Prediction synthesis failed: {error_messages}")

    return {
        **state,
        "prediction_summary": "Prediction could not be generated due to errors.",
        "total_latency_ms": (
            state.get("orchestration_latency_ms", 0)
            + state.get("ensemble_latency_ms", 0)
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
    }


def build_simple_prediction_graph(
    model_clients: Optional[Dict[str, Any]] = None,
) -> StateGraph:
    """
    Build a simplified prediction graph without context enrichment.

    Architecture:
        [orchestrate] → [combine] → END

    Args:
        model_clients: Dict mapping model_id to prediction client

    Returns:
        Compiled LangGraph workflow
    """
    orchestrator = ModelOrchestratorNode(model_clients=model_clients)
    combiner = EnsembleCombinerNode()

    workflow = StateGraph(PredictionSynthesizerState)

    workflow.add_node("orchestrate", orchestrator.execute)
    workflow.add_node("combine", combiner.execute)
    workflow.add_node("error_handler", _error_handler_node)

    workflow.set_entry_point("orchestrate")

    workflow.add_conditional_edges(
        "orchestrate",
        lambda s: "error" if s.get("status") == "failed" else "combine",
        {"combine": "combine", "error": "error_handler"},
    )

    workflow.add_edge("combine", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile()
