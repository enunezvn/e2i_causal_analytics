"""LangGraph workflow for observability_connector agent.

Observability pipeline:
  Node 1: Span Emission
    ↓
  Node 2: Metrics Aggregation

Note: This agent is primarily used via helper methods (span(), track_llm_call())
rather than being invoked in the main pipeline. The graph workflow is for
collecting and returning metrics when explicitly requested.
"""

from langgraph.graph import StateGraph, END
from .state import ObservabilityConnectorState
from .nodes import emit_spans, aggregate_metrics


def create_observability_connector_graph() -> StateGraph:
    """Create observability_connector LangGraph workflow.

    Pipeline:
        START
          ↓
        emit_spans (if events_to_log present)
          ↓
        aggregate_metrics (compute quality metrics)
          ↓
        END

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(ObservabilityConnectorState)

    # Add nodes
    workflow.add_node("emit_spans", emit_spans)
    workflow.add_node("aggregate_metrics", aggregate_metrics)

    # Define edges
    workflow.set_entry_point("emit_spans")

    # Emission → Metrics (always)
    workflow.add_edge("emit_spans", "aggregate_metrics")

    # Metrics → End
    workflow.add_edge("aggregate_metrics", END)

    return workflow.compile()
