"""LangGraph workflow for observability_connector agent.

Observability pipeline:
  Node 1: Span Emission
    ↓
  Node 2: Metrics Aggregation

Note: This agent is primarily used via helper methods (span(), track_llm_call())
rather than being invoked in the main pipeline. The graph workflow is for
collecting and returning metrics when explicitly requested.
"""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import aggregate_metrics, emit_spans
from .state import ObservabilityConnectorState


def create_observability_connector_graph() -> CompiledStateGraph:
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
    workflow.add_node("emit_spans", emit_spans)  # type: ignore[type-var,arg-type,call-overload]
    workflow.add_node("aggregate_metrics", aggregate_metrics)  # type: ignore[type-var,arg-type,call-overload]

    # Define edges
    workflow.set_entry_point("emit_spans")

    # Emission → Metrics (always)
    workflow.add_edge("emit_spans", "aggregate_metrics")

    # Metrics → End
    workflow.add_edge("aggregate_metrics", END)

    return workflow.compile()
