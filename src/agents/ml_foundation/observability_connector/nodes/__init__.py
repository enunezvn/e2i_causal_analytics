"""Nodes for observability_connector agent.

Observability workflow with 2 main nodes:
1. Span Emission - Emit spans to Opik and persist to database
2. Metrics Aggregation - Compute quality metrics from spans

Plus context management helpers:
- Create Context - Generate new trace/span context
- Extract Context - Extract context from headers
- Inject Context - Inject context into headers
"""

from .span_emitter import emit_spans
from .metrics_aggregator import aggregate_metrics
from .context_manager import create_context, extract_context, inject_context

__all__ = [
    # Main workflow nodes
    "emit_spans",
    "aggregate_metrics",
    # Context management helpers
    "create_context",
    "extract_context",
    "inject_context",
]
