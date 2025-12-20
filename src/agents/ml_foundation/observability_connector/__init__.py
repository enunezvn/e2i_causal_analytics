"""Observability Connector Agent - Cross-cutting telemetry and monitoring.

This agent wraps ALL agent operations with observability spans for:
- Distributed tracing
- Latency tracking
- Error rate monitoring
- Token usage tracking (for Hybrid/Deep agents)
- Quality metrics aggregation

Unlike other agents, this is primarily used via helper methods rather than
being invoked in the main pipeline.
"""

from .agent import ObservabilityConnectorAgent, Span
from .state import ObservabilityConnectorState

__all__ = [
    "ObservabilityConnectorAgent",
    "Span",
    "ObservabilityConnectorState",
]
