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
from .models import (
    AgentNameEnum,
    AgentTierEnum,
    LatencyStats,
    ObservabilitySpan,
    QualityMetrics,
    SpanEvent,
    SpanStatusEnum,
    TokenUsage,
    create_llm_span,
    create_span,
)
from .state import ObservabilityConnectorState

__all__ = [
    # Agent
    "ObservabilityConnectorAgent",
    "Span",
    "ObservabilityConnectorState",
    # Models
    "ObservabilitySpan",
    "SpanEvent",
    "TokenUsage",
    "LatencyStats",
    "QualityMetrics",
    # Enums
    "AgentNameEnum",
    "AgentTierEnum",
    "SpanStatusEnum",
    # Factory functions
    "create_span",
    "create_llm_span",
]
