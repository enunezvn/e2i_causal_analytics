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
from .batch_processor import (
    BatchConfig,
    BatchMetrics,
    BatchProcessor,
    get_batch_processor,
    reset_batch_processor,
)
from .cache import (
    CacheBackend,
    CacheConfig,
    CacheEntry,
    CacheMetrics,
    MetricsCache,
    get_metrics_cache,
    reset_metrics_cache,
)
from .config import (
    BatchingSettings,
    CacheSettings,
    CircuitBreakerSettings,
    ObservabilityConfig,
    OpikSettings,
    SamplingSettings,
    get_observability_config,
    reset_observability_config,
)
from .self_monitor import (
    AsyncLatencyContext,
    ComponentHealth,
    HealthStatus,
    LatencyContext,
    LatencyThresholds,
    MetricType,
    OverallHealth,
    SelfMonitor,
    SelfMonitorConfig,
    get_self_monitor,
    reset_self_monitor,
)
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
    # Batch Processor
    "BatchProcessor",
    "BatchConfig",
    "BatchMetrics",
    "get_batch_processor",
    "reset_batch_processor",
    # Metrics Cache
    "MetricsCache",
    "CacheConfig",
    "CacheMetrics",
    "CacheEntry",
    "CacheBackend",
    "get_metrics_cache",
    "reset_metrics_cache",
    # Configuration
    "ObservabilityConfig",
    "OpikSettings",
    "SamplingSettings",
    "BatchingSettings",
    "CircuitBreakerSettings",
    "CacheSettings",
    "get_observability_config",
    "reset_observability_config",
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
    # Self-Monitoring
    "SelfMonitor",
    "SelfMonitorConfig",
    "HealthStatus",
    "MetricType",
    "LatencyThresholds",
    "LatencyContext",
    "AsyncLatencyContext",
    "ComponentHealth",
    "OverallHealth",
    "get_self_monitor",
    "reset_self_monitor",
]
