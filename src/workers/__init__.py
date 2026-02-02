"""
Celery Workers Module
=====================

Multi-tier worker architecture for E2I Causal Analytics.

Worker Tiers:
- Light: Quick tasks (API calls, cache, notifications)
- Medium: Standard analytics (reports, aggregations)
- Heavy: Compute-intensive (SHAP, causal, ML, twins)

Observability (G12):
- Event consumer for task metrics
- Queue depth monitoring
- Worker health tracking
"""

from .celery_app import celery_app
from .event_consumer import (
    CeleryEventConsumer,
    CeleryMetrics,
    celery_metrics,
    extract_trace_context,
    inject_trace_context,
    traced_task,
)
from .monitoring import (
    AutoscalerMetricsProvider,
    AutoscalerRecommendation,
    CeleryQueueMonitor,
    QueueMetrics,
    get_monitoring_summary,
    get_queue_depths,
    get_queue_monitor,
    queue_metrics,
    register_monitoring_tasks,
)

__all__ = [
    # Celery app
    "celery_app",
    # Event consumer (G12)
    "CeleryEventConsumer",
    "CeleryMetrics",
    "celery_metrics",
    "inject_trace_context",
    "extract_trace_context",
    "traced_task",
    # Queue monitoring (G12/G19)
    "CeleryQueueMonitor",
    "QueueMetrics",
    "queue_metrics",
    "AutoscalerMetricsProvider",
    "AutoscalerRecommendation",
    "get_queue_monitor",
    "get_queue_depths",
    "get_monitoring_summary",
    "register_monitoring_tasks",
]
