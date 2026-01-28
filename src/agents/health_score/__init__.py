"""
E2I Health Score Agent - Tier 3 Monitoring Agent
Version: 4.2
Purpose: System health metrics and monitoring

This is a Fast Path Agent optimized for:
- Quick health checks (<5s full scope, <1s quick check)
- System status aggregation
- Dashboard metrics
- Zero LLM usage in critical path

Integration:
- Memory: Working Memory (Redis) for caching health results
- Observability: MLflow tracking for health metrics trending
- Data: Component, model, pipeline, and agent health monitoring
"""

from .agent import HealthScoreAgent, HealthScoreInput, HealthScoreOutput
from .graph import build_health_score_graph
from .health_client import (
    SupabaseHealthClient,
    SimpleHealthClient,
    get_health_client_for_testing,
)
from .mlflow_tracker import (
    HealthScoreMLflowTracker,
    HealthScoreMetrics,
    HealthScoreContext,
    create_tracker as create_mlflow_tracker,
)
from .opik_tracer import (
    HealthScoreOpikTracer,
    HealthCheckTraceContext,
    NodeSpanContext,
    get_health_score_tracer,
)
from .state import (
    AgentStatus,
    ComponentStatus,
    HealthScoreState,
    ModelMetrics,
    PipelineStatus,
)

__all__ = [
    "HealthScoreAgent",
    "HealthScoreInput",
    "HealthScoreOutput",
    "HealthScoreState",
    "ComponentStatus",
    "ModelMetrics",
    "PipelineStatus",
    "AgentStatus",
    "build_health_score_graph",
    # Health clients
    "SupabaseHealthClient",
    "SimpleHealthClient",
    "get_health_client_for_testing",
    # MLflow tracking
    "HealthScoreMLflowTracker",
    "HealthScoreMetrics",
    "HealthScoreContext",
    "create_mlflow_tracker",
    # Opik tracing
    "HealthScoreOpikTracer",
    "HealthCheckTraceContext",
    "NodeSpanContext",
    "get_health_score_tracer",
]
