"""
E2I Health Score Agent - Tier 3 Monitoring Agent
Version: 4.2
Purpose: System health metrics and monitoring

This is a Fast Path Agent optimized for:
- Quick health checks (<5s full scope, <1s quick check)
- System status aggregation
- Dashboard metrics
- Zero LLM usage in critical path
"""

from .agent import HealthScoreAgent
from .graph import build_health_score_graph
from .state import (
    AgentStatus,
    ComponentStatus,
    HealthScoreState,
    ModelMetrics,
    PipelineStatus,
)

__all__ = [
    "HealthScoreAgent",
    "HealthScoreState",
    "ComponentStatus",
    "ModelMetrics",
    "PipelineStatus",
    "AgentStatus",
    "build_health_score_graph",
]
