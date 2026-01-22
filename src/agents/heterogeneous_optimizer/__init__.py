"""Heterogeneous Optimizer Agent.

Tier 2 (Causal Analytics) agent for segment-level CATE analysis and
optimal treatment allocation.
"""

from .agent import HeterogeneousOptimizerAgent
from .graph import create_heterogeneous_optimizer_graph
from .mlflow_tracker import (
    HeterogeneousOptimizerMLflowTracker,
    HeterogeneousOptimizerMetrics,
    HeterogeneousOptimizerContext,
    create_tracker as create_mlflow_tracker,
)
from .opik_tracer import (
    HeterogeneousOptimizerOpikTracer,
    CATEAnalysisTraceContext,
    NodeSpanContext,
    get_heterogeneous_optimizer_tracer,
)
from .state import (
    CATEResult,
    HeterogeneousOptimizerState,
    PolicyRecommendation,
    SegmentProfile,
)

__all__ = [
    "HeterogeneousOptimizerAgent",
    "HeterogeneousOptimizerState",
    "CATEResult",
    "SegmentProfile",
    "PolicyRecommendation",
    "create_heterogeneous_optimizer_graph",
    # MLflow tracking
    "HeterogeneousOptimizerMLflowTracker",
    "HeterogeneousOptimizerMetrics",
    "HeterogeneousOptimizerContext",
    "create_mlflow_tracker",
    # Opik tracing
    "HeterogeneousOptimizerOpikTracer",
    "CATEAnalysisTraceContext",
    "NodeSpanContext",
    "get_heterogeneous_optimizer_tracer",
]
