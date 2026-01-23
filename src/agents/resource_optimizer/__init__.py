"""
E2I Resource Optimizer Agent
Version: 4.2
Purpose: Resource allocation optimization for pharmaceutical operations
"""

from .agent import (
    ResourceOptimizerAgent,
    ResourceOptimizerInput,
    ResourceOptimizerOutput,
    optimize_allocation,
)
from .graph import (
    build_resource_optimizer_graph,
    build_simple_optimizer_graph,
)
from .mlflow_tracker import (
    ResourceOptimizerMLflowTracker,
    ResourceOptimizerMetrics,
    OptimizationContext,
    create_tracker as create_mlflow_tracker,
)
from .opik_tracer import (
    ResourceOptimizerOpikTracer,
    OptimizationTraceContext,
    NodeSpanContext,
    get_resource_optimizer_tracer,
)
from .state import (
    AllocationResult,
    AllocationTarget,
    Constraint,
    ResourceOptimizerState,
    ScenarioResult,
)

__all__ = [
    # Agent
    "ResourceOptimizerAgent",
    "ResourceOptimizerInput",
    "ResourceOptimizerOutput",
    "optimize_allocation",
    # Graph
    "build_resource_optimizer_graph",
    "build_simple_optimizer_graph",
    # MLflow Tracking
    "ResourceOptimizerMLflowTracker",
    "ResourceOptimizerMetrics",
    "OptimizationContext",
    "create_mlflow_tracker",
    # Opik tracing
    "ResourceOptimizerOpikTracer",
    "OptimizationTraceContext",
    "NodeSpanContext",
    "get_resource_optimizer_tracer",
    # State
    "AllocationResult",
    "AllocationTarget",
    "Constraint",
    "ResourceOptimizerState",
    "ScenarioResult",
]
