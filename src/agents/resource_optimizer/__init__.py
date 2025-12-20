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
    # State
    "AllocationResult",
    "AllocationTarget",
    "Constraint",
    "ResourceOptimizerState",
    "ScenarioResult",
]
