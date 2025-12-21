"""Heterogeneous Optimizer Agent.

Tier 2 (Causal Analytics) agent for segment-level CATE analysis and
optimal treatment allocation.
"""

from .agent import HeterogeneousOptimizerAgent
from .graph import create_heterogeneous_optimizer_graph
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
]
