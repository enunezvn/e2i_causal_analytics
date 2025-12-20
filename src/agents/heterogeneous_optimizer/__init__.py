"""Heterogeneous Optimizer Agent.

Tier 2 (Causal Analytics) agent for segment-level CATE analysis and
optimal treatment allocation.
"""

from .agent import HeterogeneousOptimizerAgent
from .state import (
    HeterogeneousOptimizerState,
    CATEResult,
    SegmentProfile,
    PolicyRecommendation,
)
from .graph import create_heterogeneous_optimizer_graph

__all__ = [
    "HeterogeneousOptimizerAgent",
    "HeterogeneousOptimizerState",
    "CATEResult",
    "SegmentProfile",
    "PolicyRecommendation",
    "create_heterogeneous_optimizer_graph",
]
