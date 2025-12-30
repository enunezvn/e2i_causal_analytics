"""
E2I Causal Analytics - Discovery Algorithm Wrappers
===================================================

Wrappers for causal-learn structure learning algorithms.

Algorithms:
- GES: Greedy Equivalence Search (score-based)
- PC: Peter-Clark (constraint-based)
- FCI: Fast Causal Inference (handles latent confounders)
- LiNGAM: Linear Non-Gaussian Acyclic Model

Author: E2I Causal Analytics Team
"""

from .ges_wrapper import GESAlgorithm
from .pc_wrapper import PCAlgorithm

__all__ = [
    "GESAlgorithm",
    "PCAlgorithm",
]
