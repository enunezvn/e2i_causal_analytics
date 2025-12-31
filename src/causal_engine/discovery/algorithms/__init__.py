"""
E2I Causal Analytics - Discovery Algorithm Wrappers
===================================================

Wrappers for causal-learn structure learning algorithms.

Algorithms:
- GES: Greedy Equivalence Search (score-based)
- PC: Peter-Clark (constraint-based)
- FCI: Fast Causal Inference (handles latent confounders)
- DirectLiNGAM: Linear Non-Gaussian Acyclic Model (regression-based)
- ICA-LiNGAM: Linear Non-Gaussian Acyclic Model (ICA-based)

Author: E2I Causal Analytics Team
"""

from .fci_wrapper import FCIAlgorithm
from .ges_wrapper import GESAlgorithm
from .lingam_wrapper import DirectLiNGAMAlgorithm, ICALiNGAMAlgorithm
from .pc_wrapper import PCAlgorithm

__all__ = [
    "DirectLiNGAMAlgorithm",
    "FCIAlgorithm",
    "GESAlgorithm",
    "ICALiNGAMAlgorithm",
    "PCAlgorithm",
]
