"""
E2I Ground Truth Causal Effects Module

Stores and retrieves known causal effects for pipeline validation.
"""

from .causal_effects import (
    GroundTruthEffect,
    GroundTruthStore,
    get_ground_truth,
    validate_estimate,
)

__all__ = [
    "GroundTruthEffect",
    "GroundTruthStore",
    "get_ground_truth",
    "validate_estimate",
]
