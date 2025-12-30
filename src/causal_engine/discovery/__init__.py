"""
E2I Causal Analytics - Causal Discovery Module
==============================================

Causal structure learning with multi-algorithm ensemble.

This module provides automatic DAG discovery from observational data
using established algorithms from the causal-learn library.

Components:
- DiscoveryRunner: Orchestrates multi-algorithm ensemble discovery
- DiscoveryGate: Evaluates discovery results for acceptance
- DriverRanker: Compares causal vs predictive feature importance
- Algorithm wrappers: GES, PC (FCI, LiNGAM coming soon)

Example:
    >>> from src.causal_engine.discovery import (
    ...     DiscoveryRunner,
    ...     DiscoveryConfig,
    ...     DiscoveryAlgorithmType,
    ...     DiscoveryGate,
    ...     GateDecision,
    ... )
    >>>
    >>> # Run discovery
    >>> runner = DiscoveryRunner()
    >>> config = DiscoveryConfig(
    ...     algorithms=[DiscoveryAlgorithmType.GES, DiscoveryAlgorithmType.PC],
    ...     ensemble_threshold=0.5,
    ... )
    >>> result = await runner.discover_dag(data, config)
    >>>
    >>> # Evaluate result
    >>> gate = DiscoveryGate()
    >>> evaluation = gate.evaluate(result)
    >>> if evaluation.decision == GateDecision.ACCEPT:
    ...     print(f"Discovered DAG with {result.n_edges} edges")

Author: E2I Causal Analytics Team
"""

from .base import (
    AlgorithmResult,
    BaseDiscoveryAlgorithm,
    DiscoveredEdge,
    DiscoveryAlgorithm,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    DiscoveryResult,
    EdgeType,
    GateDecision,
)
from .driver_ranker import (
    DriverRanker,
    DriverRankingResult,
    FeatureRanking,
    ImportanceType,
)
from .gate import (
    DiscoveryGate,
    GateConfig,
    GateEvaluation,
)
from .runner import DiscoveryRunner

__all__ = [
    # Core classes
    "DiscoveryRunner",
    "DiscoveryConfig",
    "DiscoveryResult",
    "DiscoveryGate",
    "GateConfig",
    "GateEvaluation",
    "DriverRanker",
    "DriverRankingResult",
    "FeatureRanking",
    # Enums
    "DiscoveryAlgorithmType",
    "GateDecision",
    "EdgeType",
    "ImportanceType",
    # Types
    "DiscoveredEdge",
    "AlgorithmResult",
    # Protocols
    "DiscoveryAlgorithm",
    "BaseDiscoveryAlgorithm",
]
