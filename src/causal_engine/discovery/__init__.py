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
- DiscoveryCache: Caches discovery results to avoid redundant computation
- Algorithm wrappers: GES, PC, FCI, DirectLiNGAM, ICA-LiNGAM

Example:
    >>> from src.causal_engine.discovery import (
    ...     DiscoveryRunner,
    ...     DiscoveryConfig,
    ...     DiscoveryAlgorithmType,
    ...     DiscoveryGate,
    ...     GateDecision,
    ...     DiscoveryCache,
    ... )
    >>>
    >>> # Run discovery with caching
    >>> cache = DiscoveryCache()
    >>> runner = DiscoveryRunner()
    >>> config = DiscoveryConfig(
    ...     algorithms=[DiscoveryAlgorithmType.GES, DiscoveryAlgorithmType.PC],
    ...     ensemble_threshold=0.5,
    ... )
    >>>
    >>> # Check cache first
    >>> result = await cache.get(data, config)
    >>> if result is None:
    ...     result = await runner.discover_dag(data, config)
    ...     await cache.set(data, config, result)
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
from .cache import (
    CacheConfig,
    CacheStats,
    DiscoveryCache,
    get_discovery_cache,
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
from .hasher import (
    hash_config,
    hash_dataframe,
    hash_discovery_request,
    make_cache_key,
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
    # Cache
    "DiscoveryCache",
    "CacheConfig",
    "CacheStats",
    "get_discovery_cache",
    # Hashing
    "hash_dataframe",
    "hash_config",
    "hash_discovery_request",
    "make_cache_key",
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
