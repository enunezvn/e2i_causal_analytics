"""
Digital Twin Module
===================

Digital twin system for pre-screening A/B experiments using synthetic
populations. Provides twin generation, intervention simulation,
fidelity tracking, result caching, and automatic retraining.

Components:
    - TwinGenerator: Generate synthetic populations from historical data
    - SimulationEngine: Simulate intervention effects on twin populations
    - FidelityTracker: Track prediction accuracy against real outcomes
    - SimulationCache: Cache frequently simulated interventions
    - TwinRetrainingService: Automatic model retraining on fidelity degradation

Version: 1.1.0 (Phase 5+6 improvements)
"""

from .fidelity_tracker import FidelityTracker
from .models.simulation_models import (
    EffectHeterogeneity,
    FidelityGrade,
    FidelityRecord,
    InterventionConfig,
    PopulationFilter,
    SimulationRecommendation,
    SimulationResult,
    SimulationStatus,
)
from .models.twin_models import (
    Brand,
    DigitalTwin,
    Region,
    TwinModelConfig,
    TwinModelMetrics,
    TwinPopulation,
    TwinType,
)
from .retraining_service import (
    TwinRetrainingConfig,
    TwinRetrainingDecision,
    TwinRetrainingJob,
    TwinRetrainingService,
    TwinRetrainingStatus,
    TwinTriggerReason,
    get_twin_retraining_service,
)
from .simulation_cache import (
    CacheStats,
    SimulationCache,
    SimulationCacheConfig,
    get_simulation_cache,
)
from .simulation_engine import SimulationEngine
from .twin_generator import TwinGenerator
from .twin_repository import TwinRepository

__all__ = [
    # Core components
    "TwinGenerator",
    "SimulationEngine",
    "FidelityTracker",
    "TwinRepository",
    # Caching (Phase 5)
    "SimulationCache",
    "SimulationCacheConfig",
    "CacheStats",
    "get_simulation_cache",
    # Retraining (Phase 6)
    "TwinRetrainingService",
    "TwinRetrainingConfig",
    "TwinRetrainingDecision",
    "TwinRetrainingJob",
    "TwinRetrainingStatus",
    "TwinTriggerReason",
    "get_twin_retraining_service",
    # Twin models
    "DigitalTwin",
    "TwinPopulation",
    "TwinType",
    "Brand",
    "Region",
    "TwinModelConfig",
    "TwinModelMetrics",
    # Simulation models
    "InterventionConfig",
    "PopulationFilter",
    "SimulationResult",
    "SimulationStatus",
    "SimulationRecommendation",
    "EffectHeterogeneity",
    "FidelityRecord",
    "FidelityGrade",
]
