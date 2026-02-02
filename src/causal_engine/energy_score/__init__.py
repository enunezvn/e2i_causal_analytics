"""
Energy Score Module for Causal Estimator Selection

This module provides energy score-based selection for causal estimators,
replacing the legacy "first success" fallback with intelligent selection
based on estimator quality metrics.

V4.2 Enhancement - December 2024

Usage:
    from src.causal_engine.energy_score import (
        EnergyScoreCalculator,
        EstimatorSelector,
        SelectionStrategy,
    )

    # Simple API
    result = select_best_estimator(treatment, outcome, covariates)
    print(f"Best estimator: {result.selected.estimator_type}")
    print(f"Energy Score: {result.selected.energy_score:.4f}")

    # With custom configuration
    config = EstimatorSelectorConfig(
        strategy=SelectionStrategy.BEST_ENERGY_SCORE,
        max_acceptable_energy_score=0.7,
    )
    selector = EstimatorSelector(config)
    result = selector.select(treatment, outcome, covariates)
"""

# Score Calculator
# Estimator Selection
from .estimator_selector import (
    ESTIMATOR_WRAPPERS,
    BaseEstimatorWrapper,
    CausalForestWrapper,
    DRLearnerWrapper,
    EstimatorConfig,
    EstimatorResult,
    EstimatorSelector,
    EstimatorSelectorConfig,
    EstimatorType,
    LinearDMLWrapper,
    OLSWrapper,
    SelectionResult,
    SelectionStrategy,
    select_best_estimator,
)

# MLflow Integration
from .mlflow_tracker import (
    EnergyScoreMLflowTracker,
    ExperimentContext,
    create_tracker,
)
from .score_calculator import (
    EnergyScoreCalculator,
    EnergyScoreConfig,
    EnergyScoreResult,
    EnergyScoreVariant,
    compute_energy_score,
)

__all__ = [
    # Score Calculator
    "EnergyScoreCalculator",
    "EnergyScoreConfig",
    "EnergyScoreResult",
    "EnergyScoreVariant",
    "compute_energy_score",
    # Estimator Selection
    "BaseEstimatorWrapper",
    "CausalForestWrapper",
    "DRLearnerWrapper",
    "EstimatorConfig",
    "EstimatorResult",
    "EstimatorSelector",
    "EstimatorSelectorConfig",
    "EstimatorType",
    "LinearDMLWrapper",
    "OLSWrapper",
    "SelectionResult",
    "SelectionStrategy",
    "select_best_estimator",
    "ESTIMATOR_WRAPPERS",
    # MLflow Integration
    "EnergyScoreMLflowTracker",
    "ExperimentContext",
    "create_tracker",
]
