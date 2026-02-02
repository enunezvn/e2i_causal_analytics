"""
E2I Causal Engine
Version: 4.4
Purpose: Causal inference and validation utilities

This module provides:
- RefutationRunner: DoWhy-based refutation testing for causal estimate validation
- Gate decision logic (proceed/review/block)
- Database-aligned ENUMs and dataclasses
- DAG version hashing for expert review workflow
- ValidationOutcome: Feedback Learner integration (Phase 4)
- ExperimentKnowledgeStore: Past failure queries (Phase 4)
- Energy Score: Estimator selection based on quality metrics (V4.2 Enhancement)
"""

from .dag_hash import (
    compute_dag_hash,
    compute_dag_hash_from_dot,
    get_dag_changes,
    is_dag_changed,
    validate_dag_hash,
)

# V4.2 Enhancement: Energy Score-based Estimator Selection
from .energy_score import (
    # Score Calculator
    EnergyScoreCalculator,
    EnergyScoreConfig,
    # MLflow Integration
    EnergyScoreMLflowTracker,
    EnergyScoreResult,
    EnergyScoreVariant,
    EstimatorResult,
    # Estimator Selection
    EstimatorSelector,
    EstimatorSelectorConfig,
    EstimatorType,
    SelectionResult,
    SelectionStrategy,
    compute_energy_score,
    create_tracker,
    select_best_estimator,
)
from .expert_review_gate import (
    ExpertReviewGate,
    ReviewGateDecision,
    ReviewGateResult,
    check_dag_approval,
)
from .refutation_runner import (
    # Constants
    DOWHY_AVAILABLE,
    GateDecision,
    # Dataclasses
    RefutationResult,
    # Main class
    RefutationRunner,
    # ENUMs
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
    is_estimate_valid,
    # Convenience functions
    run_refutation_suite,
)
from .validation_outcome import (
    FailureCategory,
    ValidationFailurePattern,
    # Dataclasses
    ValidationOutcome,
    # ENUMs
    ValidationOutcomeType,
    # Functions
    create_validation_outcome,
    extract_failure_patterns,
)
from .validation_outcome_store import (
    ExperimentKnowledgeStore,
    InMemoryValidationOutcomeStore,
    ValidationLearning,
    # Store classes
    ValidationOutcomeStoreBase,
    get_experiment_knowledge_store,
    # Global accessors
    get_validation_outcome_store,
    log_validation_outcome,
)

__all__ = [
    # ENUMs
    "RefutationStatus",
    "GateDecision",
    "RefutationTestType",
    # Dataclasses
    "RefutationResult",
    "RefutationSuite",
    # Main class
    "RefutationRunner",
    # Convenience functions
    "run_refutation_suite",
    "is_estimate_valid",
    # Constants
    "DOWHY_AVAILABLE",
    # DAG Hashing
    "compute_dag_hash",
    "compute_dag_hash_from_dot",
    "is_dag_changed",
    "get_dag_changes",
    "validate_dag_hash",
    # Expert Review Gate
    "ExpertReviewGate",
    "ReviewGateDecision",
    "ReviewGateResult",
    "check_dag_approval",
    # Phase 4: Validation Outcomes
    "ValidationOutcomeType",
    "FailureCategory",
    "ValidationOutcome",
    "ValidationFailurePattern",
    "create_validation_outcome",
    "extract_failure_patterns",
    # Phase 4: Outcome Store
    "ValidationOutcomeStoreBase",
    "InMemoryValidationOutcomeStore",
    "ExperimentKnowledgeStore",
    "ValidationLearning",
    "get_validation_outcome_store",
    "get_experiment_knowledge_store",
    "log_validation_outcome",
    # V4.2 Enhancement: Energy Score
    "EnergyScoreCalculator",
    "EnergyScoreConfig",
    "EnergyScoreResult",
    "EnergyScoreVariant",
    "compute_energy_score",
    "EstimatorSelector",
    "EstimatorSelectorConfig",
    "EstimatorType",
    "EstimatorResult",
    "SelectionResult",
    "SelectionStrategy",
    "select_best_estimator",
    "EnergyScoreMLflowTracker",
    "create_tracker",
]
