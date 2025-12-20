"""
E2I Causal Engine
Version: 4.3
Purpose: Causal inference and validation utilities

This module provides:
- RefutationRunner: DoWhy-based refutation testing for causal estimate validation
- Gate decision logic (proceed/review/block)
- Database-aligned ENUMs and dataclasses
- DAG version hashing for expert review workflow
- ValidationOutcome: Feedback Learner integration (Phase 4)
- ExperimentKnowledgeStore: Past failure queries (Phase 4)
"""

from .refutation_runner import (
    # ENUMs
    RefutationStatus,
    GateDecision,
    RefutationTestType,
    # Dataclasses
    RefutationResult,
    RefutationSuite,
    # Main class
    RefutationRunner,
    # Convenience functions
    run_refutation_suite,
    is_estimate_valid,
    # Constants
    DOWHY_AVAILABLE,
)

from .dag_hash import (
    compute_dag_hash,
    compute_dag_hash_from_dot,
    is_dag_changed,
    get_dag_changes,
    validate_dag_hash,
)

from .expert_review_gate import (
    ExpertReviewGate,
    ReviewGateDecision,
    ReviewGateResult,
    check_dag_approval,
)

from .validation_outcome import (
    # ENUMs
    ValidationOutcomeType,
    FailureCategory,
    # Dataclasses
    ValidationOutcome,
    ValidationFailurePattern,
    # Functions
    create_validation_outcome,
    extract_failure_patterns,
)

from .validation_outcome_store import (
    # Store classes
    ValidationOutcomeStoreBase,
    InMemoryValidationOutcomeStore,
    ExperimentKnowledgeStore,
    ValidationLearning,
    # Global accessors
    get_validation_outcome_store,
    get_experiment_knowledge_store,
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
]
