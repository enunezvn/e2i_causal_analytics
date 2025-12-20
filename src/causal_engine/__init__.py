"""
E2I Causal Engine
Version: 4.3
Purpose: Causal inference and validation utilities

This module provides:
- RefutationRunner: DoWhy-based refutation testing for causal estimate validation
- Gate decision logic (proceed/review/block)
- Database-aligned ENUMs and dataclasses
- DAG version hashing for expert review workflow
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
]
