"""
E2I Causal Engine
Version: 4.3
Purpose: Causal inference and validation utilities

This module provides:
- RefutationRunner: DoWhy-based refutation testing for causal estimate validation
- Gate decision logic (proceed/review/block)
- Database-aligned ENUMs and dataclasses
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
]
