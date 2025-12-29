"""Cross-library validation module for causal inference.

B8: Validation Loop (DoWhy ↔ CausalML cross-validation)

This module provides:
- CrossValidator: Pairwise validation between DoWhy, EconML, CausalML
- ABReconciler: Reconcile causal estimates with A/B experiments
- ValidationReportGenerator: Generate comprehensive validation reports
- ConfidenceScorer: Compute confidence based on library agreement

Usage:
    from src.causal_engine.validation import CrossValidator, ABReconciler

    validator = CrossValidator()
    result = await validator.validate(
        treatment_var="marketing_spend",
        outcome_var="conversion_rate",
        data=df,
    )
"""

from src.causal_engine.validation.ab_reconciler import ABReconciler
from src.causal_engine.validation.confidence_scorer import (
    ConfidenceScorer,
    compute_pipeline_confidence,
)
from src.causal_engine.validation.cross_validator import CrossValidator
from src.causal_engine.validation.report_generator import ValidationReportGenerator
from src.causal_engine.validation.state import (
    ABExperimentResult,
    ABReconciliationResult,
    CrossValidationResult,
    LibraryEffectEstimate,
    PairwiseValidation,
    RefutationValidation,
    ValidationReport,
    ValidationReportSection,
    ValidationSummary,
)

__all__ = [
    # Core Classes
    "CrossValidator",
    "ABReconciler",
    "ValidationReportGenerator",
    "ConfidenceScorer",
    # Utility Functions
    "compute_pipeline_confidence",
    # State TypedDicts
    "LibraryEffectEstimate",
    "PairwiseValidation",
    "RefutationValidation",
    "ValidationSummary",
    "CrossValidationResult",
    "ABExperimentResult",
    "ABReconciliationResult",
    "ValidationReportSection",
    "ValidationReport",
]
