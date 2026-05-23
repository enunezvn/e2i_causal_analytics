"""Cross-library validation module for causal inference.

This module provides:
- ValidationReportGenerator: Generate comprehensive validation reports
- TypedDict state shapes: result envelopes for validation reports and
  pairwise/refutation/A-B-reconciliation result data.

#457 RETIRE (2026-05-23): ``ConfidenceScorer``, ``CrossValidator``, and
``ABReconciler`` were retired from this package as orphan code per the
audit at ``.claude/research/457_validation_orphan_audit.md`` (iter-1
codex-ACCEPTed). Pipeline ``_apply_consensus`` and
``_apply_pairwise_agreement`` (``src/causal_engine/pipeline/sequential.py``)
now own the consensus-weighting and pairwise-agreement responsibilities
that those classes would have provided. If a future workstream needs
typed-envelope cross-validation or A/B-reconciliation, restore via
``git restore --source=15787a7f src/causal_engine/validation/<file>.py``
(originally introduced in commit ``26ce1fff``, 2025-12-29).

``ValidationReportGenerator`` remains exported as out-of-scope-for-#457
per codex iter-1 LOW #3 — see follow-up tracker for its own audit.
"""

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
    # Core Classes (post-#457 RETIRE; ValidationReportGenerator only)
    "ValidationReportGenerator",
    # State TypedDicts (kept; used by ValidationReportGenerator and
    # potential future consumers; pruning is in the V-RG follow-up
    # tracker per codex iter-1 LOW #3)
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
