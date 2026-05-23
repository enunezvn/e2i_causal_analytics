"""Drift-guard tests for the validation package public surface.

Regression for #457 + #464 — the entire `src/causal_engine/validation/`
"Core Class" surface was retired in two phases:

Phase 1 (#457, PR #463, merged 2026-05-23, commit ``e2156ad5``):
  * ConfidenceScorer (was at validation/confidence_scorer.py, 602 LoC)
  * CrossValidator (was at validation/cross_validator.py, 675 LoC)
  * ABReconciler (was at validation/ab_reconciler.py, 555 LoC)
  * compute_pipeline_confidence (was a helper in confidence_scorer.py)

Phase 2 (#464, this PR):
  * ValidationReportGenerator (was at validation/report_generator.py, 837 LoC)
  * 9 TypedDict envelopes (was at validation/state.py, 189 LoC):
    LibraryEffectEstimate, PairwiseValidation, RefutationValidation,
    ValidationSummary, CrossValidationResult, ABExperimentResult,
    ABReconciliationResult, ValidationReportSection, ValidationReport

These tests assert both retires stay in effect. If any future change
re-introduces the symbols or modules without an accompanying intent
investigation per the audit framework, these tests trip loudly.

Positive-presence anchor: `_apply_consensus` and `_apply_pairwise_agreement`
in `src.causal_engine.pipeline.sequential` are the live replacements for
the Phase 1 upstream responsibilities; this test verifies they remain
importable (negative control for an over-aggressive future deletion).

Reversibility (Phase 2): full restoration ~30 min via
``git restore --source=735465ce src/causal_engine/validation/report_generator.py``
``git restore --source=735465ce src/causal_engine/validation/state.py``
``git restore --source=735465ce tests/unit/test_causal_engine/test_validation/test_report_generator.py``
plus restoring the corresponding ``__init__.py`` imports/__all__
entries. Original implementation commit: ``26ce1fff`` (2025-12-29).
"""

from __future__ import annotations

import importlib

import pytest

import src.causal_engine.validation as validation_pkg

_RETIRED_SYMBOLS = (
    # Phase 1 (#457)
    "ConfidenceScorer",
    "CrossValidator",
    "ABReconciler",
    "compute_pipeline_confidence",
    # Phase 2 (#464)
    "ValidationReportGenerator",
    "LibraryEffectEstimate",
    "PairwiseValidation",
    "RefutationValidation",
    "ValidationSummary",
    "CrossValidationResult",
    "ABExperimentResult",
    "ABReconciliationResult",
    "ValidationReportSection",
    "ValidationReport",
)

_RETIRED_MODULES = (
    # Phase 1 (#457)
    "src.causal_engine.validation.confidence_scorer",
    "src.causal_engine.validation.cross_validator",
    "src.causal_engine.validation.ab_reconciler",
    # Phase 2 (#464)
    "src.causal_engine.validation.report_generator",
    "src.causal_engine.validation.state",
)


def test_retired_symbols_absent_from_all() -> None:
    """`validation.__all__` must not advertise any retired symbol."""
    all_exports = set(getattr(validation_pkg, "__all__", []))
    leaked = sorted(all_exports & set(_RETIRED_SYMBOLS))
    assert not leaked, (
        f"validation.__all__ still advertises retired symbols {leaked}; "
        "if a re-introduction is intentional, see the audit framework "
        "(.claude/research/457_*.md + .claude/research/464_*.md) before "
        "un-retiring."
    )


def test_retired_symbols_absent_from_package_namespace() -> None:
    """Direct attribute access on the validation package must fail."""
    for symbol in _RETIRED_SYMBOLS:
        with pytest.raises(AttributeError):
            getattr(validation_pkg, symbol)


def test_retired_modules_absent() -> None:
    """All retired source modules must not be importable."""
    for module_name in _RETIRED_MODULES:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_pipeline_consensus_replacement_still_importable() -> None:
    """Negative control: positive-presence anchor for #457 replacement.

    `_apply_consensus` and `_apply_pairwise_agreement` in
    `src.causal_engine.pipeline.sequential` are the live replacements for
    the Phase 1 upstream responsibilities (ConfidenceScorer consensus
    weighting + CrossValidator pairwise agreement). They must remain
    importable; if a future PR accidentally removes the replacement
    surface, this test trips and forces an explicit decision.
    """
    sequential_mod = importlib.import_module("src.causal_engine.pipeline.sequential")
    assert hasattr(sequential_mod, "_apply_consensus"), (
        "_apply_consensus missing from pipeline.sequential — this is the "
        "live replacement for ConfidenceScorer consensus weighting; "
        "removal requires explicit decision per #457 audit framework."
    )
    assert hasattr(sequential_mod, "_apply_pairwise_agreement"), (
        "_apply_pairwise_agreement missing from pipeline.sequential — "
        "this is the live replacement for CrossValidator pairwise "
        "agreement; removal requires explicit decision per #457 audit "
        "framework."
    )
