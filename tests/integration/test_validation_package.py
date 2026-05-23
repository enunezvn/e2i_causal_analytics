"""Drift-guard tests for the validation package public surface.

Regression for #457 — three classes were retired from
`src/causal_engine/validation/` per memo R5 RETIRE-ALL-THREE recommendation
(codex iter-1 ACCEPTed 2026-05-23):

* ConfidenceScorer (was at validation/confidence_scorer.py, 602 LoC)
* CrossValidator (was at validation/cross_validator.py, 675 LoC)
* ABReconciler (was at validation/ab_reconciler.py, 555 LoC)
* compute_pipeline_confidence (was a helper in confidence_scorer.py)

These tests assert the retire stays in effect. If any future change
re-introduces the symbols at the package level without an accompanying
intent investigation per #457's audit framework, these tests trip
loudly.

Reversibility: full restoration ~30 min per class via
``git restore --source=15787a7f src/causal_engine/validation/<file>.py``
plus restoring the corresponding ``__init__.py`` imports/__all__
entries. Original implementation commit: ``26ce1fff`` (2025-12-29).
"""

from __future__ import annotations

import importlib

import pytest

import src.causal_engine.validation as validation_pkg

_RETIRED_SYMBOLS = (
    "ConfidenceScorer",
    "CrossValidator",
    "ABReconciler",
    "compute_pipeline_confidence",
)


def test_retired_symbols_absent_from_all() -> None:
    """`validation.__all__` must not advertise any retired symbol."""
    all_exports = set(getattr(validation_pkg, "__all__", []))
    leaked = sorted(all_exports & set(_RETIRED_SYMBOLS))
    assert not leaked, (
        f"validation.__all__ still advertises retired symbols {leaked}; "
        "if a re-introduction is intentional, see #457 audit framework "
        "(.claude/research/457_validation_orphan_audit.md) before un-retiring."
    )


def test_retired_symbols_absent_from_package_namespace() -> None:
    """Direct attribute access on the validation package must fail."""
    for symbol in _RETIRED_SYMBOLS:
        with pytest.raises(AttributeError):
            getattr(validation_pkg, symbol)


def test_retired_modules_absent() -> None:
    """The three retired source modules must not be importable."""
    for module_name in (
        "src.causal_engine.validation.confidence_scorer",
        "src.causal_engine.validation.cross_validator",
        "src.causal_engine.validation.ab_reconciler",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_validation_report_generator_still_exported() -> None:
    """ValidationReportGenerator stays — out-of-scope per codex iter-1 L-3.

    Negative-control: this test guards against an over-eager future PR
    accidentally retiring all four orphans. ValidationReportGenerator
    has its own follow-up tracker; until that tracker is resolved, the
    class stays. See codex iter-1 LOW #3.
    """
    from src.causal_engine.validation import ValidationReportGenerator  # noqa: F401

    assert "ValidationReportGenerator" in getattr(validation_pkg, "__all__", [])
