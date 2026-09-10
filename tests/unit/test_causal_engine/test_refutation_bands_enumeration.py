"""Every reachable refutation band under the 2026-09-10 rule (spec §6).

Sensitivity is non-critical with statuses {PASSED, WARNING, SKIPPED}; the other four
keep {PASSED, WARNING, FAILED, SKIPPED}. The enumerated table below is copied into
the lineage page's REVIEW-band callout (Task 10); if the arithmetic changes, both
this test and that callout must change together.

One correction to the plan's enumeration, measured 2026-09-10 before this file was
written: the two CRITICAL tests can never carry SKIPPED. ``_run_placebo_test`` and
``_run_random_common_cause_test`` have no SKIPPED return at all, and the #1419
budget-skip policy RAISES ``RefutationError`` for a critical skip rather than
appending a SKIPPED result (``refutation_runner.py``, "A budget-skipped CRITICAL
gate still fails the suite closed"); the three SKIPPED helpers are reachable only
from the non-critical ``data_subset`` and ``bootstrap``. Enumerating a SKIPPED
critical would add two confidence values (0.0 from all-critical-SKIPPED plus both
non-criticals FAILED, and 0.30 from one critical WARNING with the rest SKIPPED)
that no run can produce. The runner's weights and status scores are exactly what
the plan assumed — placebo/random_common_cause/sensitivity 0.25, data_subset/
bootstrap 0.125, PASSED 1.0 / WARNING 0.6 / FAILED 0.0, SKIPPED excluded from the
average — and restricting the criticals to {PASSED, WARNING, FAILED} reproduces the
plan's hand-derived BLOCK set exactly.
"""

from __future__ import annotations

import itertools

import pytest

from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationRunner,
    RefutationStatus,
    RefutationTestType,
)

P, W, F, S = (
    RefutationStatus.PASSED,
    RefutationStatus.WARNING,
    RefutationStatus.FAILED,
    RefutationStatus.SKIPPED,
)
CRITICAL = (RefutationTestType.PLACEBO_TREATMENT, RefutationTestType.RANDOM_COMMON_CAUSE)
NONCRIT = (RefutationTestType.DATA_SUBSET, RefutationTestType.BOOTSTRAP)
SENS = RefutationTestType.SENSITIVITY_E_VALUE

# Statuses each test can actually carry in a result list (see the module docstring
# for why the criticals exclude SKIPPED).
CRITICAL_STATUSES = [P, W, F]
SENS_STATUSES = [P, W, S]
NONCRIT_STATUSES = [P, W, F, S]


def _suite(pl, rcc, sens, sub, boot):
    mk = lambda n, s: RefutationResult(n, s, 0.1, 0.1)  # noqa: E731
    return [
        mk(CRITICAL[0], pl),
        mk(CRITICAL[1], rcc),
        mk(SENS, sens),
        mk(NONCRIT[0], sub),
        mk(NONCRIT[1], boot),
    ]


def _all_combos():
    yield from itertools.product(
        CRITICAL_STATUSES, CRITICAL_STATUSES, SENS_STATUSES, NONCRIT_STATUSES, NONCRIT_STATUSES
    )


def test_sensitivity_status_never_changes_a_proceed_when_the_other_tests_pass():
    r = RefutationRunner()
    for sens in (P, W, S):
        tests = _suite(P, P, sens, P, P)
        conf = r._calculate_confidence_score(tests)
        assert r._determine_gate_decision(tests, conf) == GateDecision.PROCEED, sens


def test_only_placebo_or_random_common_cause_can_block_on_their_own():
    r = RefutationRunner()
    for pl, rcc, sens, sub, boot in _all_combos():
        tests = _suite(pl, rcc, sens, sub, boot)
        conf = r._calculate_confidence_score(tests)
        gate = r._determine_gate_decision(tests, conf)
        if pl == F or rcc == F:
            assert gate == GateDecision.BLOCK
        else:
            assert gate == (
                GateDecision.PROCEED
                if conf >= 0.70
                else GateDecision.REVIEW
                if conf >= 0.50
                else GateDecision.BLOCK
            )


def test_reachable_confidence_values_without_a_critical_failure():
    """The exact set of confidence values a run can carry without a critical FAILED.
    REVIEW (0.50 <= c < 0.70) is reachable; BLOCK by confidence alone needs c < 0.50."""
    r = RefutationRunner()
    reachable = set()
    for pl, rcc, sens, sub, boot in _all_combos():
        if pl == F or rcc == F:
            continue
        tests = _suite(pl, rcc, sens, sub, boot)
        reachable.add(round(r._calculate_confidence_score(tests), 3))
    review = sorted(v for v in reachable if 0.50 <= v < 0.70)
    block = sorted(v for v in reachable if v < 0.50)
    assert review, "REVIEW must be reachable without a critical failure"
    assert 0.65 in review and 0.6 in review
    # Confidence-only BLOCK needs BOTH critical tests in WARNING and BOTH non-critical
    # tests FAILED (sensitivity PASSED/WARNING/SKIPPED): 0.40, 0.45 and 0.48. Pinned so
    # the lineage callout and this test move together.
    assert block == [0.4, 0.45, 0.48], block
    assert max(reachable) == 1.0


def test_null_finding_run_proceeds_when_everything_else_passes():
    """Owner decision 2026-09-10: a CI including zero is served, not blocked."""
    r = RefutationRunner()
    tests = _suite(P, P, W, P, P)  # sensitivity WARNING = null_finding or within
    conf = r._calculate_confidence_score(tests)
    assert conf == pytest.approx(0.90)
    assert r._determine_gate_decision(tests, conf) == GateDecision.PROCEED
