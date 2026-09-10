"""Every reachable refutation band under the 2026-09-10 rule (spec §6).

Two spaces are enumerated here and they answer different questions.

The ARITHMETIC space is every status combination the scoring code accepts, and the
tests over it document the band math: which weights produce which confidence, and
where the PROCEED/REVIEW/BLOCK cuts fall. Sensitivity is non-critical with statuses
{PASSED, WARNING, SKIPPED}; the other four are given {PASSED, WARNING, FAILED,
SKIPPED} minus what the criticals cannot carry (below).

The OPERATIONAL space is what the five refuters can actually EMIT, declared in
``EMITTABLE`` with the function and branch behind every status. It is strictly
smaller, and it is the space a leader's statement must be true in: with it, BLOCK
is reachable ONLY through a FAILED placebo or random_common_cause, never by
confidence alone. That sentence is what the lineage page's callout prints (Task
10), so it is pinned from the code here rather than from the arithmetic. If either
enumeration changes, this test and that callout must change together.

The two CRITICAL tests never carry SKIPPED, so the enumeration below gives them
only {PASSED, WARNING, FAILED}: a budget skip of a critical test raises
``RefutationError`` instead of appending a result (pinned by
``test_refutation_runner_1419.py::TestNonCriticalBudgetSkipDegrades::
test_critical_budget_skip_still_fails_closed``), and neither
``RefutationRunner._run_placebo_test`` nor ``_run_random_common_cause_test``
has a SKIPPED return; only ``_run_data_subset_test`` and ``_run_bootstrap_test``
reach the SKIPPED helpers. Enumerating a SKIPPED critical would add two
confidence values (0.0 and 0.30) that no run can produce.
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

# What each refuter can actually EMIT, read off the runner source 2026-09-10. This
# is NARROWER than the arithmetic space above; see the operational test below.
EMITTABLE = {
    # ``_run_placebo_test``: PASSED at ``p_value >= thresholds["placebo_p_value"]
    # ["pass"]`` (0.05), FAILED otherwise. The ``elif p_value >= ...["warning"]``
    # (0.10) WARNING branch is UNREACHABLE as coded — any p reaching the elif is
    # already below 0.05, so it can never be >= 0.10 (#1994 option 1: documented,
    # behaviour unchanged). No SKIPPED: the failure paths raise RefutationError.
    RefutationTestType.PLACEBO_TREATMENT: (P, F),
    # ``_run_random_common_cause_test``: delta_percent <= 20% PASSED, <= 30%
    # WARNING, else FAILED. All three reachable. No SKIPPED (failures raise).
    RefutationTestType.RANDOM_COMMON_CAUSE: (P, W, F),
    # ``_run_sensitivity_test`` via ``evalue.STATUS_BY_READING``: PASSED for
    # ``beyond_measured_confounding``; WARNING for ``within_measured_confounding``,
    # ``null_finding`` and ``unbenchmarked``; SKIPPED for the randomized design.
    # There is no FAILED outcome at all (spec §4.4).
    RefutationTestType.SENSITIVITY_E_VALUE: (P, W, S),
    # ``_run_data_subset_test``: ci_coverage >= 0.80 PASSED, >= 0.70 WARNING, else
    # FAILED; SKIPPED from the degenerate-CI, below-minimum-resamples and
    # degenerate-distribution helpers, and from the #1419 non-critical budget skip.
    RefutationTestType.DATA_SUBSET: (P, W, F, S),
    # ``_run_bootstrap_test``: ci_ratio <= 1.50 PASSED, <= 1.75 WARNING, else
    # FAILED; same four SKIPPED sources as data_subset.
    RefutationTestType.BOOTSTRAP: (P, W, F, S),
}


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


def test_block_is_only_reachable_through_a_critical_failed_with_emittable_statuses():
    """The statement the lineage callout prints: with the statuses the refuters can
    actually emit, BLOCK means a FAILED placebo or random_common_cause. Confidence
    alone never blocks — the floor without a critical FAILED is REVIEW.

    Hand arithmetic for that floor: placebo can only be PASSED here (it emits no
    WARNING), random_common_cause WARNING, sensitivity SKIPPED (excluded from the
    average), and both non-criticals FAILED:
    (0.25*1.0 + 0.25*0.6) / (0.25 + 0.25 + 0.125 + 0.125) = 0.40 / 0.75 = 0.5333.
    """
    r = RefutationRunner()
    names = list(EMITTABLE)
    review = set()
    floor = None
    for combo in itertools.product(*EMITTABLE.values()):
        tests = [RefutationResult(n, st, 0.1, 0.1) for n, st in zip(names, combo, strict=True)]
        # An all-SKIPPED suite cannot occur here: placebo emits only PASSED/FAILED.
        assert not all(t.status == S for t in tests)
        conf = round(r._calculate_confidence_score(tests), 4)
        gate = r._determine_gate_decision(tests, conf)
        critical_failed = combo[0] == F or combo[1] == F
        if gate == GateDecision.BLOCK:
            assert critical_failed, f"confidence-only BLOCK at {conf} from {combo}"
        if not critical_failed:
            floor = conf if floor is None else min(floor, conf)
            if 0.50 <= conf < 0.70:
                review.add(conf)
    assert floor == pytest.approx(0.5333, abs=1e-3)
    assert 0.50 <= floor < 0.70, "the floor without a critical failure is REVIEW"
    # Operationally reachable REVIEW values. Each is derived by hand below from one
    # witness combination, so the pin has an arithmetic independent of this loop.
    # Weights: placebo/rcc/sensitivity 0.25, subset/bootstrap 0.125; PASSED 1.0,
    # WARNING 0.6, FAILED 0.0; SKIPPED drops out of BOTH numerator and denominator.
    # Placebo is PASSED in every witness — it emits no WARNING, and FAILED would
    # block. Columns are placebo / rcc / sensitivity / subset / bootstrap.
    #   0.5333  P W S F F  (0.25 + 0.15)          / 0.75  = 0.40  / 0.75
    #   0.55    P W W F F  (0.25 + 0.15 + 0.15)   / 1.0   = 0.55  / 1.0
    #   0.625   P W W W F  (0.25 + 0.15 + 0.15 + 0.075) / 1.0 = 0.625 / 1.0
    #   0.6286  P W W F S  (0.25 + 0.15 + 0.15)   / 0.875 = 0.55  / 0.875
    #   0.6333  P W S W F  (0.25 + 0.15 + 0.075)  / 0.75  = 0.475 / 0.75
    #   0.64    P W S F S  (0.25 + 0.15)          / 0.625 = 0.40  / 0.625
    #   0.65    P P W F F  (0.25 + 0.25 + 0.15)   / 1.0   = 0.65  / 1.0
    #   0.6667  P P S F F  (0.25 + 0.25)          / 0.75  = 0.50  / 0.75
    #   0.675   P W W P F  (0.25 + 0.15 + 0.15 + 0.125) / 1.0 = 0.675 / 1.0
    assert sorted(review) == [
        0.5333,
        0.55,
        0.625,
        0.6286,
        0.6333,
        0.64,
        0.65,
        0.6667,
        0.675,
    ], sorted(review)
