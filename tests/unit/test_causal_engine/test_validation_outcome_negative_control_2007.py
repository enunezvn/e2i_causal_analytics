"""Failure-pattern extraction for the negative-control-outcome reading (#2007).

Lane G shipped ``negative_control_outcome`` as a weight-0, non-critical reading
in ``RefutationRunner._run_negative_control_test``. Before this file, the
categoriser in ``validation_outcome.py`` had no branch for it: T1's implementer
measured that a FAILED control landed in ``FailureCategory.UNKNOWN`` with the
generic "Test negative_control_outcome failed with N% change" sentence -- the one
refuter that can DETECT confounding would have been learned as "unknown".

What this pins (the runner's own contract, read from its docstring):

* ``extract_failure_patterns`` admits FAILED *and* WARNING rows (the module's
  existing behaviour for every test; a WARNING control -- moved, less than the
  claim -- is a pattern worth learning too), so both verdicts are covered.
* Category: ``UNOBSERVED_CONFOUNDING`` -- the existing member whose meaning is
  "the adjustment left confounding in the estimate". No new enum member: the
  sensitivity test's ``within_measured_confounding`` reading already uses it for
  the same mechanism read from a different instrument.
* Severity follows the verdict: FAILED (the control moved at least as much as the
  claim) -> ``high``; WARNING (moved, less than the claim) -> ``medium``.
* Description is ``details["reading"]`` verbatim -- the one sentence the runner
  composed with the control's name, interval and row basis -- so the learned row
  cannot contradict the verdict stored beside it. A row without the reading
  (a direct caller, or a row persisted without details) gets a sentence built
  from ``nc_outcome`` / ``nc_effect`` / ``nc_ci`` rather than a "% change".
"""

from __future__ import annotations

import pytest

from src.causal_engine import (
    FailureCategory,
    GateDecision,
    RefutationResult,
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
    extract_failure_patterns,
)

# Numbers from docs/demos/results/2026-09-11_negative_control_disproof/
# calibration_n1500.md (omitted-confounder fits, seed 21, n = 1500):
#   psp_enrolled -> persistent_180d, claimed +0.0879, control treatment_initiated
#   omitted +0.0895 [+0.0402, +0.1388] -> FAILED (|nc| >= |claimed|)
#   copay_support -> adherent_180d, claimed +0.1209, control treatment_initiated
#   omitted +0.0517 [+0.0005, +0.1029] -> WARNING (|nc| < |claimed|)
_FAILED_READING = (
    "A negative-control outcome the treatment cannot affect (treatment_initiated) "
    "moved by +0.089 [+0.040, +0.139] on n = 1500, at least as much as the claimed "
    "effect +0.088: the adjustment is leaking confounding."
)
_WARNING_READING = (
    "A negative-control outcome the treatment cannot affect (treatment_initiated) "
    "moved by +0.052 [+0.001, +0.103] on n = 1500, less than the claimed effect +0.121."
)


def _nc_result(
    status: RefutationStatus,
    *,
    original_effect: float,
    nc_effect: float,
    nc_ci: tuple[float, float],
    reading: str | None,
) -> RefutationResult:
    """A negative-control row shaped exactly as the runner persists it."""
    details = {
        "nc_outcome": "treatment_initiated",
        "nc_effect": nc_effect,
        "nc_ci": [nc_ci[0], nc_ci[1]],
        "nc_n": 1500,
        "rule": "negative_control_ci_vs_zero",
        "weight": 0.0,
        "critical": False,
    }
    if reading is not None:
        details["reading"] = reading
        details["message"] = reading
    return RefutationResult(
        test_name=RefutationTestType.NEGATIVE_CONTROL_OUTCOME,
        status=status,
        original_effect=original_effect,
        refuted_effect=nc_effect,
        p_value=None,
        delta_percent=100.0 * abs(nc_effect) / abs(original_effect),
        details=details,
    )


def _patterns_for(result: RefutationResult):
    suite = RefutationSuite(
        passed=True,  # weight 0, non-critical: the reading never fails the suite
        confidence_score=1.0,
        gate_decision=GateDecision.PROCEED,
        tests=[result],
    )
    return extract_failure_patterns(suite)


class TestNegativeControlFailurePattern:
    def test_failed_control_is_unobserved_confounding_high_with_the_reading(self):
        result = _nc_result(
            RefutationStatus.FAILED,
            original_effect=0.0879,
            nc_effect=0.0895,
            nc_ci=(0.0402, 0.1388),
            reading=_FAILED_READING,
        )
        patterns = _patterns_for(result)
        assert len(patterns) == 1
        pattern = patterns[0]
        assert pattern.test_name == "negative_control_outcome"
        assert pattern.category == FailureCategory.UNOBSERVED_CONFOUNDING
        assert pattern.category != FailureCategory.UNKNOWN
        assert pattern.severity == "high"
        # The description IS the runner's sentence -- never the generic "% change".
        assert pattern.description == _FAILED_READING
        assert "% change" not in pattern.description
        # The recommendation names the instrument and the mechanism, not a
        # generic "review manually".
        rec = pattern.recommendation.lower()
        assert "negative-control" in rec or "negative control" in rec
        assert "confound" in rec
        assert "review test results manually" not in rec
        # The verdict's own number rides along untouched.
        assert pattern.original_effect == pytest.approx(0.0879)
        assert pattern.refuted_effect == pytest.approx(0.0895)

    def test_warning_control_is_extracted_as_medium_with_the_reading(self):
        """WARNING rows are admitted by ``extract_failure_patterns`` for every
        test (measured on the module: FAILED and WARNING both pass its filter),
        so a control that moved less than the claim is learned as a medium
        unobserved-confounding pattern with its own sentence."""
        result = _nc_result(
            RefutationStatus.WARNING,
            original_effect=0.1209,
            nc_effect=0.0517,
            nc_ci=(0.0005, 0.1029),
            reading=_WARNING_READING,
        )
        patterns = _patterns_for(result)
        assert len(patterns) == 1
        pattern = patterns[0]
        assert pattern.category == FailureCategory.UNOBSERVED_CONFOUNDING
        assert pattern.severity == "medium"
        assert pattern.description == _WARNING_READING

    def test_passed_control_yields_no_pattern(self):
        result = _nc_result(
            RefutationStatus.PASSED,
            original_effect=0.1209,
            nc_effect=0.0172,
            nc_ci=(-0.0307, 0.0652),
            reading="A negative-control outcome the treatment cannot affect "
            "(treatment_initiated) stayed null: +0.017 [-0.031, +0.065] on n = 1500.",
        )
        assert _patterns_for(result) == []

    def test_row_without_reading_describes_the_control_not_a_percentage(self):
        """A direct caller (or a row persisted without the sentence) still gets a
        description built from the control's name, effect and interval."""
        result = _nc_result(
            RefutationStatus.FAILED,
            original_effect=0.0879,
            nc_effect=0.0895,
            nc_ci=(0.0402, 0.1388),
            reading=None,
        )
        patterns = _patterns_for(result)
        assert len(patterns) == 1
        desc = patterns[0].description
        assert "treatment_initiated" in desc
        assert "+0.089" in desc  # 0.0895 formats to +0.089 with :+.3f, as the runner prints it
        assert "0.040" in desc and "0.139" in desc
        assert "% change" not in desc
        assert patterns[0].category == FailureCategory.UNOBSERVED_CONFOUNDING
