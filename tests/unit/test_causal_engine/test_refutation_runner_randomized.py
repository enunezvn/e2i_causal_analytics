# tests/unit/test_causal_engine/test_refutation_runner_randomized.py
"""RCT-awareness of the ``sensitivity_e_value`` gate (post-#1217 follow-up).

The E-value (VanderWeele & Ding 2017) quantifies how strong UNMEASURED
CONFOUNDING would have to be to explain away an effect — a threat model that
does not apply to a genuinely randomized treatment (assignment is exogenous by
construction). ``sensitivity_e_value`` USED to be a CRITICAL gate: an honest small
standardized effect on an RCT failed ``e_value_min`` and hard-BLOCKed the whole
estimate. Verified live on the nba_triggers RCT question: gate-block with
confidence 0.67, with AND without baseline adjustment (documented in PR #1217).

Since the 2026-09-10 calibration (spec §4.4/§4.5) the test is a non-critical
READING with no FAILED outcome, so the observational path no longer BLOCKs on a
weak E-value either. The randomized contract below is UNCHANGED and still needs
its guards: ``randomized_design=True`` must produce SKIPPED (no confidence
weight, omitted from ``individual_tests``) while the observational path produces
a SCORED reading — the two must never collapse into each other, or the flag
would be indistinguishable from doing nothing.

Fix contract: callers declare ``randomized_design=True`` from DESIGN knowledge
(the dataset spec — NEVER inferred from an empty discovered backdoor, which
would fail-open observational questions where discovery simply found nothing).
The runner still computes the E-value for information but returns
status=SKIPPED (excluded from confidence, never a critical failure) with the
numbers preserved in details.
"""

import pytest

from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationRunner,
    RefutationStatus,
    RefutationTestType,
)
from tests.unit.test_causal_engine.test_refutation_runner import (
    _full_stub_causal_model,
    _stub_estimate,
)

# A weak-but-real standardized effect: point E-value ≈ 1.42, CI-bound
# E-value ≈ 1.36 — below the 1.5 warning threshold → FAILED for an
# observational run, i.e. exactly the profile that blocked the live RCT.
_WEAK_EFFECT = 0.05
_WEAK_CI = (0.04, 0.06)
_OUTCOME_STD = 0.5


class TestSensitivityTestRandomizedDesign:
    def test_randomized_design_is_skipped_not_failed(self):
        """A randomized design must never FAIL the unmeasured-confounding gate:
        the test is reported as SKIPPED (not applicable), with the message
        naming the randomized design."""
        runner = RefutationRunner()
        result = runner._run_sensitivity_test(
            original_effect=_WEAK_EFFECT,
            original_ci=_WEAK_CI,
            outcome_std=_OUTCOME_STD,
            randomized_design=True,
        )
        assert result.status == RefutationStatus.SKIPPED
        assert "randomized" in result.details["message"].lower()
        assert result.details["gate_applicable"] is False

    def test_randomized_design_still_reports_computed_evalue(self):
        """SKIPPED is not silence: the informational E-value numbers must be
        identical to what the observational path would have computed."""
        runner = RefutationRunner()
        observational = runner._run_sensitivity_test(
            original_effect=_WEAK_EFFECT,
            original_ci=_WEAK_CI,
            outcome_std=_OUTCOME_STD,
        )
        randomized = runner._run_sensitivity_test(
            original_effect=_WEAK_EFFECT,
            original_ci=_WEAK_CI,
            outcome_std=_OUTCOME_STD,
            randomized_design=True,
        )
        assert randomized.details["e_value"] == observational.details["e_value"]
        assert randomized.details["e_value_ci"] == observational.details["e_value_ci"]

    def test_observational_default_is_scored_not_skipped(self):
        """Guard that the flag is not fail-open: without it the SAME weak effect
        is SCORED (a reading that carries confidence weight), not SKIPPED.

        Pre-2026-09-10 this asserted FAILED. The sensitivity test no longer has a
        FAILED outcome (spec §4.4) — with no measured confounders supplied the
        weak effect reads ``unbenchmarked`` → WARNING. What still distinguishes
        the two paths, and what this guard exists for, is SKIPPED vs scored."""
        runner = RefutationRunner()
        result = runner._run_sensitivity_test(
            original_effect=_WEAK_EFFECT,
            original_ci=_WEAK_CI,
            outcome_std=_OUTCOME_STD,
        )
        assert result.status != RefutationStatus.SKIPPED
        assert result.status == RefutationStatus.WARNING
        assert result.details["reading"] == "unbenchmarked"
        assert result.details["gate_applicable"] is True


class TestRunAllTestsRandomizedDesign:
    def test_randomized_design_unblocks_the_gate(self):
        """The live-observed failure mode end-to-end: all four DoWhy refuters
        pass but the weak E-value hard-BLOCKs. With randomized_design=True the
        e-value is SKIPPED and the gate PROCEEDs on the real evidence."""
        runner = RefutationRunner()
        suite = runner.run_all_tests(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
            causal_model=_full_stub_causal_model(),
            identified_estimand=object(),
            estimate=_stub_estimate(),
            randomized_design=True,
        )
        by_name = {t.test_name: t for t in suite.tests}
        assert by_name[RefutationTestType.SENSITIVITY_E_VALUE].status == RefutationStatus.SKIPPED
        assert suite.gate_decision == GateDecision.PROCEED

    def test_observational_run_all_tests_scores_the_reading_and_proceeds(self):
        """The same suite WITHOUT the flag now scores the reading instead of
        blocking on it (spec §4.4/§4.5).

        Pre-2026-09-10 this asserted FAILED → BLOCK, which is the live-observed
        defect the calibration removes: four refuters PASS and the estimate was
        hard-BLOCKed by the E-value alone. The reading is now WARNING and
        non-critical, so the gate is decided by confidence: 0.25 + 0.25 +
        0.25*0.6 + 0.125 + 0.125 = 0.90 → PROCEED. The randomized path above
        still differs — SKIPPED, confidence 1.0."""
        runner = RefutationRunner()
        suite = runner.run_all_tests(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
            causal_model=_full_stub_causal_model(),
            identified_estimand=object(),
            estimate=_stub_estimate(),
        )
        by_name = {t.test_name: t for t in suite.tests}
        sensitivity = by_name[RefutationTestType.SENSITIVITY_E_VALUE]
        assert sensitivity.status == RefutationStatus.WARNING
        assert sensitivity.status != RefutationStatus.FAILED
        assert suite.confidence_score == pytest.approx(0.90)
        assert suite.gate_decision == GateDecision.PROCEED


class TestSkippedEvalueGateMechanics:
    """Green-today mechanism guards the fix relies on: SKIPPED carries no
    confidence weight and never trips the critical-failure rule."""

    def _tests(self):
        passed = [
            RefutationResult(
                test_name=name,
                status=RefutationStatus.PASSED,
                original_effect=0.08,
                refuted_effect=0.01,
            )
            for name in (
                RefutationTestType.PLACEBO_TREATMENT,
                RefutationTestType.RANDOM_COMMON_CAUSE,
                RefutationTestType.DATA_SUBSET,
                RefutationTestType.BOOTSTRAP,
            )
        ]
        skipped = RefutationResult(
            test_name=RefutationTestType.SENSITIVITY_E_VALUE,
            status=RefutationStatus.SKIPPED,
            original_effect=0.08,
            refuted_effect=0.08,
        )
        return passed + [skipped]

    def test_confidence_excludes_skipped_evalue(self):
        runner = RefutationRunner()
        assert runner._calculate_confidence_score(self._tests()) == 1.0

    def test_gate_proceeds_with_skipped_evalue(self):
        runner = RefutationRunner()
        decision = runner._determine_gate_decision(self._tests(), confidence_score=1.0)
        assert decision == GateDecision.PROCEED

    def test_legacy_format_omits_skipped_tests(self):
        """A SKIPPED (not-applicable) test must NOT appear in individual_tests:
        the legacy dict is pass/fail-shaped (``passed: bool``), so a skipped
        e-value would render as a red FAILED row in the FE while total_tests
        already excludes it (the #1205 misleading-state class). Absent key →
        every consumer three-states to None/not-narrated."""
        from src.causal_engine.refutation_runner import RefutationSuite

        suite = RefutationSuite(
            passed=True,
            confidence_score=1.0,
            tests=self._tests(),
            gate_decision=GateDecision.PROCEED,
        )
        legacy = suite.to_legacy_format()
        assert "unobserved_common_cause" not in legacy["individual_tests"]
        assert set(legacy["individual_tests"]) == {
            "placebo_treatment",
            "random_common_cause",
            "data_subset",
            "bootstrap",
        }
        assert legacy["total_tests"] == 4
        assert legacy["tests_passed"] == 4
