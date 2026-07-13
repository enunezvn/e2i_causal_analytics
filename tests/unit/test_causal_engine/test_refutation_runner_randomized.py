# tests/unit/test_causal_engine/test_refutation_runner_randomized.py
"""RCT-awareness of the ``sensitivity_e_value`` gate (post-#1217 follow-up).

The E-value (VanderWeele & Ding 2017) quantifies how strong UNMEASURED
CONFOUNDING would have to be to explain away an effect — a threat model that
does not apply to a genuinely randomized treatment (assignment is exogenous by
construction). Yet ``sensitivity_e_value`` is a CRITICAL gate: an honest small
standardized effect on an RCT fails ``e_value_min`` and hard-BLOCKs the whole
estimate. Verified live on the nba_triggers RCT question: gate-block with
confidence 0.67, with AND without baseline adjustment (documented in PR #1217).

Fix contract: callers declare ``randomized_design=True`` from DESIGN knowledge
(the dataset spec — NEVER inferred from an empty discovered backdoor, which
would fail-open observational questions where discovery simply found nothing).
The runner still computes the E-value for information but returns
status=SKIPPED (excluded from confidence, never a critical failure) with the
numbers preserved in details.
"""

from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationRunner,
    RefutationStatus,
    RefutationTestType,
)
from tests.unit.test_causal_engine.test_refutation_runner import _full_stub_causal_model

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

    def test_observational_default_still_fails_weak_evalue(self):
        """Guard: without the flag, the observational gate is unchanged — a
        weak E-value still FAILS (the gate matters exactly there)."""
        runner = RefutationRunner()
        result = runner._run_sensitivity_test(
            original_effect=_WEAK_EFFECT,
            original_ci=_WEAK_CI,
            outcome_std=_OUTCOME_STD,
        )
        assert result.status == RefutationStatus.FAILED


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
            estimate=object(),
            randomized_design=True,
        )
        by_name = {t.test_name: t for t in suite.tests}
        assert by_name[RefutationTestType.SENSITIVITY_E_VALUE].status == RefutationStatus.SKIPPED
        assert suite.gate_decision == GateDecision.PROCEED

    def test_observational_run_all_tests_still_blocks(self):
        """Guard: the same suite WITHOUT the flag keeps the pre-fix behavior —
        critical e-value failure → BLOCK (observational questions keep their
        unmeasured-confounding gate)."""
        runner = RefutationRunner()
        suite = runner.run_all_tests(
            original_effect=0.15,
            original_ci=(0.10, 0.20),
            causal_model=_full_stub_causal_model(),
            identified_estimand=object(),
            estimate=object(),
        )
        by_name = {t.test_name: t for t in suite.tests}
        assert by_name[RefutationTestType.SENSITIVITY_E_VALUE].status == RefutationStatus.FAILED
        assert suite.gate_decision == GateDecision.BLOCK


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
