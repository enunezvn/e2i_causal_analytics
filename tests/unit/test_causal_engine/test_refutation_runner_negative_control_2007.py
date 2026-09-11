"""#2007: the negative-control-outcome refutation test, a weight-0 READING.

Why this test exists. None of the platform's refuters can DETECT unmeasured
confounding: placebo and random_common_cause perturb the fit, data_subset and
bootstrap resample it, and the E-value reads how strong an unmeasured
confounder WOULD have to be -- all of them are silent when the adjustment set
is missing something that drives both arms. A negative-control outcome
(Lipsitch, Tchetgen Tchetgen & Cohen 2010) is an outcome the treatment cannot
causally affect but that shares the treatment's confounders: if the SAME
adjusted fit finds a non-null effect on it, the adjustment is leaking
confounding into the primary estimate.

Measured on the synthetic generator 2026-09-11 (``docs/demos/results/
2026-09-11_negative_control_disproof/disproof.md``, seed 21, n = 1500):
omitting an arm's declared confounders moves 3 of 9 structural nulls out of
their CI (copay_support -> treatment_initiated +0.017 [-0.031, +0.065] under
adjustment becomes +0.052 [+0.001, +0.103] without it); adjusted fits give 0/9
false positives on the nulls and detect 11/11 planted truths. The three bands
below use exactly those numbers against the claimed effect +0.121
(copay_support -> adherent_180d, adjusted).

The rule (``RefutationRunner._run_negative_control_test``)::

    PASSED   nc_lo <= 0 <= nc_hi                 (the control stayed null)
    WARNING  CI excludes 0 and |nc| <  |original| (moved, less than the claim)
    FAILED   CI excludes 0 and |nc| >= |original| (moved at least as much)
    SKIPPED  non-finite or inverted CI / effect  (negative_control_ci_unavailable)

Weight 0 for the first live period: the test is non-critical and carries no
confidence weight, so the gate and the score are IDENTICAL with and without
the row (pinned below for every band). Promotion to a weighted or critical
test is an owner decision after live counts -- a WARNING a leader reads must
be rare and concrete, and no live count exists yet. The counts a suite reports
(``tests_passed`` / ``total_tests``) DO include the reading, like the E-value.
"""

from __future__ import annotations

import math
import time as _t

import pytest

from src.causal_engine.refutation_runner import (
    NEGATIVE_CONTROL_SKIP_REASONS,
    GateDecision,
    RefutationResult,
    RefutationRunner,
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
)
from tests.unit.test_causal_engine.test_refutation_runner import (
    _full_stub_causal_model,
    _stub_estimate,
)

NC = RefutationTestType.NEGATIVE_CONTROL_OUTCOME
P, W, F, S = (
    RefutationStatus.PASSED,
    RefutationStatus.WARNING,
    RefutationStatus.FAILED,
    RefutationStatus.SKIPPED,
)

# copay_support -> adherent_180d, adjusted (disproof.md, truth +0.116)
ORIGINAL = 0.121
NC_OUTCOME = "treatment_initiated"
# copay_support -> treatment_initiated, ADJUSTED fit: the control stayed null
PASSED_NC = (NC_OUTCOME, 0.017, (-0.031, 0.065), 1500)
# the same pair with the confounders OMITTED: the control moved, less than the claim
WARNING_NC = (NC_OUTCOME, 0.052, (0.001, 0.103), 1500)
# a control that moved at least as much as the claimed effect
FAILED_NC = (NC_OUTCOME, 0.130, (0.08, 0.18), 1500)


def _run(runner: RefutationRunner, **overrides) -> RefutationSuite:
    kwargs = {
        "original_effect": 0.15,
        "original_ci": (0.10, 0.20),
        "causal_model": _full_stub_causal_model(),
        "identified_estimand": object(),
        "estimate": _stub_estimate(),
    }
    kwargs.update(overrides)
    return runner.run_all_tests(**kwargs)


def _by_name(suite: RefutationSuite) -> dict:
    return {t.test_name: t for t in suite.tests}


def _other_tests(placebo=P, rcc=P, sens=P, subset=P, boot=P) -> list:
    """The five existing tests as bare results (the scoring code reads only
    ``test_name`` and ``status``)."""
    mk = lambda n, s: RefutationResult(n, s, ORIGINAL, ORIGINAL)  # noqa: E731
    return [
        mk(RefutationTestType.PLACEBO_TREATMENT, placebo),
        mk(RefutationTestType.RANDOM_COMMON_CAUSE, rcc),
        mk(RefutationTestType.SENSITIVITY_E_VALUE, sens),
        mk(RefutationTestType.DATA_SUBSET, subset),
        mk(RefutationTestType.BOOTSTRAP, boot),
    ]


# --- (1)/(2) enum member and config entry ------------------------------------


def test_enum_member_and_default_config_entry():
    assert NC.value == "negative_control_outcome"
    assert RefutationRunner.DEFAULT_CONFIG["negative_control_outcome"] == {
        "enabled": True,
        "critical": False,
    }
    runner = RefutationRunner()
    assert runner.config["negative_control_outcome"]["critical"] is False
    # A custom config still merges per key onto the default (the #606 path).
    runner = RefutationRunner(config={"negative_control_outcome": {"enabled": False}})
    assert runner.config["negative_control_outcome"] == {"enabled": False, "critical": False}


# --- (a) the three bands with the disproof's numbers -----------------------------


def test_passed_when_the_control_ci_includes_zero():
    result = RefutationRunner()._run_negative_control_test(ORIGINAL, PASSED_NC)
    assert result.test_name == NC
    assert result.status == P
    assert result.original_effect == ORIGINAL
    assert result.refuted_effect == 0.017
    assert result.p_value is None
    assert result.delta_percent == pytest.approx(100 * 0.017 / 0.121)


def test_warning_when_the_control_moved_less_than_the_claimed_effect():
    result = RefutationRunner()._run_negative_control_test(ORIGINAL, WARNING_NC)
    assert result.status == W
    assert result.refuted_effect == 0.052
    assert result.delta_percent == pytest.approx(100 * 0.052 / 0.121)
    assert "less than the claimed effect" in result.details["reading"]


def test_failed_when_the_control_moved_at_least_as_much_as_the_claimed_effect():
    result = RefutationRunner()._run_negative_control_test(ORIGINAL, FAILED_NC)
    assert result.status == F
    assert result.refuted_effect == 0.130
    # delta_percent IS the FAILED ratio (>= 100 means FAILED once the CI excludes 0)
    assert result.delta_percent == pytest.approx(100 * 0.130 / 0.121)
    assert result.delta_percent >= 100.0
    assert "at least as much as the claimed effect" in result.details["reading"]


def test_band_boundaries_are_inclusive_where_the_rule_says_so():
    runner = RefutationRunner()
    # A CI endpoint exactly at zero still INCLUDES zero -> PASSED.
    assert runner._run_negative_control_test(ORIGINAL, ("nc", 0.02, (0.0, 0.04), 100)).status == P
    assert runner._run_negative_control_test(ORIGINAL, ("nc", -0.02, (-0.04, 0.0), 100)).status == P
    # |nc| == |original| with the CI excluding zero -> FAILED ("at least as much").
    assert runner._run_negative_control_test(ORIGINAL, ("nc", 0.121, (0.07, 0.17), 100)).status == F
    # Sign does not matter: a negative control moving the other way is the same evidence.
    assert (
        runner._run_negative_control_test(ORIGINAL, ("nc", -0.13, (-0.18, -0.08), 100)).status == F
    )
    assert runner._run_negative_control_test(-ORIGINAL, ("nc", 0.05, (0.01, 0.09), 100)).status == W


def test_zero_original_effect_gives_delta_percent_zero_not_a_division_error():
    """A claimed effect of exactly 0 has no ratio; the column carries 0.0 and the
    verdict is still decided by the CI rule (a control that moved at all moved
    'at least as much' as a zero claim)."""
    result = RefutationRunner()._run_negative_control_test(0.0, WARNING_NC)
    assert result.status == F
    assert result.delta_percent == 0.0
    assert math.isfinite(result.delta_percent)


# --- (b) SKIPPED reasons ---------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        (NC_OUTCOME, float("nan"), (-0.03, 0.06), 1500),
        (NC_OUTCOME, float("inf"), (-0.03, 0.06), 1500),
        (NC_OUTCOME, 0.017, (float("nan"), 0.06), 1500),
        (NC_OUTCOME, 0.017, (-0.03, float("inf")), 1500),
        (NC_OUTCOME, 0.017, (float("-inf"), float("inf")), 1500),
        (NC_OUTCOME, 0.017, (0.06, -0.03), 1500),  # inverted: lo > hi
    ],
)
def test_unusable_control_ci_is_an_honest_skipped(bad):
    result = RefutationRunner()._run_negative_control_test(ORIGINAL, bad)
    assert result.status == S
    assert result.details["skip_reason"] == "negative_control_ci_unavailable"
    assert result.details["reason"].startswith("negative_control_ci_unavailable")
    assert result.details["message"].startswith("negative_control_outcome skipped:")
    # Never a placeholder number: the skipped row carries the ORIGINAL effect
    # (the shape every other SKIPPED helper uses) and no finite fake interval.
    assert result.refuted_effect == ORIGINAL
    assert result.p_value is None
    assert result.delta_percent == 0.0
    assert result.details["nc_outcome"] == NC_OUTCOME
    assert result.details["nc_n"] == 1500
    assert result.details["nc_effect"] is None
    assert result.details["nc_ci"] is None
    # What was received is kept as repr strings (NaN / inf are not JSONB-safe floats).
    assert isinstance(result.details["received"]["nc_effect"], str)
    assert all(isinstance(v, str) for v in result.details["received"]["nc_ci"])
    assert result.details["weight"] == 0.0
    assert result.details["critical"] is False


def test_run_all_tests_without_a_declared_control_emits_the_skipped_reason():
    """``negative_control=None`` on an enabled test is a persisted SKIPPED row
    (reason ``no_negative_control_declared``), not silence: every non-agent
    caller and every undeclared treatment lands here, and the reason must reach
    ``skipped_tests``."""
    suite = _run(RefutationRunner())
    nc = _by_name(suite)[NC]
    assert nc.status == S
    assert nc.details["skip_reason"] == "no_negative_control_declared"
    assert nc.details["reason"].startswith("no_negative_control_declared")
    assert "cannot run" in nc.details["message"]
    legacy = suite.to_legacy_format()
    assert legacy["skipped_tests"]["negative_control_outcome"] == nc.details["message"]
    assert "negative_control_outcome" not in legacy["individual_tests"]


@pytest.mark.parametrize(
    "reason",
    [
        "negative_control_column_missing",
        "negative_control_too_few_rows",
        "negative_control_ci_unavailable",
        "negative_control_budget_exhausted",
    ],
)
def test_caller_signalled_skip_reason_passes_through(reason):
    """The node decides WHY it could not produce the tuple; the runner is the one
    place that emits the SKIPPED row, with THAT reason."""
    suite = _run(RefutationRunner(), negative_control_skip_reason=reason)
    nc = _by_name(suite)[NC]
    assert nc.status == S
    assert nc.details["skip_reason"] == reason
    assert nc.details["reason"].startswith(reason)
    assert suite.to_legacy_format()["skipped_tests"]["negative_control_outcome"].startswith(
        "negative_control_outcome skipped:"
    )


def test_skip_reason_vocabulary_is_closed():
    assert NEGATIVE_CONTROL_SKIP_REASONS == frozenset(
        {
            "no_negative_control_declared",
            "negative_control_column_missing",
            "negative_control_too_few_rows",
            "negative_control_ci_unavailable",
            "negative_control_reference_effect_non_finite",
            "negative_control_budget_exhausted",
        }
    )
    with pytest.raises(ValueError, match="negative_control_skip_reason"):
        _run(RefutationRunner(), negative_control_skip_reason="because")


@pytest.mark.parametrize(
    "reason", ["negative_control_too_few_rows", "negative_control_ci_unavailable"]
)
def test_a_caller_skip_reason_wins_over_a_tuple(caplog, reason):
    """Codex round 1 (MED): a tuple beside an explicit caller reason is a
    contradiction and NEITHER input is authoritative -- the caller saw
    something the tuple hides (a fit on too few rows, an interval the backend
    could not vouch for). The runner must not score a PASSED from numbers the
    caller itself disowned: it emits the caller's SKIPPED reason and logs that
    the tuple was discarded."""
    with caplog.at_level("WARNING", logger="src.causal_engine.refutation_runner"):
        suite = _run(
            RefutationRunner(),
            negative_control=PASSED_NC,
            negative_control_skip_reason=reason,
        )
    nc = _by_name(suite)[NC]
    assert nc.status == S
    assert nc.details["skip_reason"] == reason
    assert nc.details["nc_effect"] is None and nc.details["nc_ci"] is None
    assert any(reason in r.getMessage() and "discard" in r.getMessage() for r in caplog.records)
    assert "negative_control_outcome" in suite.to_legacy_format()["skipped_tests"]


@pytest.mark.parametrize("bad_n", [None, 0, -1, 2.5, float("nan"), "12"])
def test_invalid_or_zero_row_count_is_skipped_too_few_rows(bad_n):
    """Codex round 1 (MED): a count that is not a finite positive integer means
    the fit's row basis is unknown or empty; a zero-row tuple with a
    zero-containing interval must not read PASSED."""
    result = RefutationRunner()._run_negative_control_test(
        ORIGINAL, (NC_OUTCOME, 0.017, (-0.031, 0.065), bad_n)
    )
    assert result.status == S
    assert result.details["skip_reason"] == "negative_control_too_few_rows"
    assert result.details["reason"].startswith("negative_control_too_few_rows")
    assert result.details["nc_n"] is None
    assert result.details["nc_effect"] is None and result.details["nc_ci"] is None
    assert result.details["received"]["nc_n"] == repr(bad_n)
    assert result.refuted_effect == ORIGINAL
    assert result.delta_percent == 0.0


@pytest.mark.parametrize("bad_claim", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_claimed_effect_is_skipped_with_its_own_reason(bad_claim):
    """Codex round 1 (MED): a NaN / infinite claimed effect cannot be compared
    to anything; scoring it would fabricate PASSED and delta_percent = 0.0.
    The reason names the reference effect, not the control's interval."""
    result = RefutationRunner()._run_negative_control_test(bad_claim, PASSED_NC)
    assert result.status == S
    assert result.details["skip_reason"] == "negative_control_reference_effect_non_finite"
    assert "claimed effect is not a finite number" in result.details["reason"]
    assert result.details["received"]["original_effect"] == repr(bad_claim)
    assert result.details["nc_effect"] is None and result.details["nc_ci"] is None
    assert result.details["nc_outcome"] == NC_OUTCOME
    assert result.delta_percent == 0.0


def test_disabled_test_emits_no_row_at_all():
    runner = RefutationRunner(config={"negative_control_outcome": {"enabled": False}})
    suite = _run(runner, negative_control=PASSED_NC)
    assert NC not in _by_name(suite)
    assert "negative_control_outcome" not in suite.to_legacy_format()["skipped_tests"]


def test_run_all_tests_scores_a_declared_control_last_and_without_refit_budget(monkeypatch):
    """The comparison is arithmetic on values the caller already computed: it
    runs even when the compute deadline has passed (the two non-critical
    resample tests are budget-skipped at that point), it never feeds the
    observed per-refit average, and it is the last row so the #1419
    critical-first order of the five existing tests is unchanged."""
    runner = RefutationRunner(
        config={
            "placebo_treatment": {"num_simulations": 10},
            "random_common_cause": {"num_simulations": 10},
            "data_subset": {"num_subsets": 10},
            "bootstrap": {"num_bootstraps": 10},
        }
    )
    clock = {"now": 1000.0}
    monkeypatch.setattr(_t, "monotonic", lambda: clock["now"])
    order: list = []
    real = runner._run_test_with_tracing

    def spy(**kwargs):
        order.append(kwargs["test_name"])
        clock["now"] += 30.0  # every traced test costs 30 s on this clock
        return real(**kwargs)

    monkeypatch.setattr(runner, "_run_test_with_tracing", spy)
    # sensitivity (t=1030), placebo (1060), rcc: 1060 + 3*10 <= 1090 runs (1090) ->
    # data_subset and bootstrap: now >= deadline -> budget SKIPPED. The negative
    # control is not budget-gated and still runs at t=1090 >= deadline.
    suite = _run(runner, deadline=1090.0, negative_control=PASSED_NC)
    assert order == [
        "sensitivity_e_value",
        "placebo_treatment",
        "random_common_cause",
        "negative_control_outcome",
    ]
    by = _by_name(suite)
    assert by[RefutationTestType.DATA_SUBSET].status == S
    assert by[RefutationTestType.BOOTSTRAP].status == S
    assert by[NC].status == P
    names = [t.test_name for t in suite.tests]
    # The five existing tests keep their order; the budget-skipped rows are
    # appended after the reading, exactly as before for the other tests.
    assert names.index(NC) > names.index(RefutationTestType.RANDOM_COMMON_CAUSE)


# --- (c) details keys, delta_percent, reading / message -----------------------------


def test_details_carry_the_persisted_keys_and_the_one_sentence_reading():
    result = RefutationRunner()._run_negative_control_test(ORIGINAL, PASSED_NC)
    d = result.details
    assert d["nc_outcome"] == NC_OUTCOME
    assert d["nc_effect"] == 0.017
    assert d["nc_ci"] == [-0.031, 0.065]
    assert all(isinstance(v, float) for v in d["nc_ci"])
    assert d["nc_n"] == 1500
    assert d["rule"] == "negative_control_ci_vs_zero"
    assert d["weight"] == 0.0
    assert d["critical"] is False
    assert d["reading"] == (
        "A negative-control outcome the treatment cannot affect (treatment_initiated) "
        "stayed null: +0.017 [-0.031, +0.065] on n = 1500."
    )
    # ``message`` is the legacy ``details`` string the interpretation node prints.
    assert d["message"] == d["reading"]
    assert d["reading"].count(".") >= 1 and "\n" not in d["reading"]


def test_reading_sentences_for_the_moved_bands():
    runner = RefutationRunner()
    assert runner._run_negative_control_test(ORIGINAL, WARNING_NC).details["reading"] == (
        "A negative-control outcome the treatment cannot affect (treatment_initiated) "
        "moved by +0.052 [+0.001, +0.103] on n = 1500, less than the claimed effect +0.121."
    )
    assert runner._run_negative_control_test(ORIGINAL, FAILED_NC).details["reading"] == (
        "A negative-control outcome the treatment cannot affect (treatment_initiated) "
        "moved by +0.130 [+0.080, +0.180] on n = 1500, at least as much as the claimed "
        "effect +0.121: the adjustment is leaking confounding."
    )


# --- (d)/(e) weight 0: score and gate identical with and without the row ----------------


@pytest.mark.parametrize("nc_status", [P, W, F, S])
@pytest.mark.parametrize(
    "background",
    [
        {},  # all PASSED -> 1.0 PROCEED
        {"sens": W},  # 0.90 PROCEED (the null-finding run)
        {"rcc": W, "subset": F, "boot": F},  # 0.5333 REVIEW
        {"placebo": W, "rcc": S, "sens": S, "subset": F, "boot": F},  # 0.30 BLOCK by confidence
        {"placebo": F},  # BLOCK by a critical FAILED
        {"rcc": F, "sens": S},  # BLOCK by a critical FAILED
    ],
)
def test_confidence_and_gate_are_identical_with_and_without_the_row(background, nc_status):
    runner = RefutationRunner()
    without = _other_tests(**background)
    with_row = without + [RefutationResult(NC, nc_status, ORIGINAL, 0.13)]
    conf_without = runner._calculate_confidence_score(without)
    conf_with = runner._calculate_confidence_score(with_row)
    assert conf_with == conf_without
    assert runner._determine_gate_decision(with_row, conf_with) == runner._determine_gate_decision(
        without, conf_without
    )


def test_confidence_weight_is_explicitly_zero_not_the_unlisted_default():
    """``_calculate_confidence`` defaults an UNLISTED test to 0.1: a FAILED
    negative control under that default would pull an all-PASSED suite from
    1.0 to 0.909. Pinned at exactly 1.0 with the FAILED row."""
    runner = RefutationRunner()
    tests = _other_tests() + [RefutationResult(NC, F, ORIGINAL, 0.13)]
    assert runner._calculate_confidence_score(tests) == 1.0
    # And the row alone carries no evidence: all-SKIPPED-elsewhere fails closed as before.
    alone = [RefutationResult(NC, P, ORIGINAL, 0.017)]
    assert runner._calculate_confidence_score(alone) == 0.0


def test_a_failed_negative_control_never_blocks():
    runner = RefutationRunner()
    tests = _other_tests() + [RefutationResult(NC, F, ORIGINAL, 0.13)]
    conf = runner._calculate_confidence_score(tests)
    assert runner._determine_gate_decision(tests, conf) == GateDecision.PROCEED
    # Through run_all_tests too, against the disproof's claimed effect (the stub
    # suite's default 0.15 would read the 0.130 control as WARNING): the suite
    # PROCEEDs at 0.90 with a FAILED control.
    suite = _run(runner, original_effect=ORIGINAL, negative_control=FAILED_NC)
    assert _by_name(suite)[NC].status == F
    assert suite.gate_decision == GateDecision.PROCEED
    assert suite.confidence_score == pytest.approx(0.90)
    assert suite.passed is True


# --- (f) legacy format ---------------------------------------------------------------


def test_legacy_format_scored_row_lands_in_individual_tests_with_the_reading():
    runner = RefutationRunner()
    result = runner._run_negative_control_test(ORIGINAL, WARNING_NC)
    suite = RefutationSuite(
        passed=True,
        confidence_score=1.0,
        tests=_other_tests() + [result],
        gate_decision=GateDecision.PROCEED,
    )
    legacy = suite.to_legacy_format()
    row = legacy["individual_tests"]["negative_control_outcome"]
    assert row["test_name"] == "negative_control_outcome"
    assert row["status"] == "warning"
    assert row["passed"] is False
    assert row["new_effect"] == 0.052
    assert row["original_effect"] == ORIGINAL
    assert row["p_value"] == 0.0  # None -> 0.0, the legacy shape
    assert row["details"] == result.details["reading"]
    assert "negative_control_outcome" not in legacy["skipped_tests"]


def test_legacy_counts_include_the_reading_but_the_score_does_not():
    """``tests_passed`` / ``total_tests`` count the reading like the E-value;
    ``confidence_adjustment`` and ``overall_robust`` do not move."""
    runner = RefutationRunner()
    base = _other_tests()
    without = RefutationSuite(True, 1.0, base, GateDecision.PROCEED).to_legacy_format()
    for status, expected_passed in ((P, 6), (W, 5), (F, 5)):
        tests = base + [RefutationResult(NC, status, ORIGINAL, 0.05)]
        conf = runner._calculate_confidence_score(tests)
        gate = runner._determine_gate_decision(tests, conf)
        legacy = RefutationSuite(gate != GateDecision.BLOCK, conf, tests, gate).to_legacy_format()
        assert legacy["total_tests"] == 6
        assert legacy["tests_passed"] == expected_passed
        assert legacy["tests_failed"] == (1 if status == F else 0)
        assert legacy["confidence_adjustment"] == without["confidence_adjustment"] == 1.0
        assert legacy["overall_robust"] is without["overall_robust"] is True
        assert legacy["gate_decision"] == without["gate_decision"] == "proceed"
    skipped = base + [RefutationResult(NC, S, ORIGINAL, ORIGINAL, details={"message": "m"})]
    legacy = RefutationSuite(True, 1.0, skipped, GateDecision.PROCEED).to_legacy_format()
    assert legacy["total_tests"] == 5
    assert legacy["skipped_tests"] == {"negative_control_outcome": "m"}
