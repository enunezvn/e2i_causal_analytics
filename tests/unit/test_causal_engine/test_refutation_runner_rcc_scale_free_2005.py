"""#2005: the random_common_cause verdict is scale-free (shift in reported-SE units).

Before this change the runner scored ``delta_percent = |refuted − original| /
|original|`` against 20 % / 30 %. The denominator is the effect itself, so a
small true effect FAILED on perturbation noise a large effect absorbs: measured
2026-09-11, every one of the 7 live FAILED rows had |ATE| ≤ 0.047 with an
absolute shift SMALLER than the PASSED rows' (``docs/demos/results/
2026-09-11_rcc_scale_free/reband.md``). The rule under test measures the SAME
shift against the reported interval's standard error::

    se        = (ci_upper − ci_lower) / (2 × 1.959964)
    shift_se  = |refuted − original| / (se × sqrt(reference_n / refit_n))
    PASSED ≤ 1.0, WARNING ≤ 2.0, else FAILED

The ``sqrt(reference_n / refit_n)`` factor scales the reported SE to the frame
the refits actually ran on (#1419 subsamples the refutation frame to 5000 rows
while the reported interval comes from the full estimation frame); it is 1.0
when the two frames are the same or either count is unknown, and it is
persisted so a reader can undo it.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.errors import RefutationError
from src.causal_engine.refutation_runner import (
    RefutationRunner,
    RefutationStatus,
    RefutationTestType,
    _score_common_cause_shift,
)
from tests.unit.test_causal_engine.test_refutation_runner import (
    _make_refutation_result,
    _make_stub_causal_model,
    _stub_estimate,
)

Z975 = 1.959964
THRESHOLDS = {"pass": 1.0, "warning": 2.0}


def _rcc(runner: RefutationRunner, *, original_effect, original_ci, refuted, **kw):
    """Drive the runner's own rcc method through a stub refuter that returns
    ``refuted`` (the scoring branch is the code under test; the DoWhy call is
    exercised for real in ``test_real_dowhy_refit_scores_in_se_units``)."""
    stub = _make_stub_causal_model(
        {"random_common_cause": _make_refutation_result(new_effect=refuted, p_value=0.5)}
    )
    return runner._run_random_common_cause_test(
        original_effect=original_effect,
        original_ci=original_ci,
        causal_model=stub,
        identified_estimand=object(),
        estimate=_stub_estimate(value=original_effect),
        use_dowhy=True,
        **kw,
    )


# --- (g) thresholds ---------------------------------------------------------


def test_thresholds_are_in_se_units_and_the_percent_rule_is_gone():
    assert RefutationRunner.PASS_THRESHOLDS["common_cause_shift_se"] == {
        "pass": 1.0,
        "warning": 2.0,
    }
    assert "common_cause_delta" not in RefutationRunner.PASS_THRESHOLDS
    runner = RefutationRunner()
    assert runner.thresholds["common_cause_shift_se"] == {"pass": 1.0, "warning": 2.0}
    with pytest.raises(KeyError):
        runner.thresholds["common_cause_delta"]


# --- (a) / (b): the pure scoring helper ----------------------------------------


def test_small_true_effect_within_one_se_passes():
    """(a) effect 0.035, CI [−0.003, 0.074], refuted 0.054: Δ = 0.019 is 54 % of
    the effect (FAILED under the old rule) but 0.97 reported SE → PASSED."""
    status, details = _score_common_cause_shift(
        original_effect=0.035,
        refuted_effect=0.054,
        original_ci=(-0.003, 0.074),
        reference_n=None,
        refit_n=None,
        thresholds=THRESHOLDS,
    )
    assert status == RefutationStatus.PASSED
    assert details["shift_se_units"] == pytest.approx(0.9673, abs=1e-3)
    assert details["reference_se"] == pytest.approx(0.077 / (2 * Z975), rel=1e-6)


def test_large_effect_beyond_two_se_fails_although_delta_is_twenty_percent():
    """(b) effect 0.30, CI [0.25, 0.35], refuted 0.36: Δ% = 20 % (PASSED under
    the old rule) but Δ = 0.06 is 2.35 reported SE → FAILED."""
    status, details = _score_common_cause_shift(
        original_effect=0.30,
        refuted_effect=0.36,
        original_ci=(0.25, 0.35),
        reference_n=None,
        refit_n=None,
        thresholds=THRESHOLDS,
    )
    assert status == RefutationStatus.FAILED
    assert details["shift_se_units"] == pytest.approx(2.352, abs=1e-3)


def test_warning_band_between_one_and_two_se():
    status, details = _score_common_cause_shift(
        original_effect=0.30,
        refuted_effect=0.34,
        original_ci=(0.25, 0.35),
        reference_n=None,
        refit_n=None,
        thresholds=THRESHOLDS,
    )
    assert status == RefutationStatus.WARNING
    assert 1.0 < details["shift_se_units"] <= 2.0


# --- (f) n-scaling -------------------------------------------------------------


def test_reference_se_is_scaled_to_the_refit_frame():
    """(f) effect 0.039, CI [0.031, 0.047] reported on 37 515 rows, refuted 0.045
    on a 5 000-row refit frame: unscaled the shift is 1.47 SE (WARNING); against
    the refit frame's SE (× sqrt(37515/5000) ≈ 2.74) it is 0.54 SE → PASSED."""
    status, details = _score_common_cause_shift(
        original_effect=0.039,
        refuted_effect=0.045,
        original_ci=(0.031, 0.047),
        reference_n=37515,
        refit_n=5000,
        thresholds=THRESHOLDS,
    )
    assert status == RefutationStatus.PASSED
    assert details["reference_se_scale"] == pytest.approx(math.sqrt(37515 / 5000), rel=1e-9)
    assert details["reference_se_scale"] == pytest.approx(2.739, abs=1e-3)
    assert details["shift_se_units"] == pytest.approx(0.5367, abs=1e-3)
    assert details["reference_n"] == 37515
    assert details["refit_n"] == 5000
    # the scale is undoable: reference_se / scale is the reported SE
    assert details["reference_se"] / details["reference_se_scale"] == pytest.approx(
        details["reported_se"], rel=1e-12
    )
    unscaled, unscaled_details = _score_common_cause_shift(
        original_effect=0.039,
        refuted_effect=0.045,
        original_ci=(0.031, 0.047),
        reference_n=None,
        refit_n=None,
        thresholds=THRESHOLDS,
    )
    assert unscaled == RefutationStatus.WARNING
    assert unscaled_details["reference_se_scale"] == 1.0
    assert unscaled_details["shift_se_units"] == pytest.approx(1.470, abs=1e-3)


@pytest.mark.parametrize(
    "reference_n, refit_n",
    [
        (None, 5000),  # reference frame unknown
        (37515, None),  # refit frame unknown
        (5000, 5000),  # same frame
        (5000, 37515),  # refit frame LARGER than the reference: never shrink the SE
        (0, 5000),  # a computed 0 is not a usable count
        (37515, 0),  # a zero refit count is not a usable count either
    ],
)
def test_reference_se_is_not_scaled_without_both_counts_or_when_refit_is_not_smaller(
    reference_n, refit_n
):
    _status, details = _score_common_cause_shift(
        original_effect=0.039,
        refuted_effect=0.045,
        original_ci=(0.031, 0.047),
        reference_n=reference_n,
        refit_n=refit_n,
        thresholds=THRESHOLDS,
    )
    assert details["reference_se_scale"] == 1.0
    assert details["reference_se"] == pytest.approx(details["reported_se"], rel=1e-12)
    # a computed 0 is not a count: it is persisted as unknown (None)
    assert details["reference_n"] == (reference_n or None)
    assert details["refit_n"] == (refit_n or None)


# --- (c) degenerate / non-finite reference, decided BEFORE any refit -------------


def _model_that_must_not_refit() -> SimpleNamespace:
    def refute_estimate(*_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("refute_estimate must not be called: the interval decides first")

    return SimpleNamespace(refute_estimate=refute_estimate, _data=pd.DataFrame({"t": [0, 1]}))


def test_zero_width_reference_interval_skips_before_any_refit():
    """(c) A zero-width reported interval cannot score a shift in SE units. The
    verdict reuses ``_degenerate_ci_skip_result`` — the helper data_subset and
    bootstrap already use — so the reason vocabulary is ONE across the suite:
    it starts with ``original_ci_degenerate`` (the plan's draft said
    ``degenerate_reference_ci``; we keep the existing helper's word)."""
    runner = RefutationRunner()
    result = runner._run_random_common_cause_test(
        original_effect=0.10,
        original_ci=(0.10, 0.10),
        causal_model=_model_that_must_not_refit(),
        identified_estimand=object(),
        estimate=_stub_estimate(value=0.10),
        use_dowhy=True,
    )
    assert result.status == RefutationStatus.SKIPPED
    assert result.test_name == RefutationTestType.RANDOM_COMMON_CAUSE
    assert result.details["reason"].startswith("original_ci_degenerate")
    assert "shift" in result.details["reason"]  # names what could not be scored
    assert "%" not in result.details["message"]
    assert result.details["original_ci"] == (0.10, 0.10)
    assert result.refuted_effect == 0.10


def test_negative_width_reference_interval_skips_before_any_refit():
    runner = RefutationRunner()
    result = runner._run_random_common_cause_test(
        original_effect=0.10,
        original_ci=(0.12, 0.08),
        causal_model=_model_that_must_not_refit(),
        identified_estimand=object(),
        estimate=_stub_estimate(value=0.10),
        use_dowhy=True,
    )
    assert result.status == RefutationStatus.SKIPPED
    assert result.details["reason"].startswith("original_ci_degenerate")


@pytest.mark.parametrize(
    "original_ci",
    [(float("nan"), 0.2), (0.1, float("inf")), (float("-inf"), float("inf"))],
)
def test_non_finite_reference_interval_fails_closed_before_any_refit(original_ci):
    runner = RefutationRunner()
    with pytest.raises(RefutationError) as exc_info:
        runner._run_random_common_cause_test(
            original_effect=0.10,
            original_ci=original_ci,
            causal_model=_model_that_must_not_refit(),
            identified_estimand=object(),
            estimate=_stub_estimate(value=0.10),
            use_dowhy=True,
        )
    assert exc_info.value.details["reason"] == "original_ci_non_finite"
    assert exc_info.value.details["test_name"] == "random_common_cause"


def test_no_model_still_fails_closed_before_the_interval_is_read():
    """F-014 unchanged: a missing CausalModel raises, whatever the interval."""
    runner = RefutationRunner()
    with pytest.raises(RefutationError) as exc_info:
        runner._run_random_common_cause_test(
            original_effect=0.10,
            original_ci=(0.05, 0.15),
            causal_model=None,
            identified_estimand=None,
            estimate=None,
            use_dowhy=False,
        )
    assert exc_info.value.details["test_name"] == "random_common_cause"


# --- (d) / (e): the result row -------------------------------------------------------


def test_delta_percent_is_still_populated_for_the_db_column():
    """(d) ``causal_validations.delta_percent`` keeps its value: it is descriptive
    (``_describe_failure`` prints it) even though it no longer decides."""
    result = _rcc(RefutationRunner(), original_effect=0.30, original_ci=(0.25, 0.35), refuted=0.36)
    assert result.status == RefutationStatus.FAILED
    assert result.delta_percent == pytest.approx(20.0, abs=1e-9)
    assert result.refuted_effect == pytest.approx(0.36)
    assert result.p_value == 0.5


def test_details_carry_the_rule_its_inputs_and_an_se_units_message():
    """(e) A reader can recompute the verdict from the persisted details alone."""
    result = _rcc(
        RefutationRunner(),
        original_effect=0.039,
        original_ci=(0.031, 0.047),
        refuted=0.045,
        reference_n=37515,
        refit_n=5000,
    )
    d = result.details
    assert d["rule"] == "shift_vs_reported_se"
    assert d["reference_ci"] == (0.031, 0.047)
    assert d["reported_se"] == pytest.approx(0.016 / (2 * Z975), rel=1e-9)
    assert d["reference_se_scale"] == pytest.approx(math.sqrt(37515 / 5000), rel=1e-9)
    assert d["reference_se"] == pytest.approx(d["reported_se"] * d["reference_se_scale"], rel=1e-12)
    assert d["shift"] == pytest.approx(0.006, abs=1e-12)
    assert d["shift_se_units"] == pytest.approx(0.5367, abs=1e-3)
    assert d["reference_n"] == 37515
    assert d["refit_n"] == 5000
    assert d["thresholds_se"] == {"pass": 1.0, "warning": 2.0}
    assert (
        d["effect_strength"]
        == RefutationRunner.DEFAULT_CONFIG["random_common_cause"]["effect_strength"]
    )
    assert "SE" in d["message"]
    assert "%" not in d["message"]
    assert "0.54" in d["message"]
    assert result.status == RefutationStatus.PASSED


@pytest.mark.parametrize(
    "refuted, expected",
    [
        (0.041, RefutationStatus.PASSED),
        (0.045, RefutationStatus.WARNING),
        (0.050, RefutationStatus.FAILED),
    ],
)
def test_every_message_names_se_units_never_a_percentage(refuted, expected):
    result = _rcc(
        RefutationRunner(), original_effect=0.039, original_ci=(0.031, 0.047), refuted=refuted
    )
    assert result.status == expected
    assert "SE" in result.details["message"]
    assert "%" not in result.details["message"]


def test_custom_thresholds_override_the_se_cutoffs():
    runner = RefutationRunner(thresholds={"common_cause_shift_se": {"pass": 3.0, "warning": 4.0}})
    result = _rcc(runner, original_effect=0.30, original_ci=(0.25, 0.35), refuted=0.36)
    assert result.status == RefutationStatus.PASSED  # 2.35 SE ≤ 3.0
    assert result.details["thresholds_se"] == {"pass": 3.0, "warning": 4.0}


# --- run_all_tests threads the counts ----------------------------------------------


def _full_stub(rcc_refuted: float) -> SimpleNamespace:
    return _make_stub_causal_model(
        {
            "placebo_treatment_refuter": _make_refutation_result(new_effect=0.001, p_value=0.85),
            "random_common_cause": _make_refutation_result(new_effect=rcc_refuted, p_value=0.5),
        }
    )


def _rcc_of(suite):
    return next(t for t in suite.tests if t.test_name == RefutationTestType.RANDOM_COMMON_CAUSE)


def test_run_all_tests_passes_the_interval_and_both_counts_to_the_rcc_test():
    runner = RefutationRunner(
        config={
            "data_subset": {"enabled": False},
            "bootstrap": {"enabled": False},
            "sensitivity_e_value": {"enabled": False},
        }
    )
    frame = pd.DataFrame({"t": [0, 1] * 50, "y": np.linspace(0, 1, 100)})
    suite = runner.run_all_tests(
        original_effect=0.039,
        original_ci=(0.031, 0.047),
        data=frame,
        causal_model=_full_stub(0.045),
        identified_estimand=object(),
        estimate=_stub_estimate(value=0.039),
        treatment="t",
        outcome="y",
        reference_n=37515,
        refit_n=5000,
    )
    rcc = _rcc_of(suite)
    assert rcc.status == RefutationStatus.PASSED
    assert rcc.details["reference_ci"] == (0.031, 0.047)
    assert rcc.details["reference_n"] == 37515
    assert rcc.details["refit_n"] == 5000
    assert rcc.details["reference_se_scale"] == pytest.approx(math.sqrt(37515 / 5000), rel=1e-9)


def test_run_all_tests_defaults_refit_n_to_the_frame_and_reference_n_to_n_rows():
    """Without explicit counts: ``refit_n`` is the frame the runner was handed
    (the frame the refits run on) and ``reference_n`` follows ``n_rows`` (the
    full-frame count the reported effect came from), so a caller that already
    passes ``n_rows`` gets the scaling for free."""
    runner = RefutationRunner(
        config={
            "data_subset": {"enabled": False},
            "bootstrap": {"enabled": False},
            "sensitivity_e_value": {"enabled": False},
        }
    )
    frame = pd.DataFrame({"t": [0, 1] * 50, "y": np.linspace(0, 1, 100)})
    suite = runner.run_all_tests(
        original_effect=0.039,
        original_ci=(0.031, 0.047),
        data=frame,
        causal_model=_full_stub(0.045),
        identified_estimand=object(),
        estimate=_stub_estimate(value=0.039),
        treatment="t",
        outcome="y",
        n_rows=400,
    )
    rcc = _rcc_of(suite)
    assert rcc.details["refit_n"] == 100
    assert rcc.details["reference_n"] == 400
    assert rcc.details["reference_se_scale"] == pytest.approx(2.0, rel=1e-9)
    # and with neither n_rows nor data-derived reference: no scaling
    suite2 = runner.run_all_tests(
        original_effect=0.039,
        original_ci=(0.031, 0.047),
        data=frame,
        causal_model=_full_stub(0.045),
        identified_estimand=object(),
        estimate=_stub_estimate(value=0.039),
        treatment="t",
        outcome="y",
    )
    rcc2 = _rcc_of(suite2)
    assert rcc2.details["refit_n"] == 100
    assert rcc2.details["reference_n"] is None
    assert rcc2.details["reference_se_scale"] == 1.0


# --- one real DoWhy refit -------------------------------------------------------------


def _ols_ci(df: pd.DataFrame, treatment: str, outcome: str, covariate: str):
    """95 % interval of the treatment slope of ``y ~ 1 + t + c`` (classical OLS)."""
    X = np.column_stack(
        [np.ones(len(df)), df[treatment].to_numpy(float), df[covariate].to_numpy(float)]
    )
    y = df[outcome].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = float(resid @ resid) / (len(y) - X.shape[1])
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = math.sqrt(cov[1, 1])
    return float(beta[1]), (float(beta[1] - Z975 * se), float(beta[1] + Z975 * se))


def test_real_dowhy_refit_scores_in_se_units():
    """End to end on a real ``dowhy.CausalModel`` (backdoor linear regression,
    300 rows, one measured confounder, 5 simulations). A random common cause
    carries no information about the outcome, so the refit stays within the
    reported interval's noise: the shift must not FAIL, and every persisted
    field must be finite."""
    pytest.importorskip("dowhy")
    from dowhy import CausalModel

    rng = np.random.default_rng(2005)
    n = 300
    c = rng.normal(size=n)
    t = (rng.normal(size=n) + 0.8 * c > 0).astype(int)
    y = 0.5 * t + 1.0 * c + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"t": t, "y": y, "c": c})
    ate, ci = _ols_ci(df, "t", "y", "c")

    model = CausalModel(data=df, treatment="t", outcome="y", common_causes=["c"])
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
    assert float(estimate.value) == pytest.approx(ate, abs=1e-6)

    runner = RefutationRunner(config={"random_common_cause": {"num_simulations": 5}})
    result = runner._run_random_common_cause_test(
        original_effect=ate,
        original_ci=ci,
        causal_model=model,
        identified_estimand=estimand,
        estimate=estimate,
        use_dowhy=True,
        reference_n=n,
        refit_n=n,
    )
    assert result.status in (RefutationStatus.PASSED, RefutationStatus.WARNING)
    assert result.details["rule"] == "shift_vs_reported_se"
    assert result.details["reference_se_scale"] == 1.0
    assert math.isfinite(result.details["shift_se_units"])
    assert result.details["shift_se_units"] <= 2.0
    assert math.isfinite(result.delta_percent)
    assert result.p_value is not None and 0.0 <= result.p_value <= 1.0
    assert "SE" in result.details["message"] and "%" not in result.details["message"]


# --- the cutoffs are inclusive (codex round-1 LOW) -----------------------------------


@pytest.mark.parametrize(
    "refuted, expected",
    [
        (0.5, RefutationStatus.PASSED),  # exactly 1.0 SE
        (0.5000001, RefutationStatus.WARNING),  # just above 1.0 SE
        (1.0, RefutationStatus.WARNING),  # exactly 2.0 SE
        (1.0000001, RefutationStatus.FAILED),  # just above 2.0 SE
    ],
)
def test_cutoffs_are_inclusive_at_exactly_one_and_two_se(refuted, expected):
    """CI (−1.959964·0.5, +1.959964·0.5) around 0 has ``reported_se`` exactly 0.5
    (``x / (2x)`` is exact in IEEE arithmetic), so a refit at 0.5 is exactly
    1.0 SE and at 1.0 exactly 2.0 SE: both sit ON the cutoff and stay in the
    lower band."""
    half = Z975 * 0.5
    status, details = _score_common_cause_shift(
        original_effect=0.0,
        refuted_effect=refuted,
        original_ci=(-half, half),
        reference_n=None,
        refit_n=None,
        thresholds=THRESHOLDS,
    )
    assert details["reported_se"] == 0.5
    assert status == expected


# --- unusable counts never weaken the verdict (whole-diff codex F2) ---------------


@pytest.mark.parametrize(
    "reference_n, refit_n",
    [
        (float("inf"), 5000),
        (37515, float("nan")),
        (37515.5, 5000),  # fractional
        ("37515", 5000),  # string
        (True, 5000),  # bool is an int subclass
        (-37515, 5000),
        (37515, -1),
    ],
)
def test_unusable_counts_keep_scale_one_and_persist_as_unknown(reference_n, refit_n):
    """effect 0.30, CI [0.25, 0.35], refuted 0.36 is FAILED at 2.35 SE; an inf,
    NaN, fractional, string, bool or negative count must not scale it to PASSED."""
    status, details = _score_common_cause_shift(
        original_effect=0.30,
        refuted_effect=0.36,
        original_ci=(0.25, 0.35),
        reference_n=reference_n,
        refit_n=refit_n,
        thresholds=THRESHOLDS,
    )
    assert status == RefutationStatus.FAILED
    assert details["reference_se_scale"] == 1.0
    assert details["reference_n"] is None or isinstance(details["reference_n"], int)
    assert details["refit_n"] is None or isinstance(details["refit_n"], int)
    assert not (
        details["reference_n"]
        and details["refit_n"]
        and details["refit_n"] < details["reference_n"]
    )


def test_integral_float_counts_are_accepted_as_counts():
    """Counts round-trip through JSON as floats on some paths; 37515.0 is a count."""
    _status, details = _score_common_cause_shift(
        original_effect=0.039,
        refuted_effect=0.045,
        original_ci=(0.031, 0.047),
        reference_n=37515.0,
        refit_n=5000.0,
        thresholds=THRESHOLDS,
    )
    assert details["reference_n"] == 37515 and details["refit_n"] == 5000
    assert details["reference_se_scale"] == pytest.approx(math.sqrt(37515 / 5000), rel=1e-9)


# --- a SKIPPED critical test is forwarded with criticality-neutral wording (F4) ----


def test_skipped_rcc_is_forwarded_to_legacy_skipped_tests_without_non_critical_wording():
    runner = RefutationRunner(
        config={
            "data_subset": {"enabled": False},
            "bootstrap": {"enabled": False},
            "sensitivity_e_value": {"enabled": False},
        }
    )
    frame = pd.DataFrame({"t": [0, 1] * 50, "y": np.linspace(0, 1, 100)})
    suite = runner.run_all_tests(
        original_effect=0.039,
        original_ci=(0.039, 0.039),
        data=frame,
        causal_model=_full_stub(0.045),
        identified_estimand=object(),
        estimate=_stub_estimate(value=0.039),
        treatment="t",
        outcome="y",
    )
    rcc = _rcc_of(suite)
    assert rcc.status == RefutationStatus.SKIPPED
    legacy = suite.to_legacy_format()
    assert "random_common_cause" not in legacy["individual_tests"]
    forwarded = legacy["skipped_tests"]["random_common_cause"]
    assert forwarded == rcc.details["message"]
    assert "non-critical" not in forwarded
    assert "non-critical" not in rcc.details["reason"]
    assert "random_common_cause skipped" in forwarded
    # the placebo PASSED alone decides the suite: no confidence-only BLOCK
    assert suite.gate_decision.value == "proceed"
