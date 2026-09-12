"""#2022: ``sensitivity_analyzer`` derives its benchmark inputs from the frame.

One composed answer used to state TWO E-values for ONE estimate. ``sensitivity_analyzer``
took ``naive_ate`` and ``baseline_risk`` as planner-bound parameters, but no composable
tool output carries either (``EffectEstimate`` has neither field), so the planner could
only invent them — observed live 2026-09-11 on the Kisqali cohort as ``baseline_risk=0.5``
on a continuous outcome and ``naive_ate`` = the ADJUSTED ate, which forces the
measured-confounding benchmark to exactly 1.00. The refutation suite, on the same
estimate, derived the same quantities from the frame and reported a different number.

These tests pin the fix: with a frame in context the tool derives the naive contrast, the
baseline risk (binary outcome only) and the outcome SD with the SAME ``evalue`` helpers
``RefutationRunner`` uses, so the two engines state ONE number for one estimate.

Real tools, real DoWhy refutation suite, real frames; no mocks. The expected E-values are
computed from the frame with an INDEPENDENT reference implementation (the published
VanderWeele-Ding formulas), not by calling the helper under test.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr
from src.agents.tool_composer.errors import ToolRefusalError

SMD_TO_LOG_RR = 0.91  # Chinn (2000) / VanderWeele-Ding; the published constant.


def _continuous_frame(n: int = 1200, seed: int = 2022) -> pd.DataFrame:
    """Confounded frame with a CONTINUOUS outcome (the live Kisqali shape).

    Confounding is modest relative to the effect, so the reading is
    ``beyond_measured_confounding`` — the case whose shared message quotes the point
    E-value, the CI-bound E-value AND the benchmark, which is what lets the cross-engine
    test below compare all three against the refutation suite's own prose.
    """
    rng = np.random.default_rng(seed)
    severity = rng.normal(0.0, 1.0, n)
    academic = (rng.random(n) < 0.4).astype(int)
    t = (0.9 * severity + 0.5 * academic + rng.normal(0.0, 1.0, n) > 0.3).astype(int)
    y = 0.30 * t + 0.08 * severity + 0.05 * academic + rng.normal(0.0, 0.15, n)
    return pd.DataFrame(
        {
            "treatment_arm": t,
            "adherence_rate": y,
            "disease_severity": severity,
            "academic_hcp": academic,
        }
    )


def _binary_frame(n: int = 1200, seed: int = 22) -> pd.DataFrame:
    """Confounded frame with a BINARY outcome (the risk-ratio path)."""
    rng = np.random.default_rng(seed)
    severity = rng.normal(0.0, 1.0, n)
    t = (0.8 * severity + rng.normal(0.0, 1.0, n) > 0.0).astype(int)
    p = 1.0 / (1.0 + np.exp(-(-0.4 + 0.7 * t + 0.5 * severity)))
    return pd.DataFrame(
        {
            "treatment_arm": t,
            "responded": (rng.random(n) < p).astype(int),
            "disease_severity": severity,
        }
    )


def _reference_e_value_smd(effect: float, outcome_std: float) -> float:
    """Independent VanderWeele-Ding E-value for a standardized mean difference."""
    rr = math.exp(SMD_TO_LOG_RR * abs(effect) / outcome_std)
    rr = rr if rr >= 1.0 else 1.0 / rr
    return rr + math.sqrt(rr) * math.sqrt(rr - 1.0) if rr > 1.0 else 1.0


def _estimate(df: pd.DataFrame, treatment: str, outcome: str, confounders: list[str]) -> dict:
    return tr.causal_effect_estimator(
        treatment=treatment, outcome=outcome, confounders=confounders, estimation_data=df
    ).model_dump()


# ---------------------------------------------------------------------------
# 1. The defect itself: two engines, one estimate, one number.
# ---------------------------------------------------------------------------
def test_sensitivity_and_refutation_state_one_e_value_for_one_estimate() -> None:
    df = _continuous_frame()
    est = _estimate(df, "treatment_arm", "adherence_rate", ["disease_severity", "academic_hcp"])
    assert est["ci_lower"] is not None

    report = tr.sensitivity_analyzer(
        ate=est["ate"],
        ci_lower=est["ci_lower"],
        ci_upper=est["ci_upper"],
        treatment="treatment_arm",
        outcome="adherence_rate",
        confounders=["disease_severity", "academic_hcp"],
        estimation_data=df,
    )

    # Independent reference: the SD of the rows the estimate came from.
    sd = float(np.std(np.asarray(df["adherence_rate"], dtype=float)))
    bound = min(abs(est["ci_lower"]), abs(est["ci_upper"]))
    assert report["e_value_point"] == pytest.approx(
        _reference_e_value_smd(est["ate"], sd), rel=1e-9
    )
    assert report["e_value_ci"] == pytest.approx(_reference_e_value_smd(bound, sd), rel=1e-9)
    assert report["conversion"] == "standardized_difference"

    # The benchmark is the REAL naive-vs-adjusted contrast, never the 1.00 an
    # invented ``naive_ate = ate`` produced.
    t = np.asarray(df["treatment_arm"], dtype=float)
    y = np.asarray(df["adherence_rate"], dtype=float)
    naive = float(y[t == 1].mean() - y[t == 0].mean())
    assert naive != pytest.approx(est["ate"], rel=1e-6)
    rr_naive = math.exp(SMD_TO_LOG_RR * abs(naive) / sd)
    rr_adj = math.exp(SMD_TO_LOG_RR * abs(est["ate"]) / sd)
    expected_benchmark = max(rr_naive / rr_adj, rr_adj / rr_naive)
    assert report["benchmark"] == pytest.approx(expected_benchmark, rel=1e-9)
    assert report["benchmark_basis"] == "joint_naive_vs_adjusted"

    # And the refutation suite, on the SAME estimate, states the SAME numbers.
    suite = tr.refutation_runner(
        estimate_id="one-number",
        treatment="treatment_arm",
        outcome="adherence_rate",
        confounders=["disease_severity", "academic_hcp"],
        estimation_data=df,
    )
    # Pinned so the prose comparison below cannot go vacuous: only the beyond/within
    # messages quote all three numbers, and only ``beyond`` quotes the CI bound.
    assert report["reading"] == "beyond_measured_confounding"
    prose = suite["refutation_results"]["individual_tests"]["unobserved_common_cause"]["details"]
    assert f"{report['e_value_point']:.2f}" in prose
    assert f"{report['e_value_ci']:.2f}" in prose
    assert f"{report['benchmark']:.2f}" in prose


# ---------------------------------------------------------------------------
# 2. The two invented inputs are gone: derived, or refused.
# ---------------------------------------------------------------------------
def test_a_continuous_outcome_yields_no_baseline_risk_and_a_supplied_one_is_refused() -> None:
    df = _continuous_frame()
    est = _estimate(df, "treatment_arm", "adherence_rate", ["disease_severity"])

    derived = tr.sensitivity_analyzer(
        ate=est["ate"],
        ci_lower=est["ci_lower"],
        ci_upper=est["ci_upper"],
        treatment="treatment_arm",
        outcome="adherence_rate",
        confounders=["disease_severity"],
        estimation_data=df,
    )
    assert derived["conversion"] == "standardized_difference"

    with pytest.raises(ToolRefusalError, match="baseline_risk"):
        tr.sensitivity_analyzer(
            ate=est["ate"],
            ci_lower=est["ci_lower"],
            ci_upper=est["ci_upper"],
            treatment="treatment_arm",
            outcome="adherence_rate",
            estimation_data=df,
            baseline_risk=0.5,
        )


def test_a_supplied_naive_ate_is_refused_because_nothing_emits_one() -> None:
    df = _continuous_frame()
    with pytest.raises(ToolRefusalError, match="naive_ate"):
        tr.sensitivity_analyzer(
            ate=0.11,
            ci_lower=0.10,
            ci_upper=0.12,
            treatment="treatment_arm",
            outcome="adherence_rate",
            estimation_data=df,
            naive_ate=0.11,
        )
    # Refused with no frame too: no tool output carries it there either.
    with pytest.raises(ToolRefusalError, match="naive_ate"):
        tr.sensitivity_analyzer(ate=0.11, ci_lower=0.10, ci_upper=0.12, naive_ate=0.11)


def test_a_binary_outcome_derives_the_control_arm_rate_and_takes_the_risk_ratio_path() -> None:
    df = _binary_frame()
    est = _estimate(df, "treatment_arm", "responded", ["disease_severity"])

    report = tr.sensitivity_analyzer(
        ate=est["ate"],
        ci_lower=est["ci_lower"],
        ci_upper=est["ci_upper"],
        treatment="treatment_arm",
        outcome="responded",
        confounders=["disease_severity"],
        estimation_data=df,
    )

    t = np.asarray(df["treatment_arm"], dtype=float)
    y = np.asarray(df["responded"], dtype=float)
    p0 = float(y[t == 0].mean())
    rr = (p0 + est["ate"]) / p0
    rr = rr if rr >= 1.0 else 1.0 / rr
    assert report["conversion"] == "risk_ratio"
    assert report["e_value_point"] == pytest.approx(
        rr + math.sqrt(rr) * math.sqrt(rr - 1.0), rel=1e-9
    )


# ---------------------------------------------------------------------------
# 3. Fail-closed: a frame the tool cannot read is refused, never guessed around.
# ---------------------------------------------------------------------------
def test_a_frame_without_the_bound_columns_is_refused() -> None:
    df = _continuous_frame()
    with pytest.raises(ToolRefusalError, match="not in the DataFrame"):
        tr.sensitivity_analyzer(
            ate=0.11,
            ci_lower=0.10,
            ci_upper=0.12,
            treatment="treatment_arm",
            outcome="no_such_column",
            estimation_data=df,
        )


def test_a_frame_in_context_without_a_bound_treatment_or_outcome_is_refused() -> None:
    # The frame is right there in kwargs; serving an unstandardized, unbenchmarked
    # E-value from it would be the plausible-wrong number this issue is about.
    df = _continuous_frame()
    with pytest.raises(ToolRefusalError, match="treatment"):
        tr.sensitivity_analyzer(ate=0.11, ci_lower=0.10, ci_upper=0.12, estimation_data=df)


def test_without_a_frame_the_calculation_is_refused_because_the_effect_has_no_scale() -> None:
    """No frame means no outcome SD, and without one the E-value is uninterpretable.

    ``evalue`` then reads the RAW effect as though it were already a standardized
    difference, so the number moves with the outcome's UNITS. Measured on the merge-base
    ``2b43ee85e`` (this predates the derivation fix): the same effect expressed as a
    proportion and as percentage points gave ``e_value_point`` 1.4183 and 17910.0854 —
    four orders of magnitude apart, with nothing in the answer to contradict either.
    A caveat on an unsupported number is a labeling fix; the calculation is refused.
    """
    for ate, lo, hi in ((0.1, 0.05, 0.15), (10.0, 5.0, 15.0)):
        with pytest.raises(ToolRefusalError, match="standard deviation"):
            tr.sensitivity_analyzer(ate=ate, ci_lower=lo, ci_upper=hi)
    # Point-only reporting survives — but only when a usable frame supplies the scale.
    with pytest.raises(ToolRefusalError, match="standard deviation"):
        tr.sensitivity_analyzer(ate=0.3, ci_lower=None)


def test_a_supplied_but_unusable_frame_is_refused_rather_than_read_as_absent() -> None:
    # ``_extract_dataframe_from_kwargs`` returns None for a non-DataFrame, so a malformed
    # value would otherwise take the frame-absent path and report "no frame was supplied"
    # about data the caller DID supply. Absent and supplied-but-unusable are different.
    for junk in ({"t": [0, 1], "y": [0, 1]}, "estimation_data", 42):
        with pytest.raises(ToolRefusalError, match="not a DataFrame"):
            tr.sensitivity_analyzer(
                ate=0.11,
                ci_lower=0.10,
                ci_upper=0.12,
                treatment="treatment_arm",
                outcome="adherence_rate",
                estimation_data=junk,
            )


def _unscoreable_covariate_frame(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Continuous treatment (no naive contrast) + a CONSTANT confounder (unscoreable).

    So the benchmark has no joint basis and no covariate factor, while a covariate WAS
    measured — the ``measured_unscoreable`` state.
    """
    rng = np.random.default_rng(seed)
    dose = rng.normal(0.0, 1.0, n)
    return pd.DataFrame(
        {"dose": dose, "response": 0.3 * dose + rng.normal(0.0, 1.0, n), "constant_cov": 1.0}
    )


def test_point_only_keeps_the_measured_but_unscoreable_benchmark_basis() -> None:
    # Whether the estimate carried sampling uncertainty must not change whether
    # confounders were MEASURED. ``classify`` applies the covariates_measured
    # correction; the point-only path must apply the same one.
    frame = _unscoreable_covariate_frame()
    bound = {
        "treatment": "dose",
        "outcome": "response",
        "confounders": ["constant_cov"],
        "estimation_data": frame,
    }
    with_interval = tr.sensitivity_analyzer(ate=0.3, ci_lower=0.2, ci_upper=0.4, **bound)
    point_only = tr.sensitivity_analyzer(ate=0.3, ci_lower=None, **bound)

    assert with_interval["benchmark"] is None and point_only["benchmark"] is None
    assert with_interval["benchmark_basis"] == "measured_unscoreable"
    assert point_only["benchmark_basis"] == with_interval["benchmark_basis"]


# ---------------------------------------------------------------------------
# 4. The planning contract no longer offers the invented inputs.
# ---------------------------------------------------------------------------
def test_the_registered_schema_offers_the_frame_columns_not_the_invented_inputs() -> None:
    from src.tool_registry import get_registry

    declared = {p.name for p in get_registry().get_schema("sensitivity_analyzer").input_parameters}
    assert {"treatment", "outcome", "confounders"} <= declared
    assert "naive_ate" not in declared
    assert "baseline_risk" not in declared
