"""Red-first fail-closed / real-compute assertions for the honest-tools cleanup
(SHARD R1, findings F3/F4/F7).

Each previously-fabricated tool in ``tool_registrations.py`` now either computes
its output from a caller-supplied real ``pandas.DataFrame`` / scalar inputs, or
FAILS CLOSED with a descriptive ``RuntimeError`` — never returning a
plausible-but-fake placeholder, and never crashing with a bare ``AttributeError``
on a string/empty input.

Tests build their OWN DataFrames (allowed — the anti-mock rule forbids
fabricating data INSIDE tool bodies, not in tests).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr


# ---------------------------------------------------------------------------
# Task 1 — sensitivity_analyzer: real VanderWeele E-value (F3)
# ---------------------------------------------------------------------------
def test_sensitivity_analyzer_varies_with_input_and_matches_hand_value():
    # RR = exp(0.91*ate); for ate=0.5 -> RR = exp(0.455) ~= 1.57620.
    # E = RR + sqrt(RR*(RR-1)) = 1.57620 + sqrt(1.57620*0.57620) ~= 2.53942.
    out = tr.sensitivity_analyzer(ate=0.5, ci_lower=0.1)
    rr = math.exp(0.91 * 0.5)
    expected_point = rr + math.sqrt(rr * (rr - 1.0))
    assert out["e_value_point"] == pytest.approx(expected_point, rel=1e-9)

    # A larger ate must yield a strictly larger point E-value (it VARIES).
    out_big = tr.sensitivity_analyzer(ate=1.0, ci_lower=0.2)
    assert out_big["e_value_point"] > out["e_value_point"]

    # CI bound whose RR crosses 1.0 -> e_value_ci == 1.0 (E-value floor).
    out_cross = tr.sensitivity_analyzer(ate=0.5, ci_lower=0.0)  # RR_ci = exp(0)=1.0
    assert out_cross["e_value_ci"] == pytest.approx(1.0, abs=1e-12)

    # robustness derived from the computed E-value, NOT hardcoded "moderate".
    assert out_big["robustness"] in {"weak", "moderate", "strong"}


def test_sensitivity_analyzer_fail_closes_on_non_finite():
    with pytest.raises(RuntimeError):
        tr.sensitivity_analyzer(ate=float("nan"), ci_lower=0.1)
    with pytest.raises(RuntimeError):
        tr.sensitivity_analyzer(ate=0.5, ci_lower=float("inf"))


# ---------------------------------------------------------------------------
# Task 2 — psi_calculator: real PSI from a DataFrame or fail-close (F3)
# ---------------------------------------------------------------------------
def _psi_reference(baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
    # Independent reference implementation for the test (NOT imported from src).
    edges = np.quantile(baseline, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    b_counts = np.histogram(baseline, bins=edges)[0].astype(float)
    c_counts = np.histogram(current, bins=edges)[0].astype(float)
    b_pct = np.clip(b_counts / b_counts.sum(), 1e-6, None)
    c_pct = np.clip(c_counts / c_counts.sum(), 1e-6, None)
    return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))


def test_psi_calculator_matches_reference_and_varies():
    rng = np.random.default_rng(3)
    n = 500
    df = pd.DataFrame(
        {
            "period": ["baseline"] * n + ["current"] * n,
            "score": np.concatenate(
                [rng.normal(0.0, 1.0, n), rng.normal(0.8, 1.0, n)]  # shifted -> drift
            ),
        }
    )
    out = tr.psi_calculator(
        feature="score",
        baseline_period="baseline",
        current_period="current",
        estimation_data=df,
        period_column="period",
    )
    ref = _psi_reference(
        df.loc[df["period"] == "baseline", "score"],
        df.loc[df["period"] == "current", "score"],
    )
    assert out["psi"] == pytest.approx(ref, rel=1e-9)
    assert out["psi"] > 0.1  # genuine shift -> drift, not the old hardcoded 0.08


def test_psi_calculator_fail_closes_without_dataframe():
    with pytest.raises(RuntimeError):
        tr.psi_calculator(feature="score", baseline_period="baseline", current_period="current")
    # F4-style: a string passed where the frame is expected must NOT AttributeError.
    with pytest.raises(RuntimeError):
        tr.psi_calculator(
            feature="score",
            baseline_period="baseline",
            current_period="current",
            estimation_data="not-a-frame",
        )


# ---------------------------------------------------------------------------
# Task 3 — distribution_comparator: real KS test or fail-close (F3)
# ---------------------------------------------------------------------------
def test_distribution_comparator_matches_scipy_and_fail_closes():
    from scipy.stats import ks_2samp

    rng = np.random.default_rng(5)
    n = 400
    df = pd.DataFrame(
        {
            "period": ["p1"] * n + ["p2"] * n,
            "x": np.concatenate([rng.normal(0, 1, n), rng.normal(1.5, 1, n)]),
            "y": np.concatenate([rng.normal(0, 1, n), rng.normal(0, 1, n)]),
        }
    )
    out = tr.distribution_comparator(
        features=["x", "y"],
        period_1="p1",
        period_2="p2",
        estimation_data=df,
        period_column="period",
    )
    by_feature = {c["feature"]: c for c in out["comparisons"]}
    ks_x = ks_2samp(df.loc[df.period == "p1", "x"], df.loc[df.period == "p2", "x"])
    assert by_feature["x"]["ks_statistic"] == pytest.approx(ks_x.statistic, rel=1e-9)
    assert by_feature["x"]["p_value"] == pytest.approx(ks_x.pvalue, rel=1e-6)
    assert by_feature["x"]["drift_detected"] is True  # genuine shift
    assert by_feature["y"]["drift_detected"] is False  # same distribution


def test_distribution_comparator_fail_closes_on_string_input():
    with pytest.raises(RuntimeError):
        tr.distribution_comparator(features=["x"], period_1="p1", period_2="p2")
    with pytest.raises(RuntimeError):
        tr.distribution_comparator(
            features=["x"],
            period_1="p1",
            period_2="p2",
            estimation_data="not-a-frame",
        )


# ---------------------------------------------------------------------------
# Task 4 — cohort_statistics: real stats + dict type-guard (F3, F4)
# ---------------------------------------------------------------------------
def test_cohort_statistics_computes_real_demographics_from_frame():
    df = pd.DataFrame(
        {
            "patient_id": [f"P{i}" for i in range(6)],
            "age": [40, 50, 60, 70, 30, 80],
            "gender": ["male", "female", "male", "female", "female", "male"],
        }
    )
    cohort_result = {"total_eligible": len(df), "eligible_patient_ids": list(df["patient_id"])}
    out = tr.cohort_statistics(cohort_result=cohort_result, estimation_data=df)
    assert out.cohort_size == 6
    assert out.demographics["age_mean"] == pytest.approx(55.0)  # NOT hardcoded 52.3
    assert out.demographics["age_mean"] != 52.3


def test_cohort_statistics_fail_closes_on_string_cohort_result():
    # F4: a string where a dict is expected must RuntimeError, not AttributeError.
    df = pd.DataFrame({"age": [40, 50]})
    with pytest.raises(RuntimeError):
        tr.cohort_statistics(cohort_result="not-a-dict", estimation_data=df)


def test_cohort_statistics_fail_closes_without_frame():
    with pytest.raises(RuntimeError):
        tr.cohort_statistics(cohort_result={"total_eligible": 3})


# ---------------------------------------------------------------------------
# Task 5 — cohort_validator: real completeness + dict type-guard (F3, F4)
# ---------------------------------------------------------------------------
def test_cohort_validator_real_completeness_from_frame():
    # 8 of 10 cells populated -> completeness 0.8, NOT hardcoded 0.95.
    df = pd.DataFrame({"age": [40, 50, 60, None, 70], "gender": ["m", "f", None, "f", "m"]})
    cohort_result = {"total_eligible": 150}
    out = tr.cohort_validator(cohort_result=cohort_result, min_cohort_size=100, estimation_data=df)
    assert out.is_valid is True
    completeness_check = next(c for c in out.validation_checks if c["check"] == "data_completeness")
    assert completeness_check["actual"] == pytest.approx(0.8)
    assert completeness_check["actual"] != 0.95  # not the old hardcoded value


def test_cohort_validator_fail_closes_on_string_cohort_result():
    # F4: a string must RuntimeError (the old body would AttributeError on .get).
    with pytest.raises(RuntimeError):
        tr.cohort_validator(cohort_result="not-a-dict", min_cohort_size=100)


def test_cohort_validator_fail_closes_without_frame():
    with pytest.raises(RuntimeError):
        tr.cohort_validator(cohort_result={"total_eligible": 150}, min_cohort_size=100)


# ---------------------------------------------------------------------------
# Task 6 — cohort_builder & refutation_runner: intentional fail-close (F3)
# ---------------------------------------------------------------------------
def test_cohort_builder_fail_closes_and_directs_to_agent():
    with pytest.raises(RuntimeError, match="cohort_constructor"):
        tr.cohort_builder(brand="Kisqali", indication="HR+ breast cancer")
    # No fabricated P001/P002/P003 should ever appear.


def test_refutation_runner_fail_closes_and_directs_to_run_refutation():
    with pytest.raises(RuntimeError, match="run_refutation"):
        tr.refutation_runner(estimate_id="est-123")


# ---------------------------------------------------------------------------
# Task 7 — F7: document the two data contracts in the module docstring
# ---------------------------------------------------------------------------
def test_module_docstring_documents_both_data_contracts():
    doc = tr.__doc__ or ""
    assert "estimation_data" in doc
    assert "_extract_dataframe_from_kwargs" in doc
    assert "discover_dag" in doc  # the Dict-contract tool is named explicitly


def test_sensitivity_analyzer_handles_protective_effect():
    # Protective effect (ate < 0): RR = exp(0.91*ate) < 1.0, so _e_value_from_rr
    # inverts it to 1/RR. By construction this is SYMMETRIC with the harmful
    # +|ate| case — exercises the rr < 1.0 inversion branch.
    out = tr.sensitivity_analyzer(ate=-0.5, ci_lower=-0.8)
    rr = 1.0 / math.exp(0.91 * 0.5)  # exp(-0.455) < 1.0
    rr = 1.0 / rr  # the inversion _e_value_from_rr applies
    expected_point = rr + math.sqrt(rr * (rr - 1.0))
    assert out["e_value_point"] == pytest.approx(expected_point, rel=1e-9)

    # Protective -0.5 yields the SAME point E-value as harmful +0.5 (symmetry).
    out_harmful = tr.sensitivity_analyzer(ate=0.5, ci_lower=0.1)
    assert out["e_value_point"] == pytest.approx(out_harmful["e_value_point"], rel=1e-9)
