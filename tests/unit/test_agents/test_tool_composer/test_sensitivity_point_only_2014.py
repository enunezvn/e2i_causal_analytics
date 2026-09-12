"""#2014: ``sensitivity_analyzer`` without a confidence interval reports the point E-value only.

``causal_effect_estimator`` now returns ``ci_lower`` / ``ci_upper`` = ``None`` when no real
sampling uncertainty exists. The planner maps those fields into ``sensitivity_analyzer``
by name, so ``None`` arrives there: it must report the point E-value, say the interval is
unavailable, and give no reading that needs an interval (null / beyond / within all do).

Real tools, real DoWhy, real executor reference resolution; no mocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr
from src.agents.tool_composer.errors import ToolInputError
from src.agents.tool_composer.executor import PlanExecutor
from src.causal_engine import evalue


def _frame(n: int, seed: int = 2014) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = rng.normal(0.0, 1.0, n)
    t = (c + rng.normal(0.0, 1.0, n) > 0.3).astype(int)
    y = 0.4 * t + 0.8 * c + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"treatment": t, "outcome": y, "confounder_a": c})


def _binary_frame(n: int, seed: int = 2014) -> pd.DataFrame:
    """Same shape as ``_frame`` with a BINARY outcome, so a baseline risk is derivable."""
    rng = np.random.default_rng(seed)
    c = rng.normal(0.0, 1.0, n)
    t = (c + rng.normal(0.0, 1.0, n) > 0.3).astype(int)
    p = 1.0 / (1.0 + np.exp(-(-0.3 + 0.6 * t + 0.5 * c)))
    return pd.DataFrame(
        {"treatment": t, "outcome": (rng.random(n) < p).astype(int), "confounder_a": c}
    )


def _assert_point_only(report: dict, expected_point: float) -> None:
    assert report["e_value_point"] == pytest.approx(expected_point, rel=1e-12)
    assert report["e_value_ci"] is None
    assert report["reading"] == "interval_unavailable"
    assert report["benchmark"] is None
    assert "interval" in report["headline"].lower()
    assert "no confidence interval" in report["interpretation"].lower()


@pytest.mark.parametrize("ci", [{}, {"ci_lower": None, "ci_upper": None}])
def test_no_interval_reports_the_point_e_value_only(ci) -> None:
    report = tr.sensitivity_analyzer(ate=0.3, **ci)

    _assert_point_only(report, evalue.e_value_from_rr(evalue.rr_from_smd(0.3)))
    assert report["conversion"] == "standardized_difference"


def test_no_interval_on_the_risk_ratio_path() -> None:
    # #2022: the baseline risk is DERIVED — the control-arm rate of a binary outcome —
    # so the risk-ratio path is reached from the frame, not from a caller's number.
    frame = _binary_frame(800)
    report = tr.sensitivity_analyzer(
        ate=0.1,
        ci_lower=None,
        treatment="treatment",
        outcome="outcome",
        confounders=["confounder_a"],
        estimation_data=frame,
    )

    t = np.asarray(frame["treatment"], dtype=float)
    y = np.asarray(frame["outcome"], dtype=float)
    rr = evalue.rr_from_risk_difference(0.1, float(y[t == 0].mean()))
    assert rr is not None
    assert report["e_value_point"] == pytest.approx(evalue.e_value_from_rr(rr), rel=1e-12)
    assert report["e_value_ci"] is None
    assert report["reading"] == "interval_unavailable"
    assert report["conversion"] == "risk_ratio"


def test_no_interval_with_a_supplied_baseline_risk_is_refused() -> None:
    # #2022: nothing emits a baseline risk, so a supplied one is invented by construction.
    with pytest.raises(RuntimeError, match="baseline_risk"):
        tr.sensitivity_analyzer(ate=0.9, ci_lower=None, baseline_risk=0.3)


def test_half_an_interval_is_refused_not_mirrored_from_nothing() -> None:
    # An upper bound alone used to be unreachable (ci_lower was required); mirroring it
    # would invent a lower bound the estimator never produced.
    with pytest.raises(ToolInputError, match="ci_upper"):
        tr.sensitivity_analyzer(ate=0.3, ci_lower=None, ci_upper=0.5)


def test_ci_lower_is_declared_optional() -> None:
    from src.tool_registry import get_registry

    params = {p.name: p for p in get_registry().get_schema("sensitivity_analyzer").input_parameters}
    assert params["ci_lower"].required is False


def test_the_estimators_none_interval_survives_executor_resolution_into_sensitivity() -> None:
    # Three rows, three OLS parameters: the real estimator returns no interval.
    estimate = tr.causal_effect_estimator(
        treatment="treatment",
        outcome="outcome",
        confounders=["confounder_a"],
        estimation_data=_frame(600).iloc[:3].reset_index(drop=True),
    ).model_dump()
    assert estimate["ci_lower"] is None and estimate["ci_upper"] is None

    resolved = PlanExecutor()._resolve_inputs(
        {"ate": "$step_1.ate", "ci_lower": "$step_1.ci_lower", "ci_upper": "$step_1.ci_upper"},
        {"step_1": estimate},
        {},
    )
    assert resolved == {"ate": estimate["ate"], "ci_lower": None, "ci_upper": None}

    report = tr.sensitivity_analyzer(**resolved)

    assert report["e_value_ci"] is None
    assert report["reading"] == "interval_unavailable"


def test_a_real_interval_still_gives_the_full_reading() -> None:
    estimate = tr.causal_effect_estimator(
        treatment="treatment",
        outcome="outcome",
        confounders=["confounder_a"],
        estimation_data=_frame(600),
    ).model_dump()
    assert estimate["ci_lower"] is not None

    report = tr.sensitivity_analyzer(
        ate=estimate["ate"], ci_lower=estimate["ci_lower"], ci_upper=estimate["ci_upper"]
    )

    assert report["e_value_ci"] is not None
    assert report["reading"] != "interval_unavailable"


def test_no_interval_still_reports_the_measured_confounding_benchmark() -> None:
    # The benchmark is a point quantity (spec 2.5: benchmark the point estimate, state
    # precision separately), so the frame-derived naive contrast still yields it; only the
    # verdict, which needs the interval to rule out a null finding, is withheld.
    # #2022: the contrast and the SD come from the frame, not from caller-bound numbers.
    frame = _frame(600)
    report = tr.sensitivity_analyzer(
        ate=0.1,
        ci_lower=None,
        treatment="treatment",
        outcome="outcome",
        confounders=["confounder_a"],
        estimation_data=frame,
    )

    t = np.asarray(frame["treatment"], dtype=float)
    y = np.asarray(frame["outcome"], dtype=float)
    expected = evalue.joint_confounding_benchmark(
        float(y[t == 1].mean() - y[t == 0].mean()),
        0.1,
        baseline_risk=None,
        outcome_std=float(np.std(y)),
    )
    assert expected is not None
    assert report["benchmark"] == pytest.approx(expected, rel=1e-12)
    assert report["benchmark_basis"] == "joint_naive_vs_adjusted"
    assert report["reading"] == "interval_unavailable"
    assert f"{expected:.2f}" in report["interpretation"]
