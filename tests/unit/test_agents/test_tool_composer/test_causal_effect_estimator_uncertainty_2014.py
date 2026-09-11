"""#2014: ``causal_effect_estimator`` reports real sampling uncertainty or none at all.

Before: ``_derive_ci_and_p_value`` read only ``ate_ci_lower/ate_ci_upper``, ignored
the ``standard_error`` DoWhy already returned, and fell to a proxy built from library
agreement — ATE +/- 0.001 with p = 0.001 whenever one library ran. On the real Kisqali
cohort (n = 8,730) the real 95 % half-widths were 0.0088 and 0.0273.

Now:
* the CI is ATE +/- z * SE and the p-value the matching two-sided normal p, from the
  HC1 SE of DoWhy's own OLS fit;
* when no real uncertainty exists the CI and p-value are ``None`` with the reason;
* ``method`` is no longer offered — the tool runs DoWhy linear regression and says so,
  and a request for another estimator is refused instead of echoed;
* the output names what was estimated, per unit for a non-binary treatment.

Real SequentialPipeline / DoWhy on real deterministic frames. The reference SEs are the
OLS / HC1 formulas written out in numpy, independent of DoWhy and statsmodels.
"""

from __future__ import annotations

import asyncio
import math

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr
from src.agents.tool_composer.errors import ToolInputError
from src.causal_engine.pipeline import SequentialPipeline

Z_95 = 1.959963984540054


def _frame(kind: str, n: int = 600, seed: int = 2014) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = rng.normal(0.0, 1.0, n)
    if kind == "count":
        t = np.clip(np.round(1.5 + c + rng.normal(0.0, 1.0, n)), 0, 5).astype(int)
    else:
        t = (c + rng.normal(0.0, 1.0, n) > 0.3).astype(int)
    if kind == "binary_outcome":
        # linear-probability design: P(y=1) = 0.25 + 0.2 t + 0.1 c, clipped
        p = np.clip(0.25 + 0.2 * t + 0.1 * c, 0.02, 0.98)
        y = (rng.uniform(0.0, 1.0, n) < p).astype(int)
    else:
        y = 0.4 * t + 0.8 * c + rng.normal(0.0, 1.0, n) * (1.0 + 2.0 * (t > 0))
    return pd.DataFrame({"treatment": t, "outcome": y, "confounder_a": c})


def _ols_reference(df: pd.DataFrame) -> dict:
    x = np.column_stack([np.ones(len(df)), df["treatment"], df["confounder_a"]]).astype(float)
    y = df["outcome"].to_numpy(dtype=float)
    n, k = x.shape
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    nonrobust = xtx_inv * (resid @ resid) / (n - k)
    hc1 = xtx_inv @ ((x * (resid**2)[:, None]).T @ x) @ xtx_inv * n / (n - k)
    return {
        "coef": float(beta[1]),
        "se_nonrobust": float(np.sqrt(nonrobust[1, 1])),
        "se_hc1": float(np.sqrt(hc1[1, 1])),
    }


def _estimate(df: pd.DataFrame, **kwargs) -> tr.EffectEstimate:
    return tr.causal_effect_estimator(
        treatment="treatment",
        outcome="outcome",
        confounders=["confounder_a"],
        estimation_data=df,
        **kwargs,
    )


def _assert_hc1_interval(result: tr.EffectEstimate, ref: dict) -> None:
    assert result.ate == pytest.approx(ref["coef"], abs=1e-9)
    assert result.standard_error == pytest.approx(ref["se_hc1"], rel=1e-9)
    assert result.ci_lower is not None and result.ci_upper is not None
    half_width = (result.ci_upper - result.ci_lower) / 2.0
    assert half_width == pytest.approx(Z_95 * ref["se_hc1"], rel=1e-9)
    assert half_width == pytest.approx(1.96 * ref["se_hc1"], rel=1e-4)
    assert (result.ci_upper + result.ci_lower) / 2.0 == pytest.approx(result.ate, abs=1e-12)
    expected_p = math.erfc(abs(ref["coef"] / ref["se_hc1"]) / math.sqrt(2.0))
    assert result.p_value == pytest.approx(expected_p, rel=1e-6, abs=1e-300)
    assert result.uncertainty_method == "ols_hc1_normal"
    assert "HC1" in result.uncertainty_note
    assert result.method == "backdoor.linear_regression"


def test_binary_treatment_continuous_outcome_ci_is_z_times_the_hc1_se() -> None:
    df = _frame("continuous")
    ref = _ols_reference(df)

    result = _estimate(df)

    _assert_hc1_interval(result, ref)
    assert result.effect_scale == "binary_contrast"
    assert "treatment = 1" in result.estimand and "treatment = 0" in result.estimand


def test_binary_outcome_uses_the_robust_se_not_the_nonrobust_one() -> None:
    df = _frame("binary_outcome")
    ref = _ols_reference(df)
    assert abs(ref["se_hc1"] - ref["se_nonrobust"]) / ref["se_nonrobust"] > 0.01

    result = _estimate(df)

    _assert_hc1_interval(result, ref)
    assert result.standard_error != pytest.approx(ref["se_nonrobust"], rel=1e-3)


def test_count_treatment_is_labelled_as_a_per_unit_effect() -> None:
    df = _frame("count")
    assert df["treatment"].nunique() > 2
    ref = _ols_reference(df)

    result = _estimate(df)

    _assert_hc1_interval(result, ref)
    assert result.effect_scale == "per_unit"
    assert "one-unit increase in treatment" in result.estimand


def test_no_standard_error_yields_none_and_a_reason_never_a_number() -> None:
    # Three rows, three OLS parameters: zero residual degrees of freedom, so no SE exists.
    df = _frame("continuous").iloc[:3].reset_index(drop=True)

    result = _estimate(df)

    assert math.isfinite(result.ate)
    assert result.ci_lower is None
    assert result.ci_upper is None
    assert result.p_value is None
    assert result.standard_error is None
    assert result.uncertainty_method == "not_computed"
    assert "standard error" in result.uncertainty_note


def test_primary_library_without_an_estimate_yields_none_and_a_reason() -> None:
    # Impact-flow wording routes NetworkX as the primary library: the reported effect is
    # DoWhy's (the only estimate), but the primary result carries no uncertainty for it.
    df = _frame("continuous")

    result = _estimate(
        df, query="How does the impact flow through the network path from treatment to outcome?"
    )

    assert result.ci_lower is None and result.ci_upper is None and result.p_value is None
    assert result.uncertainty_method == "not_computed"
    assert "networkx" in result.uncertainty_note.lower()


def test_a_consensus_of_several_libraries_has_no_interval() -> None:
    df = _frame("continuous", n=400)
    output = asyncio.run(
        SequentialPipeline().execute(
            {
                "query": "Estimate the causal effect of treatment on outcome.",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "confounders": ["confounder_a"],
                "effect_modifiers": None,
                "data_source": "test",
                "filters": None,
                "estimation_data": df,
                "mode": "sequential",
                "libraries_enabled": ["dowhy", "econml"],
                "cross_validate": None,
            }
        )
    )
    primary = output["primary_result"]
    consensus = output["consensus_effect"]
    assert primary["standard_error"] is not None
    assert consensus != pytest.approx(primary["causal_effect"], abs=1e-6)

    uncertainty = tr._derive_uncertainty(ate=consensus, primary_result=primary)

    assert uncertainty["ci_lower"] is None and uncertainty["p_value"] is None
    assert uncertainty["uncertainty_method"] == "not_computed"
    assert "consensus" in uncertainty["uncertainty_note"]


def test_a_library_interval_is_used_with_its_back_derived_se() -> None:
    df = _frame("continuous", n=400)
    output = asyncio.run(
        SequentialPipeline().execute(
            {
                "query": "Estimate the causal effect of treatment on outcome.",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "confounders": ["confounder_a"],
                "effect_modifiers": None,
                "data_source": "test",
                "filters": None,
                "estimation_data": df,
                "mode": "sequential",
                "libraries_enabled": ["econml"],
                "cross_validate": None,
            }
        )
    )
    primary = output["primary_result"]
    lo, hi = primary["ate_ci_lower"], primary["ate_ci_upper"]

    uncertainty = tr._derive_uncertainty(ate=output["consensus_effect"], primary_result=primary)

    assert output["consensus_effect"] == pytest.approx(primary["ate"], abs=1e-12)
    assert (uncertainty["ci_lower"], uncertainty["ci_upper"]) == (lo, hi)
    se = (hi - lo) / (2.0 * Z_95)
    assert uncertainty["standard_error"] == pytest.approx(se, rel=1e-12)
    assert uncertainty["p_value"] == pytest.approx(
        math.erfc(abs(primary["ate"] / se) / math.sqrt(2.0)), rel=1e-9
    )
    assert uncertainty["uncertainty_method"] == "library_interval"


def test_method_is_not_offered_to_the_planner() -> None:
    from src.tool_registry import get_registry

    declared = {
        p.name for p in get_registry().get_schema("causal_effect_estimator").input_parameters
    }

    assert "method" not in declared
    assert "method" not in tr.EffectEstimatorInput.model_fields


def test_a_request_for_another_estimator_is_refused_not_echoed() -> None:
    df = _frame("continuous", n=200)

    with pytest.raises(ToolInputError, match="backdoor.linear_regression"):
        _estimate(df, method="backdoor.propensity_score_matching")

    assert _estimate(df, method="backdoor.linear_regression").method == (
        "backdoor.linear_regression"
    )


def test_the_proxy_formula_is_gone() -> None:
    assert not hasattr(tr, "_derive_ci_and_p_value")
    assert not hasattr(tr, "_derive_p_value_from_confidence")


def test_the_synthesizer_sees_every_field_even_with_many_confounders(sample_decomposition) -> None:
    """The synthesizer truncates each step's JSON at 1,000 characters; the uncertainty
    note and the estimand must survive it however many confounders were adjusted for."""
    from datetime import datetime, timezone

    from src.agents.tool_composer.models.composition_models import (
        ExecutionStatus,
        ExecutionTrace,
        StepResult,
        SynthesisInput,
        ToolInput,
        ToolOutput,
    )
    from src.agents.tool_composer.synthesizer import ResponseSynthesizer

    rng = np.random.default_rng(7)
    df = _frame("count")
    names = [f"a_rather_long_confounder_name_{i:02d}" for i in range(12)]
    for name in names:
        df[name] = rng.normal(0.0, 1.0, len(df))
    estimate = tr.causal_effect_estimator(
        treatment="treatment", outcome="outcome", confounders=names, estimation_data=df
    ).model_dump()
    now = datetime.now(timezone.utc)
    trace = ExecutionTrace(plan_id="plan_2014")
    trace.add_result(
        StepResult(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="causal_effect_estimator",
            input=ToolInput(tool_name="causal_effect_estimator", parameters={}),
            output=ToolOutput(tool_name="causal_effect_estimator", success=True, result=estimate),
            status=ExecutionStatus.COMPLETED,
            started_at=now,
            completed_at=now,
        )
    )

    formatted = ResponseSynthesizer(llm_client=None)._format_results(
        SynthesisInput(
            original_query="effect?", decomposition=sample_decomposition, execution_trace=trace
        )
    )

    assert "(truncated)" not in formatted
    for key in tr.EffectEstimate.model_fields:
        assert f'"{key}"' in formatted
