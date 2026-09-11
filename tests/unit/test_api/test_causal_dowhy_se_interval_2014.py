"""#2014 sweep: the causal API routes drop DoWhy's standard error instead of using it.

The same key mismatch as ``causal_effect_estimator``: DoWhy's pipeline executor returns
its uncertainty as ``standard_error`` (the HC1 SE of its OLS fit), while

* ``POST /causal/pipeline/sequential``'s per-library payload (``_extract_library_payload``)
  read only the effect, method and estimand for DoWhy — its stage carried
  ``ci_lower`` / ``ci_upper`` / ``p_value`` = None and the SE was discarded;
* ``GET /causal/treatment-effects``'s DoWhy fallback (EconML failed) reported the SE and
  its p-value but left the CI None ("linear_regression provides only an SE").

Nothing was fabricated; a real interval was dropped. Real SequentialPipeline / DoWhy on a
real deterministic frame.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from src.api.routes import causal as causal_routes
from src.causal.stats import z_score_for_confidence


def _frame(n: int = 600, seed: int = 2014) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = rng.normal(0.0, 1.0, n)
    t = (c + rng.normal(0.0, 1.0, n) > 0.3).astype(int)
    y = 0.4 * t + 0.8 * c + rng.normal(0.0, 1.0, n) * (1.0 + 2.0 * t)
    return pd.DataFrame({"treatment": t, "outcome": y, "confounder_a": c})


def _run_dowhy() -> tuple:
    pipeline = causal_routes._SurfaceCSequentialPipeline(fail_fast=False)
    output = asyncio.run(
        pipeline.execute(
            {
                "query": "Estimate the causal effect of treatment on outcome.",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "confounders": ["confounder_a"],
                "effect_modifiers": None,
                "data_source": "test",
                "filters": None,
                "estimation_data": _frame(),
                "mode": "sequential",
                "libraries_enabled": ["dowhy"],
                "cross_validate": None,
            }
        )
    )
    return output, pipeline.last_state


def test_sequential_pipeline_dowhy_stage_carries_the_interval_from_its_se() -> None:
    output, state = _run_dowhy()
    dowhy = state["dowhy_result"]["result"]
    effect, se = dowhy["causal_effect"], dowhy["standard_error"]
    assert se is not None and dowhy["standard_error_method"] == "ols_hc1"
    z = z_score_for_confidence(0.95)

    payload = causal_routes._extract_library_payload("dowhy", output, state=state)

    assert payload["ci_lower"] == pytest.approx(effect - z * se, rel=1e-12)
    assert payload["ci_upper"] == pytest.approx(effect + z * se, rel=1e-12)
    assert payload["p_value"] == pytest.approx(causal_routes._te_pvalue_from_z(effect, se))
    assert payload["standard_error"] == se
    assert payload["standard_error_method"] == "ols_hc1"


def test_dowhy_interval_helper_returns_nothing_without_an_se() -> None:
    assert causal_routes._dowhy_interval({"causal_effect": 0.3, "standard_error": None}) is None
    assert causal_routes._dowhy_interval({"causal_effect": 0.3, "standard_error": 0.0}) is None


def test_treatment_effects_dowhy_fallback_reports_the_interval() -> None:
    # A frame with no confounders: EconML fails closed (no covariates), DoWhy estimates.
    rng = np.random.default_rng(5)
    n = 300
    t = (rng.normal(0.0, 1.0, n) > 0).astype(int)
    y = 0.5 * t + rng.normal(0.0, 1.0, n) * (1.0 + t)
    spec = causal_routes._TEFrameSpec(
        frame=pd.DataFrame({"t": t, "y": y}), treatment_var="t", outcome_var="y", confounders=[]
    )
    x = np.column_stack([np.ones(n), t]).astype(float)
    xtx_inv = np.linalg.inv(x.T @ x)
    resid = y - x @ (xtx_inv @ x.T @ y)
    hc1 = xtx_inv @ ((x * (resid**2)[:, None]).T @ x) @ xtx_inv * n / (n - 2)
    z = z_score_for_confidence(0.95)

    response = asyncio.run(
        causal_routes._run_treatment_effect_estimate("initiation", "Kisqali", spec)
    )

    assert response.estimator == "backdoor.linear_regression"  # the DoWhy fallback ran
    assert response.std_error == pytest.approx(float(np.sqrt(hc1[1, 1])), rel=1e-9)
    assert response.ci_lower == pytest.approx(response.ate - z * response.std_error, rel=1e-12)
    assert response.ci_upper == pytest.approx(response.ate + z * response.std_error, rel=1e-12)
