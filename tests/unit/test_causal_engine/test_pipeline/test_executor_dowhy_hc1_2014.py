"""#2014: the DoWhy executor's linear-regression standard error is heteroskedasticity-robust (HC1).

Measured on the deployed image (real Kisqali cohort, n = 8,730): DoWhy's
``estimate.value`` equals the statsmodels OLS treatment coefficient on the identical
design (|diff| <= 1.7e-16) and ``get_standard_error()`` is that fit's NONROBUST SE
(0.004487 / 0.013931). On a binary outcome the regression is a linear-probability
model, heteroskedastic by construction, so the executor now reports the HC1 SE of the
same fit (0.004437 / 0.013543 there) and names it.

Real DoWhy on a real deterministic frame; the reference SE is the HC1 sandwich written
out in numpy, independent of statsmodels and of DoWhy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.causal_engine.pipeline.executors.dowhy import DoWhyExecutor


def _heteroskedastic_frame(n: int = 600, seed: int = 2014) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = rng.normal(0.0, 1.0, n)
    t = (c + rng.normal(0.0, 1.0, n) > 0.3).astype(int)
    # noise variance triples in the treated arm -> HC1 and nonrobust SEs differ
    y = 0.4 * t + 0.8 * c + rng.normal(0.0, 1.0, n) * (1.0 + 2.0 * t)
    return pd.DataFrame({"treatment": t, "outcome": y, "confounder_a": c})


def _ols_reference(df: pd.DataFrame) -> dict:
    """Treatment coefficient with its nonrobust and HC1 SEs, by explicit matrix algebra."""
    x = np.column_stack([np.ones(len(df)), df["treatment"], df["confounder_a"]]).astype(float)
    y = df["outcome"].to_numpy(dtype=float)
    n, k = x.shape
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    nonrobust = xtx_inv * (resid @ resid) / (n - k)
    meat = (x * (resid**2)[:, None]).T @ x
    hc1 = xtx_inv @ meat @ xtx_inv * n / (n - k)
    return {
        "coef": float(beta[1]),
        "se_nonrobust": float(np.sqrt(nonrobust[1, 1])),
        "se_hc1": float(np.sqrt(hc1[1, 1])),
    }


async def _execute(df: pd.DataFrame, method: str | None = None) -> dict:
    state: dict = {
        "treatment_var": "treatment",
        "outcome_var": "outcome",
        "confounders": ["confounder_a"],
        "estimation_data": df,
    }
    if method is not None:
        state["filters"] = {"dowhy_method": method}
    result = await DoWhyExecutor().execute(state, {})  # type: ignore[arg-type]
    assert result["success"], result.get("error")
    return result["result"]


async def test_linear_regression_standard_error_is_hc1_of_the_same_fit() -> None:
    df = _heteroskedastic_frame()
    ref = _ols_reference(df)
    # The frame must discriminate: HC1 and nonrobust differ by far more than tolerance.
    assert abs(ref["se_hc1"] - ref["se_nonrobust"]) / ref["se_nonrobust"] > 0.05

    payload = await _execute(df)

    assert abs(payload["causal_effect"] - ref["coef"]) < 1e-9
    assert abs(payload["standard_error"] - ref["se_hc1"]) < 1e-9 * max(1.0, ref["se_hc1"])
    assert payload["standard_error_method"] == "ols_hc1"


async def test_a_method_without_an_analytic_se_reports_none_and_no_bootstrap() -> None:
    df = _heteroskedastic_frame()

    payload = await _execute(df, method="backdoor.propensity_score_weighting")

    assert payload["standard_error"] is None
    assert payload["standard_error_method"] is None
