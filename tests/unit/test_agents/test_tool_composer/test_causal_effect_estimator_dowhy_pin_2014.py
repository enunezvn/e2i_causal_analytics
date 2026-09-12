"""#2014 (owner GO): ``causal_effect_estimator`` runs DoWhy, whatever its column names say.

The pipeline router classifies the tool's generated sentence by keyword, so a column
name decided which libraries ran: over every ordered column pair of the real Kisqali
cohort frame, the 168 pairs naming ``payer_category`` routed to EconML + CausalML
("cate" is a heterogeneity keyword and "payer_category" contains it). A one-hot column
such as ``payer_category_commercial`` does the same to a perfectly estimable binary
treatment, and the tool then returned an EconML + CausalML blend with no interval.

The tool now pins ``libraries_enabled=["dowhy", "networkx"]``. The router makes the
FIRST forced library primary; that is measured here on a real run in both orders
rather than read off ``router.route``.

Real SequentialPipeline / DoWhy on real deterministic frames. The reference SE is the
HC1 sandwich written out in numpy.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr
from src.causal_engine.pipeline import SequentialPipeline
from src.causal_engine.pipeline.router import CausalLibrary, LibraryRouter

Z_95 = 1.959963984540054
TREATMENT = "payer_category_commercial"


def _frame(n: int = 600, seed: int = 2014) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = rng.normal(0.0, 1.0, n)
    t = (c + rng.normal(0.0, 1.0, n) > 0.3).astype(int)
    y = 0.4 * t + 0.8 * c + rng.normal(0.0, 1.0, n) * (1.0 + 2.0 * t)
    return pd.DataFrame(
        {
            TREATMENT: t,
            "outcome": y,
            "confounder_a": c,
            "payer_category": np.where(t == 1, "commercial", "medicare"),
        }
    )


def _hc1(df: pd.DataFrame) -> tuple:
    x = np.column_stack([np.ones(len(df)), df[TREATMENT], df["confounder_a"]]).astype(float)
    y = df["outcome"].to_numpy(dtype=float)
    n, k = x.shape
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    cov = xtx_inv @ ((x * (resid**2)[:, None]).T @ x) @ xtx_inv * n / (n - k)
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def test_a_keyword_routed_treatment_is_estimated_by_dowhy_with_the_hc1_interval() -> None:
    df = _frame()
    # The sentence the tool routes on sends this treatment to EconML + CausalML.
    sentence = (
        f"Estimate the causal effect of {TREATMENT} on outcome "
        "using method='backdoor.linear_regression'."
    )
    assert LibraryRouter().route(sentence).primary_library is CausalLibrary.ECONML
    coef, se = _hc1(df)

    result = tr.causal_effect_estimator(
        treatment=TREATMENT, outcome="outcome", confounders=["confounder_a"], estimation_data=df
    )

    assert result.method == "backdoor.linear_regression"
    assert result.uncertainty_method == "ols_hc1_normal"
    assert result.ate == pytest.approx(coef, abs=1e-9)
    assert result.standard_error == pytest.approx(se, rel=1e-9)
    assert (result.ci_upper - result.ci_lower) / 2.0 == pytest.approx(Z_95 * se, rel=1e-9)


def test_a_query_override_no_longer_changes_the_libraries() -> None:
    # Impact-flow wording used to make NetworkX primary (no interval); heterogeneity
    # wording used to route EconML + CausalML.
    df = _frame()
    coef, se = _hc1(df)
    for query in (
        "How does the impact flow through the network path from treatment to outcome?",
        "How does the treatment effect vary by segment?",
    ):
        result = tr.causal_effect_estimator(
            treatment=TREATMENT,
            outcome="outcome",
            confounders=["confounder_a"],
            estimation_data=df,
            query=query,
        )
        assert result.method == "backdoor.linear_regression", query
        assert result.standard_error == pytest.approx(se, rel=1e-9), query


@pytest.mark.parametrize(
    ("libraries", "dowhy_primary"),
    [(["dowhy", "networkx"], True), (["networkx", "dowhy"], False)],
)
def test_the_first_forced_library_is_primary(libraries, dowhy_primary) -> None:
    output = asyncio.run(
        SequentialPipeline().execute(
            {
                "query": "How does the treatment effect vary by segment?",
                "treatment_var": TREATMENT,
                "outcome_var": "outcome",
                "confounders": ["confounder_a"],
                "effect_modifiers": None,
                "data_source": "test",
                "filters": None,
                "estimation_data": _frame(),
                "mode": "sequential",
                "libraries_enabled": libraries,
                "cross_validate": None,
            }
        )
    )

    assert sorted(output["libraries_used"]) == ["dowhy", "networkx"]
    assert ("dowhy_method" in output["primary_result"]) is dowhy_primary
