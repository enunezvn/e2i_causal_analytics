"""#2014 (codex round 3): only a SAMPLING standard error may weight the consensus.

``_se_for_library`` read CausalML's ``ate_ci_*`` as a sampling interval. It is
``std(predicted uplift) / sqrt(n)`` (``uplift/base.py`` documents ``ate_std`` as a
dispersion, not an SE), so its "precision" dominated the inverse-variance consensus.
Measured on the real Kisqali cohort (treatment_arm -> adherence_rate, n = 8,730):
DoWhy 0.1096 (HC1 SE 0.0044), EconML 0.0916 (SE 0.0047), CausalML 0.0005 with a
pseudo-SE of 1.75e-5 -> consensus 0.0005, 200x below both estimators with a real SE.
The EconML S/T/X-learner and OrthoForest-fallback intervals are the same
``std(CATE) / sqrt(n)`` construction (#1188 measured it ~50x too narrow).

Real DoWhy / EconML / CausalML on a real deterministic frame; the pipeline subclass only
keeps the final state it already builds.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from src.causal_engine.pipeline.sequential import SequentialPipeline, _se_for_library


class _StateKeepingPipeline(SequentialPipeline):
    def _create_output(self, state):  # type: ignore[no-untyped-def]
        self.final_state = state
        return super()._create_output(state)


def _run(libraries: list) -> dict:
    rng = np.random.default_rng(2014)
    n = 400
    c = rng.normal(0.0, 1.0, n)
    t = (c + rng.normal(0.0, 1.0, n) > 0.3).astype(int)
    y = 0.4 * t + 0.8 * c + rng.normal(0.0, 1.0, n)
    pipeline = _StateKeepingPipeline()
    asyncio.run(
        pipeline.execute(
            {
                "query": "Estimate the causal effect of treatment on outcome.",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "confounders": ["confounder_a"],
                "effect_modifiers": None,
                "data_source": "test",
                "filters": None,
                "estimation_data": pd.DataFrame({"treatment": t, "outcome": y, "confounder_a": c}),
                "mode": "sequential",
                "libraries_enabled": libraries,
                "cross_validate": None,
            }
        )
    )
    return pipeline.final_state


def test_causalml_contributes_no_standard_error_to_the_consensus() -> None:
    state = _run(["dowhy", "econml", "causalml"])
    assert (state.get("uplift_summary") or {}).get("ate_ci_lower") is not None

    assert _se_for_library(state, "causalml") is None
    assert _se_for_library(state, "dowhy") is not None
    assert _se_for_library(state, "econml") is not None
    assert state["consensus_weighting"] == "confidence"


def test_dowhy_and_econml_sampling_ses_still_weight_by_precision() -> None:
    state = _run(["dowhy", "econml"])

    assert state["consensus_weighting"] == "inverse_variance"


def test_an_econml_dispersion_interval_is_not_a_standard_error() -> None:
    half = 1.959963984540054 * 0.1
    interval = {"ate_ci_lower": 2.0 - half, "ate_ci_upper": 2.0 + half}
    for estimator, expected in (
        ("linear_dml", 0.1),
        ("causal_forest", 0.1),
        ("drlearner", 0.1),
        ("ols", 0.1),
        ("s_learner", None),
        ("t_learner", None),
        ("x_learner", None),
        ("ortho_forest", None),
    ):
        state = {"econml_result": {"result": {"estimator": estimator, **interval}}}
        se = _se_for_library(state, "econml")  # type: ignore[arg-type]
        if expected is None:
            assert se is None, estimator
        else:
            assert se is not None and abs(se - expected) < 1e-12, estimator
