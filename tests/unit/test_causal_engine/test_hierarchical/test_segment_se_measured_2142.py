"""#2142 — a segment's CI and SE are EconML's sampling uncertainty, or absent.

``_compute_ci`` asked ``model.effect_inference(X)`` for ``conf_int_mean``. On
econml 0.16 (pinned; the prod container runs it) that returns
``NormalInferenceResults``, which has no ``conf_int_mean`` — it lives on
``PopulationSummaryResults`` (``ate_inference``). The ``hasattr`` check therefore
always failed and every causal_forest / linear_dml segment got the fallback
``cate_mean ± z·cate_std/√n``: the spread of fitted per-unit CATEs, not the
estimator's uncertainty. Measured on the #2027 cert frame: linear_dml SE 0.0471
vs fallback 0.0041, causal_forest 0.0919 vs 0.0045. ``_se_from_ci`` repeated
the same ``std/√n`` stand-in when no CI existed.

Both now fail closed: no measured interval → ``None``, and the aggregating
bridges exclude and list the segment (#2027).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.hierarchical.segment_cate import (
    SegmentCATECalculator,
    SegmentCATEConfig,
)


def _planted_frame(n: int = 400, seed: int = 7):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    treatment = (rng.random(n) < 1.0 / (1.0 + np.exp(-0.5 * x1))).astype(int)
    outcome = 0.5 * treatment + x1 + 0.5 * x2 + rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2}), treatment, outcome


@pytest.mark.heavy_ml
@pytest.mark.asyncio
async def test_linear_dml_segment_se_is_econml_mean_effect_stderr():
    X, treatment, outcome = _planted_frame()
    calc = SegmentCATECalculator(SegmentCATEConfig(estimator_type="linear_dml", random_state=42))
    result = await calc.compute(X, treatment, outcome, segment_id=0, segment_name="s0")
    assert result.success, result.error_message

    # Positive control: an identical LinearDML fit, asked through econml's own
    # population summary.
    from econml.dml import LinearDML
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    rf = {
        "n_estimators": 50,
        "min_samples_leaf": 5,
        "min_impurity_decrease": 1e-7,
        "random_state": 42,
    }
    ref = LinearDML(
        model_y=RandomForestRegressor(**rf),
        model_t=RandomForestClassifier(**rf),
        discrete_treatment=True,
        random_state=42,
    )
    ref.fit(outcome, treatment, X=X.values, W=X.values)
    summary = ref.ate_inference(X.values)
    ref_lo, ref_hi = (float(v) for v in summary.conf_int_mean(alpha=0.05))

    assert result.cate_mean == pytest.approx(float(summary.mean_point), rel=1e-9)
    assert result.ci_lower == pytest.approx(ref_lo, rel=1e-9)
    assert result.ci_upper == pytest.approx(ref_hi, rel=1e-9)
    assert result.cate_se == pytest.approx(float(summary.stderr_mean), rel=1e-6)
    # The old stand-in (per-unit dispersion / √n) is an order of magnitude smaller.
    assert result.cate_se > 5 * result.cate_std / np.sqrt(len(X))


@pytest.mark.heavy_ml
@pytest.mark.asyncio
async def test_causal_forest_segment_se_is_not_the_dispersion_stand_in():
    X, treatment, outcome = _planted_frame()
    calc = SegmentCATECalculator(SegmentCATEConfig(estimator_type="causal_forest"))
    result = await calc.compute(X, treatment, outcome, segment_id=0, segment_name="s0")
    assert result.success, result.error_message

    assert result.cate_se is not None and result.ci_lower is not None
    assert result.ci_lower < result.cate_mean < result.ci_upper
    assert result.cate_se > 5 * result.cate_std / np.sqrt(len(X))


def test_model_without_inference_yields_no_interval():
    calc = SegmentCATECalculator(SegmentCATEConfig())
    lo, hi = calc._compute_ci(object(), X=np.zeros((20, 1)))
    assert lo is None and hi is None


def test_inference_that_raises_yields_no_interval():
    class _Broken:
        def ate_inference(self, X):
            raise ValueError("inference unavailable")

    calc = SegmentCATECalculator(SegmentCATEConfig())
    lo, hi = calc._compute_ci(_Broken(), X=np.zeros((20, 1)))
    assert lo is None and hi is None


@pytest.mark.asyncio
async def test_segment_without_an_interval_has_no_se():
    # compute_ci=False: no interval, and the CATEs vary (cate_std > 0), which is
    # exactly when the old std/√n stand-in produced a plausible-looking SE.
    X, treatment, outcome = _planted_frame(n=200)
    calc = SegmentCATECalculator(SegmentCATEConfig(estimator_type="t_learner", compute_ci=False))
    result = await calc.compute(X, treatment, outcome, segment_id=0, segment_name="s0")

    assert result.success, result.error_message
    assert result.cate_std is not None and result.cate_std > 0
    assert result.ci_lower is None and result.ci_upper is None
    assert result.cate_se is None
