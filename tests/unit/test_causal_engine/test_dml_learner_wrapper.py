"""``dml_learner``: econml's general ``DML`` with a flexible (featurized) final stage.

``EstimatorType.DML_LEARNER`` was declared in V4.2 (523a9f2a2, "General DML
framework with flexible final stage") with a DB CHECK slot and a speed rank, but
no wrapper was ever registered, so nothing could instantiate it. This pins the
implementation and every seam an estimator has to cross to be servable:

  * the selector can build it (``ESTIMATOR_WRAPPERS``) -- and EVERY declared
    ``EstimatorType`` can, so a declared-but-unbuildable member cannot recur;
  * its fit reports an HONEST sampling interval (the estimation node fails closed
    on a CI-less winner, ``estimation.py`` ``missing_ci_bounds_on_success``);
  * the estimation node labels it as itself (the old ``estimator_to_method``
    default would have reported a ``dml_learner`` win as ``CausalForestDML``);
  * it is forceable through the agent API allowlist;
  * the refutation node rebuilds the SAME model in DoWhy (round trip: the
    reconstructed ATE reproduces the reported one).

Design measured 2026-09-19 (``docs/demos/results/2026-09-19_dml_learner/
disproof.md``): a non-parametric final stage (``NonParamDML``) has NO analytic
inference in econml 0.16 ("Only point estimates are available!"), and bootstrap
took 148 s at n=2000 with an RMS-pointwise interval ~8x the MC spread. The
featurized ``DML`` with a ``StatsModelsLinearRegression`` final stage has
analytic ``ate_inference``; with GradientBoosting nuisances (50 rounds) it
measured |bias| <= 0.008 and coverage 25/25 on constant / linear / quadratic
CATE DGPs (RF leaf-50 nuisances: +0.06 bias, coverage 0.84-0.88, even at a
CONSTANT effect).

Fast: ~300-row synthetic frames, no Monte Carlo.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from src.causal_engine import nuisance_config
from src.causal_engine.energy_score.estimator_selector import (
    ESTIMATOR_WRAPPERS,
    EstimatorConfig,
    EstimatorSelectorConfig,
    EstimatorType,
)
from src.causal_engine.estimator_registry import (
    AGENT_FORCEABLE_ESTIMATORS,
    FORCEABLE_ESTIMATOR_TYPE_BY_ALIAS,
    get_estimator_spec,
)

DML_METHOD = "backdoor.econml.dml.DML"


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(1919)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-(0.5 * x1 - 0.3 * x2)))).astype(int)
    y = (0.4 + 0.2 * x2**2) * t + 0.8 * x1 - 0.5 * x2 + rng.normal(scale=0.5, size=n)
    return t, y, pd.DataFrame({"x1": x1, "x2": x2})


@pytest.fixture(scope="module")
def fitted(frame):
    t, y, covariates = frame
    wrapper = ESTIMATOR_WRAPPERS[EstimatorType.DML_LEARNER](
        EstimatorConfig(EstimatorType.DML_LEARNER)
    )
    return wrapper, wrapper.fit(t, y, covariates)


# ---------------------------------------------------------------- registry


def test_every_declared_estimator_type_is_instantiable():
    """A declared ``EstimatorType`` with no wrapper is silently skipped by the
    selector (``estimator_type in ESTIMATOR_WRAPPERS`` filter), which is how
    ``dml_learner`` sat unbuildable since V4.2."""
    missing = [e.value for e in EstimatorType if e not in ESTIMATOR_WRAPPERS]
    assert missing == []
    for est_type, wrapper_cls in ESTIMATOR_WRAPPERS.items():
        assert wrapper_cls(EstimatorConfig(est_type)).estimator_type is est_type


def test_default_tournament_is_unchanged():
    """V4.2 intent: dml_learner is a SUPPORTED estimator, not a default entrant.
    Adding it to Auto would change which estimator live runs serve."""
    chain = [c.estimator_type for c in EstimatorSelectorConfig().estimators]
    assert chain == [
        EstimatorType.CAUSAL_FOREST,
        EstimatorType.LINEAR_DML,
        EstimatorType.DRLEARNER,
        EstimatorType.OLS,
    ]


# ---------------------------------------------------------------- the fit


def test_fit_is_econml_dml_with_the_shared_models(fitted):
    from econml.dml import DML, LinearDML
    from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression

    _, result = fitted
    assert result.success, result.error_message
    assert result.estimator_type is EstimatorType.DML_LEARNER
    model = result.raw_estimate
    # The general DML class itself -- not the LinearDML special case.
    assert type(model) is DML and not isinstance(model, LinearDML)
    assert isinstance(model.featurizer, nuisance_config.RankSafePolynomialFeatures)
    assert model.featurizer.get_params() == nuisance_config.dml_learner_featurizer().get_params()
    assert isinstance(model.model_final, StatsModelsLinearRegression)
    models_y = [m for fold in model.models_y for m in fold]
    models_t = [m for fold in model.models_t for m in fold]
    assert models_y and models_t
    expected = nuisance_config.dml_learner_gb_params()
    for m in models_y:
        assert isinstance(m, GradientBoostingRegressor)
        assert {k: m.get_params()[k] for k in expected} == expected
    for m in models_t:
        assert isinstance(m, GradientBoostingClassifier)
        assert {k: m.get_params()[k] for k in expected} == expected


def test_fit_reports_an_honest_sampling_interval(fitted):
    """The interval comes from econml ``ate_inference`` (analytic, via the
    statsmodels final stage) -- not the ``std(cate)/sqrt(n)`` heterogeneity
    spread #1188 measured ~50x too narrow."""
    _, result = fitted
    assert result.ate_ci_lower is not None and result.ate_ci_upper is not None
    assert result.ate_ci_lower < result.ate < result.ate_ci_upper
    assert result.ate_std is not None and result.ate_std > 0
    cate = np.asarray(result.cate, dtype=float).ravel()
    heterogeneity_spread = float(np.std(cate) / np.sqrt(cate.size))
    assert result.ate_std != pytest.approx(heterogeneity_spread)
    assert result.propensity_scores is not None


def test_fit_is_deterministic(frame, fitted):
    t, y, covariates = frame
    wrapper, first = fitted
    second = wrapper.fit(t, y, covariates)
    assert second.ate == pytest.approx(first.ate, abs=1e-12)
    assert second.ate_ci_lower == pytest.approx(first.ate_ci_lower, abs=1e-12)


def test_binary_and_one_hot_features_are_rank_safe_and_inference_valid():
    """Squares/interactions of binary indicators must not make the final-stage
    covariance singular while still returning a seemingly valid CI."""
    rng = np.random.default_rng(77)
    n = 600
    region_a = rng.binomial(1, 0.35, n)
    region_b = rng.binomial(1, 0.25, n) * (1 - region_a)
    continuous = rng.normal(size=n)
    treatment = rng.binomial(
        1, 1 / (1 + np.exp(-(0.6 * region_a - 0.3 * region_b + 0.2 * continuous)))
    )
    outcome = (
        (0.4 + 0.35 * region_a + 0.2 * continuous**2) * treatment
        + 0.7 * continuous
        + rng.normal(scale=0.7, size=n)
    )
    covariates = pd.DataFrame(
        {
            # A real mixed bool/float frame would otherwise materialize as an
            # object ndarray at the EconML boundary.
            "region_a": region_a.astype(bool),
            "region_b": region_b,
            "continuous": continuous,
        }
    )
    wrapper = ESTIMATOR_WRAPPERS[EstimatorType.DML_LEARNER](
        EstimatorConfig(EstimatorType.DML_LEARNER)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = wrapper.fit(treatment, outcome, covariates)

    assert result.success, result.error_message
    assert result.ate_ci_lower < result.ate < result.ate_ci_upper
    assert not any("inference will be invalid" in str(w.message).lower() for w in caught)
    fitted_featurizer = result.raw_estimate.featurizer_
    expanded = fitted_featurizer.transform(covariates.to_numpy(dtype=float))
    assert np.linalg.matrix_rank(expanded) == expanded.shape[1]


def test_invalid_final_stage_inference_warning_fails_closed(monkeypatch, frame):
    """EconML can warn and still return numeric intervals; those numbers must
    never be promoted as valid inference."""
    import econml.dml

    class _WarningDML:
        def __init__(self, **_kwargs):
            pass

        def fit(self, *_args, **_kwargs):
            warnings.warn(
                "Co-variance matrix is underdetermined. Inference will be invalid!",
                UserWarning,
                stacklevel=2,
            )
            return self

    monkeypatch.setattr(econml.dml, "DML", _WarningDML)
    treatment, outcome, covariates = frame
    wrapper = ESTIMATOR_WRAPPERS[EstimatorType.DML_LEARNER](
        EstimatorConfig(EstimatorType.DML_LEARNER)
    )
    result = wrapper.fit(treatment, outcome, covariates)

    assert result.success is False
    assert "inference is not identified" in (result.error_message or "")


def test_real_dml_heterogeneity_test_emits_segments_with_intervals():
    from src.causal_engine.heterogeneity import analyze_heterogeneity

    rng = np.random.default_rng(123)
    n = 500
    x = rng.normal(size=n)
    binary = rng.binomial(1, 0.4, n)
    treatment = rng.binomial(1, 1 / (1 + np.exp(-(0.4 * x + 0.4 * binary))))
    outcome = (
        (0.3 + 1.2 * x + 0.6 * binary) * treatment
        + 0.5 * x
        + 0.4 * binary
        + rng.normal(scale=0.6, size=n)
    )
    covariates = pd.DataFrame({"x": x, "binary": binary})
    wrapper = ESTIMATOR_WRAPPERS[EstimatorType.DML_LEARNER](
        EstimatorConfig(EstimatorType.DML_LEARNER)
    )
    fitted_result = wrapper.fit(treatment, outcome, covariates)
    assert fitted_result.success, fitted_result.error_message

    diagnostic = analyze_heterogeneity(
        model=fitted_result.raw_estimate,
        X=covariates.values,
        cate=fitted_result.cate,
    )
    assert diagnostic.cate_available is True
    assert diagnostic.detected is True
    assert diagnostic.p_value is not None and diagnostic.p_value < 0.05
    assert len(diagnostic.segments) == 2
    assert all(
        segment["cate_ci_lower"] < segment["cate_ci_upper"] for segment in diagnostic.segments
    )


# ---------------------------------------------------------------- estimation node


def test_estimation_node_labels_a_dml_learner_win_as_itself():
    spec = get_estimator_spec(EstimatorType.DML_LEARNER)
    assert spec.result_method == "dml_learner"


def test_dml_learner_is_forceable_end_to_end():
    """Forced through the agent API -> accepted by the node's allowlist -> mapped
    to the real selector type (so the forced run evaluates ONLY dml_learner)."""
    assert "dml_learner" in AGENT_FORCEABLE_ESTIMATORS
    assert FORCEABLE_ESTIMATOR_TYPE_BY_ALIAS["dml_learner"] is EstimatorType.DML_LEARNER


def test_dml_learner_is_in_the_sampling_interval_set():
    from src.causal_engine.pipeline.sequential import ECONML_SAMPLING_INTERVAL_ESTIMATORS

    assert "dml_learner" in ECONML_SAMPLING_INTERVAL_ESTIMATORS


# ---------------------------------------------------------------- refutation


def test_refutation_resolves_dml_learner_to_econml_dml():
    from src.agents.causal_impact.nodes.refutation import _resolve_dowhy_method

    assert _resolve_dowhy_method({"selected_estimator": "dml_learner"}) == DML_METHOD
    assert _resolve_dowhy_method({"method": "dml_learner"}) == DML_METHOD


def test_reconstruction_params_mirror_the_wrapper():
    from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression

    from src.agents.causal_impact.nodes.refutation import _reconstruction_nuisance_init_params

    p1 = _reconstruction_nuisance_init_params(DML_METHOD, discrete_treatment=True)
    p2 = _reconstruction_nuisance_init_params(DML_METHOD, discrete_treatment=True)
    assert set(p1) == {"model_y", "model_t", "model_final", "featurizer"}
    assert isinstance(p1["model_y"], GradientBoostingRegressor)
    assert isinstance(p1["model_t"], GradientBoostingClassifier)
    assert isinstance(p1["model_final"], StatsModelsLinearRegression)
    assert isinstance(p1["featurizer"], nuisance_config.RankSafePolynomialFeatures)
    for key in p1:
        assert p1[key] is not p2[key], key  # DoWhy fits them in place
    cont = _reconstruction_nuisance_init_params(DML_METHOD, discrete_treatment=False)
    assert isinstance(cont["model_t"], GradientBoostingRegressor)


def test_dml_branch_does_not_capture_the_other_dml_classes():
    """``"DML" in method`` would also match LinearDML / CausalForestDML."""
    from src.agents.causal_impact.nodes.refutation import _reconstruction_nuisance_init_params

    linear = _reconstruction_nuisance_init_params(
        "backdoor.econml.dml.LinearDML", discrete_treatment=True
    )
    assert set(linear) == {"model_y", "model_t"}
    assert (
        _reconstruction_nuisance_init_params(
            "backdoor.econml.dml.CausalForestDML", discrete_treatment=True
        )
        == {}
    )


def test_refutation_reconstruction_reproduces_the_reported_ate(frame, fitted):
    """Round trip: the DoWhy rebuild the refuters run on must be the SAME model
    whose ATE was reported -- not merely inside the 0.1 tolerance guard."""
    from src.agents.causal_impact.nodes.refutation import _reconstruct_dowhy_artifacts

    t, y, covariates = frame
    _, result = fitted
    data = covariates.assign(treatment=t, outcome=y)
    _model, _estimand, estimate = _reconstruct_dowhy_artifacts(
        data=data,
        treatment="treatment",
        outcome="outcome",
        common_causes=list(covariates.columns),
        estimation_result={
            "selected_estimator": "dml_learner",
            "method": "dml_learner",
            "ate": result.ate,
        },
    )
    assert float(estimate.value) == pytest.approx(result.ate, abs=1e-6)
