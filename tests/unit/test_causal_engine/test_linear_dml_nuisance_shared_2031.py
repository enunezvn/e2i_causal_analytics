"""#2031: both production LinearDML sites build their RF nuisances from ONE config.

Measured 2026-09-12 (``docs/demos/results/2026-09-12_estimator_calibration_2031/
disproof.md``): with ``min_samples_leaf=5`` the production LinearDML's seed-to-seed
spread reached 0.89 SE on one live pair and flipped its CI-vs-zero verdict; at leaf
50 the spread is <= 0.38 SE on every live pair with SE / bias / coverage unchanged.
The refutation node's reconstructed-vs-reported tolerance guard only means anything
if the DoWhy rebuild fits the SAME estimator production reported, so the two sites
are pinned to ``src.causal_engine.nuisance_config`` rather than to each other.

Fast: one LinearDML fit on a 200-row synthetic frame, no DGP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.agents.causal_impact.nodes.refutation import _reconstruction_nuisance_init_params
from src.causal_engine import nuisance_config
from src.causal_engine.energy_score.estimator_selector import (
    EstimatorConfig,
    EstimatorType,
    LinearDMLWrapper,
)

LINEAR_DML_METHOD = "backdoor.econml.dml.LinearDML"


def test_shared_params_pin_leaf_50():
    params = nuisance_config.linear_dml_rf_params()
    assert params["min_samples_leaf"] == 50
    assert params == {
        "n_estimators": 50,
        "min_samples_leaf": 50,
        "min_impurity_decrease": 1e-7,
        "random_state": 42,
    }
    # Fresh dict each call -- a caller mutating one must not leak into production.
    params["min_samples_leaf"] = 1
    assert nuisance_config.linear_dml_rf_params()["min_samples_leaf"] == 50


def test_factories_return_fresh_models_with_the_shared_params():
    y1, y2 = nuisance_config.linear_dml_model_y(), nuisance_config.linear_dml_model_y()
    t1, t2 = nuisance_config.linear_dml_model_t(), nuisance_config.linear_dml_model_t()
    assert isinstance(y1, RandomForestRegressor) and not is_classifier(y1)
    assert isinstance(t1, RandomForestClassifier) and is_classifier(t1)
    assert y1 is not y2 and t1 is not t2
    expected = nuisance_config.linear_dml_rf_params()
    for model in (y1, y2, t1, t2):
        got = model.get_params()
        assert {k: got[k] for k in expected} == expected


@pytest.fixture(scope="module")
def tiny_frame():
    rng = np.random.default_rng(2031)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-(0.5 * x1 - 0.3 * x2)))).astype(int)
    y = 0.4 * t + 0.8 * x1 - 0.5 * x2 + rng.normal(scale=0.5, size=n)
    return t, y, pd.DataFrame({"x1": x1, "x2": x2})


def test_wrapper_fits_with_the_shared_nuisance_params(tiny_frame):
    """The reported estimator's FITTED nuisances (econml ``LinearDML.models_y`` /
    ``models_t`` -> nested [mc_iter][fold] lists of the underlying sklearn models,
    econml 0.16 ``dml/dml.py:269-294``) carry the shared params -- leaf 50."""
    t, y, covariates = tiny_frame
    result = LinearDMLWrapper(EstimatorConfig(EstimatorType.LINEAR_DML)).fit(t, y, covariates)
    assert result.success, result.error_message
    fitted = result.raw_estimate
    expected = nuisance_config.linear_dml_rf_params()
    models_y = [m for fold in fitted.models_y for m in fold]
    models_t = [m for fold in fitted.models_t for m in fold]
    assert models_y and models_t
    for model in models_y:
        assert isinstance(model, RandomForestRegressor)
        assert model.get_params()["min_samples_leaf"] == expected["min_samples_leaf"]
        assert {k: model.get_params()[k] for k in expected} == expected
    for model in models_t:
        assert isinstance(model, RandomForestClassifier)
        assert model.get_params()["min_samples_leaf"] == expected["min_samples_leaf"]
        assert {k: model.get_params()[k] for k in expected} == expected


def test_reconstruction_lineardml_uses_the_shared_params():
    expected = nuisance_config.linear_dml_rf_params()
    p1 = _reconstruction_nuisance_init_params(LINEAR_DML_METHOD, discrete_treatment=True)
    p2 = _reconstruction_nuisance_init_params(LINEAR_DML_METHOD, discrete_treatment=True)
    assert set(p1) == {"model_y", "model_t"}
    assert isinstance(p1["model_y"], RandomForestRegressor)
    assert isinstance(p1["model_t"], RandomForestClassifier)
    for key in ("model_y", "model_t"):
        got = p1[key].get_params()
        assert {k: got[k] for k in expected} == expected, key
        # Fresh objects each call: DoWhy fits them in place.
        assert p1[key] is not p2[key]


def test_reconstruction_other_branches_unchanged():
    """Pin the pre-#2031 behaviour of every other branch (read from the function):
    continuous-treatment LinearDML -> RF REGRESSOR for model_t (not the classifier
    factory); DRLearner -> GradientBoosting nuisances + StatsModelsLinearRegression
    final; forest / plain-linear -> ``{}``."""
    from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    cont = _reconstruction_nuisance_init_params(LINEAR_DML_METHOD, discrete_treatment=False)
    assert isinstance(cont["model_t"], RandomForestRegressor)
    assert not is_classifier(cont["model_t"])
    expected = nuisance_config.linear_dml_rf_params()
    assert {k: cont["model_t"].get_params()[k] for k in expected} == expected

    dr = _reconstruction_nuisance_init_params(
        "backdoor.econml.dr.DRLearner", discrete_treatment=True
    )
    assert set(dr) == {"model_regression", "model_propensity", "model_final"}
    assert isinstance(dr["model_regression"], GradientBoostingRegressor)
    assert isinstance(dr["model_propensity"], GradientBoostingClassifier)
    assert isinstance(dr["model_final"], StatsModelsLinearRegression)
    assert dr["model_regression"].get_params()["n_estimators"] == 50
    assert dr["model_regression"].get_params()["random_state"] == 42
    assert dr["model_propensity"].get_params()["n_estimators"] == 50
    assert dr["model_propensity"].get_params()["random_state"] == 42

    for method in ("backdoor.econml.dml.CausalForestDML", "backdoor.linear_regression"):
        assert _reconstruction_nuisance_init_params(method, discrete_treatment=True) == {}
