"""Single source of truth for the production LinearDML nuisance models (#2031)
and the ``dml_learner`` (general econml ``DML``) models (bottom of module).

Two production sites fit a ``LinearDML`` with RandomForest outcome / treatment
nuisances and MUST build them from this module:

  * ``src/causal_engine/energy_score/estimator_selector.py`` ``LinearDMLWrapper``
    -- the estimator whose ATE is reported to chat.
  * ``src/agents/causal_impact/nodes/refutation.py``
    ``_reconstruction_nuisance_init_params`` -- the DoWhy rebuild the refutation
    suite runs on. Its reconstructed-vs-reported tolerance guard only means
    anything if the rebuilt estimator matches production; a drift between the
    two sites would have the guard compare two DIFFERENT estimators.

Why ``min_samples_leaf = 50`` (measured 2026-09-12, ``docs/demos/results/
2026-09-12_estimator_calibration_2031/disproof.md``): at leaf 5 the seed-to-seed
spread of the production fit reached 0.89 SE on one live pair
(``treatment_arm -> persistent_180d``: seed 42 = 0.034 vs 6-seed mean 0.066) and
flipped its CI-vs-zero verdict; at leaf 50 the spread is <= 0.38 SE on every live
pair, every other served ATE moves <= 0.014, and SE / bias / coverage are
unchanged on the planted-truth DGP at n = 1500 and n = 5000.

No heavy imports at module level -- sklearn is imported inside the factories so
importing this module stays free for the memory-capped prod container.
"""

from __future__ import annotations

from typing import Any, Dict

LINEAR_DML_MIN_SAMPLES_LEAF = 50
LINEAR_DML_N_ESTIMATORS = 50
LINEAR_DML_MIN_IMPURITY_DECREASE = 1e-7
LINEAR_DML_RANDOM_STATE = 42


def linear_dml_rf_params() -> Dict[str, Any]:
    """The RandomForest constructor kwargs both production sites use (fresh dict)."""
    return {
        "n_estimators": LINEAR_DML_N_ESTIMATORS,
        "min_samples_leaf": LINEAR_DML_MIN_SAMPLES_LEAF,
        "min_impurity_decrease": LINEAR_DML_MIN_IMPURITY_DECREASE,
        "random_state": LINEAR_DML_RANDOM_STATE,
    }


def linear_dml_model_y() -> Any:
    """Fresh RandomForestRegressor outcome nuisance for the production LinearDML."""
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**linear_dml_rf_params())


def linear_dml_model_t() -> Any:
    """Fresh RandomForestClassifier treatment nuisance for the production LinearDML."""
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(**linear_dml_rf_params())


# --------------------------------------------------------------------------
# ``dml_learner``: econml's general ``DML`` with a flexible (featurized) final
# stage. Same two-site contract as LinearDML above: the selector's
# ``DMLLearnerWrapper`` and the refutation node's DoWhy rebuild
# (``backdoor.econml.dml.DML``) both build from these factories.
#
# Why this shape (measured 2026-09-19, ``docs/demos/results/
# 2026-09-19_dml_learner/disproof.md``):
#   * The final stage is a degree-2 polynomial featurization of X fed to a
#     ``StatsModelsLinearRegression`` -- CATE non-linear in X, yet econml's
#     analytic ``ate_inference`` still applies. A non-parametric final stage
#     (``NonParamDML`` + GBR/RF) has NO analytic inference on econml 0.16, and
#     the estimation node fails closed on a CI-less winner.
#   * Nuisances are GradientBoosting, NOT the LinearDML RF-leaf-50 pair: with
#     the RF pair the flexible final stage absorbed residual confounding
#     (+0.06 bias at a CONSTANT effect, coverage 0.84-0.90). With GB
#     nuisances (50 rounds, depth 3; 25 seeds, n=2000) bias -0.008 / -0.006
#     and coverage 25/25 on constant / quadratic-CATE DGPs. 100 rounds was no
#     better (bias -0.016 / -0.019, coverage 0.92 / 0.96) at ~2x the fit time.
# --------------------------------------------------------------------------

DML_LEARNER_GB_N_ESTIMATORS = 50
DML_LEARNER_GB_MAX_DEPTH = 3
DML_LEARNER_RANDOM_STATE = 42
DML_LEARNER_FEATURIZER_DEGREE = 2


def dml_learner_gb_params() -> Dict[str, Any]:
    """The GradientBoosting constructor kwargs both dml_learner sites use (fresh dict)."""
    return {
        "n_estimators": DML_LEARNER_GB_N_ESTIMATORS,
        "max_depth": DML_LEARNER_GB_MAX_DEPTH,
        "random_state": DML_LEARNER_RANDOM_STATE,
    }


def dml_learner_model_y() -> Any:
    """Fresh GradientBoostingRegressor outcome nuisance for dml_learner."""
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(**dml_learner_gb_params())


def dml_learner_model_t(discrete_treatment: bool = True) -> Any:
    """Fresh treatment nuisance: a classifier for a binary treatment, else a regressor."""
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    cls = GradientBoostingClassifier if discrete_treatment else GradientBoostingRegressor
    return cls(**dml_learner_gb_params())


def dml_learner_featurizer() -> Any:
    """Fresh degree-2 polynomial featurizer (the flexible part of the final stage).

    ``include_bias=False``: econml adds the CATE intercept itself
    (``fit_cate_intercept=True``); a bias column would duplicate it.
    """
    from sklearn.preprocessing import PolynomialFeatures

    return PolynomialFeatures(degree=DML_LEARNER_FEATURIZER_DEGREE, include_bias=False)


def dml_learner_model_final() -> Any:
    """Fresh linear final stage exposing prediction stderr (honest ATE inference).

    ``fit_intercept=False``: the intercept is econml's ``fit_cate_intercept``.
    """
    from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression

    return StatsModelsLinearRegression(fit_intercept=False)
