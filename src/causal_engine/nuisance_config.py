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


class RankSafePolynomialFeatures:
    """Degree-2 features with deterministic constant/collinearity pruning.

    Plain ``PolynomialFeatures`` makes inference invalid on mixed production
    designs: for a binary indicator ``x**2 == x`` and mutually-exclusive
    one-hot interactions can be identically zero.  EconML's statsmodels final
    stage warns about the singular covariance matrix but still returns numbers.
    This sklearn-compatible transformer removes constant and linearly dependent
    columns at fit time and applies the same projection at prediction time.

    Imports stay inside methods so importing ``nuisance_config`` remains cheap.
    """

    def __init__(self, degree: int = DML_LEARNER_FEATURIZER_DEGREE, rank_tolerance: float = 1e-10):
        self.degree = degree
        self.rank_tolerance = rank_tolerance

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"degree": self.degree, "rank_tolerance": self.rank_tolerance}

    def set_params(self, **params: Any) -> "RankSafePolynomialFeatures":
        for key, value in params.items():
            if key not in {"degree", "rank_tolerance"}:
                raise ValueError(f"Unknown parameter {key!r}")
            setattr(self, key, value)
        return self

    def fit(self, X: Any, y: Any = None) -> "RankSafePolynomialFeatures":
        import numpy as np
        from scipy.linalg import qr
        from sklearn.preprocessing import PolynomialFeatures

        self.polynomial_ = PolynomialFeatures(degree=self.degree, include_bias=False)
        expanded = np.asarray(self.polynomial_.fit_transform(X), dtype=float)
        if expanded.ndim != 2 or expanded.shape[1] == 0:
            raise ValueError("dml_learner featurizer produced no candidate columns")

        scale = np.maximum(np.max(np.abs(expanded), axis=0), 1.0)
        varying = np.ptp(expanded, axis=0) > self.rank_tolerance * scale
        candidate_indices = np.flatnonzero(varying)
        if candidate_indices.size == 0:
            raise ValueError("dml_learner effect modifiers contain no varying features")

        # Rank selection must not depend on the units of the input columns
        # (for example, an age-squared term versus a binary flag).  Normalize
        # each candidate before pivoted QR, then retain the corresponding
        # unscaled polynomial columns for the actual model fit.
        candidates = expanded[:, candidate_indices]
        normalized_candidates = candidates / scale[candidate_indices]
        _q, r, pivots = qr(normalized_candidates, mode="economic", pivoting=True)
        diagonal = np.abs(np.diag(r))
        threshold = (
            self.rank_tolerance
            * max(candidates.shape)
            * (float(diagonal.max()) if diagonal.size else 1.0)
        )
        rank = int(np.sum(diagonal > threshold))
        if rank < 1:
            raise ValueError("dml_learner effect-modifier basis has zero numerical rank")

        # Keep original PolynomialFeatures order for stable coefficient names;
        # QR pivoting is used only to choose an independent subset.
        self.selected_indices_ = np.sort(candidate_indices[np.asarray(pivots[:rank], dtype=int)])
        self.n_features_in_ = int(np.asarray(X).shape[1])
        return self

    def transform(self, X: Any) -> Any:
        import numpy as np
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, ("polynomial_", "selected_indices_"))
        expanded = np.asarray(self.polynomial_.transform(X), dtype=float)
        return expanded[:, self.selected_indices_]

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, ("polynomial_", "selected_indices_"))
        names = self.polynomial_.get_feature_names_out(input_features)
        return names[self.selected_indices_]


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
    """Fresh rank-safe degree-2 featurizer for the flexible final stage.

    EconML adds the CATE intercept itself.  The transformer also removes
    constant and collinear terms (notably ``binary_x**2 == binary_x``), which
    keeps the statsmodels covariance matrix identified on encoded categoricals.
    """
    return RankSafePolynomialFeatures(degree=DML_LEARNER_FEATURIZER_DEGREE)


def dml_learner_model_final() -> Any:
    """Fresh linear final stage exposing prediction stderr (honest ATE inference).

    ``fit_intercept=False``: the intercept is econml's ``fit_cate_intercept``.
    """
    from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression

    return StatsModelsLinearRegression(fit_intercept=False)


# --------------------------------------------------------------------------
# Whole init-param dicts, one per production estimator with a DoWhy rebuild.
# The selector wrappers and ``refutation._reconstruction_nuisance_init_params``
# both call these, so the rebuilt estimator cannot drift from the reported one.
# --------------------------------------------------------------------------


def linear_dml_init_params(discrete_treatment: bool = True) -> Dict[str, Any]:
    """LinearDML nuisances; a continuous treatment gets the RF REGRESSOR for model_t."""
    return {
        "model_y": linear_dml_model_y(),
        "model_t": linear_dml_model_t() if discrete_treatment else linear_dml_model_y(),
    }


def drlearner_init_params() -> Dict[str, Any]:
    """DRLearner models: GB nuisances + a LINEAR statsmodels final stage.

    The final stage is the only one exposing prediction stderr, i.e. the only way
    DRLearner yields an honest population-ATE sampling interval (#1188); only the
    CATE(X) surface is linear-in-X.
    """
    from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    return {
        "model_regression": GradientBoostingRegressor(n_estimators=50, random_state=42),
        "model_propensity": GradientBoostingClassifier(n_estimators=50, random_state=42),
        "model_final": StatsModelsLinearRegression(),
    }


def dml_learner_init_params(discrete_treatment: bool = True) -> Dict[str, Any]:
    """The general ``DML`` (``dml_learner``) models: GB nuisances, featurized final stage."""
    return {
        "model_y": dml_learner_model_y(),
        "model_t": dml_learner_model_t(discrete_treatment=discrete_treatment),
        "model_final": dml_learner_model_final(),
        "featurizer": dml_learner_featurizer(),
    }


# --------------------------------------------------------------------------
# Energy-score propensity model. Every selector wrapper (and ``dml_learner``)
# hands the energy score a logistic propensity fit on the design it estimated
# on; the wrappers' own nuisances above are untouched by this.
#
# Why a StandardScaler pipeline (measured 2026-09-22, ``docs/demos/results/
# 2026-09-22_optum_biologic_persistence_cert/timing_probe2_*.json``): on the
# real Optum design (n=15,209 x 77 mixed-unit columns -- ages, counts, one-hot
# dummies) the bare ``LogisticRegressionCV(cv=3, max_iter=500)`` took 441.4 s
# (lbfgs grinding to its iteration cap: a truncated, non-converged solution)
# versus 5.8 s standardised -- 441 of the production LinearDML wrapper's 605 s,
# while the LinearDML fit itself took 13 s. An L2-penalised logistic fit is
# not scale-equivariant either, so standardising inside the model makes the
# propensity scores a function of the data alone, not of its units
# (tests/unit/test_causal_engine/test_energy_score/test_propensity_scale_invariance.py).
# --------------------------------------------------------------------------

PROPENSITY_CV_FOLDS = 3
PROPENSITY_MAX_ITER = 500


def propensity_model() -> Any:
    """Fresh ``StandardScaler -> LogisticRegressionCV`` pipeline for the energy-score propensity."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(cv=PROPENSITY_CV_FOLDS, max_iter=PROPENSITY_MAX_ITER),
    )
