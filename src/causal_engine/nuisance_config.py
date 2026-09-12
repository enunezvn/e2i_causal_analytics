"""Single source of truth for the production LinearDML nuisance models (#2031).

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
