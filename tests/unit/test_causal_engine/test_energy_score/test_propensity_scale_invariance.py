"""The energy-score propensity model must not depend on covariate UNITS.

Every selector wrapper (and ``dml_learner``) fits a
``LogisticRegressionCV(cv=3, max_iter=500)`` on the raw design to hand the
energy score its propensity scores. Measured 2026-09-22 on the Optum
biologic-persistence frame (n=15,209 x 77 mixed-unit columns: ages, counts,
one-hot dummies; ``docs/demos/results/2026-09-22_optum_biologic_persistence_cert/
timing_probe2_persistent_at_180d_g28.json``): that fit took **441.4 s** on the
unscaled design (lbfgs grinding to its 500-iteration cap, i.e. a truncated,
non-converged solution) versus **5.8 s** after ``StandardScaler`` -- 76x -- and
it was 441 of the production ``LinearDMLWrapper.fit``'s 605 s while the
LinearDML fit itself took 13 s. An L2-penalised logistic fit is not
scale-equivariant, so the unscaled model also returns DIFFERENT propensities
for the same data expressed in different units. Standardising inside the
model makes the propensity scores a function of the data alone.

Fast: n=400 rows, k=5 covariates, two fits per wrapper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.energy_score.estimator_selector import (
    ESTIMATOR_WRAPPERS,
    EstimatorConfig,
    EstimatorType,
)

RESCALE = 1e4  # e.g. an age-in-days column next to 0/1 dummies


@pytest.fixture(scope="module")
def design():
    rng = np.random.default_rng(11)
    n, k = 400, 5
    X = rng.normal(size=(n, k))
    logit = 1.5 * X[:, 0] + 0.5 * X[:, 1]
    t = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    y = 0.5 * t + X[:, 0] + 0.3 * X[:, 2] + rng.normal(size=n)
    unit = pd.DataFrame(X, columns=[f"x{i}" for i in range(k)])
    rescaled = unit.copy()
    rescaled["x0"] = rescaled["x0"] * RESCALE
    return t, y, unit, rescaled


# OrthoForestWrapper is left out: it fails to FIT this (and any unit-scale) frame
# today with a pre-existing, propensity-unrelated econml TypeError ("'<' not
# supported between instances of 'int' and 'NoneType'"), so it never reaches its
# propensity step. Forceable only (no default priority) -- reported as a follow-up.
WRAPPERS_UNDER_TEST = sorted(
    (e for e in ESTIMATOR_WRAPPERS if e is not EstimatorType.ORTHO_FOREST), key=lambda e: e.value
)


@pytest.mark.parametrize("estimator_type", WRAPPERS_UNDER_TEST)
def test_propensity_scores_are_invariant_to_covariate_units(design, estimator_type: EstimatorType):
    t, y, unit, rescaled = design
    wrapper_cls = ESTIMATOR_WRAPPERS[estimator_type]

    a = wrapper_cls(EstimatorConfig(estimator_type)).fit(t, y, unit)
    b = wrapper_cls(EstimatorConfig(estimator_type)).fit(t, y, rescaled)

    assert a.success and b.success, (a.error_message, b.error_message)
    assert a.propensity_scores is not None and b.propensity_scores is not None
    assert np.all((a.propensity_scores > 0) & (a.propensity_scores < 1))
    np.testing.assert_allclose(
        a.propensity_scores,
        b.propensity_scores,
        atol=1e-6,
        err_msg=f"{estimator_type.value}: propensity scores moved with covariate units",
    )


# --- The one wrapper whose ESTIMATE consumes the propensity (codex r2 MED) ------------


def test_forced_xlearner_estimate_is_invariant_to_covariate_units(design):
    """XLearner combines its two CATE surfaces with the propensity
    (``ps * tau_0 + (1 - ps) * tau_1``), so a unit-dependent propensity made its
    CATE/ATE unit-dependent too. Its base learners are gradient boosting
    (unit-invariant), so with a unit-invariant propensity the whole estimate is."""
    t, y, unit, rescaled = design
    wrapper_cls = ESTIMATOR_WRAPPERS[EstimatorType.X_LEARNER]

    a = wrapper_cls(EstimatorConfig(EstimatorType.X_LEARNER)).fit(t, y, unit)
    b = wrapper_cls(EstimatorConfig(EstimatorType.X_LEARNER)).fit(t, y, rescaled)

    assert a.success and b.success, (a.error_message, b.error_message)
    np.testing.assert_allclose(a.cate, b.cate, atol=1e-6)
    assert abs(a.ate - b.ate) < 1e-6
