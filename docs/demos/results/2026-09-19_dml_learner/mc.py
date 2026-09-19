"""Planted-truth Monte Carlo behind the dml_learner design (2026-09-19).

Reproduces every table in disproof.md. Run from the repo root:

    python docs/demos/results/2026-09-19_dml_learner/mc.py

Standalone (econml / sklearn only, no ``src`` import). ~6 min on the droplet.
DGP: X ~ N(0, I_4), P(T=1|X) = logistic(X0) (X0 confounds), Y = X0 + sin(X3)
+ tau(X) * T + N(0, 1), n = 2000. Every tau below has population ATE 0.5.
"""

import warnings

import numpy as np
from econml.dml import DML, LinearDML, NonParamDML
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")
TRUE_ATE = 0.5
SEEDS = range(25)
TAUS = {
    "constant": lambda X: np.full(len(X), 0.5),
    "linear": lambda X: 0.5 + 0.3 * X[:, 1],
    "quadratic": lambda X: 0.1 + 0.4 * X[:, 2] ** 2,
}


def dgp(seed, kind, n=2000):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    T = rng.binomial(1, 1 / (1 + np.exp(-X[:, 0])))
    Y = X[:, 0] + np.sin(X[:, 3]) + TAUS[kind](X) * T + rng.normal(size=n)
    return X, T, Y


def rf():  # the production LinearDML nuisance pair (nuisance_config, #2031)
    kw = {
        "n_estimators": 50,
        "min_samples_leaf": 50,
        "min_impurity_decrease": 1e-7,
        "random_state": 42,
    }
    return {"model_y": RandomForestRegressor(**kw), "model_t": RandomForestClassifier(**kw)}


def gb(rounds):
    kw = {"n_estimators": rounds, "max_depth": 3, "random_state": 42}
    return {"model_y": GradientBoostingRegressor(**kw), "model_t": GradientBoostingClassifier(**kw)}


def featurized_dml(nuis):
    return DML(
        **nuis,
        model_final=StatsModelsLinearRegression(fit_intercept=False),
        featurizer=PolynomialFeatures(degree=2, include_bias=False),
        discrete_treatment=True,
        random_state=42,
    )


def run(label, make, kinds):
    for kind in kinds:
        ates, covered = [], []
        for s in SEEDS:
            X, T, Y = dgp(s, kind)
            m = make()
            m.fit(Y, T, X=X, W=X)
            ates.append(float(m.effect(X).mean()))
            try:
                lo, hi = m.ate_inference(X).conf_int_mean()
                covered.append(lo <= TRUE_ATE <= hi)
            except AttributeError:
                pass
        a = np.array(ates)
        cov = f"{np.mean(covered):.2f}" if covered else "n/a (no analytic inference)"
        print(
            f"{label:34s} {kind:9s} bias {a.mean() - TRUE_ATE:+.4f}  MC sd {a.std():.4f}  coverage {cov}"
        )


if __name__ == "__main__":
    kinds = list(TAUS)
    run(
        "LinearDML, RF leaf-50 (production)",
        lambda: LinearDML(**rf(), discrete_treatment=True, random_state=42),
        kinds,
    )
    run(
        "NonParamDML(RF final), RF leaf-50",
        lambda: NonParamDML(
            **rf(),
            model_final=RandomForestRegressor(
                n_estimators=100, min_samples_leaf=20, random_state=42
            ),
            discrete_treatment=True,
            random_state=42,
        ),
        kinds,
    )
    run("DML poly-2, RF leaf-50", lambda: featurized_dml(rf()), kinds)
    run("DML poly-2, GB 100 rounds", lambda: featurized_dml(gb(100)), kinds)
    run("DML poly-2, GB 50 rounds (SHIPPED)", lambda: featurized_dml(gb(50)), kinds)
