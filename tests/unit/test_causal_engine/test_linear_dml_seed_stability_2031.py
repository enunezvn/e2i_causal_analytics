"""Calibration pin for the PRODUCTION LinearDML nuisance config (#2031).

Measured 2026-09-12 (``docs/demos/results/2026-09-12_estimator_calibration_2031/
disproof.md``): on the seed-21 planted-truth DGP at n = 1500 (the live row cap) the
production fit at RF leaf 50 has seed-SD / mean-SE 0.20 median, max < 0.4 across the
11 planted pairs, and 1.00 coverage of the population-weighted planted truth; at
leaf 5 the median was 0.52 and one live pair's spread reached 0.89 SE, flipping its
CI-vs-zero verdict. This test fits the SAME config production uses (guarded below so
the pin cannot drift from ``nuisance_config``) and gates:

  1. seed-SD / mean-SE <= 0.5 on every pair;
  2. coverage of the weighted planted truth >= 0.9 over pairs x seeds;
  3. the test's own RF params == ``linear_dml_rf_params()`` except ``random_state``.

Runtime: 11 pairs x 4 seeds = 44 fits, ~1-2 s each on the droplet.
"""

from __future__ import annotations

import numpy as np
import pytest
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.causal_engine.nuisance_config import linear_dml_rf_params
from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator
from tests.unit.test_causal_engine.test_sensitivity_calibration import N_ROWS, _planted_pairs

pytestmark = pytest.mark.heavy_ml

SEEDS = (42, 7, 123, 2024)
MAX_SEED_SD_OVER_SE = 0.5
MIN_COVERAGE = 0.9


def _rf_params(seed: int) -> dict:
    """Production params with only ``random_state`` swapped for the sweep seed."""
    return {**linear_dml_rf_params(), "random_state": seed}


def _fit(df, treatment, outcome, covariates, seed):
    Y = df[outcome].to_numpy(dtype=float)
    T = df[treatment].to_numpy(dtype=int)
    X = df[covariates].to_numpy(dtype=float)
    m = LinearDML(
        model_y=RandomForestRegressor(**_rf_params(seed)),
        model_t=RandomForestClassifier(**_rf_params(seed)),
        discrete_treatment=True,
        random_state=seed,
    )
    m.fit(Y, T, X=X, W=X)
    inf = m.ate_inference(X)
    lo, hi = (float(v) for v in inf.conf_int_mean())
    return float(inf.mean_point), float(inf.stderr_mean), lo, hi


@pytest.fixture(scope="module")
def frame():
    cfg = GeneratorConfig(
        seed=21, n_records=N_ROWS, brand=Brand.REMIBRUTINIB, dgp_type=DGPType.HETEROGENEOUS
    )
    return PatientGenerator(cfg).generate()


def test_sweep_config_is_production_config():
    """The pin fits production's RF params verbatim, differing ONLY in random_state --
    so a future change to ``nuisance_config`` re-measures itself here."""
    prod = linear_dml_rf_params()
    assert set(prod) == {
        "n_estimators",
        "min_samples_leaf",
        "min_impurity_decrease",
        "random_state",
    }
    prod_no_seed = {k: v for k, v in prod.items() if k != "random_state"}
    for seed in SEEDS:
        mine = _rf_params(seed)
        assert mine.pop("random_state") == seed
        assert mine == prod_no_seed


@pytest.mark.timeout(600)
def test_production_config_is_seed_stable_and_covers_the_planted_truth(frame):
    pairs = _planted_pairs(frame)
    assert len(pairs) == 11, [p[:2] for p in pairs]
    truth = frame.attrs["true_ate_by_arm"]

    rows = []
    for arm, outcome, covs in pairs:
        true_ate = float(truth[arm][outcome]["ate"])
        fits = [_fit(frame, arm, outcome, covs, seed) for seed in SEEDS]
        ates = np.array([f[0] for f in fits])
        ses = np.array([f[1] for f in fits])
        covered = [lo <= true_ate <= hi for _, _, lo, hi in fits]
        rows.append(
            {
                "pair": f"{arm} -> {outcome}",
                "true": true_ate,
                "mean_ate": float(ates.mean()),
                "seed_sd": float(ates.std(ddof=1)),
                "mean_se": float(ses.mean()),
                "ratio": float(ates.std(ddof=1) / ses.mean()),
                "covered": sum(covered),
            }
        )

    print("\n#2031 production LinearDML seed stability (seed-21 DGP, n=%d)" % N_ROWS)
    print(f"{'pair':44s} {'true':>7s} {'mean':>7s} {'sd':>7s} {'se':>7s} {'sd/se':>6s} cov")
    for r in rows:
        print(
            f"{r['pair']:44s} {r['true']:7.4f} {r['mean_ate']:7.4f} {r['seed_sd']:7.4f} "
            f"{r['mean_se']:7.4f} {r['ratio']:6.2f} {r['covered']}/{len(SEEDS)}"
        )

    unstable = {r["pair"]: round(r["ratio"], 3) for r in rows if r["ratio"] > MAX_SEED_SD_OVER_SE}
    assert not unstable, f"seed-SD / SE above {MAX_SEED_SD_OVER_SE}: {unstable}"

    coverage = sum(r["covered"] for r in rows) / (len(rows) * len(SEEDS))
    print(f"coverage of the population-weighted planted truth: {coverage:.3f}")
    assert coverage >= MIN_COVERAGE, {r["pair"]: r["covered"] for r in rows}
