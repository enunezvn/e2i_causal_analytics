# `dml_learner`: design disproofs (2026-09-19)

`EstimatorType.DML_LEARNER` was declared in V4.2 (523a9f2a2, planned as the
"General DML framework with flexible final stage") but no wrapper was ever
registered, so nothing could build it. These are the measurements that chose how
to build it. `mc.py` in this directory reproduces every table (econml 0.16.0).
DGP: n = 2000, four covariates, X0 confounds treatment and outcome, 25 seeds per
cell, true ATE 0.5 in every cell.

## 1. A non-parametric final stage cannot be served

`NonParamDML` (GBR or RF final stage) fits, but `ate_inference` raises
`AttributeError: Only point estimates are available!`. The estimation node fails
closed on a winner with no CI (`missing_ci_bounds_on_success`), so a CI-less
`dml_learner` would fail every run it won.

Bootstrap inference is not a way out: at n = 2000 it took **148 s** for one fit,
and its `conf_int_mean` gave SE 0.41, about 8× the MC spread. That SE is the RMS
of the pointwise CATE SEs, not a sampling interval for the ATE.

**Chosen shape:** econml's general `DML` class. X passes through a degree-2
polynomial featurizer before a `StatsModelsLinearRegression` final stage. The
CATE is non-linear in X, and `ate_inference` stays analytic.

## 2. The nuisances decide the bias

| estimator | constant | linear | quadratic |
|---|---|---|---|
| LinearDML, RF leaf-50 (production) | +0.020 / 0.96 | +0.021 / 0.96 | +0.020 / 0.92 |
| NonParamDML (RF final), RF leaf-50 | +0.063 / n/a | +0.059 / n/a | +0.048 / n/a |
| DML poly-2, RF leaf-50 | +0.062 / 0.84 | +0.063 / 0.88 | +0.062 / 0.88 |
| DML poly-2, GB 100 rounds | −0.016 / 0.92 | −0.014 / 0.92 | −0.019 / 0.96 |
| **DML poly-2, GB 50 rounds (shipped)** | **−0.008 / 1.00** | **−0.005 / 1.00** | **−0.006 / 1.00** |

Each cell is bias / 95% CI coverage. MC sd is about 0.047 throughout.

With the production RF leaf-50 nuisances, *every* flexible final stage carries
about +0.06 bias, and it does so **even when the true effect is constant**. So
the bias is not heterogeneity being mis-modelled. The flexible stage is absorbing
residual confounding that the coarse nuisances left behind. Gradient-boosted
nuisances remove it.

Reusing the LinearDML nuisance pair would have been the "consistent" choice, and
it would have shipped a biased estimator with 0.84 coverage.

The shipped interval is mildly conservative. Mean reported SE is 0.052 on the
constant DGP and 0.053 on the quadratic DGP, against an MC sd of 0.049 and 0.047
(1.08× and 1.12×). That explains the 75/75 coverage. Wide is honest; narrow was
the #1188 failure.

## 3. Cost

Single wrapper fit at n = 5000 with 9 covariates (the tournament's subsample cap):

| estimator | fit |
|---|---|
| OLS | 1.0 s |
| LinearDML | 7.2 s |
| **dml_learner, GB 50** | **7.6 s** |
| DRLearner | 8.1 s |
| CausalForest | 9.7 s |
| dml_learner, GB 100 | 23.2 s |

50 rounds is no worse than 100 on bias or coverage and is about 3× cheaper. The
refutation suite re-fits the selected estimator dozens of times, so this
matters. Speed rank: 3, the same bucket as DRLearner.

## 4. Round trip

The refutation node rebuilds the estimator in DoWhy as `backdoor.econml.dml.DML`,
from the same `nuisance_config` factories. On the unit-test frame it reproduces
the wrapper's ATE within 1e-6. If the featurizer is dropped from the rebuild, the
round-trip test fails; that failure was confirmed by planting exactly that change.

## Not measured

- **Live gold-standard frames.** Everything above is synthetic planted truth.
  Behaviour on `patient_journeys` / `hcp_adoption` needs a post-deploy forced run.
- **Covariate count.** The degree-2 featurizer grows as d(d+3)/2: 54 features at
  9 covariates, 230 at 20. On very small n with many covariates, expect wide
  intervals rather than a failure.
