# Negative-control interval source (T4 disproof, 2026-09-11)

Seed-21 HETEROGENEOUS Remibrutinib frame, n = 1500, two negative-control pairs, the reconstruction's
own `CausalModel` build (`_reconstruction_nuisance_init_params`, `effect_modifiers = common_causes`).

## `estimate.get_confidence_intervals()` (`ci_availability_dowhy.py`)

| method | value | CI | cost |
|---|---|---|---|
| econml LinearDML / CausalForestDML / DRLearner | finite | **NaN** | 0 s |
| backdoor.linear_regression | finite | finite | **64–94 s** (bootstrap) |
| backdoor.propensity_score_weighting | finite | finite | **19–21 s** (bootstrap) |

Not a usable interval source for the three production methods.

## econml inference on DoWhy's own encoded effect-modifier frame (`ci_availability_econml_on_dowhy_frame.py`)

`estimate.estimator.estimator.ate_inference(np.asarray(estimate.estimator._effect_modifiers, float))`
reproduces `estimate.value` exactly on all three methods (LinearDML +0.0153 / +0.0302, DRLearner
+0.0115 / +0.0269, CausalForestDML +0.0017 / +0.0263) in ≤ 0.3 s. A hand-built X in the caller's column
order does NOT (−1.16 vs +0.03 on one pair): DoWhy sorts the effect-modifier columns. The node must use
DoWhy's frame and assert the identity before trusting the interval.
