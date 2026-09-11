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

## `backdoor.linear_regression` interval source (T4, `scratch lr_ci.py` re-run on the same frame)

The reconstruction passes the common causes as EFFECT MODIFIERS, so DoWhy's OLS design is
`[const, T, W…, T·X…]` and `estimate.value` = `mean(predict(T=1) − predict(T=0))` =
`b_T + Σ b_TX·mean(X)`, **not** the treatment coefficient. Measured, `copay_support → treatment_initiated`
with `[insurance_access_score, disease_severity]`: `params[1]` = **−0.1234** (conf_int row [−0.271, +0.024])
vs `value` = **+0.0034**. DoWhy's own `_estimate_confidence_intervals` raises `NotImplementedError` there.
The delta-method contrast `c = mean(F(T=1) − F(T=0), axis=0)` built with DoWhy's own `_build_features`
gives `t_test(c).effect` = +0.003352 (identical) with CI [−0.0451, +0.0518] in 0.02 s; with no modifiers
it collapses to the conf_int row exactly (+0.060903, [+0.0100, +0.1118]). The node uses the contrast.

## Node fit time (`nc_fit_timing.py`, real `_fit_negative_control`, LinearDML, seed 21, n = 1500)

| arm → control | confounders | build | interval | `_fit_negative_control` | reading |
|---|---|---|---|---|---|
| copay_support → treatment_initiated | insurance_access_score, disease_severity | 1.02 s | 0.006 s | **1.12 s** | +0.0153 [−0.0329, +0.0634] n=1500 |
| psp_enrolled → treatment_initiated | disease_severity, engagement_score, academic_hcp | 1.14 s | 0.004 s | **1.17 s** | +0.0022 [−0.0452, +0.0495] n=1500 |
| rep_detailing_high → persistent_180d | academic_hcp, engagement_score | 0.95 s | 0.004 s | **0.98 s** | +0.0281 [−0.0222, +0.0784] n=1500 |

The interval point reproduces `estimate.value` to the last digit on all three; the copay_support point
matches the econml-on-DoWhy-frame measurement above (+0.0153).
