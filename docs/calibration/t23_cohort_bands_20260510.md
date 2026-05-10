# T2.3 Cohort-Derived Honest Bands — ADVISORY OBSERVABILITY ONLY

> **⚠ ADVISORY OBSERVABILITY ONLY** — these bands are computed from cohort metrics whose values were used to argue for the band's correctness. They are **NOT** thresholds for deployer enforcement. They MUST NOT be promoted to enforcement until the un-touched-cohort fit is complete. See [`t26_future_cohort_plan_20260510.md`](t26_future_cohort_plan_20260510.md) for the promotion criteria.

**Date:** 2026-05-10
**Plan:** v4 §6 G4 — calibration-protocol artifacts (codex-rescue HIGH-4 fix)
**Code surface:**
- `src/agents/ml_foundation/model_trainer/nodes/evaluator.py:254` `_emit_cohort_derived_honest_band` (helper that emits the band onto `validation_metrics`)
- `src/agents/ml_foundation/model_trainer/nodes/evaluator.py:73-98` `T2_3_HONEST_BAND_*_DEFAULT` constants (band-derivation parameters)
- Lifecycle marker: `T23_BAND_LIFECYCLE_STATE = "advisory"` (added 2026-05-10)

---

## 1. What the cohort-derived honest band is

The "honest band" is a per-cohort range `[honest_band_lo, honest_band_hi]` of test AUC values that a deployable model **could plausibly achieve without leakage**. It is derived per-run from:

- `baseline_test_auc` (stratified-dummy baseline AUC; see `_compute_baseline_test_metrics`)
- `permutation_null_p99` (upper tail of the empirical permutation null AUC distribution)
- `permutation_auc_std` (std of the permutation null distribution)
- `T2_3_HONEST_BAND_MIN_LIFT_DEFAULT = 0.05`, `MAX_LIFT_DEFAULT = 0.30`, `CEILING_DEFAULT = 0.95`, `NOISE_SIGMA_DEFAULT = 1.0`

The band derivation:
- **Lower bound** (`honest_band_lo`) = max(baseline + min_lift, perm_null_p99 + noise_sigma × perm_auc_std). I.e., the test AUC must beat the baseline by at least 5pp AND be at least 1σ above the perm-null p99 to count as "operationally meaningful + statistically distinguishable from noise."
- **Upper bound** (`honest_band_hi`) = min(ceiling, baseline + max_lift). Caps at 0.95 absolute or baseline+30pp, whichever is lower — RWD claims-only AUC > 0.95 is essentially never honest per published claims-research norms.

The band is emitted onto `validation_metrics` for every run via PR #121 — it replaces the hardcoded `[0.62, 0.68]` literal that was previously baked into `synthetic_rwd_realistic.py`.

---

## 2. Why this is advisory observability ONLY

### The band's parameters were chosen on cohorts whose results drove the choice

The four constants (`MIN_LIFT_DEFAULT`, `MAX_LIFT_DEFAULT`, `CEILING_DEFAULT`, `NOISE_SIGMA_DEFAULT`) were calibrated against:

- **Synthetic regime calibration** (PR #84): `synthetic_rwd_realistic` was tuned to land within `[0.62, 0.68]` — values that themselves were anchored to published claims-only research (psoriasis 0.67, atopic dermatitis 0.63, severe asthma 0.66; see codex 2026-05-09 CSU-benchmark research note in MEMORY.md). The synthetic-regime band was a published-literature anchor.
- **CSU n=9607 calibration** (PR #106): the observed `val_AUC=0.6592` validated that the `[0.62, 0.68]` literal was a viable target for a real cohort. The 5pp lift over the stratified-dummy baseline was confirmed empirically post-hoc.
- **Optum n=1294 calibration** (PR #116): the observed `cv_5fold_roc_auc_mean=0.6795` lands inside the same `[0.62, 0.68]` band; this is a corroboration rather than a discovery.

The constants are therefore conditioned on observed cohort metrics. **Promoting them to enforcement thresholds is data-snooping**:

- A new cohort that produces a band-violating AUC could be either (a) a leaky cohort (legitimate observability signal) or (b) a cohort whose feature-correlation structure simply differs from the calibration cohorts (false positive on the band).
- Without an un-touched cohort to test the band's fitness, we cannot distinguish (a) from (b).

### What we are willing to claim

- The band is **descriptively useful**: it surfaces the cohort-conditional honest range in a structured way.
- The band is **drift-detection-grade**: a sudden shift in the band relative to historical runs is a signal worth investigating.
- The band is **NOT enforcement-grade**: violating the band SHOULD NOT block model deployment in advisory mode.

This document formalizes the second claim. The function-level docstring on `_emit_cohort_derived_honest_band` was updated in PR for plan v4 §6 G4 to point here.

---

## 3. Drift-monitoring use case (the only allowed use)

The cohort-derived honest band is suitable for the following observability uses:

### A. Per-cohort historical drift detection

For a cohort with at least 10 prior pipeline runs, compute the median `honest_band_lo` and `honest_band_hi` across those runs. Flag runs where:

- Either bound shifts by > 5pp from the historical median.
- The band width (`honest_band_hi - honest_band_lo`) shifts by > 0.10 from historical median (suggests a change in baseline OR perm-null structure).
- The reported `honest_band_violated == True` AND the deviation is in the same direction (above/below) as the prior runs' violations.

### B. Cross-cohort sanity check

When deploying a model trained on cohort X to a new cohort Y, compare the cohort-X honest band to the cohort-Y baseline AUC. If the cohort-Y baseline alone falls **above** the cohort-X honest band's upper bound, the model trained on X likely cannot generalize to Y — the noise structure is too different.

### C. Internal anomaly detection

If `honest_band_lo > honest_band_hi` (the helper emits None for both in this case + a warning log), the cohort has a degenerate baseline + ceiling combination. Flag for cohort review; do NOT treat as a model-training failure.

### D. Cohort-comparison dashboards

The band values are JSON-promoted scalars (PR #121) suitable for inclusion in dashboards. Operators can compare bands across cohorts to visualize cohort heterogeneity.

---

## 4. What the cohort-derived bands are NOT

- **NOT a deployment gate.** `_emit_cohort_derived_honest_band` is pure observability — it does NOT mutate `success_criteria`, `success_criteria_met`, or `success_criteria_results`. The deployer (`model_deployer/nodes/registry_manager.py`) does not consume `honest_band_violated`.
- **NOT a substitute for the T2.6 deployer-input thresholds.** `compute_deployer_input_metrics` reads `permutation_pvalue`, `calibration_error`, and `cv_5fold_roc_auc_{std,mean}` — NOT the honest band bounds. See [`t26_literature_anchors_20260510.md`](t26_literature_anchors_20260510.md) for the deployer thresholds.
- **NOT a calibration certificate.** The honest band gates on AUC, not on calibration. A model whose AUC lies inside the honest band but whose ECE is 0.40 (`poor` band) is still un-deployable — the deployer enforcement (eventually T2.6c) catches this separately via the calibration-quality category.
- **NOT a validity claim about the perm-null derivation.** The band combines baseline-derived AND permutation-derived components; if the permutation test is degenerate (single-class y, missing `y_proba`), the perm-null component is dropped and the band is baseline-only. Operators should check `honest_band_perm_null_p99` is not None before treating the band as fully derived.

---

## 5. Lifecycle marker and graduation criteria

**Code constant:** `T23_BAND_LIFECYCLE_STATE = "advisory"` (in `evaluator.py`, near `_emit_cohort_derived_honest_band`). Possible future values:

- `"advisory"` — current state. Band emitted; no deployer impact; no enforcement.
- `"shadow"` — band is added to `compute_deployer_input_metrics` as an INPUT but the resulting category is observability only. (Same lifecycle stage as T2.6b shadow reporting today.)
- `"enforced"` — band violations rejected at deployer; would-be denial reasons graduate to actual denials.

**Graduation criteria (advisory → shadow):**
1. At least one un-touched cohort has been onboarded per [`t26_future_cohort_plan_20260510.md`](t26_future_cohort_plan_20260510.md) §2 Path A or Path B.
2. The new cohort's honest band derivation has been **observed for at least 10 pipeline runs** without operator complaints about the band's tightness.
3. Band-derivation constants (`MIN_LIFT_DEFAULT`, `MAX_LIFT_DEFAULT`, `CEILING_DEFAULT`, `NOISE_SIGMA_DEFAULT`) are reviewed and possibly re-fit using the new cohort's observed range.

**Graduation criteria (shadow → enforced):**
1. Shadow-mode operations have run for at least one quarter on the new cohort.
2. Domain-expert sign-off (medical leader or causal-analytics governance) on the band derivation parameters.
3. The deployer is updated to read `honest_band_violated` from `validation_metrics` and reject promotion when True.
4. A rollback plan exists: a single config flag to re-disable enforcement if the band proves over-tight in production.

---

## 6. Why the literature-anchored T2.6 thresholds are different

The deployer-input thresholds in `compute_deployer_input_metrics` (T2.6a) are anchored to peer-reviewed literature, NOT to cohort metrics. See [`t26_literature_anchors_20260510.md`](t26_literature_anchors_20260510.md) §2-§5 for the citations.

Critically:

- **Fisher α=0.05** for the permutation-test p-value (T2_6A_SIGNAL_MARGINAL_PVALUE_MAX) is a 100-year-old hypothesis-testing convention; it is independent of any cohort.
- **Naeini 2015 ECE bands** are based on the AAAI 2015 ECE evaluation; they are independent of any cohort in our corpus.
- **Bouckaert & Frank 2004 CV-stability bands** are based on PAKDD 2004 cross-validation variance research; they are independent of any cohort in our corpus.

The literature-anchored thresholds can be advisory-mode-enforced today (i.e., we generate would-be denial reasons via `compute_advisory_denial_reasons` without an actual denial). The honest band CANNOT, because the band parameters are conditioned on our cohort metrics.

---

## 7. Cross-references

- **Code:** `_emit_cohort_derived_honest_band` (`evaluator.py:254`); `T23_BAND_LIFECYCLE_STATE` (in same file).
- **Companion docs:**
  - [`t26_literature_anchors_20260510.md`](t26_literature_anchors_20260510.md) — peer-reviewed deployer thresholds.
  - [`t26_future_cohort_plan_20260510.md`](t26_future_cohort_plan_20260510.md) — promotion roadmap.
  - [`t22_perm_anchored_synth_20260510.md`](t22_perm_anchored_synth_20260510.md) — synthetic-only T2.2 sweep.
- **Test pin:** `tests/integration/test_csu_val_auc_measurement.py` enforces the historical `[0.62, 0.68]` band; it is a regression pin, NOT a deployment threshold.
