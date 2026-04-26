# Tier-0 Remediation — Baseline Metrics (Block 0)

Captured before any code changes on `feat/tier0-mlops-hardening`. Source: single
`python scripts/run_tier0_test.py` execution. These numbers are the
**threshold-tuned-on-test** baseline (the bug Block 1A fixes).

## Run metadata

| Field | Value |
|-------|-------|
| Branch | `feat/tier0-mlops-hardening` |
| Base SHA (off `main`) | `d9907bbaf20fd1ea7fdcc1cf9d21f1b317d77508` |
| Run timestamp | 2026-04-26 00:18:20 UTC |
| Experiment ID | `tier0_e2e_62ede312` |
| Cohort size | 1500 patients |
| Total duration | 226.4 s |
| QC gate | PASSED |
| Step status | 9 success / 1 warning / 1 failed (Step 7 MODEL DEPLOYER — out of scope per plan) |
| Run report | `docs/results/tier0_pipeline_run_20260426_001820.md` |

## Selected best model

LogisticRegression (chosen by algorithm comparison: LR=0.574 AUC > LightGBM=0.552 > RF=0.548 > XGB=0.530 on test).

## Class distribution (severe imbalance, minority ratio 14.78 %)

| Split | n | Class 0 | Class 1 |
|-------|---|---------|---------|
| Train | 900 | 767 (85.2 %) | 133 (14.8 %) |
| Validation | 300 | 256 (85.3 %) | 44 (14.7 %) |
| Test | 225 | 192 (85.3 %) | 33 (14.7 %) |

Imbalance remediation: SMOTE (resampled 900 → 1534, minority 14.8 % → 50.0 %).

## Threshold (current bug — tuned on test)

`optimal_threshold = 0.4982` chosen by `_compute_optimal_threshold(y_test, y_test_proba)`
at `src/agents/ml_foundation/model_trainer/nodes/evaluator.py:536`. Block 1A relocates this
to validation; expect AUC unchanged, precision/recall to shift.

## Validation-set metrics (n=300, evaluated at the test-tuned threshold)

| Metric | Value |
|--------|-------|
| roc_auc | 0.6942 |
| pr_auc | 0.2848 |
| accuracy | 0.6600 |
| precision | 0.2456 |
| recall | 0.6364 |
| f1_score | 0.3544 |
| f1_macro | 0.5618 |
| f1_weighted | 0.7084 |
| precision_class_1 | 0.2456 |
| recall_class_1 | 0.6364 |
| mcc | 0.2190 |
| brier_score | 0.2369 |

Validation confusion matrix at threshold 0.4982: TN=169, FP=87, FN=14, TP=30 → 117 predicted positives.

## Test-set metrics (n=225, evaluated at the test-tuned threshold)

| Metric | Value |
|--------|-------|
| auc_roc | 0.5740 |
| accuracy | 0.6600 |
| precision | 0.2065 |
| recall | 0.5758 |
| f1_score | 0.3040 |
| positive_predictions | 117 |

## Verdict (from the script's usefulness checks)

**MARGINAL** — model barely exceeds random chance.
Permutation test: `signal=RANDOM` (p=0.0700, shuffled AUC=0.4944).
Stratified 5-fold: AUC=0.6316 ± 0.0384, PR-AUC=0.2419 ± 0.0386, MCC=0.1668 ± 0.0518.
F1-optimal threshold (cross-validated): 0.4600.
ECE: 0.3398 → 0.0691 after isotonic calibration.

## Top features by SHAP

1. `prior_treatments` 0.1165
2. `hcp_visits` 0.0603
3. `data_quality_score` 0.0202
4. `geographic_region` 0.0183
5. `brand` 0.0176
6. `age_group` 0.0013

## Preamble spot-check (closes plan §preamble)

| Plan claim | Verified location | Status |
|------------|-------------------|--------|
| #6: threshold tuned on test | `evaluator.py:536` — `_compute_optimal_threshold(y_test, y_test_proba)` | ✓ matches |
| #11: leakage detection runs pre-transform | `data_preparer/graph.py:90` — `add_edge("run_ge_validation", "detect_leakage")`, `detect_leakage` routes to `transform_data` | ✓ matches |
| #4: feast schedules in beat | `workers/celery_app.py:209-225` — incremental (6h), freshness (4h), full-weekly (7d) | ✓ matches |

## What Block 1A is expected to move

After the fix, the threshold is tuned on validation, frozen, then applied to test. Predicted:

- Validation AUC unchanged (AUC is threshold-free).
- Validation precision/recall/F1 may shift (validation now used to pick threshold; metrics at that threshold land on validation itself).
- Test precision/recall will change because the test threshold is no longer fitted on test.
- `validation_metrics["chosen_threshold"]` becomes the source-of-truth, and re-runs become reproducible w.r.t. threshold selection.

Block 1A's commit will append a follow-up table comparing post-fix numbers against this baseline.
