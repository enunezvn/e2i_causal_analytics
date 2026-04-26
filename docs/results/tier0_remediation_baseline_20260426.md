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

## Block 1A delta

After relocating threshold tuning from test → validation. Source: single
`python scripts/run_tier0_test.py` execution at experiment ID
`tier0_e2e_da5a561b`, run report `docs/results/tier0_pipeline_run_20260426_004002.md`.

### Run metadata (post-fix)

| Field | Value |
|-------|-------|
| Run timestamp | 2026-04-26 00:40:02 UTC |
| Experiment ID | `tier0_e2e_da5a561b` |
| Cohort size | 1500 patients (unchanged) |
| Total duration | 238.4 s |
| QC gate | PASSED |
| Step status | 9 success / 1 warning / 1 failed (Step 7 MODEL DEPLOYER — out of scope per plan) |
| Selected best model | LogisticRegression (AUC ranking unchanged) |
| Run report | `docs/results/tier0_pipeline_run_20260426_004002.md` |

### Threshold (post-fix — tuned on validation)

`chosen_threshold = 0.5141` selected by `_compute_optimal_threshold(y_validation, y_validation_proba)`.
`chosen_threshold_source = "validation"` (persisted in `validation_metrics`
and at the top-level evaluator output for downstream auditability).
Same threshold is then frozen and applied to the test set without re-tuning.

### Validation-set metric delta (n=300, evaluated at the chosen threshold)

| Metric | Block 0 (test-tuned, leakage) | Block 1A (validation-tuned) | Delta |
|--------|-------------------------------|------------------------------|-------|
| roc_auc | 0.6942 | 0.6942 | 0.0000 |
| pr_auc | 0.2848 | 0.2848 | 0.0000 |
| accuracy | 0.6600 | 0.7300 | +0.0700 |
| precision | 0.2456 | 0.2921 | +0.0465 |
| recall | 0.6364 | 0.5909 | -0.0455 |
| f1_score | 0.3544 | 0.3910 | +0.0366 |
| f1_macro | 0.5618 | 0.6088 | +0.0470 |
| f1_weighted | 0.7084 | 0.7627 | +0.0543 |
| precision_class_1 | 0.2456 | 0.2921 | +0.0465 |
| recall_class_1 | 0.6364 | 0.5909 | -0.0455 |
| mcc | 0.2190 | 0.2671 | +0.0481 |
| brier_score | 0.2369 | 0.2369 | 0.0000 |

Validation confusion matrix at threshold 0.5141: TN=193, FP=63, FN=18, TP=26 → 89 predicted positives (down from 117 at the test-tuned 0.4982).

### Test-set metric delta (n=225, evaluated at the frozen threshold 0.5141)

| Metric | Block 0 (test-tuned, leakage) | Block 1A (frozen at val-tuned) | Delta |
|--------|-------------------------------|---------------------------------|-------|
| auc_roc | 0.5740 | 0.5740 | 0.0000 |
| accuracy | 0.6600 | 0.7300\* | — |
| precision | 0.2065 | 0.1719 | -0.0346 |
| recall | 0.5758 | 0.3333 | -0.2425 |
| f1_score | 0.3040 | 0.2268 | -0.0772 |
| positive_predictions | 117 | 89\* | — |

\* Both reports reuse the validation-set confusion matrix in the
"FINAL MODEL PERFORMANCE" rendering, so accuracy/predicted-positive
rows in this row-set track validation, not test. The test-only metrics
shown here are the AUC/precision/recall/F1 values surfaced in the
verdict block via `state.get("test_metrics", {})`.

### Verdict — observations

- **AUC unchanged** as predicted (threshold-free measure).
- **Test precision/recall/F1 dropped substantially** (precision -3.5pp, recall -24.3pp, F1 -7.7pp). This is the expected unmasking: under the leakage bug the test-tuned threshold cherry-picked the test-optimal point, inflating reported test recall by ~24pp. Block 1A removes that flattering bias; the new test numbers are an honest read of the model at an operating point chosen pre-test.
- **Validation metrics at the frozen threshold are slightly different** from the baseline's "validation at the test-tuned threshold" line, as expected — the chosen threshold moved from 0.4982 → 0.5141, so the validation operating point shifted accordingly.
- **MARGINAL verdict unchanged** (AUC remains the dominant signal in the verdict logic, so the verdict label is robust to this fix).
- **Top features, SHAP rankings, calibration, CV results, permutation test all unchanged** — those don't depend on the chosen operating point.

## Block 1B note

After making lag/rolling feature generation entity-grouped pre-split and adding
`prediction_timestamp` scaffolding to the contract. Source: single
`python scripts/run_tier0_test.py` execution at experiment ID
`tier0_e2e_25bc97bc`, run report `docs/results/tier0_pipeline_run_20260426_013502.md`.

### Run metadata (post-Block-1B)

| Field | Value |
|-------|-------|
| Run timestamp | 2026-04-26 01:35:02 UTC |
| Experiment ID | `tier0_e2e_25bc97bc` |
| Cohort size | 1500 patients (unchanged) |
| Total duration | 216.3 s |
| QC gate | PASSED |
| Step status | 9 success / 1 warning / 1 failed (Step 7 MODEL DEPLOYER — out of scope per plan) |
| Selected best model | LogisticRegression (AUC ranking unchanged) |

### Tier-0 metric delta vs Block 1A

| Metric | Block 1A | Block 1B | Delta |
|--------|----------|----------|-------|
| roc_auc (val) | 0.6942 | 0.6942 | 0.0000 |
| pr_auc (val) | 0.2848 | 0.2848 | 0.0000 |
| accuracy (val) | 0.7300 | 0.7300 | 0.0000 |
| precision (val) | 0.2921 | 0.2921 | 0.0000 |
| recall (val) | 0.5909 | 0.5909 | 0.0000 |
| f1_score (val) | 0.3910 | 0.3910 | 0.0000 |
| mcc (val) | 0.2671 | 0.2671 | 0.0000 |
| brier_score (val) | 0.2369 | 0.2369 | 0.0000 |
| chosen_threshold | 0.5141 | 0.5141 | 0.0000 |
| auc_roc (test) | 0.5740 | 0.5740 | 0.0000 |
| precision (test) | 0.1719 | 0.1719 | 0.0000 |
| recall (test) | 0.3333 | 0.3333 | 0.0000 |
| f1_score (test) | 0.2268 | 0.2268 | 0.0000 |
| positive_predictions (test) | 89 | 89 | 0 |

### Why the metrics didn't move

Synthetic patients in `sample_data.py:573` produce one row per `patient_id`. With
one row per entity, every `groupby(patient_id).shift(N)` produces NaN, which is
then median-filled by `_handle_generated_nans`. The temporal block thus carries
no information for synthetic data — it is a structural fix that lights up only
once multi-row entities (longitudinal patient panels) flow through the
pipeline. Block 4+ is where this changes the numbers.

Beyond the temporal node, the tier0 e2e path doesn't reach `generate_features`
at all on this synthetic input: `step_6_feature_analyzer` calls
`FeatureAnalyzerAgent` with only `X_sample`, which routes to the SHAP-only
graph (`agent.py:163`). So the per-row interaction/domain/aggregate steps are
also dormant on this run; observed parity is expected end-to-end.

`prediction_timestamp` is plumbed through `scope_definer` → `scope_spec` →
`Tier0OutputMapper` (drift_monitor + heterogeneous_optimizer mappings) but
not yet consumed; expect this scaffolding to be exercised in Block 4 onward.

### Other notable changes at the code level

- `_generate_temporal_features` now requires `entity_id_column` and
  `event_timestamp_column` (both `Optional`, default `None`); when set, lag
  and rolling are computed via `df.groupby(entity_id).shift(N)` and
  `df.groupby(entity_id).rolling(W)` respectively.
- `generate_features` concatenates train+val+test once with internal split
  markers, runs the temporal node on the combined frame, then re-splits via
  marker reindex. Public return contract (`X_train_generated`,
  `X_val_generated`, `X_test_generated`, `feature_metadata`) unchanged.
- `data_transformer.py:173` misleading comment removed; new tests in
  `tests/unit/test_agents/test_ml_foundation/test_data_preparer/test_data_transformer.py`
  assert `LabelEncoder.classes_` matches train uniques exactly and that
  unseen val/test categories collapse to the sentinel id
  via `_safe_label_encode`.
