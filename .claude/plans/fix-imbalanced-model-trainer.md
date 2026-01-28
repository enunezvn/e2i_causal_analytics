# Plan: Fix Model Trainer for Imbalanced Classification

## Problem Statement
The model trainer produces models that predict ALL samples as class 0 (negative), resulting in 0% recall for the minority class. This happens despite:
- AUC-ROC of 0.80 (misleading metric)
- SMOTE resampling applied to training data
- Severe class imbalance (14% minority)

## Root Causes Identified

1. **SMOTE strategy doesn't include class_weight** - When "smote" is recommended, `class_weight='balanced'` is NOT applied to the model
2. **Threshold hardcoded at 0.5** - Optimal threshold is computed (could be ~0.25) but predictions use hardcoded 0.5
3. **Optimization metric is AUC only** - HPO optimizes for AUC which doesn't penalize models that predict all negatives

## Fix Strategy: Defense in Depth

Apply multiple complementary fixes to ensure models work for imbalanced data:

### Fix 1: Always Apply class_weight When Imbalance Detected
**File**: `src/agents/ml_foundation/model_trainer/nodes/hyperparameter_tuner.py`

**Change**: In `_get_fixed_params()`, apply class_weight for ANY strategy when imbalance is detected, not just "class_weight" or "combined".

```python
# Current (line ~632-658):
if imbalance_detected and recommended_strategy in ("class_weight", "combined"):
    fixed_params["class_weight"] = "balanced"

# Fix:
if imbalance_detected:  # Always apply when imbalance detected
    fixed_params["class_weight"] = "balanced"
```

### Fix 2: Use Optimal Threshold for Validation Predictions
**File**: `src/agents/ml_foundation/model_trainer/nodes/evaluator.py`

**Change**: After computing optimal threshold, use it for final validation metrics (not hardcoded 0.5).

The function `_compute_optimal_threshold()` already exists. Modify the evaluation flow to:
1. Compute optimal threshold on validation set
2. Use that threshold for final predictions
3. Return the optimal threshold in output for deployment use

### Fix 3: Add Minority Recall to HPO Objective
**File**: `src/agents/ml_foundation/model_trainer/nodes/hyperparameter_tuner.py`

**Change**: Modify HPO objective to penalize models with 0% minority recall.

```python
# Current: Optimize for AUC only
score = roc_auc_score(y_val, y_proba)

# Fix: Combined metric that requires minimum recall
auc = roc_auc_score(y_val, y_proba)
y_pred = (y_proba >= 0.5).astype(int)
minority_recall = recall_score(y_val, y_pred, pos_label=1, zero_division=0)

# Penalize if minority recall is 0
if minority_recall == 0:
    score = auc * 0.5  # Heavy penalty
else:
    score = auc * 0.7 + minority_recall * 0.3  # Weighted combination
```

## Files to Modify

| File | Change | Priority |
|------|--------|----------|
| `src/agents/ml_foundation/model_trainer/nodes/hyperparameter_tuner.py` | Apply class_weight always + modify HPO objective | High |
| `src/agents/ml_foundation/model_trainer/nodes/evaluator.py` | Use optimal threshold for predictions | Medium |
| `scripts/run_tier0_test.py` | Return optimal_threshold in output | Low |

## Implementation Order

1. **Fix 1**: Always apply class_weight (simplest, most impactful)
2. **Fix 3**: Penalize 0% recall in HPO (ensures HPO finds useful models)
3. **Fix 2**: Use optimal threshold (improves deployed model performance)

## Verification

After implementing, run the tier0 test:
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36 \
  "cd /opt/e2i_causal_analytics && source .venv/bin/activate && \
   python scripts/run_tier0_test.py --include-bentoml 2>&1 | tail -100"
```

**Expected Results**:
- Step 5 should pass (not fail)
- Confusion matrix should show TP > 0 (some positives predicted)
- Minority recall > 10% (CONFIG.min_minority_recall)
- Model usefulness verdict: "acceptable" (not "useless")

## Rollback Plan
If fixes cause issues:
1. Each fix is independent and can be reverted separately
2. Original behavior preserved by checking `imbalance_detected` flag
3. No database schema changes required
