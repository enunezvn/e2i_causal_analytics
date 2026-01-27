# Class Imbalance Detection & Feature Names Preservation - Implementation Plan

## Objective

Address two critical issues discovered in the Tier 0 ML pipeline test:
1. **Precision/Recall = 0** - Model predicts all negatives due to 91% class imbalance
2. **Feature Names Lost** - SHAP output shows indices (`feature_0`) instead of actual feature names

The solution implements LLM-assisted class imbalance detection and remediation strategy recommendation.

---

## Issue Analysis

### Issue 1: Class Imbalance (Severity: Critical)

**Root Cause**: The `patient_journeys` dataset has 91% majority class (no discontinuation). The model learns to predict all negatives, achieving 91% accuracy but 0% precision/recall on the minority class.

**Current State**:
- `model_trainer` has 9 nodes with no imbalance handling
- No detection, no resampling, no weighted metrics
- HPO doesn't include `class_weight` or `scale_pos_weight`

### Issue 2: Feature Names Lost (Severity: Medium)

**Root Cause**: In `shap_computer.py` (lines 224-233), the code falls back to generic names when the model doesn't have `feature_names_in_`. The `feature_columns` created in `data_transformer.py:300` is not preserved through to SHAP computation.

---

## Implementation Summary

| Phase | Task | Files |
|-------|------|-------|
| 0 | Data generation (pre-req) | Generate 2000+ patient samples on droplet |
| 1 | Create new nodes | 2 new files |
| 2 | Modify existing files | 6 files |
| 3 | Feature names flow | Pipeline handoff |

**All testing performed on droplet**: `ssh -i ~/.ssh/replit enunez@138.197.4.36`

---

## Phase 1: Create New Nodes

### Task 1.1: Create `detect_class_imbalance.py`

**File**: `src/agents/ml_foundation/model_trainer/nodes/detect_class_imbalance.py`

**Purpose**: Detect class imbalance in training data and use Claude to recommend optimal remediation strategy.

**Key Components**:
```python
# Severity thresholds (based on minority class ratio)
SEVERITY_THRESHOLDS = {
    "none": 0.40,      # Minority >= 40% - no action needed
    "moderate": 0.20,  # Minority 20-40% - consider weighting
    "severe": 0.05,    # Minority 5-20% - resampling recommended
    "extreme": 0.0,    # Minority < 5% - aggressive resampling + weighting
}

async def detect_class_imbalance(state: ModelTrainerState) -> Dict[str, Any]:
    """Detect class imbalance and recommend remediation strategy.

    1. Analyzes class distribution in training data
    2. Classifies severity (none/moderate/severe/extreme)
    3. Uses Claude to recommend optimal strategy
    4. Returns strategy for downstream resampling node
    """
```

**LLM Recommendation**: Uses `claude-sonnet-4-20250514` to analyze the imbalance and recommend one of:
- `smote` - Synthetic minority oversampling
- `random_oversample` - Duplicate minority samples
- `random_undersample` - Remove majority samples
- `smote_tomek` - SMOTE + Tomek links cleaning
- `class_weight` - Use class weights only (no resampling)
- `combined` - Moderate resampling + class weights

**Fallback**: If LLM unavailable, uses `_heuristic_strategy()` based on severity and sample count.

---

### Task 1.2: Create `apply_resampling.py`

**File**: `src/agents/ml_foundation/model_trainer/nodes/apply_resampling.py`

**Purpose**: Apply the recommended resampling strategy to training data only.

**CRITICAL**: Resampling is ONLY applied to training data. Validation and test sets remain untouched.

**Key Components**:
```python
async def apply_resampling(state: ModelTrainerState) -> Dict[str, Any]:
    """Apply resampling strategy to training data.

    Uses imbalanced-learn library for robust implementations.
    """

def _apply_strategy(X, y, strategy) -> tuple[np.ndarray, np.ndarray]:
    """Apply the specified resampling strategy."""
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
```

---

## Phase 2: Modify Existing Files

### Task 2.1: Update `state.py`

**File**: `src/agents/ml_foundation/model_trainer/state.py`

**Add after line 73** (after `leakage_warnings`):

```python
    # Class Imbalance Detection
    imbalance_detected: bool  # Whether imbalance was detected
    imbalance_ratio: float  # Majority/minority ratio (e.g., 10.0 means 10:1)
    minority_ratio: float  # Minority class percentage (e.g., 0.09 for 9%)
    imbalance_severity: str  # none, moderate, severe, extreme
    class_distribution: Dict[int, int]  # {0: 800, 1: 77}
    recommended_strategy: str  # smote, random_oversample, class_weight, etc.
    strategy_rationale: str  # LLM explanation for strategy choice

    # Resampling Results
    X_train_resampled: Any  # Resampled training features
    y_train_resampled: Any  # Resampled training labels
    resampling_applied: bool  # Whether resampling was actually applied
    resampling_strategy: str  # Strategy that was applied
    original_train_shape: tuple  # Shape before resampling
    resampled_train_shape: tuple  # Shape after resampling
    original_distribution: Dict[int, int]  # Class counts before
    resampled_distribution: Dict[int, int]  # Class counts after

    # Feature Names (preserved from data_preparer)
    feature_columns: List[str]  # Original feature names from data_preparer
```

---

### Task 2.2: Update `graph.py`

**File**: `src/agents/ml_foundation/model_trainer/graph.py`

**Update imports**:
```python
from .nodes import (
    check_qc_gate,
    detect_class_imbalance,  # NEW
    apply_resampling,  # NEW
    enforce_splits,
    evaluate_model,
    fit_preprocessing,
    load_splits,
    log_to_mlflow,
    save_checkpoint,
    train_model,
    tune_hyperparameters,
)
```

**New Topology (11 nodes)**:
```
[OLD - 9 nodes]
check_qc_gate → load_splits → enforce_splits → fit_preprocessing →
tune_hyperparameters → train_model → evaluate_model → log_to_mlflow → save_checkpoint

[NEW - 11 nodes]
check_qc_gate → load_splits → enforce_splits → detect_class_imbalance →
fit_preprocessing → apply_resampling → tune_hyperparameters →
train_model → evaluate_model → log_to_mlflow → save_checkpoint
```

---

### Task 2.3: Update `hyperparameter_tuner.py`

**File**: `src/agents/ml_foundation/model_trainer/nodes/hyperparameter_tuner.py`

**Add class weight handling in HPO objective**:

```python
# Add to hyperparameters based on model type
if imbalance_detected and recommended_strategy in ("class_weight", "combined"):
    if algorithm_name in ("XGBoost", "XGBClassifier"):
        # XGBoost uses scale_pos_weight
        hyperparameters["scale_pos_weight"] = majority_count / minority_count
    elif algorithm_name in ("RandomForest", "LogisticRegression"):
        hyperparameters["class_weight"] = "balanced"
    elif algorithm_name == "LightGBM":
        hyperparameters["is_unbalance"] = True
```

---

### Task 2.4: Update `evaluator.py`

**File**: `src/agents/ml_foundation/model_trainer/nodes/evaluator.py`

**Add weighted metrics for imbalanced classification**:

```python
# Always compute these for classification
metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")

# Per-class metrics (critical for imbalanced data)
metrics["precision_class_0"] = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
metrics["precision_class_1"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
metrics["recall_class_0"] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
metrics["recall_class_1"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

# Minority class recall is key metric for imbalanced data
if imbalance_detected:
    metrics["minority_recall"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics["minority_precision"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
```

---

### Task 2.5: Update `model_trainer_node.py`

**File**: `src/agents/ml_foundation/model_trainer/nodes/model_trainer_node.py`

**Use resampled data if available**:
```python
resampling_applied = state.get("resampling_applied", False)
if resampling_applied:
    X_train = state.get("X_train_resampled")
    y_train = state.get("y_train_resampled")
else:
    X_train = state.get("X_train_preprocessed")
    y_train = state.get("train_data", {}).get("y")
```

**Set feature names on model after training**:
```python
feature_columns = state.get("feature_columns")
if feature_columns:
    try:
        model.feature_names_in_ = np.array(feature_columns)
    except (AttributeError, TypeError):
        pass  # Model doesn't support this attribute
```

---

### Task 2.6: Update `shap_computer.py`

**File**: `src/agents/ml_foundation/feature_analyzer/nodes/shap_computer.py`

**Prioritize state's feature_columns** (replace lines 224-233):

```python
# Get feature names - prioritize state's feature_columns over model attributes
feature_columns = state.get("feature_columns")

if feature_columns and len(feature_columns) > 0:
    # Use preserved feature names from data_preparer
    feature_names = list(feature_columns)
    logger.info(f"Using {len(feature_names)} feature names from state.feature_columns")
elif hasattr(loaded_model, "feature_names_in_"):
    feature_names = list(loaded_model.feature_names_in_)
elif hasattr(loaded_model, "feature_name_"):
    feature_names = list(loaded_model.feature_name_)
else:
    # Fallback: use generic names (last resort)
    n_features = loaded_model.n_features_in_ if hasattr(loaded_model, "n_features_in_") else X_sample.shape[1]
    feature_names = [f"feature_{i}" for i in range(n_features)]
    logger.warning(f"Using generic feature names - feature_columns not preserved")
```

---

## Phase 3: Feature Names Flow

Ensure `feature_columns` flows through the pipeline:

1. **data_preparer** → Creates `feature_columns` at line 300 in `data_transformer.py`
2. **model_trainer input** → Pass `feature_columns` from data_preparer output
3. **model_trainer** → Preserve in state, set on model after training
4. **feature_analyzer input** → Pass `feature_columns` from model_trainer output
5. **shap_computer** → Use state's `feature_columns` for SHAP output

---

## Files Summary

| File | Action | Changes |
|------|--------|---------|
| `nodes/detect_class_imbalance.py` | CREATE | LLM-assisted imbalance detection |
| `nodes/apply_resampling.py` | CREATE | Resampling strategies |
| `state.py` | MODIFY | Add 15+ new fields |
| `graph.py` | MODIFY | Add 2 nodes, rewire edges |
| `nodes/__init__.py` | MODIFY | Export new nodes |
| `nodes/hyperparameter_tuner.py` | MODIFY | Add class_weight to HPO |
| `nodes/evaluator.py` | MODIFY | Add weighted metrics |
| `nodes/model_trainer_node.py` | MODIFY | Use resampled data, set feature names |
| `shap_computer.py` | MODIFY | Prioritize state.feature_columns |

---

## Dependencies

### Local Development
Add to `requirements.txt`:
```
imbalanced-learn>=0.12.0
```

### Droplet Installation (if needed)

**IMPORTANT**: Check if `imbalanced-learn` is already installed before installing.

```bash
# SSH to droplet
ssh -i ~/.ssh/replit enunez@138.197.4.36

# Check if already installed
/opt/e2i_causal_analytics/.venv/bin/python -c "import imblearn; print(imblearn.__version__)"

# Only install if missing (avoid dependency conflicts)
# /opt/e2i_causal_analytics/.venv/bin/pip install imbalanced-learn>=0.12.0
```

**Note**: Per CLAUDE.md critical rules, avoid `pip install` on droplet unless necessary. The production venv has pre-configured dependencies.

---

## Pre-Implementation: Data Generation

Before implementing the class imbalance solution, generate additional data points to ensure sufficient samples for SMOTE and robust testing.

### Data Generation Requirements

**Current State**: The `patient_journeys` dataset may have insufficient minority class samples for SMOTE (requires at least 6 samples for default k_neighbors=5).

**Action**: Generate additional patient journey data with the same configuration as the original dataset.

```bash
# SSH to droplet
ssh -i ~/.ssh/replit enunez@138.197.4.36

# Navigate to project and activate venv
cd /opt/e2i_causal_analytics
source .venv/bin/activate

# Check current dataset size
python -c "
from src.repositories.sample_data import SampleDataGenerator
gen = SampleDataGenerator(seed=42)
df = gen.ml_patients(n_patients=1000)
print(f'Total samples: {len(df)}')
print(f'Class distribution: {df[\"discontinuation_flag\"].value_counts().to_dict()}')
"

# If minority class < 50 samples, regenerate with larger dataset
python -c "
from src.repositories.sample_data import SampleDataGenerator
gen = SampleDataGenerator(seed=42)
# Increase to 2000 patients for more minority samples
df = gen.ml_patients(n_patients=2000)
print(f'Total samples: {len(df)}')
print(f'Class distribution: {df[\"discontinuation_flag\"].value_counts().to_dict()}')
# Minority should be ~180 samples (9% of 2000)
"
```

**Target**: At least 50 minority class samples to ensure SMOTE has sufficient neighbors.

---

## Verification (DROPLET ONLY)

**CRITICAL**: All testing MUST be performed on the production droplet at `138.197.4.36`.

### SSH to Droplet

```bash
# Connect to droplet
ssh -i ~/.ssh/replit enunez@138.197.4.36

# Navigate and activate venv
cd /opt/e2i_causal_analytics
source .venv/bin/activate
```

### Run Tier 0 Test

```bash
# Run full Tier 0 pipeline test
python scripts/run_tier0_test.py

# Expected output:
# - Imbalance detected: True
# - Severity: severe or extreme
# - Strategy: (recommended by LLM)
# - Resampling applied: True
# - Minority recall > 0 (was 0 before)
# - SHAP features: actual names (not feature_0, feature_1)
```

### Run Unit Tests for New Nodes

```bash
# Test class imbalance detection
pytest tests/unit/agents/ml_foundation/model_trainer/nodes/test_detect_class_imbalance.py -v

# Test resampling application
pytest tests/unit/agents/ml_foundation/model_trainer/nodes/test_apply_resampling.py -v

# Test evaluator with new metrics
pytest tests/unit/agents/ml_foundation/model_trainer/nodes/test_evaluator.py -v

# Test SHAP feature names
pytest tests/unit/agents/ml_foundation/feature_analyzer/nodes/test_shap_computer.py -v
```

### Verify from Local (API Check Only)

```bash
# From local machine - check API health
curl -s http://138.197.4.36:8000/health | python3 -m json.tool

# Check service status via SSH
ssh -i ~/.ssh/replit enunez@138.197.4.36 "sudo systemctl status e2i-api"
```

---

## Success Criteria

| Checkpoint | Before | After |
|------------|--------|-------|
| Imbalance detection | None | Detected with severity |
| LLM recommendation | None | Strategy with rationale |
| Resampling | None | Applied based on strategy |
| Minority recall | 0.0 | > 0.3 |
| F1-macro | ~0.48 | > 0.60 |
| SHAP feature names | `feature_0, feature_1...` | `engagement_score, conversion_rate...` |
| Per-class metrics | None | `precision_class_0`, `recall_class_1`, etc. |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM API unavailable | Heuristic fallback in `_heuristic_strategy()` |
| SMOTE fails (too few samples) | 1) Generate 2000+ samples pre-implementation, 2) Adaptive k_neighbors based on minority count |
| Overfitting from oversampling | Combine with class weights, use validation set |
| imbalanced-learn not installed | Check on droplet first; only install if missing |
| Feature names still lost | Multiple fallback sources (state → model → generic) |
| Droplet pip conflicts | Check dependency before installing; use existing venv |
| Insufficient minority samples | Phase 0 generates 2000 patients (~180 minority samples) |

---

## Reference Articles

The implementation is informed by these sources on handling imbalanced data:
1. Neptune.ai - Comprehensive guide on imbalanced classification
2. Towards Data Science - Random oversampling and undersampling
3. Towards Data Science - Practical tips for class imbalance
4. GeeksforGeeks - Imbalanced dataset handling methods
