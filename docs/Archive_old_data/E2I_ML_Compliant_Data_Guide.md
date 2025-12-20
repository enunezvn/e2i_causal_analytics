# E2I Causal Analytics - ML-Compliant Data Guide

## Overview

This guide addresses **data leakage prevention** for the E2I Causal Analytics Dashboard. The regenerated pilot data follows strict ML best practices to ensure valid model evaluation and causal inference.

## The Problem: Data Leakage

Data leakage occurs when information from outside the training dataset is used to create the model, leading to overly optimistic performance estimates that don't generalize to production.

### Common Leakage Sources in Healthcare ML:

1. **Temporal Leakage**: Using future information to predict past events
2. **Patient Leakage**: Same patient appearing in both train and test sets
3. **Preprocessing Leakage**: Computing scaling/encoding statistics on full data
4. **Target Leakage**: Including outcome information in features

## Solution Architecture

### 1. Chronological Data Splits

```
Data Timeline: Jan 2024 ────────────────────────────────────► Sep 2025

┌─────────────────────┬─────┬───────────────┬─────┬─────────────┬─────┬──────────┐
│      TRAINING       │ GAP │  VALIDATION   │ GAP │    TEST     │ GAP │ HOLDOUT  │
│    (60% of data)    │ 7d  │  (20% data)   │ 7d  │  (15% data) │ 7d  │ (5% data)│
│  Jan 2024 - Jan 2025│     │ Jan - May 2025│     │ Jun-Sep 2025│     │ Sep 2025 │
└─────────────────────┴─────┴───────────────┴─────┴─────────────┴─────┴──────────┘
```

**Key Features:**
- 7-day gaps between splits prevent information bleeding
- Patients are assigned to ONE split only (patient-level isolation)
- All patient events contained within their assigned split
- Holdout set reserved for final model evaluation only

### 2. Patient-Level Isolation

```python
# Each patient exists in exactly ONE split
patient_split_assignments = {
    "PAT-00001": "train",
    "PAT-00002": "train",
    "PAT-00003": "validation",
    "PAT-00004": "test",
    ...
}

# All events for a patient stay with them
# NO cross-contamination between splits
```

### 3. Preprocessing Isolation

```python
# WRONG: Computing on full data
scaler = StandardScaler()
scaler.fit(all_data)  # ❌ LEAKAGE!

# CORRECT: Compute ONLY on training data
scaler = StandardScaler()
scaler.fit(train_data)  # ✅ Safe

# Apply same transformation to validation/test
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)
```

Our preprocessing metadata is stored separately:
```json
// e2i_ml_compliant_preprocessing_metadata.json
{
  "feature_means": {...},
  "feature_stds": {...},
  "computed_on_split": "train",  // ← Critical!
  "computed_timestamp": "2025-11-27T..."
}
```

## Generated Data Files

### Core Split Files

| File | Description | Use Case |
|------|-------------|----------|
| `e2i_ml_compliant_train.json` | Training data (60%) | Model training |
| `e2i_ml_compliant_validation.json` | Validation data (20%) | Hyperparameter tuning |
| `e2i_ml_compliant_test.json` | Test data (15%) | Model evaluation |
| `e2i_ml_compliant_holdout.json` | Holdout data (5%) | Final evaluation |

### Supporting Files

| File | Description |
|------|-------------|
| `e2i_ml_compliant_hcp_profiles.json` | HCP reference data (shared) |
| `e2i_ml_compliant_preprocessing_metadata.json` | Scaling/encoding params |
| `e2i_ml_compliant_leakage_audit.json` | Audit trail |
| `e2i_pipeline_config.json` | Pipeline configuration |

### CSV Files by Split

Each split has separate CSV files:
```
e2i_ml_compliant_train_patient_journeys.csv
e2i_ml_compliant_train_treatment_events.csv
e2i_ml_compliant_train_ml_predictions.csv
e2i_ml_compliant_train_triggers.csv
...
e2i_ml_compliant_validation_patient_journeys.csv
...
e2i_ml_compliant_test_patient_journeys.csv
...
```

## Usage Examples

### Loading Data with Proper Isolation

```python
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load split-specific data
def load_split_data(split_name: str):
    with open(f'e2i_ml_compliant_{split_name}.json', 'r') as f:
        return json.load(f)

train_data = load_split_data('train')
val_data = load_split_data('validation')
test_data = load_split_data('test')

# Load preprocessing metadata (computed on train only)
with open('e2i_ml_compliant_preprocessing_metadata.json', 'r') as f:
    preprocess_meta = json.load(f)

# Verify preprocessing was done on training data
assert preprocess_meta['computed_on_split'] == 'train'
```

### Building a Leakage-Free Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

# Define feature columns
numerical_features = ['confidence_score', 'prediction_value', 'treatment_effect_estimate']
categorical_features = ['brand', 'geographic_region', 'journey_stage']

# Create preprocessor - will be fit ONLY on training data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Create full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier())
])

# Prepare data
X_train = pd.DataFrame(train_data['data']['ml_predictions'])
y_train = ...  # Your target variable

X_val = pd.DataFrame(val_data['data']['ml_predictions'])
y_val = ...

# Fit on training ONLY
pipeline.fit(X_train, y_train)  # ✅ Preprocessing learned here

# Evaluate on validation (transformation uses train statistics)
val_score = pipeline.score(X_val, y_val)
```

### Time-Series Cross-Validation

For time-series data, use expanding window validation:

```python
from sklearn.model_selection import TimeSeriesSplit

# Custom time-series CV that respects patient boundaries
class PatientAwareTimeSeriesSplit:
    def __init__(self, n_splits=5, gap_days=7):
        self.n_splits = n_splits
        self.gap_days = gap_days
    
    def split(self, X, y=None, groups=None):
        """
        Groups should be patient_ids to ensure patient isolation.
        """
        unique_dates = sorted(X['journey_start_date'].unique())
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        
        # Calculate split points
        n_dates = len(unique_dates)
        fold_size = n_dates // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end + self.gap_days
            test_end = test_start + fold_size
            
            train_dates = set(unique_dates[:train_end])
            test_dates = set(unique_dates[test_start:test_end])
            
            train_idx = X[X['journey_start_date'].isin(train_dates)].index
            test_idx = X[X['journey_start_date'].isin(test_dates)].index
            
            yield train_idx, test_idx
```

### Causal Inference with DoWhy

```python
import dowhy
from dowhy import CausalModel

# Load treatment events from training split only
train_events = pd.DataFrame(train_data['data']['treatment_events'])

# Define causal model
model = CausalModel(
    data=train_events,
    treatment='brand',
    outcome='outcome_indicator',
    common_causes=['age_group', 'geographic_region', 'insurance_type']
)

# Identify causal effect
identified_estimand = model.identify_effect()

# Estimate on training data
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_matching"
)

# Validate on test data (separate evaluation)
test_events = pd.DataFrame(test_data['data']['treatment_events'])
# Apply same model structure to test data for validation
```

## Leakage Audit Report

The generator produces an audit report (`e2i_ml_compliant_leakage_audit.json`):

```json
[
  {
    "audit_id": "AUDIT-0001",
    "timestamp": "2025-11-27T...",
    "check_type": "patient_split_isolation",
    "passed": true,
    "details": "All 500 patients correctly isolated to single split",
    "severity": "info"
  },
  {
    "audit_id": "AUDIT-0002",
    "timestamp": "2025-11-27T...",
    "check_type": "preprocessing_isolation",
    "passed": true,
    "details": "Preprocessing metadata correctly computed from training data only",
    "severity": "info"
  }
]
```

### Audit Checks Performed:

| Check | Description | Severity if Failed |
|-------|-------------|-------------------|
| `patient_split_isolation` | No patient in multiple splits | Critical |
| `temporal_boundary` | Events don't exceed split dates | Warning |
| `preprocessing_isolation` | Stats computed on train only | Critical |
| `outcome_leakage` | Outcome not known at prediction time | Critical |

## Best Practices Checklist

### Data Preparation
- [ ] Use chronological splits for time-series data
- [ ] Ensure patient-level isolation (no patient in multiple splits)
- [ ] Add temporal gaps between splits (7+ days recommended)
- [ ] Keep holdout set untouched until final evaluation

### Preprocessing
- [ ] Fit scalers/encoders on training data ONLY
- [ ] Store preprocessing parameters separately
- [ ] Apply same transformations to val/test
- [ ] Document what was computed and when

### Feature Engineering
- [ ] Only use information available at prediction time
- [ ] Never include target variable in features
- [ ] Verify temporal ordering of events
- [ ] Audit for indirect leakage through proxies

### Model Evaluation
- [ ] Use validation set for hyperparameter tuning
- [ ] Use test set for final model evaluation
- [ ] Use holdout only once for production decision
- [ ] Monitor for performance degradation in production

### Causal Inference Specific
- [ ] Use pre-treatment covariates only
- [ ] Validate assumptions on separate data
- [ ] Use appropriate identification strategies
- [ ] Account for time-varying confounders

## Regenerating Data

To regenerate with different parameters:

```python
from e2i_ml_compliant_data_generator import (
    E2IMLCompliantDataGenerator, 
    SplitConfig
)

# Custom split configuration
config = SplitConfig(
    train_ratio=0.70,      # More training data
    validation_ratio=0.15,
    test_ratio=0.10,
    holdout_ratio=0.05,
    temporal_gap_days=14,  # Larger gap
    patient_level_split=True
)

# Generate larger dataset
generator = E2IMLCompliantDataGenerator(
    num_patients=2000,
    num_hcps=200,
    split_config=config,
    random_seed=42  # For reproducibility
)

generator.generate_all()
generator.save_to_files("e2i_large_pilot")
```

## Integration with E2I Dashboard

### Loading Split Data in Dashboard

```javascript
// Load training data for model training views
async function loadTrainData() {
    const response = await fetch('data/e2i_ml_compliant_train.json');
    return response.json();
}

// Load test data for evaluation views
async function loadTestData() {
    const response = await fetch('data/e2i_ml_compliant_test.json');
    return response.json();
}

// Display split information
function displaySplitInfo(metadata) {
    return `
        <div class="split-info">
            <h4>Data Split: ${metadata.split}</h4>
            <p>Period: ${metadata.split_dates.start} to ${metadata.split_dates.end}</p>
            <p>Patients: ${metadata.num_patients}</p>
        </div>
    `;
}
```

### Agent Training Protocol

Each of the 8 AI agents should follow this protocol:

1. **Training Phase**: Use ONLY `train` split data
2. **Validation Phase**: Tune hyperparameters on `validation` split
3. **Testing Phase**: Evaluate final model on `test` split
4. **Production Deployment**: Monitor against `holdout` metrics

## Summary

The regenerated pilot data addresses your data leakage concerns through:

1. **Chronological Splits** - Time-ordered data partitioning
2. **Patient Isolation** - Each patient in exactly one split
3. **Temporal Gaps** - 7-day buffer between splits
4. **Preprocessing Metadata** - Statistics computed on training only
5. **Leakage Audits** - Automated verification of data integrity

This ensures that your causal analytics models will:
- Produce realistic performance estimates
- Generalize properly to new data
- Support valid causal inference
- Meet pharmaceutical regulatory requirements

---

*Document Version: 2.0*  
*Created: November 27, 2025*  
*Last Updated: November 27, 2025*  
*Status: Complete*
