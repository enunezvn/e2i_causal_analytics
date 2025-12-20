# Model Training Specialist

**Domain**: Model development, training, and optimization
**Bounded Context**: Model training pipeline, hyperparameter tuning, model selection
**Load This File**: When working on model training, experimentation, or model selection tasks

---

## Quick Reference

### Critical Constraints
- ✅ MUST log all experiments with full reproducibility info
- ✅ MUST use cross-validation for model selection
- ✅ MUST test for data leakage before training
- ✅ MUST validate on held-out data
- ❌ NEVER fit transformers on test data
- ❌ NEVER use test data for model selection
- ❌ NEVER skip experiment logging

### Key Patterns
- Use pipelines for preprocessing + model
- Set random seeds for reproducibility
- Track multiple metrics, not just one
- Version all models with metadata
- Test models before deployment

---

## Technology Stack for This Domain

### Required Libraries
```python
# Model training
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Model types (adjust for your stack)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
# import torch
# import tensorflow as tf

# Experiment tracking
import mlflow
import mlflow.sklearn

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
```

---

## Domain Architecture

### Training Pipeline Structure

```
Training Pipeline
├── Data Loading
│   ├── Load train/validation/test splits
│   └── Validate data quality
├── Preprocessing
│   ├── Handle missing values
│   ├── Encode categorical variables
│   ├── Scale numerical features
│   └── Feature engineering
├── Model Training
│   ├── Train candidate models
│   ├── Cross-validation
│   └── Hyperparameter tuning
├── Model Evaluation
│   ├── Evaluate on validation set
│   ├── Compute all metrics
│   └── Compare to baseline
├── Model Selection
│   ├── Choose best model
│   └── Final evaluation on test set
└── Model Packaging
    ├── Save model artifacts
    ├── Log to model registry
    └── Create model card
```

---

## Implementation Patterns

### Pattern 1: Training Pipeline with MLflow

```python
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, X_val, y_val, params):
    """
    Train a model with experiment tracking.

    CRITICAL: This function demonstrates:
    - Proper experiment logging
    - Pipeline usage (prevents data leakage)
    - Multiple metric tracking
    - Model versioning
    """
    with mlflow.start_run(run_name=params['experiment_name']):
        # Log parameters
        mlflow.log_params(params)

        # Log data version
        mlflow.log_param("data_version", params['data_version'])

        # Build preprocessing + model pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                random_state=params['random_state']
            ))
        ])

        # Train model
        pipeline.fit(X_train, y_train)

        # Evaluate on training set
        y_train_pred = pipeline.predict(X_train)
        train_metrics = compute_metrics(y_train, y_train_pred)
        for metric_name, metric_value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", metric_value)

        # Evaluate on validation set
        y_val_pred = pipeline.predict(X_val)
        val_metrics = compute_metrics(y_val, y_val_pred)
        for metric_name, metric_value in val_metrics.items():
            mlflow.log_metric(f"val_{metric_name}", metric_value)

        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            registered_model_name=params['model_name']
        )

        # Log artifacts
        mlflow.log_artifact("config.yaml")

        # Return model and metrics
        return pipeline, val_metrics


def compute_metrics(y_true, y_pred):
    """Compute comprehensive metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
```

### Pattern 2: Cross-Validation

```python
from sklearn.model_selection import cross_val_score, cross_validate

def evaluate_with_cross_validation(X_train, y_train, pipeline, cv=5):
    """
    Perform cross-validation for robust performance estimation.

    CRITICAL: Use cross-validation for:
    - Model selection
    - Hyperparameter tuning
    - Performance estimation

    NEVER use test set for these purposes!
    """
    # Multiple scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted'
    }

    # Perform cross-validation
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True
    )

    # Log cross-validation results
    for metric in scoring.keys():
        val_scores = cv_results[f'test_{metric}']
        mlflow.log_metric(f"cv_{metric}_mean", val_scores.mean())
        mlflow.log_metric(f"cv_{metric}_std", val_scores.std())

        # Check for overfitting
        train_scores = cv_results[f'train_{metric}']
        overfit_gap = train_scores.mean() - val_scores.mean()
        mlflow.log_metric(f"cv_{metric}_overfit_gap", overfit_gap)

        if overfit_gap > 0.1:
            logging.warning(f"Potential overfitting detected for {metric}: gap = {overfit_gap:.3f}")

    return cv_results
```

### Pattern 3: Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

def tune_hyperparameters(X_train, y_train, base_pipeline, param_distributions, n_iter=50):
    """
    Tune hyperparameters using randomized search.

    CRITICAL:
    - Use RandomizedSearchCV or GridSearchCV
    - Tune on training + validation only
    - Use cross-validation
    - Log all trials
    """
    # Randomized search with cross-validation
    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=5,
        scoring='f1_weighted',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # Fit search
    search.fit(X_train, y_train)

    # Log all trials to MLflow
    for i, params in enumerate(search.cv_results_['params']):
        with mlflow.start_run(run_name=f"trial_{i}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_score", search.cv_results_['mean_test_score'][i])
            mlflow.log_metric("std_cv_score", search.cv_results_['std_test_score'][i])

    # Log best parameters
    mlflow.log_params({"best_" + k: v for k, v in search.best_params_.items()})
    mlflow.log_metric("best_cv_score", search.best_score_)

    return search.best_estimator_


# Example param distributions
RF_PARAM_DISTRIBUTIONS = {
    'model__n_estimators': randint(50, 500),
    'model__max_depth': randint(3, 20),
    'model__min_samples_split': randint(2, 20),
    'model__min_samples_leaf': randint(1, 10),
    'model__max_features': ['sqrt', 'log2', None]
}
```

### Pattern 4: Model Comparison

```python
def compare_models(X_train, y_train, X_val, y_val, model_configs):
    """
    Train and compare multiple model types.

    CRITICAL:
    - Compare on same data splits
    - Use same evaluation metrics
    - Track all experiments
    - Use validation set for comparison (NOT test set)
    """
    results = {}

    for model_name, config in model_configs.items():
        print(f"\nTraining {model_name}...")

        with mlflow.start_run(run_name=model_name):
            # Build pipeline
            pipeline = Pipeline([
                ('preprocessor', config['preprocessor']),
                ('model', config['model'])
            ])

            # Train
            pipeline.fit(X_train, y_train)

            # Evaluate
            y_val_pred = pipeline.predict(X_val)
            y_val_proba = pipeline.predict_proba(X_val) if hasattr(pipeline, 'predict_proba') else None

            # Compute metrics
            metrics = {
                "accuracy": accuracy_score(y_val, y_val_pred),
                "precision": precision_score(y_val, y_val_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_val, y_val_pred, average='weighted', zero_division=0),
                "f1": f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
            }

            if y_val_proba is not None:
                metrics["roc_auc"] = roc_auc_score(y_val, y_val_proba, multi_class='ovr')

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model
            mlflow.sklearn.log_model(pipeline, f"model_{model_name}")

            results[model_name] = {
                "pipeline": pipeline,
                "metrics": metrics
            }

    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['f1'])
    print(f"\nBest model: {best_model_name}")
    print(f"F1 Score: {results[best_model_name]['metrics']['f1']:.4f}")

    return results, best_model_name


# Example model configs
MODEL_CONFIGS = {
    "logistic_regression": {
        "preprocessor": StandardScaler(),
        "model": LogisticRegression(random_state=42, max_iter=1000)
    },
    "random_forest": {
        "preprocessor": StandardScaler(),
        "model": RandomForestClassifier(n_estimators=100, random_state=42)
    },
    "gradient_boosting": {
        "preprocessor": StandardScaler(),
        "model": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
}
```

---

## Data Leakage Prevention

### Critical Checks Before Training

```python
def check_for_data_leakage(X_train, X_val, X_test):
    """
    Verify no data leakage between splits.

    CRITICAL: Run this before every training run!
    """
    # Check for row overlap
    train_ids = set(X_train.index)
    val_ids = set(X_val.index)
    test_ids = set(X_test.index)

    overlap_train_val = train_ids & val_ids
    overlap_train_test = train_ids & test_ids
    overlap_val_test = val_ids & test_ids

    assert len(overlap_train_val) == 0, f"Leakage: {len(overlap_train_val)} rows in both train and val"
    assert len(overlap_train_test) == 0, f"Leakage: {len(overlap_train_test)} rows in both train and test"
    assert len(overlap_val_test) == 0, f"Leakage: {len(overlap_val_test)} rows in both val and test"

    # Check temporal ordering (if applicable)
    if 'timestamp' in X_train.columns:
        assert X_train['timestamp'].max() < X_val['timestamp'].min(), "Temporal leakage: train overlaps with val"
        assert X_val['timestamp'].max() < X_test['timestamp'].min(), "Temporal leakage: val overlaps with test"

    print("✓ No data leakage detected")
```

---

## Model Validation

### Validation Checklist

```python
def validate_model_before_deployment(model, X_test, y_test, thresholds):
    """
    Validate model meets minimum requirements before deployment.

    CRITICAL: All checks must pass before deploying to production.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

    # Check thresholds
    failed_checks = []
    for metric_name, threshold in thresholds.items():
        if metrics.get(metric_name, 0) < threshold:
            failed_checks.append(f"{metric_name}: {metrics[metric_name]:.4f} < {threshold}")

    if failed_checks:
        raise ValueError(f"Model failed validation checks:\n" + "\n".join(failed_checks))

    print("✓ Model passed all validation checks")
    return metrics


# Example thresholds
MINIMUM_THRESHOLDS = {
    "accuracy": 0.75,
    "f1": 0.70,
    "roc_auc": 0.80
}
```

---

## Integration Points

### Contracts This Specialist Uses

1. **Data Contracts** (`.claude/contracts/data-contracts.md`)
   - Input: Training data schema
   - Output: Model predictions schema

2. **Model Contracts** (`.claude/contracts/model-contracts.md`)
   - Model interface specification
   - Model versioning requirements

3. **MLOps Pipeline Contracts** (`.claude/contracts/mlops-contracts.md`)
   - Experiment logging format
   - Model registry integration

### Handoff to Other Specialists

**To Model Evaluation Specialist**:
```yaml
handoff:
  from_specialist: model-training
  to_specialist: model-evaluation
  artifacts:
    - trained_model: <mlflow_run_id>
    - validation_metrics: <metrics_dict>
  next_steps:
    - Comprehensive evaluation on test set
    - Error analysis
    - Fairness evaluation
```

**To MLOps Pipeline Specialist**:
```yaml
handoff:
  from_specialist: model-training
  to_specialist: mlops-pipeline
  artifacts:
    - model_uri: <mlflow_model_uri>
    - model_version: <version>
  next_steps:
    - Package model for deployment
    - Set up monitoring
    - Configure rollback
```

---

## Common Mistakes to Avoid

| Mistake | Why It's Bad | Solution |
|---------|--------------|----------|
| Fitting scaler on full data | Data leakage | Use Pipeline, fit only on train |
| Using test set for model selection | Overly optimistic performance | Use validation set |
| Not logging experiments | Can't reproduce | Use MLflow for all experiments |
| Single metric optimization | Miss important tradeoffs | Track multiple metrics |
| No cross-validation | Unreliable estimates | Always use CV |
| Unpinned random seeds | Irreproducible | Set seeds everywhere |
| No hyperparameter tuning | Suboptimal models | Use GridSearch/RandomSearch |
| Training on all data | Can't validate | Always hold out test set |

---

## Testing Requirements

### Tests to Implement

```python
# tests/unit/test_model_training.py

def test_pipeline_no_data_leakage():
    """Test that pipeline doesn't fit on test data."""
    # Create mock train/test data
    X_train, X_test = create_mock_data()

    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])

    # Fit on train only
    pipeline.fit(X_train, y_train)

    # Verify scaler was fit on train data only
    assert pipeline.named_steps['scaler'].n_samples_seen_ == len(X_train)


def test_cross_validation_reproducible():
    """Test that cross-validation gives same results with same seed."""
    X, y = create_mock_data()
    model = RandomForestClassifier(random_state=42)

    scores1 = cross_val_score(model, X, y, cv=5, random_state=42)
    scores2 = cross_val_score(model, X, y, cv=5, random_state=42)

    assert np.allclose(scores1, scores2)


def test_model_meets_minimum_thresholds():
    """Test that trained model meets performance thresholds."""
    model, X_test, y_test = train_and_load_test_model()

    metrics = validate_model_before_deployment(
        model, X_test, y_test, MINIMUM_THRESHOLDS
    )

    # Should not raise ValueError
    assert metrics['f1'] >= MINIMUM_THRESHOLDS['f1']
```

---

## Quick Commands

```bash
# Train model with default config
python scripts/train_model.py --config config/model_config.yaml

# Train with hyperparameter tuning
python scripts/train_model.py --config config/model_config.yaml --tune

# Compare multiple models
python scripts/compare_models.py --config config/models_to_compare.yaml

# Validate model before deployment
python scripts/validate_model.py --model-uri <mlflow_uri> --test-data <path>
```

---

**Last Updated**: [Date]
**Version**: 1.0
**Owner**: [Team/Person]
