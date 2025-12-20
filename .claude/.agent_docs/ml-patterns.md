# ML/MLOps Patterns and Anti-Patterns

Quick reference for AI agents working on ML/MLOps projects. Consult during code generation, debugging, and reviews.

## Priority Order
1. **Data Leakage Prevention** - Most critical, hardest to debug
2. **Reproducibility** - Essential for debugging and compliance
3. **Model Validation** - Prevents production failures
4. **Pipeline Reliability** - Ensures system stability
5. **Performance & Scalability** - Important but optimize last

---

## Data Leakage Anti-Patterns

### ❌ Anti-Pattern: Training on Future Data
```python
# BAD: Scaling before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)
```

### ✅ Pattern: Fit on Train, Transform on Test
```python
# GOOD: Fit scaler only on training data
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform
```

### ❌ Anti-Pattern: Target Encoding Without Holdout
```python
# BAD: Encoding using full dataset
df['category_encoded'] = df.groupby('category')['target'].transform('mean')
X_train, X_test = train_test_split(df)
```

### ✅ Pattern: Proper Target Encoding
```python
# GOOD: Encoding only from training data
X_train, X_test = train_test_split(df)
encoding_map = X_train.groupby('category')['target'].mean()
X_train['category_encoded'] = X_train['category'].map(encoding_map)
X_test['category_encoded'] = X_test['category'].map(encoding_map)
```

### ❌ Anti-Pattern: Feature Selection on Full Dataset
```python
# BAD: Feature selection sees test data
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
X_train, X_test = train_test_split(X_selected)
```

### ✅ Pattern: Feature Selection in Pipeline
```python
# GOOD: Feature selection only on training data
from sklearn.pipeline import Pipeline

X_train, X_test = train_test_split(X, y)
pipeline = Pipeline([
    ('selector', SelectKBest(score_func=f_classif, k=10)),
    ('classifier', LogisticRegression())
])
pipeline.fit(X_train, y_train)
```

---

## Reproducibility Patterns

### ✅ Pattern: Comprehensive Experiment Logging
```python
# GOOD: Log everything needed for reproduction
import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "model_type": "RandomForest",
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    })

    # Log dataset version
    mlflow.log_param("data_version", data_version)
    mlflow.log_param("data_hash", hashlib.md5(data.encode()).hexdigest())

    # Train model
    model.fit(X_train, y_train)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "train_f1": train_f1,
        "val_f1": val_f1
    })

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Log artifacts
    mlflow.log_artifact("config.yaml")
    mlflow.log_artifact("feature_importance.png")
```

### ✅ Pattern: Dependency Pinning
```python
# requirements.txt - GOOD: Pin all dependencies
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
mlflow==2.5.0

# requirements.txt - BAD: Unpinned dependencies
scikit-learn
pandas
numpy
```

### ✅ Pattern: Random Seed Management
```python
# GOOD: Set all random seeds
import numpy as np
import random
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic operations (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

## Model Validation Patterns

### ✅ Pattern: Multiple Validation Metrics
```python
# GOOD: Track multiple relevant metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred, average='weighted'),
    "recall": recall_score(y_true, y_pred, average='weighted'),
    "f1": f1_score(y_true, y_pred, average='weighted'),
    "roc_auc": roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
}

# Log all metrics
for metric_name, metric_value in metrics.items():
    mlflow.log_metric(metric_name, metric_value)
```

### ✅ Pattern: Cross-Validation
```python
# GOOD: Use cross-validation for robust estimates
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    model, X_train, y_train,
    cv=5,
    scoring='f1_weighted'
)

mlflow.log_metric("cv_mean_f1", cv_scores.mean())
mlflow.log_metric("cv_std_f1", cv_scores.std())
```

### ✅ Pattern: Threshold Optimization
```python
# GOOD: Optimize threshold for business metric
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)

# Find threshold that maximizes F1
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

mlflow.log_param("optimal_threshold", optimal_threshold)
```

---

## Feature Engineering Patterns

### ✅ Pattern: Feature Validation
```python
# GOOD: Validate features before training
def validate_features(df, expected_features):
    """Validate feature set matches expectations."""
    missing = set(expected_features) - set(df.columns)
    extra = set(df.columns) - set(expected_features)

    if missing:
        raise ValueError(f"Missing features: {missing}")
    if extra:
        logging.warning(f"Extra features: {extra}")

    # Check for null values
    null_counts = df[expected_features].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found: {null_counts[null_counts > 0]}")

    # Check for infinite values
    inf_counts = np.isinf(df[expected_features].select_dtypes(include=[np.number])).sum()
    if inf_counts.any():
        raise ValueError(f"Infinite values found: {inf_counts[inf_counts > 0]}")

validate_features(X_train, EXPECTED_FEATURES)
```

### ✅ Pattern: Feature Store Integration
```python
# GOOD: Use feature store for consistency
class FeatureStore:
    def __init__(self, version):
        self.version = version
        self.feature_definitions = self._load_definitions(version)

    def get_features(self, entity_ids, feature_names, timestamp=None):
        """Get features with point-in-time correctness."""
        if timestamp is None:
            timestamp = datetime.now()

        # Point-in-time join to prevent leakage
        features = self._point_in_time_join(
            entity_ids, feature_names, timestamp
        )
        return features

    def register_features(self, features_df, metadata):
        """Register new features with version control."""
        feature_version = f"{self.version}.{metadata['feature_group']}"
        self._save_features(features_df, feature_version, metadata)

# Usage
fs = FeatureStore(version="v1.0")
features = fs.get_features(
    entity_ids=patient_ids,
    feature_names=["age", "bmi", "medication_count"],
    timestamp=training_cutoff_date
)
```

---

## Pipeline Reliability Patterns

### ✅ Pattern: Idempotent Pipelines
```python
# GOOD: Pipeline can be safely re-run
def process_data(input_path, output_path, run_id):
    """Idempotent data processing pipeline."""
    # Check if already processed
    if os.path.exists(output_path):
        existing_run_id = load_metadata(output_path)['run_id']
        if existing_run_id == run_id:
            logging.info(f"Data already processed for run {run_id}")
            return

    # Process data
    df = pd.read_parquet(input_path)
    processed_df = transform(df)

    # Atomic write with metadata
    temp_path = f"{output_path}.tmp"
    processed_df.to_parquet(temp_path)
    save_metadata(temp_path, {'run_id': run_id, 'timestamp': datetime.now()})
    os.rename(temp_path, output_path)  # Atomic operation
```

### ✅ Pattern: Pipeline Observability
```python
# GOOD: Comprehensive pipeline monitoring
class PipelineMonitor:
    def __init__(self, pipeline_name):
        self.pipeline_name = pipeline_name
        self.metrics = {}

    def log_stage(self, stage_name, input_rows, output_rows, duration):
        """Log stage execution metrics."""
        self.metrics[stage_name] = {
            'input_rows': input_rows,
            'output_rows': output_rows,
            'rows_dropped': input_rows - output_rows,
            'drop_rate': (input_rows - output_rows) / input_rows,
            'duration_seconds': duration
        }

        # Alert on anomalies
        if self.metrics[stage_name]['drop_rate'] > 0.1:
            self._alert(f"High drop rate in {stage_name}: {self.metrics[stage_name]['drop_rate']:.2%}")

    def log_data_quality(self, stage_name, df):
        """Log data quality metrics."""
        quality_metrics = {
            'null_rate': df.isnull().sum().sum() / df.size,
            'duplicate_rate': df.duplicated().sum() / len(df),
            'numerical_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns)
        }
        self.metrics[f"{stage_name}_quality"] = quality_metrics

# Usage
monitor = PipelineMonitor("training_pipeline")
monitor.log_stage("data_load", len(raw_df), len(clean_df), load_duration)
monitor.log_data_quality("data_load", clean_df)
```

---

## Model Monitoring Patterns

### ✅ Pattern: Drift Detection
```python
# GOOD: Monitor for data and model drift
from scipy.stats import ks_2samp

class DriftMonitor:
    def __init__(self, reference_data, threshold=0.05):
        self.reference_data = reference_data
        self.threshold = threshold

    def detect_feature_drift(self, current_data):
        """Detect distribution drift using KS test."""
        drift_detected = {}

        for feature in self.reference_data.columns:
            statistic, p_value = ks_2samp(
                self.reference_data[feature],
                current_data[feature]
            )

            drift_detected[feature] = {
                'drifted': p_value < self.threshold,
                'p_value': p_value,
                'statistic': statistic
            }

        return drift_detected

    def detect_prediction_drift(self, current_predictions):
        """Detect drift in prediction distribution."""
        statistic, p_value = ks_2samp(
            self.reference_predictions,
            current_predictions
        )

        return {
            'drifted': p_value < self.threshold,
            'p_value': p_value
        }

# Usage
drift_monitor = DriftMonitor(reference_data=X_train)
drift_results = drift_monitor.detect_feature_drift(X_production)

for feature, result in drift_results.items():
    if result['drifted']:
        logging.warning(f"Drift detected in feature: {feature}")
        mlflow.log_metric(f"drift_{feature}_pvalue", result['p_value'])
```

### ✅ Pattern: Performance Monitoring
```python
# GOOD: Track model performance in production
class PerformanceMonitor:
    def __init__(self, model_version):
        self.model_version = model_version
        self.predictions = []
        self.actuals = []

    def log_prediction(self, prediction, features, timestamp):
        """Log prediction for future evaluation."""
        self.predictions.append({
            'prediction': prediction,
            'features': features,
            'timestamp': timestamp,
            'model_version': self.model_version
        })

    def log_actual(self, prediction_id, actual_value, timestamp):
        """Log actual outcome when available."""
        self.actuals.append({
            'prediction_id': prediction_id,
            'actual': actual_value,
            'timestamp': timestamp
        })

    def compute_performance(self, window_days=7):
        """Compute performance metrics over recent window."""
        cutoff = datetime.now() - timedelta(days=window_days)

        # Join predictions with actuals
        recent_predictions = [p for p in self.predictions if p['timestamp'] > cutoff]
        recent_actuals = [a for a in self.actuals if a['timestamp'] > cutoff]

        # Compute metrics
        metrics = self._compute_metrics(recent_predictions, recent_actuals)

        # Alert on degradation
        if metrics['accuracy'] < self.baseline_accuracy * 0.9:
            self._alert(f"Performance degradation detected: {metrics['accuracy']:.2%}")

        return metrics
```

---

## Testing Patterns

### ✅ Pattern: Data Validation Tests
```python
# GOOD: Comprehensive data validation
import pytest
from great_expectations import DataContext

def test_data_quality(raw_data_path):
    """Test data quality constraints."""
    df = pd.read_parquet(raw_data_path)

    # Test schema
    assert set(df.columns) == set(EXPECTED_COLUMNS), "Schema mismatch"

    # Test data types
    for col, expected_dtype in EXPECTED_DTYPES.items():
        assert df[col].dtype == expected_dtype, f"Wrong dtype for {col}"

    # Test value ranges
    assert df['age'].between(0, 120).all(), "Invalid age values"
    assert df['revenue'].ge(0).all(), "Negative revenue"

    # Test for duplicates
    assert not df.duplicated(subset='patient_id').any(), "Duplicate patients"

def test_no_data_leakage(X_train, X_test):
    """Test for data leakage between train and test."""
    # Check for row overlap
    train_ids = set(X_train['patient_id'])
    test_ids = set(X_test['patient_id'])
    overlap = train_ids & test_ids
    assert len(overlap) == 0, f"Data leakage: {len(overlap)} overlapping IDs"

    # Check temporal ordering
    assert X_train['date'].max() < X_test['date'].min(), "Temporal leakage"
```

### ✅ Pattern: Model Performance Tests
```python
# GOOD: Model performance gates
def test_model_performance(model, X_test, y_test):
    """Test model meets minimum performance requirements."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Minimum performance thresholds
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= 0.75, f"Accuracy {accuracy:.2%} below threshold"

    f1 = f1_score(y_test, y_pred, average='weighted')
    assert f1 >= 0.70, f"F1 {f1:.2%} below threshold"

    auc = roc_auc_score(y_test, y_pred_proba)
    assert auc >= 0.80, f"AUC {auc:.2%} below threshold"

def test_model_fairness(model, X_test, y_test, sensitive_feature):
    """Test model fairness across groups."""
    y_pred = model.predict(X_test)

    # Compute metrics by group
    groups = X_test[sensitive_feature].unique()
    group_metrics = {}

    for group in groups:
        mask = X_test[sensitive_feature] == group
        group_metrics[group] = accuracy_score(y_test[mask], y_pred[mask])

    # Check fairness constraint
    max_disparity = max(group_metrics.values()) - min(group_metrics.values())
    assert max_disparity < 0.1, f"Fairness violation: {max_disparity:.2%} disparity"
```

---

## Quick Reference Checklist

Before merging ML/MLOps code, verify:

### Data Handling
- [ ] No data leakage (fit/transform split correctly)
- [ ] Data versioning implemented
- [ ] Data validation tests in place
- [ ] Null/infinite value handling

### Model Development
- [ ] Experiments logged with full reproducibility info
- [ ] Multiple validation metrics tracked
- [ ] Cross-validation performed
- [ ] Random seeds set

### Feature Engineering
- [ ] Feature validation implemented
- [ ] No future data in features
- [ ] Feature documentation updated
- [ ] Feature importance analyzed

### Pipeline
- [ ] Idempotent pipeline design
- [ ] Error handling and retries
- [ ] Pipeline monitoring and alerts
- [ ] Quality gates implemented

### Testing
- [ ] Data leakage tests pass
- [ ] Model performance tests pass
- [ ] Fairness tests pass (if applicable)
- [ ] Integration tests pass

### Deployment
- [ ] Model versioning in place
- [ ] Drift monitoring configured
- [ ] Performance monitoring configured
- [ ] Rollback strategy defined

---

## Common ML/MLOps Mistakes

| Mistake | Impact | Solution |
|---------|--------|----------|
| Fitting scaler on full data | Data leakage | Fit only on training data |
| No experiment tracking | Can't reproduce results | Use MLflow/Weights & Biases |
| Unpinned dependencies | Irreproducible | Pin all dependencies |
| Single validation metric | Misleading performance | Track multiple metrics |
| No drift monitoring | Silent model degradation | Implement drift detection |
| Hardcoded thresholds | Suboptimal decisions | Optimize thresholds |
| No data versioning | Can't debug issues | Version all datasets |
| No pipeline monitoring | Can't detect failures | Comprehensive logging |
| Manual deployments | Error-prone, slow | CI/CD for ML |
| No fairness testing | Bias in production | Automated fairness tests |

---

**Last Updated**: 2025-12-17
**Version**: 1.0
