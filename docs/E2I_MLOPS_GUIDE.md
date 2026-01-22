# E2I MLOps User Guide

A comprehensive guide to the Machine Learning Operations (MLOps) pipeline in E2I Causal Analytics, covering data loading, quality control, cohort determination, model training, serving, and validation.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Loading Pipeline](#data-loading-pipeline)
3. [Data Quality Control (QC)](#data-quality-control-qc)
4. [Cohort Determination](#cohort-determination)
5. [Model Selection](#model-selection)
6. [Model Training](#model-training)
7. [Hyperparameter Optimization](#hyperparameter-optimization)
8. [Model Serving](#model-serving)
9. [Testing and Validation](#testing-and-validation)
10. [Experiment Tracking](#experiment-tracking)
11. [End-to-End Workflow](#end-to-end-workflow)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The E2I MLOps pipeline is built on a **Tier 0 (ML Foundation) Agent Architecture** with 7 specialized agents that work together to ensure reproducible, validated, and production-ready machine learning models.

### Tier 0 Agents

| Agent | Purpose | SLA |
|-------|---------|-----|
| **Scope Definer** | Define ML problem scope and target outcome | <30s |
| **Cohort Constructor** | Apply eligibility criteria to build patient cohorts | <120s for 100K patients |
| **Data Preparer** | Data validation, QC, and preprocessing | <60s |
| **Feature Analyzer** | Feature engineering and selection | <120s |
| **Model Selector** | Algorithm selection and benchmarking | <120s |
| **Model Trainer** | Training with hyperparameter optimization | Variable |
| **Model Deployer** | Packaging and deployment to BentoML | <300s |
| **Observability Connector** | MLflow and Opik integration | <10s |

### MLOps Stack

| Component | Purpose |
|-----------|---------|
| **Great Expectations** | Data quality validation |
| **Pandera** | Schema validation |
| **MLflow** | Experiment tracking and model registry |
| **Optuna** | Hyperparameter optimization |
| **BentoML** | Model serving |
| **Feast** | Feature store |
| **Opik** | LLM/Agent observability |

---

## Data Loading Pipeline

### Step 1: Generate or Import Data

E2I supports both synthetic data generation and real data import.

**Synthetic Data Generation:**
```bash
# Full dataset (5K HCPs, 25K patients, 75K treatments)
python scripts/load_synthetic_data.py

# Small dataset for testing (1/10 scale)
python scripts/load_synthetic_data.py --small

# Dry run (validate without loading)
python scripts/load_synthetic_data.py --dry-run
```

**Data Generation Process (DGP) Types:**
| Type | Description | Use Case |
|------|-------------|----------|
| `simple_linear` | Linear relationships only | Baseline testing |
| `confounded` | With confounding variables | Default, realistic |
| `heterogeneous` | Treatment heterogeneity | CATE analysis |
| `time_series` | Temporal patterns | Forecasting |
| `selection_bias` | Selection bias patterns | Bias correction |

### Step 2: Batch Loading to Supabase

The `BatchLoader` handles loading with these features:
- **Batch size:** 1,000 records per batch (configurable)
- **Retry logic:** 3 retries with exponential backoff
- **Validation:** Pre-load validation enabled by default
- **Success threshold:** 95% of records must load successfully

**Loading Order (respects foreign keys):**
```
1. hcp_profiles         (no dependencies)
2. patient_journeys     (depends on HCPs)
3. treatment_events     (depends on patients)
4. ml_predictions       (depends on patients)
5. triggers             (depends on patients and HCPs)
```

### Step 3: Verify Data Load

```bash
# Check loaded data counts
psql -c "SELECT table_name, row_count FROM information_schema.tables WHERE table_schema = 'public';"

# Or via API
curl http://localhost:8000/api/data/status
```

---

## Data Quality Control (QC)

E2I implements a **3-Layer Validation Architecture** that runs before any model training.

### Layer 1: Schema Validation (Pandera)

**Purpose:** Fast-fail on structural issues (~10ms execution)

**Checks:**
- Column existence
- Data types (int, float, string, datetime)
- Nullability constraints
- Enum value validation

**Example Schema:**
```python
class BusinessMetricsSchema(pa.DataFrameModel):
    metric_id: int = pa.Field(ge=0)
    brand_id: int = pa.Field(ge=1, le=3)
    metric_name: str = pa.Field(str_length={'min_value': 1})
    metric_value: float = pa.Field(nullable=True)
    metric_date: datetime = pa.Field()
```

### Layer 2: Quality Dimension Scoring

**Purpose:** Comprehensive quality assessment on 5 dimensions

| Dimension | Weight | Threshold | Description |
|-----------|--------|-----------|-------------|
| **Completeness** | 25% | ≥90% non-null | Missing value analysis |
| **Validity** | 25% | Type compliance | Data types, no infinity values |
| **Consistency** | 20% | Cross-split match | Column/dtype matching across splits |
| **Uniqueness** | 15% | ≤5% duplicates | Duplicate detection |
| **Timeliness** | 15% | ≤30 days stale | Data freshness |

**Overall Score Formula:**
```
score = (completeness × 0.25) + (validity × 0.25) +
        (consistency × 0.20) + (uniqueness × 0.15) +
        (timeliness × 0.15)
```

**Blocking Threshold:** `score < 0.80` prevents training

### Layer 3: Great Expectations Validation

**Purpose:** Business rule validation with industry-standard framework

**Pre-defined Expectation Suites:**
| Suite | Table | Key Expectations |
|-------|-------|------------------|
| `business_metrics` | business_metrics | Brand values in [1,2,3], metric ranges |
| `predictions` | ml_predictions | Confidence scores 0-1 |
| `triggers` | triggers | Priority levels, type validation |
| `patient_journeys` | patient_journeys | Entity references, dates |
| `causal_paths` | causal_paths | Effect strength -1 to 1 |

**Running Validation:**
```python
from src.mlops.data_quality import DataQualityValidator

validator = DataQualityValidator()
result = await validator.validate_table(
    table_name="patient_journeys",
    expectation_suite="patient_journeys"
)

print(f"Status: {result.status}")  # passed, warning, failed
print(f"Success Rate: {result.success_rate}%")
```

### Leakage Detection

E2I detects **3 types of data leakage** before training:

| Type | Detection Method | Severity |
|------|------------------|----------|
| **Temporal Leakage** | Event dates after target date | BLOCKING |
| **Target Leakage** | Features with >95% correlation to target | BLOCKING |
| **Train-Test Contamination** | Duplicate indices across splits | BLOCKING |

**Running Leakage Audit:**
```bash
python scripts/run_leakage_audit.py
```

---

## Cohort Determination

The **CohortConstructor Agent** applies explicit rule-based eligibility criteria to build patient cohorts.

### Step 1: Define Cohort Configuration

```python
from src.agents.cohort_constructor import CohortConfig, Criterion, Operator

config = CohortConfig(
    cohort_name="Remibrutinib CSU Adult Patients",
    brand="remibrutinib",
    indication="csu",
    inclusion_criteria=[
        Criterion(
            field="age_at_diagnosis",
            operator=Operator.GREATER_EQUAL,
            value=18,
            description="Adult patients (≥18 years)",
            clinical_rationale="FDA approved for adults only"
        ),
        Criterion(
            field="diagnosis_code",
            operator=Operator.IN,
            value=["L50.1", "L50.8", "L50.9"],
            description="CSU diagnosis codes"
        ),
        Criterion(
            field="urticaria_severity_uas7",
            operator=Operator.GREATER_EQUAL,
            value=16,
            description="Moderate-to-severe CSU (UAS7 ≥16)"
        ),
    ],
    exclusion_criteria=[
        Criterion(
            field="active_autoimmune",
            operator=Operator.EQUAL,
            value=True,
            description="Exclude active autoimmune conditions"
        ),
    ],
    temporal_requirements=TemporalRequirements(
        lookback_days=180,   # 6 months historical data
        followup_days=90,    # 3 months outcome observation
        index_date_field="diagnosis_date"
    )
)
```

### Step 2: Pre-Built Brand Configurations

E2I includes ready-to-use configurations for each brand:

| Brand | Indication | Key Inclusion Criteria |
|-------|------------|------------------------|
| **Remibrutinib** | CSU | Age ≥18, ICD-10 L50.*, UAS7 ≥16 |
| **Fabhalta** | PNH | Age ≥18, ICD-10 D59.5*, LDH ≥1.5x ULN |
| **Fabhalta** | C3G | Age ≥18, ICD-10 N03.6/N04.6/N05.6, Proteinuria ≥1g/day |
| **Kisqali** | HR+/HER2- BC | Age ≥18, ICD-10 C50.*, HR+, HER2-, Stage IV |

**Using Pre-Built Configs:**
```python
from src.agents.cohort_constructor import CohortConfig

config = CohortConfig.from_brand("remibrutinib", "csu")
```

### Step 3: Execute Cohort Construction

```python
from src.agents.cohort_constructor import CohortConstructor

constructor = CohortConstructor(config)
eligible_df, result = constructor.construct_cohort(patient_df)

print(f"Eligible patients: {result.eligible_count}")
print(f"Exclusion rate: {result.exclusion_rate:.1%}")
```

### Step 4: Review Eligibility Log

The audit trail shows each criterion's impact:

```json
{
  "eligibility_log": [
    {
      "criterion_name": "age_at_diagnosis",
      "criterion_type": "inclusion",
      "operator": ">=",
      "value": 18,
      "removed_count": 125,
      "remaining_count": 34875
    },
    {
      "criterion_name": "diagnosis_code",
      "criterion_type": "inclusion",
      "operator": "in",
      "value": ["L50.1", "L50.8", "L50.9"],
      "removed_count": 2340,
      "remaining_count": 32535
    }
  ]
}
```

### Supported Operators

| Operator | Symbol | Example |
|----------|--------|---------|
| `EQUAL` | `==` | `age == 18` |
| `NOT_EQUAL` | `!=` | `gender != 'M'` |
| `GREATER` | `>` | `severity > 7` |
| `GREATER_EQUAL` | `>=` | `age >= 18` |
| `LESS` | `<` | `age < 65` |
| `LESS_EQUAL` | `<=` | `severity <= 10` |
| `IN` | `in` | `code in ['L50.0', 'L50.1']` |
| `NOT_IN` | `not_in` | `code not_in ['L30.0']` |
| `BETWEEN` | `between` | `age between [18, 65]` |
| `CONTAINS` | `contains` | `notes contains 'urticaria'` |

---

## Model Selection

The **Model Selector Agent** chooses optimal algorithms based on problem type and constraints.

### Step 1: Define Selection Criteria

```python
selection_request = {
    "scope_spec": {
        "problem_type": "binary_classification",
        "target_outcome": "treatment_initiated",
        "interpretability_required": True
    },
    "constraints": {
        "max_latency_ms": 100,
        "max_memory_gb": 8
    },
    "preferences": {
        "preferred_algorithms": ["XGBoost", "LightGBM"],
        "excluded_algorithms": ["DeepLearning"]
    }
}
```

### Step 2: Available Algorithms

**12 Supported Algorithms:**

| Family | Algorithm | Interpretability | Latency | Memory |
|--------|-----------|------------------|---------|--------|
| **Causal ML** | CausalForest | 0.7 | 50ms | 4GB |
| **Causal ML** | LinearDML | 0.9 | 10ms | 1GB |
| **Boosting** | XGBoost | 0.6 | 20ms | 2GB |
| **Boosting** | LightGBM | 0.6 | 15ms | 1.5GB |
| **Ensemble** | RandomForest | 0.5 | 30ms | 3GB |
| **Linear** | LogisticRegression | 1.0 | 1ms | 0.1GB |
| **Linear** | Ridge | 1.0 | 1ms | 0.1GB |
| **Linear** | Lasso | 1.0 | 1ms | 0.1GB |

### Step 3: Selection Process

```
1. Filter by Problem Type
   └── Classification vs. Regression

2. Filter by Technical Constraints
   └── Latency, memory requirements

3. Filter by Preferences
   └── Preferred/excluded algorithms

4. Filter by Interpretability
   └── If required: score ≥ 0.7

5. Rank Candidates
   └── Historical performance, speed, interpretability

6. Optional: Run Benchmark
   └── 3-fold CV on sample data
```

### Step 4: Execute Selection

```python
from src.agents.ml_foundation.model_selector import ModelSelectorAgent

selector = ModelSelectorAgent(mode="conditional")
result = await selector.run(selection_request)

print(f"Selected: {result['model_candidate']['algorithm_name']}")
print(f"Rationale: {result['selection_rationale']}")
```

---

## Model Training

The **Model Trainer Agent** trains models with strict data governance and leakage prevention.

### Data Split Policy

E2I enforces a **60/20/15/5 split** with strict validation:

| Split | Percentage | Purpose |
|-------|------------|---------|
| **Train** | 60% ± 2% | Primary model training |
| **Validation** | 20% ± 2% | Hyperparameter tuning, early stopping |
| **Test** | 15% ± 2% | Final evaluation (touched ONCE) |
| **Holdout** | 5% ± 2% | Post-deployment validation |

### Training Workflow

```
START
  ↓
check_qc_gate (MANDATORY)
  ↓
load_splits (enforce 60/20/15/5)
  ↓
enforce_splits (validate ratios ± 2%)
  ↓
fit_preprocessing (ONLY on train set)
  ↓
tune_hyperparameters (Optuna on validation)
  ↓
train_model (train set + best params)
  ↓
evaluate_model (all splits)
  ↓
log_to_mlflow (experiment tracking)
  ↓
save_checkpoint (model persistence)
  ↓
END
```

### Critical Data Governance Rules

| Rule | Enforcement |
|------|-------------|
| QC gate must pass | Training blocked if failed |
| Preprocessing fit on train only | Prevents leakage |
| HPO uses validation set only | Preserves test set |
| Test set touched ONCE | Final evaluation only |
| Holdout locked until deployment | Post-production validation |

### Execute Training

```python
from src.agents.ml_foundation.model_trainer import ModelTrainerAgent

trainer = ModelTrainerAgent()
result = await trainer.run({
    "model_candidate": selection_result["model_candidate"],
    "qc_report": qc_report,
    "experiment_id": "exp_001",
    "enable_hpo": True,
    "hpo_trials": 100,
    "success_criteria": {"min_auc": 0.8}
})

print(f"Train AUC: {result['metrics']['train_auc']}")
print(f"Test AUC: {result['metrics']['test_auc']}")
```

---

## Hyperparameter Optimization

E2I uses **Optuna** for intelligent hyperparameter search with warm-starting.

### Configuration

```yaml
# config/optuna_config.yaml
sampler:
  name: tpe
  seed: 42
  n_startup_trials: 10
  multivariate: true

pruner:
  name: median
  n_startup_trials: 5
  n_warmup_steps: 10

optimization:
  n_trials: 50
  timeout: null
  n_jobs: 1

warm_start:
  enabled: true
  min_similarity: 0.7
  max_trials_to_import: 20
```

### Search Space Example (XGBoost)

```python
search_space = {
    "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
    "max_depth": {"type": "int", "low": 3, "high": 10},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
}
```

### Warm-Starting

Optuna reuses successful hyperparameters from previous similar experiments:

1. Query `hpo_pattern_memory` for similar problems
2. Find patterns with similarity ≥ 0.7
3. Enqueue as initial trial
4. Continue exploration from best known point

### Problem-Type Defaults

| Problem Type | Metric | Direction |
|--------------|--------|-----------|
| Binary Classification | ROC-AUC | Maximize |
| Multiclass Classification | F1-Weighted | Maximize |
| Regression | Neg-RMSE | Maximize |

---

## Model Serving

E2I uses **BentoML** for production model serving with three specialized microservices.

### Service Architecture

| Service | Port | Purpose | Resources |
|---------|------|---------|-----------|
| **classification-service** | 3001 | Churn, conversion | 2 CPU, 4GB RAM |
| **regression-service** | 3002 | LTV, adoption | 2 CPU, 4GB RAM |
| **causal-service** | 3003 | CATE, treatment effects | 4 CPU, 8GB RAM |

### Deployment Workflow

```
1. Package Model
   └── Create Bento bundle with service template

2. Build Container
   └── Docker image with dependencies

3. Deploy
   └── Blue-green deployment (zero downtime)

4. Health Check
   └── Validate endpoints (30s intervals)

5. Monitor
   └── Prometheus metrics, alerts
```

### API Endpoints

```
POST /predict                   # Generic prediction
POST /{model_name}/predict     # Model-specific
GET  /health                   # Health check
GET  /metrics                  # Prometheus metrics
POST /predict_batch            # Batch predictions
```

### Making Predictions

```python
from src.api.dependencies.bentoml_client import BentoMLClient

client = BentoMLClient()

# Single prediction
result = await client.predict(
    model_name="churn_model",
    features={"tenure": 12, "engagement_score": 0.7}
)
print(f"Prediction: {result['predictions'][0]}")
print(f"Probability: {result['probabilities'][0]}")

# Batch prediction
results = await client.predict_batch(
    model_name="churn_model",
    features_list=[
        {"tenure": 12, "engagement_score": 0.7},
        {"tenure": 24, "engagement_score": 0.9},
    ]
)
```

### Response Format

```json
{
  "predictions": [0.7],
  "probabilities": [[0.3, 0.7]],
  "model_id": "churn_model_v1",
  "prediction_time_ms": 12.5,
  "_metadata": {
    "model_name": "churn_model",
    "latency_ms": 12.5,
    "timestamp": "2026-01-22T14:30:00Z"
  }
}
```

### Monitoring

**Health Status Levels:**
| Status | Condition |
|--------|-----------|
| `healthy` | All metrics normal |
| `degraded` | Error rate 5-10% OR latency 500-2000ms |
| `unhealthy` | Error rate >10% OR latency >2000ms |

**Prometheus Metrics:**
- `e2i_model_requests_total` - Request counts
- `e2i_model_request_latency_seconds` - Latency histogram
- `e2i_model_errors_total` - Error counts
- `e2i_model_health_status` - Health gauge

### Rollback

Blue-green deployment enables instant rollback:

```bash
# Check deployment status
curl http://localhost:8000/api/deployments/status

# Trigger rollback (if needed)
curl -X POST http://localhost:8000/api/deployments/rollback \
  -H "Content-Type: application/json" \
  -d '{"model_name": "churn_model", "target_version": "v1"}'
```

---

## Testing and Validation

### Split Validation

E2I validates data splits before training:

```python
from src.agents.ml_foundation.model_trainer.nodes.split_enforcer import SplitEnforcer

enforcer = SplitEnforcer()
result = enforcer.validate_splits(
    train_df=train,
    val_df=val,
    test_df=test,
    holdout_df=holdout
)

if not result.is_valid:
    print(f"Split violations: {result.violations}")
```

### Cross-Validation

For causal models, E2I validates across multiple libraries:

```python
from src.causal_engine.validation.cross_validator import CrossValidator

validator = CrossValidator()
result = await validator.validate(
    treatment="speaker_program",
    outcome="trx_lift",
    libraries=["dowhy", "econml", "causalml"]
)

print(f"Agreement Score: {result.agreement_score}")
print(f"Consensus Effect: {result.consensus_effect}")
```

### Model Evaluation Metrics

**Classification:**
| Metric | Description |
|--------|-------------|
| Accuracy | Overall correctness |
| Precision | False positive control |
| Recall | False negative control |
| F1 Score | Harmonic mean |
| AUC-ROC | Ranking performance |
| AUC-PR | Precision-recall area |

**Regression:**
| Metric | Description |
|--------|-------------|
| RMSE | Root mean squared error |
| MAE | Mean absolute error |
| R² | Coefficient of determination |
| MAPE | Mean absolute percentage error |

### Running Tests

```bash
# Run all ML tests
make test

# Run specific test categories
pytest tests/unit/test_leakage_detector.py -v
pytest tests/unit/test_split_enforcer.py -v
pytest tests/integration/test_mlops_pipeline.py -v
```

---

## Experiment Tracking

### MLflow Integration

All experiments are tracked in MLflow with full reproducibility:

**Tracked Items:**
- Parameters (hyperparameters, config)
- Metrics (train/val/test for all metrics)
- Artifacts (model, preprocessor, SHAP plots)
- Tags (experiment type, brand, problem type)

### Accessing MLflow

**Production:** http://138.197.4.36:5000
**Local:** http://localhost:5000

### Model Registry Stages

| Stage | Description | Validation |
|-------|-------------|------------|
| `development` | Initial training | None |
| `staging` | Validation phase | min_auc: 0.7 |
| `shadow` | Shadow production | min_auc: 0.75, 7 days |
| `production` | Full production | min_auc: 0.8, approval |
| `archived` | Retired | Auto after 30 days |

### Transitioning Models

```python
from src.mlops.mlflow_connector import MLflowConnector

mlflow_conn = MLflowConnector()

# Promote to staging
await mlflow_conn.transition_model_stage(
    model_name="churn_model",
    version=1,
    stage="staging"
)

# Promote to production (after validation)
await mlflow_conn.transition_model_stage(
    model_name="churn_model",
    version=1,
    stage="production"
)
```

---

## End-to-End Workflow

Here's the complete MLOps workflow from data to deployment:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA LOADING                                             │
│    python scripts/load_synthetic_data.py                   │
│    └─ Load data to Supabase with validation                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. DATA QUALITY CONTROL                                     │
│    ├─ Pandera schema validation (Layer 1)                  │
│    ├─ Quality dimension scoring (Layer 2)                  │
│    ├─ Great Expectations validation (Layer 3)              │
│    └─ Leakage detection (3 types)                          │
│    GATE: Overall score ≥ 0.80 AND no leakage               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. COHORT CONSTRUCTION                                      │
│    ├─ Apply inclusion criteria (AND logic)                 │
│    ├─ Apply exclusion criteria (AND NOT logic)             │
│    ├─ Validate temporal windows                            │
│    └─ Generate audit trail                                 │
│    OUTPUT: Eligible patient IDs + eligibility log          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. MODEL SELECTION                                          │
│    ├─ Filter by problem type                               │
│    ├─ Filter by constraints (latency, memory)              │
│    ├─ Filter by interpretability                           │
│    ├─ Rank candidates                                      │
│    └─ Optional: Run benchmark                              │
│    OUTPUT: Model candidate with hyperparameter space       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. DATA SPLITTING                                           │
│    ├─ Enforce 60/20/15/5 ratios (±2%)                      │
│    ├─ Temporal ordering validation                         │
│    ├─ Entity-level splitting (no patient overlap)          │
│    └─ Stratification for classification                    │
│    GATE: All ratios within tolerance                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. PREPROCESSING                                            │
│    ├─ Fit scalers on TRAIN ONLY                            │
│    ├─ Transform all splits                                 │
│    └─ Save preprocessor artifact                           │
│    RULE: Never fit on validation/test/holdout              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. HYPERPARAMETER OPTIMIZATION                              │
│    ├─ Load warm-start from pattern memory                  │
│    ├─ Run Optuna trials on VALIDATION set                  │
│    ├─ Prune unpromising trials (median pruner)             │
│    └─ Store best pattern for future warm-starts            │
│    OUTPUT: Best hyperparameters                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. MODEL TRAINING                                           │
│    ├─ Train on TRAIN set with best hyperparameters         │
│    ├─ Validate on VALIDATION set (early stopping)          │
│    ├─ Final evaluation on TEST set (ONCE)                  │
│    └─ Log all metrics to MLflow                            │
│    GATE: Meets success criteria (e.g., min_auc: 0.8)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 9. MODEL REGISTRATION                                       │
│    ├─ Register in MLflow Model Registry                    │
│    ├─ Stage: development → staging → production            │
│    └─ Version tracking with metadata                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 10. MODEL DEPLOYMENT                                        │
│    ├─ Package as BentoML bundle                            │
│    ├─ Build Docker container                               │
│    ├─ Blue-green deployment                                │
│    └─ Health check validation                              │
│    OUTPUT: Production endpoint serving predictions         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 11. MONITORING                                              │
│    ├─ Prometheus metrics collection                        │
│    ├─ Latency and error rate tracking                      │
│    ├─ Drift detection                                      │
│    └─ Alerting on thresholds                               │
│    CONTINUOUS: Health checks every 30 seconds              │
└─────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Data Quality Issues

**Problem:** QC gate failing
```bash
# Check quality report
SELECT * FROM ml_data_quality_reports ORDER BY run_at DESC LIMIT 1;

# Check specific dimension scores
SELECT completeness_score, validity_score, consistency_score
FROM ml_data_quality_reports
WHERE overall_status = 'failed';
```

**Solution:** Fix the lowest-scoring dimension first

### Leakage Detection

**Problem:** Leakage detected blocking training

**Check which type:**
```python
result = leakage_detector.detect(state)
print(result["leakage_issues"])
```

**Common fixes:**
- Temporal: Remove features with future dates
- Target: Remove highly correlated features
- Contamination: Re-split data ensuring no overlap

### Split Ratio Violations

**Problem:** Split ratios outside tolerance

**Check:**
```python
enforcer.validate_splits(train, val, test, holdout)
# Returns violations list
```

**Fix:** Adjust splitting logic or use stratification

### Model Training Failures

**Problem:** Training not meeting success criteria

**Check MLflow:**
```bash
# Open MLflow UI
mlflow ui --port 5000

# Or query directly
mlflow runs list --experiment-id 1
```

**Solutions:**
- Increase HPO trials
- Try different algorithms
- Check for data quality issues
- Review feature engineering

### BentoML Deployment Issues

**Problem:** Model serving unhealthy

**Check health:**
```bash
curl http://localhost:3001/health
```

**Check logs:**
```bash
docker logs classification-service
```

**Common fixes:**
- Restart service: `docker-compose restart classification-service`
- Rollback to previous version
- Check memory limits

---

## Quick Reference

### Commands

```bash
# Data loading
python scripts/load_synthetic_data.py
python scripts/load_synthetic_data.py --small --dry-run

# Quality validation
python scripts/run_leakage_audit.py
python scripts/validate_kpi_coverage.py

# Testing
make test                    # Full test suite
pytest tests/unit/ -v        # Unit tests only
pytest tests/integration/ -v # Integration tests

# MLflow
mlflow ui --port 5000        # Start MLflow UI

# BentoML
bentoml list                 # List models
bentoml serve service:svc    # Start serving
```

### Key Files

| File | Purpose |
|------|---------|
| `scripts/load_synthetic_data.py` | Data loading |
| `config/optuna_config.yaml` | HPO configuration |
| `config/mlflow/mlflow.yaml` | MLflow configuration |
| `config/model_endpoints.yaml` | Serving endpoints |
| `src/mlops/data_quality.py` | Great Expectations |
| `src/agents/cohort_constructor/` | Cohort construction |
| `src/agents/ml_foundation/` | Tier 0 agents |

### Database Tables

| Table | Purpose |
|-------|---------|
| `ml_data_quality_reports` | QC results |
| `ml_cohort_definitions` | Cohort specs |
| `ml_cohort_eligibility_log` | Eligibility audit |
| `ml_training_runs` | Training history |
| `ml_model_registry` | Model versions |
| `ml_hpo_studies` | Optuna studies |
| `ml_bentoml_services` | Deployed services |

---

*Last Updated: January 2026*
*Version: 4.3 (GEPA Prompt Optimization)*
