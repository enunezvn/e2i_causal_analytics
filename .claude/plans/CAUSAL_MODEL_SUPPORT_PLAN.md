# Causal Model Support for Model Trainer - Implementation Plan

## Objective

Fix the model_trainer agent to support causal models, enable MLflow/Opik observability, fix model_uri handoff, and re-run the Tier 0 MLOps workflow test.

---

## Current Issues

| Issue | Location | Root Cause |
|-------|----------|------------|
| Causal models return `None` | `optuna_optimizer.py:931-938` | Returns None for CausalForest, LinearDML |
| No causal model class mapping | `model_trainer_node.py:196-248` | Missing entries in `_get_model_class_dynamic` |
| No causal hyperparameter filtering | `model_trainer_node.py:250-344` | Missing entries in `_filter_hyperparameters` |
| No causal fixed params | `hyperparameter_tuner.py:553-591` | Missing entries in `_get_fixed_params` |
| model_uri not passed to Step 6 | `run_tier0_test.py:737-742` | Passes `trained_model`, Feature Analyzer needs `model_uri` |
| Fallback logic bypasses causal | `run_tier0_test.py:378-387` | Workaround forces LogisticRegression |

---

## Implementation Tasks

### Task 1: Add Causal Model Classes to `optuna_optimizer.py`

**File**: `src/mlops/optuna_optimizer.py`
**Location**: Lines 931-938 (replace existing None returns)

```python
elif algorithm_name == "CausalForest":
    from econml.dml import CausalForestDML
    return CausalForestDML

elif algorithm_name == "LinearDML":
    from econml.dml import LinearDML
    return LinearDML

elif algorithm_name == "DRLearner":
    from econml.dr import DRLearner
    return DRLearner

elif algorithm_name == "SLearner":
    from econml.metalearners import SLearner
    return SLearner

elif algorithm_name == "TLearner":
    from econml.metalearners import TLearner
    return TLearner

elif algorithm_name == "XLearner":
    from econml.metalearners import XLearner
    return XLearner
```

---

### Task 2: Add Model Class Mapping in `model_trainer_node.py`

**File**: `src/agents/ml_foundation/model_trainer/nodes/model_trainer_node.py`
**Location**: After line 239 (in `_get_model_class_dynamic`)

```python
elif algorithm_name == "CausalForest":
    from econml.dml import CausalForestDML
    return CausalForestDML

elif algorithm_name == "LinearDML":
    from econml.dml import LinearDML
    return LinearDML

elif algorithm_name in ("DRLearner", "SLearner", "TLearner", "XLearner"):
    # Meta-learners share similar interface
    from econml import metalearners, dr
    mapping = {
        "DRLearner": dr.DRLearner,
        "SLearner": metalearners.SLearner,
        "TLearner": metalearners.TLearner,
        "XLearner": metalearners.XLearner,
    }
    return mapping[algorithm_name]
```

---

### Task 3: Add Hyperparameter Filtering for Causal Models

**File**: `src/agents/ml_foundation/model_trainer/nodes/model_trainer_node.py`
**Location**: In `_filter_hyperparameters` function, add to `allowed_params` dict

```python
"CausalForest": {
    "n_estimators", "max_depth", "min_samples_leaf", "min_samples_split",
    "max_features", "inference", "n_jobs", "random_state",
    "model_y", "model_t", "discrete_treatment", "cv"
},
"LinearDML": {
    "model_y", "model_t", "discrete_treatment", "cv", "mc_iters",
    "random_state", "linear_first_stages"
},
"DRLearner": {
    "model_propensity", "model_regression", "model_final",
    "cv", "mc_iters", "random_state", "n_jobs"
},
"SLearner": {"overall_model", "cv", "random_state"},
"TLearner": {"models", "cv", "random_state"},
"XLearner": {"models", "propensity_model", "cate_models", "cv", "random_state"},
```

---

### Task 4: Add Fixed Params for Causal Models

**File**: `src/agents/ml_foundation/model_trainer/nodes/hyperparameter_tuner.py`
**Location**: In `_get_fixed_params` function, add entries

```python
"CausalForest": {
    "n_jobs": -1,
    "inference": True,
    "random_state": 42,
},
"LinearDML": {
    "cv": 3,
    "mc_iters": 3,
    "random_state": 42,
},
"DRLearner": {
    "cv": 3,
    "random_state": 42,
},
"SLearner": {"cv": 3, "random_state": 42},
"TLearner": {"cv": 3, "random_state": 42},
"XLearner": {"cv": 3, "random_state": 42},
```

---

### Task 5: Fix model_uri Handoff in Test Script

**File**: `scripts/run_tier0_test.py`

**5a.** Update step_5 result capture (line 726):
```python
state["model_uri"] = result.get("model_uri") or result.get("model_artifact_uri") or result.get("mlflow_model_uri")
```

**5b.** Update step_6 call to pass model_uri (lines 737-742):
```python
result = await step_6_feature_analyzer(
    experiment_id,
    state.get("trained_model"),
    X.iloc[:50],
    y.iloc[:50],
    model_uri=state.get("model_uri")  # Add this parameter
)
```

**5c.** Update step_6_feature_analyzer function signature (line 490-495):
```python
async def step_6_feature_analyzer(
    experiment_id: str,
    trained_model: Any,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    model_uri: Optional[str] = None  # Add this
) -> dict[str, Any]:
```

**5d.** Update input_data in step_6 (lines 504-510):
```python
input_data = {
    "experiment_id": experiment_id,
    "trained_model": trained_model,
    "model_uri": model_uri,  # Add this
    "X_sample": X_sample,
    "y_sample": y_sample,
    "max_samples": min(100, len(X_sample)),
}
```

---

### Task 6: Remove Causal Model Fallback in Test Script

**File**: `scripts/run_tier0_test.py`
**Location**: Lines 378-387

**Replace the fallback logic with proper causal model handling:**

```python
# Check if causal model - prepare treatment indicator if needed
causal_models = ["LinearDML", "CausalForest", "DoubleLasso", "SparseLinearDML", "DML",
                 "DRLearner", "SLearner", "TLearner", "XLearner"]
algo_name = model_candidate.get("algorithm_name", "")
if algo_name in causal_models:
    print_info(f"Causal model '{algo_name}' selected - treatment indicator required")
    # Note: For true causal inference, treatment must come from data
    # For testing, we can proceed with standard classification if no treatment available
```

---

### Task 7: Enable MLflow and Opik in Agents

**File**: `scripts/run_tier0_test.py`

When instantiating agents, add `enable_mlflow=True` and `enable_opik=True`:

```python
# In step functions, update agent instantiation:
agent = ScopeDefinerAgent(enable_mlflow=True, enable_opik=True)
agent = DataPreparerAgent(enable_mlflow=True, enable_opik=True)
agent = CohortConstructorAgent(enable_mlflow=True, enable_opik=True)
# ... etc for all agents
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/mlops/optuna_optimizer.py` | Add causal model classes (Task 1) |
| `src/agents/ml_foundation/model_trainer/nodes/model_trainer_node.py` | Add model mapping + hyperparameter filtering (Tasks 2, 3) |
| `src/agents/ml_foundation/model_trainer/nodes/hyperparameter_tuner.py` | Add fixed params (Task 4) |
| `scripts/run_tier0_test.py` | Fix model_uri handoff, remove fallback, enable observability (Tasks 5-7) |

---

## Answers to User Questions

### Q: How are Feast and Optuna being used in this test?

**Optuna Usage**:
- **Location**: `src/agents/ml_foundation/model_trainer/nodes/hyperparameter_tuner.py`
- **Usage**: HPO with TPE sampler and median pruner
- **Config**: `hpo_trials=10` in test (reduced from default 50 for speed)
- **Study Creation**: Creates Optuna study per algorithm (e.g., `XGBoost_hpo`)
- **Objective**: Optimizes on validation set with `roc_auc` metric for classification

**Feast Usage**:
- **Current State**: Feast is NOT actively used in the Tier 0 test
- **Why**: The test script loads data directly from Supabase `patient_journeys` table
- **Feast Purpose**: Would provide point-in-time correct feature retrieval for production ML
- **Integration Point**: `src/feature_store/` contains Feast client but not invoked in test

---

## Verification Steps

After implementation, run on droplet:

```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36

cd /opt/e2i_causal_analytics
source .venv/bin/activate

# Sync code changes
git pull origin main

# Run test with observability enabled
python scripts/run_tier0_test.py --enable-mlflow --enable-opik

# Verify MLflow
curl -s http://localhost:5000/api/2.0/mlflow/experiments/search | python3 -m json.tool

# Verify Opik traces
curl -s http://localhost:8080/api/v1/projects | python3 -m json.tool
```

**Check Opik UI**: http://138.197.4.36/opik/

---

## Success Criteria

| Checkpoint | Criteria |
|------------|----------|
| Causal model class loading | `get_model_class("CausalForest")` returns `CausalForestDML` |
| No fallback triggered | Test doesn't fall back to LogisticRegression for causal models |
| model_uri passed | Feature Analyzer receives valid model_uri |
| MLflow logging | Experiment and run visible in MLflow UI |
| Opik traces | Traces visible at http://138.197.4.36/opik/ |
| All 8 steps pass | Pipeline completes with all steps successful |

---

## Risk Notes

| Risk | Mitigation |
|------|------------|
| EconML not installed on droplet | Check `pip show econml` - should already be in venv |
| Treatment column missing | For test purposes, causal models can still run with synthetic treatment for demonstration |
| Nested model HPO complexity | Use default nested models (Ridge for model_y, LogisticRegression for model_t) |

---

## Model Selector: sklearn & MLflow Patterns

### Model Support in Benchmarking

**File**: `src/agents/ml_foundation/model_selector/nodes/benchmark_runner.py`

**All Models Supported** (lines 163-212):
| Model | Type | Package | Import |
|-------|------|---------|--------|
| **XGBoost** | Gradient Boosting | xgboost | `XGBClassifier/XGBRegressor` |
| **LightGBM** | Gradient Boosting | lightgbm | `LGBMClassifier/LGBMRegressor` |
| **RandomForest** | Ensemble | sklearn | `RandomForestClassifier/Regressor` |
| **LogisticRegression** | Linear | sklearn | `LogisticRegression` |
| **Ridge** | Linear | sklearn | `Ridge` |
| **Lasso** | Linear | sklearn | `Lasso` |
| **CausalForest** | Causal ML | econml | _Returns None (skip benchmarking)_ |
| **LinearDML** | Causal ML | econml | _Returns None (skip benchmarking)_ |

**Note**: Causal models (CausalForest, LinearDML) currently return `None` in `_create_model_instance()` because they require special handling (treatment indicators). Our implementation plan adds proper support in the model_trainer, not in benchmarking.

**Cross-Validation Pattern** (lines 234-269):
```python
from sklearn.model_selection import cross_val_score

# Classification: ROC-AUC scoring
scores = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc', n_jobs=-1)

# Regression: Negative MSE → converted to RMSE
scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1)
rmse = np.sqrt(-scores.mean())
```

**Re-ranking Formula** (60% original + 40% benchmark):
```python
combined_score = 0.6 * original_score + 0.4 * benchmark_score
# benchmark_score penalizes high variance:
benchmark_score = cv_mean - (0.5 * cv_std)
```

### MLflow Integration Pattern

**File**: `src/agents/ml_foundation/model_selector/nodes/mlflow_registrar.py`

**Async Context Manager Pattern**:
```python
async with connector.start_run(
    experiment_id=mlflow_experiment_id,
    run_name=f"model_selection_{primary_candidate['name']}",
    tags={
        "agent": "model_selector",
        "algorithm": primary_candidate["name"],
        "selection_type": "automated",
    },
) as run:
    await run.log_params(params)      # Algorithm specs, hyperparameters
    await run.log_metrics(metrics)    # selection_score, benchmark_cv_mean, cv_std
    await run.log_artifact(path)      # Selection rationale text
```

**Logged Metrics**:
- `selection_score` - Initial ranking score
- `benchmark_cv_mean` - Cross-validation mean
- `benchmark_cv_std` - Cross-validation std deviation
- `combined_score` - Final composite score
- `training_time_seconds` - Benchmark duration

---

## Feast Integration Testing

### Feast Architecture (Already Integrated)

```
┌─────────────────────────────────────────────────────────────┐
│                    Feast Feature Store                       │
├─────────────────────────────────────────────────────────────┤
│  Offline Store: Supabase PostgreSQL                         │
│  Online Store: Redis (port 6382)                            │
│  Materialization: Celery background tasks                   │
└─────────────────────────────────────────────────────────────┘
          ▲                    ▲                    ▲
          │                    │                    │
    ┌─────┴─────┐        ┌─────┴─────┐       ┌─────┴─────┐
    │data_preparer│      │model_trainer│     │prediction_│
    │feast_registrar│    │split_loader │     │synthesizer│
    └───────────┘        └───────────┘       └───────────┘
```

### Feast Test Tasks

**Task 8: Add Feast Feature Registration to Test**

**File**: `scripts/run_tier0_test.py`

Add after step_2_data_preparer (before step 3):

```python
async def step_2b_feast_registration(
    experiment_id: str,
    X: pd.DataFrame,
    feature_metadata: dict
) -> dict[str, Any]:
    """Step 2b: Register features in Feast."""
    print_header("2b", "FEAST FEATURE REGISTRATION")

    from src.feature_store.feature_analyzer_adapter import get_feature_analyzer_adapter

    adapter = get_feature_analyzer_adapter(enable_feast=True)

    # Build adapter state from data preparer output
    adapter_state = {
        "feature_metadata": feature_metadata,
        "X": X,
        "entity_key": "patient_journey_id",
    }

    result = await adapter.register_features_from_state(
        state=adapter_state,
        experiment_id=experiment_id,
        entity_key="patient_journey_id",
        owner="data_preparer",
        tags=["tier0_test", f"exp_{experiment_id}"],
    )

    print_result("features_registered", result.get("features_registered", 0))
    print_result("feast_status", result.get("status", "unknown"))

    return result
```

**Task 9: Add Feast Freshness Check**

```python
async def step_2c_feast_freshness_check(
    experiment_id: str,
    required_features: list[str]
) -> dict[str, Any]:
    """Step 2c: Check feature freshness."""
    print_header("2c", "FEAST FRESHNESS CHECK")

    from src.feature_store.feast_client import FeastClient

    client = FeastClient()

    freshness_result = await client.check_feature_freshness(
        feature_refs=required_features,
        max_staleness_hours=24.0
    )

    print_result("freshness_status", freshness_result.get("status", "unknown"))
    print_result("stalest_feature_age_hours", freshness_result.get("max_age_hours", "N/A"))

    if freshness_result.get("status") == "STALE":
        print_warning("Features are stale - consider re-materialization")
    else:
        print_success("Features are fresh")

    return freshness_result
```

**Task 10: Add Feast Online Feature Retrieval Test**

In step_8 (Observability Connector), add Feast online retrieval test:

```python
# Test online feature retrieval (prediction-time)
from src.agents.prediction_synthesizer.nodes.feast_feature_store import FeastFeatureStore

feast_store = FeastFeatureStore()
sample_entity_id = eligible_patient_ids[0] if eligible_patient_ids else "test_patient"

online_features = await feast_store.get_online_features(
    entity_id=sample_entity_id,
    feature_refs=["hcp_conversion_features:engagement_score"]
)

print_result("online_feature_retrieval", "SUCCESS" if online_features else "FAILED")
```

### Feast Verification Commands

```bash
# Check Feast CLI
feast --version

# Check Redis (online store)
redis-cli -p 6382 PING

# Check feature freshness via API
curl -s http://localhost:8000/api/features/freshness | python3 -m json.tool

# Check materialization status
curl -s http://localhost:8000/api/features/materialization/status | python3 -m json.tool

# Manual materialization trigger (if needed)
curl -X POST http://localhost:8000/api/features/materialize \
  -H "Content-Type: application/json" \
  -d '{"feature_views": ["hcp_conversion_features"], "days_back": 7}'
```

---

## Complete Test Strategy (Step-by-Step)

### Prerequisites

1. SSH access to droplet (138.197.4.36)
2. E2I API running (port 8000)
3. MLflow running (port 5000)
4. Opik running (port 5173/8080)
5. Redis running (port 6382) - for Feast online store
6. Supabase connectivity

### Test Configuration

```yaml
test_brand: Kisqali  # 877 patients, well-distributed splits
problem_type: binary_classification
target: discontinuation_flag
data_source: patient_journeys

# Data sufficiency (from Supabase):
# patient_journeys: 2,700 total (877 Kisqali)
# Split distribution: 60%/20%/15%/5% (train/val/test/holdout)
```

### Setup (Run Once)

```bash
# SSH to droplet
ssh -i ~/.ssh/replit enunez@138.197.4.36

# Navigate and activate venv
cd /opt/e2i_causal_analytics
source .venv/bin/activate

# Sync latest code
git pull origin main

# Verify services
curl -s localhost:8000/health | python3 -m json.tool
curl -s localhost:5000/health
curl -s localhost:8080/health
redis-cli -p 6382 PING
```

### Step 1: Scope Definer

```python
python3 << 'PYEOF'
import asyncio
from src.agents.ml_foundation.scope_definer import ScopeDefinerAgent

async def test_scope_definer():
    agent = ScopeDefinerAgent(enable_mlflow=True, enable_opik=True)
    result = await agent.run({
        "problem_description": "Predict patient discontinuation risk for Kisqali",
        "business_objective": "Identify high-risk patients early for intervention",
        "target_outcome": "discontinuation_flag",
        "problem_type_hint": "binary_classification"
    })
    print("=== SCOPE DEFINER OUTPUT ===")
    print(f"Experiment ID: {result.get('experiment_id')}")
    print(f"Scope Spec: {result.get('scope_spec')}")
    print(f"Success Criteria: {result.get('success_criteria')}")
    return result

result = asyncio.run(test_scope_definer())
PYEOF
```

### Step 2: Data Preparer

```python
python3 << 'PYEOF'
import asyncio
from src.agents.ml_foundation.data_preparer import DataPreparerAgent

scope_spec = {
    "experiment_id": "<from_step_1>",
    "problem_type": "binary_classification",
    "prediction_target": "discontinuation_flag",
    "minimum_samples": 50
}

async def test_data_preparer():
    agent = DataPreparerAgent(enable_mlflow=True, enable_opik=True)
    result = await agent.run({
        "scope_spec": scope_spec,
        "data_source": "patient_journeys",
        "brand": "Kisqali"
    })
    print("=== DATA PREPARER OUTPUT ===")
    print(f"QC Report: {result.get('qc_report')}")
    print(f"Gate Passed: {result.get('gate_passed')}")
    print(f"Train samples: {result.get('train_samples')}")
    return result

result = asyncio.run(test_data_preparer())
PYEOF
```

### Step 2b: Feast Feature Registration (NEW)

```python
python3 << 'PYEOF'
import asyncio
from src.feature_store.feature_analyzer_adapter import get_feature_analyzer_adapter

async def test_feast_registration():
    adapter = get_feature_analyzer_adapter(enable_feast=True)

    # Use sample feature metadata from data_preparer
    result = await adapter.register_features_from_state(
        state={"feature_metadata": {}, "entity_key": "patient_journey_id"},
        experiment_id="<from_step_1>",
        entity_key="patient_journey_id",
        owner="tier0_test",
        tags=["test"]
    )
    print("=== FEAST REGISTRATION OUTPUT ===")
    print(f"Status: {result.get('status')}")
    print(f"Features Registered: {result.get('features_registered', 0)}")
    return result

result = asyncio.run(test_feast_registration())
PYEOF
```

### Step 3: Cohort Constructor

```python
python3 << 'PYEOF'
import asyncio
from src.agents.cohort_constructor import CohortConstructorAgent

async def test_cohort():
    agent = CohortConstructorAgent(enable_mlflow=True, enable_opik=True)
    result = await agent.run({
        "brand": "Kisqali",
        "indication": "HR+/HER2- breast cancer",
        "inclusion_criteria": ["journey_status IS NOT NULL"],
        "exclusion_criteria": ["data_quality_score < 0.5"]
    })
    print("=== COHORT CONSTRUCTOR OUTPUT ===")
    print(f"Eligible count: {result.get('eligible_patient_count')}")
    print(f"Eligibility stats: {result.get('eligibility_stats')}")
    return result

result = asyncio.run(test_cohort())
PYEOF
```

### Step 4: Model Selector

```python
python3 << 'PYEOF'
import asyncio
from src.agents.ml_foundation.model_selector import ModelSelectorAgent

scope_spec = {"experiment_id": "<from_step_1>", "problem_type": "binary_classification"}
qc_report = {"gate_passed": True}
baseline_metrics = {}

async def test_model_selector():
    agent = ModelSelectorAgent(enable_mlflow=True, enable_opik=True)
    result = await agent.run({
        "scope_spec": scope_spec,
        "qc_report": qc_report,
        "baseline_metrics": baseline_metrics
    })
    print("=== MODEL SELECTOR OUTPUT ===")
    print(f"Model candidate: {result.get('model_candidate')}")
    print(f"HPO search space: {result.get('hyperparameter_search_space')}")
    print(f"Benchmark CV Mean: {result.get('benchmark_cv_mean')}")
    print(f"Combined Score: {result.get('combined_score')}")
    return result

result = asyncio.run(test_model_selector())
PYEOF
```

### Step 5: Model Trainer

```python
python3 << 'PYEOF'
import asyncio
from src.agents.ml_foundation.model_trainer import ModelTrainerAgent

async def test_trainer():
    agent = ModelTrainerAgent(enable_mlflow=True, enable_opik=True)
    result = await agent.run({
        "model_candidate": {"algorithm": "xgboost"},
        "qc_report": {"gate_passed": True},
        "hyperparameter_search_space": {},
        "hpo_trials": 10
    })
    print("=== MODEL TRAINER OUTPUT ===")
    print(f"Model URI: {result.get('model_uri') or result.get('mlflow_model_uri')}")
    print(f"Validation metrics: {result.get('validation_metrics')}")
    print(f"Success criteria met: {result.get('success_criteria_met')}")
    return result

result = asyncio.run(test_trainer())
PYEOF
```

### Step 6: Feature Analyzer

```python
python3 << 'PYEOF'
import asyncio
from src.agents.ml_foundation.feature_analyzer import FeatureAnalyzerAgent

async def test_feature_analyzer():
    agent = FeatureAnalyzerAgent(enable_mlflow=True, enable_opik=True)
    result = await agent.run({
        "model_uri": "<from_step_5>",
        "problem_type": "classification"
    })
    print("=== FEATURE ANALYZER OUTPUT ===")
    print(f"Top features: {result.get('top_features')}")
    print(f"Feature importance: {result.get('feature_importance')}")
    return result

result = asyncio.run(test_feature_analyzer())
PYEOF
```

### Step 7: Model Deployer

```python
python3 << 'PYEOF'
import asyncio
from src.agents.ml_foundation.model_deployer import ModelDeployerAgent

async def test_deployer():
    agent = ModelDeployerAgent(enable_mlflow=True, enable_opik=True)
    result = await agent.run({
        "model_uri": "<from_step_5>",
        "validation_metrics": {},
        "success_criteria_met": True
    })
    print("=== MODEL DEPLOYER OUTPUT ===")
    print(f"Deployment manifest: {result.get('deployment_manifest')}")
    print(f"Model version: {result.get('model_version')}")
    print(f"Stage: {result.get('stage')}")
    return result

result = asyncio.run(test_deployer())
PYEOF
```

### Step 8: Observability Connector

```python
python3 << 'PYEOF'
import asyncio
from src.agents.ml_foundation.observability_connector import ObservabilityConnectorAgent

async def test_observability():
    agent = ObservabilityConnectorAgent()
    result = await agent.run({
        "time_window": "1h",
        "agent_name_filter": None
    })
    print("=== OBSERVABILITY CONNECTOR OUTPUT ===")
    print(f"Logged events: {result.get('logged_events')}")
    print(f"Aggregated metrics: {result.get('aggregated_metrics')}")
    return result

result = asyncio.run(test_observability())
PYEOF
```

### Final Verification

```bash
# Check MLflow experiments
curl -s "http://localhost:5000/api/2.0/mlflow/experiments/search" | python3 -m json.tool

# Check model registry
curl -s "http://localhost:5000/api/2.0/mlflow/registered-models/list" | python3 -m json.tool

# Check Opik traces
curl -s "http://localhost:8080/api/v1/projects" | python3 -m json.tool

# Check Feast feature freshness
redis-cli -p 6382 KEYS "feast:*" | head -20

# View in browser
echo "MLflow: http://138.197.4.36:5000"
echo "Opik: http://138.197.4.36/opik/"
```

---

## Full Pipeline Command

After implementing all changes, run the complete test:

```bash
cd /opt/e2i_causal_analytics
source .venv/bin/activate

# Full pipeline with all observability
python scripts/run_tier0_test.py --enable-mlflow --enable-opik

# Or run specific steps
python scripts/run_tier0_test.py --step 5 --enable-mlflow  # Just model training
```

---

## Success Criteria Summary

| Checkpoint | Criteria |
|------------|----------|
| Data Loading | ≥ 800 patients loaded for Kisqali |
| QC Gate | `gate_passed = True` |
| Feast Registration | Features registered successfully |
| Feast Freshness | Status = FRESH (< 24h staleness) |
| Cohort Size | ≥ 30 eligible patients |
| Model Selection | Benchmark CV mean logged to MLflow |
| Model Training | Validation AUC ≥ 0.60 |
| SHAP Computation | Feature importance computed |
| Model Registry | Model registered in MLflow |
| Observability | Traces visible in Opik |
| Online Features | Feast online retrieval works |
