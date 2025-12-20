# MLOps Tools Configuration

**Last Updated**: 2025-12-18
**Update Cadence**: When tool versions change or new integrations added
**Owner**: MLOps Team
**Purpose**: MLOps tool configuration and integration patterns

> Context file for E2I Causal Analytics V4 ML Foundation layer.
> Provides configuration details for 7 integrated MLOps tools.

## Overview

The ML Foundation (Tier 0) integrates with 7 specialized MLOps tools to provide complete ML lifecycle management. This document defines configurations, integration patterns, and usage guidelines.

---

## Tool Matrix

| Tool | Version | Primary Agents | Purpose |
|------|---------|----------------|---------|
| MLflow | ≥2.16.0 | model_trainer, model_selector, model_deployer | Experiment tracking, model registry |
| Opik | ≥0.2.0 | observability_connector, feature_analyzer | LLM/agent observability |
| Great Expectations | ≥1.0.0 | data_preparer | Data quality validation |
| Feast | ≥0.40.0 | data_preparer, model_trainer | Feature store |
| Optuna | ≥3.6.0 | model_trainer | Hyperparameter optimization |
| SHAP | ≥0.46.0 | feature_analyzer | Model interpretability |
| BentoML | ≥1.3.0 | model_deployer | Model serving |

---

## Implementation Status

**System Maturity**: Infrastructure configured, agent integrations in progress

### Tool Integration Status

| Tool | Config Status | Code Integration | Agent Status | Notes |
|------|---------------|------------------|--------------|-------|
| **MLflow** | ✅ Complete | ⚠️ Verify | model_trainer ❌, model_selector ❌, model_deployer ❌ | Config ready, awaiting Tier 0 agents |
| **Opik** | ✅ Complete | ⚠️ Verify | observability_connector ❌, feature_analyzer ❌ | Config ready, awaiting Tier 0 agents |
| **Great Expectations** | ✅ Complete | ⚠️ Verify | data_preparer ❌ | Config ready, awaiting data_preparer |
| **Feast** | ✅ Complete | ⚠️ Verify | data_preparer ❌, model_trainer ❌ | Config ready, awaiting Tier 0 agents |
| **Optuna** | ✅ Complete | ⚠️ Verify | model_trainer ❌ | Config ready, awaiting model_trainer |
| **SHAP** | ✅ Complete | ✅ Partial | feature_analyzer ❌ | src/mlops/shap_explainer_realtime.py exists |
| **BentoML** | ✅ Complete | ⚠️ Verify | model_deployer ❌ | Config ready, awaiting model_deployer |

**Legend**:
- ✅ Complete: Fully configured and ready
- ⚠️ Verify: Configuration exists but integration needs verification
- ❌ Not implemented: Agent or integration not yet coded
- ✅ Partial: Some code exists, but incomplete integration

### Integration Readiness Matrix

| Category | Status | Details |
|----------|--------|---------|
| **Configurations** | ✅ 100% | All 7 tools configured in agent_config.yaml |
| **Environment Variables** | ⚠️ Verify | Need to check .env setup |
| **Dependencies** | ✅ 100% | All tools in requirements.txt |
| **Code Integration** | ⚠️ 14% | Only SHAP has partial code (shap_explainer_realtime.py) |
| **Agent Implementation** | ❌ 0% | No Tier 0 agents implemented (0 of 7) |
| **Database Support** | ✅ 100% | ml_* tables support all tools |

**Overall MLOps Integration**: ~40% complete (config done, code/agents pending)

### Critical Path for Full Integration

**Phase 1: Data Pipeline (HIGHEST PRIORITY)**
1. ✅ Feast configuration
2. ✅ Great Expectations configuration
3. ❌ **Implement data_preparer agent** (BLOCKER)
4. ❌ Integrate Feast feature store
5. ❌ Integrate Great Expectations QC gates

**Phase 2: Model Training**
6. ✅ MLflow configuration
7. ✅ Optuna configuration
8. ❌ **Implement model_trainer agent** (depends on data_preparer)
9. ❌ Integrate MLflow experiment tracking
10. ❌ Integrate Optuna hyperparameter optimization

**Phase 3: Model Analysis**
11. ✅ SHAP configuration
12. ⚠️ SHAP partial code exists
13. ❌ **Implement feature_analyzer agent**
14. ❌ Integrate SHAP explainability

**Phase 4: Model Deployment**
15. ✅ BentoML configuration
16. ❌ **Implement model_deployer agent** (depends on model_trainer)
17. ❌ Integrate BentoML serving

**Phase 5: Observability**
18. ✅ Opik configuration
19. ❌ **Implement observability_connector agent**
20. ❌ Integrate Opik tracing for all agents

### Verification Commands

**Check MLOps tool installations**:
```bash
# Verify all tools are installed with correct versions
pip list | grep -E "mlflow|opik|optuna|feast|great-expectations|bentoml|shap"

# Expected output:
# mlflow                    2.16.0 (or higher)
# opik                      0.2.0 (or higher)
# optuna                    3.6.0 (or higher)
# feast                     0.40.0 (or higher)
# great-expectations        1.0.0 (or higher)
# bentoml                   1.3.0 (or higher)
# shap                      0.46.0 (or higher)
```

**Check configuration files**:
```bash
# Verify agent_config.yaml has MLOps tools configured
grep -A 5 "mlflow\|opik\|feast\|optuna" config/agent_config.yaml

# Check database tables for ML support
psql -d e2i_causal_analytics -c "\\dt ml_*"

# Expected tables:
# ml_experiments, ml_data_quality_reports, ml_feature_store, ml_model_registry,
# ml_training_runs, ml_shap_analyses, ml_deployments, ml_observability_spans
```

**Test SHAP integration** (partial code exists):
```bash
# Verify SHAP code exists
ls -la src/mlops/shap_explainer_realtime.py

# Test SHAP import
python -c "from src.mlops.shap_explainer_realtime import SHAPExplainer; print('✅ SHAP integration found')"
```

### Next Steps for Full MLOps Integration

**Immediate Actions** (Required before using MLOps tools):
1. **Verify environment variables** - Check .env has all required MLOps tool credentials
2. **Test tool connectivity** - Verify MLflow tracking server, Opik API, etc. are accessible
3. **Implement data_preparer agent** - CRITICAL blocker for entire Tier 0
4. **Create integration tests** - Test each tool's configuration and connectivity

**Medium-term Actions** (After Tier 0 agent implementation):
1. Implement model_trainer with MLflow + Optuna integration
2. Implement feature_analyzer with SHAP integration
3. Implement model_deployer with BentoML integration
4. Implement observability_connector with Opik integration
5. Create end-to-end MLOps pipeline test

**Long-term Actions** (Production readiness):
1. Set up production MLflow tracking server
2. Configure Feast online/offline stores
3. Implement Great Expectations data quality monitoring
4. Set up Opik production tracing
5. Deploy models with BentoML to production

### Known Limitations

| Limitation | Impact | Workaround | Timeline |
|------------|--------|------------|----------|
| No Tier 0 agents implemented | Cannot use any MLOps tools | Manual ML workflows | Q1 2025 |
| SHAP integration incomplete | Limited explainability | Manual SHAP analysis | After feature_analyzer implemented |
| MLflow not production-ready | Local tracking only | Use local MLflow server | After model_trainer implemented |
| No Feast feature store | Manual feature engineering | Direct database queries | After data_preparer implemented |

---

## 1. MLflow

### Purpose
Experiment tracking, model versioning, and registry management.

### Configuration
```yaml
mlflow:
  tracking_uri: "${MLFLOW_TRACKING_URI:-http://localhost:5000}"
  artifact_location: "s3://e2i-mlflow-artifacts"
  experiment_prefix: "e2i"
  
  registry:
    model_name_pattern: "e2i-{brand}-{model_type}"
    stages: ["None", "Staging", "Production", "Archived"]
    
  autolog:
    enabled: true
    log_models: true
    log_input_examples: true
    log_model_signatures: true
```

### Agent Integration
| Agent | Usage |
|-------|-------|
| model_trainer | Log runs, parameters, metrics, artifacts |
| model_selector | Query registry for candidate models |
| model_deployer | Transition model stages, fetch production models |

### Key APIs
```python
# model_trainer
mlflow.start_run(experiment_id=exp_id, run_name=run_name)
mlflow.log_params(hyperparameters)
mlflow.log_metrics({"rmse": rmse, "r2": r2})
mlflow.sklearn.log_model(model, "model")

# model_selector
mlflow.search_registered_models(filter_string="name LIKE 'e2i-%'")

# model_deployer
mlflow.register_model(model_uri, model_name)
client.transition_model_version_stage(name, version, stage)
```

---

## 2. Opik (Comet)

### Purpose
LLM and agent observability, tracing hybrid/deep agent execution.

### Configuration
```yaml
opik:
  api_key: "${OPIK_API_KEY}"
  project_name: "e2i-causal-analytics"
  workspace: "e2i-team"
  
  tracing:
    enabled: true
    sample_rate: 1.0  # 100% for dev, reduce in prod
    
  spans:
    include_inputs: true
    include_outputs: true
    max_output_length: 10000
```

### Agent Integration
| Agent | Usage |
|-------|-------|
| observability_connector | Create spans for all agent executions |
| feature_analyzer | Trace LLM interpretation calls |
| All Hybrid/Deep agents | Automatic span creation via decorator |

### Key APIs
```python
from opik import track, span

@track(name="causal_impact_agent")
async def execute(self, state: AgentState) -> AgentState:
    with span("graph_builder"):
        dag = await self._build_graph(state)
    
    with span("llm_interpretation"):
        interpretation = await self._interpret(dag)
    
    return result
```

---

## 3. Great Expectations

### Purpose
Data quality validation and QC gating before ML training.

### Configuration
```yaml
great_expectations:
  data_docs_site: "s3://e2i-ge-docs"
  checkpoint_store: "s3://e2i-ge-checkpoints"
  
  default_expectations:
    - expect_column_values_to_not_be_null
    - expect_column_values_to_be_unique
    - expect_column_values_to_be_in_set
    
  qc_gate:
    block_on_failure: true
    min_pass_rate: 0.95
    critical_expectations:
      - expect_table_row_count_to_be_between
      - expect_column_values_to_not_be_null
```

### Agent Integration
| Agent | Usage |
|-------|-------|
| data_preparer | Run expectation suites, generate QC reports |

### Key APIs
```python
import great_expectations as gx

context = gx.get_context()
suite = context.get_expectation_suite("e2i_patient_journeys")
results = context.run_checkpoint(checkpoint_name="qc_checkpoint")

if not results.success:
    raise QCGateBlockedError("Data quality check failed")
```

### Expectation Suites
| Suite | Target Table | Critical Checks |
|-------|--------------|-----------------|
| `e2i_patient_journeys` | patient_journeys | patient_id uniqueness, required fields |
| `e2i_treatment_events` | treatment_events | valid treatment types, date ordering |
| `e2i_hcp_profiles` | hcp_profiles | valid specialties, region codes |

---

## 4. Feast

### Purpose
Feature store for consistent feature serving across training and inference.

### Configuration
```yaml
feast:
  project: "e2i_features"
  provider: "aws"
  
  online_store:
    type: "dynamodb"
    region: "us-east-1"
    
  offline_store:
    type: "redshift"
    cluster_id: "e2i-analytics"
    
  registry:
    path: "s3://e2i-feast-registry/registry.pb"
```

### Agent Integration
| Agent | Usage |
|-------|-------|
| data_preparer | Define feature views, materialize features |
| model_trainer | Retrieve training datasets with point-in-time joins |
| prediction_synthesizer | Online feature retrieval for inference |

### Key APIs
```python
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo/")

# Training (offline)
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "patient_features:days_since_diagnosis",
        "hcp_features:total_patients",
        "engagement_features:recent_touches"
    ]
).to_df()

# Inference (online)
feature_vector = store.get_online_features(
    features=["patient_features:risk_score"],
    entity_rows=[{"patient_id": "P001"}]
).to_dict()
```

### Feature Views
| View | Entity | Features |
|------|--------|----------|
| `patient_features` | patient_id | days_since_diagnosis, treatment_count, risk_factors |
| `hcp_features` | hcp_id | specialty, patient_volume, engagement_rate |
| `engagement_features` | patient_id | recent_touches, channel_preferences |

---

## 5. Optuna

### Purpose
Hyperparameter optimization with efficient search strategies.

### Configuration
```yaml
optuna:
  storage: "postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}/optuna"
  
  study_defaults:
    direction: "minimize"
    sampler: "TPESampler"
    pruner: "MedianPruner"
    
  search:
    n_trials: 100
    timeout: 3600  # 1 hour max
    n_jobs: 4
    
  integration:
    mlflow_tracking: true
    log_best_params: true
```

### Agent Integration
| Agent | Usage |
|-------|-------|
| model_trainer | Create studies, run optimization trials |

### Key APIs
```python
import optuna
from optuna.integration import MLflowCallback

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    }
    
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    return mean_squared_error(y_val, model.predict(X_val))

study = optuna.create_study(
    study_name="e2i-conversion-model",
    storage=storage_url,
    direction="minimize"
)

study.optimize(
    objective,
    n_trials=100,
    callbacks=[MLflowCallback(metric_name="mse")]
)
```

---

## 6. SHAP

### Purpose
Model interpretability through Shapley value explanations.

### Configuration
```yaml
shap:
  explainer_type: "auto"  # TreeExplainer, KernelExplainer, etc.
  
  computation:
    max_samples: 1000
    check_additivity: true
    
  storage:
    save_values: true
    table: "ml_shap_analyses"
    
  visualization:
    plot_types: ["summary", "waterfall", "force", "dependence"]
    max_display: 20
```

### Agent Integration
| Agent | Usage |
|-------|-------|
| feature_analyzer | Compute SHAP values, identify interactions |
| explainer | Generate natural language feature explanations |

### Key APIs
```python
import shap

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Feature importance
feature_importance = pd.DataFrame({
    "feature": X_test.columns,
    "importance": np.abs(shap_values).mean(axis=0)
}).sort_values("importance", ascending=False)

# Interaction effects
interaction_values = explainer.shap_interaction_values(X_test)

# Store results
shap_analysis = SHAPAnalysis(
    experiment_id=exp_id,
    model_version=model_version,
    feature_importance=feature_importance.to_dict(),
    top_interactions=extract_top_interactions(interaction_values)
)
```

---

## 7. BentoML

### Purpose
Model serving with containerized endpoints.

### Configuration
```yaml
bentoml:
  bento_store: "s3://e2i-bento-store"
  
  service:
    name: "e2i-prediction-service"
    runners:
      - name: "conversion_model"
        resources:
          cpu: 2
          memory: "4Gi"
          
  deployment:
    target: "kubernetes"  # or "aws-lambda", "aws-sagemaker"
    namespace: "e2i-models"
    
  serving:
    timeout: 30
    max_batch_size: 100
    max_latency_ms: 100
```

### Agent Integration
| Agent | Usage |
|-------|-------|
| model_deployer | Build bentos, deploy services |
| prediction_synthesizer | Call deployed endpoints |

### Key APIs
```python
import bentoml

# Save model to BentoML
saved_model = bentoml.sklearn.save_model(
    "conversion_model",
    model,
    signatures={"predict": {"batchable": True}},
    custom_objects={"preprocessor": preprocessor}
)

# Create service
@bentoml.service(resources={"cpu": "2", "memory": "4Gi"})
class E2IPredictionService:
    model = bentoml.models.get("conversion_model:latest")
    
    @bentoml.api
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        return self.model.predict(input_data)

# Build and deploy
bentoml.build("service:E2IPredictionService")
bentoml.deploy("e2i-prediction-service", target="kubernetes")
```

---

## Environment Variables

Required environment variables for MLOps tools:

```bash
# MLflow
MLFLOW_TRACKING_URI=http://mlflow.e2i.internal:5000
MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com

# Opik
OPIK_API_KEY=your-api-key
OPIK_WORKSPACE=e2i-team

# Great Expectations
GE_DATA_DOCS_S3_BUCKET=e2i-ge-docs

# Feast
FEAST_REGISTRY_PATH=s3://e2i-feast-registry/registry.pb

# Optuna
OPTUNA_STORAGE_URL=postgresql://user:pass@host/optuna

# BentoML
BENTOML_HOME=/opt/bentoml
BENTOML_S3_ENDPOINT_URL=https://s3.amazonaws.com
```

---

## Cross-Tool Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ML Foundation Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [Feast]              [Great Expectations]                          │
│  Feature Store  ───►  Data Quality Check  ───► QC Gate              │
│       │                      │                   │                   │
│       │                      ▼                   │                   │
│       │               [data_preparer]            │                   │
│       │                      │                   │                   │
│       ▼                      ▼                   ▼                   │
│  ┌─────────────────────────────────────────────────────┐            │
│  │                   [model_trainer]                    │            │
│  │  Optuna (HPO) ──► Training ──► MLflow (logging)     │            │
│  └─────────────────────────────────────────────────────┘            │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────┐            │
│  │                 [feature_analyzer]                   │            │
│  │  SHAP (values) ──► Opik (tracing) ──► Interpretation│            │
│  └─────────────────────────────────────────────────────┘            │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────┐            │
│  │                  [model_deployer]                    │            │
│  │  MLflow (registry) ──► BentoML (serving)            │            │
│  └─────────────────────────────────────────────────────┘            │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────┐            │
│  │              [observability_connector]               │            │
│  │  Opik (spans) ──► All Tier 1-5 agents               │            │
│  └─────────────────────────────────────────────────────┘            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Common Issues

| Issue | Tool | Resolution |
|-------|------|------------|
| MLflow connection timeout | MLflow | Check `MLFLOW_TRACKING_URI`, verify network access |
| QC gate blocks unexpectedly | Great Expectations | Review expectation suite, check `min_pass_rate` |
| Feature retrieval slow | Feast | Ensure features are materialized, check online store |
| HPO trials failing | Optuna | Check storage connection, review pruning settings |
| SHAP computation OOM | SHAP | Reduce `max_samples`, use `TreeExplainer` for tree models |
| BentoML deployment fails | BentoML | Check Kubernetes permissions, verify resource limits |

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-08 | Initial V4 creation with 7 MLOps tools |
| 2025-12-08 | Added cross-tool data flow diagram |
| 2025-12-08 | Added troubleshooting section |
