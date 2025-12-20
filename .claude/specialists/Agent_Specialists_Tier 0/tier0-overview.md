# Tier 0: ML Foundation - Agent Specialist Overview

**Architecture Tier**: 0 (Foundation Layer)
**Total Agents**: 7
**Purpose**: Complete ML lifecycle from problem definition to production deployment
**Dependencies**: None (foundation for all other tiers)

---

## 🎯 Tier Overview

Tier 0 (ML Foundation) is the **base layer** of the E2I 18-agent, 6-tier architecture. All Tiers 1-5 depend on models, features, and infrastructure established by these 7 agents.

### Tier 0 in the Complete Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          E2I AGENT ARCHITECTURE V4                       │
│                          18 Agents, 6 Tiers                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ★ TIER 0: ML FOUNDATION (THIS TIER) ★                                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────┐ │
│  │  SCOPE  │→│  DATA   │→│  MODEL  │→│  MODEL  │→│ FEATURE │→│ MODEL │ │
│  │ DEFINER │ │PREPARER │ │SELECTOR │ │ TRAINER │ │ANALYZER │ │DEPLOY │ │
│  │ (Std)   │ │(Std/QC) │ │ (Std)   │ │ (Std)   │ │(Hybrid) │ │ (Std) │ │
│  │  5s     │ │  60s    │ │  120s   │ │ varies  │ │  120s   │ │  30s  │ │
│  └─────────┘ └────┬────┘ └─────────┘ └─────────┘ └─────────┘ └───────┘ │
│                   │      ┌─────────────────────┐                        │
│              QC GATE     │    OBSERVABILITY    │ (Cross-cutting)        │
│           (blocks on     │     CONNECTOR       │                        │
│            failure)      │    (Async, 100ms)   │                        │
│                          └─────────────────────┘                        │
│                                    ▼                                     │
│  ═══════════════════════════════════════════════════════════════════════│
│                                                                          │
│  TIER 1-5: CONSUME TIER 0 OUTPUTS                                       │
│  - prediction_synthesizer ← Deployed models                             │
│  - drift_monitor ← Baseline metrics                                     │
│  - explainer ← SHAP analyses                                            │
│  - causal_impact ← Feature relationships                                │
│  - orchestrator ← Model metadata                                        │
│  - health_score ← Observability data                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 The 7 ML Foundation Agents

### Agent Classification Matrix

| # | Agent | Type | Model | Latency | Critical Path | Primary Output |
|---|-------|------|-------|---------|---------------|----------------|
| 1 | **Scope Definer** | Standard | None | <5s | Yes (ML) | ScopeSpec, SuccessCriteria |
| 2 | **Data Preparer** | Standard | None | <60s | Yes (QC Gate) | QCReport, BaselineMetrics |
| 3 | **Model Selector** | Standard | None | <120s | Yes (ML) | ModelCandidate, SelectionRationale |
| 4 | **Model Trainer** | Standard | None | Variable | Yes (ML) | TrainedModel, ValidationMetrics |
| 5 | **Feature Analyzer** | **Hybrid** | Sonnet | <120s | No | SHAPAnalysis, FeatureImpacts |
| 6 | **Model Deployer** | Standard | None | <30s | No | DeploymentManifest, VersionRecord |
| 7 | **Observability Connector** | Standard (Async) | None | <100ms | No (cross-cutting) | Spans, QualityMetrics |

---

## 🔗 Agent Pipeline Flow

### Linear Pipeline (Happy Path)

```
1. SCOPE DEFINER
   └─▶ Defines: ML problem, success criteria, constraints
   └─▶ Writes: ml_experiments table
   └─▶ Next: data_preparer

2. DATA PREPARER
   └─▶ Validates: Data quality with Great Expectations
   └─▶ Computes: Baseline metrics for drift monitoring
   └─▶ Checks: Data leakage, schema conformance
   └─▶ Writes: ml_data_quality_reports, ml_feature_store
   └─▶ CRITICAL: QC Gate - blocks pipeline if QC fails
   └─▶ Next: model_selector

3. MODEL SELECTOR
   └─▶ Searches: Algorithm registry for candidates
   └─▶ Filters: By constraints (latency, interpretability, etc.)
   └─▶ Ranks: Candidates by predicted performance
   └─▶ Writes: ml_model_registry (stage: development)
   └─▶ Next: model_trainer

4. MODEL TRAINER
   └─▶ MANDATORY: Checks QC passed (gate enforcement)
   └─▶ Enforces: ML splits (60/20/15/5)
   └─▶ Trains: Model with Optuna hyperparameter tuning
   └─▶ Tracks: MLflow experiment logging
   └─▶ Validates: Against success criteria
   └─▶ Writes: ml_training_runs
   └─▶ Next: feature_analyzer

5. FEATURE ANALYZER (HYBRID)
   └─▶ Computes: SHAP values (no LLM)
   └─▶ Detects: Feature interactions (no LLM)
   └─▶ Interprets: Importance narratives (Sonnet LLM)
   └─▶ Writes: ml_shap_analyses, semantic_memory
   └─▶ Next: model_deployer

6. MODEL DEPLOYER
   └─▶ Promotes: Model to staging/production
   └─▶ Deploys: BentoML endpoint
   └─▶ Manages: Model lifecycle (stage transitions)
   └─▶ Enables: Rollback procedures
   └─▶ Writes: ml_deployments, ml_model_registry (stage: production)
   └─▶ Handoff: To Tier 1-5 agents

7. OBSERVABILITY CONNECTOR (Cross-Cutting)
   └─▶ Spans: All Tier 0 operations
   └─▶ Metrics: Quality, latency, errors
   └─▶ Emits: Opik traces
   └─▶ Writes: ml_observability_spans
   └─▶ Consumed by: health_score, orchestrator
```

### The QC Gate (Critical Workflow)

```
data_preparer
    │
    ├─ QC PASS ──▶ Continue pipeline ──▶ model_trainer
    │
    └─ QC FAIL ──▶ Block pipeline ──▶ status: "blocked"
                   │
                   └─▶ Manual intervention or re-run with fixed data
```

**CRITICAL**: `model_trainer` MUST check QC status before training. Never train on failed QC data.

---

## 🗄️ Database Schema (Tier 0 Tables)

### 8 New ML Tables (Migration 007)

| Table | Primary Writers | Primary Readers | Purpose |
|-------|-----------------|-----------------|---------|
| `ml_experiments` | scope_definer | data_preparer, model_selector, model_trainer | Experiment metadata |
| `ml_data_quality_reports` | data_preparer | model_trainer (gate), drift_monitor | QC validation results |
| `ml_feature_store` | data_preparer | model_trainer, prediction_synthesizer | Feature definitions |
| `ml_model_registry` | model_selector, model_deployer | prediction_synthesizer, orchestrator | Model versions |
| `ml_training_runs` | model_trainer | feature_analyzer, model_deployer, feedback_learner | Training execution |
| `ml_shap_analyses` | feature_analyzer | explainer, causal_impact | Interpretability results |
| `ml_deployments` | model_deployer | prediction_synthesizer, health_score | Deployment records |
| `ml_observability_spans` | observability_connector | health_score, orchestrator | Telemetry data |

---

## 🛠️ MLOps Tool Stack

### 7 Integrated MLOps Tools

| Tool | Version | Primary Agents | Purpose |
|------|---------|----------------|---------|
| **MLflow** | ≥2.10 | model_trainer, model_selector, model_deployer | Experiment tracking, model registry, versioning |
| **Opik** | ≥0.1 | observability_connector, feature_analyzer | LLM/agent observability, trace visualization |
| **Great Expectations** | ≥0.18 | data_preparer | Data quality validation, QC gate enforcement |
| **Feast** | ≥0.35 | data_preparer, model_trainer | Feature store, feature serving |
| **Optuna** | ≥3.5 | model_trainer | Hyperparameter optimization, automated tuning |
| **SHAP** | ≥0.44 | feature_analyzer | Model interpretability, feature importance |
| **BentoML** | ≥1.2 | model_deployer | Model serving, endpoint management |

---

## 🔄 Handoffs to Tiers 1-5

### Tier 0 Outputs Consumed by Other Tiers

| Output | From Agent | To Agent (Tier) | Purpose |
|--------|-----------|-----------------|---------|
| Deployed model endpoints | model_deployer | prediction_synthesizer (Tier 4) | Run inference for predictions |
| SHAP analyses | feature_analyzer | explainer (Tier 5) | Generate natural language explanations |
| Feature relationships | feature_analyzer | causal_impact (Tier 2) | Inform causal graph construction |
| Baseline metrics | data_preparer | drift_monitor (Tier 3) | Detect distribution/concept drift |
| Observability data | observability_connector | health_score (Tier 3) | System health monitoring |
| Model metadata | model_deployer | orchestrator (Tier 1) | Route queries by model capability |

---

## 🧠 Memory Architecture

### Memory Types Used by Tier 0

#### Working Memory (Redis)
- **All agents**: Active context, scratchpad, immediate state
- **TTL**: 86400 seconds (24 hours)
- **Use**: Current operation context

#### Episodic Memory (Supabase/pgvector)
- **scope_definer**: Past scope definitions for similar use cases
- **data_preparer**: Historical QC reports and patterns
- **model_trainer**: Training run logs and outcomes
- **model_deployer**: Deployment history and rollback events
- **observability_connector**: Span archives

#### Procedural Memory (Supabase/pgvector)
- **scope_definer**: Successful scope patterns by use case type
- **data_preparer**: Effective validation patterns
- **model_selector**: Algorithm selection heuristics
- **model_trainer**: Winning hyperparameter configurations
- **model_deployer**: Successful deployment procedures

#### Semantic Memory (FalkorDB/Graphity)
- **feature_analyzer**: Feature relationships, interaction graphs
- **Feeds into**: causal_impact (Tier 2), explainer (Tier 5)

---

## 📐 Agent Type Patterns

### Standard Pattern (6 agents)

Linear node flow, computational focus, minimal or no LLM usage.

```
[Input] → [Computation] → [Validation] → [Persistence] → [Output]
```

**Agents**: Scope Definer, Data Preparer, Model Selector, Model Trainer, Model Deployer, Observability Connector

**Characteristics**:
- Deterministic execution
- No LLM inference (computation only)
- Fast execution (<120s except model_trainer)
- Error handling via computational fallbacks

### Hybrid Pattern (1 agent)

Computation nodes + Deep reasoning node. Separates deterministic execution from interpretation.

```
[SHAP Computation] → [Interaction Detection] → [LLM Interpretation] → [Output]
     (no LLM)              (no LLM)                (Sonnet)
```

**Agent**: Feature Analyzer

**Characteristics**:
- Computation phase: Fast, deterministic
- Interpretation phase: LLM-powered, slower
- Clear separation of concerns
- Fallback: Skip interpretation if LLM fails

---

## ⚠️ Critical Workflows & Constraints

### 1. QC Gate Enforcement

**MANDATORY PATTERN**:

```python
# In model_trainer agent - MUST CHECK QC BEFORE TRAINING
async def execute(self, state: ModelTrainerState) -> ModelTrainerState:
    # STEP 1: MANDATORY QC CHECK
    qc_report = await self._fetch_qc_report(state["experiment_id"])

    if qc_report.status == QCStatus.FAILED:
        return {
            **state,
            "errors": [{"node": "model_trainer", "error": "QC gate blocked"}],
            "status": "blocked"  # Note: "blocked" not "failed"
        }

    # STEP 2: Proceed with training only if QC passed
    # ...
```

**Why Critical**: Prevents training on corrupted/invalid data that would produce unreliable models.

### 2. ML Split Enforcement

**MANDATORY PATTERN**:

```python
# Always use these splits
TRAIN_RATIO = 0.60  # 60%
VAL_RATIO = 0.20    # 20%
TEST_RATIO = 0.15   # 15%
HOLDOUT_RATIO = 0.05  # 5%

# NEVER access test/holdout data during training/validation
# NEVER use data from future time periods in historical predictions (temporal leakage)
```

**Why Critical**: Prevents data leakage that would invalidate model performance estimates.

### 3. Sequential Dependencies

**Pipeline Order (MUST BE RESPECTED)**:

```
scope_definer → data_preparer → model_selector → model_trainer →
feature_analyzer → model_deployer
```

**Why Critical**: Each agent depends on outputs from previous agents. Out-of-order execution will fail.

### 4. Observability Spanning

**MANDATORY PATTERN**:

```python
# All Tier 0 operations MUST emit spans
from src.mlops.opik_connector import emit_span

async def execute(self, state: AgentState) -> AgentState:
    with emit_span("agent_operation", metadata=state.metadata):
        # Agent logic here
        result = await self._do_work(state)
    return result
```

**Why Critical**: Without spans, health_score and orchestrator lack visibility into Tier 0 operations.

---

## 🚨 Common Pitfalls & Anti-Patterns

### ❌ NEVER Do These

1. **Skip QC Check in Training**
   ```python
   # ❌ WRONG - Training without QC check
   async def execute(self, state):
       model = await train_model(state["data"])
       return {"model": model}
   ```
   ```python
   # ✅ CORRECT - Always check QC first
   async def execute(self, state):
       qc_report = await get_qc_report(state["experiment_id"])
       if qc_report.status == "failed":
           return {"status": "blocked"}
       model = await train_model(state["data"])
       return {"model": model}
   ```

2. **Data Leakage in Preprocessing**
   ```python
   # ❌ WRONG - Fitting scaler on all data
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)  # Leaks test data!
   X_train, X_test = train_test_split(X_scaled)
   ```
   ```python
   # ✅ CORRECT - Fit only on train, transform on test
   X_train, X_test = train_test_split(X)
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)  # Only transform
   ```

3. **Missing Observability Spans**
   ```python
   # ❌ WRONG - No span emission
   async def deploy_model(self, model):
       endpoint = await bentoml.deploy(model)
       return endpoint
   ```
   ```python
   # ✅ CORRECT - Emit span
   async def deploy_model(self, model):
       with emit_span("model_deployment", model_id=model.id):
           endpoint = await bentoml.deploy(model)
       return endpoint
   ```

4. **Unversioned Models**
   ```python
   # ❌ WRONG - Model not registered
   model = train_model(data)
   pickle.dump(model, "model.pkl")  # No versioning!
   ```
   ```python
   # ✅ CORRECT - Register in MLflow
   model = train_model(data)
   mlflow.sklearn.log_model(model, "model", registered_model_name="churn_predictor")
   ```

5. **Stale Baselines Not Updated**
   ```python
   # ❌ WRONG - drift_monitor uses old baselines
   # data_preparer computes new baselines but doesn't notify drift_monitor
   ```
   ```python
   # ✅ CORRECT - Update drift_monitor when baselines change
   baselines = await data_preparer.compute_baselines(data)
   await drift_monitor.update_baselines(baselines)
   ```

---

## 📁 Specialist File Locations

### Individual Agent Specialists

| Agent | Specialist File | Description |
|-------|-----------------|-------------|
| Scope Definer | [ml_foundation/scope_definer.md](../ml_foundation/scope_definer.md) | ML problem definition, success criteria validation |
| Data Preparer | [ml_foundation/data_preparer.md](../ml_foundation/data_preparer.md) | QC gating, Great Expectations, baseline computation |
| Model Selector | [ml_foundation/model_selector.md](../ml_foundation/model_selector.md) | Algorithm registry, constraint filtering, ranking |
| Model Trainer | [ml_foundation/model_trainer.md](../ml_foundation/model_trainer.md) | Split enforcement, Optuna tuning, MLflow tracking |
| Feature Analyzer | [ml_foundation/feature_analyzer.md](../ml_foundation/feature_analyzer.md) | SHAP computation + LLM interpretation (hybrid) |
| Model Deployer | [ml_foundation/model_deployer.md](../ml_foundation/model_deployer.md) | Stage lifecycle, BentoML deployment, rollback |
| Observability Connector | [ml_foundation/observability_connector.md](../ml_foundation/observability_connector.md) | Opik spans, cross-cutting telemetry |

### Tier 0 Overview
| Document | File | Description |
|----------|------|-------------|
| **Tier Overview** | [ml_foundation/CLAUDE.md](../ml_foundation/CLAUDE.md) | Comprehensive Tier 0 guide, data flow, implementation patterns |
| **MLOps Integration** | [MLOps_Integration/mlops_integration.md](../MLOps_Integration/mlops_integration.md) | 7 MLOps tool integrations, configuration |

---

## 📋 Integration Contracts

### Contract Files

| Contract | File | Purpose |
|----------|------|---------|
| **Tier 0 Contracts** | [../../contracts/Tier-Specific Contracts/tier0-contracts.md](../../contracts/Tier-Specific Contracts/tier0-contracts.md) | All ML Foundation agent input/output contracts |
| Base Contract | [../../contracts/Base Structures/base-contract.md](../../contracts/Base Structures/base-contract.md) | Base input/output structures |
| Orchestrator Contracts | [../../contracts/Orchestrator Contracts/orchestrator-contracts.md](../../contracts/Orchestrator Contracts/orchestrator-contracts.md) | Orchestrator dispatch/receive (Tier 1 ↔ Tier 0) |

---

## 🧪 Testing Requirements

### Mandatory Tests for All Tier 0 Agents

```
tests/unit/test_agents/test_ml_foundation/test_<agent_name>/
├── test_<node_1>.py              # Unit test for each node
├── test_<node_2>.py
├── test_integration.py           # Full graph flow
├── test_performance.py           # Latency SLA compliance
├── test_error_handling.py        # Fallback chains
├── test_qc_gate.py               # QC gate enforcement (data_preparer, model_trainer)
├── test_split_compliance.py      # ML split validation (data_preparer, model_trainer)
└── test_mlops_integration.py     # Tool integration (all agents)
```

### Key Test Scenarios

1. **End-to-End Pipeline**: scope_definer → data_preparer → model_selector → model_trainer → feature_analyzer → model_deployer
2. **QC Gate Blocking**: Verify model_trainer blocks when QC fails
3. **Data Leakage Detection**: Verify data_preparer catches temporal/target leakage
4. **SHAP Accuracy**: Verify feature_analyzer SHAP values match ground truth
5. **Deployment Rollback**: Verify model_deployer rollback procedures work
6. **Observability Coverage**: Verify observability_connector captures all Tier 0 spans

---

## ⚡ Performance Budgets

| Agent | SLA | Model Budget | Notes |
|-------|-----|--------------|-------|
| Scope Definer | <5s | None | Fast problem definition |
| Data Preparer | <60s | None | QC validation may take time |
| Model Selector | <120s | None | Registry search and ranking |
| Model Trainer | Variable | None | Depends on model complexity, dataset size |
| Feature Analyzer | <120s | Sonnet | SHAP computation + interpretation |
| Model Deployer | <30s | None | Fast deployment to staging/production |
| Observability Connector | <100ms | None | Async span emission |

---

## 🔧 Error Handling Standards

### Error Categories (Tier 0 Specific)

```python
class ErrorCategory(Enum):
    VALIDATION = "validation"        # Invalid input
    COMPUTATION = "computation"      # Algorithm failure
    TIMEOUT = "timeout"              # SLA exceeded
    DEPENDENCY = "dependency"        # External service failure
    QC_GATE = "qc_gate"             # QC validation failed (Tier 0)
    ML_PIPELINE = "ml_pipeline"      # Pipeline sequencing error (Tier 0)
```

### Status Values

```python
status: Literal["pending", "processing", "completed", "failed", "blocked"]

# "blocked" - Used when QC gate prevents progress (not an error, intentional block)
# "failed" - Used for actual errors (timeouts, exceptions, validation failures)
```

### Fallback Chains

#### For Computational Agents

```python
# Example: model_trainer fallback
ComputationFallbackChain([
    "CausalForest",    # Primary (most accurate, slowest)
    "LinearDML",       # Fallback 1 (faster, less accurate)
    "OLS"              # Fallback 2 (fastest, least accurate)
])
```

#### For Hybrid Agent (feature_analyzer)

```python
# LLM interpretation fallback
ModelFallbackChain([
    "claude-sonnet-4-20250514",  # Primary
    "claude-haiku-4-20250414",   # Fallback 1
    "template_response"          # Fallback 2 (static template)
])
```

---

## 📖 Related Documentation

### E2I Architecture
- [AGENT-INDEX-V4.md](../AGENT-INDEX-V4.md) - Complete 18-agent architecture
- [SPECIALIST-INDEX-V4.md](../SPECIALIST-INDEX-V4.md) - All specialist files index

### Framework Resources
- [.agent_docs/ml-patterns.md](../../.agent_docs/ml-patterns.md) - ML best practices, data leakage prevention
- [.agent_docs/coding-patterns.md](../../.agent_docs/coding-patterns.md) - General coding standards
- [.agent_docs/testing-patterns.md](../../.agent_docs/testing-patterns.md) - Testing strategies

### E2I Context
- [context/summary-v4.md](../../context/summary-v4.md) - E2I project summary
- [context/mlops-tools.md](../../context/mlops-tools.md) - MLOps stack configuration
- [context/kpi-dictionary.md](../../context/kpi-dictionary.md) - KPI definitions

---

## 🎯 Quick Reference: When to Use Tier 0 Agents

### By Query/Task Type

| User Query | Primary Agent | Supporting Agents |
|-----------|---------------|-------------------|
| "Train a new model for X" | model_trainer | scope_definer, data_preparer, model_selector |
| "What features matter most for Y?" | feature_analyzer | explainer (Tier 5) |
| "Deploy model to production" | model_deployer | model_trainer (prerequisite) |
| "Validate data quality for Z" | data_preparer | - |
| "Is the model still accurate?" | drift_monitor (Tier 3) | data_preparer (baselines) |
| "How is the ML pipeline performing?" | health_score (Tier 3) | observability_connector |
| "Define ML problem for A" | scope_definer | - |
| "Which algorithm should we use?" | model_selector | scope_definer |

### By Development Phase

| Phase | Primary Agents | Output |
|-------|----------------|--------|
| **Problem Definition** | scope_definer | ScopeSpec, SuccessCriteria |
| **Data Validation** | data_preparer | QCReport, BaselineMetrics |
| **Model Selection** | model_selector | ModelCandidate |
| **Model Training** | model_trainer | TrainedModel, ValidationMetrics |
| **Model Interpretation** | feature_analyzer | SHAPAnalysis, FeatureImpacts |
| **Model Deployment** | model_deployer | DeploymentManifest |
| **Monitoring Setup** | observability_connector | Spans, QualityMetrics |

---

## ✅ Development Checklist

### Adding a New Tier 0 Agent

- [ ] Define agent tier (0), type (Standard/Hybrid), SLA
- [ ] Create specialist file in `ml_foundation/<agent-name>.md`
- [ ] Implement state definition with TypedDict
- [ ] Implement node classes with async execute methods
- [ ] Add QC gate check if agent consumes data_preparer outputs
- [ ] Enforce ML splits if agent accesses training data
- [ ] Emit observability spans for all operations
- [ ] Add integration contracts to `tier0-contracts.md`
- [ ] Add unit tests, integration tests, performance tests
- [ ] Update AGENT-INDEX-V4.md and SPECIALIST-INDEX-V4.md
- [ ] Register MLOps tools in `mlops-tools.md`
- [ ] Add database migrations if new tables needed

### Modifying an Existing Tier 0 Agent

- [ ] Read agent specialist file completely
- [ ] Understand state definition and node flow
- [ ] Verify QC gate handling (if applicable)
- [ ] Verify ML split compliance (if applicable)
- [ ] Make changes in isolation (one node at a time)
- [ ] Update integration contracts if interface changes
- [ ] Run existing tests before adding new ones
- [ ] Update observability spans if new operations added
- [ ] Update specialist file documentation
- [ ] Update AGENT-INDEX-V4.md if tier/type/SLA changes

---

**Version**: V4.0
**Last Updated**: 2025-12-17
**Tier**: 0 (ML Foundation)
**Total Agents**: 7
**Agent Types**: 6 Standard, 1 Hybrid
**Critical Workflows**: QC Gate, ML Split Enforcement, Sequential Dependencies, Observability Spanning
