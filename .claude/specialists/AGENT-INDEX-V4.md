# E2I Causal Analytics: Agent Specialist Index V4

## Architecture Overview

The E2I Causal Analytics platform implements an **18-agent, 6-tier architecture** for pharmaceutical commercial operations. Each agent has a specialist documentation file (CLAUDE.md) containing complete LangGraph implementation details.

**V4 Changes**: Added Tier 0 (ML Foundation) with 7 new agents for the complete ML lifecycle.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          E2I AGENT ARCHITECTURE V4                           │
│                          18 Agents, 6 Tiers                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TIER 0: ML FOUNDATION (NEW)                                                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │  SCOPE  │ │  DATA   │ │  MODEL  │ │  MODEL  │ │ FEATURE │ │  MODEL  │   │
│  │ DEFINER │ │PREPARER │ │SELECTOR │ │ TRAINER │ │ANALYZER │ │DEPLOYER │   │
│  │ (Std)   │ │ (Std)   │ │ (Std)   │ │ (Std)   │ │(Hybrid) │ │ (Std)   │   │
│  │  5s     │ │  60s    │ │  120s   │ │ varies  │ │  120s   │ │  30s    │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│                          ┌─────────────────────┐                            │
│                          │    OBSERVABILITY    │ (Cross-cutting)            │
│                          │     CONNECTOR       │                            │
│                          │    (Async, 100ms)   │                            │
│                          └─────────────────────┘                            │
│                                    │                                         │
│  ═══════════════════════════════════════════════════════════════════════════│
│                                                                              │
│  TIER 1: ORCHESTRATION                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         ORCHESTRATOR                                 │    │
│  │            Intent Classification → Routing → Dispatch                │    │
│  │                        (Standard, <2s)                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│  ───────────────────────────────────────────────────────────────────────────│
│                                                                              │
│  TIER 2: CAUSAL INFERENCE                                                    │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                   │
│  │ CAUSAL IMPACT │  │ GAP ANALYZER  │  │ HETEROGENEOUS │                   │
│  │   (Hybrid)    │  │  (Standard)   │  │   OPTIMIZER   │                   │
│  │   120s max    │  │   20s max     │  │  (Standard)   │                   │
│  └───────────────┘  └───────────────┘  └───────────────┘                   │
│                                                                              │
│  ───────────────────────────────────────────────────────────────────────────│
│                                                                              │
│  TIER 3: DESIGN & MONITORING                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                   │
│  │  EXPERIMENT   │  │ DRIFT MONITOR │  │ HEALTH SCORE  │                   │
│  │   DESIGNER    │  │  (Standard)   │  │  (Standard)   │                   │
│  │   (Hybrid)    │  │   10s max     │  │   5s max      │                   │
│  │   60s max     │  └───────────────┘  └───────────────┘                   │
│  └───────────────┘                                                          │
│                                                                              │
│  ───────────────────────────────────────────────────────────────────────────│
│                                                                              │
│  TIER 4: ML PREDICTIONS                                                      │
│  ┌───────────────────────────┐  ┌───────────────────────────┐              │
│  │  PREDICTION SYNTHESIZER   │  │   RESOURCE OPTIMIZER      │              │
│  │       (Standard)          │  │       (Standard)          │              │
│  │        15s max            │  │        20s max            │              │
│  └───────────────────────────┘  └───────────────────────────┘              │
│                                                                              │
│  ───────────────────────────────────────────────────────────────────────────│
│                                                                              │
│  TIER 5: SELF-IMPROVEMENT                                                    │
│  ┌───────────────────────────┐  ┌───────────────────────────┐              │
│  │       EXPLAINER           │  │    FEEDBACK LEARNER       │              │
│  │        (Deep)             │  │     (Deep, Async)         │              │
│  │       45s max             │  │      No RT limit          │              │
│  └───────────────────────────┘  └───────────────────────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Agent Classification Matrix

### Tier 0: ML Foundation (NEW in V4)

| Agent | Tier | Type | Model | Latency | Critical Path |
|-------|------|------|-------|---------|---------------|
| Scope Definer | 0 | Standard | None | <5s | Yes (ML pipeline) |
| Data Preparer | 0 | Standard | None | <60s | Yes (QC Gate) |
| Model Selector | 0 | Standard | None | <120s | Yes (ML pipeline) |
| Model Trainer | 0 | Standard | None | Variable | Yes (ML pipeline) |
| Feature Analyzer | 0 | **Hybrid** | Sonnet | <120s | No |
| Model Deployer | 0 | Standard | None | <30s | No |
| Observability Connector | 0 | Standard (Async) | None | <100ms | No (cross-cutting) |

### Tiers 1-5: Causal Analytics (Unchanged from V3)

| Agent | Tier | Type | Model | Latency | Critical Path |
|-------|------|------|-------|---------|---------------|
| Orchestrator | 1 | Standard (Fast) | Haiku/Sonnet | <2s (strict) | Yes |
| Causal Impact | 2 | Hybrid | Sonnet + Opus | <120s | Yes |
| Gap Analyzer | 2 | Standard | Sonnet | <20s | No |
| Heterogeneous Optimizer | 2 | Standard | Sonnet | <150s | No |
| Experiment Designer | 3 | Hybrid | Sonnet + Opus | <60s | No |
| Drift Monitor | 3 | Standard (Fast) | Haiku/Sonnet | <10s | No |
| Health Score | 3 | Standard (Fast) | Haiku | <5s | No |
| Prediction Synthesizer | 4 | Standard | Sonnet | <15s | No |
| Resource Optimizer | 4 | Standard | Sonnet | <20s | No |
| Explainer | 5 | Deep | Opus/Sonnet | <45s | No |
| Feedback Learner | 5 | Deep (Async) | Opus | Async | No |

## Specialist File Locations

### Tier 0: ML Foundation (NEW)
| Agent | Specialist File | Status |
|-------|-----------------|--------|
| Scope Definer | [scope_definer/CLAUDE.md](specialists/ml_foundation/scope_definer/CLAUDE.md) | ✅ Complete |
| Data Preparer | [data_preparer/CLAUDE.md](specialists/ml_foundation/data_preparer/CLAUDE.md) | ✅ Complete |
| Model Selector | [model_selector/CLAUDE.md](specialists/ml_foundation/model_selector/CLAUDE.md) | ✅ Complete |
| Model Trainer | [model_trainer/CLAUDE.md](specialists/ml_foundation/model_trainer/CLAUDE.md) | ✅ Complete |
| Feature Analyzer | [feature_analyzer/CLAUDE.md](specialists/ml_foundation/feature_analyzer/CLAUDE.md) | ✅ Complete |
| Model Deployer | [model_deployer/CLAUDE.md](specialists/ml_foundation/model_deployer/CLAUDE.md) | ✅ Complete |
| Observability Connector | [observability_connector/CLAUDE.md](specialists/ml_foundation/observability_connector/CLAUDE.md) | ✅ Complete |
| **Tier Overview** | [ml_foundation/CLAUDE.md](specialists/ml_foundation/CLAUDE.md) | ✅ Complete |
| **MLOps Integration** | [mlops/CLAUDE.md](specialists/mlops/CLAUDE.md) | ✅ Complete |

### Tier 1: Orchestration
| Agent | Specialist File | Status |
|-------|-----------------|--------|
| Orchestrator | [orchestrator-agent.md](specialists/orchestrator-agent.md) | ✅ Complete |

### Tier 2: Causal Inference
| Agent | Specialist File | Status |
|-------|-----------------|--------|
| Causal Impact | [causal-impact.md](specialists/causal-impact.md) | ✅ Complete |
| Gap Analyzer | [gap-analyzer.md](specialists/gap-analyzer.md) | ✅ Complete |
| Heterogeneous Optimizer | [heterogeneous-optimizer.md](specialists/heterogeneous-optimizer.md) | ✅ Complete |

### Tier 3: Design & Monitoring
| Agent | Specialist File | Status |
|-------|-----------------|--------|
| Experiment Designer | [experiment-designer.md](specialists/experiment-designer.md) | ✅ Complete |
| Drift Monitor | [drift-monitor.md](specialists/drift-monitor.md) | ✅ Complete |
| Health Score | [health-score.md](specialists/health-score.md) | ✅ Complete |

### Tier 4: ML Predictions
| Agent | Specialist File | Status |
|-------|-----------------|--------|
| Prediction Synthesizer | [prediction-synthesizer.md](specialists/prediction-synthesizer.md) | ✅ Complete |
| Resource Optimizer | [resource-optimizer.md](specialists/resource-optimizer.md) | ✅ Complete |

### Tier 5: Self-Improvement
| Agent | Specialist File | Status |
|-------|-----------------|--------|
| Explainer | [explainer.md](specialists/explainer.md) | ✅ Complete |
| Feedback Learner | [feedback-learner.md](specialists/feedback-learner.md) | ✅ Complete |

## Agent Type Patterns

### Standard Pattern (Fast Path)
- Linear node flow
- Minimal or no LLM usage
- Computational focus
- **Agents**: Orchestrator, Gap Analyzer, Heterogeneous Optimizer, Drift Monitor, Health Score, Prediction Synthesizer, Resource Optimizer, **Scope Definer, Data Preparer, Model Selector, Model Trainer, Model Deployer, Observability Connector**

```
[Input] → [Node 1] → [Node 2] → [Node 3] → [Output]
```

### Hybrid Pattern
- Computation nodes + Deep reasoning node
- Separates deterministic execution from interpretation
- **Agents**: Causal Impact, Experiment Designer, **Feature Analyzer**

```
[Input] → [Computation Nodes] → [Deep Reasoning Node] → [Output]
```

### Deep Pattern
- Extended reasoning throughout
- High latency tolerance
- Async execution possible
- **Agents**: Explainer, Feedback Learner

```
[Input] → [Context Assembly] → [Deep Reasoning] → [Generation] → [Output]
```

## Model Tier Strategy

### No LLM (Tier 0 Standard Agents)
- **Use for**: Computation-only ML pipeline agents
- **Latency target**: Varies by operation
- **Agents**: Scope Definer, Data Preparer, Model Selector, Model Trainer, Model Deployer, Observability Connector

### Haiku (claude-haiku-4-20250414)
- **Use for**: Fast classification, simple routing, quick checks
- **Latency target**: <500ms
- **Agents**: Orchestrator (classification), Health Score

### Sonnet (claude-sonnet-4-20250514)
- **Use for**: Most analytical tasks, balanced speed/quality
- **Latency target**: <30s
- **Agents**: Most Tier 2-4 agents, **Feature Analyzer (interpretation)**

### Opus (claude-opus-4-20250514)
- **Use for**: Complex synthesis, deep reasoning, high-stakes decisions
- **Latency target**: <120s
- **Agents**: Explainer, Feedback Learner, Experiment Designer (validity audit)

## Fallback Chain Strategy

### LLM-Based Agents (Tiers 1-5)

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Primary   │─────►│  Secondary  │─────►│   Fallback  │
│   (Opus)    │      │  (Sonnet)   │      │   (Haiku)   │
└─────────────┘      └─────────────┘      └─────────────┘
    timeout/             timeout/             minimal
    error                error                response
```

### Computation Agents (Tier 0)

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│CausalForest │─────►│  LinearDML  │─────►│     OLS     │
│  (EconML)   │      │  (EconML)   │      │ (sklearn)   │
└─────────────┘      └─────────────┘      └─────────────┘
    timeout/             timeout/             basic
    error                error                estimate
```

## Tier 0 ↔ Tier 1-5 Integration

### Data Flow: ML Foundation → Causal Analytics

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TIER 0: ML FOUNDATION                          │
│                                                                          │
│  scope_definer → data_preparer → model_selector → model_trainer         │
│                        │                              │                  │
│                        ▼                              ▼                  │
│                   QC GATE                       feature_analyzer         │
│                   (blocks if failed)                  │                  │
│                                                       ▼                  │
│                                                 model_deployer           │
│                                                       │                  │
│                        observability_connector (cross-cutting spans)     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         TIER 1-5: CAUSAL ANALYTICS                      │
│                                                                          │
│  orchestrator → causal_impact / gap_analyzer / heterogeneous_optimizer  │
│       │                          │                                       │
│       ▼                          ▼                                       │
│  drift_monitor ←─────── model predictions consumed ──────────────────►  │
│  (uses baselines from       prediction_synthesizer                       │
│   data_preparer)            resource_optimizer                           │
│                                   │                                      │
│                                   ▼                                      │
│                        explainer / feedback_learner                      │
└────────────────────────────────────────────────────────────────────────┘
```

### Cross-Tier Dependencies

| Tier 0 Agent | Consumes From | Provides To |
|--------------|---------------|-------------|
| data_preparer | - | drift_monitor (baselines) |
| model_trainer | - | prediction_synthesizer (models) |
| feature_analyzer | - | causal_impact (feature relationships) |
| model_deployer | - | prediction_synthesizer (endpoints) |
| observability_connector | all agents | health_score (metrics) |

## Integration Contract Index

See [contracts/](contracts/) directory for detailed integration specifications:

| Contract | Description |
|----------|-------------|
| [tier0-contracts.md](contracts/tier0-contracts.md) | **NEW**: ML Foundation agent contracts |
| [orchestrator-dispatch.yaml](contracts/orchestrator-dispatch.yaml) | Orchestrator → Agent dispatch format |
| [agent-handoff.yaml](contracts/agent-handoff.yaml) | Agent → Orchestrator response format |
| [tier2-contracts.md](contracts/tier2-contracts.md) | Causal inference agent contracts |
| [tier3-contracts.md](contracts/tier3-contracts.md) | Design & monitoring agent contracts |
| [tier4-contracts.md](contracts/tier4-contracts.md) | ML prediction agent contracts |
| [tier5-contracts.md](contracts/tier5-contracts.md) | Self-improvement agent contracts |

## Context Summary Index

See [context/](context/) directory for context management:

| Context | Description |
|---------|-------------|
| [brand-context.md](context/brand-context.md) | Brand-specific context (Kisqali, Fabhalta, Remibrutinib) |
| [kpi-dictionary.md](context/kpi-dictionary.md) | KPI definitions and causal relationships |
| [experiment-history.md](context/experiment-history.md) | Historical experiment outcomes |
| [mlops-tools.md](context/mlops-tools.md) | **NEW**: MLOps tool configurations |

## Quick Reference: Agent Selection

### By Query Type

| Query Pattern | Primary Agent | Supporting Agents |
|---------------|---------------|-------------------|
| "What is the effect of X on Y?" | Causal Impact | Explainer |
| "Where are the gaps?" | Gap Analyzer | Resource Optimizer |
| "Who responds best to X?" | Heterogeneous Optimizer | Prediction Synthesizer |
| "Design an experiment for X" | Experiment Designer | Causal Impact |
| "Is the model still accurate?" | Drift Monitor | Health Score, **Data Preparer** |
| "How is the system performing?" | Health Score | Drift Monitor, **Observability Connector** |
| "Predict Y for entity Z" | Prediction Synthesizer | Explainer, **Model Deployer** |
| "How should we allocate budget?" | Resource Optimizer | Gap Analyzer |
| "Explain the analysis" | Explainer | - |
| "Train a new model for X" | **Model Trainer** | Scope Definer, Data Preparer, Model Selector |
| "What features matter most?" | **Feature Analyzer** | Explainer |
| "Deploy model to production" | **Model Deployer** | Model Trainer |

### By Latency Requirement

| Requirement | Agents |
|-------------|--------|
| Real-time (<2s) | Orchestrator |
| Fast (<10s) | Health Score, Drift Monitor, **Scope Definer** |
| Standard (<30s) | Gap Analyzer, Prediction Synthesizer, Resource Optimizer, **Model Deployer** |
| Extended (<120s) | Causal Impact, Heterogeneous Optimizer, **Model Selector, Feature Analyzer** |
| Deep (<180s) | Experiment Designer, Explainer |
| Async (no limit) | Feedback Learner, **Observability Connector** |
| Variable | **Model Trainer, Data Preparer** |

## MLOps Tool Matrix (Tier 0)

| Tool | Primary Agents | Purpose |
|------|----------------|---------|
| MLflow | model_trainer, model_selector, model_deployer | Experiment tracking, model registry |
| Opik | observability_connector, feature_analyzer | LLM/agent observability |
| Great Expectations | data_preparer | Data quality validation |
| Feast | data_preparer, model_trainer | Feature store |
| Optuna | model_trainer | Hyperparameter optimization |
| SHAP | feature_analyzer | Model interpretability |
| BentoML | model_deployer | Model serving |

## Development Guidelines

### Adding a New Agent

1. Determine agent tier (0-5) and type (Standard/Hybrid/Deep)
2. Create specialist file (CLAUDE.md) following template
3. Implement state definition with TypedDict
4. Implement node classes with async execute methods
5. Assemble graph with StateGraph
6. Add integration contracts to appropriate tier file
7. Add to Orchestrator's intent mapping (if Tier 1-5)
8. Add tests in `tests/unit/test_agents/`
9. Update AGENT-INDEX and SPECIALIST-INDEX

### Modifying an Existing Agent

1. Read the relevant specialist file (CLAUDE.md) completely
2. Understand the state definition and node flow
3. Make changes in isolation (one node at a time)
4. Update integration contracts if interface changes
5. Run existing tests before adding new ones
6. Update specialist file documentation

### Testing Requirements

All agents must have:
- Unit tests for each node
- Integration tests for graph flow
- Performance tests for latency
- Error handling tests for fallback chains

**Tier 0 Additional Requirements:**
- QC gate enforcement tests
- ML split compliance tests
- MLOps tool integration tests

## Performance Budgets

| Tier | Latency Budget | Model Budget | Notes |
|------|----------------|--------------|-------|
| 0 | Variable | None (computation) | QC gate may block |
| 1 | <2s (strict) | Haiku preferred | Critical path |
| 2 | <120s | Sonnet primary | Causal computation |
| 3 | <60s | Mixed | Monitoring fast path |
| 4 | <20s | Sonnet | Prediction serving |
| 5 | No hard limit | Opus preferred | Deep reasoning |

## Error Handling Standards

All agents must implement:

1. **Error Classification**
   ```python
   class ErrorCategory(Enum):
       VALIDATION = "validation"
       COMPUTATION = "computation"
       TIMEOUT = "timeout"
       DEPENDENCY = "dependency"
       QC_GATE = "qc_gate"          # NEW: Tier 0
       ML_PIPELINE = "ml_pipeline"   # NEW: Tier 0
   ```

2. **Fallback Chains**
   - Model degradation (Opus → Sonnet → Haiku)
   - Method degradation (CausalForest → LinearDML → OLS)
   - Graceful partial results

3. **Status Tracking**
   ```python
   status: Literal["pending", "processing", "completed", "failed", "blocked"]
   ```

4. **QC Gate Enforcement (Tier 0)**
   ```python
   if qc_report.status == QCStatus.FAILED:
       raise QCGateBlockedError("Training blocked: QC failed")
   ```

---

## Change Log

| Date | Change | Reason |
|------|--------|--------|
| 2025-12-08 | V4: Added Tier 0 ML Foundation (7 agents) | Complete ML lifecycle support |
| 2025-12-08 | V4: Total agents increased 11 → 18 | ML Foundation expansion |
| 2025-12-08 | V4: Added MLOps tool matrix | Tier 0 tool integration |
| 2025-12-08 | V4: Added cross-tier integration diagram | Tier 0 ↔ Tier 1-5 data flow |
| 2025-12-04 | Moved Experiment Designer from Tier 2 to Tier 3 | Standardize tier assignments |
| 2025-12-04 | Updated architecture diagram | Reflect tier correction |
