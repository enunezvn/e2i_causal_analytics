# E2I Causal Analytics: Agent Specialist Index V4

## Architecture Overview

The E2I Causal Analytics platform uses an **18-agent, 6-tier architecture** with 100% KPI coverage. This document serves as the master index for all agent specialist files.

**V4 Changes**: Added Tier 0 (ML Foundation) with 7 new agents, 8 new database tables, and MLOps tool integrations.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          E2I AGENT ARCHITECTURE V4                           │
│                          18 Agents, 6 Tiers                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TIER 0: ML FOUNDATION (NEW)                                                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │  SCOPE  │→│  DATA   │→│  MODEL  │→│  MODEL  │→│ FEATURE │→│  MODEL  │   │
│  │ DEFINER │ │PREPARER │ │SELECTOR │ │ TRAINER │ │ANALYZER │ │DEPLOYER │   │
│  │ (Std)   │ │(Std/QC) │ │ (Std)   │ │ (Std)   │ │(Hybrid) │ │ (Std)   │   │
│  │  5s     │ │  60s    │ │  120s   │ │ varies  │ │  120s   │ │  30s    │   │
│  └─────────┘ └────┬────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│                   │      ┌─────────────────────┐                            │
│              QC GATE     │    OBSERVABILITY    │ (Cross-cutting)            │
│           (blocks on     │     CONNECTOR       │                            │
│            failure)      │    (Async, 100ms)   │                            │
│                          └─────────────────────┘                            │
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

### Summary by Tier

| Tier | Name | Agents | Types | Primary Purpose |
|------|------|--------|-------|-----------------|
| 0 | ML Foundation | 7 | 6 Standard, 1 Hybrid | ML lifecycle, MLOps |
| 1 | Orchestration | 1 | 1 Standard (Fast) | Routing, dispatch |
| 2 | Causal Inference | 3 | 1 Hybrid, 2 Standard | Causal analytics |
| 3 | Design & Monitoring | 3 | 1 Hybrid, 2 Standard | Experiments, health |
| 4 | ML Predictions | 2 | 2 Standard | Forecasting, optimization |
| 5 | Self-Improvement | 2 | 2 Deep | Explanation, learning |
| **Total** | | **18** | **13 Std, 3 Hybrid, 2 Deep** | |

### Full Classification Matrix

| Agent | Tier | Type | Model Tier | Latency | Critical Path |
|-------|------|------|------------|---------|---------------|
| **Scope Definer** | 0 | Standard | None | <5s | Yes (ML) |
| **Data Preparer** | 0 | Standard | None | <60s | Yes (QC Gate) |
| **Model Selector** | 0 | Standard | None | <120s | Yes (ML) |
| **Model Trainer** | 0 | Standard | None | Variable | Yes (ML) |
| **Feature Analyzer** | 0 | Hybrid | Sonnet | <120s | No |
| **Model Deployer** | 0 | Standard | None | <30s | No |
| **Observability Connector** | 0 | Standard (Async) | None | <100ms | No |
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

## Specialist File Index

### Tier 0: ML Foundation (NEW)

| Agent | File | Description |
|-------|------|-------------|
| **Tier Overview** | [ml_foundation/CLAUDE.md](specialists/ml_foundation/CLAUDE.md) | Tier 0 overview, data flow, MLOps integration |
| Scope Definer | [scope_definer/CLAUDE.md](specialists/ml_foundation/scope_definer/CLAUDE.md) | ML problem definition, success criteria |
| Data Preparer | [data_preparer/CLAUDE.md](specialists/ml_foundation/data_preparer/CLAUDE.md) | QC gating, Great Expectations, baselines |
| Model Selector | [model_selector/CLAUDE.md](specialists/ml_foundation/model_selector/CLAUDE.md) | Algorithm registry, constraint filtering |
| Model Trainer | [model_trainer/CLAUDE.md](specialists/ml_foundation/model_trainer/CLAUDE.md) | Split enforcement, Optuna tuning |
| Feature Analyzer | [feature_analyzer/CLAUDE.md](specialists/ml_foundation/feature_analyzer/CLAUDE.md) | SHAP computation + LLM interpretation |
| Model Deployer | [model_deployer/CLAUDE.md](specialists/ml_foundation/model_deployer/CLAUDE.md) | Stage lifecycle, BentoML deployment |
| Observability Connector | [observability_connector/CLAUDE.md](specialists/ml_foundation/observability_connector/CLAUDE.md) | Opik spans, cross-cutting telemetry |
| **MLOps Integration** | [mlops/CLAUDE.md](specialists/mlops/CLAUDE.md) | 7 MLOps tool integrations |

### Tier 1: Orchestration
| Agent | File | Description |
|-------|------|-------------|
| Orchestrator | [orchestrator-agent.md](specialists/orchestrator-agent.md) | Intent classification, routing, dispatch |

### Tier 2: Causal Inference
| Agent | File | Description |
|-------|------|-------------|
| Causal Impact | [causal-impact.md](specialists/causal-impact.md) | DoWhy/EconML causal effect estimation |
| Gap Analyzer | [gap-analyzer.md](specialists/gap-analyzer.md) | ROI opportunity detection |
| Heterogeneous Optimizer | [heterogeneous-optimizer.md](specialists/heterogeneous-optimizer.md) | CATE estimation, segment optimization |

### Tier 3: Design & Monitoring
| Agent | File | Description |
|-------|------|-------------|
| Experiment Designer | [experiment-designer.md](specialists/experiment-designer.md) | Experiment design with validity audit |
| Drift Monitor | [drift-monitor.md](specialists/drift-monitor.md) | Data/model/concept drift detection |
| Health Score | [health-score.md](specialists/health-score.md) | System health monitoring |

### Tier 4: ML Predictions
| Agent | File | Description |
|-------|------|-------------|
| Prediction Synthesizer | [prediction-synthesizer.md](specialists/prediction-synthesizer.md) | Multi-model ensemble predictions |
| Resource Optimizer | [resource-optimizer.md](specialists/resource-optimizer.md) | Resource allocation optimization |

### Tier 5: Self-Improvement
| Agent | File | Description |
|-------|------|-------------|
| Explainer | [explainer.md](specialists/explainer.md) | Natural language explanations |
| Feedback Learner | [feedback-learner.md](specialists/feedback-learner.md) | Async learning from feedback |

## Agent Type Patterns

### Standard Pattern
Linear node flow, computational focus, minimal or no LLM usage.

```
[Input] → [Node 1] → [Node 2] → [Node 3] → [Output]
```

**Tier 0 Examples**: Scope Definer, Data Preparer, Model Selector, Model Trainer, Model Deployer, Observability Connector
**Tier 1-5 Examples**: Orchestrator, Gap Analyzer, Heterogeneous Optimizer, Drift Monitor, Health Score, Prediction Synthesizer, Resource Optimizer

### Hybrid Pattern
Computation nodes + Deep reasoning nodes. Separates deterministic execution from interpretation.

```
[Computation] → [Computation] → [Deep Reasoning] → [Output]
```

**Tier 0 Examples**: Feature Analyzer (SHAP → Interactions → Interpretation)
**Tier 1-5 Examples**: Causal Impact (compute → interpret), Experiment Designer (design → compute → audit → generate)

### Deep Pattern
Extended reasoning throughout. High latency tolerance, often async.

```
[Context] → [Deep Reasoning] → [Generation] → [Output]
```

**Examples**: Explainer, Feedback Learner

## Integration Contracts

All agents follow standardized contracts for inter-agent communication:

### Contract Files
| Contract | File | Purpose |
|----------|------|---------|
| **Tier 0 Contracts** | [contracts/tier0-contracts.md](contracts/tier0-contracts.md) | **NEW**: ML Foundation agents |
| Base Contract | [contracts/base-contract.md](contracts/base-contract.md) | Base input/output structures |
| Orchestrator Contracts | [contracts/orchestrator-contracts.md](contracts/orchestrator-contracts.md) | Orchestrator dispatch/receive |
| Tier 2 Contracts | [contracts/tier2-contracts.md](contracts/tier2-contracts.md) | Causal inference agents |
| Tier 3 Contracts | [contracts/tier3-contracts.md](contracts/tier3-contracts.md) | Design & monitoring agents |
| Tier 4 Contracts | [contracts/tier4-contracts.md](contracts/tier4-contracts.md) | ML prediction agents |
| Tier 5 Contracts | [contracts/tier5-contracts.md](contracts/tier5-contracts.md) | Self-improvement agents |

## Context Summaries

Context files for maintaining state across conversations:

| Context | File | Purpose |
|---------|------|---------|
| Brand Context | [context/brands.md](context/brands.md) | Remibrutinib, Fabhalta, Kisqali |
| KPI Dictionary | [context/kpis.md](context/kpis.md) | KPI definitions and relationships |
| Data Sources | [context/data-sources.md](context/data-sources.md) | Available data assets |
| Experiment History | [context/experiment-history.md](context/experiment-history.md) | Historical experiment outcomes |
| **MLOps Tools** | [context/mlops-tools.md](context/mlops-tools.md) | **NEW**: MLOps configuration |

## Database Tables by Tier

### Tier 0: ML Foundation (8 NEW Tables)

| Table | Primary Agent | Purpose |
|-------|---------------|---------|
| `ml_experiments` | scope_definer | Experiment tracking |
| `ml_data_quality_reports` | data_preparer | QC validation results |
| `ml_feature_store` | data_preparer | Feature definitions |
| `ml_model_registry` | model_selector, model_deployer | Model versions |
| `ml_training_runs` | model_trainer | Training execution |
| `ml_shap_analyses` | feature_analyzer | Interpretability results |
| `ml_deployments` | model_deployer | Deployment records |
| `ml_observability_spans` | observability_connector | Telemetry data |

### Tiers 1-5: Existing Tables

See database schema documentation for complete list.

## MLOps Tool Matrix (Tier 0)

| Tool | Version | Primary Agents | Purpose |
|------|---------|----------------|---------|
| MLflow | ≥2.10 | model_trainer, model_selector, model_deployer | Experiment tracking, registry |
| Opik | ≥0.1 | observability_connector, feature_analyzer | LLM/agent observability |
| Great Expectations | ≥0.18 | data_preparer | Data quality validation |
| Feast | ≥0.35 | data_preparer, model_trainer | Feature store |
| Optuna | ≥3.5 | model_trainer | Hyperparameter optimization |
| SHAP | ≥0.44 | feature_analyzer | Model interpretability |
| BentoML | ≥1.2 | model_deployer | Model serving |

## Development Guidelines

### Creating a New Agent

1. **Determine Classification**
   - Tier: 0 (ML Foundation), 1-5 (Causal Analytics)
   - Type: Standard, Hybrid, or Deep
   - Model: None, Haiku, Sonnet, or Opus

2. **Create Specialist File**
   ```
   # Tier 0
   specialists/ml_foundation/<agent-name>/CLAUDE.md
   
   # Tiers 1-5
   specialists/<agent-name>.md
   ```

3. **Define State**
   - Input fields
   - Computation outputs
   - Final outputs
   - Error handling

4. **Implement Nodes**
   - Each node is async
   - Handle timeouts
   - Include fallbacks

5. **Build Graph**
   - Use LangGraph StateGraph
   - Add conditional edges
   - Include error handlers

6. **Define Contracts**
   - Input BaseModel
   - Output BaseModel
   - Handoff YAML format

7. **Update Index Files**
   - AGENT-INDEX-V4.md
   - SPECIALIST-INDEX-V4.md

### Testing Requirements

Each agent must have:
```
tests/unit/test_agents/test_<agent_name>/
├── test_<node_1>.py
├── test_<node_2>.py
├── test_integration.py
└── test_performance.py
```

**Tier 0 Additional:**
```
tests/unit/test_agents/test_<agent_name>/
├── test_qc_gate.py           # QC gate enforcement
├── test_split_compliance.py  # ML split validation
└── test_mlops_integration.py # Tool integration
```

## Performance Budgets

| Tier | Latency Budget | Model Budget | Notes |
|------|----------------|--------------|-------|
| 0 | Variable | None (computation) | QC gate may block |
| 1 | <2s total | Haiku preferred | Critical path |
| 2 | <120s per agent | Sonnet primary | Causal computation |
| 3 | <60s per agent | Mixed | Monitoring fast path |
| 4 | <20s per agent | Sonnet | Prediction serving |
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
       QC_GATE = "qc_gate"          # Tier 0
       ML_PIPELINE = "ml_pipeline"   # Tier 0
   ```

2. **Fallback Chains**
   - Model degradation (Opus → Sonnet → Haiku)
   - Method degradation (CausalForest → LinearDML → OLS)
   - Graceful partial results

3. **Status Tracking**
   ```python
   status: Literal["pending", "processing", "completed", "failed", "blocked"]
   ```

## Quick Reference

### Common Import Pattern
```python
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
import operator
```

### Standard Node Template
```python
class NodeName:
    async def execute(self, state: AgentState) -> AgentState:
        start_time = time.time()
        
        try:
            # Node logic here
            result = await self._process(state)
            
            latency = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "result": result,
                "latency_ms": latency,
                "status": "next_status"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "node_name", "error": str(e)}],
                "status": "failed"
            }
```

### Tier 0 QC Gate Pattern
```python
# In model_trainer, MUST check QC before training
async def execute(self, state: ModelTrainerState) -> ModelTrainerState:
    # MANDATORY QC CHECK
    qc_report = await self._fetch_qc_report(state["experiment_id"])
    
    if qc_report.status == QCStatus.FAILED:
        return {
            **state,
            "errors": [{"node": "model_trainer", "error": "QC gate blocked"}],
            "status": "blocked"  # Note: "blocked" not "failed"
        }
    
    # Proceed with training...
```

### Graph Assembly Template
```python
def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("node1", node1.execute)
    workflow.add_node("node2", node2.execute)
    workflow.add_node("error_handler", error_handler)
    
    workflow.set_entry_point("node1")
    
    workflow.add_conditional_edges(
        "node1",
        lambda s: "error" if s.get("status") == "failed" else "node2",
        {"node2": "node2", "error": "error_handler"}
    )
    
    workflow.add_edge("node2", END)
    workflow.add_edge("error_handler", END)
    
    return workflow.compile()
```

---

## Change Log

| Date | Change | Reason |
|------|--------|--------|
| 2025-12-08 | V4: Added Tier 0 ML Foundation section | 7 new agents |
| 2025-12-08 | V4: Added MLOps tool matrix | Tier 0 tool integration |
| 2025-12-08 | V4: Added database tables by tier | 8 new ml_ tables |
| 2025-12-08 | V4: Updated agent count 11 → 18 | ML Foundation expansion |
| 2025-12-08 | V4: Added QC gate pattern | Tier 0 critical workflow |
| 2025-12-04 | Moved Experiment Designer to Tier 3 | Standardize tier assignments |
