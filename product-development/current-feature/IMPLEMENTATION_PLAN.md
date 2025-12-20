# E2I Causal Analytics: 18-Agent Implementation Plan

**Version**: 1.0
**Date**: December 18, 2024
**Project**: E2I Causal Analytics Platform v4.2
**Status**: In Progress

---

## Executive Summary

This document provides the complete implementation plan for building the 18-agent, 6-tier architecture of the E2I Causal Analytics platform using the Claude Code Development Framework.

**Implementation Approach**: Bottom-up, tier-by-tier implementation
**Total Agents**: 18 (7 Tier 0, 2 Tier 1, 3 Tier 2, 3 Tier 3, 2 Tier 4, 2 Tier 5)
**Estimated Timeline**: 6 phases
**Current Status**: 3/18 agents partially implemented

---

## Current State Assessment

### ✅ Completed (Partial)

| Agent | Tier | Status | Components |
|-------|------|--------|------------|
| orchestrator | 1 | Partial | classifier/, router.py, router_v42.py, tools/ |
| tool_composer | 1 | Partial | composer.py, decomposer.py, executor.py |
| experiment_designer | 3 | Partial | tools/simulate_intervention_tool.py, validate_twin_fidelity_tool.py |

### ❌ Not Started (15 agents)

**Tier 0 (7 agents)**: scope_definer, data_preparer, model_selector, model_trainer, feature_analyzer, model_deployer, observability_connector

**Tier 2 (3 agents)**: causal_impact, gap_analyzer, heterogeneous_optimizer

**Tier 3 (2 agents)**: drift_monitor, health_score

**Tier 4 (2 agents)**: prediction_synthesizer, resource_optimizer

**Tier 5 (2 agents)**: explainer, feedback_learner

---

## Framework Resources Available

### Specialist Documentation
Location: `.claude/specialists/`

- **Tier 0**: `.claude/specialists/Agent_Specialists_Tier 0/` (7 CLAUDE.md files)
- **Tier 1-5**: `.claude/specialists/Agent_Specialists_Tiers 1-5/` (11 .md files)
- **System Specialists**: `.claude/specialists/system/` (NLP, Causal, RAG, API, Frontend, Database, Testing, DevOps)
- **ML Foundation**: `.claude/specialists/ml_foundation/`
- **MLOps Integration**: `.claude/specialists/MLOps_Integration/`
- **Agent Index**: `AGENT-INDEX-V4.md`, `SPECIALIST-INDEX-V4.md`

### Integration Contracts
Location: `.claude/contracts/`

- `base-contract.md` - Base contract patterns
- `tier0-contracts.md` - ML Foundation agent contracts
- `orchestrator-contracts.md` - Orchestrator-specific contracts
- `tier2-contracts.md` - Causal inference contracts
- `tier3-contracts.md` - Monitoring agent contracts
- `tier4-contracts.md` - ML prediction contracts
- `tier5-contracts.md` - Self-improvement contracts
- `orchestrator-dispatch.yaml` - Dispatch format
- `agent-handoff.yaml` - Response format
- `inter-agent.yaml` - Agent-to-agent communication
- `integration-contracts.md` - Cross-tier integration
- `data-contracts.md` - Data schema contracts

### Context Files
Location: `.claude/context/`

- `brand-context.md` - Brand-specific information (Kisqali, Fabhalta, Remibrutinib)
- `kpi-dictionary.md` - 46+ KPI definitions
- `experiment-history.md` - Historical experiment outcomes
- `mlops-tools.md` - MLOps stack configuration (MLflow, Opik, Great Expectations, etc.)

### Product Requirements
Location: `product-development/current-feature/PRD/`

- `product-features-specifications.md` - Feature specifications
- `technical-architecture-high-level.md` - System architecture
- `success-metrics-kpis.md` - Success criteria
- Other PRD sections (sharded for efficiency)

---

## Implementation Strategy

### Bottom-Up Tier Implementation

**Rationale**:
1. Tier 0 provides foundation data/models consumed by upper tiers
2. Tier 1 orchestrator needs all agents available for routing
3. Dependencies flow upward (Tier 0 → Tier 5)
4. Enables incremental integration testing

### Dependency Flow

```
Tier 0 (ML Foundation)
    ↓ (provides: clean data, trained models, baselines)
Tier 1 (Orchestration)
    ↓ (routes queries to)
Tier 2 (Causal Analytics) + Tier 3 (Monitoring)
    ↓ (provides: causal estimates, health metrics)
Tier 4 (ML Predictions)
    ↓ (provides: predictions for explanation)
Tie2r 5 (Self-Improvement)
```

### Cross-Tier Dependencies

| Tier 0 Agent | Provides To | What |
|--------------|-------------|------|
| data_preparer | drift_monitor | Data quality baselines |
| model_trainer | prediction_synthesizer | Trained model artifacts |
| feature_analyzer | causal_impact | Feature relationship insights |
| model_deployer | prediction_synthesizer | Model serving endpoints |
| observability_connector | health_score | System metrics |

---

## Phase 1: Tier 0 - ML Foundation (7 Agents)

**Priority**: HIGHEST (Foundation for all other tiers)

### Agents to Implement

#### 1. data_preparer ⭐ START HERE
**Specialist**: `.claude/specialists/Agent_Specialists_Tier 0/data_preparer/CLAUDE.md`
**Contract**: `.claude/contracts/tier0-contracts.md`
**Type**: Standard (No LLM)
**Latency**: <60s
**Critical**: YES (QC gate blocks training)

**Responsibilities**:
- Data loading and validation
- Missing value handling
- Feature engineering
- QC gate enforcement (blocks if QC fails)
- Integration with Great Expectations

**Outputs**:
- Validated datasets
- QC reports
- Data quality baselines for drift_monitor

**Dependencies**:
- Great Expectations (data validation)
- Feast (feature store)

---

#### 2. scope_definer
**Specialist**: `.claude/specialists/Agent_Specialists_Tier 0/scope_definer/CLAUDE.md`
**Contract**: `.claude/contracts/tier0-contracts.md`
**Type**: Standard (No LLM)
**Latency**: <5s
**Critical**: YES (defines ML pipeline scope)

**Responsibilities**:
- Translate business questions to ML scopes
- Define prediction targets
- Identify relevant data sources
- Set ML problem type (classification, regression, etc.)

**Outputs**:
- ML scope definitions
- Problem type classification
- Data source recommendations

---

#### 3. model_selector
**Specialist**: `.claude/specialists/Agent_Specialists_Tier 0/model_selector/CLAUDE.md`
**Contract**: `.claude/contracts/tier0-contracts.md`
**Type**: Standard (No LLM)
**Latency**: <120s
**Critical**: YES (selects appropriate algorithm)

**Responsibilities**:
- Algorithm selection based on data characteristics
- Model family recommendation (tree, linear, ensemble)
- Hyperparameter search space definition

**Outputs**:
- Selected model family
- Recommended algorithms
- Hyperparameter search space

**Dependencies**:
- Optuna (hyperparameter optimization)

---

#### 4. model_trainer
**Specialist**: `.claude/specialists/Agent_Specialists_Tier 0/model_trainer/CLAUDE.md`
**Contract**: `.claude/contracts/tier0-contracts.md`
**Type**: Standard (No LLM)
**Latency**: Variable (depends on model complexity)
**Critical**: YES (trains models)

**Responsibilities**:
- Model training with cross-validation
- Hyperparameter tuning with Optuna
- MLflow experiment tracking
- Model performance evaluation

**Outputs**:
- Trained model artifacts
- Performance metrics
- MLflow run IDs

**Dependencies**:
- MLflow (experiment tracking, model registry)
- Optuna (hyperparameter tuning)
- scikit-learn, EconML, DoWhy

---

#### 5. feature_analyzer
**Specialist**: `.claude/specialists/Agent_Specialists_Tier 0/feature_analyzer/CLAUDE.md`
**Contract**: `.claude/contracts/tier0-contracts.md`
**Type**: Hybrid (Computation + Sonnet for interpretation)
**Latency**: <120s
**Critical**: NO

**Responsibilities**:
- Feature importance calculation (SHAP)
- Feature interaction detection
- Feature selection recommendations
- Natural language interpretation (via Sonnet)

**Outputs**:
- Feature importance rankings
- Feature interaction insights
- Recommendations for causal_impact agent

**Dependencies**:
- SHAP (model interpretability)
- Claude Sonnet (interpretation)

---

#### 6. model_deployer
**Specialist**: `.claude/specialists/Agent_Specialists_Tier 0/model_deployer/CLAUDE.md`
**Contract**: `.claude/contracts/tier0-contracts.md`
**Type**: Standard (No LLM)
**Latency**: <30s
**Critical**: NO (deployment is async)

**Responsibilities**:
- Model deployment to serving infrastructure
- Version management
- Endpoint creation
- Health check setup

**Outputs**:
- Deployed model endpoints
- Version metadata
- Health check URLs

**Dependencies**:
- BentoML (model serving)
- MLflow (model registry)

---

#### 7. observability_connector
**Specialist**: `.claude/specialists/Agent_Specialists_Tier 0/observability_connector/CLAUDE.md`
**Contract**: `.claude/contracts/tier0-contracts.md`
**Type**: Standard Async (No LLM)
**Latency**: <100ms
**Critical**: NO (cross-cutting, non-blocking)

**Responsibilities**:
- Metrics collection from all agents
- Integration with Opik (LLM observability)
- Span tracking for distributed tracing
- Health metrics aggregation

**Outputs**:
- Metrics for health_score agent
- Distributed traces
- Performance analytics

**Dependencies**:
- Opik (observability platform)
- OpenTelemetry (tracing)

---

### Phase 1 Implementation Order

```
1. data_preparer      (QC gate critical, blocks training)
   ↓
2. scope_definer      (defines what to train)
   ↓
3. model_selector     (chooses algorithm)
   ↓
4. model_trainer      (trains models)
   ↓
5. feature_analyzer   (interprets features, hybrid agent)
   ↓
6. model_deployer     (deploys to production)
   ↓
7. observability_connector (cross-cutting monitoring)
```

### Phase 1 Success Criteria

- ✅ All 7 agents have LangGraph implementations
- ✅ QC gate in data_preparer blocks bad data
- ✅ MLflow tracking works for all experiments
- ✅ Integration tests pass for Tier 0 pipeline
- ✅ Contracts validated for all Tier 0 agents

---

## Phase 2: Tier 1 - Orchestration (2 Agents)

**Priority**: HIGH (Needed to route queries)

### Agents to Complete

#### 1. orchestrator (COMPLETE PARTIAL)
**Specialist**: `.claude/specialists/Agent_Specialists_Tiers 1-5/orchestrator-agent.md`
**Contract**: `.claude/contracts/orchestrator-contracts.md`
**Type**: Standard (Fast) - Haiku/Sonnet
**Latency**: <2s (STRICT)
**Critical**: YES (routes all queries)

**Status**: Partial implementation exists
- ✅ Has: classifier/, router.py, router_v42.py, tools/
- ❌ Needs: Integration with all Tier 2-5 agents, full dispatch logic

**Responsibilities**:
- 4-stage query classification
- Agent routing and dispatch
- Response synthesis
- Error handling and fallback

**Dependencies**: All Tier 2-5 agents must be implemented first

---

#### 2. tool_composer (COMPLETE PARTIAL)
**Specialist**: `.claude/specialists/tool-composer.md`
**Contract**: `.claude/contracts/orchestrator-contracts.md`
**Type**: Standard - Sonnet
**Latency**: <15s for multi-faceted queries
**Critical**: YES (handles complex queries)

**Status**: Partial implementation exists
- ✅ Has: composer.py, decomposer.py, executor.py
- ❌ Needs: Full integration with orchestrator, dependency detection

**Responsibilities**:
- Multi-faceted query decomposition
- Tool orchestration (parallel vs sequential)
- Result synthesis

**Dependencies**: All Tier 2-5 agents for tool availability

---

### Phase 2 Implementation Order

```
1. Complete tool_composer (needed by orchestrator)
   ↓
2. Complete orchestrator (uses tool_composer)
   ↓
3. Test end-to-end query routing
```

### Phase 2 Success Criteria

- ✅ Orchestrator routes queries to all 18 agents correctly
- ✅ Tool_composer handles multi-faceted queries
- ✅ <2s latency for orchestrator (P95)
- ✅ 94%+ successful orchestration rate
- ✅ Graceful degradation on agent failures

---

## Phase 3: Tier 2 - Causal Analytics (3 Agents)

**Priority**: HIGH (Core product value)

### Agents to Implement

#### 1. causal_impact
**Specialist**: `.claude/specialists/Agent_Specialists_Tiers 1-5/causal-impact.md`
**Contract**: `.claude/contracts/tier2-contracts.md`
**Type**: Hybrid (DoWhy/EconML + Sonnet/Opus)
**Latency**: <120s
**Critical**: YES (core causal analysis)

**Responsibilities**:
- Causal effect estimation (ATE, CATE, ITE)
- 5-stage refutation testing
- Causal graph construction
- Natural language interpretation

**Methods**:
- Propensity Score Matching
- Inverse Propensity Weighting
- Doubly Robust Estimation
- Instrumental Variables
- Regression Discontinuity
- Difference-in-Differences

**Dependencies**:
- DoWhy, EconML
- Claude Sonnet/Opus (interpretation)
- feature_analyzer (Tier 0) for feature insights

---

#### 2. gap_analyzer
**Specialist**: `.claude/specialists/Agent_Specialists_Tiers 1-5/gap-analyzer.md`
**Contract**: `.claude/contracts/tier2-contracts.md`
**Type**: Standard - Sonnet
**Latency**: <20s
**Critical**: NO

**Responsibilities**:
- ROI opportunity identification
- Performance gap analysis
- Optimization recommendations

**Dependencies**:
- causal_impact (causal effect estimates)

---

#### 3. heterogeneous_optimizer
**Specialist**: `.claude/specialists/Agent_Specialists_Tiers 1-5/heterogeneous-optimizer.md`
**Contract**: `.claude/contracts/tier2-contracts.md`
**Type**: Standard - Sonnet
**Latency**: <150s
**Critical**: NO

**Responsibilities**:
- Treatment effect heterogeneity (CATE)
- Segment-level analysis
- Personalization recommendations

**Dependencies**:
- EconML (CATE estimators)
- causal_impact (base causal estimates)

---

### Phase 3 Implementation Order

```
1. causal_impact      (core causal engine)
   ↓
2. gap_analyzer       (uses causal estimates)
   ↓
3. heterogeneous_optimizer (segment-level CATE)
```

### Phase 3 Success Criteria

- ✅ 87%+ causal estimates pass all 5 refutation tests
- ✅ <30s for simple causal queries
- ✅ <120s for CATE analysis
- ✅ Full statistical reporting (p-values, CIs, effect sizes)

---

## Phase 4: Tier 3 - Monitoring (3 Agents)

**Priority**: MEDIUM (Monitoring and experimentation)

### Agents to Implement

#### 1. experiment_designer (COMPLETE PARTIAL)
**Specialist**: `.claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md`
**Contract**: `.claude/contracts/tier3-contracts.md`
**Type**: Hybrid (Computation + Sonnet/Opus)
**Latency**: <60s
**Critical**: NO

**Status**: Partial implementation exists
- ✅ Has: tools/simulate_intervention_tool.py, validate_twin_fidelity_tool.py
- ❌ Needs: Full graph implementation, A/B test design logic

**Responsibilities**:
- A/B test design
- Digital twin pre-screening
- Sample size calculation
- Experiment validity audit

---

#### 2. drift_monitor
**Specialist**: `.claude/specialists/Agent_Specialists_Tiers 1-5/drift-monitor.md`
**Contract**: `.claude/contracts/tier3-contracts.md`
**Type**: Standard (Fast) - Haiku/Sonnet
**Latency**: <10s
**Critical**: YES (detects model degradation)

**Responsibilities**:
- Data drift detection (statistical tests)
- Model drift detection (performance degradation)
- Alert triggering
- Drift reporting

**Dependencies**:
- data_preparer (Tier 0) for baselines
- model_trainer (Tier 0) for model performance history

---

#### 3. health_score
**Specialist**: `.claude/specialists/Agent_Specialists_Tiers 1-5/health-score.md`
**Contract**: `.claude/contracts/tier3-contracts.md`
**Type**: Standard (Fast) - Haiku
**Latency**: <5s
**Critical**: YES (SLA monitoring)

**Responsibilities**:
- System health metrics
- SLA compliance checking
- Agent performance monitoring
- Alert aggregation

**Dependencies**:
- observability_connector (Tier 0) for metrics

---

### Phase 4 Implementation Order

```
1. drift_monitor      (detects degradation)
   ↓
2. health_score       (monitors system health)
   ↓
3. experiment_designer (A/B test + digital twin)
```

### Phase 4 Success Criteria

- ✅ <24 hours drift detection from onset
- ✅ <5s health score calculation
- ✅ Digital twin fidelity >80%
- ✅ Automated alerts for drift/degradation

---

## Phase 5: Tier 4 - ML Predictions (2 Agents)

**Priority**: MEDIUM (Prediction and optimization)

### Agents to Implement

#### 1. prediction_synthesizer
**Specialist**: `.claude/specialists/Agent_Specialists_Tiers 1-5/prediction-synthesizer.md`
**Contract**: `.claude/contracts/tier4-contracts.md`
**Type**: Standard - Sonnet
**Latency**: <15s
**Critical**: NO

**Responsibilities**:
- Multi-model ensemble predictions
- Prediction aggregation
- Confidence interval calculation

**Dependencies**:
- model_trainer (Tier 0) for models
- model_deployer (Tier 0) for endpoints

---

#### 2. resource_optimizer
**Specialist**: `.claude/specialists/Agent_Specialists_Tiers 1-5/resource-optimizer.md`
**Contract**: `.claude/contracts/tier4-contracts.md`
**Type**: Standard - Sonnet
**Latency**: <20s
**Critical**: NO

**Responsibilities**:
- Resource allocation optimization
- Budget allocation recommendations
- ROI maximization

**Dependencies**:
- gap_analyzer (Tier 2) for opportunities
- causal_impact (Tier 2) for effect estimates

---

### Phase 5 Implementation Order

```
1. prediction_synthesizer (ensemble predictions)
   ↓
2. resource_optimizer (allocation optimization)
```

### Phase 5 Success Criteria

- ✅ >80% prediction accuracy for churn/conversion models
- ✅ <15s prediction latency (P95)
- ✅ Optimal resource allocation recommendations

---

## Phase 6: Tier 5 - Self-Improvement (2 Agents)

**Priority**: LOW (Enhancement features)

### Agents to Implement

#### 1. explainer
**Specialist**: `.claude/specialists/Agent_Specialists_Tiers 1-5/explainer.md`
**Contract**: `.claude/contracts/tier5-contracts.md`
**Type**: Deep - Opus/Sonnet
**Latency**: <45s
**Critical**: YES (regulatory compliance)

**Responsibilities**:
- Natural language explanations
- SHAP value interpretation
- Causal path explanation
- Visualization generation

**Dependencies**:
- SHAP API (real-time explanations)
- All agents (provides explanations for all outputs)

---

#### 2. feedback_learner
**Specialist**: `.claude/specialists/Agent_Specialists_Tiers 1-5/feedback-learner.md`
**Contract**: `.claude/contracts/tier5-contracts.md`
**Type**: Deep Async - Opus
**Latency**: No limit (async)
**Critical**: NO

**Responsibilities**:
- User feedback incorporation
- Model retraining triggers
- Pattern learning from interactions

**Dependencies**:
- All agents (learns from all interactions)
- model_trainer (Tier 0) for retraining

---

### Phase 6 Implementation Order

```
1. explainer          (SHAP + NL explanations)
   ↓
2. feedback_learner   (self-improvement loop)
```

### Phase 6 Success Criteria

- ✅ <300ms SHAP explanation latency (P95)
- ✅ Natural language summaries for all predictions
- ✅ Audit trail for regulatory compliance
- ✅ Feedback loop triggers retraining

---

## Per-Agent Implementation Workflow

### Standard Workflow for Each Agent

#### Step 1: Load Specialist Documentation
```bash
Read: .claude/specialists/Agent_Specialists_Tier X/{agent_name}/CLAUDE.md
```
- Understand agent responsibilities
- Review state definition (TypedDict)
- Study node flow and dependencies
- Note latency budgets and performance requirements

#### Step 2: Review Integration Contracts
```bash
Read: .claude/contracts/tierX-contracts.md
Read: .claude/contracts/integration-contracts.md
```
- Understand input/output schemas
- Review error handling requirements
- Check cross-tier dependencies
- Validate against base contracts

#### Step 3: Review Context (if needed)
```bash
Read: .claude/context/kpi-dictionary.md       # KPI-related agents
Read: .claude/context/brand-context.md        # Brand-specific logic
Read: .claude/context/mlops-tools.md          # Tier 0 agents
```

#### Step 4: Implement Agent Structure
```
src/agents/{agent_name}/
├── __init__.py
├── state.py              # TypedDict state definition
├── nodes/                # Node implementations
│   ├── __init__.py
│   ├── node_1.py        # Each node has async execute method
│   ├── node_2.py
│   └── ...
├── graph.py              # LangGraph assembly
├── tools/                # Agent-specific tools (if any)
│   ├── __init__.py
│   └── tool_1.py
└── prompts.py            # LLM prompts (Hybrid/Deep agents only)
```

#### Step 5: Implement Tests
```
tests/unit/test_agents/test_{agent_name}/
├── __init__.py
├── test_nodes.py         # Unit tests for each node
├── test_graph.py         # Integration tests for graph flow
├── test_performance.py   # Latency/performance tests
├── test_contracts.py     # Contract compliance tests
└── fixtures.py           # Test data fixtures
```

#### Step 6: Validate Contracts
- ✅ State schema matches contract
- ✅ Output format matches contract
- ✅ Error handling matches contract
- ✅ Latency meets tier budgets
- ✅ Integration with dependent agents works

---

## Agent Type Patterns

### Standard Pattern (13 agents)
**Agents**: Most Tier 0, 2-4 agents

```python
# Linear node flow, minimal/no LLM
[Input] → [Node 1] → [Node 2] → [Node 3] → [Output]
```

**Example**: data_preparer, gap_analyzer, drift_monitor

---

### Hybrid Pattern (3 agents)
**Agents**: causal_impact, experiment_designer, feature_analyzer

```python
# Computation nodes + Deep reasoning node
[Input] → [Computation Nodes] → [Deep Reasoning Node] → [Output]
```

**Example**: feature_analyzer
- Computation: SHAP calculation
- Deep reasoning: Sonnet interprets SHAP values

---

### Deep Pattern (2 agents)
**Agents**: explainer, feedback_learner

```python
# Extended reasoning throughout
[Input] → [Context Assembly] → [Deep Reasoning] → [Generation] → [Output]
```

**Example**: explainer
- Uses Opus for complex causal explanations

---

## Performance Budgets by Tier

| Tier | Latency Budget | Model Budget | Notes |
|------|----------------|--------------|-------|
| 0 | Variable | None (computation) | QC gate may block |
| 1 | <2s (strict) | Haiku preferred | Critical path |
| 2 | <120s | Sonnet primary | Causal computation |
| 3 | <60s | Mixed | Monitoring fast path |
| 4 | <20s | Sonnet | Prediction serving |
| 5 | No hard limit | Opus preferred | Deep reasoning |

---

## Error Handling Standards

All agents must implement:

### 1. Error Classification
```python
class ErrorCategory(Enum):
    VALIDATION = "validation"
    COMPUTATION = "computation"
    TIMEOUT = "timeout"
    DEPENDENCY = "dependency"
    QC_GATE = "qc_gate"          # Tier 0
    ML_PIPELINE = "ml_pipeline"   # Tier 0
```

### 2. Fallback Chains
- **Model degradation**: Opus → Sonnet → Haiku
- **Method degradation**: CausalForest → LinearDML → OLS
- **Graceful partial results**: Return what's available

### 3. Status Tracking
```python
status: Literal["pending", "processing", "completed", "failed", "blocked"]
```

### 4. QC Gate Enforcement (Tier 0)
```python
if qc_report.status == QCStatus.FAILED:
    raise QCGateBlockedError("Training blocked: QC failed")
```

---

## Testing Requirements

### All Agents Must Have

1. **Unit Tests** (for each node)
   - Test individual node logic
   - Mock dependencies
   - Test error handling

2. **Integration Tests** (for graph flow)
   - Test complete graph execution
   - Test state transitions
   - Test cross-node data flow

3. **Performance Tests** (for latency)
   - Measure P50, P95, P99 latencies
   - Ensure within tier budget
   - Test under load

4. **Contract Tests** (for compliance)
   - Validate input/output schemas
   - Test error response format
   - Verify integration contracts

### Tier 0 Additional Requirements

- **QC gate enforcement tests**
- **ML split compliance tests** (prevent data leakage)
- **MLOps tool integration tests** (MLflow, Opik, etc.)

---

## Success Criteria by Phase

### Phase 1 (Tier 0)
- ✅ All 7 agents implemented
- ✅ QC gate blocks bad data
- ✅ MLflow tracking works
- ✅ Integration tests pass
- ✅ Contracts validated

### Phase 2 (Tier 1)
- ✅ Orchestrator routes to all 18 agents
- ✅ Tool_composer handles complex queries
- ✅ <2s orchestrator latency (P95)
- ✅ 94%+ orchestration success rate

### Phase 3 (Tier 2)
- ✅ 87%+ causal estimates pass refutation
- ✅ <30s simple causal queries
- ✅ <120s CATE analysis
- ✅ Full statistical reporting

### Phase 4 (Tier 3)
- ✅ <24h drift detection
- ✅ <5s health score
- ✅ >80% twin fidelity
- ✅ Automated alerts

### Phase 5 (Tier 4)
- ✅ >80% prediction accuracy
- ✅ <15s prediction latency
- ✅ Optimal allocation recommendations

### Phase 6 (Tier 5)
- ✅ <300ms SHAP latency
- ✅ NL summaries for all predictions
- ✅ Audit trail complete
- ✅ Feedback loop operational

---

## Integration Testing Strategy

### Tier-by-Tier Integration

After each phase, run integration tests:

1. **Tier 0 Integration**: Test ML pipeline end-to-end
   ```
   scope_definer → data_preparer → model_selector → model_trainer
   → feature_analyzer → model_deployer
   ```

2. **Tier 0-1 Integration**: Test orchestrator routing to Tier 0
   ```
   orchestrator → data_preparer (via query)
   ```

3. **Tier 0-2 Integration**: Test causal agents using Tier 0 models
   ```
   causal_impact → model_trainer (uses trained models)
   ```

4. **Cross-Tier Integration**: Test full query flow
   ```
   User Query → orchestrator → tool_composer → causal_impact
   → feature_analyzer → explainer → Response
   ```

### End-to-End Integration Tests

Final validation with real user scenarios:

```
Test 1: "Why did Kisqali TRx drop 15% in Q3?"
Expected flow: orchestrator → causal_impact → explainer

Test 2: "Should we increase rep visits by 20% in Northeast?"
Expected flow: orchestrator → tool_composer →
  [experiment_designer, gap_analyzer, causal_impact] → explainer

Test 3: "Show me ROI for Q3 campaigns and predict Q4 impact"
Expected flow: orchestrator → tool_composer →
  [gap_analyzer, prediction_synthesizer] → explainer
```

---

## Development Best Practices

### Follow Framework Patterns

1. **Coding Patterns**: `.claude/.agent_docs/coding-patterns.md`
2. **Error Handling**: `.claude/.agent_docs/error-handling.md`
3. **ML Patterns**: `.claude/.agent_docs/ml-patterns.md`
4. **Testing Patterns**: `.claude/.agent_docs/testing-patterns.md`

### Code Review Checklist

All PRs use: `.claude/.agent_docs/code-review-checklist.md`

**Priority**: Security → Data Leakage → Correctness → Performance

### ML-Specific Requirements

When working on ML components:
1. **Prevent data leakage** (see ml-patterns.md)
2. **Track experiments** (MLflow for all training)
3. **Validate models** (performance thresholds)
4. **Test thoroughly** (leakage tests, performance tests)

---

## Technology Stack by Tier

### Tier 0 (ML Foundation)
- **ML**: scikit-learn, EconML, DoWhy
- **MLOps**: MLflow, Opik, Optuna, SHAP, BentoML, Great Expectations, Feast
- **Data**: pandas, numpy
- **Testing**: pytest

### Tier 1 (Orchestration)
- **Framework**: LangGraph, LangChain
- **LLM**: Claude (Anthropic API) - Haiku/Sonnet
- **NLP**: fastText, rapidfuzz

### Tier 2 (Causal Analytics)
- **Causal**: DoWhy, EconML, NetworkX
- **LLM**: Claude Sonnet/Opus
- **Stats**: scipy, statsmodels

### Tier 3 (Monitoring)
- **Monitoring**: Prometheus metrics, OpenTelemetry
- **LLM**: Haiku/Sonnet
- **Stats**: scipy (drift tests)

### Tier 4 (ML Predictions)
- **ML**: scikit-learn, ensemble methods
- **LLM**: Sonnet

### Tier 5 (Self-Improvement)
- **Interpretability**: SHAP
- **LLM**: Opus/Sonnet
- **Visualization**: matplotlib, plotly

### Shared Infrastructure
- **Database**: Supabase (PostgreSQL + pgvector)
- **Cache**: Redis
- **Graph**: FalkorDB (Neo4j-compatible)
- **API**: FastAPI, Pydantic
- **Logging**: loguru

---

## Risk Management

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| QC gate too strict | Blocks valid training | Configurable thresholds, manual override |
| Orchestrator timeout | User queries fail | Fallback to simpler agents, async processing |
| Causal refutation fails | No causal estimates | Fallback to correlational analysis, flag uncertainty |
| Model drift undetected | Poor predictions | Automated alerts, daily drift checks |
| SHAP timeout | No explanations | Model-agnostic fallback, async explanation generation |

### Implementation Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Tier dependencies break | Integration failures | Contract validation, integration tests |
| Performance degradation | Latency SLA violations | Performance testing, profiling, optimization |
| Data leakage in ML | Invalid models | Leakage audits, split validation tests |
| Contract drift | Agent incompatibility | Automated contract validation in CI/CD |

---

## Next Steps

### Immediate (Phase 1)
1. ✅ Document implementation plan (this document)
2. ⏳ Implement data_preparer agent (Tier 0)
3. Implement scope_definer agent
4. Implement model_selector agent
5. Implement model_trainer agent
6. Implement feature_analyzer agent
7. Implement model_deployer agent
8. Implement observability_connector agent
9. Run Tier 0 integration tests

### Short-Term (Phase 2-3)
10. Complete orchestrator agent
11. Complete tool_composer agent
12. Implement causal_impact agent
13. Implement gap_analyzer agent
14. Implement heterogeneous_optimizer agent
15. Run Tier 0-2 integration tests

### Medium-Term (Phase 4-6)
16. Complete experiment_designer agent
17. Implement drift_monitor, health_score
18. Implement prediction_synthesizer, resource_optimizer
19. Implement explainer, feedback_learner
20. Run full end-to-end integration tests

---

## Document Maintenance

**Update Frequency**: After each phase completion

**Change Log**:
- v1.0 (2024-12-18): Initial implementation plan created
- Future: Track agent completion status, phase milestones

**Ownership**: Product Management + Engineering Leadership

---

**END OF IMPLEMENTATION PLAN**
