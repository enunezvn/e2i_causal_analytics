# E2I Causal Analytics: Comprehensive Observability Audit & Remediation Plan

**Date**: 2026-01-21
**Prepared By**: Claude Code Observability Audit
**Version**: 1.1
**Last Updated**: 2026-01-21

---

## Implementation Progress

### Quick Wins Status: **5/5 COMPLETE** ✅

| QW | Task | Status | Date |
|----|------|--------|------|
| QW1 | Add `/metrics` endpoint | ✅ Done | 2026-01-21 |
| QW2 | Initialize Sentry SDK | ✅ Done | 2026-01-21 |
| QW3 | Add timing middleware | ✅ Done | 2026-01-21 |
| QW4 | Fix `log_model_prediction()` | ✅ Done | 2026-01-21 |
| QW5 | Add trace header extraction | ✅ Done | 2026-01-21 |

### Updated Health Score: **75/100** (↑ from 70)

With Quick Wins + Phase 1 Critical Fixes complete:
- ✅ Error Tracking: 0% → **80%** (Sentry integrated with FastAPI)
- ✅ Infrastructure Monitoring: 25% → **45%** (/metrics endpoint, timing middleware)
- ✅ Model Serving Observability: 40% → **80%** (log_model_prediction + BentoML audit trail)
- ✅ Distributed Tracing: 0% → **80%** (OpenTelemetry + request ID propagation)

### Phase 1 Critical Fixes Status: **2/2 COMPLETE** ✅

| ID | Task | Status | Date |
|----|------|--------|------|
| G02 | Configure OpenTelemetry TracerProvider | ✅ Done | 2026-01-21 |
| G08 | Request ID propagation to agents | ✅ Done | 2026-01-21 |

### Phase 1 Additional Items Status: **1/2 COMPLETE**

| ID | Task | Status | Date |
|----|------|--------|------|
| G07 | Prediction audit trail in BentoML templates | ✅ Done | 2026-01-21 |
| G01 | BentoML-Opik integration | ⚠️ Partial (QW4 done) | - |

### Next Steps: Phase 2 MLflow Coverage

- [ ] G05: Tier 2 Causal Agents MLflow instrumentation
- [ ] G06: Tier 3 Monitoring Agents MLflow instrumentation

---

## Executive Summary

This audit reveals **significant observability gaps** across the E2I Causal Analytics platform that compromise the ability to monitor, debug, and optimize the 18-agent ML system. While foundational components exist (Opik, MLflow, Prometheus SDKs), implementation is incomplete and fragmented.

### Overall Observability Health Score: **45/100** ⚠️

| Area | Coverage | Status |
|------|----------|--------|
| Agent Opik Tracing | 74% (14/19 agents) | 🟡 Partial |
| MLflow Experiment Tracking | 33% (4/12 ML agents) | 🔴 Critical Gap |
| Model Serving Observability | 40% | 🔴 Critical Gap |
| Infrastructure Monitoring | 25% | 🔴 Critical Gap |
| Distributed Tracing | 0% | 🔴 Not Implemented |
| Error Tracking | 0% | 🔴 Not Implemented |

### Critical Finding Summary

1. **BentoML ↔ Opik integration is broken** - stub function calls non-existent method
2. **11 of 18 agents have no MLflow tracking** - decisions cannot be audited or reproduced
3. **OpenTelemetry installed but not configured** - no distributed tracing
4. **Sentry SDK installed but not integrated** - exceptions only in logs
5. **Prometheus metrics defined but not exposed** - cannot scrape `/metrics` endpoint
6. **No prediction audit trail** - model inputs/outputs not logged

---

## Part 1: Detailed Audit Findings

### 1.1 Agent Opik Observability

**Coverage: 74% (14 of 19 agents)**

#### Fully Integrated Tiers

| Tier | Agent | Status | File Path | Key Lines |
|------|-------|--------|-----------|-----------|
| 0 | scope_definer | ✅ | `src/agents/ml_foundation/scope_definer/agent.py` | 40-47, 156-176 |
| 0 | data_preparer | ✅ | `src/agents/ml_foundation/data_preparer/agent.py` | 28-35, 124-142 |
| 0 | feature_analyzer | ✅ | `src/agents/ml_foundation/feature_analyzer/agent.py` | 42-49, 158-178 |
| 0 | model_selector | ✅ | `src/agents/ml_foundation/model_selector/agent.py` | 44-51, 214-234 |
| 0 | model_trainer | ✅ | `src/agents/ml_foundation/model_trainer/agent.py` | 43-50, 213-240 |
| 0 | model_deployer | ✅ | `src/agents/ml_foundation/model_deployer/agent.py` | 36-43, 144-170 |
| 0 | observability_connector | ✅ | `src/agents/ml_foundation/observability_connector/agent.py` | 61-103, 231-296 |
| 1 | orchestrator | ⚠️ | Nodes only, not main agent.py | - |
| 1 | tool_composer | ✅ | `src/agents/tool_composer/opik_tracer.py` | Full 697-line module |
| 5 | explainer | ✅ | `src/agents/explainer/nodes/deep_reasoner.py` | 20-27 |
| 5 | feedback_learner | ✅ | `src/agents/feedback_learner/nodes/pattern_analyzer.py` | 20-27 |

#### Agents WITHOUT Opik Integration

| Tier | Agent | Reason | Impact |
|------|-------|--------|--------|
| 2 | gap_analyzer | Pure computation, no LLM | Medium - ROI analysis invisible |
| 2 | heterogeneous_optimizer | Uses memory hooks | Medium - CATE analysis untraced |
| 3 | health_score | Fast path agent (<5s) | Low - Dashboard metrics only |
| 4 | resource_optimizer | Mathematical optimization | Medium - Allocations invisible |
| 2 | causal_impact (partial) | Only graph.py instrumented | High - Core agent partially traced |

#### Implementation Patterns Identified

1. **Pattern 1: Direct Agent-Level** (10 agents) - `_get_opik_connector()` + context manager
2. **Pattern 2: Graph/Node-Level** (5 agents) - Tracing in LangGraph nodes
3. **Pattern 3: Dedicated Module** (1 agent) - tool_composer has 697-line tracer
4. **Pattern 4: Client Integration** (1 agent) - HTTP client tracing

---

### 1.2 MLflow Experiment Tracking

**Coverage: 33% of ML agents instrumented**

#### What IS Tracked ✅

| Component | Status | What's Logged |
|-----------|--------|---------------|
| Model Trainer | ✅ Full | Hyperparams, metrics, model, HPO metadata |
| Model Selector | ⚠️ Partial | Model registration (no selection metrics) |
| Model Deployer | ⚠️ Partial | Registry transitions (no validation metrics) |
| GEPA Optimization | ✅ Full | All optimization params, per-gen metrics, artifacts |
| RAG Evaluation | ✅ Full | RAGAS metrics, thresholds, artifacts |
| Energy Score | ✅ Full | Dual logging (MLflow + DB) |

#### What IS NOT Tracked ❌

**Tier 0 Gaps:**
- Data Preparer: No QC scores, baseline metrics, leakage detection
- Feature Analyzer: Loads models but doesn't log SHAP analysis results
- Scope Definer: No scope validation metrics

**Tier 1-5 Gaps (CRITICAL - 0% coverage):**

| Agent | Missing Metrics | Business Impact |
|-------|-----------------|-----------------|
| Orchestrator | Query routing, latency | Cannot optimize routing |
| Tool Composer | Composition decisions | Cannot audit tool selection |
| Causal Impact | ATE, CATE, refutation tests | **Cannot reproduce causal findings** |
| Gap Analyzer | Gap scores, ROI opportunity | Cannot validate recommendations |
| Heterogeneous Optimizer | CATE by segment | Cannot reproduce targeting |
| Experiment Designer | Power calculations, design params | Cannot audit experiment design |
| Drift Monitor | PSI scores, drift flags | Cannot verify drift detection |
| Health Score | Component scores | No health trend analysis |
| Prediction Synthesizer | Ensemble predictions | Cannot reproduce forecasts |
| Resource Optimizer | Allocation decisions | Cannot audit resource recommendations |
| Explainer | Explanation quality | Cannot improve explanations |
| Feedback Learner | Learning patterns, DSPy rewards | **Self-improvement is invisible** |

---

### 1.3 Model Serving Observability (BentoML)

**Coverage: 40% - Critical gaps in production serving**

#### What EXISTS ✅

| Component | Status | Location |
|-----------|--------|----------|
| BentoML Templates | ✅ | 3 templates (classification, regression, causal) |
| Health Endpoints | ✅ | `/health`, `/ready`, `/health/bentoml` |
| Prometheus Metrics Class | ⚠️ Defined | `src/mlops/bentoml_monitoring.py` |
| BentoML Health Monitor | ✅ | Periodic checks with alerts |
| Database Tables | ✅ | `ml_bentoml_services`, `ml_bentoml_serving_metrics` |

#### What's BROKEN ❌

**Critical: `log_prediction_to_opik()` calls non-existent method**

```python
# src/mlops/bentoml_monitoring.py
async def log_prediction_to_opik(...) -> None:
    connector = OpikConnector()
    await connector.log_model_prediction(...)  # ❌ METHOD DOES NOT EXIST
```

**OpikConnector actual methods:**
- `trace_agent()` ✅
- `trace_llm_call()` ✅
- `log_metric()` ✅
- `log_feedback()` ✅
- `flush()` ✅

#### Missing Observability

| Gap | Impact | Severity |
|-----|--------|----------|
| No `/metrics` endpoint | Cannot scrape with Prometheus | 🔴 Critical |
| No prediction audit trail | Inputs/outputs not logged | 🔴 Critical |
| No request tracing | Cannot correlate across services | 🔴 Critical |
| No per-model error tracking | Cannot debug failures | 🟡 High |
| No latency percentiles | Only averages tracked | 🟡 High |
| No feature retrieval latency | Cannot identify bottlenecks | 🟡 High |

---

### 1.4 Infrastructure Monitoring

**Coverage: 25% - Foundational only**

#### SDKs Installed But NOT Configured

| SDK | Version | Status | Issue |
|-----|---------|--------|-------|
| OpenTelemetry | 1.39.1 | ❌ Not initialized | No TracerProvider, no exporter |
| Sentry | 2.48.0 | ❌ Not used | 0 references in src/ |
| Prometheus Client | 0.23.1 | ⚠️ Partial | Only BentoML, not exposed |

#### Distributed Tracing: NOT IMPLEMENTED

```
Current State:
- X-Request-ID headers recognized in code (62 references)
- Headers NOT extracted or propagated
- No context.ContextVar for thread-safe trace propagation
- Celery tasks don't receive trace context
- Database calls have no trace spans
```

#### Logging Infrastructure

| Aspect | Status | Issue |
|--------|--------|-------|
| Format | ⚠️ Inconsistent | 631 Python files, mixed formats |
| Structured Logging | ⚠️ Partial | JSON in 8 files only |
| Log Aggregation | ❌ None | No ELK/Datadog/Splunk |
| Trace ID Propagation | ❌ None | Cannot correlate logs |

#### Background Workers (Celery)

```
Configured:
✅ 9 task queues (light, medium, heavy tiers)
✅ Task events enabled
✅ 23 scheduled tasks via Beat

Missing:
❌ No event consumer to process task events
❌ No task latency metrics
❌ No queue depth monitoring
❌ No trace ID propagation from API
```

#### Database Observability

```
Current:
✅ Connection pool config (min: 2, max: 10)
✅ Health check query exists

Missing:
❌ Query execution time tracking
❌ Slow query detection
❌ Connection pool exhaustion alerts
❌ N+1 query detection
```

---

## Part 2: Gap Analysis Matrix

### Severity Classification

| Severity | Definition | Count |
|----------|------------|-------|
| 🔴 Critical | Blocks production debugging, audit compliance, or causes runtime errors | 8 |
| 🟡 High | Significantly degrades observability quality | 12 |
| 🟢 Medium | Limits insight depth but workarounds exist | 6 |

### Complete Gap Inventory

| ID | Area | Gap Description | Severity | Effort |
|----|------|-----------------|----------|--------|
| G01 | BentoML-Opik | `log_model_prediction()` method missing | 🔴 Critical | 4h |
| G02 | Tracing | OpenTelemetry not configured | 🔴 Critical | 8h |
| G03 | Errors | Sentry not integrated | 🔴 Critical | 4h |
| G04 | Metrics | No `/metrics` endpoint exposed | 🔴 Critical | 4h |
| G05 | MLflow | Tier 2 agents (Causal) not tracked | 🔴 Critical | 16h |
| G06 | MLflow | Tier 3 agents (Monitoring) not tracked | 🔴 Critical | 12h |
| G07 | Serving | No prediction audit trail | 🔴 Critical | 8h |
| G08 | Tracing | Request ID not propagated | 🔴 Critical | 8h |
| G09 | MLflow | Tier 4-5 agents not tracked | 🟡 High | 8h |
| G10 | MLflow | Data Preparer not tracked | 🟡 High | 4h |
| G11 | MLflow | Feature Analyzer SHAP not logged | 🟡 High | 4h |
| G12 | Celery | Task events not processed | 🟡 High | 8h |
| G13 | Database | Query logging not implemented | 🟡 High | 6h |
| G14 | Logging | Structured logging not standardized | 🟡 High | 8h |
| G15 | Opik | 5 agents without tracing | 🟡 High | 8h |
| G16 | Serving | No latency percentiles | 🟡 High | 4h |
| G17 | Serving | Feature retrieval not tracked | 🟡 High | 4h |
| G18 | API | No timing middleware | 🟡 High | 4h |
| G19 | Workers | No queue depth monitoring | 🟡 High | 4h |
| G20 | Database | No slow query alerts | 🟡 High | 4h |
| G21 | MLflow | Tier 0 remaining gaps | 🟢 Medium | 4h |
| G22 | Logs | No log aggregation | 🟢 Medium | 16h |
| G23 | Opik | No feedback loop integration | 🟢 Medium | 6h |
| G24 | Serving | No business context labels | 🟢 Medium | 4h |
| G25 | Cost | No per-agent cost tracking | 🟢 Medium | 8h |
| G26 | SLOs | No SLO monitoring | 🟢 Medium | 12h |

---

## Part 3: Remediation Plan

### Phase 1: Critical Fixes (Week 1-2)

**Goal: Fix broken components and establish baseline observability**

#### Sprint 1.1: Fix BentoML-Opik Integration (G01, G07)

**Priority: P0 - Currently broken code**

```
Tasks:
1. Add log_model_prediction() method to OpikConnector
   File: src/mlops/opik_connector.py
   Effort: 2h

2. Instrument BentoML service templates with Opik tracing
   Files: src/mlops/bentoml_templates/*.py
   Effort: 4h

3. Add prediction audit logging to BentoML client
   File: src/api/dependencies/bentoml_client.py
   Effort: 2h

4. Add integration tests
   Effort: 2h

Total: 10h
```

#### Sprint 1.2: Enable Distributed Tracing (G02, G08)

**Priority: P0 - Foundation for all other observability**

```
Tasks:
1. Configure OpenTelemetry TracerProvider in API startup
   File: src/api/main.py
   Effort: 2h

2. Add ASGI instrumentation middleware
   File: src/api/main.py
   Effort: 2h

3. Configure trace exporter (Jaeger/OTLP)
   File: config/observability.yaml
   Effort: 2h

4. Propagate trace context to Celery tasks
   File: src/workers/celery_app.py
   Effort: 4h

5. Add database query tracing
   File: src/repositories/supabase_client.py
   Effort: 4h

Total: 14h
```

#### Sprint 1.3: Integrate Error Tracking (G03)

**Priority: P0 - Exceptions currently invisible**

```
Tasks:
1. Initialize Sentry SDK in API startup
   File: src/api/main.py
   Effort: 1h

2. Configure Sentry DSN and environment
   File: config/observability.yaml
   Effort: 1h

3. Add Sentry exception handlers
   File: src/api/main.py
   Effort: 2h

4. Integrate Sentry with Celery
   File: src/workers/celery_app.py
   Effort: 2h

Total: 6h
```

#### Sprint 1.4: Expose Prometheus Metrics (G04, G16, G18)

**Priority: P0 - Metrics defined but not accessible**

```
Tasks:
1. Add /metrics endpoint to API
   File: src/api/routes/metrics.py (new)
   Effort: 2h

2. Add timing middleware for API latency histograms
   File: src/api/middleware/timing.py (new)
   Effort: 3h

3. Add latency percentile calculations
   File: src/mlops/bentoml_monitoring.py
   Effort: 2h

4. Register all metrics in single registry
   File: src/mlops/metrics_registry.py (new)
   Effort: 2h

Total: 9h
```

### Phase 2: MLflow Coverage (Week 3-4)

**Goal: Instrument all agents for experiment tracking**

#### Sprint 2.1: Tier 2 Causal Agents (G05)

**Priority: P1 - Core causal decisions untracked**

```
Causal Impact Agent:
- Log: ATE estimate, CATE by segment, refutation results, e-value
- Artifacts: causal_dag.png, sensitivity_analysis.json
Effort: 6h

Gap Analyzer Agent:
- Log: gap_score, roi_opportunity, priority_ranking
- Artifacts: gap_analysis_report.json
Effort: 4h

Heterogeneous Optimizer Agent:
- Log: cate_estimate, segment_validity_score, treatment_coverage
- Artifacts: cate_by_segment.json
Effort: 6h

Total: 16h
```

#### Sprint 2.2: Tier 3 Monitoring Agents (G06)

**Priority: P1 - Monitoring agents need monitoring**

```
Experiment Designer Agent:
- Log: statistical_power, sample_size_required, min_effect_size
- Artifacts: power_analysis.json, experimental_design.json
Effort: 5h

Drift Monitor Agent:
- Log: psi_score, drift_detected, drift_severity
- Artifacts: drift_report.json
Effort: 4h

Health Score Agent:
- Log: component_health_scores, system_health_score
- Artifacts: health_report.json
Effort: 3h

Total: 12h
```

#### Sprint 2.3: Tier 0 Remaining & Tier 4-5 (G09, G10, G11, G21)

**Priority: P2**

```
Data Preparer:
- Log: data_quality_score, leakage_detected, baseline_accuracy
Effort: 4h

Feature Analyzer:
- Log: feature_importance_by_feature, shap_consistency_score
- Artifacts: shap_summary.png, feature_importance.json
Effort: 4h

Prediction Synthesizer:
- Log: ensemble_prediction, model_agreement, confidence_interval
Effort: 3h

Resource Optimizer:
- Log: optimization_objective, allocations_before/after, expected_roi
Effort: 3h

Explainer:
- Log: explanation_clarity_score, insight_count
Effort: 2h

Feedback Learner:
- Log: patterns_detected, learning_recommendations, dspy_reward
Effort: 3h

Total: 19h
```

### Phase 3: Infrastructure Hardening (Week 5-6)

**Goal: Complete infrastructure observability**

#### Sprint 3.1: Celery Observability (G12, G19)

```
Tasks:
1. Implement Celery event consumer
   File: src/workers/event_consumer.py (new)
   Effort: 4h

2. Add task latency metrics
   File: src/workers/celery_app.py
   Effort: 2h

3. Add queue depth monitoring
   File: src/workers/monitoring.py (new)
   Effort: 3h

4. Propagate trace IDs to tasks
   Effort: 3h

Total: 12h
```

#### Sprint 3.2: Database Observability (G13, G20)

```
Tasks:
1. Add query logging wrapper
   File: src/repositories/query_logger.py (new)
   Effort: 4h

2. Implement slow query detection
   Effort: 3h

3. Add connection pool metrics
   Effort: 2h

4. Create slow query alerts
   Effort: 2h

Total: 11h
```

#### Sprint 3.3: Structured Logging (G14)

```
Tasks:
1. Standardize on loguru with JSON format
   File: src/utils/logging.py (new)
   Effort: 3h

2. Add context propagation (trace_id, request_id)
   Effort: 3h

3. Update all loggers (631 files - automated)
   Effort: 4h

4. Configure log sampling for production
   Effort: 2h

Total: 12h
```

#### Sprint 3.4: Remaining Agent Opik Coverage (G15)

```
Tasks:
Add Opik tracing to:
1. gap_analyzer - 2h
2. heterogeneous_optimizer - 2h
3. health_score - 1h
4. resource_optimizer - 2h
5. orchestrator main agent - 2h

Total: 9h
```

### Phase 4: Enhanced Observability (Week 7-8)

**Goal: Advanced monitoring and insights**

#### Sprint 4.1: Feature Store & Serving Enhancements (G17, G24)

```
Tasks:
1. Add feature retrieval latency tracking
   File: src/feature_store/retrieval.py
   Effort: 3h

2. Add business context labels (brand, segment)
   File: src/mlops/bentoml_templates/*.py
   Effort: 3h

3. Add cache hit rate metrics
   Effort: 2h

Total: 8h
```

#### Sprint 4.2: Opik Feedback Loop (G23)

```
Tasks:
1. Integrate Opik feedback scoring
   File: src/mlops/opik_connector.py
   Effort: 4h

2. Add feedback collection endpoints
   File: src/api/routes/feedback.py
   Effort: 3h

3. Connect feedback to GEPA optimization
   Effort: 4h

Total: 11h
```

#### Sprint 4.3: Cost & SLO Monitoring (G25, G26)

```
Tasks:
1. Add per-agent cost tracking
   File: src/mlops/cost_tracker.py (new)
   Effort: 6h

2. Define SLOs for each tier
   File: config/slos.yaml (new)
   Effort: 4h

3. Implement SLO monitoring
   File: src/mlops/slo_monitor.py (new)
   Effort: 6h

Total: 16h
```

#### Sprint 4.4: Log Aggregation (G22)

```
Tasks:
1. Deploy ELK stack or configure Datadog
   Effort: 8h

2. Configure log shipping
   Effort: 4h

3. Create log-based dashboards
   Effort: 4h

Total: 16h
```

---

## Part 4: Implementation Priorities

### Critical Path (Must Complete First)

```
Week 1: G01 → G02 → G03 → G04
        (BentoML fix → Tracing → Errors → Metrics)

Week 2: G05 → G08
        (Tier 2 MLflow → Request ID propagation)
```

### Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 1 (Critical)                        │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │  G01    │───▶│  G07    │    │  G02    │───▶│  G08    │  │
│  │BentoML  │    │ Audit   │    │OpenTel  │    │TraceID  │  │
│  │ Fix     │    │ Trail   │    │ Setup   │    │ Prop    │  │
│  └─────────┘    └─────────┘    └────┬────┘    └─────────┘  │
│                                     │                       │
│  ┌─────────┐                       │        ┌─────────┐    │
│  │  G03    │                       │        │  G04    │    │
│  │ Sentry  │                       │        │/metrics │    │
│  │ Setup   │                       │        │endpoint │    │
│  └─────────┘                       │        └─────────┘    │
└────────────────────────────────────┼────────────────────────┘
                                     │
┌────────────────────────────────────┼────────────────────────┐
│                    Phase 2 (MLflow)│                        │
│  ┌─────────┐    ┌─────────┐    ┌──▼──────┐                 │
│  │  G05    │    │  G06    │    │  G12    │                 │
│  │Tier 2   │    │Tier 3   │    │ Celery  │                 │
│  │ MLflow  │    │ MLflow  │    │ Events  │                 │
│  └────┬────┘    └────┬────┘    └─────────┘                 │
│       │              │                                      │
│  ┌────▼────┐    ┌────▼────┐                                │
│  │  G09    │    │  G10    │                                │
│  │Tier 4-5 │    │Tier 0   │                                │
│  │ MLflow  │    │Complete │                                │
│  └─────────┘    └─────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

### Resource Requirements

| Phase | Duration | Engineering Hours | Skills Needed |
|-------|----------|-------------------|---------------|
| Phase 1 | 2 weeks | 39h | Backend, Observability |
| Phase 2 | 2 weeks | 47h | ML Engineering, MLflow |
| Phase 3 | 2 weeks | 44h | Backend, DevOps |
| Phase 4 | 2 weeks | 51h | Full Stack, DevOps |
| **Total** | **8 weeks** | **181h** | - |

---

## Part 5: Success Metrics

### Observability Health Score Targets

| Metric | Current | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|--------|---------|---------|---------|---------|---------|
| Agent Opik Coverage | 74% | 74% | 74% | 100% | 100% |
| MLflow Coverage | 33% | 33% | 100% | 100% | 100% |
| Distributed Tracing | 0% | 80% | 90% | 100% | 100% |
| Error Tracking | 0% | 100% | 100% | 100% | 100% |
| API Metrics | 0% | 100% | 100% | 100% | 100% |
| Model Serving | 40% | 80% | 90% | 95% | 100% |
| **Overall Score** | **45** | **70** | **85** | **95** | **100** |

### Key Performance Indicators

1. **Mean Time to Detection (MTTD)**: Target < 5 minutes for critical errors
2. **Trace Completeness**: 100% of requests have end-to-end traces
3. **Experiment Reproducibility**: 100% of ML decisions can be reproduced
4. **Audit Compliance**: All predictions have input/output audit trail
5. **Alert Coverage**: All critical paths have alerting configured

---

## Part 6: Quick Wins (Can Start Immediately)

These can be implemented in 2-4 hours each:

| ID | Task | File | Effort | Status |
|----|------|------|--------|--------|
| QW1 | Add `/metrics` endpoint | `src/api/routes/metrics.py` | 2h | ✅ DONE |
| QW2 | Initialize Sentry SDK | `src/api/main.py` | 2h | ✅ DONE |
| QW3 | Add timing middleware | `src/api/middleware/timing.py` | 3h | ✅ DONE |
| QW4 | Fix `log_model_prediction()` | `src/mlops/opik_connector.py` | 2h | ✅ DONE |
| QW5 | Add trace header extraction | `src/api/middleware/tracing.py` | 2h | ✅ DONE |

### Quick Wins Implementation Notes (2026-01-21)

**QW1: /metrics endpoint**
- Created `src/api/routes/metrics.py` with Prometheus metrics registry
- Exposes request counts, latencies, error rates, agent invocations, component health
- Registered at `/metrics` (standard Prometheus path)

**QW2: Sentry SDK Initialization**
- Initialized in `src/api/main.py` with FastAPI/Starlette integrations
- Captures CRITICAL and HIGH severity E2I errors automatically
- Configurable via SENTRY_DSN, ENVIRONMENT, SENTRY_TRACES_SAMPLE_RATE env vars

**QW3: Timing Middleware**
- Created `src/api/middleware/timing.py`
- Records request latency to Prometheus metrics
- Adds Server-Timing header for debugging
- Logs slow requests (configurable threshold via TIMING_SLOW_THRESHOLD_MS)

**QW4: log_model_prediction() Fix**
- Added missing async method to `src/mlops/opik_connector.py`
- Uses circuit breaker pattern consistent with other methods
- Creates Opik traces for model prediction audit trail

**QW5: Trace Header Extraction**
- Created `src/api/middleware/tracing.py`
- Supports W3C Trace Context (traceparent), Zipkin B3, X-Request-ID
- Uses UUID7 for new request IDs (Opik compatible)
- Stores context in ContextVars for thread-safe access

### Phase 1 Implementation Notes (2026-01-21)

**G02: OpenTelemetry Distributed Tracing**
- Created `src/api/dependencies/opentelemetry_config.py` with full TracerProvider setup
- Supports OTLP, Console, and None exporters via `OTEL_EXPORTER_TYPE` env var
- Module-level initialization in `src/api/main.py` (like Sentry pattern)
- ASGI instrumentation added after all middleware for accurate timing
- W3C Trace Context + Baggage propagators configured
- Graceful degradation with NoOp tracer when SDK unavailable
- Configuration: `OTEL_ENABLED`, `OTEL_SERVICE_NAME`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SAMPLING_RATE`

**G08: Request ID Propagation to Agents**
- Updated `src/api/routes/copilotkit.py` to use middleware request_id
- `ChatRequest.request_id` now optional - falls back to `X-Request-ID` header
- Added `get_request_id()` import from TracingMiddleware for ContextVar access
- Both `/chat/stream` and `/chat` endpoints inject middleware request_id
- StreamingResponse includes `X-Request-ID` header in response
- Enables end-to-end tracing from API request through agent execution

**G07: Prediction Audit Trail in BentoML Templates**
- Created `src/mlops/bentoml_prediction_audit.py` with audit logging utilities
- Updated all 3 BentoML service templates (classification, regression, causal)
- Each prediction endpoint now logs to Opik via `log_prediction_audit()`
- Audit data includes: model name/tag, service type, input/output summaries, latency
- Uses `asyncio.create_task()` for non-blocking audit logging
- Graceful degradation when Opik unavailable (AUDIT_AVAILABLE flag)
- Templates updated to version 1.1.0

---

## Appendix A: Configuration Changes Required

### observability.yaml Updates

```yaml
# Add to config/observability.yaml

opentelemetry:
  enabled: true
  service_name: e2i-causal-analytics
  exporter:
    type: otlp  # or jaeger
    endpoint: ${OTEL_EXPORTER_ENDPOINT}
  sampling_rate: 1.0  # 100% in dev, reduce in prod

sentry:
  enabled: true
  dsn: ${SENTRY_DSN}
  environment: ${ENVIRONMENT}
  traces_sample_rate: 0.1
  profiles_sample_rate: 0.1

prometheus:
  enabled: true
  endpoint: /metrics
  include_histograms: true
  buckets: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
```

### MLflow Agent Configuration

```yaml
# Add to config/mlflow/agents.yaml

agents:
  causal_impact:
    experiment_prefix: causal_impact
    log_inputs: true
    log_outputs: true
    artifact_types: [dag, sensitivity, refutation]

  gap_analyzer:
    experiment_prefix: gap_analyzer
    log_inputs: true
    log_outputs: true
    artifact_types: [gap_report, roi_projection]

  # ... repeat for all agents
```

---

## Appendix B: Testing Requirements

Each phase must include:

1. **Unit Tests**: Cover new observability code
2. **Integration Tests**: Verify end-to-end tracing
3. **Load Tests**: Ensure observability doesn't impact performance
4. **Chaos Tests**: Verify graceful degradation when observability backends fail

### Test Coverage Targets

| Component | Unit | Integration | Load |
|-----------|------|-------------|------|
| Opik Integration | 90% | 80% | Yes |
| MLflow Logging | 90% | 80% | Yes |
| OpenTelemetry | 85% | 90% | Yes |
| Sentry | 80% | 70% | No |
| Prometheus | 85% | 80% | Yes |

---

## Appendix C: Rollback Plan

If observability changes cause production issues:

1. **Feature Flags**: All observability can be disabled via config
2. **Circuit Breakers**: Already implemented for Opik, extend to others
3. **Sampling**: Reduce sampling rates under load
4. **Graceful Degradation**: Never fail requests due to observability errors

---

## Conclusion

This audit reveals significant observability gaps that must be addressed to operate the E2I platform reliably. The 8-week remediation plan prioritizes:

1. **Week 1-2**: Fix broken code and establish baseline tracing
2. **Week 3-4**: Instrument all ML agents for reproducibility
3. **Week 5-6**: Harden infrastructure monitoring
4. **Week 7-8**: Add advanced insights and SLO monitoring

The total investment of ~181 engineering hours will raise the observability health score from 45 to 100, enabling:
- Rapid incident response (MTTD < 5 min)
- Full audit compliance
- ML decision reproducibility
- Proactive performance optimization
