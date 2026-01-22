# Observability Remediation Execution Plan

**Status**: Ready for Execution
**Created**: 2026-01-22
**Total Effort**: ~181 engineering hours
**Duration**: 8 weeks

---

## Execution Checklist

### Phase 1: Critical Fixes (Weeks 1-2)

#### Sprint 1.1: Fix BentoML-Opik Integration
**Priority: P0 | Effort: 10h | Assignee: ___________**

- [ ] **Task 1.1.1**: Add `log_model_prediction()` method to OpikConnector (2h)
  - File: `src/mlops/opik_connector.py`
  - Add async method with same pattern as `trace_agent()`
  - Include: model_name, input_data, output_data, latency_ms, metadata
  - Test: Unit test with mocked Opik client

- [ ] **Task 1.1.2**: Instrument BentoML classification template (1.5h)
  - File: `src/mlops/bentoml_templates/classification_service.py`
  - Add Opik tracing decorator to predict endpoint
  - Log: input features, prediction, probability, latency

- [ ] **Task 1.1.3**: Instrument BentoML regression template (1.5h)
  - File: `src/mlops/bentoml_templates/regression_service.py`
  - Add Opik tracing decorator to predict endpoint
  - Log: input features, prediction, confidence interval, latency

- [ ] **Task 1.1.4**: Instrument BentoML causal template (1.5h)
  - File: `src/mlops/bentoml_templates/causal_service.py`
  - Add Opik tracing decorator to predict endpoint
  - Log: treatment, outcome, CATE estimate, latency

- [ ] **Task 1.1.5**: Add tracing to BentoML client (1.5h)
  - File: `src/api/dependencies/bentoml_client.py`
  - Wrap HTTP calls with Opik spans
  - Propagate trace context headers

- [ ] **Task 1.1.6**: Add integration tests (2h)
  - File: `tests/integration/test_bentoml_opik.py`
  - Test: End-to-end prediction tracing
  - Verify: Spans appear in Opik

**Completion Criteria**:
- [ ] `log_prediction_to_opik()` no longer throws runtime error
- [ ] All 3 BentoML templates have Opik instrumentation
- [ ] Integration test passes

---

#### Sprint 1.2: Enable Distributed Tracing
**Priority: P0 | Effort: 14h | Assignee: ___________**

- [ ] **Task 1.2.1**: Configure OpenTelemetry TracerProvider (2h)
  - File: `src/api/main.py`
  - Add TracerProvider initialization at startup
  - Configure resource with service name, version, environment

- [ ] **Task 1.2.2**: Add ASGI instrumentation middleware (2h)
  - File: `src/api/main.py`
  - Install: `opentelemetry-instrumentation-asgi`
  - Wrap FastAPI app with OpenTelemetryMiddleware

- [ ] **Task 1.2.3**: Configure trace exporter (2h)
  - File: `config/observability.yaml`
  - Add OTLP exporter configuration
  - Support: Jaeger, OTLP, or console (for dev)
  ```yaml
  opentelemetry:
    enabled: true
    service_name: e2i-causal-analytics
    exporter:
      type: otlp
      endpoint: ${OTEL_EXPORTER_ENDPOINT}
  ```

- [ ] **Task 1.2.4**: Propagate trace context to Celery tasks (4h)
  - File: `src/workers/celery_app.py`
  - Create: `src/workers/tracing.py`
  - Extract trace context from task headers
  - Inject trace context when dispatching tasks
  - Test with sample task chain

- [ ] **Task 1.2.5**: Add database query tracing (4h)
  - File: `src/repositories/supabase_client.py`
  - Wrap query methods with trace spans
  - Log: query type, table, duration, row count
  - Add span attributes for slow query detection

**Completion Criteria**:
- [ ] API requests create trace spans automatically
- [ ] Trace ID visible in logs
- [ ] Celery tasks continue parent traces
- [ ] Database queries appear as child spans

---

#### Sprint 1.3: Integrate Error Tracking
**Priority: P0 | Effort: 6h | Assignee: ___________**

- [ ] **Task 1.3.1**: Initialize Sentry SDK (1h)
  - File: `src/api/main.py`
  - Add: `sentry_sdk.init()` at startup
  - Configure DSN from environment variable

- [ ] **Task 1.3.2**: Configure Sentry settings (1h)
  - File: `config/observability.yaml`
  ```yaml
  sentry:
    enabled: true
    dsn: ${SENTRY_DSN}
    environment: ${ENVIRONMENT}
    traces_sample_rate: 0.1
    release: ${VERSION}
  ```

- [ ] **Task 1.3.3**: Add Sentry exception handlers (2h)
  - File: `src/api/main.py`
  - Capture unhandled exceptions
  - Add user context (if available)
  - Add request context

- [ ] **Task 1.3.4**: Integrate Sentry with Celery (2h)
  - File: `src/workers/celery_app.py`
  - Install: `sentry-sdk[celery]`
  - Capture task failures
  - Add task context to error reports

**Completion Criteria**:
- [ ] Unhandled exceptions appear in Sentry dashboard
- [ ] Errors include stack traces and context
- [ ] Celery task failures captured

---

#### Sprint 1.4: Expose Prometheus Metrics
**Priority: P0 | Effort: 9h | Assignee: ___________**

- [ ] **Task 1.4.1**: Create metrics endpoint (2h)
  - File: `src/api/routes/metrics.py` (new)
  - Expose `/metrics` endpoint
  - Return Prometheus text format
  - Exclude from auth middleware

- [ ] **Task 1.4.2**: Create timing middleware (3h)
  - File: `src/api/middleware/timing.py` (new)
  - Track request latency histogram
  - Track request count by endpoint, method, status
  - Add request size gauge

- [ ] **Task 1.4.3**: Add latency percentiles (2h)
  - File: `src/mlops/bentoml_monitoring.py`
  - Configure histogram buckets for p50, p95, p99
  - Add summary metrics for percentiles

- [ ] **Task 1.4.4**: Create centralized metrics registry (2h)
  - File: `src/mlops/metrics_registry.py` (new)
  - Single registry for all metrics
  - Naming convention: `e2i_{component}_{metric}_{unit}`
  - Include standard labels: service, environment

**Completion Criteria**:
- [ ] `/metrics` endpoint returns Prometheus format
- [ ] API latency histogram available
- [ ] Can scrape with `curl localhost:8000/metrics`

---

### Phase 2: MLflow Agent Coverage (Weeks 3-4)

#### Sprint 2.1: Tier 2 Causal Agents
**Priority: P1 | Effort: 16h | Assignee: ___________**

- [ ] **Task 2.1.1**: Instrument Causal Impact Agent (6h)
  - File: `src/agents/causal_impact/agent.py`
  - Create: `src/agents/causal_impact/mlflow_logger.py`
  - Log parameters:
    - [ ] treatment_column
    - [ ] outcome_column
    - [ ] causal_method (dowhy/econml)
    - [ ] confounders list
  - Log metrics:
    - [ ] ate_estimate
    - [ ] ate_ci_lower, ate_ci_upper
    - [ ] e_value
    - [ ] refutation_passed (bool per test)
  - Log artifacts:
    - [ ] causal_dag.json
    - [ ] refutation_results.json
    - [ ] sensitivity_analysis.json

- [ ] **Task 2.1.2**: Instrument Gap Analyzer Agent (4h)
  - File: `src/agents/gap_analyzer/agent.py`
  - Create: `src/agents/gap_analyzer/mlflow_logger.py`
  - Log metrics:
    - [ ] gap_score
    - [ ] roi_opportunity
    - [ ] priority_ranking
    - [ ] actionability_score
  - Log artifacts:
    - [ ] gap_analysis_report.json
    - [ ] roi_projections.json

- [ ] **Task 2.1.3**: Instrument Heterogeneous Optimizer Agent (6h)
  - File: `src/agents/heterogeneous_optimizer/agent.py`
  - Create: `src/agents/heterogeneous_optimizer/mlflow_logger.py`
  - Log metrics:
    - [ ] cate_estimate (per segment)
    - [ ] segment_validity_score
    - [ ] treatment_coverage_pct
    - [ ] best_segment_id
  - Log artifacts:
    - [ ] cate_by_segment.json
    - [ ] segment_rankings.json

**Completion Criteria**:
- [ ] All Tier 2 agents log to MLflow
- [ ] Causal findings can be reproduced from logged artifacts
- [ ] Experiment hierarchy: `e2i/causal/{agent_name}/{run_id}`

---

#### Sprint 2.2: Tier 3 Monitoring Agents
**Priority: P1 | Effort: 12h | Assignee: ___________**

- [ ] **Task 2.2.1**: Instrument Experiment Designer Agent (5h)
  - File: `src/agents/experiment_designer/agent.py`
  - Log metrics:
    - [ ] statistical_power
    - [ ] sample_size_required
    - [ ] min_detectable_effect
    - [ ] alpha, beta
  - Log artifacts:
    - [ ] power_analysis.json
    - [ ] experimental_design.json
    - [ ] pre_registration.json

- [ ] **Task 2.2.2**: Instrument Drift Monitor Agent (4h)
  - File: `src/agents/drift_monitor/agent.py`
  - Log metrics:
    - [ ] psi_score
    - [ ] drift_detected (bool)
    - [ ] drift_severity (0-1)
    - [ ] features_drifted (count)
  - Log artifacts:
    - [ ] distribution_comparison.json
    - [ ] drift_report.json

- [ ] **Task 2.2.3**: Instrument Health Score Agent (3h)
  - File: `src/agents/health_score/agent.py`
  - Log metrics:
    - [ ] system_health_score
    - [ ] component_scores (per component)
    - [ ] degraded_components (count)
  - Log artifacts:
    - [ ] health_report.json

**Completion Criteria**:
- [ ] All Tier 3 agents log to MLflow
- [ ] Drift detection decisions are auditable
- [ ] Experiment designs can be reviewed historically

---

#### Sprint 2.3: Remaining Agents (Tier 0 gaps + Tier 4-5)
**Priority: P2 | Effort: 19h | Assignee: ___________**

- [ ] **Task 2.3.1**: Instrument Data Preparer Agent (4h)
  - Log: data_quality_score, leakage_detected, baseline_accuracy, sample_count
  - Artifacts: data_quality_report.json, leakage_analysis.json

- [ ] **Task 2.3.2**: Instrument Feature Analyzer Agent (4h)
  - Log: feature_importance (dict), shap_consistency_score
  - Artifacts: shap_summary.png, feature_importance.json

- [ ] **Task 2.3.3**: Instrument Prediction Synthesizer Agent (3h)
  - Log: ensemble_prediction, model_agreement, confidence_interval_width
  - Artifacts: individual_predictions.json

- [ ] **Task 2.3.4**: Instrument Resource Optimizer Agent (3h)
  - Log: optimization_objective, allocations_delta, expected_roi
  - Artifacts: allocation_plan.json

- [ ] **Task 2.3.5**: Instrument Explainer Agent (2h)
  - Log: explanation_clarity_score, insight_count, recommendation_count
  - Artifacts: explanation.json

- [ ] **Task 2.3.6**: Instrument Feedback Learner Agent (3h)
  - Log: patterns_detected, learning_recommendations, dspy_reward
  - Artifacts: learning_report.json, pattern_analysis.json

**Completion Criteria**:
- [ ] 100% of agents have MLflow instrumentation
- [ ] All ML decisions are reproducible

---

### Phase 3: Infrastructure Hardening (Weeks 5-6)

#### Sprint 3.1: Celery Observability
**Priority: P1 | Effort: 12h | Assignee: ___________**

- [ ] **Task 3.1.1**: Implement Celery event consumer (4h)
  - File: `src/workers/event_consumer.py` (new)
  - Subscribe to: task-sent, task-received, task-started, task-succeeded, task-failed
  - Emit Prometheus metrics per event type

- [ ] **Task 3.1.2**: Add task latency metrics (2h)
  - Histogram: `e2i_celery_task_duration_seconds`
  - Labels: task_name, queue, status

- [ ] **Task 3.1.3**: Add queue depth monitoring (3h)
  - File: `src/workers/monitoring.py` (new)
  - Gauge: `e2i_celery_queue_length`
  - Poll Redis for queue sizes
  - Alert threshold configuration

- [ ] **Task 3.1.4**: Propagate trace IDs to tasks (3h)
  - Inject: trace_id, span_id into task headers
  - Extract: in task pre-run signal
  - Create child span for task execution

**Completion Criteria**:
- [ ] Task execution metrics visible in Prometheus
- [ ] Queue depth alerts configured
- [ ] Tasks have trace context

---

#### Sprint 3.2: Database Observability
**Priority: P1 | Effort: 11h | Assignee: ___________**

- [ ] **Task 3.2.1**: Add query logging wrapper (4h)
  - File: `src/repositories/query_logger.py` (new)
  - Wrap all Supabase client methods
  - Log: query_type, table, duration_ms, row_count
  - Emit as trace spans

- [ ] **Task 3.2.2**: Implement slow query detection (3h)
  - Threshold: configurable (default 100ms)
  - Log slow queries with full context
  - Counter: `e2i_db_slow_queries_total`

- [ ] **Task 3.2.3**: Add connection pool metrics (2h)
  - Gauge: `e2i_db_pool_size`
  - Gauge: `e2i_db_pool_available`
  - Gauge: `e2i_db_pool_waiting`

- [ ] **Task 3.2.4**: Create slow query alerts (2h)
  - Alert rule in Prometheus
  - Threshold: >100ms for >1% of queries

**Completion Criteria**:
- [ ] All DB queries logged with duration
- [ ] Slow query alerts firing
- [ ] Connection pool health visible

---

#### Sprint 3.3: Structured Logging
**Priority: P1 | Effort: 12h | Assignee: ___________**

- [ ] **Task 3.3.1**: Create logging configuration module (3h)
  - File: `src/utils/logging.py` (new)
  - Use loguru with JSON formatter
  - Include: timestamp, level, logger, message, trace_id, request_id

- [ ] **Task 3.3.2**: Add context propagation (3h)
  - Use contextvars for trace_id, request_id
  - Auto-inject into all log records
  - Propagate through async calls

- [ ] **Task 3.3.3**: Update existing loggers (4h)
  - Run codemod to replace `logging.getLogger()`
  - Update 631 Python files
  - Verify log format consistency

- [ ] **Task 3.3.4**: Configure log sampling (2h)
  - Sample debug logs in production
  - Keep 100% of error/warning logs
  - Configurable via observability.yaml

**Completion Criteria**:
- [ ] All logs in JSON format
- [ ] Trace IDs in every log entry
- [ ] Can filter logs by trace_id

---

#### Sprint 3.4: Complete Agent Opik Coverage
**Priority: P1 | Effort: 9h | Assignee: ___________**

- [ ] **Task 3.4.1**: Add Opik to gap_analyzer (2h)
- [ ] **Task 3.4.2**: Add Opik to heterogeneous_optimizer (2h)
- [ ] **Task 3.4.3**: Add Opik to health_score (1h)
- [ ] **Task 3.4.4**: Add Opik to resource_optimizer (2h)
- [ ] **Task 3.4.5**: Add Opik to orchestrator main agent (2h)

**Completion Criteria**:
- [ ] 100% of agents have Opik tracing
- [ ] All LLM calls tracked with token usage

---

### Phase 4: Enhanced Observability (Weeks 7-8)

#### Sprint 4.1: Feature Store & Serving Enhancements
**Priority: P2 | Effort: 8h | Assignee: ___________**

- [ ] **Task 4.1.1**: Add feature retrieval latency tracking (3h)
  - File: `src/feature_store/retrieval.py`
  - Histogram: `e2i_feature_retrieval_seconds`
  - Labels: feature_view, entity_type

- [ ] **Task 4.1.2**: Add business context labels (3h)
  - Add to all BentoML templates: brand, segment, region
  - Pass through prediction requests
  - Include in Opik spans

- [ ] **Task 4.1.3**: Add cache metrics (2h)
  - Counter: `e2i_feature_cache_hits_total`
  - Counter: `e2i_feature_cache_misses_total`
  - Gauge: `e2i_feature_cache_hit_rate`

**Completion Criteria**:
- [ ] Feature retrieval latency visible
- [ ] Can filter metrics by brand/segment
- [ ] Cache efficiency measurable

---

#### Sprint 4.2: Opik Feedback Loop
**Priority: P2 | Effort: 11h | Assignee: ___________**

- [ ] **Task 4.2.1**: Integrate Opik feedback scoring (4h)
  - File: `src/mlops/opik_connector.py`
  - Add `log_feedback()` integration
  - Support: thumbs up/down, rating, comment

- [ ] **Task 4.2.2**: Add feedback collection endpoints (3h)
  - File: `src/api/routes/feedback.py` (new)
  - POST `/api/feedback/{trace_id}`
  - Link feedback to original trace

- [ ] **Task 4.2.3**: Connect feedback to GEPA optimization (4h)
  - File: `src/optimization/gepa/integration/opik_integration.py`
  - Use feedback scores as optimization signal
  - Track feedback-driven improvements

**Completion Criteria**:
- [ ] Users can provide feedback on agent responses
- [ ] Feedback linked to traces
- [ ] GEPA uses feedback for optimization

---

#### Sprint 4.3: Cost & SLO Monitoring
**Priority: P2 | Effort: 16h | Assignee: ___________**

- [ ] **Task 4.3.1**: Add per-agent cost tracking (6h)
  - File: `src/mlops/cost_tracker.py` (new)
  - Track: LLM tokens, compute time, API calls
  - Calculate cost per agent, per query
  - Emit as metrics and log to DB

- [ ] **Task 4.3.2**: Define SLOs (4h)
  - File: `config/slos.yaml` (new)
  ```yaml
  slos:
    tier_0:
      latency_p99: 5s
      error_rate: 0.1%
    tier_1:
      latency_p99: 10s
      error_rate: 0.5%
    # ... per tier
  ```

- [ ] **Task 4.3.3**: Implement SLO monitoring (6h)
  - File: `src/mlops/slo_monitor.py` (new)
  - Calculate error budget
  - Alert on SLO breach
  - Dashboard integration

**Completion Criteria**:
- [ ] Cost per agent visible
- [ ] SLOs defined for all tiers
- [ ] SLO breach alerts configured

---

#### Sprint 4.4: Log Aggregation
**Priority: P2 | Effort: 16h | Assignee: ___________**

- [ ] **Task 4.4.1**: Deploy log aggregation stack (8h)
  - Option A: ELK (Elasticsearch, Logstash, Kibana)
  - Option B: Grafana Loki
  - Option C: Datadog
  - Configure log shipping from containers

- [ ] **Task 4.4.2**: Configure log shipping (4h)
  - Add log driver to Docker containers
  - Configure retention policy
  - Set up log rotation

- [ ] **Task 4.4.3**: Create log dashboards (4h)
  - Error rate by service
  - Request flow visualization
  - Trace-to-log correlation

**Completion Criteria**:
- [ ] All logs centralized
- [ ] Can search logs by trace_id
- [ ] Error dashboards operational

---

## Progress Tracking

### Phase Summary

| Phase | Sprint | Status | Start Date | End Date | Notes |
|-------|--------|--------|------------|----------|-------|
| 1 | 1.1 BentoML-Opik | ⬜ Not Started | | | |
| 1 | 1.2 Distributed Tracing | ⬜ Not Started | | | |
| 1 | 1.3 Error Tracking | ⬜ Not Started | | | |
| 1 | 1.4 Prometheus Metrics | ⬜ Not Started | | | |
| 2 | 2.1 Tier 2 MLflow | ⬜ Not Started | | | |
| 2 | 2.2 Tier 3 MLflow | ⬜ Not Started | | | |
| 2 | 2.3 Remaining Agents | ⬜ Not Started | | | |
| 3 | 3.1 Celery Observability | ⬜ Not Started | | | |
| 3 | 3.2 Database Observability | ⬜ Not Started | | | |
| 3 | 3.3 Structured Logging | ⬜ Not Started | | | |
| 3 | 3.4 Agent Opik Coverage | ⬜ Not Started | | | |
| 4 | 4.1 Feature Store | ⬜ Not Started | | | |
| 4 | 4.2 Opik Feedback | ⬜ Not Started | | | |
| 4 | 4.3 Cost & SLO | ⬜ Not Started | | | |
| 4 | 4.4 Log Aggregation | ⬜ Not Started | | | |

### Health Score Progress

| Milestone | Target Score | Achieved | Date |
|-----------|--------------|----------|------|
| Baseline | 45 | ✅ | 2026-01-22 |
| Phase 1 Complete | 70 | ⬜ | |
| Phase 2 Complete | 85 | ⬜ | |
| Phase 3 Complete | 95 | ⬜ | |
| Phase 4 Complete | 100 | ⬜ | |

---

## Appendix: File Index

### New Files to Create

| File | Phase | Sprint |
|------|-------|--------|
| `src/api/routes/metrics.py` | 1 | 1.4 |
| `src/api/middleware/timing.py` | 1 | 1.4 |
| `src/api/middleware/tracing.py` | 1 | 1.2 |
| `src/mlops/metrics_registry.py` | 1 | 1.4 |
| `src/workers/tracing.py` | 1 | 1.2 |
| `src/workers/event_consumer.py` | 3 | 3.1 |
| `src/workers/monitoring.py` | 3 | 3.1 |
| `src/repositories/query_logger.py` | 3 | 3.2 |
| `src/utils/logging.py` | 3 | 3.3 |
| `src/api/routes/feedback.py` | 4 | 4.2 |
| `src/mlops/cost_tracker.py` | 4 | 4.3 |
| `src/mlops/slo_monitor.py` | 4 | 4.3 |
| `config/slos.yaml` | 4 | 4.3 |
| `tests/integration/test_bentoml_opik.py` | 1 | 1.1 |

### Files to Modify

| File | Phase | Sprint | Changes |
|------|-------|--------|---------|
| `src/api/main.py` | 1 | 1.2, 1.3 | OTEL init, Sentry init |
| `src/mlops/opik_connector.py` | 1 | 1.1 | Add log_model_prediction() |
| `src/mlops/bentoml_templates/*.py` | 1 | 1.1 | Add Opik tracing |
| `src/workers/celery_app.py` | 1,3 | 1.2,1.3 | Trace propagation, Sentry |
| `src/repositories/supabase_client.py` | 1 | 1.2 | Query tracing |
| `config/observability.yaml` | 1 | 1.2,1.3 | OTEL, Sentry config |
| `src/agents/causal_impact/agent.py` | 2 | 2.1 | MLflow logging |
| `src/agents/gap_analyzer/agent.py` | 2 | 2.1 | MLflow logging |
| `src/agents/heterogeneous_optimizer/agent.py` | 2 | 2.1 | MLflow logging |
| `src/agents/experiment_designer/agent.py` | 2 | 2.2 | MLflow logging |
| `src/agents/drift_monitor/agent.py` | 2 | 2.2 | MLflow logging |
| `src/agents/health_score/agent.py` | 2 | 2.2 | MLflow logging |
| (All remaining agents) | 2 | 2.3 | MLflow logging |
| `src/feature_store/retrieval.py` | 4 | 4.1 | Latency tracking |
