# Opik Implementation To-Do List

**Project**: E2I Causal Analytics
**Component**: LLM/Agent Observability
**Status**: Phase 2 Complete
**Last Updated**: 2025-12-21

---

## Overview

| Phase | Description | Est. Hours | Status |
|-------|-------------|------------|--------|
| Phase 1 | Core Infrastructure | 4-6 | ✅ Complete |
| Phase 2 | Agent Integration | 3-4 | ✅ Complete |
| Phase 3 | Production Features | 4-5 | Not Started |
| Testing | Updates + New | 3-4 | In Progress |
| **Total** | | **14-19** | |

---

## Phase 1: Core Infrastructure (4-6 hours)

### 1.1 OpikConnector Class
**File**: `src/mlops/opik_connector.py` (NEW)

- [ ] Create `OpikConnector` singleton class
- [ ] Implement `trace_agent()` async context manager
- [ ] Implement `trace_llm_call()` for LLM tracking
- [ ] Implement `log_metric()` for custom metrics
- [ ] Implement `log_feedback()` for trace feedback
- [ ] Add error handling with graceful degradation
- [ ] Add configuration loading from environment
- [ ] Write unit tests for OpikConnector

**Dependencies**: None
**Reference**: `.claude/specialists/MLOps_Integration/mlops_integration.md:269-453`

---

### 1.2 ObservabilitySpanRepository
**File**: `src/repositories/observability_span.py` (NEW)

- [ ] Create `ObservabilitySpanRepository` class extending `BaseRepository`
- [ ] Implement `insert_span()` for single span insert
- [ ] Implement `insert_spans_batch()` for batch inserts
- [ ] Implement `get_spans_by_time_window()` with window parsing
- [ ] Implement `get_spans_by_trace_id()` for trace reconstruction
- [ ] Implement `get_spans_by_agent()` for agent-specific queries
- [ ] Implement `get_latency_stats()` using `v_agent_latency_summary` view
- [ ] Implement `delete_old_spans()` for retention cleanup
- [ ] Write unit tests for repository methods

**Dependencies**: Phase 1.3 (Pydantic models)
**Reference**: `src/repositories/base.py`

---

### 1.3 Pydantic Models
**File**: `src/agents/ml_foundation/observability_connector/models.py` (NEW)

- [ ] Create `ObservabilitySpan` model matching database schema
- [ ] Create `SpanEvent` model for events within spans
- [ ] Create `LatencyStats` model for percentile data
- [ ] Create `QualityMetrics` model for aggregated metrics
- [ ] Create `TokenUsage` model for LLM tracking
- [ ] Add validation and default factories

**Dependencies**: None
**Reference**: `database/ml/mlops_tables.sql:425-479`

---

### 1.4 Update Exports
- [ ] Update `src/mlops/__init__.py` to export `OpikConnector`
- [ ] Update `src/repositories/__init__.py` to export `ObservabilitySpanRepository`

**Dependencies**: Phase 1.1, 1.2

---

### Phase 1 Validation Checkpoint
- [ ] All 65 existing unit tests pass
- [ ] New unit tests pass for OpikConnector
- [ ] New unit tests pass for ObservabilitySpanRepository
- [ ] Pydantic models validate correctly
- [ ] Imports work without circular dependencies

---

## Phase 2: Agent Integration (3-4 hours) ✅ COMPLETE

### 2.1 Update span_emitter.py
**File**: `src/agents/ml_foundation/observability_connector/nodes/span_emitter.py`

- [x] Import `OpikConnector` from `src/mlops/opik_connector`
- [x] Import `ObservabilitySpanRepository` from `src/repositories/`
- [x] Import `ObservabilitySpan` model
- [x] Replace mock Opik emission (lines 48-62) with real SDK calls
- [x] Replace mock database writes (lines 65-84) with repository calls
- [x] Add error handling for Opik failures (fallback to DB-only)
- [x] Add error handling for database failures
- [x] Update unit tests with proper mocking

**Dependencies**: Phase 1 complete
**Reference**: Current mock code in `span_emitter.py:48-84`

---

### 2.2 Update metrics_aggregator.py
**File**: `src/agents/ml_foundation/observability_connector/nodes/metrics_aggregator.py`

- [x] Import `ObservabilitySpanRepository`
- [x] Replace `_get_mock_spans()` (lines 111-172) with repository queries
- [x] Use `get_latency_stats()` for fast percentile queries
- [x] Add time window parsing (1h, 24h, 7d)
- [ ] Add caching for frequently accessed metrics (deferred to Phase 3)
- [x] Update unit tests

**Dependencies**: Phase 1.2 complete
**Reference**: SQL view `v_agent_latency_summary`

---

### 2.3 Update agent.py
**File**: `src/agents/ml_foundation/observability_connector/agent.py`

- [x] Import `OpikConnector` and `ObservabilitySpanRepository`
- [x] Add `opik_connector` attribute initialization in `__init__`
- [x] Add `span_repository` attribute with Supabase client
- [x] Pass dependencies to nodes via state or dependency injection
- [x] Update `_emit_span_async()` to use real connector

**Dependencies**: Phase 2.1, 2.2 complete

---

### Phase 2 Validation Checkpoint
- [x] Spans successfully emit to Opik dashboard (when API key configured)
- [x] Spans persist to `ml_observability_spans` table in Supabase
- [x] `get_quality_metrics` returns real data from database (falls back to mock when DB unavailable)
- [x] All unit tests pass (2687 passed, including 100 observability + 30 mlops tests)
- [x] Circular import issue fixed (lazy imports in span_emitter.py)
- [ ] Manual test: Run agent workflow and verify traces in Opik UI (requires deployment)

---

## Phase 3: Production Features (4-5 hours)

### 3.1 Batch Processing
**File**: `src/agents/ml_foundation/observability_connector/batch_processor.py` (NEW)

- [ ] Create `BatchProcessor` class
- [ ] Implement memory buffer (max 100 spans OR 5 seconds)
- [ ] Implement `add_span()` to buffer
- [ ] Implement `flush()` to emit batch
- [ ] Handle partial failures gracefully
- [ ] Track batch metrics (size, duration, success rate)
- [ ] Add background task for periodic flush
- [ ] Write unit tests

**Dependencies**: Phase 2 complete

---

### 3.2 Circuit Breaker
**File**: `src/mlops/opik_connector.py` (UPDATE)

- [ ] Add `CircuitBreaker` class or use existing library
- [ ] Track consecutive failures (threshold: 5)
- [ ] Implement CLOSED → OPEN transition
- [ ] Implement HALF-OPEN state after 30 seconds
- [ ] Fall back to database-only logging when circuit open
- [ ] Add metrics for circuit state changes
- [ ] Write unit tests for circuit breaker logic

**Dependencies**: Phase 2 complete

---

### 3.3 Metrics Caching
**File**: `src/agents/ml_foundation/observability_connector/cache.py` (NEW)

- [ ] Create `MetricsCache` class
- [ ] Support Redis backend (primary)
- [ ] Support in-memory fallback
- [ ] Implement TTL: 60s for "1h" window, 300s for "24h"
- [ ] Implement cache key: `obs_metrics:{window}:{agent}`
- [ ] Implement cache invalidation on new span insertion
- [ ] Write unit tests

**Dependencies**: Phase 2 complete

---

### 3.4 Configuration File
**File**: `config/observability.yaml` (NEW)

- [ ] Create YAML configuration file
- [ ] Add Opik settings (enabled, project, workspace, API key env)
- [ ] Add sampling settings (default_rate, production_rate, always_sample_errors)
- [ ] Add batching settings (enabled, max_size, max_wait)
- [ ] Add circuit breaker settings (threshold, timeout)
- [ ] Add retention settings (ttl_days, cleanup_batch_size)
- [ ] Update `OpikConnector` to load from config file

**Dependencies**: None (can be done in parallel)

---

### 3.5 Self-Monitoring
**File**: `src/agents/ml_foundation/observability_connector/self_monitor.py` (NEW)

- [ ] Track span emission latency
- [ ] Track Opik API response times
- [ ] Track database write latency
- [ ] Emit health spans every 60 seconds
- [ ] Add alert thresholds for degraded performance
- [ ] Write unit tests

**Dependencies**: Phase 3.1, 3.2 complete

---

### Phase 3 Validation Checkpoint
- [ ] Batch processing handles 100+ spans/second
- [ ] Circuit breaker trips after 5 consecutive failures
- [ ] Circuit breaker recovers after 30 seconds
- [ ] Metrics caching reduces query latency by 80%
- [ ] Configuration file loads correctly
- [ ] Self-monitoring emits health spans

---

## Testing (3-4 hours)

### Unit Tests
**Location**: `tests/unit/test_agents/test_ml_foundation/test_observability_connector/`

- [ ] Add tests for `OpikConnector` (mocked SDK)
- [ ] Add tests for `ObservabilitySpanRepository` (mocked DB)
- [ ] Add tests for `BatchProcessor`
- [ ] Add tests for `CircuitBreaker`
- [ ] Add tests for `MetricsCache`
- [ ] Update existing tests with new dependencies

---

### Integration Tests
**Location**: `tests/integration/test_observability_integration.py` (NEW)

- [ ] Test end-to-end span emission to Opik
- [ ] Test database write and query round-trip
- [ ] Test metrics computation from real data
- [ ] Test batch processing under load
- [ ] Test circuit breaker behavior
- [ ] Test cross-agent context propagation

---

### Load Tests
**Location**: `tests/load/test_observability_load.py` (NEW)

- [ ] Test 100 concurrent span emissions
- [ ] Test 1000 spans in batch
- [ ] Test metrics query under load
- [ ] Test recovery from Opik outage

---

## Final Validation

### Contract Compliance
- [ ] Run contract validation script
- [ ] Achieve 100% compliance (currently 60%)
- [ ] Update `CONTRACT_VALIDATION.md` with results

### Documentation
- [ ] Update `.claude/specialists/ml_foundation/observability_connector.md` if needed
- [ ] Update `.claude/context/implementation-status.md`
- [ ] Update `README.md` with Opik setup instructions

### Production Readiness
- [ ] All tests pass in CI
- [ ] Load tests pass (100 concurrent spans)
- [ ] Opik dashboard shows traces correctly
- [ ] Database contains persisted spans
- [ ] Configuration validated

---

## Environment Setup Checklist

Before starting implementation:

- [ ] Verify `opik>=0.2.0` is in `requirements.txt`
- [ ] Set `OPIK_API_KEY` environment variable
- [ ] Verify `ml_observability_spans` table exists in Supabase
- [ ] Verify `v_agent_latency_summary` view exists
- [ ] Test Opik connectivity: `opik.Opik().health_check()`

---

## Quick Reference

### Key Files

| File | Purpose |
|------|---------|
| `src/mlops/opik_connector.py` | Opik SDK wrapper (CREATE) |
| `src/repositories/observability_span.py` | Span repository (CREATE) |
| `src/agents/ml_foundation/observability_connector/models.py` | Pydantic models (CREATE) |
| `src/agents/ml_foundation/observability_connector/nodes/span_emitter.py` | Replace mocks (UPDATE) |
| `src/agents/ml_foundation/observability_connector/nodes/metrics_aggregator.py` | Replace mocks (UPDATE) |
| `src/agents/ml_foundation/observability_connector/agent.py` | Add dependencies (UPDATE) |
| `config/observability.yaml` | Configuration (CREATE) |

### Commands

```bash
# Run unit tests
pytest tests/unit/test_agents/test_ml_foundation/test_observability_connector/ -v

# Run integration tests
pytest tests/integration/test_observability_integration.py -v

# Validate contract compliance
python scripts/validate_observability_contract.py

# Check Opik connectivity
python -c "from opik import Opik; print(Opik().health_check())"
```

---

## Notes

- Priority: HIGH → Phases 1 & 2 are blocking for proper telemetry
- Phase 3 features are MEDIUM priority (production hardening)
- Update `docs/MLFLOW_INTEGRATION_TODO.md` to remove Opik deprecation note
- Consider creating a PRP for tracking this work: `.claude/PRPs/features/active/opik-integration.md`
