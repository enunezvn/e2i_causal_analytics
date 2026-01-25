# Tier 2 Causal Inference Agents - Enhancement Plan

**Created**: 2026-01-24
**Updated**: 2026-01-24 (All Phases Complete)
**Status**: ✅ COMPLETE
**Priority**: Non-blocking enhancements for production-ready agents

---

## Executive Summary

This plan addresses four non-blocking enhancements to the Tier 2 Causal Inference agents (causal_impact, gap_analyzer, heterogeneous_optimizer). The infrastructure for most enhancements is already in place but not activated in production code paths.

---

## Phase 1: Memory Contribution Hooks (Gap Analyzer)

**Complexity**: Low
**Dependencies**: Redis (port 6382), Supabase episodic memory tables
**Status**: ✅ COMPLETE (already implemented in codebase)

### Task 1.1: Integrate Memory Contribution in Graph ✅

**Files Modified**:
- `src/agents/gap_analyzer/nodes/formatter.py` - Lines 89-92, 116-148

**Implementation** (already in codebase):
- `FormatterNode._contribute_to_memory()` method calls memory hooks
- Async, non-blocking with graceful error handling
- Returns `{episodic_stored, working_cached}` counts

### Task 1.2: Add Context Retrieval to Gap Detector ✅

**Files Modified**:
- `src/agents/gap_analyzer/nodes/gap_detector.py` - `_get_memory_context()` method

**Implementation** (already in codebase):
- Uses `get_gap_analyzer_memory_hooks()` for context retrieval
- Retrieves from both working memory (Redis) and episodic memory (Supabase)
- Integrates historical analyses into current gap detection

### Task 1.3: Configure Memory Backend Connections ✅

**Files Verified**:
- `src/memory/working_memory.py` - `RedisWorkingMemory` class with port 6382
- `src/memory/episodic_memory.py` - Supabase pgvector integration

**Implementation** (already in codebase):
- Redis URL from env: `REDIS_URL` default `redis://localhost:6382`
- 24h TTL for working memory cache
- Graceful fallback on connection failures (returns empty results)

### Acceptance Criteria

- [x] Gap analysis results cached in Redis (24h TTL)
- [x] Gap analysis summaries stored in Supabase episodic memory
- [x] Historical ROI data retrievable for similar analyses
- [x] Memory contribution adds <100ms overhead
- [x] Graceful degradation if backends unavailable

---

## Phase 2: DSPy Sender Integration

**Complexity**: Medium
**Dependencies**: Feedback Learner DSPy Receiver role, GEPA optimizer
**Status**: ✅ COMPLETE

### Task 2.1: Create Unified Signal Router ✅ COMPLETE

**Files**:
- `src/agents/tier2_signal_router.py` (already existed)
- `src/agents/feedback_learner/dspy_receiver.py` (created)

**Implementation**:
- `Tier2SignalRouter` class with batching and async delivery
- `submit_signal()` and `flush()` methods
- Convenience functions: `route_causal_impact_signal()`, `route_gap_analyzer_signal()`, `route_heterogeneous_optimizer_signal()`
- Metrics tracking and graceful fallback
- Singleton access via `get_signal_router()`
- `TrainingSignalReceiver` class for receiving signals in feedback_learner
- Signal-to-FeedbackItem conversion for feedback learning cycle

### Task 2.2: Activate DSPy Signals in Causal Impact Agent ✅ COMPLETE

**Files** (already implemented):
- `src/agents/causal_impact/nodes/interpretation.py` - `_collect_dspy_signal()` method

**Implementation**:
- Imports `route_causal_impact_signal` from tier2_signal_router
- Collects signals with all workflow phases (graph building, estimation, energy score, refutation, sensitivity, interpretation)
- Routes to feedback_learner at workflow completion (line 496)

### Task 2.3: Activate DSPy Signals in Gap Analyzer Agent ✅ COMPLETE

**Files Modified**:
- `src/agents/gap_analyzer/nodes/formatter.py` - Added routing to `_collect_dspy_signal()`

**Implementation**:
- Added import of `route_gap_analyzer_signal` from tier2_signal_router
- Routes signal after update_prioritization() call

### Task 2.4: Activate DSPy Signals in Heterogeneous Optimizer Agent ✅ COMPLETE

**Files** (already implemented):
- `src/agents/heterogeneous_optimizer/nodes/profile_generator.py` - `_collect_dspy_signal()` method

**Implementation**:
- Imports `route_heterogeneous_optimizer_signal` from tier2_signal_router
- Collects signals with all workflow phases (CATE estimation, segment discovery, policy learning)
- Routes to feedback_learner at workflow completion (line 387)

### Acceptance Criteria

- [x] All three agents collect training signals during workflow execution
- [x] Signals include all phase metrics
- [x] Reward computation returns values between 0.0 and 1.0
- [x] Signals routed to feedback_learner within 5 seconds of workflow completion
- [x] Signal collection adds <50ms overhead
- [x] Unit tests verify signal structure
- [x] Integration test validates end-to-end signal flow

---

## Phase 3: Multi-Library Stress Testing Expansion

**Complexity**: Medium
**Dependencies**: Existing causal engine components
**Status**: ✅ COMPLETE

### Task 3.1: Create Cross-Library Estimator Comparison Tests ✅ COMPLETE

**File**: `tests/unit/test_causal_engine/test_cross_validation/test_estimator_comparison.py` (456 lines)

**Tests Implemented**:
- DoWhy OLS vs EconML LinearDML
- DoWhy IPW vs EconML DRLearner
- EconML CausalForestDML vs CausalML UpliftRandomForest
- Effect size recovery tests
- Confidence interval overlap tests

### Task 3.2: Create Cross-Library Robustness Tests ✅ COMPLETE

**File**: `tests/unit/test_causal_engine/test_cross_validation/test_refutation_consistency.py` (470 lines)

**Tests Implemented**:
- Placebo treatment consistency tests
- Random common cause consistency tests
- Bootstrap refutation consistency tests
- Cross-estimator gate consistency tests

### Task 3.3: Create CATE Consistency Tests ✅ COMPLETE

**File**: `tests/unit/test_causal_engine/test_cross_validation/test_cate_consistency.py` (512 lines)

**Tests Implemented**:
- CATE sign consistency across methods
- Segment ranking consistency (Spearman correlation)
- High responder identification consistency
- Homogeneous effect uniformity tests
- Continuous heterogeneity detection
- Cross-method individual CATE correlation

### Task 3.4: Create Energy Score Cross-Validation Tests ✅ COMPLETE

**File**: `tests/unit/test_causal_engine/test_energy_score/test_cross_validation.py` (461 lines)

**Tests Implemented**:
- Energy score vs ground truth accuracy correlation
- Selection strategy outcomes (best_energy, first_success)
- Energy score reproducibility and stability
- Quality tier threshold verification
- Nonlinear data challenge tests
- Noisy data behavior tests

### Test Data Requirements

- Synthetic datasets with known causal effects (ATE = 0.5, 1.0, 2.0)
- Datasets with heterogeneous effects (CATE varying by segment)
- Datasets with confounding
- Sample sizes: 1K, 5K, 10K rows

### Acceptance Criteria

- [x] Cross-library estimator tests pass with <10% relative error
- [x] Refutation consistency tests verify same gate decision across estimators
- [x] CATE ranking tests show Spearman correlation >0.7 across methods
- [x] Energy score selection tests validate correct estimator ranking
- [x] Tests run within pytest memory limits (4 workers)

---

## Phase 4: Causal Discovery Stress Testing

**Complexity**: Medium
**Dependencies**: Discovery module
**Status**: ✅ COMPLETE

### Task 4.1: Create Large-Scale Discovery Tests ✅ COMPLETE

**File**: `tests/stress/test_discovery_scale.py` (435 lines)

**Test Fixtures**:
- 10K rows, 10 variables
- 50K rows, 20 variables
- 100K rows, 30 variables

**Tests Implemented**:
- GES algorithm scale tests (10K, 50K, 100K rows)
- PC algorithm scale tests (10K, 50K, 100K rows)
- LiNGAM algorithm scale tests (10K, 50K, 100K rows)
- Edge recovery accuracy tests
- Convergence with scale tests

### Task 4.2: Create Parallel Algorithm Execution Tests ✅ COMPLETE

**File**: `tests/stress/test_discovery_parallel.py` (450 lines)

**Tests Implemented**:
- Parallel speedup vs sequential execution
- ProcessPoolExecutor serialization tests
- Data serialization roundtrip verification
- Timeout behavior tests
- Memory cleanup after parallel runs
- Repeated parallel runs stability
- Worker process isolation tests
- Scale tests with 50K rows

### Task 4.3: Create Cache Performance Tests ✅ COMPLETE

**File**: `tests/stress/test_discovery_cache.py` (320 lines)

**Tests Implemented**:
- Cache hit performance (<100ms target)
- Cache reduces time by >90%
- Cache hash stability (same data = same hash)
- Different data produces different hash
- FIFO eviction behavior
- Hit rate tracking accuracy
- Cache overhead measurement
- Large data caching (50K rows)

### Task 4.4: Create Graph Builder Integration Stress Tests ✅ COMPLETE

**File**: `tests/stress/test_graph_builder_scale.py` (422 lines)

**Tests Implemented**:
- auto_discover on 5K, 20K, 50K rows
- Gate decision quality tests
- Gate confidence validation
- Latency comparison: manual vs auto_discover
- Augmentation with high-confidence edges

### Infrastructure Requirements

- Tests in separate `tests/stress/` directory (not run in CI by default)
- Mark tests with `@pytest.mark.stress`
- Consider running on droplet for accurate timing
- Memory profiling with tracemalloc

### Acceptance Criteria

- [x] GES algorithm completes on 100K rows in <60 seconds
- [x] PC algorithm completes on 100K rows in <120 seconds
- [x] Memory usage stays under 8GB for 100K row datasets
- [x] Ensemble voting produces consistent results at scale
- [x] Cache reduces repeated query time by >90%
- [x] Parallel execution shows speedup for 3+ algorithms
- [x] Graph builder with auto_discover meets <30s target

---

## Execution Order

```
Phase 1: Memory Contribution Hooks
   └── Task 1.1 → Task 1.2 → Task 1.3

Phase 2: DSPy Sender Integration (depends on Phase 1 patterns)
   └── Task 2.1 (router) → Tasks 2.2, 2.3, 2.4 (can parallelize)

Phase 3: Multi-Library Stress Testing (parallel with Phases 1-2)
   └── Tasks 3.1-3.4 (can parallelize)

Phase 4: Causal Discovery Stress Testing (can run last)
   └── Tasks 4.1-4.4 (can parallelize)
```

---

## Dependency Graph

```
Memory Hooks (Phase 1)
    │
    └──► DSPy Sender Integration (Phase 2)
              │
              └──► Feedback Learner (external)

Multi-Library Tests (Phase 3)  ──► Existing causal engine (no blockers)

Discovery Stress Tests (Phase 4) ──► Discovery module (no blockers)
```

---

## Risk Assessment

### High Risk

| Risk | Mitigation |
|------|------------|
| DSPy/GEPA Compatibility | Validate signal schema against feedback_learner expectations |
| Memory Backend Availability | Add graceful fallback, never block on memory failures |

### Medium Risk

| Risk | Mitigation |
|------|------------|
| Performance Overhead | Use async operations, batch where possible, monitor with Opik |
| Large-Scale Test Stability | Set appropriate timeouts, run in isolated environment |

### Low Risk

| Risk | Mitigation |
|------|------------|
| Test Organization | Follow existing naming conventions |

---

## Key Files Reference

| Purpose | File |
|---------|------|
| Gap Analyzer memory hooks | `src/agents/gap_analyzer/memory_hooks.py` |
| Tier 2 Signal Router | `src/agents/tier2_signal_router.py` |
| DSPy Signal Receiver | `src/agents/feedback_learner/dspy_receiver.py` |
| Causal Impact DSPy integration | `src/agents/causal_impact/dspy_integration.py` |
| Causal Impact signal routing | `src/agents/causal_impact/nodes/interpretation.py` |
| Gap Analyzer DSPy integration | `src/agents/gap_analyzer/dspy_integration.py` |
| Gap Analyzer signal routing | `src/agents/gap_analyzer/nodes/formatter.py` |
| Heterogeneous Optimizer DSPy | `src/agents/heterogeneous_optimizer/dspy_integration.py` |
| Heterogeneous Optimizer routing | `src/agents/heterogeneous_optimizer/nodes/profile_generator.py` |
| Discovery runner | `src/causal_engine/discovery/runner.py` |
| Existing discovery tests | `tests/unit/test_causal_engine/test_discovery/` |
| Cross-library estimator tests | `tests/unit/test_causal_engine/test_cross_validation/test_estimator_comparison.py` |
| Refutation consistency tests | `tests/unit/test_causal_engine/test_cross_validation/test_refutation_consistency.py` |
| CATE consistency tests | `tests/unit/test_causal_engine/test_cross_validation/test_cate_consistency.py` |
| Energy score cross-validation | `tests/unit/test_causal_engine/test_energy_score/test_cross_validation.py` |
| Discovery scale tests | `tests/stress/test_discovery_scale.py` |
| Discovery parallel tests | `tests/stress/test_discovery_parallel.py` |
| Discovery cache tests | `tests/stress/test_discovery_cache.py` |
| Graph builder scale tests | `tests/stress/test_graph_builder_scale.py` |
| Signal router unit tests | `tests/unit/test_agents/test_tier2_signal_routing/test_signal_router.py` |
| Signal receiver unit tests | `tests/unit/test_agents/test_tier2_signal_routing/test_signal_receiver.py` |
| E2E signal flow tests | `tests/integration/test_signal_flow/test_e2e_signal_flow.py` |

---

## Complexity Summary

| Phase | Enhancement | Complexity | Tasks |
|-------|-------------|------------|-------|
| 1 | Memory Contribution Hooks | Low | 3 |
| 2 | DSPy Sender Integration | Medium | 4 |
| 3 | Multi-Library Stress Testing | Medium | 4 |
| 4 | Causal Discovery Stress Testing | Medium | 4 |
| **Total** | | | **15** |
