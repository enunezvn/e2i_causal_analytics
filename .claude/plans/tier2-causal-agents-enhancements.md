# Tier 2 Causal Inference Agents - Enhancement Plan

**Created**: 2026-01-24
**Status**: PLANNED
**Priority**: Non-blocking enhancements for production-ready agents

---

## Executive Summary

This plan addresses four non-blocking enhancements to the Tier 2 Causal Inference agents (causal_impact, gap_analyzer, heterogeneous_optimizer). The infrastructure for most enhancements is already in place but not activated in production code paths.

---

## Phase 1: Memory Contribution Hooks (Gap Analyzer)

**Complexity**: Low
**Dependencies**: Redis (port 6382), Supabase episodic memory tables

### Task 1.1: Integrate Memory Contribution in Graph

**Files to Modify**:
- `src/agents/gap_analyzer/nodes/formatter.py`

**Implementation**:
```python
from src.agents.gap_analyzer.memory_hooks import contribute_to_memory

# At end of formatter node:
memory_counts = await contribute_to_memory(
    result=formatted_output,
    state=state,
    session_id=state.get("session_id"),
    region=state.get("region"),
)
```

### Task 1.2: Add Context Retrieval to Gap Detector

**Files to Modify**:
- `src/agents/gap_analyzer/nodes/gap_detector.py`

**Implementation**:
- Import memory hooks
- Retrieve episodic context at start of gap detection
- Use historical analyses to inform current gap detection

### Task 1.3: Configure Memory Backend Connections

**Files to Verify**:
- `src/memory/working_memory.py`
- `src/memory/episodic_memory.py`

**Actions**:
- Verify Redis connection on port 6382
- Verify Supabase episodic memory tables exist
- Add fallback behavior if backends unavailable

### Acceptance Criteria

- [ ] Gap analysis results cached in Redis (24h TTL)
- [ ] Gap analysis summaries stored in Supabase episodic memory
- [ ] Historical ROI data retrievable for similar analyses
- [ ] Memory contribution adds <100ms overhead
- [ ] Graceful degradation if backends unavailable

---

## Phase 2: DSPy Sender Integration

**Complexity**: Medium
**Dependencies**: Feedback Learner DSPy Receiver role, GEPA optimizer

### Task 2.1: Create Unified Signal Router

**New File**: `src/agents/tier2_signal_router.py`

**Purpose**: Central router to batch and send signals to feedback_learner

**Implementation**:
```python
class Tier2SignalRouter:
    """Batches training signals from Tier 2 agents for feedback_learner."""

    async def submit_signal(self, agent_name: str, signal: TrainingSignal) -> None:
        """Queue signal for batched delivery."""

    async def flush(self) -> int:
        """Send queued signals to feedback_learner, return count."""
```

### Task 2.2: Activate DSPy Signals in Causal Impact Agent

**Files to Modify**:
- `src/agents/causal_impact/graph.py`
- `src/agents/causal_impact/nodes/interpretation.py`

**Implementation**:
1. Import signal collector:
   ```python
   from src.agents.causal_impact.dspy_integration import get_causal_impact_signal_collector
   ```
2. Collect signals at workflow completion
3. Route signals to feedback_learner

### Task 2.3: Activate DSPy Signals in Gap Analyzer Agent

**Files to Modify**:
- `src/agents/gap_analyzer/graph.py`
- `src/agents/gap_analyzer/nodes/formatter.py`

### Task 2.4: Activate DSPy Signals in Heterogeneous Optimizer Agent

**Files to Modify**:
- `src/agents/heterogeneous_optimizer/graph.py`
- `src/agents/heterogeneous_optimizer/nodes/profile_generator.py`

### Acceptance Criteria

- [ ] All three agents collect training signals during workflow execution
- [ ] Signals include all phase metrics
- [ ] Reward computation returns values between 0.0 and 1.0
- [ ] Signals routed to feedback_learner within 5 seconds of workflow completion
- [ ] Signal collection adds <50ms overhead
- [ ] Unit tests verify signal structure
- [ ] Integration test validates end-to-end signal flow

---

## Phase 3: Multi-Library Stress Testing Expansion

**Complexity**: Medium
**Dependencies**: Existing causal engine components

### Task 3.1: Create Cross-Library Estimator Comparison Tests

**New File**: `tests/unit/test_causal_engine/test_cross_validation/test_estimator_comparison.py`

**Tests**:
- DoWhy OLS vs EconML LinearDML
- DoWhy IPW vs EconML DRLearner
- EconML CausalForestDML vs CausalML UpliftRandomForest

### Task 3.2: Create Cross-Library Robustness Tests

**New File**: `tests/unit/test_causal_engine/test_cross_validation/test_refutation_consistency.py`

**Tests**:
- Same refutation tests across different estimators
- Verify consistent gate decisions (proceed/block/review)

### Task 3.3: Create CATE Consistency Tests

**New File**: `tests/unit/test_causal_engine/test_cross_validation/test_cate_consistency.py`

**Tests**:
- Compare CATE estimates from EconML CausalForestDML, DRLearner, CausalML uplift
- Test segment ranking consistency
- Test CATE sign consistency

### Task 3.4: Create Energy Score Cross-Validation Tests

**New File**: `tests/unit/test_causal_engine/test_energy_score/test_cross_validation.py`

**Tests**:
- Verify energy score ranking correlates with ground truth accuracy
- Test selection strategy outcomes

### Test Data Requirements

- Synthetic datasets with known causal effects (ATE = 0.5, 1.0, 2.0)
- Datasets with heterogeneous effects (CATE varying by segment)
- Datasets with confounding
- Sample sizes: 1K, 5K, 10K rows

### Acceptance Criteria

- [ ] Cross-library estimator tests pass with <10% relative error
- [ ] Refutation consistency tests verify same gate decision across estimators
- [ ] CATE ranking tests show Spearman correlation >0.7 across methods
- [ ] Energy score selection tests validate correct estimator ranking
- [ ] Tests run within pytest memory limits (4 workers)

---

## Phase 4: Causal Discovery Stress Testing

**Complexity**: Medium
**Dependencies**: Discovery module

### Task 4.1: Create Large-Scale Discovery Tests

**New File**: `tests/stress/test_discovery_scale.py`

**Test Fixtures**:
- 10K rows, 10 variables
- 50K rows, 20 variables
- 100K rows, 30 variables

**Tests**:
- Each algorithm individually (GES, PC, FCI, LiNGAM)
- Runtime and memory measurement
- Convergence verification

### Task 4.2: Create Parallel Algorithm Execution Tests

**New File**: `tests/stress/test_discovery_parallel.py`

**Tests**:
- `use_process_pool=True` path with large data
- ProcessPoolExecutor serialization
- Timeout behavior with slow algorithms
- Memory cleanup after parallel runs

### Task 4.3: Create Cache Performance Tests

**New File**: `tests/stress/test_discovery_cache.py`

**Tests**:
- Cache hit performance with repeated queries
- Cache hash stability
- Cache eviction behavior
- Cache overhead vs computation savings

### Task 4.4: Create Graph Builder Integration Stress Tests

**New File**: `tests/stress/test_graph_builder_scale.py`

**Tests**:
- `graph_builder.py` with auto_discover=True on large datasets
- Gate decision quality with scale
- Augmentation path with high-confidence edges
- End-to-end latency vs manual DAG construction

### Infrastructure Requirements

- Tests in separate `tests/stress/` directory (not run in CI by default)
- Mark tests with `@pytest.mark.stress`
- Consider running on droplet for accurate timing
- Add memory profiling decorators

### Acceptance Criteria

- [ ] GES algorithm completes on 100K rows in <60 seconds
- [ ] PC algorithm completes on 100K rows in <120 seconds
- [ ] Memory usage stays under 8GB for 100K row datasets
- [ ] Ensemble voting produces consistent results at scale
- [ ] Cache reduces repeated query time by >90%
- [ ] Parallel execution shows speedup for 3+ algorithms
- [ ] Graph builder with auto_discover meets <30s target

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
| Causal Impact DSPy integration | `src/agents/causal_impact/dspy_integration.py` |
| Gap Analyzer DSPy integration | `src/agents/gap_analyzer/dspy_integration.py` |
| Heterogeneous Optimizer DSPy | `src/agents/heterogeneous_optimizer/dspy_integration.py` |
| Discovery runner | `src/causal_engine/discovery/runner.py` |
| Existing discovery tests | `tests/unit/test_causal_engine/test_discovery/` |

---

## Complexity Summary

| Phase | Enhancement | Complexity | Tasks |
|-------|-------------|------------|-------|
| 1 | Memory Contribution Hooks | Low | 3 |
| 2 | DSPy Sender Integration | Medium | 4 |
| 3 | Multi-Library Stress Testing | Medium | 4 |
| 4 | Causal Discovery Stress Testing | Medium | 4 |
| **Total** | | | **15** |
