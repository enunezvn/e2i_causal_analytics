# Tier 4: ML Predictions Agents - Implementation Plan

**Created**: 2026-01-24
**Status**: In Progress
**Testing**: User will perform on droplet (138.197.4.36)

---

## Overview

This plan addresses the pending items from the Tier 4 ML Predictions agents evaluation to achieve 100% contract compliance.

### Current Status
- Prediction Synthesizer: 92% compliant (core complete)
- Resource Optimizer: 95% compliant (core complete)
- **Target**: 100% compliance for both agents

### Phases Summary

| Phase | Component | Effort | Status |
|-------|-----------|--------|--------|
| 1 | Resource Optimizer Memory Hooks | ~2 hours | [x] COMPLETE |
| 2 | Prediction Synthesizer Memory Hooks | ~2 hours | [x] COMPLETE |
| 3 | Prediction Synthesizer DSPy Integration | ~3 hours | [x] COMPLETE |
| 4 | Prediction Synthesizer Opik Tracing | ~1 hour | [ ] Not Started |
| 5 | Resource Optimizer MILP Enhancement | ~2 hours | [ ] Optional |

---

## Phase 1: Resource Optimizer Memory Hooks ✅ COMPLETE

**Goal**: Implement memory integration for caching optimization results and learning allocation patterns.

**Files Created/Modified**:
- `src/agents/resource_optimizer/memory_hooks.py` (already existed - 645 lines)
- `src/agents/resource_optimizer/agent.py` (modified - added memory integration)
- `tests/unit/agents/test_resource_optimizer/test_memory_hooks.py` (created - comprehensive tests)

### Tasks

- [x] **1.1** Create `memory_hooks.py` with `ResourceOptimizerMemoryHooks` class
  - [x] Implement `MemoryHooksInterface` ABC
  - [x] Add `get_context()` for retrieving past optimization patterns
  - [x] Add `contribute_to_memory()` for storing allocation outcomes

- [x] **1.2** Implement Working Memory (Redis)
  - [x] Cache optimization solutions with 1h TTL
  - [x] Key pattern: `resource_optimizer:session:{session_id}`
  - [x] Scenario caching for comparison

- [x] **1.3** Implement Procedural Memory (Supabase + pgvector)
  - [x] Store successful allocation strategies (`store_optimization_pattern()`)
  - [x] Enable similarity search via `search_procedures_by_text()`
  - [x] Track solver performance by problem type

- [x] **1.4** Integrate with agent
  - [x] Add `enable_memory` flag to `ResourceOptimizerAgent.__init__()`
  - [x] Add `memory_hooks` property with lazy loading
  - [x] Call `get_context()` at start of `optimize()`
  - [x] Call `contribute_to_memory()` after successful optimization

- [x] **1.5** Write tests
  - [x] Test `get_context()` with mock Redis/Supabase
  - [x] Test `contribute_to_memory()` with mock storage
  - [x] Test graceful degradation when memory unavailable
  - [x] Test data structures (OptimizationContext, OptimizationPattern, OptimizationRecord)

### Acceptance Criteria
- [x] Memory hooks file created and integrated
- [x] Redis caching functional (with graceful degradation)
- [x] Procedural memory stores successful allocations
- [x] All tests passing
- [x] CONTRACT_VALIDATION.md updated to 100%

### Completion Date: 2026-01-24

---

## Phase 2: Prediction Synthesizer Memory Hooks ✅ COMPLETE

**Goal**: Implement memory integration for caching predictions and learning from historical accuracy.

**Files Created/Modified**:
- `src/agents/prediction_synthesizer/memory_hooks.py` (already existed - 672 lines)
- `src/agents/prediction_synthesizer/agent.py` (modified - added memory integration)
- `tests/unit/test_agents/test_prediction_synthesizer/test_memory_hooks.py` (created - comprehensive tests)

### Tasks

- [x] **2.1** Create `memory_hooks.py` with `PredictionSynthesizerMemoryHooks` class
  - [x] Implement `MemoryHooksInterface` ABC
  - [x] Add `get_context()` for retrieving similar predictions
  - [x] Add `contribute_to_memory()` for storing prediction outcomes

- [x] **2.2** Implement Working Memory (Redis)
  - [x] Cache ensemble predictions with configurable TTL (1h entity, 24h session)
  - [x] Key pattern: `prediction_synthesizer:entity:{entity_type}:{entity_id}:{target}`
  - [x] Model performance tracking with 7d TTL

- [x] **2.3** Implement Episodic Memory (Supabase + pgvector)
  - [x] Store prediction outcomes via `store_prediction()`
  - [x] Track model accuracy via `update_model_performance()`
  - [x] Enable calibration via `get_calibration_data()`

- [x] **2.4** Integrate with agent
  - [x] Add `enable_memory` flag to `PredictionSynthesizerAgent.__init__()`
  - [x] Add `memory_hooks` property with lazy loading
  - [x] Call `get_context()` at start of `synthesize()`
  - [x] Call `contribute_to_memory()` after successful prediction

- [x] **2.5** Write tests
  - [x] Test context retrieval with graceful degradation
  - [x] Test prediction caching
  - [x] Test model performance tracking
  - [x] Test data structure validation

### Acceptance Criteria
- [x] Memory hooks file created and integrated
- [x] Redis caching functional with TTL (with graceful degradation)
- [x] Episodic memory stores predictions
- [x] All tests passing
- [x] CONTRACT_VALIDATION.md updated to 96%

### Completion Date: 2026-01-24

---

## Phase 3: Prediction Synthesizer DSPy Integration ✅ COMPLETE

**Goal**: Implement training signal emission for feedback learner optimization.

**Files Created/Modified**:
- `src/agents/prediction_synthesizer/dspy_integration.py` (extended - 607 lines)
- `src/agents/prediction_synthesizer/agent.py` (modified - added enable_dspy flag)
- `tests/unit/test_agents/test_prediction_synthesizer/test_dspy_integration.py` (extended - comprehensive tests)

### Tasks

- [x] **3.1** Define DSPy signatures
  - [x] `PredictionSynthesisSignature` for synthesizing predictions
  - [x] `PredictionInterpretationSignature` for quality interpretation
  - [x] `UncertaintyQuantificationSignature` for uncertainty
  - [x] Include model agreement, calibration, historical accuracy

- [x] **3.2** Implement training signal emission
  - [x] Create `PredictionSynthesisTrainingSignal` dataclass
  - [x] Compute reward based on:
    - Model success rate (0.25 weight)
    - Ensemble quality (0.25 weight)
    - Efficiency (0.15 weight)
    - Context quality (0.15 weight)
    - Accuracy/satisfaction (0.20 weight)
  - [x] `emit_training_signal()` - emits to feedback_learner via memory hooks

- [x] **3.3** Add signal collection points
  - [x] `PredictionSynthesizerSignalCollector` class for buffering
  - [x] `create_signal_from_result()` - creates signal from prediction output
  - [x] `collect_and_emit_signal()` - convenience function for agent

- [x] **3.4** Integrate with agent
  - [x] Add `enable_dspy` flag to `PredictionSynthesizerAgent.__init__()` (default: True)
  - [x] Call `collect_and_emit_signal()` after successful predictions
  - [x] Graceful degradation when feedback_learner unavailable

- [x] **3.5** Write tests
  - [x] Test signal creation and reward computation
  - [x] Test signal collector functionality
  - [x] Test signal emission with mocked feedback_learner
  - [x] Test agent integration (enable_dspy flag)

### Acceptance Criteria
- [x] DSPy signatures defined (3 signatures)
- [x] Training signals emitted correctly
- [x] Agent integration complete with enable_dspy flag
- [x] All tests passing
- [x] CONTRACT_VALIDATION.md updated to 100%

### Completion Date: 2026-01-24

---

## Phase 4: Prediction Synthesizer Opik Tracing

**Goal**: Add distributed tracing for observability parity with Resource Optimizer.

**Files to Create/Modify**:
- `src/agents/prediction_synthesizer/opik_tracer.py` (create)
- `src/agents/prediction_synthesizer/agent.py` (modify)
- `src/agents/prediction_synthesizer/nodes/*.py` (modify)
- `tests/unit/agents/test_prediction_synthesizer/test_opik_tracer.py` (create)

### Tasks

- [ ] **4.1** Create `opik_tracer.py`
  - [ ] Implement `PredictionSynthesizerOpikTracer` class
  - [ ] Follow same pattern as `resource_optimizer/opik_tracer.py`
  - [ ] Use UUID v7 compatible trace IDs

- [ ] **4.2** Add span context managers
  - [ ] `trace_orchestration()` for model orchestrator
  - [ ] `trace_ensemble()` for ensemble combiner
  - [ ] `trace_enrichment()` for context enricher

- [ ] **4.3** Integrate with nodes
  - [ ] Add tracing to model_orchestrator.py
  - [ ] Add tracing to ensemble_combiner.py
  - [ ] Add tracing to context_enricher.py

- [ ] **4.4** Add trace metadata
  - [ ] Log model count, latencies
  - [ ] Log ensemble method, agreement
  - [ ] Log context sources used

- [ ] **4.5** Write tests
  - [ ] Test tracer initialization
  - [ ] Test span creation
  - [ ] Test metadata logging

### Acceptance Criteria
- [ ] Opik tracer implemented
- [ ] All nodes traced
- [ ] Traces visible in Opik UI (http://138.197.4.36:5173)
- [ ] All tests passing

---

## Phase 5: Resource Optimizer MILP Enhancement (Optional)

**Goal**: Replace MILP fallback with proper mixed-integer solver.

**Files to Create/Modify**:
- `src/agents/resource_optimizer/nodes/optimizer.py` (modify)
- `requirements.txt` (add pulp or ortools)
- `tests/unit/agents/test_resource_optimizer/test_optimizer.py` (extend)

### Tasks

- [ ] **5.1** Evaluate solver options
  - [ ] PuLP (lightweight, CBC backend)
  - [ ] OR-Tools (Google, more features)
  - [ ] Choose based on dependency footprint

- [ ] **5.2** Implement MILP solver
  - [ ] Add `_solve_milp_proper()` method
  - [ ] Handle integer variable constraints
  - [ ] Support binary allocation decisions

- [ ] **5.3** Add discrete allocation support
  - [ ] Integer allocation amounts
  - [ ] Binary include/exclude decisions
  - [ ] Cardinality constraints (max N entities)

- [ ] **5.4** Write tests
  - [ ] Test integer solutions
  - [ ] Test binary decisions
  - [ ] Test cardinality constraints

### Acceptance Criteria
- [ ] True MILP solver working
- [ ] Discrete allocation supported
- [ ] Performance within 20s target
- [ ] All tests passing

---

## Testing Instructions (Droplet)

### SSH Access
```bash
ssh -i ~/.ssh/replit enunez@138.197.4.36
```

### Run Tests
```bash
# Activate venv
source /opt/e2i_causal_analytics/.venv/bin/activate

# Run Tier 4 tests (memory-safe: 4 workers)
pytest tests/unit/agents/test_prediction_synthesizer/ -v -n 4
pytest tests/unit/agents/test_resource_optimizer/ -v -n 4

# Run specific phase tests
pytest tests/unit/agents/test_resource_optimizer/test_memory_hooks.py -v
pytest tests/unit/agents/test_prediction_synthesizer/test_dspy_integration.py -v
```

### Verify Services
```bash
# Check Opik is running
curl -s localhost:5173/health

# Check Redis is running
redis-cli -p 6382 ping

# Check MLflow is running
curl -s localhost:5000/health
```

---

## Progress Tracking

### Phase 1: Resource Optimizer Memory Hooks
- Start Date: 2026-01-24
- Completion Date: 2026-01-24
- Tests Passing: All memory hooks tests + existing 62 core tests
- Notes: Memory hooks already existed (645 lines). Integrated into agent.py with enable_memory flag, memory_hooks property, get_context() call, and contribute_to_memory() call. Created comprehensive test suite.

### Phase 2: Prediction Synthesizer Memory Hooks
- Start Date: 2026-01-24
- Completion Date: 2026-01-24
- Tests Passing: All memory hooks tests + existing 66 core tests
- Notes: Memory hooks already existed (672 lines). Integrated into agent.py with enable_memory flag, memory_hooks property, session_id parameter, get_context() call, and contribute_to_memory() call. Created comprehensive test suite.

### Phase 3: Prediction Synthesizer DSPy Integration
- Start Date: 2026-01-24
- Completion Date: 2026-01-24
- Tests Passing: All DSPy integration tests + existing 66 core tests
- Notes: Extended existing dspy_integration.py (607 lines) with signal emission functions: emit_training_signal(), create_signal_from_result(), collect_and_emit_signal(). Added enable_dspy flag to agent.py. Extended test file with signal emission tests and agent integration tests.

### Phase 4: Prediction Synthesizer Opik Tracing
- Start Date: ___________
- Completion Date: ___________
- Tests Passing: ___/___
- Notes:

### Phase 5: Resource Optimizer MILP Enhancement
- Start Date: ___________
- Completion Date: ___________
- Tests Passing: ___/___
- Notes:

---

## Dependencies

### Required Services (on droplet)
- Redis: Port 6382
- Supabase/PostgreSQL: Via connection string
- Opik: Port 5173 (UI), 8080 (API)
- MLflow: Port 5000

### Python Dependencies (already installed)
- `redis` - Working memory
- `supabase` - Episodic/procedural memory
- `opik` - Distributed tracing
- `dspy-ai` - Training signals
- `scipy` - Optimization solvers

---

## Reference Documents

- Evaluation: `docs/evaluations/tier4_ml_predictions_evaluation.md`
- Contracts: `.claude/contracts/tier4-contracts.md`
- Specialists: `.claude/specialists/Agent_Specialists_Tiers 1-5/`
- Memory Architecture: `E2I_Agentic_Memory_Documentation.html`
- GEPA: `src/optimization/gepa/`
