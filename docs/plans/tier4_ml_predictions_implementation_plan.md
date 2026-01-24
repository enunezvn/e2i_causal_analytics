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
| 1 | Resource Optimizer Memory Hooks | ~2 hours | [ ] Not Started |
| 2 | Prediction Synthesizer Memory Hooks | ~2 hours | [ ] Not Started |
| 3 | Prediction Synthesizer DSPy Integration | ~3 hours | [ ] Not Started |
| 4 | Prediction Synthesizer Opik Tracing | ~1 hour | [ ] Not Started |
| 5 | Resource Optimizer MILP Enhancement | ~2 hours | [ ] Optional |

---

## Phase 1: Resource Optimizer Memory Hooks

**Goal**: Implement memory integration for caching optimization results and learning allocation patterns.

**Files to Create/Modify**:
- `src/agents/resource_optimizer/memory_hooks.py` (create)
- `src/agents/resource_optimizer/agent.py` (modify)
- `src/agents/resource_optimizer/nodes/impact_projector.py` (modify)
- `tests/unit/agents/test_resource_optimizer/test_memory_hooks.py` (create)

### Tasks

- [ ] **1.1** Create `memory_hooks.py` with `ResourceOptimizerMemoryHooks` class
  - [ ] Implement `MemoryHooksInterface` ABC
  - [ ] Add `get_context()` for retrieving past optimization patterns
  - [ ] Add `contribute_to_memory()` for storing allocation outcomes

- [ ] **1.2** Implement Working Memory (Redis)
  - [ ] Cache optimization solutions with 24h TTL
  - [ ] Key pattern: `ro:solution:{problem_signature}`
  - [ ] Include constraint hash for cache matching

- [ ] **1.3** Implement Procedural Memory (Supabase + pgvector)
  - [ ] Store successful allocation strategies
  - [ ] Enable similarity search for warm starts
  - [ ] Track solver performance by problem type

- [ ] **1.4** Integrate with agent
  - [ ] Add memory hooks to `ResourceOptimizerAgent.__init__()`
  - [ ] Call `get_context()` in problem formulator
  - [ ] Call `contribute_to_memory()` in impact projector

- [ ] **1.5** Write tests
  - [ ] Test `get_context()` with mock Redis/Supabase
  - [ ] Test `contribute_to_memory()` with mock storage
  - [ ] Test graceful degradation when memory unavailable

### Acceptance Criteria
- [ ] Memory hooks file created and integrated
- [ ] Redis caching functional
- [ ] Procedural memory stores successful allocations
- [ ] All tests passing
- [ ] CONTRACT_VALIDATION.md updated to 100%

---

## Phase 2: Prediction Synthesizer Memory Hooks

**Goal**: Implement memory integration for caching predictions and learning from historical accuracy.

**Files to Create/Modify**:
- `src/agents/prediction_synthesizer/memory_hooks.py` (create)
- `src/agents/prediction_synthesizer/agent.py` (modify)
- `src/agents/prediction_synthesizer/nodes/context_enricher.py` (modify)
- `tests/unit/agents/test_prediction_synthesizer/test_memory_hooks.py` (create)

### Tasks

- [ ] **2.1** Create `memory_hooks.py` with `PredictionSynthesizerMemoryHooks` class
  - [ ] Implement `MemoryHooksInterface` ABC
  - [ ] Add `get_context()` for retrieving similar predictions
  - [ ] Add `contribute_to_memory()` for storing prediction outcomes

- [ ] **2.2** Implement Working Memory (Redis)
  - [ ] Cache ensemble predictions with configurable TTL
  - [ ] Key pattern: `ps:prediction:{entity_id}:{target}:{feature_hash}`
  - [ ] Include confidence threshold for cache validity

- [ ] **2.3** Implement Episodic Memory (Supabase + pgvector)
  - [ ] Store prediction outcomes with actual results (when available)
  - [ ] Track model accuracy over time
  - [ ] Enable calibration based on historical performance

- [ ] **2.4** Integrate with agent
  - [ ] Add memory hooks to `PredictionSynthesizerAgent.__init__()`
  - [ ] Check cache in `synthesize()` before running graph
  - [ ] Call `contribute_to_memory()` after successful prediction

- [ ] **2.5** Write tests
  - [ ] Test cache hit/miss scenarios
  - [ ] Test episodic memory storage
  - [ ] Test historical accuracy integration

### Acceptance Criteria
- [ ] Memory hooks file created and integrated
- [ ] Redis caching functional with TTL
- [ ] Episodic memory stores predictions
- [ ] All tests passing
- [ ] CONTRACT_VALIDATION.md updated to reflect progress

---

## Phase 3: Prediction Synthesizer DSPy Integration

**Goal**: Implement training signal emission for feedback learner optimization.

**Files to Create/Modify**:
- `src/agents/prediction_synthesizer/dspy_integration.py` (complete)
- `src/agents/prediction_synthesizer/nodes/ensemble_combiner.py` (modify)
- `tests/unit/agents/test_prediction_synthesizer/test_dspy_integration.py` (create)

### Tasks

- [ ] **3.1** Define DSPy signatures
  - [ ] `EvidenceSynthesisSignature` for prediction evidence
  - [ ] `PredictionQualitySignature` for quality scoring
  - [ ] Include model agreement, calibration, historical accuracy

- [ ] **3.2** Implement training signal emission
  - [ ] Create `PredictionTrainingSignal` class
  - [ ] Compute reward based on:
    - Model agreement (0.4 weight)
    - Calibration accuracy (0.4 weight)
    - Historical accuracy (0.2 weight)
  - [ ] Emit signals to feedback_learner queue

- [ ] **3.3** Add signal collection points
  - [ ] Collect in ensemble combiner after aggregation
  - [ ] Include prediction context in signal
  - [ ] Add batch_id for correlation

- [ ] **3.4** Integrate with GEPA optimizer
  - [ ] Register agent with GEPA metric
  - [ ] Configure optimization budget (light/medium/heavy)
  - [ ] Enable A/B testing for prompt variants

- [ ] **3.5** Write tests
  - [ ] Test signal computation
  - [ ] Test reward calculation
  - [ ] Test GEPA integration

### Acceptance Criteria
- [ ] DSPy signatures defined
- [ ] Training signals emitted correctly
- [ ] GEPA integration configured
- [ ] All tests passing
- [ ] CONTRACT_VALIDATION.md updated to 100%

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
- Start Date: ___________
- Completion Date: ___________
- Tests Passing: ___/___
- Notes:

### Phase 2: Prediction Synthesizer Memory Hooks
- Start Date: ___________
- Completion Date: ___________
- Tests Passing: ___/___
- Notes:

### Phase 3: Prediction Synthesizer DSPy Integration
- Start Date: ___________
- Completion Date: ___________
- Tests Passing: ___/___
- Notes:

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
