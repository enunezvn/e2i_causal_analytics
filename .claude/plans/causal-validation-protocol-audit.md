# E2I Causal Validation Protocol v1.0 - Implementation Audit Plan

**Created**: 2025-12-30
**Status**: ✅ COMPLETED
**Completed**: 2025-12-30
**Protocol Version**: v1.0 (December 2025)

---

## Executive Summary

This audit compares the **E2I Causal Validation Protocol v1.0** documentation against the actual codebase implementation. The audit is organized into 4 execution waves with context-window-friendly task batches.

### Implementation Status Overview

| Component | Status | Notes |
|-----------|--------|-------|
| 5-Node Workflow | ✅ IMPLEMENTED | GraphBuilder → Estimation → Refutation → Sensitivity → Interpretation |
| DoWhy Refutation (5 tests) | ✅ IMPLEMENTED | All thresholds match protocol (43/43 tests pass) |
| Expert Review System | ✅ IMPLEMENTED | DAG hashing, approval workflow, expiration |
| Synthetic Benchmarks | ✅ IMPLEMENTED | 3 DGPs with comprehensive tests (47/47 pass) |
| Database Schema | ✅ IMPLEMENTED | Tables, views, functions present |
| Gate Decision Logic | ✅ IMPLEMENTED | proceed/review/block thresholds correct |
| Feedback Learner Integration | ✅ FIXED | ValidationOutcome wired to FeedbackLearnerState |
| Observability (Opik) | ✅ FIXED | Per-test Opik tracing added to RefutationRunner |

---

## Critical Files

```
# Core Implementation
src/agents/causal_impact/graph.py           # 5-node LangGraph workflow
src/agents/causal_impact/nodes/             # Node implementations
src/causal_engine/refutation_runner.py      # DoWhy refutation tests + thresholds
src/causal_engine/expert_review_gate.py     # Expert approval workflow
src/causal_engine/validation_outcome.py     # Feedback Learner integration (needs wiring)

# Database Schema
database/ml/010_causal_validation_tables.sql

# Tests
tests/synthetic/conftest.py                 # Synthetic DGP fixtures
tests/synthetic/test_simple_linear.py
tests/synthetic/test_confounded_moderate.py
tests/synthetic/test_heterogeneous_cate.py
tests/unit/test_causal_engine/              # Unit tests

# Contracts
.claude/contracts/tier2-contracts.md
```

---

## Wave 1: Core Validation Tests (30 min)

### 1.1 Run Refutation Suite Tests
```bash
pytest tests/unit/test_causal_engine/test_refutation_runner.py -n 2 -v
```
**Verify**: All 5 DoWhy tests pass with correct thresholds:
- `placebo_treatment`: p-value > 0.05
- `random_common_cause`: delta < 20%
- `data_subset`: 80% CI coverage
- `bootstrap`: CI ratio < 0.50
- `sensitivity_e_value`: E-value >= 2.0

### 1.2 Run Synthetic Benchmark Tests (Small Batch)
```bash
# Test 1: Simple linear (no confounding)
pytest tests/synthetic/test_simple_linear.py -n 2 -v

# Test 2: Confounded moderate
pytest tests/synthetic/test_confounded_moderate.py -n 2 -v

# Test 3: Heterogeneous CATE
pytest tests/synthetic/test_heterogeneous_cate.py -n 2 -v
```
**Verify**:
- `simple_linear`: True ATE = 0.50 recovered
- `confounded_moderate`: True ATE = 0.30 recovered after adjustment
- `heterogeneous_cate`: Segment CATEs (low=0.10, medium=0.20, high=0.40)

---

## Wave 2: Database & Gate Logic (25 min)

### 2.1 Verify Database Schema
**File**: `database/ml/010_causal_validation_tables.sql`

Check ENUMs:
- [ ] `refutation_test_type`: 5 values (placebo_treatment, random_common_cause, data_subset, bootstrap, sensitivity_e_value)
- [ ] `validation_status`: 4 values (passed, failed, warning, skipped)
- [ ] `gate_decision`: 3 values (proceed, review, block)
- [ ] `expert_review_type`: 4 values

Check Tables:
- [ ] `causal_validations`: Has estimate_id, test_type, status, original_effect, refuted_effect, p_value, delta_percent, confidence_score, gate_decision
- [ ] `expert_reviews`: Has dag_version_hash, reviewer_id, approval_status, valid_from, valid_until

Check Functions:
- [ ] `is_dag_approved(dag_hash, brand)` returns BOOLEAN
- [ ] `get_validation_gate(estimate_id)` returns gate_decision
- [ ] `can_use_estimate(estimate_id, dag_hash)` returns BOOLEAN

### 2.2 Verify Gate Decision Logic
**File**: `src/causal_engine/refutation_runner.py`

Check thresholds:
- [ ] PROCEED: confidence >= 0.70, all critical tests passed
- [ ] REVIEW: confidence 0.50-0.70
- [ ] BLOCK: confidence < 0.50 OR critical test failed

Critical tests:
- [ ] placebo_treatment
- [ ] random_common_cause
- [ ] sensitivity_e_value

```bash
pytest tests/unit/test_causal_engine/ -k "gate_decision" -n 2 -v
```

---

## Wave 3: Expert Review & Integration (35 min)

### 3.1 Verify Expert Review Gate
**File**: `src/causal_engine/expert_review_gate.py`

Check implementation:
- [ ] `ReviewGateDecision` enum: PROCEED, PENDING_REVIEW, RENEWAL_REQUIRED, BLOCKED
- [ ] `ExpertReviewGate.check_approval()` returns ReviewGateResult
- [ ] DAG hash comparison logic
- [ ] Expiration tracking (valid_from, valid_until)
- [ ] Renewal workflow

```bash
pytest tests/unit/test_causal_engine/ -k "expert" -n 2 -v
```

### 3.2 Audit Feedback Learner Integration (GAP)
**File**: `src/causal_engine/validation_outcome.py`

Current state:
- [x] `ValidationOutcome` dataclass exists
- [x] `FailureCategory` enum exists (6 categories)
- [x] `extract_failure_patterns()` function exists
- [ ] **GAP**: ValidationOutcome NOT imported in Feedback Learner

**Required Fix**:
1. Import ValidationOutcome in `src/agents/feedback_learner/state.py`
2. Add validation_outcomes field to FeedbackLearnerState
3. Wire causal_impact agent to emit ValidationOutcome
4. Feedback Learner consumes validation outcomes for learning

```bash
pytest tests/unit/test_causal_engine/ -k "validation_outcome" -n 2 -v
```

---

## Wave 4: Workflow & Observability (30 min)

### 4.1 Verify 5-Node Workflow
**File**: `src/agents/causal_impact/graph.py`

Check workflow wiring:
- [ ] GraphBuilder node: DAG construction (<10s SLA)
- [ ] Estimation node: ATE/CATE calculation (<30s SLA)
- [ ] Refutation node: 5 DoWhy tests as quality gate (<15s SLA)
- [ ] Sensitivity node: E-value analysis (<5s SLA)
- [ ] Interpretation node: LLM narrative (<30s SLA)

Check conditional routing:
- [ ] `should_continue_after_estimation()`: Skip to interpretation if estimation fails
- [ ] `should_continue_after_refutation()`: Block interpretation if gate=BLOCK

### 4.2 Verify Node Implementations
**Directory**: `src/agents/causal_impact/nodes/`

- [ ] `graph_builder.py`: NetworkX DAG construction
- [ ] `estimation.py`: DoWhy/EconML estimators, Energy Score (V4.2)
- [ ] `refutation.py`: Calls RefutationRunner, updates state with gate_decision
- [ ] `sensitivity.py`: E-value calculation
- [ ] `interpretation.py`: Claude narrative generation

### 4.3 Audit Observability (GAP)
**Current**: MLflow tracking present in `src/causal_engine/energy_score/mlflow_tracker.py`
**Missing**: Opik tracing not integrated in causal validation

**Required Fix**:
1. Add Opik trace decorators to refutation tests
2. Log validation outcomes to Opik
3. Create spans for each refutation test

---

## Identified Gaps & Remediation

### Gap 1: Feedback Learner Integration (Priority: HIGH)
**Location**: `src/causal_engine/validation_outcome.py` → `src/agents/feedback_learner/`

**Tasks**:
1. Add `validation_outcomes: List[ValidationOutcome]` to FeedbackLearnerState
2. Create `ValidationOutcomeCollector` node in feedback_learner
3. Wire causal_impact agent to emit ValidationOutcome on each run
4. Add learning logic for failure pattern recognition

### Gap 2: Opik Tracing (Priority: MEDIUM)
**Location**: `src/causal_engine/refutation_runner.py`

**Tasks**:
1. Import `opik.track` decorator
2. Add `@track(project_name='e2i-causal-validation')` to `run_suite()`
3. Create child spans for each refutation test
4. Log gate_decision and confidence_score as span metadata

---

## Test Execution Commands (Resource-Constrained)

```bash
# Wave 1: Core (run sequentially due to memory)
pytest tests/unit/test_causal_engine/test_refutation_runner.py -n 2 --timeout=60
pytest tests/synthetic/test_simple_linear.py -n 2 --timeout=120
pytest tests/synthetic/test_confounded_moderate.py -n 2 --timeout=120

# Wave 2: Database/Gate
pytest tests/unit/test_causal_engine/ -k "gate" -n 2 --timeout=60

# Wave 3: Expert Review
pytest tests/unit/test_causal_engine/ -k "expert" -n 2 --timeout=60

# Wave 4: Integration (if tests exist)
pytest tests/unit/test_agents/test_causal_impact/ -n 2 --timeout=180
```

---

## Success Criteria

- [x] All 5 DoWhy refutation tests implemented with correct thresholds ✅ (43/43 tests pass)
- [x] Expert review gate blocks estimates without DAG approval ✅
- [x] Synthetic benchmarks recover true ATEs within tolerance ✅ (47/47 tests pass)
- [x] Gate decision logic: PROCEED (>=0.70), REVIEW (0.50-0.70), BLOCK (<0.50) ✅
- [x] Database schema matches protocol specification ✅
- [x] **GAP FIX**: Feedback Learner receives ValidationOutcome ✅ (Added to FeedbackLearnerState)
- [x] **GAP FIX**: Opik tracing added to validation workflow ✅ (RefutationRunner per-test tracing)

---

## Notes

- **Memory constraint**: Max 4 pytest workers (`-n 4`), prefer `-n 2` for safety
- **Timeout**: 60s per test file, 180s for integration tests
- **Protocol source**: `docs/E2I_Causal_Validation_Protocol.html`
- **ML Foundation reference**: `docs/E2I_ML_Foundation_Data_Flow.html`
