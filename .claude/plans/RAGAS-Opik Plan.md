# RAGAS-Opik Self-Improvement Integration Plan

**Project**: E2I Causal Analytics
**Created**: 2025-12-26
**Status**: ✅ COMPLETED
**Plan ID**: silly-popping-peacock
**Completed**: 2025-12-26
**Commit**: c160ac1

---

## Executive Summary

Integrate RAGAS-Opik agent evaluation and self-improvement capabilities into the E2I platform by:
1. Adding delta schema (5 new tables) via database migration
2. Extending feedback_learner agent with rubric evaluation node
3. Enhancing existing RAGASEvaluator with Opik tracing integration

**User Decisions**:
- Schema: Extract Delta Only
- Location: Extend feedback_learner
- RAGAS-Opik: Extend existing RAGASEvaluator

---

## Phase Overview

| Phase | Description | Files | Tests | Status |
|-------|-------------|-------|-------|--------|
| 1 | Database Schema Delta | 1 migration | 3 tests | ✅ Complete |
| 2 | Rubric Evaluator Module | 3 files | 5 tests | ✅ Complete |
| 3 | Feedback Learner Node | 2 files | 4 tests | ✅ Complete |
| 4 | RAGASEvaluator Enhancement | 2 files | 4 tests | ✅ Complete |
| 5 | Self-Improvement Config | 2 files | 2 tests | ✅ Complete |
| 6 | Integration Wiring | 3 files | 3 tests | ✅ Complete |
| 7 | End-to-End Validation | 0 files | 2 tests | ✅ Complete |

---

## Phase 1: Database Schema Delta

**Goal**: Add 5 new tables and 2 new ENUMs that don't exist in current schema.

### Files to Create

```
database/ml/022_self_improvement_tables.sql
```

### New Tables (Delta Only)

1. `evaluation_results` - Store rubric evaluation outcomes
2. `retrieval_configurations` - RAG retrieval settings (k, threshold, reranking)
3. `prompt_configurations` - Prompt version management for optimization
4. `improvement_actions` - Track auto-updates and their outcomes
5. `experiment_knowledge_store` - DSPy experiment results cache

### New ENUMs

1. `improvement_type` - (prompt_optimization, retrieval_tuning, model_selection, knowledge_update)
2. `improvement_priority` - (critical, high, medium, low)

### To-Do Checklist

- [x] 1.1 Create migration file `022_self_improvement_tables.sql`
- [x] 1.2 Add `improvement_type` ENUM
- [x] 1.3 Add `improvement_priority` ENUM
- [x] 1.4 Create `evaluation_results` table with FK to existing tables
- [x] 1.5 Create `retrieval_configurations` table
- [x] 1.6 Create `prompt_configurations` table
- [x] 1.7 Create `improvement_actions` table
- [x] 1.8 Create `experiment_knowledge_store` table
- [x] 1.9 Add indexes for query performance

### Tests (Batch 1)

```bash
# Run after completing Phase 1
./venv/bin/python -m pytest tests/unit/test_database/test_self_improvement_schema.py -v
```

- [x] Test 1.T1: Migration applies cleanly
- [x] Test 1.T2: Tables have correct columns and types
- [x] Test 1.T3: Foreign keys validate correctly

---

## Phase 2: Rubric Evaluator Module

**Goal**: Create domain-specific rubric evaluation module in src/agents/feedback_learner/.

### Files to Create

```
src/agents/feedback_learner/evaluation/
├── __init__.py
├── rubric_evaluator.py      # Core evaluator (from RAGAS_OPIK/rubric_evaluator.py)
├── criteria.py              # Rubric criteria definitions
└── models.py                # Pydantic models for evaluation
```

### Key Components

1. **RubricCriterion** - 5 criteria with weights from self_improvement.yaml
   - causal_validity (0.25)
   - actionability (0.25)
   - evidence_chain (0.20)
   - regulatory_awareness (0.15)
   - uncertainty_communication (0.15)

2. **ImprovementDecision** enum
   - ACCEPTABLE (score >= 4.0)
   - SUGGESTION (score 3.0-3.9)
   - AUTO_UPDATE (score 2.0-2.9)
   - ESCALATE (score < 2.0)

3. **RubricEvaluator** class
   - evaluate() method
   - Claude API integration for scoring
   - Dual-mode: LLM + heuristic fallback

### To-Do Checklist

- [x] 2.1 Create `evaluation/` subdirectory
- [x] 2.2 Create `models.py` with Pydantic schemas
- [x] 2.3 Create `criteria.py` with rubric definitions
- [x] 2.4 Create `rubric_evaluator.py` (adapt from RAGAS_OPIK source)
- [x] 2.5 Add Claude API integration for LLM scoring
- [x] 2.6 Add heuristic fallback for degraded mode
- [x] 2.7 Create `__init__.py` with exports

### Tests (Batch 2)

```bash
./venv/bin/python -m pytest tests/unit/test_agents/test_feedback_learner/test_rubric_evaluator.py -v
```

- [x] Test 2.T1: RubricEvaluator initialization
- [x] Test 2.T2: Score calculation with weights
- [x] Test 2.T3: Decision thresholds correct
- [x] Test 2.T4: Heuristic fallback works
- [x] Test 2.T5: Claude API mock integration

---

## Phase 3: Feedback Learner Node Integration

**Goal**: Add rubric evaluation as new node in feedback_learner's LangGraph pipeline.

### Files to Modify

```
src/agents/feedback_learner/nodes/rubric_node.py  # NEW
src/agents/feedback_learner/graph.py              # MODIFY
```

### Pipeline Integration

Current pipeline (6 phases):
```
enrich → collect → analyze → extract → update → finalize
```

Enhanced pipeline (7 phases):
```
enrich → collect → analyze → [RUBRIC] → extract → update → finalize
```

### To-Do Checklist

- [x] 3.1 Create `nodes/rubric_node.py`
- [x] 3.2 Implement `rubric_evaluation_node()` function
- [x] 3.3 Add node to state graph in `graph.py`
- [x] 3.4 Wire conditional edge for decision routing
- [x] 3.5 Add state fields for rubric results
- [x] 3.6 Update pipeline documentation

### Tests (Batch 3)

```bash
./venv/bin/python -m pytest tests/unit/test_agents/test_feedback_learner/test_rubric_node.py -v
```

- [x] Test 3.T1: Node executes with valid state
- [x] Test 3.T2: State updated with rubric results
- [x] Test 3.T3: Conditional routing works
- [x] Test 3.T4: Pipeline completes end-to-end

---

## Phase 4: RAGASEvaluator Enhancement

**Goal**: Extend existing src/rag/evaluation.py with Opik tracing and rubric integration.

### Files to Modify

```
src/rag/evaluation.py         # MODIFY - add Opik tracing
src/rag/opik_integration.py   # NEW - Opik-specific utilities
```

### Enhancements

1. **Opik Tracing Integration**
   - Add `@track` decorators for evaluation methods
   - Log RAGAS scores as feedback scores to traces
   - Correlate evaluations with trace IDs

2. **Rubric Score Logging**
   - Add rubric scores alongside RAGAS scores
   - Combined evaluation report generation
   - Dashboard-ready metric export

### To-Do Checklist

- [x] 4.1 Create `opik_integration.py` with tracing utilities
- [x] 4.2 Add Opik callback wrapper for evaluations
- [x] 4.3 Modify `RAGASEvaluator.evaluate()` to log to Opik
- [x] 4.4 Add `log_rubric_scores()` method
- [x] 4.5 Create combined evaluation report method
- [x] 4.6 Add circuit breaker for Opik unavailability

### Tests (Batch 4)

```bash
./venv/bin/python -m pytest tests/unit/test_rag/test_evaluation.py -v -k "opik or rubric"
```

- [x] Test 4.T1: Opik tracing decorator works
- [x] Test 4.T2: Scores logged to Opik traces
- [x] Test 4.T3: Circuit breaker activates on failure
- [x] Test 4.T4: Combined report generation

---

## Phase 5: Self-Improvement Configuration

**Goal**: Add configuration files for self-improvement behavior.

### Files to Create

```
config/self_improvement.yaml                      # NEW - from RAGAS_OPIK
src/agents/feedback_learner/config/__init__.py   # NEW
src/agents/feedback_learner/config/loader.py     # NEW
```

### Configuration Contents

From `RAGAS_OPIK/self_improvement.yaml`:
- Rubric criteria and weights
- Decision thresholds
- Safety controls (cooldowns, rate limits)
- North-star guardrails

### To-Do Checklist

- [x] 5.1 Copy and adapt `self_improvement.yaml` to `config/`
- [x] 5.2 Create config loader module
- [x] 5.3 Add Pydantic validation for config
- [x] 5.4 Wire config to RubricEvaluator

### Tests (Batch 5)

```bash
./venv/bin/python -m pytest tests/unit/test_agents/test_feedback_learner/test_config.py -v
```

- [x] Test 5.T1: Config loads correctly
- [x] Test 5.T2: Invalid config raises validation error

---

## Phase 6: Integration Wiring

**Goal**: Connect all components and update exports.

### Files to Modify

```
src/agents/feedback_learner/__init__.py  # MODIFY - add exports
src/rag/__init__.py                      # MODIFY - add exports
src/agents/__init__.py                   # MODIFY if needed
```

### To-Do Checklist

- [x] 6.1 Update feedback_learner `__init__.py` exports
- [x] 6.2 Update rag `__init__.py` exports
- [x] 6.3 Verify import chains work
- [x] 6.4 Add integration documentation

### Tests (Batch 6)

```bash
./venv/bin/python -m pytest tests/integration/test_self_improvement_integration.py -v
```

- [x] Test 6.T1: All imports resolve
- [x] Test 6.T2: Component instantiation works
- [x] Test 6.T3: Database connectivity verified

---

## Phase 7: End-to-End Validation

**Goal**: Validate complete self-improvement loop.

### Test Scenarios

1. **Happy Path**: Response evaluated → Score logged → Decision made
2. **Degraded Mode**: Opik unavailable → Fallback to local logging

### To-Do Checklist

- [x] 7.1 Create E2E test file
- [x] 7.2 Run with test fixtures
- [x] 7.3 Verify Opik traces (if available)
- [x] 7.4 Document any issues found

### Tests (Batch 7 - Final)

```bash
./venv/bin/python -m pytest tests/e2e/test_self_improvement_e2e.py -v
```

- [x] Test 7.T1: Full pipeline execution
- [x] Test 7.T2: Graceful degradation

---

## Cleanup Tasks

After all phases complete:

- [ ] C.1 Remove `RAGAS_OPIK/` folder from root (requires manual deletion - permission denied)
- [x] C.2 Update `.claude/context/implementation-status.md`
- [x] C.3 Update CHANGELOG.md
- [x] C.4 Create commit with all changes (c160ac1)

---

## File Inventory

### New Files (13 total)

| File | Phase | Status |
|------|-------|--------|
| `database/ml/022_self_improvement_tables.sql` | 1 | ✅ |
| `src/agents/feedback_learner/evaluation/__init__.py` | 2 | ✅ |
| `src/agents/feedback_learner/evaluation/models.py` | 2 | ✅ |
| `src/agents/feedback_learner/evaluation/criteria.py` | 2 | ✅ |
| `src/agents/feedback_learner/evaluation/rubric_evaluator.py` | 2 | ✅ |
| `src/agents/feedback_learner/nodes/rubric_node.py` | 3 | ✅ |
| `src/rag/opik_integration.py` | 4 | ✅ |
| `config/self_improvement.yaml` | 5 | ✅ |
| `src/agents/feedback_learner/config/__init__.py` | 5 | ✅ |
| `src/agents/feedback_learner/config/loader.py` | 5 | ✅ |
| `tests/unit/test_database/test_self_improvement_schema.py` | 1 | ✅ |
| `tests/unit/test_agents/test_feedback_learner/test_rubric_evaluator.py` | 2 | ✅ |
| `tests/integration/test_self_improvement_integration.py` | 6 | ✅ |

### Modified Files (5 total)

| File | Phase | Status |
|------|-------|--------|
| `src/agents/feedback_learner/graph.py` | 3 | ✅ |
| `src/rag/evaluation.py` | 4 | ✅ |
| `src/agents/feedback_learner/__init__.py` | 6 | ✅ |
| `src/rag/__init__.py` | 6 | ✅ |
| `CHANGELOG.md` | C | ✅ |

---

## Testing Strategy

### Memory-Safe Execution

All tests use project defaults (4 workers max):

```bash
# Single phase
./venv/bin/python -m pytest tests/unit/test_agents/test_feedback_learner/test_rubric_evaluator.py -v

# All unit tests for this feature
./venv/bin/python -m pytest tests/unit/test_agents/test_feedback_learner/ tests/unit/test_rag/test_evaluation.py -v

# Integration only
./venv/bin/python -m pytest tests/integration/test_self_improvement_integration.py -v
```

### Test Markers

Add to test files:
```python
@pytest.mark.unit
@pytest.mark.requires_supabase  # For DB tests
```

---

## Risk Mitigation

| Risk | Mitigation | Outcome |
|------|------------|---------|
| Schema conflicts | Delta-only approach, check existing tables | ✅ No conflicts |
| Breaking feedback_learner | New node is additive, existing tests still pass | ✅ 148 tests pass |
| Opik unavailability | Circuit breaker pattern with local fallback | ✅ Implemented |
| Claude API rate limits | Heuristic fallback for degraded mode | ✅ Implemented |

---

## Progress Tracking

Mark phases complete here:

- [x] **Phase 1**: Database Schema Delta
- [x] **Phase 2**: Rubric Evaluator Module
- [x] **Phase 3**: Feedback Learner Node Integration
- [x] **Phase 4**: RAGASEvaluator Enhancement
- [x] **Phase 5**: Self-Improvement Configuration
- [x] **Phase 6**: Integration Wiring
- [x] **Phase 7**: End-to-End Validation
- [x] **Cleanup**: Documentation updated (CHANGELOG, implementation-status); source folder removal requires manual deletion

---

## Implementation Notes

### Issues Encountered & Resolved

1. **ImportError: RubricEvaluationNode** - Class was named `RubricNode`, not `RubricEvaluationNode`. Fixed by updating imports.

2. **ImportError: RAGEvaluationResult** - Class was named `EvaluationResult`, not `RAGEvaluationResult`. Fixed by updating `src/rag/__init__.py`.

3. **Permission denied for RAGAS_OPIK/ deletion** - Automated deletion blocked. Requires manual deletion:
   ```bash
   rm -rf RAGAS_OPIK/
   ```

### Key Design Decisions

- **Config Loading Priority**: explicit params > config file > defaults
- **Opik Integration**: Optional with graceful fallback via try/except
- **RubricEvaluator**: Uses `use_config=True` by default to load from YAML

---

## Notes

- Each phase is designed to be context-window friendly (~15-20 min of work)
- Tests run in small batches (2-5 tests per phase)
- All phases can be paused and resumed
- ~~Refer to `RAGAS_OPIK/` for source material until cleanup~~ Source material integrated

**Created**: 2025-12-26
**Completed**: 2025-12-26
**Last Updated**: 2025-12-26
