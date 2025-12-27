# Energy Score Integration Plan

**Feature**: Energy Score-based Estimator Selection for Causal Impact Agent
**Version**: V4.2
**Created**: 2025-12-26
**Status**: ✅ IMPLEMENTATION COMPLETE

---

## Executive Summary

Integrate the Energy Score Enhancement into the Causal Impact Agent, replacing single-method estimation with energy score-based estimator selection. The enhancement evaluates all causal estimators and selects the one with the lowest energy score (best quality).

**Key Changes**:
- Replace single estimator fallback with multi-estimator evaluation
- Add energy score computation (3 components: treatment_balance 35%, outcome_fit 45%, propensity_calibration 20%)
- Integrate with existing MLflow tracking
- Update DSPy training signals
- Apply database migration for `estimator_evaluations` table

---

## Phase 1: Foundation - Module Setup
**Duration**: ~30 min | **Risk**: Low | **Dependencies**: None

### Objective
Copy enhancement files and establish module structure.

### Files to Create

| File | Source | Description |
|------|--------|-------------|
| `src/causal_engine/energy_score/__init__.py` | New | Module exports |
| `src/causal_engine/energy_score/score_calculator.py` | `energy_score.py` | Core calculator |
| `src/causal_engine/energy_score/estimator_selector.py` | `estimator_selection.py` | Selection logic |
| `src/causal_engine/energy_score/mlflow_tracker.py` | `mlflow_integration.py` | MLflow logging |

### Tasks
- [ ] Create `src/causal_engine/energy_score/` directory
- [ ] Copy `energy_score.py` → `score_calculator.py`
- [ ] Copy `estimator_selection.py` → `estimator_selector.py`
- [ ] Copy `mlflow_integration.py` → `mlflow_tracker.py`
- [ ] Create `__init__.py` with all exports
- [ ] Update `src/causal_engine/__init__.py` to include energy_score

### Validation
```bash
python -c "from src.causal_engine.energy_score import EnergyScoreCalculator, EstimatorSelector; print('OK')"
```

---

## Phase 2: State Updates
**Duration**: ~45 min | **Risk**: Medium | **Dependencies**: Phase 1

### Objective
Extend CausalImpactState with energy score fields.

### Files to Modify

| File | Changes |
|------|---------|
| `src/agents/causal_impact/state.py` | Add EnergyScoreData, extend EstimationResult, extend CausalImpactState |

### New TypedDicts

```python
class EnergyScoreData(TypedDict, total=False):
    score: float  # 0-1, lower is better
    treatment_balance_score: float
    outcome_fit_score: float
    propensity_calibration: float
    ci_lower: NotRequired[float]
    ci_upper: NotRequired[float]
    computation_time_ms: float
```

### EstimationResult Extensions
```python
# Add to EstimationResult:
selection_strategy: Literal["first_success", "best_energy", "ensemble"]
selected_estimator: str
energy_score: NotRequired[float]
energy_score_data: NotRequired[EnergyScoreData]
all_estimators_evaluated: NotRequired[List[Dict[str, Any]]]
selection_reason: NotRequired[str]
energy_score_gap: NotRequired[float]
```

### CausalImpactState Extensions
```python
# Add to CausalImpactState:
energy_score_enabled: NotRequired[bool]
estimator_selection_result: NotRequired[Dict[str, Any]]
energy_score_latency_ms: NotRequired[float]
```

### Validation
```bash
pytest tests/unit/test_agents/test_causal_impact/test_estimation.py -v -n 4
```

---

## Phase 3: Estimation Node Modification
**Duration**: ~90 min | **Risk**: High | **Dependencies**: Phases 1, 2

### Objective
Replace single-method estimation with EstimatorSelector.

### Files to Modify

| File | Changes |
|------|---------|
| `src/agents/causal_impact/nodes/estimation.py` | Add energy score selection path |

### Implementation

1. **Add imports**:
```python
from src.causal_engine.energy_score import (
    EstimatorSelector,
    EstimatorSelectorConfig,
    SelectionStrategy,
)
```

2. **Add new method** `_select_estimator_with_energy_score()`:
   - Extract treatment/outcome/covariates arrays
   - Configure EstimatorSelector
   - Run selection
   - Convert to EstimationResult format

3. **Modify `execute()`**:
   - Check `state.parameters.use_energy_score` (default: True)
   - Check `state.parameters.selection_strategy` (default: "best_energy")
   - If explicit `method` provided, use legacy path (backward compatibility)
   - Otherwise, use energy score selection

### Backward Compatibility
- If `state.parameters.method` is set → legacy single-method path
- If `state.parameters.use_energy_score = False` → legacy path
- Default (no parameters) → energy score selection

### Tests to Create
Create `tests/unit/test_agents/test_causal_impact/test_energy_score_selection.py`:
- `test_energy_score_selection_enabled_by_default`
- `test_legacy_mode_with_explicit_method`
- `test_first_success_strategy`
- `test_energy_score_in_result`

### Validation
```bash
pytest tests/unit/test_agents/test_causal_impact/test_energy_score_selection.py -v -n 4
pytest tests/unit/test_agents/test_causal_impact/test_estimation.py -v -n 4
```

---

## Phase 4: MLflow & DSPy Integration
**Duration**: ~60 min | **Risk**: Medium | **Dependencies**: Phase 3

### Objective
Integrate energy score metrics into MLflow and DSPy training signals.

### Files to Modify

| File | Changes |
|------|---------|
| `src/agents/causal_impact/graph.py` | Update `_extract_mlflow_metrics()`, `_extract_mlflow_result_tags()` |
| `src/agents/causal_impact/dspy_integration.py` | Add energy score fields to CausalAnalysisTrainingSignal |

### MLflow Metrics to Add
- `energy_score` - Selected estimator's score
- `energy_score_gap` - Gap between best and second-best
- `n_estimators_evaluated` - Count evaluated
- `n_estimators_succeeded` - Count successful
- `energy_score_{estimator}` - Per-estimator scores

### MLflow Tags to Add
- `selection_strategy` - "best_energy" or "first_success"
- `selected_estimator` - e.g., "causal_forest"
- `energy_score_enabled` - "true" or "false"

### DSPy Signal Fields to Add
```python
# Add to CausalAnalysisTrainingSignal:
energy_score_enabled: bool = False
selection_strategy: str = ""
selected_estimator: str = ""
energy_score: float = 0.0
energy_score_gap: float = 0.0
n_estimators_evaluated: int = 0
n_estimators_succeeded: int = 0
```

### Update compute_reward()
Add 5% weight for energy score quality: `0.05 * (1.0 - energy_score)`

### Validation
```bash
pytest tests/unit/test_agents/test_causal_impact/test_dspy_integration.py -v -n 4
```

---

## Phase 5: Database & Integration Tests
**Duration**: ~45 min | **Risk**: Medium | **Dependencies**: Phases 3, 4

### Objective
Apply database migration and create comprehensive tests.

### Files to Create/Copy

| File | Action |
|------|--------|
| `database/migrations/011_energy_score_enhancement.sql` | Copy from enhancement |
| `tests/unit/test_causal_engine/test_energy_score/conftest.py` | Create fixtures |
| `tests/unit/test_causal_engine/test_energy_score/test_score_calculator.py` | Copy/adapt |
| `tests/unit/test_causal_engine/test_energy_score/test_estimator_selector.py` | Copy/adapt |
| `tests/integration/test_causal_impact_energy_score.py` | Create |

### Database Objects Created
- **Table**: `estimator_evaluations`
- **View**: `v_estimator_performance`
- **View**: `v_energy_score_trends`
- **View**: `v_selection_comparison`
- **Function**: `log_estimator_evaluation()`

### Domain Vocabulary Updates
Add to `config/domain_vocabulary.yaml`:
- `estimator_types`: causal_forest, linear_dml, drlearner, ols, etc.
- `selection_strategies`: first_success, best_energy, ensemble
- `energy_score_variants`: standard, weighted, doubly_robust

### Testing Strategy (Memory-Safe)
```bash
# Unit tests (small batches, 4 workers max)
pytest tests/unit/test_causal_engine/test_energy_score/test_score_calculator.py -v -n 4
pytest tests/unit/test_causal_engine/test_energy_score/test_estimator_selector.py -v -n 4

# Integration tests (2 workers for heavier tests)
pytest tests/integration/test_causal_impact_energy_score.py -v -n 2

# Full regression
make test
```

### Validation
```bash
# Apply migration
psql $DATABASE_URL -f database/migrations/011_energy_score_enhancement.sql

# Verify schema
psql $DATABASE_URL -c "\d estimator_evaluations"

# Full test suite
make test
```

---

## Critical Files Summary

### Files to Create
```
src/causal_engine/energy_score/
├── __init__.py
├── score_calculator.py      # From energy_score.py
├── estimator_selector.py    # From estimator_selection.py
└── mlflow_tracker.py        # From mlflow_integration.py

tests/unit/test_causal_engine/test_energy_score/
├── conftest.py
├── test_score_calculator.py
└── test_estimator_selector.py

tests/unit/test_agents/test_causal_impact/
└── test_energy_score_selection.py

tests/integration/
└── test_causal_impact_energy_score.py

database/migrations/
└── 011_energy_score_enhancement.sql
```

### Files to Modify
```
src/agents/causal_impact/state.py              # Add energy score TypedDicts
src/agents/causal_impact/nodes/estimation.py   # Add EstimatorSelector integration
src/agents/causal_impact/graph.py              # Add MLflow metrics/tags
src/agents/causal_impact/dspy_integration.py   # Add signal fields
src/causal_engine/__init__.py                  # Export energy_score module
config/domain_vocabulary.yaml                   # Add estimator enums
```

---

## Quality Tiers Reference

| Tier | Max Score | Description | Color |
|------|-----------|-------------|-------|
| Excellent | 0.25 | High confidence | Green |
| Good | 0.45 | Reasonable confidence | Blue |
| Acceptable | 0.65 | Use with caution | Amber |
| Poor | 0.80 | Low confidence | Red |
| Unreliable | 1.00 | Results unreliable | Gray |

---

## Supported Estimators

| Estimator | Library | Default Priority |
|-----------|---------|------------------|
| causal_forest | EconML | 1 |
| linear_dml | EconML | 2 |
| drlearner | EconML | 3 |
| ols | sklearn | 10 (fallback) |

---

## Rollback Strategy

### Per-Phase Rollback
- **Phase 1-2**: Delete new files, revert state.py changes
- **Phase 3**: Set `use_energy_score=False` in config
- **Phase 4**: Revert graph.py and dspy_integration.py
- **Phase 5**: Drop `estimator_evaluations` table (no data loss risk)

### Emergency Disable
```python
# In estimation node config
parameters = {"use_energy_score": False}
```

---

## Progress Tracking

### Phase 1: Foundation ✅ COMPLETED
- [x] Create directory structure
- [x] Copy and rename source files
- [x] Create __init__.py
- [x] Update causal_engine exports
- [x] Verify imports

### Phase 2: State Updates ✅ COMPLETED
- [x] Add EnergyScoreData TypedDict
- [x] Extend EstimationResult
- [x] Extend CausalImpactState
- [x] Run existing tests (145 passed)

### Phase 3: Estimation Node ✅ COMPLETED
- [x] Add energy score imports
- [x] Implement _select_estimator_with_energy_score()
- [x] Modify execute() with conditional logic
- [x] Create test_energy_score_selection.py (18 tests)
- [x] Run tests (18 new + 15 existing = 33 passed)

### Phase 4: MLflow & DSPy ✅ COMPLETED
- [x] Update _extract_mlflow_metrics() (energy score, gap, per-estimator scores)
- [x] Update _extract_mlflow_result_tags() (selection_strategy, selected_estimator, quality_tier)
- [x] Add DSPy signal fields (7 new fields)
- [x] Update compute_reward() (5% energy score weight)
- [x] Run tests (28 passed)

### Phase 5: Database & Integration ✅ COMPLETED
- [x] Create database migration (database/ml/023_energy_score_tables.sql)
- [x] Create enums: estimator_type, selection_strategy, quality_tier
- [x] Create table: estimator_evaluations
- [x] Create views: v_estimator_performance, v_energy_score_trends, v_selection_comparison
- [x] Create function: log_estimator_evaluation()
- [x] Run full causal impact test suite (163 passed)

---

## Documentation Updated

The following documentation has already been updated (prior to this plan):
- `.claude/specialists/Agent_Specialists_Tiers 1-5/causal-impact.md`
- `.claude/contracts/Tier-Specific Contracts/tier2-contracts.md`

---

## Notes

- **Memory Safety**: All test commands use `-n 4` max workers
- **DSPy Tests**: Use `@pytest.mark.xdist_group(name="dspy_integration")` marker
- **Backward Compatibility**: Legacy `first_success` strategy remains available
- **Context Window**: Phases designed to complete in single sessions
