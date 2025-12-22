# Phase 11: Scope Definer Agent Completion

**Goal**: Complete ML problem scoping

**Status**: Not Started

**Dependencies**: None (can run in parallel)

---

## Tasks

- [ ] **Task 11.1**: Complete `nodes/problem_analyzer.py`
  - Parse user request
  - Classify problem type (classification/regression/causal)
  - Extract target variable

- [ ] **Task 11.2**: Complete `nodes/data_assessor.py`
  - Check data availability
  - Assess data quality
  - Identify potential features

- [ ] **Task 11.3**: Complete `nodes/scope_generator.py`
  - Generate scope document
  - Define success metrics
  - Estimate complexity

- [ ] **Task 11.4**: Wire up LangGraph flow
  - Define state transitions
  - Add validation checkpoints

- [ ] **Task 11.5**: Add scope validation tests
  - Test problem classification
  - Test data assessment
  - Test scope generation

- [ ] **Task 11.6**: Document scope definition patterns
  - Best practices
  - Example scope documents
  - Common patterns

---

## Files to Modify

| File | Action | Description |
|------|--------|-------------|
| `src/agents/ml_foundation/scope_definer/nodes/problem_analyzer.py` | Modify | Add analysis |
| `src/agents/ml_foundation/scope_definer/nodes/data_assessor.py` | Modify | Add assessment |
| `src/agents/ml_foundation/scope_definer/nodes/scope_generator.py` | Modify | Add generation |
| `src/agents/ml_foundation/scope_definer/graph.py` | Modify | Wire up flow |
| `tests/` | Create | Tests |

---

## Problem Types

### Classification
- Binary: Churn prediction, HCP response
- Multi-class: Segment classification

### Regression
- Continuous: Sales forecasting, share prediction
- Time series: Trend prediction

### Causal Inference
- ATE: Average treatment effect
- CATE: Conditional average treatment effect
- Counterfactual: What-if analysis

---

## Output Contract

```python
class ScopeDefinition(TypedDict):
    problem_type: Literal["classification", "regression", "causal"]
    target_variable: str
    feature_candidates: List[str]
    data_requirements: Dict[str, Any]
    success_metrics: List[str]
    complexity_estimate: Literal["low", "medium", "high"]
    estimated_data_rows: int
    constraints: List[str]
```

---

## Example Scope Document

```yaml
scope:
  name: "HCP Churn Prediction"
  problem_type: "classification"
  target: "churned"
  features:
    - engagement_score
    - last_visit_days
    - prescription_volume
    - tier
  success_metrics:
    - "AUC > 0.80"
    - "Precision > 0.70"
  constraints:
    - "No PHI data"
    - "Regional compliance"
```

---

## Progress Log

| Date | Update |
|------|--------|
| 2024-12-22 | Phase created |

---

## Blockers

None currently.
