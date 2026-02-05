---
name: DoWhy Causal Estimation Workflow
version: "1.0"
description: End-to-end DoWhy causal estimation workflow for pharma analytics
triggers:
  - causal estimation
  - DoWhy analysis
  - ATE calculation
  - CATE analysis
  - causal effect estimation
agents:
  - causal_impact
categories:
  - causal-inference
---

# DoWhy Causal Estimation Workflow

Complete workflow for causal estimation using DoWhy and EconML.

## Phase 1: DAG Construction

Build the causal directed acyclic graph (DAG) from domain knowledge.

### Steps

1. Identify treatment, outcome, and confounder variables
2. Define causal relationships
3. Validate DAG with domain experts

### Code Template

```python
import dowhy
from dowhy import CausalModel

model = CausalModel(
    data=df,
    treatment='treatment_var',
    outcome='outcome_var',
    graph=dag_string
)
```

## Phase 2: Identification

Identify the causal estimand using the DAG.

## Phase 3: Estimation

Estimate the causal effect using appropriate estimators.

### Energy Score Weighting

The Energy Score combines multiple metrics for estimator selection:
- Treatment balance weight: 0.35
- Outcome fit weight: 0.45
- Complexity penalty: 0.20

### CATE Estimation

For heterogeneous treatment effects, use CausalForestDML:

```python
from econml.dml import CausalForestDML

est = CausalForestDML(
    model_y='auto',
    model_t='auto',
    n_estimators=200
)
```

## Phase 4: Refutation Testing

Test the robustness of estimates using refutation methods.

### Available Refuters

- **Placebo treatment**: Replace treatment with random placebo variable
- **random_common_cause**: Add random common cause to check sensitivity
- **Data subset**: Test on random subsets of data
- **Bootstrap**: Bootstrap confidence intervals

## Phase 5: Sensitivity Analysis

Assess sensitivity to unmeasured confounding.

### E-value Calculation

The E-value quantifies the minimum strength of unmeasured confounding needed to explain away the effect.

### Rosenbaum Bounds

Test sensitivity under varying levels of hidden bias.

## Phase 6: Interpretation

Present results for different audiences.

### Executive Summary

High-level findings for leadership:
- Effect size and direction
- Confidence level
- Business implications

### Analyst Report

Detailed findings for analytics team:
- Point estimates and confidence intervals
- Robustness check results
- Sensitivity analysis findings

### Data Scientist Deep Dive

Technical details for the data science team:
- Estimator comparison
- Model diagnostics
- Assumption verification
