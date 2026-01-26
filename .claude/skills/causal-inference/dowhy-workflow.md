---
name: DoWhy Causal Estimation Workflow
version: 1.0
description: End-to-end DoWhy/EconML workflow with pharma-specific procedures
triggers:
  - causal estimation
  - DoWhy analysis
  - effect estimation
  - causal impact
  - ATE calculation
  - CATE analysis
  - treatment effect
agents:
  - causal_impact
  - experiment_designer
categories:
  - estimation
  - refutation
  - sensitivity
---

# DoWhy Causal Estimation Workflow

## Overview

```
Query → DAG Construction → Estimation → Refutation → Sensitivity → Interpretation
```

---

## Phase 1: DAG Construction

### Input Requirements
- Treatment variable (what we're testing)
- Outcome variable (what we're measuring)
- Identified confounders (from confounder-identification.md)
- Time ordering (cause must precede effect)

### DoWhy Graph Specification

```python
from dowhy import CausalModel

# Build causal graph
model = CausalModel(
    data=df,
    treatment='hcp_targeting_intensity',
    outcome='nrx_volume',
    common_causes=['territory_potential', 'hcp_specialty',
                   'baseline_volume', 'payer_mix'],
    instruments=['rep_turnover'],  # If using IV
    effect_modifiers=['hcp_segment'],  # For CATE analysis
)
```

### Graph Validation Checklist

- [ ] All confounders are pre-treatment
- [ ] No colliders included
- [ ] No mediators (unless mediation analysis)
- [ ] Time ordering is correct
- [ ] Effect modifiers identified for heterogeneity

---

## Phase 2: Estimation with Fallback Chain

### Energy Score Selection (V4.2)

Instead of first-success, use energy score to select best estimator:

```
Energy Score = 0.35 × Treatment Balance + 0.45 × Outcome Fit + 0.20 × Propensity Calibration
```

### Quality Tiers

| Score Range | Quality | Action |
|-------------|---------|--------|
| ≤0.25 | Excellent | Use with high confidence |
| ≤0.45 | Good | Use with normal confidence |
| ≤0.65 | Acceptable | Use with caveats |
| ≤0.80 | Poor | Consider alternative methods |
| >0.80 | Unreliable | Do not use without redesign |

### Estimator Priority Chain

Run all estimators, score each, select best:

#### 1. Causal Forest DML (EconML)
Best for heterogeneous effects

```python
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

model_y = GradientBoostingRegressor(n_estimators=100, max_depth=5)
model_t = GradientBoostingClassifier(n_estimators=100, max_depth=5)

cf = CausalForestDML(model_y=model_y, model_t=model_t)
cf.fit(Y, T, X=effect_modifiers, W=confounders)
ate = cf.ate(X)
cate = cf.effect(X)
```

#### 2. Linear DML
Good for linear relationships

```python
from econml.dml import LinearDML

ldml = LinearDML(model_y=model_y, model_t=model_t)
ldml.fit(Y, T, X=effect_modifiers, W=confounders)
ate = ldml.ate(X)
ci = ldml.ate_interval(X, alpha=0.05)
```

#### 3. Backdoor Linear Regression
Simple, interpretable

```python
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression",
    confidence_intervals=True,
)
```

#### 4. Propensity Score Weighting
When overlap is good

```python
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_weighting",
    method_params={"weighting_scheme": "ips_weight"}
)
```

### Positivity Check

Before estimation, verify treatment overlap:

```python
from sklearn.linear_model import LogisticRegression

# Estimate propensity scores
ps_model = LogisticRegression()
ps_model.fit(confounders, treatment)
propensity = ps_model.predict_proba(confounders)[:, 1]

# Check overlap
overlap_ok = (propensity.min() > 0.05) and (propensity.max() < 0.95)
```

If overlap is poor, consider:
- Trimming extreme propensity scores
- Using matching instead of weighting
- Restricting to common support region

---

## Phase 3: Refutation Testing

Run ALL refutation tests in parallel:

### 1. Placebo Treatment Test

```python
res_placebo = model.refute_estimate(
    identified_estimand, estimate,
    method_name="placebo_treatment_refuter"
)
# Effect should be ~0 with random treatment
```

**Pass criteria**: |effect| < 0.1 × original effect

### 2. Random Common Cause Test

```python
res_random = model.refute_estimate(
    identified_estimand, estimate,
    method_name="random_common_cause"
)
# Effect should be stable with random confounder
```

**Pass criteria**: |change| < 10% of original effect

### 3. Data Subset Test

```python
res_subset = model.refute_estimate(
    identified_estimand, estimate,
    method_name="data_subset_refuter",
    subset_fraction=0.8
)
# Effect should be stable across subsets
```

**Pass criteria**: Effect within original 95% CI

### 4. Bootstrap Validation

```python
res_bootstrap = model.refute_estimate(
    identified_estimand, estimate,
    method_name="bootstrap_refuter",
    num_simulations=100
)
```

### Refutation Summary

| Test | Pass Condition | Fail Action |
|------|----------------|-------------|
| Placebo | Effect ≈ 0 | Check for data leakage |
| Random cause | Stable | Add more confounders |
| Subset | Stable | Check for outliers |
| Bootstrap | Narrow CI | Get more data |

---

## Phase 4: Sensitivity Analysis

### E-value Calculation

```python
import numpy as np

def calculate_e_value(rr):
    """Calculate E-value for a risk ratio."""
    if rr >= 1:
        e_value = rr + np.sqrt(rr * (rr - 1))
    else:
        rr_inv = 1 / rr
        e_value = rr_inv + np.sqrt(rr_inv * (rr_inv - 1))
    return e_value

# For the estimate
rr = np.exp(estimate.value)  # If log scale
e_value = calculate_e_value(rr)
```

### Interpretation Framework

| E-value | Interpretation |
|---------|----------------|
| < 1.5 | Easily explained by unmeasured confounding |
| 1.5-2.0 | Moderate robustness |
| 2.0-3.0 | Good robustness |
| > 3.0 | Strong robustness |

### Benchmark Against Observed Confounders

Compare E-value to strongest observed confounder:

```
If E-value > max(observed confounder effects) × 1.5:
    → Effect is likely robust
Else:
    → Unmeasured confounding is plausible concern
```

---

## Phase 5: Interpretation by Audience

### For Executives

```markdown
**Key Finding**: [Treatment] causes a [X%] increase in [Outcome].

**Business Impact**: This represents [$Y] in additional revenue.

**Confidence**: [High/Medium/Low] based on robustness checks.

**Recommendation**: [Specific action to take].
```

### For Analysts

```markdown
**Causal Effect**: ATE = [X] (95% CI: [lower, upper])

**Method**: [Estimator used] selected via energy score ([score])

**Robustness**: [X/4] refutation tests passed

**Sensitivity**: E-value = [Y], suggesting [interpretation]

**Limitations**: [Key assumption concerns]
```

### For Data Scientists

```markdown
**Estimation**: ATE = [X] ± [SE], p < [p-value]

**Estimator**: [Method] with energy score [score]
  - Treatment balance: [score component]
  - Outcome fit: [score component]
  - Propensity calibration: [score component]

**Identification**: Backdoor criterion satisfied via [confounders]

**Refutation Results**:
  - Placebo: effect = [X] (expected ~0)
  - Random cause: Δ = [X]% (expected <10%)
  - Subset: within CI = [Yes/No]
  - Bootstrap: SE = [X]

**Sensitivity**: E-value = [Y]
  - Required confounder strength: RR > [Z]
  - Observed max confounder: RR = [W]
  - Assessment: [Robust/Concerning]

**Assumptions at risk**: [Specific concerns]
```

---

## Validation Gates (V4.1)

Before proceeding to interpretation, verify:

| Gate | Criterion | Action if Failed |
|------|-----------|------------------|
| Positivity | Propensity ∈ [0.05, 0.95] | Trim or match |
| Refutation | ≥3/4 tests pass | Investigate failures |
| E-value | > 1.5 | Report with strong caveats |
| Energy score | ≤ 0.65 | Consider alternative design |

---

## Common Estimation Pitfalls

### 1. Collider Bias

**Problem**: Adjusting for a variable caused by both treatment and outcome

**Detection**: Review DAG for collider structures

**Fix**: Remove collider from adjustment set

### 2. Mediator Adjustment

**Problem**: Adjusting for mediator blocks causal path

**Detection**: Check if variable is post-treatment

**Fix**: Remove mediator unless doing mediation analysis

### 3. Positivity Violations

**Problem**: Some covariate combinations have no treated/untreated units

**Detection**: Check propensity score distribution

**Fix**: Trim sample or use matching

### 4. Model Misspecification

**Problem**: Functional form assumptions violated

**Detection**: Residual plots, cross-validation

**Fix**: Use flexible estimators (forest, ensemble)

---

## Code Templates

### Full DoWhy Pipeline

```python
import dowhy
from dowhy import CausalModel
import pandas as pd

async def run_causal_analysis(df, treatment, outcome, confounders):
    """Run full DoWhy causal analysis pipeline."""

    # Phase 1: Build model
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=confounders,
    )

    # Identify estimand
    estimand = model.identify_effect()

    # Phase 2: Estimate
    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
        confidence_intervals=True,
    )

    # Phase 3: Refute
    refutations = {
        'placebo': model.refute_estimate(estimand, estimate,
                    method_name="placebo_treatment_refuter"),
        'random_cause': model.refute_estimate(estimand, estimate,
                    method_name="random_common_cause"),
        'subset': model.refute_estimate(estimand, estimate,
                    method_name="data_subset_refuter"),
    }

    # Phase 4: Sensitivity
    e_value = calculate_e_value(estimate.value)

    return {
        'estimate': estimate,
        'refutations': refutations,
        'e_value': e_value,
    }
```

### EconML CATE Analysis

```python
from econml.dml import CausalForestDML

async def estimate_heterogeneous_effects(df, Y, T, X, W):
    """Estimate conditional average treatment effects."""

    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(),
        model_t=GradientBoostingClassifier(),
        n_estimators=100,
    )

    cf.fit(Y, T, X=X, W=W)

    # Overall ATE
    ate = cf.ate(X)
    ate_ci = cf.ate_interval(X, alpha=0.05)

    # Individual CATEs
    cate = cf.effect(X)
    cate_ci = cf.effect_interval(X, alpha=0.05)

    # Feature importance for heterogeneity
    importance = cf.feature_importances_

    return {
        'ate': ate,
        'ate_ci': ate_ci,
        'cate': cate,
        'cate_ci': cate_ci,
        'importance': importance,
    }
```
