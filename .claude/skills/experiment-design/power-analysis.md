---
name: Power Analysis and Sample Size Calculation
version: 1.0
description: Sample size calculation procedures for pharmaceutical commercial experiments
triggers:
  - sample size
  - power analysis
  - statistical power
  - minimum detectable effect
  - MDE calculation
  - experiment sizing
agents:
  - experiment_designer
categories:
  - methodology
  - statistics
---

# Power Analysis and Sample Size Calculation

## Overview

Power analysis determines the sample size needed to detect a meaningful effect with acceptable confidence. In pharmaceutical commercial contexts, this ensures experiments are adequately powered while balancing practical constraints.

---

## Key Parameters

### 1. Significance Level (α)
- **Standard**: 0.05 (5% false positive rate)
- **Conservative**: 0.01 (1% for high-stakes decisions)
- **Exploratory**: 0.10 (10% for pilot studies)

### 2. Statistical Power (1-β)
- **Standard**: 0.80 (80% chance to detect true effect)
- **High**: 0.90 (90% for critical experiments)
- **Minimum**: 0.70 (acceptable for pilots)

### 3. Minimum Detectable Effect (MDE)
The smallest effect size worth detecting. Must be:
- Practically meaningful
- Achievable based on historical benchmarks
- Worth the investment to implement

### 4. Baseline Metrics
| Metric | Typical Baseline | Variance |
|--------|-----------------|----------|
| TRx per HCP | 10-50/month | CV 0.5-1.0 |
| NRx conversion | 5-15% | SD 3-5% |
| Market share | 10-40% | SD 5-10% |
| Adherence (PDC) | 60-80% | SD 15-20% |

---

## Sample Size Formulas

### Two-Sample t-test (Continuous Outcome)

```python
import scipy.stats as stats
import numpy as np

def sample_size_ttest(mde, sd, alpha=0.05, power=0.80):
    """
    Calculate sample size for two-sample t-test.

    Args:
        mde: Minimum detectable effect (absolute difference)
        sd: Standard deviation of outcome
        alpha: Significance level (default 0.05)
        power: Statistical power (default 0.80)

    Returns:
        n: Sample size per group
    """
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)

    n = 2 * ((z_alpha + z_beta) * sd / mde) ** 2
    return int(np.ceil(n))
```

### Two-Sample Proportion Test (Binary Outcome)

```python
def sample_size_proportion(p1, p2, alpha=0.05, power=0.80):
    """
    Calculate sample size for two-proportion z-test.

    Args:
        p1: Control proportion
        p2: Treatment proportion (p1 + expected lift)
        alpha: Significance level
        power: Statistical power

    Returns:
        n: Sample size per group
    """
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)

    p_pooled = (p1 + p2) / 2

    n = ((z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) +
          z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) /
         (p2 - p1)) ** 2

    return int(np.ceil(n))
```

### Cluster Randomized Trial

```python
def sample_size_cluster(mde, sd, icc, cluster_size, alpha=0.05, power=0.80):
    """
    Calculate number of clusters for cluster randomized trial.

    Args:
        mde: Minimum detectable effect
        sd: Standard deviation of outcome
        icc: Intraclass correlation coefficient
        cluster_size: Average size per cluster (e.g., patients per HCP)
        alpha: Significance level
        power: Statistical power

    Returns:
        k: Number of clusters per group
    """
    # Design effect for clustering
    deff = 1 + (cluster_size - 1) * icc

    # Individual-level sample size
    n_individual = sample_size_ttest(mde, sd, alpha, power)

    # Inflate for clustering
    n_cluster_total = n_individual * deff

    # Number of clusters
    k = int(np.ceil(n_cluster_total / cluster_size))

    return k
```

---

## Pharma-Specific Considerations

### Typical Intraclass Correlations (ICC)

| Clustering Level | Typical ICC | Range |
|-----------------|-------------|-------|
| Patients within HCP | 0.05-0.15 | 0.01-0.25 |
| HCPs within territory | 0.02-0.10 | 0.01-0.20 |
| Territories within region | 0.01-0.05 | 0.00-0.10 |

### Expected Effect Sizes (Historical Benchmarks)

| Intervention | Expected Lift | Historical Range |
|--------------|--------------|------------------|
| HCP targeting pilot | 10-20% NRx | 5-25% |
| Digital engagement | 5-15% reach | 3-20% |
| Patient support program | 5-10% persistence | 3-15% |
| MSL engagement | 8-15% advocacy | 5-20% |
| Rep call frequency increase | 3-8% TRx | 2-12% |

### Variance Inflation Factors

Apply these multipliers to account for real-world complications:

| Factor | Multiplier | When to Apply |
|--------|------------|---------------|
| Missing data (10%) | 1.1 | Standard |
| Missing data (20%) | 1.25 | High attrition expected |
| Clustering ignored | 1.5-2.0 | Cluster design but analyzed individually |
| Multiple comparisons | 1.3-1.5 | Multiple outcomes |
| Interim analyses | 1.05-1.1 | Planned interim looks |

---

## Power Analysis Decision Tree

```
                    ┌─────────────────────────────────────┐
                    │    What is your primary outcome?    │
                    └─────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                  ▼
            ┌───────────────┐                ┌───────────────┐
            │  Continuous   │                │    Binary     │
            │  (TRx, NRx)   │                │  (conversion) │
            └───────────────┘                └───────────────┘
                    │                                  │
                    ▼                                  ▼
            Use t-test formula               Use proportion formula
                    │                                  │
                    └────────────────┬────────────────┘
                                     ▼
                    ┌─────────────────────────────────────┐
                    │    Is the design clustered?         │
                    └─────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                  ▼
                  Yes                                 No
                    │                                  │
                    ▼                                  ▼
            Apply design effect              Use individual-level
            (1 + (m-1) × ICC)                sample size
```

---

## Sample Size Table (Quick Reference)

### Continuous Outcomes (80% power, α=0.05)

| Effect Size (d) | SD Ratio | N per Group |
|-----------------|----------|-------------|
| 0.2 (small) | 1.0 | 394 |
| 0.3 | 1.0 | 176 |
| 0.5 (medium) | 1.0 | 64 |
| 0.8 (large) | 1.0 | 26 |

### Binary Outcomes (80% power, α=0.05)

| Baseline Rate | Lift | N per Group |
|---------------|------|-------------|
| 10% | 5% (→15%) | 407 |
| 10% | 10% (→20%) | 131 |
| 20% | 5% (→25%) | 619 |
| 20% | 10% (→30%) | 172 |
| 30% | 10% (→40%) | 203 |
| 30% | 15% (→45%) | 91 |

---

## MDE Calculation (Reverse Power Analysis)

When sample size is constrained, calculate the minimum detectable effect:

```python
def mde_from_sample(n, sd, alpha=0.05, power=0.80):
    """
    Calculate MDE given fixed sample size.

    Args:
        n: Sample size per group
        sd: Standard deviation of outcome
        alpha: Significance level
        power: Statistical power

    Returns:
        mde: Minimum detectable effect
    """
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)

    mde = (z_alpha + z_beta) * sd * np.sqrt(2/n)
    return mde
```

---

## Output Format

### Power Analysis Report Template

```markdown
## Power Analysis: [Experiment Name]

### Study Parameters
- **Primary Outcome**: [Metric]
- **Study Design**: [Individual/Cluster RCT]
- **Significance Level (α)**: [0.05]
- **Statistical Power (1-β)**: [0.80]

### Baseline Assumptions
- **Control Rate/Mean**: [X]
- **Standard Deviation**: [Y] (or variance [Z])
- **Historical Effect Size**: [Reference]

### Effect Size Justification
- **MDE**: [X%] ([Absolute value])
- **Rationale**: [Why this effect is meaningful and achievable]
- **Historical Benchmark**: [Prior experiment or literature]

### Sample Size Calculation
- **Formula Used**: [t-test/proportion/cluster]
- **Raw Calculation**: [N] per group
- **Inflation Factors Applied**:
  - [Factor 1]: [Multiplier]
  - [Factor 2]: [Multiplier]
- **Final Sample Size**: [N] per group, [2N] total

### Clustering Adjustment (if applicable)
- **ICC**: [X]
- **Cluster Size**: [Y]
- **Design Effect**: [Z]
- **Number of Clusters**: [K] per group

### Sensitivity Analysis
| Power | MDE | Sample Size |
|-------|-----|-------------|
| 70% | [X] | [N] |
| 80% | [X] | [N] |
| 90% | [X] | [N] |

### Feasibility Assessment
- **Available Population**: [N]
- **Required Sample**: [N]
- **Feasibility**: [Yes/No/Marginal]
- **Recommendations**: [If not feasible, alternatives]
```

---

## Common Pitfalls

### 1. Ignoring Clustering
**Problem**: Analyzing clustered data as if independent
**Impact**: Type I error rate inflates (false positives)
**Solution**: Always check for clustering and apply design effect

### 2. Optimistic Effect Sizes
**Problem**: Assuming larger effects than realistic
**Impact**: Underpowered study
**Solution**: Use conservative historical benchmarks

### 3. Ignoring Multiple Comparisons
**Problem**: Testing many outcomes without correction
**Impact**: Inflated false positive rate
**Solution**: Adjust α or pre-specify primary outcome

### 4. Not Planning for Attrition
**Problem**: Sample depletes during study
**Impact**: Underpowered final analysis
**Solution**: Inflate sample size for expected dropout
