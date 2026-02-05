---
name: Power Analysis and Sample Size Calculation
version: "1.0"
description: Power analysis and sample size calculation for pharma experiments
triggers:
  - sample size
  - power analysis
  - statistical power
  - minimum detectable effect
  - MDE calculation
agents:
  - experiment_designer
categories:
  - experiment-design
---

# Power Analysis and Sample Size Calculation

Procedures for calculating statistical power and required sample sizes for pharmaceutical experiments.

## Sample Size Formulas

### Two-Sample t-test

```python
def sample_size_ttest(effect_size, alpha=0.05, power=0.80):
    """Calculate sample size for two-sample t-test."""
    from scipy import stats
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n))
```

### Proportion Test

For binary outcomes (e.g., conversion rates):

```python
def sample_size_proportion(p1, p2, alpha=0.05, power=0.80):
    """Calculate sample size for proportion test."""
    pass
```

## Cluster Randomization

When randomizing at the HCP or territory level instead of patient level.

### Design Effect

Design Effect = 1 + (m - 1) * ICC

Where:
- m = average cluster size
- ICC = intra-cluster correlation coefficient

### ICC Benchmarks for Pharma

- HCP-level clustering: ICC ≈ 0.02-0.05
- Territory-level clustering: ICC ≈ 0.05-0.15

## Pharma-Specific Benchmarks

### Typical Effect Sizes

| Metric | Small | Medium | Large |
|--------|-------|--------|-------|
| TRx lift | 2-5% | 5-10% | >10% |
| NRx lift | 3-8% | 8-15% | >15% |
| Market share | 0.5-1pp | 1-2pp | >2pp |

## MDE (Minimum Detectable Effect)

The minimum detectable effect is the smallest effect size that can be detected with specified power.

### MDE Calculation

MDE = (z_alpha + z_beta) * sqrt(2 * sigma^2 / n)

### Practical Significance

Always consider whether the MDE is practically meaningful for business decisions.
