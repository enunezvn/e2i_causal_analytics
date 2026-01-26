---
name: Experiment Validity Threat Assessment
version: 1.0
description: 6-threat taxonomy for experiment design validation
triggers:
  - validity threats
  - experiment validation
  - internal validity
  - external validity
  - selection bias
  - confounding assessment
agents:
  - experiment_designer
  - causal_impact
categories:
  - methodology
  - validation
---

# Experiment Validity Threat Assessment

## 6-Threat Taxonomy

### 1. Selection Bias

**Definition**: Non-random assignment creates pre-existing differences between groups.

**Manifestations in Pharma**:
- High-volume HCPs self-select into programs
- Early adopters differ from mainstream
- Healthier patients enroll in studies

**Detection Methods**:
- Compare baseline characteristics (t-tests, chi-square)
- Check balance on key confounders
- Examine enrollment patterns

**Mitigations**:
| Mitigation | When to Use |
|------------|-------------|
| Randomization | Gold standard, when feasible |
| Stratified randomization | When key confounders are known |
| Matching | When randomization isn't possible |
| Propensity score weighting | Observational settings |
| Regression discontinuity | Natural thresholds exist |

**Mitigation Specificity Required**: Don't just say "randomize." Specify:
- Randomization unit (patient, HCP, territory)
- Stratification variables
- Block size
- Allocation ratio

---

### 2. Confounding

**Definition**: Unmeasured variables affect both treatment and outcome.

**Manifestations in Pharma**:
- Territory potential affects targeting AND outcomes
- HCP motivation affects engagement AND prescribing
- Market dynamics affect interventions AND results

**Detection Methods**:
- Sensitivity analysis (E-value)
- Negative control outcomes
- Instrumental variable tests

**Mitigations**:
| Mitigation | When to Use |
|------------|-------------|
| Measure and adjust | Confounder is measurable |
| Instrumental variables | Valid instrument available |
| Difference-in-differences | Pre-post data available |
| Regression discontinuity | Sharp threshold exists |
| Synthetic controls | Single treated unit |

---

### 3. Measurement Error

**Definition**: Outcome or treatment is measured with error.

**Manifestations in Pharma**:
- Data source lag creates incomplete outcomes
- Attribution windows miss delayed effects
- Self-reported outcomes have bias

**Detection Methods**:
- Compare across data sources
- Test-retest reliability
- Examine measurement timing

**Mitigations**:
| Mitigation | When to Use |
|------------|-------------|
| Multiple data sources | Cross-validation possible |
| Longer observation windows | Effects may be delayed |
| Objective outcomes | Avoid self-report bias |
| Standardized definitions | Ensure consistency |

---

### 4. Contamination

**Definition**: Control group is exposed to treatment.

**Manifestations in Pharma**:
- Control HCPs learn from treated colleagues
- Patients switch between HCPs
- National campaigns affect both groups

**Detection Methods**:
- Measure treatment exposure in control
- Geographic spillover analysis
- Network analysis for information spread

**Mitigations**:
| Mitigation | When to Use |
|------------|-------------|
| Geographic separation | Clusters are distinct |
| Cluster randomization | Prevent within-cluster spillover |
| Waitlist control | Eventual treatment for all |
| Intent-to-treat analysis | Despite contamination |

---

### 5. Temporal Effects

**Definition**: Time-varying factors affect outcomes independent of treatment.

**Manifestations in Pharma**:
- Seasonality in respiratory conditions
- Competitor launches during study
- Policy changes affecting access
- COVID-19 effects on healthcare

**Detection Methods**:
- Examine outcome trends pre-intervention
- Identify concurrent events
- Test for time × treatment interactions

**Mitigations**:
| Mitigation | When to Use |
|------------|-------------|
| Concurrent control | Separates time from treatment |
| Difference-in-differences | Pre-post comparison |
| Interrupted time series | Long baseline available |
| Event study design | Sharp intervention timing |

---

### 6. Attrition

**Definition**: Differential dropout between groups.

**Manifestations in Pharma**:
- Non-responders drop out faster
- Side effects cause selective attrition
- Loss to follow-up in rare disease

**Detection Methods**:
- Compare attrition rates between groups
- Analyze attrition by baseline characteristics
- Test for differential attrition on outcomes

**Mitigations**:
| Mitigation | When to Use |
|------------|-------------|
| Intent-to-treat analysis | Primary analysis |
| Multiple imputation | Handle missing data |
| Sensitivity analysis | Bound effects under attrition |
| Per-protocol secondary | Complier effects |

---

## Validity Scoring Framework

### For Each Threat, Assess:

1. **Likelihood** (1-5)
   - 1: Very unlikely
   - 3: Possible
   - 5: Very likely

2. **Severity** (1-5)
   - 1: Minor bias
   - 3: Moderate bias
   - 5: Severe bias, could invalidate results

3. **Mitigation Quality** (1-5)
   - 1: No mitigation
   - 3: Partial mitigation
   - 5: Fully mitigated

### Validity Score Calculation

```
Threat Score = Likelihood × Severity / Mitigation Quality

Overall Validity = Σ(Threat Scores) / Number of Threats
```

### Interpretation

| Overall Score | Validity | Recommendation |
|---------------|----------|----------------|
| < 5 | Strong | Proceed |
| 5-10 | Moderate | Proceed with caution |
| 10-15 | Weak | Redesign recommended |
| > 15 | Unacceptable | Do not proceed |

---

## Validity Audit Template

```markdown
## Validity Threat Assessment: [Experiment Name]

### 1. Selection Bias
- Likelihood: [1-5] - [Rationale]
- Severity: [1-5] - [Rationale]
- Mitigation: [Description]
- Mitigation Quality: [1-5]
- Residual Concern: [None/Low/Medium/High]

### 2. Confounding
[Same structure]

### 3. Measurement Error
[Same structure]

### 4. Contamination
[Same structure]

### 5. Temporal Effects
[Same structure]

### 6. Attrition
[Same structure]

### Overall Assessment
- Validity Score: [X]
- Classification: [Strong/Moderate/Weak]
- Recommendation: [Proceed/Caution/Redesign]
- Key Risks: [Top 2-3 concerns]
```

---

## Quick Reference: Threat-Mitigation Matrix

| Threat | Primary Mitigation | Secondary Mitigation |
|--------|-------------------|---------------------|
| Selection Bias | Randomization | Propensity matching |
| Confounding | Measure & adjust | IV/DiD |
| Measurement Error | Multiple sources | Longer windows |
| Contamination | Geographic separation | Cluster randomization |
| Temporal Effects | Concurrent control | DiD |
| Attrition | ITT analysis | Multiple imputation |
