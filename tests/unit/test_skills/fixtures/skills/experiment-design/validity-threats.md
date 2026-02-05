---
name: Experiment Validity Threat Assessment
version: "1.0"
description: Framework for assessing validity threats in pharma experiments
triggers:
  - validity threats
  - experiment validation
  - internal validity
  - external validity
  - threat assessment
agents:
  - experiment_designer
categories:
  - experiment-design
---

# Experiment Validity Threat Assessment

Framework for identifying and mitigating validity threats in pharmaceutical experiments.

## Threat Taxonomy

### 1. Selection Bias

Non-random assignment of HCP or patient groups.

#### Pharma Manifestations

- Higher-volume HCPs more likely to be targeted
- Patient self-selection into treatment
- Geographic clustering of enrollment

#### Mitigation Strategies

- Randomization of treatment assignment
- Propensity score matching
- Stratified sampling

### 2. Confounding

Unmeasured variables affecting both treatment and outcome.

#### Pharma Manifestations

- Territory potential confounding targeting effects
- HCP specialty confounding prescribing patterns
- Patient severity confounding treatment outcomes

#### Mitigation

- DAG-based confounder identification
- Instrumental variable approaches

### 3. Measurement Error

Inaccurate measurement of treatment or outcome variables.

#### Pharma Manifestations

- Claims data lag
- Incomplete capture of free samples
- Attribution window misspecification

#### Mitigation

- Multiple data source validation
- Sensitivity analysis on measurement windows

### 4. Contamination

Cross-group exposure between treatment and control.

#### Pharma Manifestations

- HCPs in control group exposed to peer influence
- Patient switching between treatment arms
- Digital marketing spillover

#### Mitigation

- Geographic separation of arms
- Intent-to-treat analysis

### 5. Temporal Effects

Time-related threats to validity.

#### Pharma Manifestations

- Seasonal prescription patterns
- Formulary change timing
- Product launch effects

#### Mitigation

- Time-series decomposition
- Pre-post comparison with adequate washout

### 6. Attrition

Differential dropout from the study.

#### Pharma Manifestations

- HCP disengagement from program
- Patient discontinuation
- Territory realignment during study

#### Mitigation

- Intent-to-treat analysis
- Sensitivity analysis for missing data

## Validity Score Framework

### Scoring Dimensions

- **Likelihood**: Probability that the threat is present (1-5)
- **Severity**: Impact on conclusions if threat is realized (1-5)
- **Detectability**: Ability to detect the threat (1-5)

### Validity Score Calculation

Validity Score = Likelihood * Severity * (6 - Detectability)

### Interpretation

- Score < 20: Low risk
- Score 20-50: Moderate risk, mitigation recommended
- Score > 50: High risk, mitigation required
