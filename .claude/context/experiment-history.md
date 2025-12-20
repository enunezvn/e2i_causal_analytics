# Experiment History

**Last Updated**: 2025-12-18
**Update Cadence**: After each experiment completion or monthly review (whichever comes first)
**Owner**: Experiment Design Team
**Purpose**: Organizational learning and experiment design guidance

## Overview

This document catalogs historical experiments conducted through the E2I Causal Analytics platform. This history informs the Experiment Designer agent's organizational learning context and helps avoid repeating known pitfalls.

**When to Update This Document:**
1. **After experiment completion**: Add final results, learnings, and recommendations
2. **After interim analysis**: Update interim results and conditional power calculations
3. **Monthly review**: Check for pending updates, refresh organizational defaults
4. **New experiment launch**: Add experiment entry with design details
5. **Pitfall discovery**: Add to Common Pitfalls section when new issues identified

**Maintenance Guidelines:**
- Keep experiment entries in chronological order (by start date)
- Archive experiments older than 3 years to separate document
- Update organizational defaults annually (January)
- Cross-reference MLflow experiment IDs for technical details

---

## MLflow Integration

**Relationship to MLflow Experiment Tracking:**

This document serves a **complementary but distinct purpose** from MLflow experiment tracking:

### This Document (experiment-history.md)
**Purpose**: Business context and organizational learning
**Focus**:
- High-level experiment design and results
- Strategic learnings and pitfalls
- Organizational knowledge accumulation
- Recommendation for future experiments

**Content**:
- Experiment hypotheses and business rationale
- Design decisions (randomization, stratification, power analysis)
- Business outcome metrics (NRx, TRx, adherence, market share)
- Qualitative learnings and recommendations
- Common pitfalls and organizational defaults

**Audience**: Product managers, commercial strategists, experiment designers
**Update Cadence**: After experiment completion or major milestone

### MLflow Tracking System
**Purpose**: Technical execution and model metrics
**Focus**:
- Technical run parameters and hyperparameters
- Model training metrics (RMSE, R², AUC, etc.)
- Feature importance and SHAP values
- Model artifacts and versioning

**Content**:
- Hyperparameter configurations (e.g., `n_estimators=500`, `max_depth=10`)
- Training/validation metrics per epoch/iteration
- Model performance metrics
- Computational resources and runtime
- Model artifacts (pickle files, ONNX models, etc.)

**Audience**: Data scientists, ML engineers, model developers
**Update Cadence**: Real-time during model training

### Cross-Referencing

**From experiment-history.md to MLflow:**
Each experiment in this document should reference the corresponding MLflow experiment ID(s) for technical details:

```yaml
EXP-2024-001:
  mlflow_tracking:
    experiment_id: "kisqali-hcp-targeting-2024"
    runs:
      - causal_forest_v1: "run_id_abc123"
      - causal_forest_v2: "run_id_def456"
    location: "http://mlflow.e2i.internal:5000/#/experiments/12"
```

**From MLflow to experiment-history.md:**
MLflow runs should tag the business experiment ID for traceability:

```python
mlflow.set_tag("business_experiment_id", "EXP-2024-001")
mlflow.set_tag("brand", "Kisqali")
mlflow.set_tag("experiment_type", "cluster_rct")
```

### When to Use Each

| Question | Use |
|----------|-----|
| "What was the business impact of the HCP targeting experiment?" | experiment-history.md |
| "What were the technical model metrics for the causal forest?" | MLflow |
| "Should we run a similar experiment for another brand?" | experiment-history.md |
| "Which hyperparameters performed best for the propensity model?" | MLflow |
| "What design pitfalls should we avoid?" | experiment-history.md |
| "How did model performance change across training runs?" | MLflow |

### Integration Example

**Business View** (experiment-history.md):
```yaml
EXP-2024-001: Kisqali HCP Targeting Optimization
Status: Completed
Primary Outcome: NRx lift of 18.1% (CI: 12-24%, p<0.001)
Learning: CATE-based targeting outperforms decile-based rules
```

**Technical View** (MLflow):
```yaml
Experiment: kisqali-hcp-targeting-2024
Run: causal_forest_v1
Parameters:
  n_estimators: 500
  max_depth: 10
  min_samples_leaf: 100
Metrics:
  r_score: 0.82
  calibration_slope: 1.03
  shap_top_feature: "historical_trx"
Artifacts:
  - causal_forest.pkl
  - shap_values.csv
  - feature_importance.png
```

**Cross-Reference**:
- experiment-history.md entry links to MLflow experiment ID
- MLflow runs tagged with `business_experiment_id: EXP-2024-001`
- Both documents updated after experiment completion

---

## Experiment Registry

### EXP-2024-001: Kisqali HCP Targeting Optimization

| Property | Value |
|----------|-------|
| **Status** | Completed |
| **Brand** | Kisqali |
| **Start Date** | 2024-02-01 |
| **End Date** | 2024-05-31 |
| **Design Type** | Cluster RCT |
| **Primary Outcome** | NRx |

**Hypothesis:**
Enhanced AI-driven HCP targeting (using causal ML) will increase NRx by 10% compared to traditional targeting rules.

**Design Details:**
```yaml
treatment:
  name: "AI-Driven Targeting"
  description: "CATE-based HCP prioritization using causal forest"
  units: territories
  
control:
  name: "Traditional Targeting"
  description: "Decile-based prioritization using historical TRx"
  
randomization:
  level: territory
  n_treatment: 45
  n_control: 45
  stratification: [region, territory_potential_decile]
  
power_analysis:
  mde: 0.10
  power: 0.80
  alpha: 0.05
  icc: 0.15
```

**Results:**
```yaml
primary_outcome:
  treatment_mean: 127.3
  control_mean: 107.8
  ate: 0.181  # 18.1% lift
  ci_95: [0.12, 0.24]
  p_value: 0.0003
  
robustness:
  placebo_test: passed
  subset_validation: passed
  sensitivity_r2: 0.18
```

**Learnings:**
- CATE-based targeting significantly outperforms decile-based
- Effect concentrated in middle-decile HCPs (deciles 4-7)
- Territory-level randomization appropriate for this intervention
- 3-month observation period sufficient for NRx outcomes

**Assumption Violations:**
- Minor contamination between adjacent territories (~5%)
- One region had unusual competitive activity (excluded from analysis)

---

### EXP-2024-002: Patient Support Call Cadence

| Property | Value |
|----------|-------|
| **Status** | Completed |
| **Brand** | Kisqali |
| **Start Date** | 2024-04-01 |
| **End Date** | 2024-09-30 |
| **Design Type** | Individual RCT |
| **Primary Outcome** | Adherence (PDC) |

**Hypothesis:**
Increased patient support call frequency (weekly vs. monthly) will improve 6-month adherence by 8%.

**Design Details:**
```yaml
treatment:
  name: "Weekly Support Calls"
  description: "Weekly proactive outreach from nurse navigators"
  units: patients
  
control:
  name: "Monthly Support Calls"
  description: "Standard monthly check-in calls"
  
randomization:
  level: patient
  n_treatment: 850
  n_control: 850
  stratification: [age_group, prior_therapy, region]
  
power_analysis:
  mde: 0.08
  power: 0.85
  alpha: 0.05
```

**Results:**
```yaml
primary_outcome:
  treatment_mean: 0.847
  control_mean: 0.782
  ate: 0.065  # 6.5pp improvement (below MDE)
  ci_95: [0.042, 0.088]
  p_value: 0.012
  
secondary_outcomes:
  persistence_6mo:
    treatment: 0.78
    control: 0.71
    ate: 0.07
  patient_satisfaction:
    treatment: 4.2
    control: 3.8
    ate: 0.4
```

**Learnings:**
- Adherence improvement real but smaller than expected
- Strongest effect in first 3 months of therapy
- Cost-effectiveness questionable at weekly cadence
- Biweekly may be optimal (follow-up experiment recommended)

**Assumption Violations:**
- 12% patient attrition (balanced across arms)
- Call completion rate varied by region

---

### EXP-2024-003: Fabhalta KOL Engagement Intensity

| Property | Value |
|----------|-------|
| **Status** | Completed |
| **Brand** | Fabhalta |
| **Start Date** | 2024-06-01 |
| **End Date** | 2024-11-30 |
| **Design Type** | Quasi-experimental (DiD) |
| **Primary Outcome** | Referral Rate |

**Hypothesis:**
Intensive KOL engagement program will increase specialist referral rates by 15%.

**Design Details:**
```yaml
treatment:
  name: "Intensive KOL Program"
  description: "Enhanced speaker programs, advisory boards, data sharing"
  units: PNH_centers
  
control:
  name: "Standard Engagement"
  description: "Standard medical affairs engagement"
  
design:
  type: difference_in_differences
  treatment_group: "Early adopter centers"
  control_group: "Later adopter centers"
  pre_period: "2024-01 to 2024-05"
  post_period: "2024-06 to 2024-11"
  
matching:
  method: propensity_score
  variables: [center_size, pnh_volume, academic_status, region]
```

**Results:**
```yaml
primary_outcome:
  did_estimate: 0.22  # 22% increase in referral rate
  ci_95: [0.11, 0.33]
  p_value: 0.008
  
parallel_trends:
  pre_trend_test: passed
  visualization: confirmed
  
heterogeneity:
  academic_centers: +28%
  community_centers: +15%
```

**Learnings:**
- KOL engagement highly effective for rare disease
- Academic centers respond more strongly
- Long-term relationship building matters more than frequency
- Data sharing (RWE) particularly valued

**Assumption Violations:**
- Parallel trends assumption holds (verified)
- Selection into treatment not fully random (controlled via matching)

---

### EXP-2024-004: Digital HCP Engagement (Failed)

| Property | Value |
|----------|-------|
| **Status** | Completed (Null Result) |
| **Brand** | Remibrutinib |
| **Start Date** | 2024-03-01 |
| **End Date** | 2024-06-30 |
| **Design Type** | Cluster RCT |
| **Primary Outcome** | Awareness Score |

**Hypothesis:**
Digital-only HCP engagement will achieve comparable awareness to in-person engagement at lower cost.

**Design Details:**
```yaml
treatment:
  name: "Digital-Only Engagement"
  description: "Webinars, email, digital content only"
  units: territories
  
control:
  name: "Traditional + Digital"
  description: "Standard hybrid engagement model"
  
randomization:
  level: territory
  n_treatment: 30
  n_control: 30
```

**Results:**
```yaml
primary_outcome:
  treatment_mean: 3.2
  control_mean: 4.1
  ate: -0.9  # Digital-only WORSE
  ci_95: [-1.3, -0.5]
  p_value: 0.002
  
secondary_outcomes:
  nrx:
    ate: -0.15  # 15% lower NRx
    p_value: 0.04
```

**Learnings:**
- **CRITICAL**: Digital-only engagement insufficient for launch brand
- In-person engagement provides irreplaceable value
- Digital should complement, not replace, in-person
- Pre-launch requires relationship building

**Recommendation:**
Do NOT use digital-only engagement for launch brands. This experiment provides strong evidence against this approach.

---

### EXP-2024-005: Sample Allocation Optimization

| Property | Value |
|----------|-------|
| **Status** | In Progress |
| **Brand** | Kisqali |
| **Start Date** | 2024-10-01 |
| **End Date** | 2025-03-31 |
| **Design Type** | Adaptive RCT |
| **Primary Outcome** | NRx |

**Hypothesis:**
CATE-optimized sample allocation will outperform uniform allocation by 8%.

**Design Details:**
```yaml
treatment:
  name: "CATE-Optimized Samples"
  description: "Samples allocated based on estimated treatment effect"
  
control:
  name: "Uniform Allocation"
  description: "Equal samples per HCP in target list"
  
design:
  type: adaptive_rct
  initial_allocation: 50/50
  adaptation_schedule: monthly
  
power_analysis:
  mde: 0.08
  power: 0.80
```

**Interim Results (Month 3):**
```yaml
interim_analysis:
  treatment_mean: 45.2
  control_mean: 42.8
  estimated_ate: 0.056
  conditional_power: 0.62
  recommendation: continue
```

---

## Common Pitfalls Learned

### Design Pitfalls

| Pitfall | Experiments Affected | Mitigation |
|---------|---------------------|------------|
| Insufficient power | EXP-2024-002 | Use historical effect sizes, add 20% buffer |
| Contamination | EXP-2024-001 | Geographic buffers, cluster larger |
| Selection bias | EXP-2024-003 | Propensity matching, DiD when possible |
| Attrition | EXP-2024-002 | Intent-to-treat analysis, over-recruit |

### Assumption Violations

| Assumption | Violation Frequency | Detection Method | Remedy |
|------------|---------------------|------------------|--------|
| SUTVA | 25% of experiments | Spillover tests | Cluster randomization |
| Parallel trends | 10% of DiD | Pre-trend visualization | Synthetic control |
| No unmeasured confounding | Unknown | Sensitivity analysis | Bound effects |
| Positivity | 15% of experiments | Overlap checks | Trim extremes |

### Effect Size Miscalibration

| Domain | Typical Expected | Typical Actual | Recommendation |
|--------|------------------|----------------|----------------|
| HCP targeting | 15% | 10-20% | Well-calibrated |
| Patient support | 10% | 5-8% | Reduce expectations |
| Digital engagement | 8% | 2-5% | Much lower than expected |
| KOL programs | 12% | 15-25% | Higher than expected |

---

## Organizational Defaults

Based on historical experiments, use these defaults:

### Sample Size Defaults

| Outcome Type | ICC | MDE (realistic) | Power |
|--------------|-----|-----------------|-------|
| NRx | 0.15 | 10% | 0.80 |
| TRx | 0.12 | 8% | 0.80 |
| Adherence | 0.08 | 6% | 0.85 |
| Market Share | 0.20 | 5% | 0.80 |

### Duration Defaults

| Outcome | Minimum Duration | Recommended |
|---------|------------------|-------------|
| NRx | 3 months | 4-6 months |
| Adherence | 6 months | 6-12 months |
| Persistence | 6 months | 12 months |
| Market Share | 6 months | 12 months |

### Stratification Variables

Always stratify by:
1. Region (geographic)
2. Baseline potential (decile)
3. Prior brand engagement (history)

Consider stratifying by:
- HCP specialty
- Practice type
- Payer mix

---

## Knowledge Base Updates

### December 2025 (Planned)

**Expected Additions:**
- EXP-2024-005 final results (Sample Allocation Optimization)
- Remibrutinib launch experiments (Q1 2025)
- Cross-brand meta-analysis of targeting effectiveness
- New experiments:
  - Patient onboarding optimization (Fabhalta)
  - Omnichannel messaging sequence testing (Kisqali)
  - Payer strategy experiments (Remibrutinib)

**Planned Updates:**
- Refresh organizational defaults with 2024 complete dataset
- Add launch brand vs. mature brand effect size calibration
- Update common pitfalls with 2024 learnings

---

### December 2024

**Last Updated**: 2024-12-01

**Recent Additions:**
- EXP-2024-005 interim results
- Updated effect size calibration table
- Added digital engagement failure case (EXP-2024-004)
- Added MLflow Integration section

**Pending Updates:**
- EXP-2024-005 final results (March 2025)
- New Remibrutinib launch experiments
- Cross-brand meta-analysis
