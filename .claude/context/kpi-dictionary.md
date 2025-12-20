# KPI Dictionary

## Overview

This document defines all Key Performance Indicators (KPIs) used in the E2I Causal Analytics platform, including their causal relationships, measurement specifications, and analytical considerations.

---

## Prescription Metrics

### NRx (New Prescriptions)

| Property | Value |
|----------|-------|
| **Definition** | Number of new prescriptions written for a brand |
| **Granularity** | Weekly, by HCP, territory, or brand |
| **Data Source** | IQVIA/Symphony claims data |
| **Lag** | 2-4 weeks |

**Causal Relationships:**
```
Drivers → NRx:
├── HCP Engagement (+)
├── HCP Targeting (+)
├── Sample Drops (+)
├── Patient Identification Programs (+)
├── Competitor Activity (-)
└── Payer Restrictions (-)

NRx → Outcomes:
├── Market Share (+)
├── Revenue (+)
└── TRx (lagged, +)
```

**Analytical Considerations:**
- Seasonality in Q4 (holiday effects)
- Territory size normalization required
- New HCP vs. existing HCP distinction important

---

### TRx (Total Prescriptions)

| Property | Value |
|----------|-------|
| **Definition** | Total prescriptions filled (new + refills) |
| **Granularity** | Weekly, by HCP, territory, or brand |
| **Data Source** | IQVIA/Symphony claims data |
| **Lag** | 2-4 weeks |

**Causal Relationships:**
```
Drivers → TRx:
├── NRx (lagged, +)
├── Persistence (+)
├── Refill Rate (+)
├── Patient Support Programs (+)
└── Adherence (+)

TRx → Outcomes:
├── Revenue (+)
├── Patient Outcomes (+)
└── Market Share (+)
```

**Analytical Considerations:**
- Decompose into NRx + refills for causal analysis
- Patient cohort effects (time on therapy)
- Dosing schedule affects refill patterns

---

### NBRx (New-to-Brand Prescriptions)

| Property | Value |
|----------|-------|
| **Definition** | Prescriptions from patients new to the brand |
| **Granularity** | Weekly, by HCP, territory |
| **Data Source** | IQVIA/Symphony patient-level data |
| **Lag** | 2-4 weeks |

**Causal Relationships:**
```
Drivers → NBRx:
├── Competitive Switching (+)
├── New Patient Starts (+)
├── Brand Preference (+)
├── Access/Formulary (+)
└── Competitor Stockouts (+)

NBRx → Outcomes:
├── Market Share Change (+)
├── Growth Trajectory (+)
└── TRx (future, +)
```

---

## Engagement Metrics

### HCP Reach

| Property | Value |
|----------|-------|
| **Definition** | Number of HCPs engaged by sales force |
| **Granularity** | Monthly, by territory, rep |
| **Data Source** | CRM (Veeva) |
| **Lag** | Real-time |

**Causal Relationships:**
```
Drivers → HCP Reach:
├── Sales Force Size (+)
├── Territory Design (+)
├── Call Planning (+)
└── Access Constraints (-)

HCP Reach → Outcomes:
├── HCP Awareness (+)
├── NRx (lagged, +)
└── Brand Perception (+)
```

---

### Call Frequency

| Property | Value |
|----------|-------|
| **Definition** | Average calls per HCP per period |
| **Granularity** | Monthly, by HCP decile, territory |
| **Data Source** | CRM (Veeva) |
| **Lag** | Real-time |

**Causal Relationships:**
```
Drivers → Call Frequency:
├── Targeting Algorithm (+)
├── HCP Accessibility (+)
├── Rep Capacity (-)
└── Priority Tier (+)

Call Frequency → Outcomes:
├── NRx (+, diminishing returns)
├── Message Retention (+)
└── HCP Satisfaction (inverted U)
```

**Analytical Considerations:**
- Diminishing returns typically after 4-6 calls/quarter
- Interaction with call quality matters
- Segment-specific optimal frequency

---

### Sample Drops

| Property | Value |
|----------|-------|
| **Definition** | Number of product samples provided to HCP |
| **Granularity** | Monthly, by HCP, territory |
| **Data Source** | Sample tracking system |
| **Lag** | Real-time |

**Causal Relationships:**
```
Drivers → Sample Drops:
├── Rep Visits (+)
├── Sample Inventory (+)
├── HCP Request (+)
└── Regulatory Limits (-)

Sample Drops → Outcomes:
├── NRx (+, short-term)
├── Trial Rate (+)
├── Patient Start (+)
└── Long-term TRx (?)
```

**Analytical Considerations:**
- Strong short-term effect, unclear long-term
- Regulatory limits create natural experiments
- Patient vs. HCP directed sampling

---

## Patient Metrics

### Adherence Rate

| Property | Value |
|----------|-------|
| **Definition** | Proportion of days covered (PDC) by medication |
| **Granularity** | Monthly, patient cohort |
| **Data Source** | Claims data |
| **Lag** | 30-60 days |

**Causal Relationships:**
```
Drivers → Adherence:
├── Patient Support Programs (+)
├── Side Effect Profile (-)
├── Dosing Convenience (+)
├── Cost/Copay (-)
└── HCP Communication (+)

Adherence → Outcomes:
├── Clinical Outcomes (+)
├── Persistence (+)
├── TRx (+)
└── Healthcare Costs (-)
```

---

### Persistence Rate

| Property | Value |
|----------|-------|
| **Definition** | Proportion of patients still on therapy at time T |
| **Granularity** | 3mo, 6mo, 12mo cohorts |
| **Data Source** | Claims data |
| **Lag** | Time-dependent |

**Causal Relationships:**
```
Drivers → Persistence:
├── Efficacy (+)
├── Tolerability (+)
├── Patient Support (+)
├── HCP Follow-up (+)
└── Out-of-Pocket Cost (-)

Persistence → Outcomes:
├── Long-term TRx (+)
├── Clinical Outcomes (+)
├── Patient Satisfaction (+)
└── LTV (+)
```

---

### Patient Identification Rate

| Property | Value |
|----------|-------|
| **Definition** | Rate of eligible patients identified for treatment |
| **Granularity** | Monthly, by HCP, territory |
| **Data Source** | EHR data, claims data |
| **Lag** | Variable |

**Causal Relationships:**
```
Drivers → Patient ID Rate:
├── Disease Awareness (+)
├── Diagnostic Tools (+)
├── HCP Education (+)
├── Screening Programs (+)
└── EMR Alerts (+)

Patient ID Rate → Outcomes:
├── Treatment Starts (+)
├── NRx (+)
├── Appropriate Treatment (+)
└── Patient Outcomes (+)
```

---

## Market Metrics

### Market Share

| Property | Value |
|----------|-------|
| **Definition** | Brand TRx / Total class TRx |
| **Granularity** | Weekly/Monthly, by geography |
| **Data Source** | IQVIA/Symphony |
| **Lag** | 2-4 weeks |

**Causal Relationships:**
```
Drivers → Market Share:
├── NRx Growth (+)
├── Persistence (+)
├── Competitor Weakness (+)
├── Formulary Position (+)
└── Clinical Data (+)

Market Share → Outcomes:
├── Revenue (+)
├── Bargaining Power (+)
└── Investment Justification (+)
```

---

### Share of Voice

| Property | Value |
|----------|-------|
| **Definition** | Brand calls / Total class calls by territory |
| **Granularity** | Monthly, by territory |
| **Data Source** | CRM data |
| **Lag** | Real-time |

**Causal Relationships:**
```
Drivers → Share of Voice:
├── Sales Force Investment (+)
├── Call Frequency (+)
├── Rep Prioritization (+)
└── Competitor Activity (-)

Share of Voice → Outcomes:
├── HCP Awareness (+)
├── Brand Preference (+)
├── Market Share (lagged, +)
└── NRx (+)
```

---

## Causal Confounders Reference

### Common Confounders by KPI

| Confounder | Affects | Direction | Mitigation |
|------------|---------|-----------|------------|
| Territory Potential | NRx, TRx, Market Share | + | Stratify by decile |
| HCP Specialty | All HCP metrics | Variable | Segment analysis |
| Patient Demographics | Persistence, Adherence | Variable | Propensity matching |
| Payer Mix | All prescription metrics | Variable | Include as covariate |
| Seasonality | NRx, Adherence | Cyclical | Time fixed effects |
| Prior Treatment | Persistence, Outcomes | Variable | Stratify by history |
| Geographic Region | All | Variable | Regional fixed effects |
| Practice Type | HCP metrics | Variable | Segment analysis |

### Instrumental Variables

| Instrument | For Treatment | Validity | Strength |
|------------|---------------|----------|----------|
| Rep Turnover | HCP Engagement | Exogenous shock | Medium |
| Weather | Call Frequency | Exogenous | Weak |
| Policy Changes | Formulary Position | Natural experiment | Strong |
| Competitor Stockout | Competitive Switching | Natural experiment | Strong |
| Distance to HCP | Call Frequency | Geographic | Medium |

---

## Complete KPI Reference (46 KPIs)

**Last Updated**: 2025-12-18
**Source**: config/kpi_definitions.yaml
**Total KPIs**: 46 (100% calculable from database schema)

### KPI Categories

| Category | KPIs | Status |
|----------|------|--------|
| WS1: Data Quality | 9 | ✅ 100% Calculable |
| WS1: Model Performance | 9 | ✅ 100% Calculable |
| WS2: Trigger Performance | 8 | ✅ 100% Calculable |
| WS3: Business Impact | 10 | ✅ 100% Calculable |
| Brand-Specific | 5 | ✅ 100% Calculable |
| Causal Metrics | 5 | ✅ 100% Calculable |

---

## WS1: Data Quality KPIs (9 KPIs)

### WS1-DQ-001: Source Coverage - Patients
- **Definition**: Percentage of eligible patients present in source vs reference universe
- **Formula**: `covered_patients / reference_patients`
- **Tables**: patient_journeys, reference_universe
- **Target**: ≥85% | **Warning**: 70% | **Critical**: 50%
- **Frequency**: Daily

### WS1-DQ-002: Source Coverage - HCPs
- **Definition**: Percentage of priority HCPs present in source vs universe
- **Formula**: `covered_hcps / reference_hcps`
- **Tables**: hcp_profiles, reference_universe
- **Target**: ≥80% | **Warning**: 65% | **Critical**: 45%
- **Frequency**: Daily

### WS1-DQ-003: Cross-Source Match Rate ⭐ NEW in V3
- **Definition**: Percentage of entities linkable across data sources
- **Formula**: `records_matched / total_records`
- **Tables**: data_source_tracking
- **Columns**: match_rate_vs_claims, match_rate_vs_ehr, match_rate_vs_specialty
- **View**: v_kpi_cross_source_match
- **Target**: ≥75% | **Warning**: 60% | **Critical**: 40%
- **Frequency**: Daily

### WS1-DQ-004: Stacking Lift ⭐ NEW in V3
- **Definition**: Incremental value from combining multiple data sources
- **Formula**: `(stacked_value - baseline) / baseline`
- **Tables**: data_source_tracking
- **Columns**: stacking_lift_percentage, stacking_eligible_records, stacking_applied_records
- **View**: v_kpi_stacking_lift
- **Target**: ≥15% | **Warning**: 10% | **Critical**: 5%
- **Frequency**: Daily

### WS1-DQ-005: Completeness Pass Rate
- **Definition**: 1 minus null rate across brand-critical fields
- **Formula**: `1 - (null_critical / total_records)`
- **Tables**: patient_journeys
- **Target**: ≥95% | **Warning**: 90% | **Critical**: 80%
- **Frequency**: Daily

### WS1-DQ-006: Geographic Consistency
- **Definition**: Max absolute gap between source share and universe share across regions
- **Formula**: `max_region(|share_source - share_universe|)`
- **Tables**: patient_journeys, reference_universe
- **Target**: ≤5% | **Warning**: 10% | **Critical**: 20%
- **Frequency**: Weekly

### WS1-DQ-007: Data Lag (Median) ⭐ NEW in V3
- **Definition**: Median days from service date to availability in warehouse
- **Formula**: `median(ingestion_timestamp - source_timestamp)`
- **Tables**: patient_journeys
- **Columns**: source_timestamp, ingestion_timestamp, data_lag_hours
- **View**: v_kpi_data_lag
- **Target**: ≤3 days | **Warning**: 7 days | **Critical**: 14 days
- **Frequency**: Daily

### WS1-DQ-008: Label Quality (IAA) ⭐ NEW in V3
- **Definition**: Inter-annotator agreement score for labeled data
- **Formula**: `avg(agreement_score) for iaa_groups`
- **Tables**: ml_annotations
- **Columns**: iaa_group_id, annotation_value, annotation_confidence
- **View**: v_kpi_label_quality
- **Target**: ≥85% | **Warning**: 70% | **Critical**: 60%
- **Frequency**: Weekly

### WS1-DQ-009: Time-to-Release (TTR) ⭐ NEW in V3
- **Definition**: Hours from source data timestamp to pipeline completion
- **Formula**: `run_completed_at - source_data_timestamp`
- **Tables**: etl_pipeline_metrics
- **Columns**: source_data_timestamp, run_completed_at, time_to_release_hours
- **View**: v_kpi_time_to_release
- **Target**: ≤24 hours | **Warning**: 48 hours | **Critical**: 72 hours
- **Frequency**: Daily

---

## WS1: Model Performance KPIs (9 KPIs)

### WS1-MP-001: ROC-AUC
- **Definition**: Area Under the ROC Curve
- **Formula**: `∫TPR d(FPR)`
- **Tables**: ml_predictions
- **Columns**: model_auc
- **Target**: ≥0.80 | **Warning**: 0.70 | **Critical**: 0.60
- **Frequency**: Daily

### WS1-MP-002: PR-AUC ⭐ NEW in V3
- **Definition**: Area under the Precision-Recall curve
- **Formula**: `∫Precision d(Recall)`
- **Tables**: ml_predictions
- **Columns**: model_pr_auc
- **Target**: ≥0.70 | **Warning**: 0.55 | **Critical**: 0.40
- **Frequency**: Daily

### WS1-MP-003: F1 Score
- **Definition**: Harmonic mean of precision and recall
- **Formula**: `2 * (precision * recall) / (precision + recall)`
- **Tables**: ml_predictions
- **Columns**: model_precision, model_recall
- **Target**: ≥0.75 | **Warning**: 0.60 | **Critical**: 0.45
- **Frequency**: Daily

### WS1-MP-004: Recall@Top-K ⭐ NEW in V3
- **Definition**: Recall achieved when selecting top K predictions
- **Formula**: `TP_at_K / total_positives`
- **Tables**: ml_predictions
- **Columns**: rank_metrics (JSONB: recall_at_5, recall_at_10, recall_at_20)
- **Target**: ≥0.60 | **Warning**: 0.45 | **Critical**: 0.30
- **Frequency**: Daily

### WS1-MP-005: Brier Score ⭐ NEW in V3
- **Definition**: Mean squared error of probability predictions
- **Formula**: `mean((p - y)^2)`
- **Tables**: ml_predictions
- **Columns**: brier_score
- **Target**: ≤0.15 | **Warning**: 0.25 | **Critical**: 0.35
- **Frequency**: Daily

### WS1-MP-006: Calibration Slope
- **Definition**: Slope of predicted vs actual probability regression
- **Formula**: `logistic_regression(y ~ predicted_prob).slope`
- **Tables**: ml_predictions
- **Columns**: calibration_score
- **Target**: ~1.0 | **Warning**: 0.8 | **Critical**: 0.6
- **Frequency**: Weekly

### WS1-MP-007: SHAP Coverage
- **Definition**: Percentage of predictions with SHAP explanations
- **Formula**: `count(shap_values IS NOT NULL) / total_predictions`
- **Tables**: ml_predictions
- **Columns**: shap_values
- **Target**: ≥95% | **Warning**: 80% | **Critical**: 60%
- **Frequency**: Daily

### WS1-MP-008: Fairness Gap (ΔRecall)
- **Definition**: Max difference in recall across protected groups
- **Formula**: `max_group(recall) - min_group(recall)`
- **Tables**: ml_predictions
- **Columns**: fairness_metrics (JSONB)
- **Target**: ≤0.05 | **Warning**: 0.10 | **Critical**: 0.20
- **Frequency**: Weekly

### WS1-MP-009: Feature Drift (PSI)
- **Definition**: Population Stability Index measuring feature distribution shift
- **Formula**: `Σ_b (q_b - p_b) * ln(q_b / p_b)`
- **Tables**: ml_preprocessing_metadata, ml_predictions
- **Columns**: feature_distributions
- **Target**: ≤0.10 | **Warning**: 0.20 | **Critical**: 0.25
- **Frequency**: Daily

---

## WS2: Trigger Performance KPIs (8 KPIs)

### WS2-TR-001: Trigger Precision
- **Definition**: Percentage of fired triggers resulting in positive outcome
- **Formula**: `true_positives / (true_positives + false_positives)`
- **Tables**: triggers
- **Target**: ≥70% | **Warning**: 55% | **Critical**: 40%
- **Frequency**: Daily

### WS2-TR-002: Trigger Recall
- **Definition**: Percentage of positive outcomes preceded by a trigger
- **Formula**: `true_positives / (true_positives + false_negatives)`
- **Tables**: triggers, treatment_events
- **Target**: ≥60% | **Warning**: 45% | **Critical**: 30%
- **Frequency**: Daily

### WS2-TR-003: Action Rate Uplift
- **Definition**: Incremental action rate vs control group
- **Formula**: `(action_rate_treatment - action_rate_control) / action_rate_control`
- **Tables**: triggers
- **Target**: ≥15% | **Warning**: 10% | **Critical**: 5%
- **Frequency**: Weekly

### WS2-TR-004: Acceptance Rate
- **Definition**: Percentage of delivered triggers accepted by reps
- **Formula**: `count(accepted) / count(delivered)`
- **Tables**: triggers
- **Columns**: acceptance_status
- **Target**: ≥60% | **Warning**: 45% | **Critical**: 30%
- **Frequency**: Daily

### WS2-TR-005: False Alert Rate
- **Definition**: Percentage of triggers marked as false positives
- **Formula**: `count(false_positive) / total_triggers`
- **Tables**: triggers
- **Columns**: false_positive_flag
- **Target**: ≤10% | **Warning**: 20% | **Critical**: 30%
- **Frequency**: Daily

### WS2-TR-006: Override Rate
- **Definition**: Percentage of triggers overridden by users
- **Formula**: `count(overridden) / count(delivered)`
- **Tables**: triggers
- **Target**: ≤15% | **Warning**: 25% | **Critical**: 40%
- **Frequency**: Daily

### WS2-TR-007: Lead Time
- **Definition**: Median days between trigger and outcome
- **Formula**: `median(outcome_date - trigger_date)`
- **Tables**: triggers
- **Columns**: lead_time_days
- **Target**: ≤14 days | **Warning**: 21 days | **Critical**: 30 days
- **Frequency**: Weekly

### WS2-TR-008: Change-Fail Rate (CFR) ⭐ NEW in V3
- **Definition**: Percentage of trigger changes that resulted in worse outcomes
- **Formula**: `count(change_failed) / count(changed_triggers)`
- **Tables**: triggers
- **Columns**: previous_trigger_id, change_type, change_failed, change_outcome_delta
- **View**: v_kpi_change_fail_rate
- **Target**: ≤10% | **Warning**: 20% | **Critical**: 30%
- **Frequency**: Weekly

---

## WS3: Business Impact KPIs (10 KPIs)

### WS3-BI-001: Monthly Active Users (MAU) ⭐ NEW in V3
- **Definition**: Unique users with at least one session in past 30 days
- **Formula**: `count(distinct user_id) where session_start >= now() - 30 days`
- **Tables**: user_sessions
- **View**: v_kpi_active_users
- **Target**: ≥2000 | **Warning**: 1500 | **Critical**: 1000
- **Frequency**: Daily

### WS3-BI-002: Weekly Active Users (WAU) ⭐ NEW in V3
- **Definition**: Unique users with at least one session in past 7 days
- **Formula**: `count(distinct user_id) where session_start >= now() - 7 days`
- **Tables**: user_sessions
- **View**: v_kpi_active_users
- **Target**: ≥1200 | **Warning**: 900 | **Critical**: 600
- **Frequency**: Daily

### WS3-BI-003: Patient Touch Rate
- **Definition**: Percentage of eligible patients with trigger-driven touchpoint
- **Formula**: `patients_with_trigger / eligible_patients`
- **Tables**: triggers, patient_journeys
- **Target**: ≥40% | **Warning**: 30% | **Critical**: 20%
- **Frequency**: Weekly

### WS3-BI-004: HCP Coverage
- **Definition**: Percentage of priority HCPs with active engagement
- **Formula**: `count(covered) / total_priority_hcps`
- **Tables**: hcp_profiles
- **Columns**: coverage_status
- **Target**: ≥75% | **Warning**: 60% | **Critical**: 45%
- **Frequency**: Weekly

### WS3-BI-005: Total Prescriptions (TRx)
- **Definition**: Total prescription volume
- **Formula**: `count(event_type = 'prescription')`
- **Tables**: treatment_events
- **Frequency**: Daily

### WS3-BI-006: New Prescriptions (NRx)
- **Definition**: First-time prescriptions for a patient
- **Formula**: `count(first_prescription)`
- **Tables**: treatment_events
- **Frequency**: Daily

### WS3-BI-007: New-to-Brand Prescriptions (NBRx)
- **Definition**: First prescription of specific brand for a patient
- **Formula**: `count(first_brand_prescription)`
- **Tables**: treatment_events
- **Frequency**: Daily

### WS3-BI-008: TRx Share
- **Definition**: Brand prescription share of total category
- **Formula**: `brand_trx / category_trx`
- **Tables**: treatment_events
- **Target**: ≥30% | **Warning**: 20% | **Critical**: 10%
- **Frequency**: Weekly

### WS3-BI-009: Conversion Rate
- **Definition**: Percentage of triggers resulting in prescription
- **Formula**: `prescriptions_after_trigger / triggers_delivered`
- **Tables**: triggers, treatment_events
- **Target**: ≥8% | **Warning**: 5% | **Critical**: 2%
- **Frequency**: Weekly

### WS3-BI-010: Return on Investment (ROI)
- **Definition**: Value generated per dollar invested
- **Formula**: `value_captured / cost_invested`
- **Tables**: business_metrics, agent_activities
- **Columns**: roi, roi_estimate
- **Target**: ≥3.0 | **Warning**: 2.0 | **Critical**: 1.0
- **Frequency**: Monthly

---

## Brand-Specific KPIs (5 KPIs)

### BR-001: Remi - AH Uncontrolled %
- **Brand**: Remibrutinib
- **Definition**: Percentage of antihistamine patients with uncontrolled symptoms
- **Formula**: `uncontrolled_patients / ah_patients`
- **Tables**: patient_journeys, treatment_events
- **Target**: ≤40% | **Warning**: 50% | **Critical**: 60%
- **Frequency**: Weekly

### BR-002: Remi - Intent-to-Prescribe Δ ⭐ NEW in V3
- **Brand**: Remibrutinib
- **Definition**: Change in HCP intent-to-prescribe score after intervention
- **Formula**: `post_intent - pre_intent`
- **Tables**: hcp_intent_surveys
- **Columns**: intent_to_prescribe_score, intent_to_prescribe_change, previous_survey_id
- **View**: v_kpi_intent_to_prescribe
- **Target**: ≥0.5 points | **Warning**: 0.3 points | **Critical**: 0.0 points
- **Scale**: 1-7 point scale
- **Frequency**: Monthly

### BR-003: Fabhalta - % PNH Tested
- **Brand**: Fabhalta
- **Definition**: Percentage of eligible patients tested for PNH
- **Formula**: `pnh_tested / eligible_patients`
- **Tables**: treatment_events
- **Target**: ≥60% | **Warning**: 45% | **Critical**: 30%
- **Frequency**: Weekly

### BR-004: Kisqali - Dx Adoption
- **Brand**: Kisqali
- **Definition**: Time from diagnosis to first Kisqali prescription
- **Formula**: `median(first_kisqali_date - diagnosis_date)`
- **Tables**: patient_journeys, treatment_events
- **Target**: ≤30 days | **Warning**: 45 days | **Critical**: 60 days
- **Frequency**: Weekly

### BR-005: Kisqali - Oncologist Reach
- **Brand**: Kisqali
- **Definition**: Percentage of oncologists with Kisqali engagement
- **Formula**: `engaged_oncologists / total_oncologists`
- **Tables**: hcp_profiles, triggers
- **Target**: ≥70% | **Warning**: 55% | **Critical**: 40%
- **Frequency**: Weekly

---

## Causal Metrics KPIs (5 KPIs)

### CM-001: Average Treatment Effect (ATE)
- **Definition**: Average causal effect of treatment on outcome
- **Formula**: `E[Y(1) - Y(0)]`
- **Tables**: ml_predictions
- **Columns**: treatment_effect_estimate
- **Frequency**: Weekly (on-demand)

### CM-002: Conditional ATE (CATE)
- **Definition**: Treatment effect conditioned on segment
- **Formula**: `E[Y(1) - Y(0) | X=x]`
- **Tables**: ml_predictions
- **Columns**: heterogeneous_effect, segment_assignment
- **Frequency**: Weekly (on-demand)

### CM-003: Causal Impact
- **Definition**: Estimated causal effect size on causal paths
- **Formula**: Computed by causal_impact agent using DoWhy/EconML
- **Tables**: causal_paths
- **Columns**: causal_effect_size, confidence_level
- **Frequency**: On-demand

### CM-004: Counterfactual Outcome
- **Definition**: Predicted outcome under alternative treatment
- **Formula**: `E[Y(a') | do(A=a), X]`
- **Tables**: ml_predictions
- **Columns**: counterfactual_outcome
- **Frequency**: On-demand

### CM-005: Mediation Effect
- **Definition**: Effect mediated through intermediate variables
- **Formula**: `indirect_effect / total_effect`
- **Tables**: causal_paths
- **Columns**: mediators_identified, pathway_details
- **Frequency**: On-demand

---

## Database Implementation Reference

**All 46 KPIs are 100% calculable from the database schema.**

### Core Tables for KPI Calculation

| Table | KPIs Supported | Primary Use |
|-------|----------------|-------------|
| patient_journeys | WS1-DQ-001, WS1-DQ-005, WS1-DQ-006, WS1-DQ-007, WS3-BI-003, BR-001, BR-004 | Patient coverage, journeys, diagnosis |
| hcp_profiles | WS1-DQ-002, WS3-BI-004, BR-005 | HCP coverage, engagement |
| treatment_events | WS3-BI-005, WS3-BI-006, WS3-BI-007, WS3-BI-008, WS3-BI-009, BR-001, BR-003, BR-004 | Prescriptions, treatments |
| triggers | WS2-TR-001 through WS2-TR-008, WS3-BI-003, WS3-BI-009, BR-005 | Trigger performance |
| ml_predictions | WS1-MP-001 through WS1-MP-009, CM-001 through CM-004 | Model performance, causal metrics |
| data_source_tracking | WS1-DQ-003, WS1-DQ-004 | Data quality, source matching |
| ml_annotations | WS1-DQ-008 | Label quality |
| etl_pipeline_metrics | WS1-DQ-009 | Pipeline performance |
| user_sessions | WS3-BI-001, WS3-BI-002 | User engagement |
| hcp_intent_surveys | BR-002 | Intent tracking |
| reference_universe | WS1-DQ-001, WS1-DQ-002, WS1-DQ-006 | Coverage calculations |
| business_metrics | WS3-BI-010 | ROI, business metrics |
| causal_paths | CM-003, CM-005 | Causal relationships |

### Helper Views (8 views for optimized KPI queries)

| View | KPIs | Purpose |
|------|------|---------|
| v_kpi_cross_source_match | WS1-DQ-003 | Daily cross-source match rates by source |
| v_kpi_stacking_lift | WS1-DQ-004 | Stacking lift percentages |
| v_kpi_data_lag | WS1-DQ-007 | Data lag statistics (avg, median, p95) |
| v_kpi_label_quality | WS1-DQ-008 | Label quality and IAA metrics |
| v_kpi_time_to_release | WS1-DQ-009 | TTR by pipeline |
| v_kpi_change_fail_rate | WS2-TR-008 | Change-fail rate by change type |
| v_kpi_active_users | WS3-BI-001, WS3-BI-002 | MAU, WAU, DAU counts |
| v_kpi_intent_to_prescribe | BR-002 | Intent scores by brand/month |

### Database Schema Files

| Schema Component | File Location |
|------------------|---------------|
| Core Schema (V3) | database/core/e2i_ml_complete_v3_schema.sql |
| V4 ML Tables | database/ml/mlops_tables.sql (migration 007) |
| Helper Views | Defined in core schema |
| Split Registry | ml_split_registry, ml_patient_split_assignments |

### Calculation Queries

All KPIs can be calculated using standard SQL queries against these tables. Examples:

```sql
-- WS1-DQ-003: Cross-Source Match Rate
SELECT
    source_type,
    match_rate_vs_claims,
    match_rate_vs_ehr,
    match_rate_vs_specialty
FROM data_source_tracking
WHERE date = CURRENT_DATE;

-- WS3-BI-001: Monthly Active Users
SELECT COUNT(DISTINCT user_id) as mau
FROM user_sessions
WHERE session_start >= CURRENT_DATE - INTERVAL '30 days';

-- WS1-MP-001: ROC-AUC
SELECT
    model_name,
    model_version,
    AVG(model_auc) as avg_auc
FROM ml_predictions
WHERE prediction_date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY model_name, model_version;
```

---

## Measurement Specifications

### Data Quality Requirements

| Metric | Completeness | Accuracy | Timeliness |
|--------|--------------|----------|------------|
| NRx | >95% | ±5% | 2 weeks |
| TRx | >95% | ±5% | 2 weeks |
| HCP Reach | >99% | ±1% | Real-time |
| Call Frequency | >99% | ±1% | Real-time |
| Adherence | >80% | ±10% | 30 days |
| Market Share | >95% | ±3% | 2 weeks |

### Aggregation Rules

| Metric | Territory | Region | National |
|--------|-----------|--------|----------|
| NRx | Sum | Sum | Sum |
| TRx | Sum | Sum | Sum |
| Adherence | Weighted Avg | Weighted Avg | Weighted Avg |
| Market Share | Recalculate | Recalculate | Recalculate |
| HCP Reach | Sum | Sum | Deduplicated |
