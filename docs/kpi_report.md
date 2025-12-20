# E2I KPI Coverage Validation Report
Generated: 2025-12-02 14:18:23

## Summary

- **Total KPIs**: 46
- **Passed**: 46 ✅
- **Failed**: 0 ❌
- **Warnings**: 0 ⚠️
- **V3 New**: 14

## By Workstream

### Brand (5/5)

| ID | KPI | Status | V3 New | Notes |
|---|---|---|---|---|
| BR-001 | Remi - AH Uncontrolled % | ✅ |  | Database validation passed |
| BR-002 | Remi - Intent-to-Prescribe Δ | ✅ | ✓ | V3: NEW hcp_intent_surveys table |
| BR-003 | Fabhalta - % PNH Tested | ✅ |  | Database validation passed |
| BR-004 | Kisqali - Dx Adoption | ✅ |  | Database validation passed |
| BR-005 | Kisqali - Oncologist Reach | ✅ |  | Database validation passed |

### Causal (5/5)

| ID | KPI | Status | V3 New | Notes |
|---|---|---|---|---|
| CM-001 | Average Treatment Effect (ATE) | ✅ |  | Database validation passed |
| CM-002 | Conditional ATE (CATE) | ✅ |  | Database validation passed |
| CM-003 | Causal Impact | ✅ |  | Database validation passed |
| CM-004 | Counterfactual Outcome | ✅ |  | Database validation passed |
| CM-005 | Mediation Effect | ✅ |  | Database validation passed |

### WS1 (18/18)

| ID | KPI | Status | V3 New | Notes |
|---|---|---|---|---|
| WS1-DQ-001 | Source Coverage - Patients | ✅ | ✓ | V3: reference_universe table |
| WS1-DQ-002 | Source Coverage - HCPs | ✅ | ✓ | V3: reference_universe table |
| WS1-DQ-003 | Cross-source Match Rate | ✅ | ✓ | V3: NEW data_source_tracking table |
| WS1-DQ-004 | Stacking Lift | ✅ | ✓ | V3: NEW data_source_tracking table |
| WS1-DQ-005 | Completeness Pass Rate | ✅ |  | Database validation passed |
| WS1-DQ-006 | Geographic Consistency | ✅ |  | Database validation passed |
| WS1-DQ-007 | Data Lag (Median) | ✅ | ✓ | V3: NEW fields in patient_journeys |
| WS1-DQ-008 | Label Quality (IAA) | ✅ | ✓ | V3: NEW ml_annotations table |
| WS1-DQ-009 | Time-to-Release (TTR) | ✅ | ✓ | V3: NEW etl_pipeline_metrics table |
| WS1-MP-001 | ROC-AUC | ✅ |  | Database validation passed |
| WS1-MP-002 | PR-AUC | ✅ | ✓ | V3: NEW field model_pr_auc |
| WS1-MP-003 | F1 Score | ✅ |  | Database validation passed |
| WS1-MP-004 | Recall@Top-K | ✅ | ✓ | V3: NEW field rank_metrics (JSONB) |
| WS1-MP-005 | Brier Score | ✅ | ✓ | V3: NEW field brier_score |
| WS1-MP-006 | Calibration Slope | ✅ |  | Database validation passed |
| WS1-MP-007 | SHAP Coverage | ✅ |  | Database validation passed |
| WS1-MP-008 | Fairness Gap (ΔRecall) | ✅ |  | Database validation passed |
| WS1-MP-009 | Feature Drift (PSI) | ✅ |  | Database validation passed |

### WS2 (8/8)

| ID | KPI | Status | V3 New | Notes |
|---|---|---|---|---|
| WS2-TR-001 | Trigger Precision | ✅ |  | Database validation passed |
| WS2-TR-002 | Trigger Recall | ✅ |  | Database validation passed |
| WS2-TR-003 | Action Rate Uplift | ✅ |  | Database validation passed |
| WS2-TR-004 | Acceptance Rate | ✅ |  | Database validation passed |
| WS2-TR-005 | False Alert Rate | ✅ |  | Database validation passed |
| WS2-TR-006 | Override Rate | ✅ |  | Database validation passed |
| WS2-TR-007 | Lead Time | ✅ |  | Database validation passed |
| WS2-TR-008 | Change-Fail Rate (CFR) | ✅ | ✓ | V3: NEW change tracking fields |

### WS3 (10/10)

| ID | KPI | Status | V3 New | Notes |
|---|---|---|---|---|
| WS3-BI-001 | Monthly Active Users (MAU) | ✅ | ✓ | V3: NEW user_sessions table |
| WS3-BI-002 | Weekly Active Users (WAU) | ✅ | ✓ | V3: NEW user_sessions table |
| WS3-BI-003 | Patient Touch Rate | ✅ |  | Database validation passed |
| WS3-BI-004 | HCP Coverage | ✅ |  | Database validation passed |
| WS3-BI-005 | Total Prescriptions (TRx) | ✅ |  | Database validation passed |
| WS3-BI-006 | New Prescriptions (NRx) | ✅ |  | Database validation passed |
| WS3-BI-007 | New-to-Brand (NBRx) | ✅ |  | Database validation passed |
| WS3-BI-008 | TRx Share | ✅ |  | Database validation passed |
| WS3-BI-009 | Conversion Rate | ✅ |  | Database validation passed |
| WS3-BI-010 | ROI | ✅ |  | Database validation passed |

## V3 New Tables & Fields

### New Tables

- `user_sessions`
- `data_source_tracking`
- `ml_annotations`
- `etl_pipeline_metrics`
- `hcp_intent_surveys`
- `reference_universe`
- `agent_registry`

### New KPI Helper Views

- `v_kpi_cross_source_match`
- `v_kpi_stacking_lift`
- `v_kpi_data_lag`
- `v_kpi_label_quality`
- `v_kpi_time_to_release`
- `v_kpi_change_fail_rate`
- `v_kpi_active_users`
- `v_kpi_intent_to_prescribe`