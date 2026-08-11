/**
 * KPI catalog (GENERATED -- do not edit by hand)
 * =============================================
 *
 * Every KPI in the registry, with the alias forms a user or model might
 * type and the Flint semantic type its values should be charted as.
 *
 * Source:    config/kpi_definitions.yaml
 * Generator: scripts/gen_kpi_catalog.py
 *
 * Regenerate with `python3 scripts/gen_kpi_catalog.py` after editing the
 * YAML. `kpi-catalog.test.ts` fails if this file drifts from the YAML.
 *
 * @module lib/kpi-catalog.generated
 */

/** Flint semantic type governing axis formatting for a KPI's values. */
export type KpiSemanticType = 'Percentage' | 'Count' | 'Number' | 'Duration' | 'Score';

export interface KpiCatalogEntry {
  /** Registry code, e.g. 'WS3-BI-005'. */
  id: string;
  /** YAML key, e.g. 'trx'. */
  key: string;
  /** Display name, e.g. 'Total Prescriptions (TRx)'. */
  name: string;
  /** Registry workstream the KPI belongs to. */
  workstream: string;
  /** Flint semantic type for this KPI's values. */
  semanticType: KpiSemanticType;
  /** Declared threshold target, drawn as the goal on a KPI Card.
   *  Absent for KPIs the registry gives no target. */
  target?: number;
  /** Normalized strings that resolve to this KPI. */
  aliases: string[];
}

export const KPI_CATALOG: readonly KpiCatalogEntry[] = [
  {"id": "WS1-DQ-001", "key": "source_coverage_patients", "name": "Source Coverage - Patients", "workstream": "ws1_data_quality", "semanticType": "Percentage", "target": 0.85, "aliases": ["source_coverage_patients", "ws1_dq_001"]},
  {"id": "WS1-DQ-002", "key": "source_coverage_hcps", "name": "Source Coverage - HCPs", "workstream": "ws1_data_quality", "semanticType": "Percentage", "target": 0.8, "aliases": ["source_coverage_hcps", "ws1_dq_002"]},
  {"id": "WS1-DQ-003", "key": "cross_source_match_rate", "name": "Cross-source Match Rate", "workstream": "ws1_data_quality", "semanticType": "Percentage", "target": 0.75, "aliases": ["cross_source_match_rate", "ws1_dq_003"]},
  {"id": "WS1-DQ-004", "key": "stacking_lift", "name": "Stacking Lift", "workstream": "ws1_data_quality", "semanticType": "Percentage", "target": 0.15, "aliases": ["stacking_lift", "ws1_dq_004"]},
  {"id": "WS1-DQ-005", "key": "completeness_pass_rate", "name": "Completeness Pass Rate", "workstream": "ws1_data_quality", "semanticType": "Percentage", "target": 0.95, "aliases": ["completeness_pass_rate", "ws1_dq_005"]},
  {"id": "WS1-DQ-006", "key": "geographic_consistency", "name": "Geographic Consistency Gap", "workstream": "ws1_data_quality", "semanticType": "Percentage", "target": 0.05, "aliases": ["geographic_consistency", "geographic_consistency_gap", "ws1_dq_006"]},
  {"id": "WS1-DQ-007", "key": "data_lag_median", "name": "Data Lag (Median)", "workstream": "ws1_data_quality", "semanticType": "Duration", "target": 3, "aliases": ["data_lag", "data_lag_(median)", "data_lag_median", "median", "ws1_dq_007"]},
  {"id": "WS1-DQ-009", "key": "time_to_release", "name": "Time-to-Release (TTR)", "workstream": "ws1_data_quality", "semanticType": "Duration", "target": 24, "aliases": ["time_to_release", "time_to_release_(ttr)", "ttr", "ws1_dq_009"]},
  {"id": "WS1-MP-001", "key": "roc_auc", "name": "ROC-AUC", "workstream": "ws1_model_performance", "semanticType": "Score", "target": 0.8, "aliases": ["roc_auc", "ws1_mp_001"]},
  {"id": "WS1-MP-002", "key": "pr_auc", "name": "PR-AUC", "workstream": "ws1_model_performance", "semanticType": "Score", "target": 0.7, "aliases": ["pr_auc", "ws1_mp_002"]},
  {"id": "WS1-MP-003", "key": "f1_score", "name": "F1 Score", "workstream": "ws1_model_performance", "semanticType": "Score", "target": 0.65, "aliases": ["f1_score", "ws1_mp_003"]},
  {"id": "WS1-MP-004", "key": "recall_at_top_k", "name": "Recall@Top-K", "workstream": "ws1_model_performance", "semanticType": "Score", "target": 0.6, "aliases": ["recall@top_k", "recall_at_top_k", "ws1_mp_004"]},
  {"id": "WS1-MP-005", "key": "brier_score", "name": "Brier Score", "workstream": "ws1_model_performance", "semanticType": "Score", "target": 0.185, "aliases": ["brier_score", "ws1_mp_005"]},
  {"id": "WS1-MP-006", "key": "calibration_slope", "name": "Calibration Slope Deviation", "workstream": "ws1_model_performance", "semanticType": "Score", "aliases": ["calibration_slope", "calibration_slope_deviation", "ws1_mp_006"]},
  {"id": "WS1-MP-007", "key": "shap_coverage", "name": "SHAP Coverage", "workstream": "ws1_model_performance", "semanticType": "Percentage", "target": 0.95, "aliases": ["shap_coverage", "ws1_mp_007"]},
  {"id": "WS1-MP-009", "key": "feature_drift_psi", "name": "Feature Drift (PSI)", "workstream": "ws1_model_performance", "semanticType": "Score", "target": 0.1, "aliases": ["feature_drift", "feature_drift_(psi)", "feature_drift_psi", "psi", "ws1_mp_009"]},
  {"id": "WS2-TR-001", "key": "trigger_precision", "name": "Trigger Precision", "workstream": "ws2_triggers", "semanticType": "Percentage", "target": 0.7, "aliases": ["trigger_precision", "ws2_tr_001"]},
  {"id": "WS2-TR-002", "key": "trigger_recall", "name": "Trigger Recall", "workstream": "ws2_triggers", "semanticType": "Percentage", "target": 0.6, "aliases": ["trigger_recall", "ws2_tr_002"]},
  {"id": "WS2-TR-003", "key": "action_rate_uplift", "name": "Action Rate Uplift", "workstream": "ws2_triggers", "semanticType": "Percentage", "target": 0.15, "aliases": ["action_rate_uplift", "ws2_tr_003"]},
  {"id": "WS2-TR-004", "key": "acceptance_rate", "name": "Acceptance Rate", "workstream": "ws2_triggers", "semanticType": "Percentage", "target": 0.6, "aliases": ["acceptance_rate", "ws2_tr_004"]},
  {"id": "WS2-TR-005", "key": "false_alert_rate", "name": "False Alert Rate", "workstream": "ws2_triggers", "semanticType": "Percentage", "target": 0.1, "aliases": ["false_alert_rate", "ws2_tr_005"]},
  {"id": "WS2-TR-006", "key": "override_rate", "name": "Override Rate", "workstream": "ws2_triggers", "semanticType": "Percentage", "target": 0.15, "aliases": ["override_rate", "ws2_tr_006"]},
  {"id": "WS2-TR-007", "key": "lead_time_days", "name": "Lead Time", "workstream": "ws2_triggers", "semanticType": "Duration", "target": 14, "aliases": ["lead_time", "lead_time_days", "ws2_tr_007"]},
  {"id": "WS2-TR-008", "key": "change_fail_rate", "name": "Change-Fail Rate (CFR)", "workstream": "ws2_triggers", "semanticType": "Percentage", "target": 0.1, "aliases": ["cfr", "change_fail_rate", "change_fail_rate_(cfr)", "ws2_tr_008"]},
  {"id": "WS2-TR-009", "key": "trigger_funnel_conversion", "name": "Trigger Funnel Conversion", "workstream": "ws2_triggers", "semanticType": "Percentage", "aliases": ["trigger_funnel_conversion", "ws2_tr_009"]},
  {"id": "WS3-BI-001", "key": "active_users_mau", "name": "Monthly Active Users (MAU)", "workstream": "ws3_business", "semanticType": "Count", "target": 2000, "aliases": ["active_users_mau", "mau", "monthly_active_users", "monthly_active_users_(mau)", "ws3_bi_001"]},
  {"id": "WS3-BI-002", "key": "active_users_wau", "name": "Weekly Active Users (WAU)", "workstream": "ws3_business", "semanticType": "Count", "target": 1200, "aliases": ["active_users_wau", "wau", "weekly_active_users", "weekly_active_users_(wau)", "ws3_bi_002"]},
  {"id": "WS3-BI-003", "key": "patient_touch_rate", "name": "Patient Touch Rate", "workstream": "ws3_business", "semanticType": "Percentage", "target": 0.4, "aliases": ["patient_touch_rate", "ws3_bi_003"]},
  {"id": "WS3-BI-004", "key": "hcp_coverage", "name": "HCP Coverage", "workstream": "ws3_business", "semanticType": "Percentage", "target": 0.75, "aliases": ["hcp_coverage", "ws3_bi_004"]},
  {"id": "WS3-BI-005", "key": "trx", "name": "Total Prescriptions (TRx)", "workstream": "ws3_business", "semanticType": "Count", "aliases": ["total_prescriptions", "total_prescriptions_(trx)", "trx", "ws3_bi_005"]},
  {"id": "WS3-BI-006", "key": "nrx", "name": "New Prescriptions (NRx)", "workstream": "ws3_business", "semanticType": "Count", "aliases": ["new_prescriptions", "new_prescriptions_(nrx)", "nrx", "ws3_bi_006"]},
  {"id": "WS3-BI-007", "key": "nbrx", "name": "New-to-Brand Prescriptions (NBRx)", "workstream": "ws3_business", "semanticType": "Count", "aliases": ["nbrx", "new_to_brand_prescriptions", "new_to_brand_prescriptions_(nbrx)", "ws3_bi_007"]},
  {"id": "WS3-BI-008", "key": "trx_share", "name": "TRx Share", "workstream": "ws3_business", "semanticType": "Percentage", "target": 0.3, "aliases": ["trx_share", "ws3_bi_008"]},
  {"id": "WS3-BI-009", "key": "conversion_rate", "name": "Conversion Rate", "workstream": "ws3_business", "semanticType": "Percentage", "target": 0.08, "aliases": ["conversion_rate", "ws3_bi_009"]},
  {"id": "WS3-BI-010", "key": "roi", "name": "Return on Investment", "workstream": "ws3_business", "semanticType": "Number", "target": 3.0, "aliases": ["return_on_investment", "roi", "ws3_bi_010"]},
  {"id": "BR-001", "key": "remi_ah_uncontrolled_pct", "name": "Remi - AH Uncontrolled %", "workstream": "brand_specific", "semanticType": "Percentage", "target": 0.4, "aliases": ["br_001", "remi_ah_uncontrolled_%", "remi_ah_uncontrolled_pct"]},
  {"id": "BR-002", "key": "remi_intent_to_prescribe_delta", "name": "Remi - Intent-to-Prescribe Δ", "workstream": "brand_specific", "semanticType": "Score", "target": 0.5, "aliases": ["br_002", "remi_intent_to_prescribe_delta", "remi_intent_to_prescribe_δ"]},
  {"id": "BR-003", "key": "fabhalta_pnh_tested_pct", "name": "Fabhalta - % PNH Tested", "workstream": "brand_specific", "semanticType": "Percentage", "target": 0.6, "aliases": ["br_003", "fabhalta_%_pnh_tested", "fabhalta_pnh_tested_pct"]},
  {"id": "BR-004", "key": "kisqali_dx_adoption", "name": "Kisqali - Dx Adoption", "workstream": "brand_specific", "semanticType": "Duration", "target": 30, "aliases": ["br_004", "kisqali_dx_adoption"]},
  {"id": "BR-005", "key": "kisqali_oncologist_reach", "name": "Kisqali - Oncologist Reach", "workstream": "brand_specific", "semanticType": "Percentage", "target": 0.7, "aliases": ["br_005", "kisqali_oncologist_reach"]},
  {"id": "CM-001", "key": "treatment_effect_ate", "name": "Average Treatment Effect (ATE)", "workstream": "causal_metrics", "semanticType": "Number", "aliases": ["ate", "average_treatment_effect", "average_treatment_effect_(ate)", "cm_001", "treatment_effect_ate"]},
  {"id": "CM-002", "key": "treatment_effect_cate", "name": "Conditional ATE (CATE)", "workstream": "causal_metrics", "semanticType": "Number", "aliases": ["cate", "cm_002", "conditional_ate", "conditional_ate_(cate)", "treatment_effect_cate"]},
  {"id": "CM-003", "key": "causal_impact", "name": "Causal Impact", "workstream": "causal_metrics", "semanticType": "Number", "aliases": ["causal_impact", "cm_003"]},
  {"id": "CM-004", "key": "counterfactual_outcome", "name": "Counterfactual Outcome", "workstream": "causal_metrics", "semanticType": "Number", "aliases": ["cm_004", "counterfactual_outcome"]},
  {"id": "CM-005", "key": "mediation_effect", "name": "Mediation Effect", "workstream": "causal_metrics", "semanticType": "Number", "aliases": ["cm_005", "mediation_effect"]},
] as const;

/** region_type enum labels (US census regions) — SSOT: src/services/enum_labels.py (#1538). */
export const REGION_LABELS: readonly string[] = ["northeast", "south", "midwest", "west"] as const;

/**
 * Folded region alias -> enum label, mirroring enum_labels.REGION_ALIASES
 * (the platform's one region synonym table). Keys are folded the way
 * `fold_region_key` folds: casefolded with space/hyphen/underscore removed —
 * `resolveRegion` in kpi-alias.ts folds lookups the same way.
 */
export const REGION_ALIAS_MAP: Readonly<Record<string, string>> = {"central": "midwest", "midwest": "midwest", "mw": "midwest", "ne": "northeast", "newengland": "northeast", "northeast": "northeast", "northwest": "west", "nw": "west", "pacific": "west", "se": "south", "south": "south", "southeast": "south", "southern": "south", "southwest": "south", "sw": "south", "west": "west", "western": "west"} as const;
