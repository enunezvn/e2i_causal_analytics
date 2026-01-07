/**
 * KPI Dictionary Content Data
 * ===========================
 *
 * Static content for KPI Dictionary tables copied from mock dashboard.
 * Organized by workstream:
 * - WS1: Data Coverage & Quality KPIs
 * - WS1: Model Performance KPIs
 * - WS2: Trigger Performance KPIs
 * - WS3: Business Impact KPIs
 * - Statistical & Causal Methods
 *
 * @module data/kpi-dictionary-content
 */

import type { KPITableRow } from '../components/visualizations/dashboard/KPITable';

// ============================================================================
// WS1 - Data Coverage & Quality KPIs
// ============================================================================

export const WS1_DATA_COVERAGE_KPIS: KPITableRow[] = [
  {
    name: 'Source Coverage - Patients',
    definition: '% of eligible patients present in source vs reference universe',
    formula: 'coverage = covered_patients / reference_patients',
  },
  {
    name: 'Source Coverage - HCPs',
    definition: '% of priority HCPs present in source vs universe',
    formula: 'coverage = covered_hcps / reference_hcps',
  },
  {
    name: 'Geographic Consistency',
    definition: 'Max absolute gap between source share and universe share across regions',
    formula: 'max_region |share_source - share_universe|',
  },
  {
    name: 'Data Lag (Median)',
    definition: 'Median days from service date to availability in warehouse',
    formula: 'median(datediff(ingested_ts, svc_date))',
  },
  {
    name: 'Cross-source Match Rate',
    definition: '% entities linkable across sources A & B',
    formula: 'match_rate = |A∩B| / |A∪B|',
  },
  {
    name: 'Stacking Lift',
    definition: 'Incremental value from combining multiple data sources',
    formula: 'lift = (stacked - baseline) / baseline',
  },
  {
    name: 'Completeness Pass Rate',
    definition: '1 − null rate across brand-critical fields',
    formula: 'pass = 1 − (null_critical / total)',
  },
  {
    name: 'Feature Drift (PSI)',
    definition: 'Population Stability Index measuring shift in feature distributions',
    formula: 'PSI = Σ_b (q_b - p_b) * ln(q_b / p_b)',
  },
];

// ============================================================================
// WS1 - Model Performance KPIs
// ============================================================================

export const WS1_MODEL_PERFORMANCE_KPIS: KPITableRow[] = [
  {
    name: 'ROC-AUC',
    definition: 'Area Under the ROC Curve - probability that a positive is ranked above a negative',
    formula: 'AUC = ∫TPR d(FPR)',
  },
  {
    name: 'PR-AUC',
    definition: 'Area under the Precision–Recall curve; preferable under class imbalance',
    formula: '∫Precision d(Recall)',
  },
  {
    name: 'F1 Score',
    definition: 'Harmonic mean of precision and recall at chosen threshold',
    formula: 'F1 = 2 * (Precision * Recall) / (Precision + Recall)',
  },
  {
    name: 'Recall@Top-K',
    definition: 'Fraction of all positives captured within the top K highest-scored cases',
    formula: 'TP_in_topK / Total_Positives',
  },
  {
    name: 'Brier Score',
    definition: 'Mean squared error of probabilistic predictions; lower is better',
    formula: 'Brier = (1/N) * Σ(p_i - y_i)²',
  },
  {
    name: 'Calibration Slope',
    definition: 'Slope from regressing outcomes on logit of predicted probabilities; 1.0 is ideal',
    formula: 'logit(Pr(y=1)) = α + β*logit(p̂)',
  },
  {
    name: 'SHAP Coverage',
    definition: '% of predictions where Top-N features explain ≥τ% of total attribution',
    formula: 'count(top_n_shap ≥ τ) / total_predictions',
  },
  {
    name: 'Label Quality (IAA)',
    definition: 'Inter-Annotator Agreement measuring label consistency',
    formula: "Cohen's κ or Krippendorff's α",
  },
  {
    name: 'Fairness Gap (ΔRecall)',
    definition: 'Worst-case recall gap across protected or relevant subgroups',
    formula: 'max_g(Recall_g) - min_g(Recall_g)',
  },
];

// ============================================================================
// WS2 - Trigger Performance KPIs
// ============================================================================

export const WS2_TRIGGER_PERFORMANCE_KPIS: KPITableRow[] = [
  {
    name: 'Precision (Trigger)',
    definition: 'Of all triggers fired, % that led to intended outcome within look-ahead window',
    formula: 'Precision = TP / (TP + FP)',
  },
  {
    name: 'Recall (Trigger)',
    definition: 'Of all outcomes that occurred, % preceded by a trigger within look-back window',
    formula: 'Recall = TP / (TP + FN)',
  },
  {
    name: 'Lead Time',
    definition: 'Median days from trigger to outcome for true positives',
    formula: 'median(outcome_ts - trigger_ts)',
  },
  {
    name: 'Action Rate Uplift',
    definition: 'Increase in desired actions among exposed vs matched control HCPs',
    formula: 'Uplift = Pr(action|exposed) - Pr(action|control)',
  },
  {
    name: 'Acceptance Rate',
    definition: '% of presented recommendations accepted (not snoozed/overridden)',
    formula: 'Acceptance = #accepted / #presented',
  },
  {
    name: 'Time-to-Release (TTR)',
    definition: 'Calendar days from change approval to production availability',
    formula: 'TTR = date_prod - date_approve',
  },
  {
    name: 'Change-Fail Rate (CFR)',
    definition: '% of releases requiring rollback or hotfix within 7 days',
    formula: 'CFR = #rollbacks(7d) / #releases',
  },
  {
    name: 'False Alert Rate',
    definition: '% of fired alerts without intended outcome within T days',
    formula: 'FalseAlert = #NoOutcomeInT / #Triggers',
  },
  {
    name: 'Override Rate',
    definition: '% of alerts manually overridden or marked "not relevant"',
    formula: 'Override = #overridden / #presented',
  },
];

// ============================================================================
// WS3 - Business Impact KPIs
// ============================================================================

export interface KPITableRowWithSection extends KPITableRow {
  /** Optional section header (e.g., "General Platform KPIs") */
  sectionHeader?: string;
}

export const WS3_BUSINESS_IMPACT_KPIS: KPITableRowWithSection[] = [
  // General Platform KPIs
  {
    sectionHeader: 'General Platform KPIs',
    name: 'Active Users (MAU/WAU)',
    definition: 'Monthly/Weekly Active Users performing ≥1 relevant action',
    formula: 'MAU = #{unique users with ≥1 action in month}',
  },
  {
    name: 'HCP Coverage',
    definition: '% of priority HCP list with ≥1 brand-aligned touch linked to triggers',
    formula: '%HCPs touched = touched_HCPs / target_HCPs',
  },
  {
    name: 'Patient Touch Rate',
    definition: '% of target patient list reviewed/flagged/acted upon in care settings',
    formula: '%Patients touched = target_patients_w_≥1_action / target_patients',
  },
  // Brand-Specific KPIs
  {
    sectionHeader: 'Brand-Specific KPIs',
    name: 'Remi - AH Uncontrolled %',
    definition: '% of antihistamine-uncontrolled CSU patients identified by model',
    formula: '%Identified = #AH_uncontrolled_flagged / #AH_uncontrolled_total',
  },
  {
    name: 'Remi - Intent-to-Prescribe Δ',
    definition: 'Change in HCPs reporting "likely to prescribe" vs baseline',
    formula: 'ITP_Δ = ITP_current - ITP_baseline',
  },
  {
    name: 'Fabhalta - % PNH Tested',
    definition: '% of suspected PNH patients who received flow cytometry testing',
    formula: '%tested = #suspected_with_test / #suspected_total',
  },
  {
    name: 'Kisqali - Oncologist Reach',
    definition: '% of priority oncologists with ≥1 brand-aligned engagement',
    formula: '%reached = #priority_HCPs_touched / #priority_HCPs',
  },
  {
    name: 'Kisqali - Dx Adoption',
    definition: '% priority oncologists with Oncotype Dx-guided prescriptions',
    formula: 'Uplift = Pr(adoption|test_aligned) - Pr(adoption|BAU)',
  },
];

// ============================================================================
// Statistical & Causal Methods
// ============================================================================

export interface MethodTableRow {
  /** Method name */
  method: string;
  /** Method description */
  description: string;
  /** Application in dashboard */
  application: string;
}

export const STATISTICAL_CAUSAL_METHODS: MethodTableRow[] = [
  {
    method: 'Difference-in-Differences (DiD)',
    description: 'Compares changes over time between treatment and control groups to isolate causal effect',
    application: 'Used for action rate uplift, ITP changes, testing rate improvements',
  },
  {
    method: 'Causal Forests',
    description: 'Non-parametric method for heterogeneous treatment effects using random forests',
    application: 'Identifies which HCP/patient subgroups benefit most from interventions',
  },
  {
    method: 'Double ML (DML)',
    description: 'Uses ML for nuisance parameters while maintaining valid causal inference',
    application: 'Estimates treatment effects with high-dimensional confounders in EMR data',
  },
  {
    method: 'Instrumental Variables (IV)',
    description: 'Uses external variation to identify causal effects when unmeasured confounding exists',
    application: 'Data lag analysis using system delays as instruments',
  },
  {
    method: 'Synthetic Control',
    description: 'Creates weighted combination of control units to match treated unit pre-intervention',
    application: 'Brand-specific impact assessment (Remi patient identification)',
  },
  {
    method: 'Regression Discontinuity',
    description: 'Exploits discontinuous treatment assignment at threshold for causal identification',
    application: 'Coverage threshold effects (70% threshold for model performance)',
  },
  {
    method: 'Mediation Analysis',
    description: 'Decomposes total effect into direct and indirect pathways',
    application: 'ITP improvement pathway: targeting→relevance→engagement',
  },
  {
    method: 'Meta-learners (S/T/X)',
    description: 'ML methods that estimate CATE by combining base learners differently',
    application: 'Heterogeneous treatment effect estimation across HCP segments',
  },
  {
    method: 'G-computation',
    description: 'Standardization method for estimating causal effects under sequential treatments',
    application: 'Multi-source data stacking lift analysis',
  },
  {
    method: 'Propensity Score Matching',
    description: 'Matches treated and control units on probability of treatment',
    application: 'Creating balanced comparison groups for trigger effectiveness',
  },
];

// ============================================================================
// All KPI Sections for easy import
// ============================================================================

export const KPI_DICTIONARY_SECTIONS = {
  ws1DataCoverage: {
    title: 'WS1 - Data Coverage & Quality KPIs',
    emoji: '📊',
    data: WS1_DATA_COVERAGE_KPIS,
  },
  ws1ModelPerformance: {
    title: 'WS1 - Model Performance KPIs',
    emoji: '🤖',
    data: WS1_MODEL_PERFORMANCE_KPIS,
  },
  ws2TriggerPerformance: {
    title: 'WS2 - Trigger Performance KPIs',
    emoji: '🎯',
    data: WS2_TRIGGER_PERFORMANCE_KPIS,
  },
  ws3BusinessImpact: {
    title: 'WS3 - Business Impact KPIs',
    emoji: '📈',
    data: WS3_BUSINESS_IMPACT_KPIS,
  },
  statisticalMethods: {
    title: 'Statistical & Causal Methods',
    emoji: '📐',
    data: STATISTICAL_CAUSAL_METHODS,
  },
};
