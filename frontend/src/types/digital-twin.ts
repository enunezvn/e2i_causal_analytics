/**
 * Digital Twin API Types
 * ======================
 *
 * TypeScript types for the E2I Digital Twin simulation endpoints.
 * Based on src/api/routes/digital_twin.py backend schemas.
 *
 * Supports:
 * - Twin simulation for interventions
 * - Simulation history and detail retrieval
 * - Fidelity validation against actual experiments
 * - Twin model management
 * - Fidelity tracking and reporting
 *
 * @module types/digital-twin
 */

// =============================================================================
// ENUMS
// =============================================================================

/**
 * Types of digital twins
 */
export enum TwinType {
  HCP = 'hcp',
  PATIENT = 'patient',
  TERRITORY = 'territory',
}

/**
 * Pharmaceutical brands
 */
export enum Brand {
  REMIBRUTINIB = 'Remibrutinib',
  FABHALTA = 'Fabhalta',
  KISQALI = 'Kisqali',
}

/**
 * Simulation status values
 */
export enum SimulationStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
}

/**
 * Simulation recommendations
 */
export enum Recommendation {
  DEPLOY = 'deploy',
  SKIP = 'skip',
  REFINE = 'refine',
}

/**
 * Fidelity grade values
 */
export enum FidelityGrade {
  EXCELLENT = 'excellent',
  GOOD = 'good',
  FAIR = 'fair',
  POOR = 'poor',
  UNVALIDATED = 'unvalidated',
}

// =============================================================================
// LEGACY ENUMS (Kept for backward compatibility)
// =============================================================================

/**
 * Types of interventions that can be simulated
 * @deprecated Use intervention_type string field instead
 */
export enum InterventionType {
  HCP_ENGAGEMENT = 'hcp_engagement',
  PATIENT_SUPPORT = 'patient_support',
  PRICING = 'pricing',
  REP_TRAINING = 'rep_training',
  DIGITAL_MARKETING = 'digital_marketing',
  FORMULARY_ACCESS = 'formulary_access',
}

/**
 * @deprecated Use Recommendation enum instead
 */
export enum RecommendationType {
  DEPLOY = 'deploy',
  SKIP = 'skip',
  REFINE = 'refine',
  ANALYZE = 'analyze',
}

/**
 * Confidence levels for simulation results
 */
export enum ConfidenceLevel {
  HIGH = 'high',
  MEDIUM = 'medium',
  LOW = 'low',
}

// =============================================================================
// REQUEST TYPES
// =============================================================================

/**
 * Configuration for an intervention to simulate
 */
export interface InterventionConfigRequest {
  /** Type of intervention (email_campaign, call_frequency_increase, etc.) */
  intervention_type: string;
  /** Channel: email, call, in_person, digital */
  channel?: string;
  /** Frequency: daily, weekly, monthly */
  frequency?: string;
  /** Duration in weeks (1-52) */
  duration_weeks?: number;
  /** Content type: clinical_data, patient_stories, etc. */
  content_type?: string;
  /** Personalization level: none, standard, high */
  personalization_level?: string;
  /** Target segment identifier */
  target_segment?: string;
  /** Target deciles (1-10) */
  target_deciles?: number[];
  /** Target specialty list */
  target_specialties?: string[];
  /** Target region list */
  target_regions?: string[];
  /** Treatment intensity multiplier (0.1-10.0) */
  intensity_multiplier?: number;
  /** Additional parameters */
  extra_params?: Record<string, unknown>;
}

/**
 * Filters for selecting twin population
 */
export interface PopulationFilterRequest {
  /** Filter by specialties */
  specialties?: string[];
  /** Filter by deciles (1-10) */
  deciles?: number[];
  /** Filter by regions */
  regions?: string[];
  /** Filter by adoption stages */
  adoption_stages?: string[];
  /** Minimum baseline outcome */
  min_baseline_outcome?: number;
  /** Maximum baseline outcome */
  max_baseline_outcome?: number;
}

/**
 * Request to run a twin simulation
 */
export interface SimulateRequest {
  /** Intervention configuration */
  intervention: InterventionConfigRequest;
  /** Target brand */
  brand: Brand | string;
  /** Twin type (default: hcp) */
  twin_type?: TwinType | string;
  /** Population filters */
  population_filters?: PopulationFilterRequest;
  /** Number of twins to simulate (100-100000) */
  twin_count?: number;
  /** Confidence level for CI (0.8-0.99) */
  confidence_level?: number;
  /** Calculate heterogeneous effects */
  calculate_heterogeneity?: boolean;
  /** Specific model ID to use */
  model_id?: string;
  /** Link to experiment design */
  experiment_design_id?: string;
}

/**
 * Request to validate simulation against actual results
 */
export interface ValidateFidelityRequest {
  /** Simulation ID to validate */
  simulation_id: string;
  /** Actual experiment ID */
  experiment_id: string;
  /** Actual Average Treatment Effect */
  actual_ate: number;
  /** Actual CI lower bound */
  actual_ci_lower?: number;
  /** Actual CI upper bound */
  actual_ci_upper?: number;
  /** Actual sample size */
  actual_sample_size?: number;
  /** Notes on validation */
  validation_notes?: string;
  /** Known confounding factors */
  confounding_factors?: string[];
  /** Validator identifier */
  validated_by?: string;
}

// =============================================================================
// RESPONSE TYPES
// =============================================================================

/**
 * Heterogeneous effects across subgroups
 */
export interface EffectHeterogeneityResponse {
  /** Effects by specialty */
  by_specialty: Record<string, Record<string, number>>;
  /** Effects by decile */
  by_decile: Record<string, Record<string, number>>;
  /** Effects by region */
  by_region: Record<string, Record<string, number>>;
  /** Effects by adoption stage */
  by_adoption_stage: Record<string, Record<string, number>>;
  /** Top performing segments */
  top_segments: Array<Record<string, unknown>>;
}

/**
 * Response from a simulation run
 */
export interface SimulationResponse {
  /** Unique simulation ID */
  simulation_id: string;
  /** Model ID used */
  model_id: string;
  /** Type of intervention */
  intervention_type: string;
  /** Brand */
  brand: string;
  /** Twin type */
  twin_type: string;
  /** Number of twins simulated */
  twin_count: number;
  /** Simulated Average Treatment Effect */
  simulated_ate: number;
  /** CI lower bound */
  simulated_ci_lower: number;
  /** CI upper bound */
  simulated_ci_upper: number;
  /** Standard error */
  simulated_std_error: number;
  /** Cohen's d effect size */
  effect_size_cohens_d?: number;
  /** Statistical power */
  statistical_power?: number;
  /** Recommendation (deploy/skip/refine) */
  recommendation: Recommendation;
  /** Recommendation rationale */
  recommendation_rationale: string;
  /** Recommended sample size */
  recommended_sample_size?: number;
  /** Recommended duration in weeks */
  recommended_duration_weeks?: number;
  /** Simulation confidence score */
  simulation_confidence: number;
  /** Whether fidelity warning is present */
  fidelity_warning: boolean;
  /** Fidelity warning reason */
  fidelity_warning_reason?: string;
  /** Model fidelity score */
  model_fidelity_score?: number;
  /** Simulation status */
  status: SimulationStatus;
  /** Error message if failed */
  error_message?: string;
  /** Execution time in ms */
  execution_time_ms: number;
  /** Whether effect is statistically significant */
  is_significant: boolean;
  /** Effect direction (positive/negative/neutral) */
  effect_direction: string;
  /** Creation timestamp */
  created_at: string;
}

/**
 * Detailed simulation response including heterogeneity
 */
export interface SimulationDetailResponse extends SimulationResponse {
  /** Population filters used */
  population_filters: Record<string, unknown>;
  /** Effect heterogeneity by subgroup */
  effect_heterogeneity: EffectHeterogeneityResponse;
  /** Full intervention config */
  intervention_config: Record<string, unknown>;
  /** Completion timestamp */
  completed_at?: string;
}

/**
 * Summary item for simulation list
 */
export interface SimulationListItem {
  /** Simulation ID */
  simulation_id: string;
  /** Intervention type */
  intervention_type: string;
  /** Brand */
  brand: string;
  /** Twin type */
  twin_type: string;
  /** Number of twins */
  twin_count: number;
  /** Simulated ATE */
  simulated_ate: number;
  /** Recommendation */
  recommendation: Recommendation;
  /** Status */
  status: SimulationStatus;
  /** Creation timestamp */
  created_at: string;
}

/**
 * Response for listing simulations
 */
export interface SimulationListResponse {
  /** Total count of simulations */
  total_count: number;
  /** Simulation list */
  simulations: SimulationListItem[];
  /** Current page */
  page: number;
  /** Page size */
  page_size: number;
}

/**
 * Fidelity validation record
 */
export interface FidelityRecordResponse {
  /** Tracking record ID */
  tracking_id: string;
  /** Simulation ID */
  simulation_id: string;
  /** Linked experiment ID */
  experiment_id?: string;
  /** Simulated ATE */
  simulated_ate: number;
  /** Simulated CI lower */
  simulated_ci_lower?: number;
  /** Simulated CI upper */
  simulated_ci_upper?: number;
  /** Actual ATE from experiment */
  actual_ate?: number;
  /** Actual CI lower */
  actual_ci_lower?: number;
  /** Actual CI upper */
  actual_ci_upper?: number;
  /** Actual sample size */
  actual_sample_size?: number;
  /** Prediction error (actual - simulated) */
  prediction_error?: number;
  /** Absolute prediction error */
  absolute_error?: number;
  /** Whether simulated CI covered actual */
  ci_coverage?: boolean;
  /** Fidelity grade */
  fidelity_grade: FidelityGrade;
  /** Validation notes */
  validation_notes?: string;
  /** Known confounding factors */
  confounding_factors: string[];
  /** Record creation timestamp */
  created_at: string;
  /** Validation timestamp */
  validated_at?: string;
  /** Validator identifier */
  validated_by?: string;
}

/**
 * Summary of a twin generator model
 */
export interface TwinModelSummary {
  /** Model ID */
  model_id: string;
  /** Model name */
  model_name: string;
  /** Twin type */
  twin_type: string;
  /** Brand */
  brand: string;
  /** Algorithm used */
  algorithm: string;
  /** R² score */
  r2_score?: number;
  /** RMSE */
  rmse?: number;
  /** Number of training samples */
  training_samples: number;
  /** Whether model is active */
  is_active: boolean;
  /** Creation timestamp */
  created_at: string;
}

/**
 * Detailed twin model information
 */
export interface TwinModelDetailResponse extends TwinModelSummary {
  /** Model description */
  model_description?: string;
  /** Feature columns used */
  feature_columns: string[];
  /** Target column */
  target_column: string;
  /** Cross-validation mean score */
  cv_mean?: number;
  /** Cross-validation std deviation */
  cv_std?: number;
  /** Feature importances */
  feature_importances: Record<string, number>;
  /** Top features list */
  top_features: string[];
  /** Training duration in seconds */
  training_duration_seconds: number;
  /** Model configuration */
  config: Record<string, unknown>;
}

/**
 * Response for listing models
 */
export interface ModelListResponse {
  /** Total count */
  total_count: number;
  /** List of models */
  models: TwinModelSummary[];
}

/**
 * Fidelity history for a model
 */
export interface FidelityHistoryResponse {
  /** Model ID */
  model_id: string;
  /** Total validation records */
  total_validations: number;
  /** Average fidelity score */
  average_fidelity_score?: number;
  /** Distribution by grade */
  grade_distribution: Record<string, number>;
  /** Fidelity records */
  records: FidelityRecordResponse[];
}

/**
 * Aggregated fidelity report for a model
 */
export interface FidelityReportResponse {
  /** Model ID */
  model_id: string;
  /** Total validations */
  total_validations: number;
  /** Average fidelity score */
  average_fidelity_score: number;
  /** CI coverage rate */
  coverage_rate: number;
  /** Grade distribution */
  grade_distribution: Record<string, number>;
  /** Trend direction */
  trend: string;
  /** Whether model is degrading */
  is_degrading: boolean;
  /** Degradation rate if applicable */
  degradation_rate?: number;
  /** Recommendation for the model */
  recommendation: string;
  /** Report generation timestamp */
  generated_at: string;
}

/**
 * Health status for Digital Twin service
 */
export interface DigitalTwinHealthResponse {
  /** Service health status */
  status: string;
  /** Service name */
  service: string;
  /** Number of models available */
  models_available: number;
  /** Number of pending simulations */
  simulations_pending: number;
  /** Timestamp of last simulation */
  last_simulation_at?: string;
}

// =============================================================================
// LEGACY TYPES (Kept for backward compatibility)
// =============================================================================

/**
 * @deprecated Use SimulateRequest instead
 */
export interface SimulationRequest {
  /** Type of intervention to simulate */
  intervention_type: InterventionType;
  /** Target brand for the intervention */
  brand: string;
  /** Sample size for treatment group */
  sample_size: number;
  /** Duration in days */
  duration_days: number;
  /** Target regions (optional) */
  target_regions?: string[];
  /** Target HCP segments (optional) */
  target_segments?: string[];
  /** Budget allocation in dollars */
  budget?: number;
  /** Custom intervention parameters */
  parameters?: Record<string, unknown>;
}

/**
 * @deprecated Use separate scenario simulations
 */
export interface ScenarioComparisonRequest {
  /** Base scenario (control) */
  base_scenario: SimulationRequest;
  /** Alternative scenarios to compare */
  alternative_scenarios: SimulationRequest[];
  /** Metrics to compare */
  comparison_metrics?: string[];
}

/**
 * Confidence interval for estimated values
 */
export interface ConfidenceInterval {
  /** Lower bound (2.5th percentile) */
  lower: number;
  /** Point estimate (median) */
  estimate: number;
  /** Upper bound (97.5th percentile) */
  upper: number;
}

/**
 * Primary outcome metrics from simulation
 * @deprecated Use SimulationResponse fields directly
 */
export interface SimulationOutcomes {
  /** Average Treatment Effect */
  ate: ConfidenceInterval;
  /** Conditional Average Treatment Effect by segment */
  cate_by_segment?: Record<string, ConfidenceInterval>;
  /** Expected TRx lift */
  trx_lift: ConfidenceInterval;
  /** Expected NRx lift */
  nrx_lift: ConfidenceInterval;
  /** Expected market share change */
  market_share_change: ConfidenceInterval;
  /** ROI projection */
  roi: ConfidenceInterval;
  /** Number needed to treat for one additional TRx */
  nnt?: number;
}

/**
 * Fidelity metrics for the simulation model
 * @deprecated Use model fidelity endpoints
 */
export interface FidelityMetrics {
  /** Overall fidelity score (0-1) */
  overall_score: number;
  /** Data coverage score */
  data_coverage: number;
  /** Model calibration score */
  calibration: number;
  /** Temporal alignment score */
  temporal_alignment: number;
  /** Feature completeness score */
  feature_completeness: number;
  /** Confidence level based on fidelity */
  confidence_level: ConfidenceLevel;
  /** Warnings or limitations */
  warnings?: string[];
}

/**
 * Sensitivity analysis results
 * @deprecated Part of detailed simulation response
 */
export interface SensitivityResult {
  /** Parameter name */
  parameter: string;
  /** Base value */
  base_value: number;
  /** Low scenario value */
  low_value: number;
  /** High scenario value */
  high_value: number;
  /** Impact on ATE at low value */
  ate_at_low: number;
  /** Impact on ATE at high value */
  ate_at_high: number;
  /** Sensitivity score (how much it affects outcome) */
  sensitivity_score: number;
}

/**
 * @deprecated Use SimulationResponse.recommendation fields
 */
export interface SimulationRecommendation {
  /** Recommendation type */
  type: RecommendationType;
  /** Confidence in recommendation */
  confidence: ConfidenceLevel;
  /** Primary rationale */
  rationale: string;
  /** Supporting evidence points */
  evidence: string[];
  /** Suggested refinements (if type is REFINE) */
  suggested_refinements?: Record<string, unknown>;
  /** Risk factors to consider */
  risk_factors?: string[];
  /** Expected value if recommendation followed */
  expected_value?: number;
}

/**
 * Time series projection data point
 * @deprecated Part of detailed simulation
 */
export interface ProjectionDataPoint {
  /** Date of projection */
  date: string;
  /** Projected value with intervention */
  with_intervention: number;
  /** Projected value without intervention */
  without_intervention: number;
  /** Uncertainty band lower */
  lower_bound: number;
  /** Uncertainty band upper */
  upper_bound: number;
}

/**
 * @deprecated Use SimulationListResponse instead
 */
export interface SimulationHistoryResponse {
  /** List of simulations */
  simulations: Array<{
    simulation_id: string;
    created_at: string;
    intervention_type: InterventionType;
    brand: string;
    ate_estimate: number;
    recommendation_type: RecommendationType;
  }>;
  /** Total count */
  total: number;
  /** Pagination offset */
  offset: number;
  /** Pagination limit */
  limit: number;
}

/**
 * @deprecated Use comparison via separate simulation calls
 */
export interface ScenarioComparisonResult {
  /** Base scenario simulation */
  base_result: SimulationResponse;
  /** Alternative scenario simulations */
  alternative_results: SimulationResponse[];
  /** Comparison summary */
  comparison: {
    /** Best performing scenario index (0 = base) */
    best_scenario_index: number;
    /** Metric-by-metric comparison */
    metric_comparison: Record<string, number[]>;
    /** Summary recommendation */
    summary: string;
  };
}

// =============================================================================
// UI COMPONENT TYPES
// =============================================================================

/**
 * Simulation panel form values
 */
export interface SimulationFormValues {
  interventionType: string;
  brand: Brand | string;
  twinType: TwinType | string;
  twinCount: number;
  durationWeeks: number;
  targetDeciles: number[];
  targetSpecialties: string[];
  targetRegions: string[];
  personalizationLevel: string;
  channel?: string;
}

/**
 * Scenario card display data
 */
export interface ScenarioCardData {
  id: string;
  title: string;
  interventionType: string;
  ate: number;
  ciLower: number;
  ciUpper: number;
  recommendation: Recommendation;
  confidence: number;
  isSelected?: boolean;
}

/**
 * Model card display data
 */
export interface ModelCardData {
  id: string;
  name: string;
  twinType: TwinType | string;
  brand: Brand | string;
  r2Score?: number;
  fidelityGrade?: FidelityGrade;
  isActive: boolean;
  lastUsed?: string;
}
