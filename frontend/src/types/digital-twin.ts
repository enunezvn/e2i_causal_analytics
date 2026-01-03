/**
 * Digital Twin API Types
 * ======================
 *
 * TypeScript types for the E2I Digital Twin simulation endpoints.
 * Used for intervention pre-screening and scenario analysis.
 *
 * @module types/digital-twin
 */

// =============================================================================
// ENUMS
// =============================================================================

/**
 * Types of interventions that can be simulated
 */
export enum InterventionType {
  /** HCP engagement campaigns */
  HCP_ENGAGEMENT = 'hcp_engagement',
  /** Patient support programs */
  PATIENT_SUPPORT = 'patient_support',
  /** Pricing changes */
  PRICING = 'pricing',
  /** Rep training programs */
  REP_TRAINING = 'rep_training',
  /** Digital marketing campaigns */
  DIGITAL_MARKETING = 'digital_marketing',
  /** Formulary access initiatives */
  FORMULARY_ACCESS = 'formulary_access',
}

/**
 * Simulation recommendation outcomes
 */
export enum RecommendationType {
  /** Deploy intervention as designed */
  DEPLOY = 'deploy',
  /** Skip this intervention */
  SKIP = 'skip',
  /** Refine intervention parameters */
  REFINE = 'refine',
  /** Run additional analysis */
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
 * Request to run a digital twin simulation
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
 * Request to compare multiple scenarios
 */
export interface ScenarioComparisonRequest {
  /** Base scenario (control) */
  base_scenario: SimulationRequest;
  /** Alternative scenarios to compare */
  alternative_scenarios: SimulationRequest[];
  /** Metrics to compare */
  comparison_metrics?: string[];
}

// =============================================================================
// RESPONSE TYPES
// =============================================================================

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
 * Recommendation from the digital twin
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
 * Complete simulation response
 */
export interface SimulationResponse {
  /** Unique simulation ID */
  simulation_id: string;
  /** Timestamp of simulation */
  created_at: string;
  /** Input request */
  request: SimulationRequest;
  /** Primary outcomes */
  outcomes: SimulationOutcomes;
  /** Model fidelity metrics */
  fidelity: FidelityMetrics;
  /** Sensitivity analysis */
  sensitivity: SensitivityResult[];
  /** Recommendation */
  recommendation: SimulationRecommendation;
  /** Time series projections */
  projections: ProjectionDataPoint[];
  /** Execution time in ms */
  execution_time_ms: number;
}

/**
 * Scenario comparison result
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

/**
 * List of historical simulations
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

// =============================================================================
// UI COMPONENT TYPES
// =============================================================================

/**
 * Simulation panel form values
 */
export interface SimulationFormValues {
  interventionType: InterventionType;
  brand: string;
  sampleSize: number;
  durationDays: number;
  targetRegions: string[];
  targetSegments: string[];
  budget: number | undefined;
}

/**
 * Scenario card display data
 */
export interface ScenarioCardData {
  id: string;
  title: string;
  interventionType: InterventionType;
  ate: ConfidenceInterval;
  roi: ConfidenceInterval;
  recommendation: RecommendationType;
  confidence: ConfidenceLevel;
  isSelected?: boolean;
}
