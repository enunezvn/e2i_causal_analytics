/**
 * A/B Testing & Experiments API Types
 * ====================================
 *
 * TypeScript interfaces for the E2I A/B Testing API.
 * Based on src/api/routes/experiments.py backend schemas.
 *
 * Supports:
 * - Randomization (simple, stratified, block)
 * - Enrollment management
 * - Interim analysis with alpha spending
 * - Results and heterogeneous effects
 * - SRM detection
 * - Digital Twin fidelity tracking
 * - Experiment monitoring and alerts
 *
 * @module types/experiments
 */

// =============================================================================
// ENUMS
// =============================================================================

/**
 * Randomization methods
 */
export enum RandomizationMethod {
  SIMPLE = 'simple',
  STRATIFIED = 'stratified',
  BLOCK = 'block',
}

/**
 * Enrollment status values
 */
export enum EnrollmentStatus {
  ACTIVE = 'active',
  WITHDRAWN = 'withdrawn',
  COMPLETED = 'completed',
}

/**
 * Analysis types
 */
export enum AnalysisType {
  INTERIM = 'interim',
  FINAL = 'final',
}

/**
 * Analysis methods
 */
export enum AnalysisMethod {
  ITT = 'itt',
  PER_PROTOCOL = 'per_protocol',
}

/**
 * Interim analysis stopping decisions
 */
export enum StoppingDecision {
  CONTINUE = 'continue',
  STOP_EFFICACY = 'stop_efficacy',
  STOP_FUTILITY = 'stop_futility',
}

/**
 * Alert severity levels
 */
export enum AlertSeverity {
  CRITICAL = 'critical',
  WARNING = 'warning',
  INFO = 'info',
}

/**
 * Experiment health status
 */
export enum ExperimentHealthStatus {
  HEALTHY = 'healthy',
  WARNING = 'warning',
  CRITICAL = 'critical',
}

// =============================================================================
// REQUEST TYPES
// =============================================================================

/**
 * Unit to randomize
 */
export interface RandomizeUnit {
  /** Unit identifier */
  unit_id: string;
  /** Unit type (hcp, patient, territory) */
  unit_type: string;
  /** Optional stratification variables */
  [key: string]: unknown;
}

/**
 * Request to randomize units to experiment variants
 */
export interface RandomizeRequest {
  /** List of units to randomize */
  units: RandomizeUnit[];
  /** Randomization method */
  method?: RandomizationMethod;
  /** Columns to use for stratification */
  strata_columns?: string[];
  /** Allocation ratio by variant */
  allocation_ratio?: Record<string, number>;
  /** Block size for block randomization */
  block_size?: number;
}

/**
 * Request to enroll a unit in an experiment
 */
export interface EnrollUnitRequest {
  /** Unit identifier (HCP, patient, etc.) */
  unit_id: string;
  /** Type of unit (hcp, patient, territory) */
  unit_type: string;
  /** When consent was obtained */
  consent_timestamp?: string;
  /** Eligibility criteria evaluation results */
  eligibility_criteria_met?: Record<string, unknown>;
}

/**
 * Request to withdraw a unit from an experiment
 */
export interface WithdrawRequest {
  /** Reason for withdrawal */
  reason: string;
}

/**
 * Request to trigger an interim analysis
 */
export interface TriggerInterimAnalysisRequest {
  /** Specific analysis number (auto-detected if not provided) */
  analysis_number?: number;
  /** Force analysis even if milestone not reached */
  force?: boolean;
}

/**
 * Request to trigger experiment monitoring
 */
export interface TriggerMonitorRequest {
  /** Specific experiments to check (all active if not provided) */
  experiment_ids?: string[];
  /** Check for SRM */
  check_srm?: boolean;
  /** Check enrollment rates */
  check_enrollment?: boolean;
  /** Check Digital Twin fidelity */
  check_fidelity?: boolean;
  /** SRM p-value threshold */
  srm_threshold?: number;
}

/**
 * Parameters for getting assignments
 */
export interface GetAssignmentsParams {
  /** Filter by variant */
  variant?: string;
  /** Filter by unit type */
  unit_type?: string;
  /** Max assignments to return */
  limit?: number;
  /** Pagination offset */
  offset?: number;
}

/**
 * Parameters for getting experiment results
 */
export interface GetResultsParams {
  /** Type of analysis (interim or final) */
  analysis_type?: AnalysisType;
  /** Analysis method (ITT or per-protocol) */
  analysis_method?: AnalysisMethod;
  /** Force recomputation */
  recompute?: boolean;
}

// =============================================================================
// RESPONSE TYPES
// =============================================================================

/**
 * Result of a single assignment
 */
export interface AssignmentResult {
  /** Assignment ID */
  assignment_id: string;
  /** Experiment ID */
  experiment_id: string;
  /** Unit ID */
  unit_id: string;
  /** Unit type */
  unit_type: string;
  /** Assigned variant */
  variant: string;
  /** Assignment timestamp */
  assigned_at: string;
  /** Randomization method used */
  randomization_method: string;
  /** Stratification key if applicable */
  stratification_key?: Record<string, unknown>;
  /** Block ID if block randomization */
  block_id?: string;
}

/**
 * Response from randomization
 */
export interface RandomizeResponse {
  /** Experiment ID */
  experiment_id: string;
  /** Total units randomized */
  total_units: number;
  /** Assignment results */
  assignments: AssignmentResult[];
  /** Count by variant */
  variant_counts: Record<string, number>;
  /** Method used */
  randomization_method: string;
  /** Timestamp */
  timestamp: string;
}

/**
 * Response listing assignments
 */
export interface AssignmentsListResponse {
  /** Experiment ID */
  experiment_id: string;
  /** Total count */
  total_count: number;
  /** Assignments */
  assignments: AssignmentResult[];
}

/**
 * Result of enrollment
 */
export interface EnrollmentResult {
  /** Enrollment ID */
  enrollment_id: string;
  /** Assignment ID */
  assignment_id: string;
  /** Experiment ID */
  experiment_id: string;
  /** Unit ID */
  unit_id: string;
  /** Assigned variant */
  variant: string;
  /** Enrollment timestamp */
  enrolled_at: string;
  /** Enrollment status */
  enrollment_status: EnrollmentStatus;
  /** Consent timestamp */
  consent_timestamp?: string;
}

/**
 * Enrollment statistics for an experiment
 */
export interface EnrollmentStatsResponse {
  /** Experiment ID */
  experiment_id: string;
  /** Total enrolled */
  total_enrolled: number;
  /** Active count */
  active_count: number;
  /** Withdrawn count */
  withdrawn_count: number;
  /** Completed count */
  completed_count: number;
  /** Enrollment rate per day */
  enrollment_rate_per_day: number;
  /** Breakdown by variant */
  variant_breakdown: Record<string, number>;
  /** Enrollment trend over time */
  enrollment_trend: Array<Record<string, unknown>>;
}

/**
 * Result of interim analysis
 */
export interface InterimAnalysisResult {
  /** Analysis ID */
  analysis_id: string;
  /** Experiment ID */
  experiment_id: string;
  /** Analysis number (1-indexed) */
  analysis_number: number;
  /** When analysis was performed */
  performed_at: string;
  /** Information fraction (0-1) */
  information_fraction: number;
  /** Cumulative alpha spent */
  alpha_spent: number;
  /** Adjusted alpha threshold */
  adjusted_alpha: number;
  /** Test statistic */
  test_statistic: number;
  /** P-value */
  p_value: number;
  /** Conditional power */
  conditional_power: number;
  /** Stopping decision */
  decision: StoppingDecision;
  /** Metrics snapshot */
  metrics_snapshot: Record<string, unknown>;
}

/**
 * List of interim analyses
 */
export interface InterimAnalysesListResponse {
  /** Experiment ID */
  experiment_id: string;
  /** Total analyses */
  total_analyses: number;
  /** Analyses */
  analyses: Array<{
    analysis_id: string;
    analysis_number: number;
    performed_at: string;
    information_fraction: number;
    p_value: number;
    decision: string;
  }>;
}

/**
 * Experiment analysis results
 */
export interface ExperimentResults {
  /** Result ID */
  result_id: string;
  /** Experiment ID */
  experiment_id: string;
  /** Analysis type */
  analysis_type: AnalysisType;
  /** Analysis method */
  analysis_method: AnalysisMethod;
  /** When results were computed */
  computed_at: string;
  /** Primary metric name */
  primary_metric: string;
  /** Control group mean */
  control_mean: number;
  /** Treatment group mean */
  treatment_mean: number;
  /** Effect estimate */
  effect_estimate: number;
  /** Effect CI lower bound */
  effect_ci_lower: number;
  /** Effect CI upper bound */
  effect_ci_upper: number;
  /** P-value */
  p_value: number;
  /** Sample size in control */
  sample_size_control: number;
  /** Sample size in treatment */
  sample_size_treatment: number;
  /** Statistical power */
  statistical_power: number;
  /** Whether statistically significant */
  is_significant: boolean;
  /** Secondary metrics results */
  secondary_metrics?: Record<string, unknown>;
  /** Segment-level results */
  segment_results?: Record<string, unknown>;
}

/**
 * Segment results response
 */
export interface SegmentResultsResponse {
  /** Experiment ID */
  experiment_id: string;
  /** Segments analyzed */
  segments_analyzed: string[];
  /** Results by segment */
  segment_results: Record<string, unknown>;
}

/**
 * SRM check result
 */
export interface SRMCheckResult {
  /** Check ID */
  check_id: string;
  /** Experiment ID */
  experiment_id: string;
  /** Check timestamp */
  checked_at: string;
  /** Expected allocation ratio */
  expected_ratio: Record<string, number>;
  /** Actual counts */
  actual_counts: Record<string, number>;
  /** Chi-squared statistic */
  chi_squared_statistic: number;
  /** P-value */
  p_value: number;
  /** Whether SRM was detected */
  is_srm_detected: boolean;
  /** Investigation notes */
  investigation_notes?: string;
}

/**
 * SRM checks list response
 */
export interface SRMChecksListResponse {
  /** Experiment ID */
  experiment_id: string;
  /** Total checks */
  total_checks: number;
  /** Count where SRM detected */
  srm_detected_count: number;
  /** Check history */
  checks: Array<{
    check_id: string;
    checked_at: string;
    expected_ratio: Record<string, number>;
    actual_counts: Record<string, number>;
    chi_squared: number;
    p_value: number;
    is_srm_detected: boolean;
  }>;
}

/**
 * Digital Twin fidelity comparison
 */
export interface FidelityComparison {
  /** Comparison ID */
  comparison_id: string;
  /** Experiment ID */
  experiment_id: string;
  /** Twin simulation ID */
  twin_simulation_id: string;
  /** Comparison timestamp */
  comparison_timestamp: string;
  /** Predicted effect from Digital Twin */
  predicted_effect: number;
  /** Actual observed effect */
  actual_effect: number;
  /** Prediction error */
  prediction_error: number;
  /** Whether CI covered actual */
  confidence_interval_coverage: boolean;
  /** Overall fidelity score (0-1) */
  fidelity_score: number;
  /** Calibration adjustment */
  calibration_adjustment?: Record<string, unknown>;
}

/**
 * Fidelity comparisons list response
 */
export interface FidelityComparisonsResponse {
  /** Experiment ID */
  experiment_id: string;
  /** Total comparisons */
  total_comparisons: number;
  /** Average fidelity score */
  average_fidelity_score: number;
  /** Comparison history */
  comparisons: Array<{
    comparison_id: string;
    twin_simulation_id: string;
    timestamp: string;
    predicted_effect: number;
    actual_effect: number;
    prediction_error: number;
    fidelity_score: number;
  }>;
}

/**
 * Experiment monitoring alert
 */
export interface MonitorAlert {
  /** Alert ID */
  alert_id: string;
  /** Alert type (srm, enrollment_slow, etc.) */
  alert_type: string;
  /** Severity level */
  severity: AlertSeverity;
  /** Experiment ID */
  experiment_id: string;
  /** Experiment name */
  experiment_name: string;
  /** Alert message */
  message: string;
  /** Additional details */
  details: Record<string, unknown>;
  /** Recommended action */
  recommended_action: string;
  /** Alert timestamp */
  timestamp: string;
}

/**
 * Experiment health summary
 */
export interface ExperimentHealthSummary {
  /** Experiment ID */
  experiment_id: string;
  /** Experiment name */
  experiment_name: string;
  /** Health status */
  health_status: ExperimentHealthStatus;
  /** Total enrolled */
  total_enrolled: number;
  /** Enrollment rate per day */
  enrollment_rate_per_day: number;
  /** Current information fraction (0-1) */
  current_information_fraction: number;
  /** Whether SRM detected */
  has_srm: boolean;
  /** Number of active alerts */
  active_alerts: number;
  /** Last health check timestamp */
  last_checked: string;
}

/**
 * Response from experiment monitoring
 */
export interface MonitorResponse {
  /** Number of experiments checked */
  experiments_checked: number;
  /** Healthy count */
  healthy_count: number;
  /** Warning count */
  warning_count: number;
  /** Critical count */
  critical_count: number;
  /** Experiment summaries */
  experiments: ExperimentHealthSummary[];
  /** Active alerts */
  alerts: MonitorAlert[];
  /** Summary message */
  monitor_summary: string;
  /** Recommended actions */
  recommended_actions: string[];
  /** Check latency in milliseconds */
  check_latency_ms: number;
  /** Check timestamp */
  timestamp: string;
}

/**
 * Experiment alerts list response
 */
export interface ExperimentAlertsResponse {
  /** Experiment ID */
  experiment_id: string;
  /** Total alerts */
  total_alerts: number;
  /** Critical count */
  critical_count: number;
  /** Warning count */
  warning_count: number;
  /** Alerts */
  alerts: Array<{
    alert_id: string;
    alert_type: string;
    severity: string;
    message: string;
    details: Record<string, unknown>;
    timestamp: string;
  }>;
}

/**
 * Withdrawal response
 */
export interface WithdrawResponse {
  /** Status */
  status: string;
  /** Enrollment ID */
  enrollment_id: string;
  /** Experiment ID */
  experiment_id: string;
  /** Withdrawal reason */
  reason: string;
  /** Withdrawal timestamp */
  withdrawn_at: string;
}
