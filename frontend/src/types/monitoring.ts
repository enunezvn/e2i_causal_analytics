/**
 * Model Monitoring & Drift Detection Types
 * =========================================
 *
 * TypeScript interfaces for the E2I Monitoring API.
 * Based on src/api/routes/monitoring.py backend schemas.
 *
 * @module types/monitoring
 */

// =============================================================================
// ENUMS
// =============================================================================

/**
 * Types of drift detection
 */
export enum DriftType {
  DATA = 'data',
  MODEL = 'model',
  CONCEPT = 'concept',
  ALL = 'all',
}

/**
 * Drift severity levels
 */
export enum DriftSeverity {
  NONE = 'none',
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical',
}

/**
 * Alert status values
 */
export enum AlertStatus {
  ACTIVE = 'active',
  ACKNOWLEDGED = 'acknowledged',
  RESOLVED = 'resolved',
  SNOOZED = 'snoozed',
}

/**
 * Actions that can be taken on alerts
 */
export enum AlertAction {
  ACKNOWLEDGE = 'acknowledge',
  RESOLVE = 'resolve',
  SNOOZE = 'snooze',
}

/**
 * Reasons for triggering retraining
 */
export enum TriggerReason {
  DATA_DRIFT = 'data_drift',
  MODEL_DRIFT = 'model_drift',
  CONCEPT_DRIFT = 'concept_drift',
  PERFORMANCE_DEGRADATION = 'performance_degradation',
  SCHEDULED = 'scheduled',
  MANUAL = 'manual',
}

/**
 * Status of retraining job
 */
export enum RetrainingStatus {
  PENDING = 'pending',
  APPROVED = 'approved',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  FAILED = 'failed',
  ROLLED_BACK = 'rolled_back',
  CANCELLED = 'cancelled',
}

// =============================================================================
// DRIFT DETECTION MODELS
// =============================================================================

/**
 * Request to trigger drift detection
 */
export interface TriggerDriftDetectionRequest {
  /** Model version/ID to check for drift */
  model_id: string;
  /** Time window for comparison (e.g., '7d', '14d', '30d') */
  time_window?: string;
  /** Specific features to check (null = all available) */
  features?: string[];
  /** Enable data drift detection */
  check_data_drift?: boolean;
  /** Enable model drift detection */
  check_model_drift?: boolean;
  /** Enable concept drift detection */
  check_concept_drift?: boolean;
  /** Optional brand filter */
  brand?: string;
}

/**
 * Single drift detection result
 */
export interface DriftResult {
  /** Feature or metric name */
  feature: string;
  /** Type of drift detected */
  drift_type: DriftType;
  /** Statistical test value */
  test_statistic: number;
  /** P-value from statistical test */
  p_value: number;
  /** Whether drift was detected */
  drift_detected: boolean;
  /** Severity level */
  severity: DriftSeverity;
  /** Baseline time period label */
  baseline_period: string;
  /** Current time period label */
  current_period: string;
}

/**
 * Response from drift detection
 */
export interface DriftDetectionResponse {
  /** Celery task ID (if async) */
  task_id: string;
  /** Model that was checked */
  model_id: string;
  /** Detection status */
  status: string;
  /** Overall drift severity (0-1) */
  overall_drift_score: number;
  /** Number of features checked */
  features_checked: number;
  /** Features with detected drift */
  features_with_drift: string[];
  /** Detailed drift results */
  results: DriftResult[];
  /** Human-readable summary */
  drift_summary: string;
  /** Recommended actions */
  recommended_actions: string[];
  /** Detection time in ms */
  detection_latency_ms: number;
  /** Detection timestamp (ISO 8601) */
  timestamp: string;
}

/**
 * Historical drift record
 */
export interface DriftHistoryItem {
  id: string;
  model_version: string;
  feature_name: string;
  drift_type: string;
  drift_score: number;
  severity: string;
  detected_at: string;
  baseline_start: string;
  baseline_end: string;
  current_start: string;
  current_end: string;
}

/**
 * Response for drift history query
 */
export interface DriftHistoryResponse {
  model_id: string;
  total_records: number;
  records: DriftHistoryItem[];
}

/**
 * Parameters for drift history query
 */
export interface DriftHistoryParams {
  model_id: string;
  feature_name?: string;
  days?: number;
  limit?: number;
}

// =============================================================================
// ALERT MODELS
// =============================================================================

/**
 * Alert record
 */
export interface AlertItem {
  id: string;
  model_version: string;
  alert_type: string;
  severity: string;
  title: string;
  description: string;
  status: AlertStatus;
  triggered_at: string;
  acknowledged_at?: string;
  acknowledged_by?: string;
  resolved_at?: string;
  resolved_by?: string;
}

/**
 * Response for alert listing
 */
export interface AlertListResponse {
  total_count: number;
  active_count: number;
  alerts: AlertItem[];
}

/**
 * Parameters for alert list query
 */
export interface AlertListParams {
  model_id?: string;
  status?: AlertStatus;
  severity?: DriftSeverity;
  limit?: number;
}

/**
 * Request to update an alert
 */
export interface AlertActionRequest {
  /** Action to take on the alert */
  action: AlertAction;
  /** User performing the action */
  user_id?: string;
  /** Optional notes about the action */
  notes?: string;
  /** Snooze until this time (for snooze action) */
  snooze_until?: string;
}

// =============================================================================
// MONITORING RUN MODELS
// =============================================================================

/**
 * Monitoring run record
 */
export interface MonitoringRunItem {
  id: string;
  model_version: string;
  run_type: string;
  started_at: string;
  completed_at?: string;
  features_checked: number;
  drift_detected_count: number;
  alerts_generated: number;
  duration_ms: number;
  error_message?: string;
}

/**
 * Response for monitoring runs query
 */
export interface MonitoringRunsResponse {
  model_id?: string;
  total_runs: number;
  runs: MonitoringRunItem[];
}

/**
 * Parameters for monitoring runs query
 */
export interface MonitoringRunsParams {
  model_id?: string;
  days?: number;
  limit?: number;
}

// =============================================================================
// MODEL HEALTH MODELS
// =============================================================================

/**
 * Summary of model health status
 */
export interface ModelHealthSummary {
  model_id: string;
  /** Health status: healthy, warning, critical */
  overall_health: 'healthy' | 'warning' | 'critical';
  last_check?: string;
  drift_score: number;
  active_alerts: number;
  last_retrained?: string;
  /** Performance trend: stable, improving, degrading */
  performance_trend: 'stable' | 'improving' | 'degrading';
  recommendations: string[];
}

// =============================================================================
// PERFORMANCE TRACKING MODELS
// =============================================================================

/**
 * Request to record model performance metrics
 */
export interface RecordPerformanceRequest {
  /** Model version/ID */
  model_id: string;
  /** Predicted labels */
  predictions: number[];
  /** Actual labels */
  actuals: number[];
  /** Prediction probability scores (for AUC) */
  prediction_scores?: number[];
}

/**
 * Single performance metric
 */
export interface PerformanceMetricItem {
  metric_name: string;
  metric_value: number;
  recorded_at: string;
}

/**
 * Response for performance trend query
 */
export interface PerformanceTrendResponse {
  model_id: string;
  metric_name: string;
  current_value: number;
  baseline_value: number;
  change_percent: number;
  /** Trend direction: improving, stable, degrading */
  trend: 'improving' | 'stable' | 'degrading';
  is_significant: boolean;
  alert_threshold_breached: boolean;
  history: PerformanceMetricItem[];
}

/**
 * Response from recording performance
 */
export interface PerformanceRecordResponse {
  model_id: string;
  recorded_at: string;
  sample_size: number;
  metrics: Record<string, number>;
  alerts_generated: number;
}

/**
 * Performance alert
 */
export interface PerformanceAlertItem {
  metric_name: string;
  current_value: number;
  baseline_value: number;
  change_percent: number;
  trend: string;
  severity: string;
  message: string;
}

/**
 * Response for performance alerts query
 */
export interface PerformanceAlertsResponse {
  model_id: string;
  alert_count: number;
  alerts: PerformanceAlertItem[];
}

/**
 * Parameters for performance trend query
 */
export interface PerformanceTrendParams {
  model_id: string;
  metric_name?: string;
  days?: number;
}

/**
 * Response for model comparison
 */
export interface ModelComparisonResponse {
  model_id: string;
  other_model_id: string;
  metric_name: string;
  model_value: number;
  other_model_value: number;
  difference: number;
  difference_percent: number;
  better_model: string;
}

// =============================================================================
// RETRAINING MODELS
// =============================================================================

/**
 * Request to trigger model retraining
 */
export interface TriggerRetrainingRequest {
  /** Reason for retraining */
  reason: TriggerReason;
  /** Additional notes */
  notes?: string;
  /** Auto-approve retraining */
  auto_approve?: boolean;
}

/**
 * Request to mark retraining as complete
 */
export interface CompleteRetrainingRequest {
  /** Performance metric after retraining */
  performance_after: number;
  /** Whether retraining was successful */
  success?: boolean;
  /** Additional notes */
  notes?: string;
}

/**
 * Request to rollback a retraining
 */
export interface RollbackRetrainingRequest {
  /** Reason for rollback */
  reason: string;
}

/**
 * Response for retraining evaluation
 */
export interface RetrainingDecisionResponse {
  model_id: string;
  should_retrain: boolean;
  confidence: number;
  reasons: string[];
  trigger_factors: Record<string, unknown>;
  cooldown_active: boolean;
  cooldown_ends_at?: string;
  recommended_action: string;
}

/**
 * Response for retraining job
 */
export interface RetrainingJobResponse {
  job_id: string;
  model_version: string;
  status: RetrainingStatus;
  trigger_reason: string;
  triggered_at: string;
  triggered_by: string;
  approved_at?: string;
  started_at?: string;
  completed_at?: string;
  performance_before?: number;
  performance_after?: number;
  notes?: string;
}

// =============================================================================
// ASYNC TASK STATUS
// =============================================================================

/**
 * Status of an async task (Celery)
 */
export interface TaskStatusResponse {
  task_id: string;
  status: string;
  ready: boolean;
  result?: DriftDetectionResponse;
  error?: string;
}

/**
 * Response for production sweep
 */
export interface ProductionSweepResponse {
  task_id: string;
  status: string;
  message: string;
  time_window: string;
}
