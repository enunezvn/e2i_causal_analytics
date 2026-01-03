/**
 * Monitoring API Client
 * =====================
 *
 * TypeScript API client functions for the E2I Model Monitoring endpoints.
 * Uses the shared apiClient for consistent error handling and interceptors.
 *
 * Endpoints:
 * - Drift detection and history
 * - Alert management
 * - Monitoring runs
 * - Model health
 * - Performance tracking
 * - Retraining triggers
 *
 * @module api/monitoring
 */

import { get, post } from '@/lib/api-client';
import type {
  AlertActionRequest,
  AlertItem,
  AlertListParams,
  AlertListResponse,
  CompleteRetrainingRequest,
  DriftDetectionResponse,
  DriftHistoryParams,
  DriftHistoryResponse,
  ModelComparisonResponse,
  ModelHealthSummary,
  MonitoringRunsParams,
  MonitoringRunsResponse,
  PerformanceAlertsResponse,
  PerformanceRecordResponse,
  PerformanceTrendParams,
  PerformanceTrendResponse,
  ProductionSweepResponse,
  RecordPerformanceRequest,
  RetrainingDecisionResponse,
  RetrainingJobResponse,
  RollbackRetrainingRequest,
  TaskStatusResponse,
  TriggerDriftDetectionRequest,
  TriggerRetrainingRequest,
} from '@/types/monitoring';

// =============================================================================
// MONITORING API ENDPOINTS
// =============================================================================

const MONITORING_BASE = '/monitoring';

// =============================================================================
// DRIFT DETECTION ENDPOINTS
// =============================================================================

/**
 * Trigger drift detection for a model.
 *
 * Runs data, model, and concept drift detection based on request parameters.
 * By default runs asynchronously via Celery.
 *
 * @param request - Drift detection parameters
 * @param asyncMode - If true, returns immediately with task ID (default: true)
 * @returns Detection results or task ID for async polling
 *
 * @example
 * ```typescript
 * const result = await triggerDriftDetection({
 *   model_id: 'propensity_v2.1.0',
 *   time_window: '7d',
 *   check_data_drift: true
 * });
 * ```
 */
export async function triggerDriftDetection(
  request: TriggerDriftDetectionRequest,
  asyncMode: boolean = true
): Promise<DriftDetectionResponse> {
  return post<DriftDetectionResponse, TriggerDriftDetectionRequest>(
    `${MONITORING_BASE}/drift/detect`,
    request,
    { params: { async_mode: asyncMode } }
  );
}

/**
 * Get status of an async drift detection task.
 *
 * @param taskId - Celery task ID
 * @returns Task status and results if complete
 *
 * @example
 * ```typescript
 * const status = await getDriftDetectionStatus('task_abc123');
 * if (status.ready && status.result) {
 *   console.log(status.result.drift_summary);
 * }
 * ```
 */
export async function getDriftDetectionStatus(
  taskId: string
): Promise<TaskStatusResponse> {
  return get<TaskStatusResponse>(
    `${MONITORING_BASE}/drift/status/${encodeURIComponent(taskId)}`
  );
}

/**
 * Get the latest drift status for a model.
 *
 * @param modelId - Model version/ID
 * @param limit - Maximum drift results to return (default: 10)
 * @returns Latest drift detection results
 *
 * @example
 * ```typescript
 * const drift = await getLatestDriftStatus('propensity_v2.1.0');
 * console.log(`Drift score: ${drift.overall_drift_score}`);
 * ```
 */
export async function getLatestDriftStatus(
  modelId: string,
  limit: number = 10
): Promise<DriftDetectionResponse> {
  return get<DriftDetectionResponse>(
    `${MONITORING_BASE}/drift/latest/${encodeURIComponent(modelId)}`,
    { limit }
  );
}

/**
 * Get drift detection history for a model.
 *
 * @param params - Query parameters with model ID and filters
 * @returns Historical drift records
 *
 * @example
 * ```typescript
 * const history = await getDriftHistory({
 *   model_id: 'propensity_v2.1.0',
 *   feature_name: 'days_since_last_visit',
 *   days: 30
 * });
 * ```
 */
export async function getDriftHistory(
  params: DriftHistoryParams
): Promise<DriftHistoryResponse> {
  return get<DriftHistoryResponse>(
    `${MONITORING_BASE}/drift/history/${encodeURIComponent(params.model_id)}`,
    {
      feature_name: params.feature_name,
      days: params.days,
      limit: params.limit,
    }
  );
}

// =============================================================================
// ALERT ENDPOINTS
// =============================================================================

/**
 * List drift alerts.
 *
 * @param params - Optional filters for model, status, severity
 * @returns List of alerts matching criteria
 *
 * @example
 * ```typescript
 * const alerts = await listAlerts({
 *   status: AlertStatus.ACTIVE,
 *   severity: DriftSeverity.HIGH
 * });
 * console.log(`Active alerts: ${alerts.active_count}`);
 * ```
 */
export async function listAlerts(
  params?: AlertListParams
): Promise<AlertListResponse> {
  return get<AlertListResponse>(`${MONITORING_BASE}/alerts`, {
    model_id: params?.model_id,
    status: params?.status,
    severity: params?.severity,
    limit: params?.limit,
  });
}

/**
 * Get a specific alert by ID.
 *
 * @param alertId - Alert UUID
 * @returns Alert details
 *
 * @example
 * ```typescript
 * const alert = await getAlert('alert-uuid-123');
 * console.log(`Alert: ${alert.title} - ${alert.status}`);
 * ```
 */
export async function getAlert(alertId: string): Promise<AlertItem> {
  return get<AlertItem>(
    `${MONITORING_BASE}/alerts/${encodeURIComponent(alertId)}`
  );
}

/**
 * Perform an action on an alert (acknowledge, resolve, snooze).
 *
 * @param alertId - Alert UUID
 * @param request - Action to perform
 * @returns Updated alert
 *
 * @example
 * ```typescript
 * const updated = await updateAlert('alert-uuid-123', {
 *   action: AlertAction.ACKNOWLEDGE,
 *   user_id: 'user_123',
 *   notes: 'Investigating the issue'
 * });
 * ```
 */
export async function updateAlert(
  alertId: string,
  request: AlertActionRequest
): Promise<AlertItem> {
  return post<AlertItem, AlertActionRequest>(
    `${MONITORING_BASE}/alerts/${encodeURIComponent(alertId)}/action`,
    request
  );
}

// =============================================================================
// MONITORING RUNS ENDPOINTS
// =============================================================================

/**
 * List monitoring runs.
 *
 * @param params - Optional filters for model and time range
 * @returns List of monitoring runs
 *
 * @example
 * ```typescript
 * const runs = await listMonitoringRuns({
 *   model_id: 'propensity_v2.1.0',
 *   days: 7
 * });
 * ```
 */
export async function listMonitoringRuns(
  params?: MonitoringRunsParams
): Promise<MonitoringRunsResponse> {
  return get<MonitoringRunsResponse>(`${MONITORING_BASE}/runs`, {
    model_id: params?.model_id,
    days: params?.days,
    limit: params?.limit,
  });
}

// =============================================================================
// MODEL HEALTH ENDPOINTS
// =============================================================================

/**
 * Get overall health summary for a model.
 *
 * Aggregates drift status, alerts, and performance metrics.
 *
 * @param modelId - Model version/ID
 * @returns Health summary with recommendations
 *
 * @example
 * ```typescript
 * const health = await getModelHealth('propensity_v2.1.0');
 * if (health.overall_health === 'critical') {
 *   console.log('Model needs attention:', health.recommendations);
 * }
 * ```
 */
export async function getModelHealth(
  modelId: string
): Promise<ModelHealthSummary> {
  return get<ModelHealthSummary>(
    `${MONITORING_BASE}/health/${encodeURIComponent(modelId)}`
  );
}

// =============================================================================
// PERFORMANCE TRACKING ENDPOINTS
// =============================================================================

/**
 * Record model performance metrics.
 *
 * Calculates and persists standard ML metrics (accuracy, precision, recall, F1, AUC).
 *
 * @param request - Performance data including predictions and actuals
 * @param asyncMode - If true, processes asynchronously via Celery (default: true)
 * @returns Recorded performance metrics
 *
 * @example
 * ```typescript
 * const result = await recordPerformance({
 *   model_id: 'propensity_v2.1.0',
 *   predictions: [1, 0, 1, 1, 0],
 *   actuals: [1, 0, 1, 0, 0],
 *   prediction_scores: [0.85, 0.23, 0.91, 0.67, 0.12]
 * });
 * ```
 */
export async function recordPerformance(
  request: RecordPerformanceRequest,
  asyncMode: boolean = true
): Promise<PerformanceRecordResponse> {
  return post<PerformanceRecordResponse, RecordPerformanceRequest>(
    `${MONITORING_BASE}/performance/record`,
    request,
    { params: { async_mode: asyncMode } }
  );
}

/**
 * Get performance trend for a model.
 *
 * Analyzes metric trends over time and detects degradation.
 *
 * @param params - Query parameters with model ID and metric
 * @returns Performance trend analysis
 *
 * @example
 * ```typescript
 * const trend = await getPerformanceTrend({
 *   model_id: 'propensity_v2.1.0',
 *   metric_name: 'accuracy',
 *   days: 30
 * });
 * console.log(`Trend: ${trend.trend}, Change: ${trend.change_percent}%`);
 * ```
 */
export async function getPerformanceTrend(
  params: PerformanceTrendParams
): Promise<PerformanceTrendResponse> {
  return get<PerformanceTrendResponse>(
    `${MONITORING_BASE}/performance/${encodeURIComponent(params.model_id)}/trend`,
    {
      metric_name: params.metric_name,
      days: params.days,
    }
  );
}

/**
 * Check for performance-related alerts.
 *
 * Analyzes all tracked metrics for degradation.
 *
 * @param modelId - Model version/ID
 * @returns List of performance alerts
 *
 * @example
 * ```typescript
 * const alerts = await getPerformanceAlerts('propensity_v2.1.0');
 * if (alerts.alert_count > 0) {
 *   console.log('Performance issues detected:', alerts.alerts);
 * }
 * ```
 */
export async function getPerformanceAlerts(
  modelId: string
): Promise<PerformanceAlertsResponse> {
  return get<PerformanceAlertsResponse>(
    `${MONITORING_BASE}/performance/${encodeURIComponent(modelId)}/alerts`
  );
}

/**
 * Compare performance between two model versions.
 *
 * @param modelId - First model version
 * @param otherModelId - Second model version
 * @param metricName - Metric to compare (default: 'accuracy')
 * @returns Comparison results
 *
 * @example
 * ```typescript
 * const comparison = await compareModelPerformance(
 *   'propensity_v2.1.0',
 *   'propensity_v2.0.0',
 *   'accuracy'
 * );
 * console.log(`Better model: ${comparison.better_model}`);
 * ```
 */
export async function compareModelPerformance(
  modelId: string,
  otherModelId: string,
  metricName: string = 'accuracy'
): Promise<ModelComparisonResponse> {
  return get<ModelComparisonResponse>(
    `${MONITORING_BASE}/performance/${encodeURIComponent(modelId)}/compare/${encodeURIComponent(otherModelId)}`,
    { metric_name: metricName }
  );
}

// =============================================================================
// PRODUCTION SWEEP ENDPOINTS
// =============================================================================

/**
 * Trigger drift detection sweep for all production models.
 *
 * Queues Celery tasks for each production model.
 *
 * @param timeWindow - Time window for drift comparison (default: '7d')
 * @returns Summary of queued tasks
 *
 * @example
 * ```typescript
 * const sweep = await triggerProductionSweep('7d');
 * console.log(`Sweep queued: ${sweep.task_id}`);
 * ```
 */
export async function triggerProductionSweep(
  timeWindow: string = '7d'
): Promise<ProductionSweepResponse> {
  return post<ProductionSweepResponse, undefined>(
    `${MONITORING_BASE}/sweep/production`,
    undefined,
    { params: { time_window: timeWindow } }
  );
}

// =============================================================================
// RETRAINING ENDPOINTS
// =============================================================================

/**
 * Evaluate whether a model needs retraining.
 *
 * Analyzes drift scores, performance trends, and other factors.
 *
 * @param modelId - Model version/ID
 * @returns Retraining decision with reasoning
 *
 * @example
 * ```typescript
 * const decision = await evaluateRetrainingNeed('propensity_v2.1.0');
 * if (decision.should_retrain) {
 *   console.log('Reasons:', decision.reasons);
 * }
 * ```
 */
export async function evaluateRetrainingNeed(
  modelId: string
): Promise<RetrainingDecisionResponse> {
  return post<RetrainingDecisionResponse, undefined>(
    `${MONITORING_BASE}/retraining/evaluate/${encodeURIComponent(modelId)}`,
    undefined
  );
}

/**
 * Trigger model retraining.
 *
 * Creates a retraining job and optionally auto-approves it.
 *
 * @param modelId - Model version/ID
 * @param request - Retraining parameters
 * @param triggeredBy - User or system triggering retraining (default: 'ui_user')
 * @returns Created retraining job
 *
 * @example
 * ```typescript
 * const job = await triggerRetraining('propensity_v2.1.0', {
 *   reason: TriggerReason.DATA_DRIFT,
 *   notes: 'Significant drift detected',
 *   auto_approve: false
 * });
 * ```
 */
export async function triggerRetraining(
  modelId: string,
  request: TriggerRetrainingRequest,
  triggeredBy: string = 'ui_user'
): Promise<RetrainingJobResponse> {
  return post<RetrainingJobResponse, TriggerRetrainingRequest>(
    `${MONITORING_BASE}/retraining/trigger/${encodeURIComponent(modelId)}`,
    request,
    { params: { triggered_by: triggeredBy } }
  );
}

/**
 * Get status of a retraining job.
 *
 * @param jobId - Retraining job ID
 * @returns Retraining job details
 *
 * @example
 * ```typescript
 * const job = await getRetrainingStatus('job-uuid-123');
 * console.log(`Status: ${job.status}`);
 * ```
 */
export async function getRetrainingStatus(
  jobId: string
): Promise<RetrainingJobResponse> {
  return get<RetrainingJobResponse>(
    `${MONITORING_BASE}/retraining/status/${encodeURIComponent(jobId)}`
  );
}

/**
 * Mark a retraining job as complete.
 *
 * @param jobId - Retraining job ID
 * @param request - Completion details
 * @returns Updated retraining job
 *
 * @example
 * ```typescript
 * const job = await completeRetraining('job-uuid-123', {
 *   performance_after: 0.92,
 *   success: true,
 *   notes: 'Model retrained successfully'
 * });
 * ```
 */
export async function completeRetraining(
  jobId: string,
  request: CompleteRetrainingRequest
): Promise<RetrainingJobResponse> {
  return post<RetrainingJobResponse, CompleteRetrainingRequest>(
    `${MONITORING_BASE}/retraining/${encodeURIComponent(jobId)}/complete`,
    request
  );
}

/**
 * Rollback a completed retraining.
 *
 * Reverts to previous model version.
 *
 * @param jobId - Retraining job ID
 * @param request - Rollback reason
 * @returns Updated retraining job
 *
 * @example
 * ```typescript
 * const job = await rollbackRetraining('job-uuid-123', {
 *   reason: 'Performance degradation on validation set'
 * });
 * ```
 */
export async function rollbackRetraining(
  jobId: string,
  request: RollbackRetrainingRequest
): Promise<RetrainingJobResponse> {
  return post<RetrainingJobResponse, RollbackRetrainingRequest>(
    `${MONITORING_BASE}/retraining/${encodeURIComponent(jobId)}/rollback`,
    request
  );
}

/**
 * Trigger retraining evaluation sweep for all production models.
 *
 * @returns Summary of queued tasks
 *
 * @example
 * ```typescript
 * const sweep = await triggerRetrainingSweep();
 * console.log(`Sweep queued: ${sweep.task_id}`);
 * ```
 */
export async function triggerRetrainingSweep(): Promise<ProductionSweepResponse> {
  return post<ProductionSweepResponse, undefined>(
    `${MONITORING_BASE}/retraining/sweep`,
    undefined
  );
}
