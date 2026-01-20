/**
 * A/B Testing & Experiments API Client
 * =====================================
 *
 * TypeScript API client functions for the E2I A/B Testing endpoints.
 * Uses the shared apiClient for consistent error handling and interceptors.
 *
 * Endpoints:
 * - Randomization
 * - Enrollment management
 * - Interim analysis
 * - Results and segment analysis
 * - SRM detection
 * - Digital Twin fidelity
 * - Experiment monitoring
 *
 * @module api/experiments
 */

import { get, post, del } from '@/lib/api-client';
import type {
  AlertSeverity,
  AssignmentsListResponse,
  EnrollmentResult,
  EnrollmentStatsResponse,
  EnrollUnitRequest,
  ExperimentAlertsResponse,
  ExperimentHealthSummary,
  ExperimentResults,
  FidelityComparison,
  FidelityComparisonsResponse,
  GetAssignmentsParams,
  GetResultsParams,
  InterimAnalysesListResponse,
  InterimAnalysisResult,
  MonitorResponse,
  RandomizeRequest,
  RandomizeResponse,
  SegmentResultsResponse,
  SRMCheckResult,
  SRMChecksListResponse,
  TriggerInterimAnalysisRequest,
  TriggerMonitorRequest,
  WithdrawRequest,
  WithdrawResponse,
} from '@/types/experiments';

// =============================================================================
// EXPERIMENTS API ENDPOINTS
// =============================================================================

const EXPERIMENTS_BASE = '/experiments';

// =============================================================================
// RANDOMIZATION ENDPOINTS
// =============================================================================

/**
 * Randomize units to experiment variants.
 *
 * Supports simple, stratified, and block randomization methods.
 *
 * @param experimentId - Experiment ID
 * @param request - Randomization parameters and units
 * @returns Assignment results for all units
 *
 * @example
 * ```typescript
 * const result = await randomizeUnits('exp_abc123', {
 *   units: [
 *     { unit_id: 'hcp_001', unit_type: 'hcp', region: 'northeast' },
 *     { unit_id: 'hcp_002', unit_type: 'hcp', region: 'southwest' },
 *   ],
 *   method: RandomizationMethod.STRATIFIED,
 *   strata_columns: ['region'],
 *   allocation_ratio: { control: 0.5, treatment: 0.5 },
 * });
 *
 * console.log(`Randomized ${result.total_units} units`);
 * console.log(`Control: ${result.variant_counts.control}`);
 * console.log(`Treatment: ${result.variant_counts.treatment}`);
 * ```
 */
export async function randomizeUnits(
  experimentId: string,
  request: RandomizeRequest
): Promise<RandomizeResponse> {
  return post<RandomizeResponse, RandomizeRequest>(
    `${EXPERIMENTS_BASE}/${encodeURIComponent(experimentId)}/randomize`,
    request
  );
}

/**
 * Get experiment assignments.
 *
 * @param experimentId - Experiment ID
 * @param params - Filter and pagination parameters
 * @returns List of assignments
 *
 * @example
 * ```typescript
 * const assignments = await getAssignments('exp_abc123', {
 *   variant: 'treatment',
 *   limit: 50,
 * });
 *
 * assignments.assignments.forEach(a => {
 *   console.log(`${a.unit_id}: ${a.variant}`);
 * });
 * ```
 */
export async function getAssignments(
  experimentId: string,
  params?: GetAssignmentsParams
): Promise<AssignmentsListResponse> {
  return get<AssignmentsListResponse>(
    `${EXPERIMENTS_BASE}/${encodeURIComponent(experimentId)}/assignments`,
    params as Record<string, unknown> | undefined
  );
}

// =============================================================================
// ENROLLMENT ENDPOINTS
// =============================================================================

/**
 * Enroll a unit in an experiment.
 *
 * The unit must already be assigned to a variant via randomization.
 *
 * @param experimentId - Experiment ID
 * @param request - Enrollment details
 * @returns Enrollment result
 *
 * @example
 * ```typescript
 * const enrollment = await enrollUnit('exp_abc123', {
 *   unit_id: 'hcp_001',
 *   unit_type: 'hcp',
 *   consent_timestamp: new Date().toISOString(),
 *   eligibility_criteria_met: { specialty: true, experience: true },
 * });
 *
 * console.log(`Enrolled in ${enrollment.variant} arm`);
 * ```
 */
export async function enrollUnit(
  experimentId: string,
  request: EnrollUnitRequest
): Promise<EnrollmentResult> {
  return post<EnrollmentResult, EnrollUnitRequest>(
    `${EXPERIMENTS_BASE}/${encodeURIComponent(experimentId)}/enroll`,
    request
  );
}

/**
 * Withdraw a unit from an experiment.
 *
 * @param experimentId - Experiment ID
 * @param enrollmentId - Enrollment ID
 * @param request - Withdrawal details
 * @returns Confirmation of withdrawal
 *
 * @example
 * ```typescript
 * const result = await withdrawUnit('exp_abc123', 'enr_xyz789', {
 *   reason: 'Subject requested withdrawal',
 * });
 *
 * console.log(`Withdrawn at ${result.withdrawn_at}`);
 * ```
 */
export async function withdrawUnit(
  experimentId: string,
  enrollmentId: string,
  request: WithdrawRequest
): Promise<WithdrawResponse> {
  return del<WithdrawResponse>(
    `${EXPERIMENTS_BASE}/${encodeURIComponent(experimentId)}/enrollments/${encodeURIComponent(enrollmentId)}`,
    { data: request }
  );
}

/**
 * Get enrollment statistics for an experiment.
 *
 * @param experimentId - Experiment ID
 * @returns Enrollment statistics
 *
 * @example
 * ```typescript
 * const stats = await getEnrollmentStats('exp_abc123');
 * console.log(`Total: ${stats.total_enrolled}`);
 * console.log(`Active: ${stats.active_count}`);
 * console.log(`Rate: ${stats.enrollment_rate_per_day}/day`);
 * ```
 */
export async function getEnrollmentStats(
  experimentId: string
): Promise<EnrollmentStatsResponse> {
  return get<EnrollmentStatsResponse>(
    `${EXPERIMENTS_BASE}/${encodeURIComponent(experimentId)}/enrollments`
  );
}

// =============================================================================
// INTERIM ANALYSIS ENDPOINTS
// =============================================================================

/**
 * Trigger an interim analysis for an experiment.
 *
 * Uses O'Brien-Fleming alpha spending for multiple comparisons.
 *
 * @param experimentId - Experiment ID
 * @param request - Analysis parameters
 * @param asyncMode - If true, runs asynchronously (default: false)
 * @returns Interim analysis results
 *
 * @example
 * ```typescript
 * const result = await triggerInterimAnalysis('exp_abc123', {
 *   analysis_number: 2,
 *   force: false,
 * });
 *
 * if (result.decision === StoppingDecision.STOP_EFFICACY) {
 *   console.log('Early stopping for efficacy!');
 * }
 * console.log(`P-value: ${result.p_value}, Adjusted alpha: ${result.adjusted_alpha}`);
 * ```
 */
export async function triggerInterimAnalysis(
  experimentId: string,
  request: TriggerInterimAnalysisRequest,
  asyncMode: boolean = false
): Promise<InterimAnalysisResult> {
  return post<InterimAnalysisResult, TriggerInterimAnalysisRequest>(
    `${EXPERIMENTS_BASE}/${encodeURIComponent(experimentId)}/interim-analysis`,
    request,
    { params: { async_mode: asyncMode } }
  );
}

/**
 * List all interim analyses for an experiment.
 *
 * @param experimentId - Experiment ID
 * @returns List of interim analyses
 *
 * @example
 * ```typescript
 * const analyses = await listInterimAnalyses('exp_abc123');
 * analyses.analyses.forEach(a => {
 *   console.log(`Analysis ${a.analysis_number}: ${a.decision}`);
 * });
 * ```
 */
export async function listInterimAnalyses(
  experimentId: string
): Promise<InterimAnalysesListResponse> {
  return get<InterimAnalysesListResponse>(
    `${EXPERIMENTS_BASE}/${encodeURIComponent(experimentId)}/interim-analyses`
  );
}

// =============================================================================
// RESULTS ENDPOINTS
// =============================================================================

/**
 * Get experiment results.
 *
 * @param experimentId - Experiment ID
 * @param params - Analysis parameters
 * @returns Experiment analysis results
 *
 * @example
 * ```typescript
 * const results = await getExperimentResults('exp_abc123', {
 *   analysis_type: AnalysisType.FINAL,
 *   analysis_method: AnalysisMethod.ITT,
 * });
 *
 * console.log(`Effect: ${results.effect_estimate} [${results.effect_ci_lower}, ${results.effect_ci_upper}]`);
 * console.log(`Significant: ${results.is_significant}`);
 * ```
 */
export async function getExperimentResults(
  experimentId: string,
  params?: GetResultsParams
): Promise<ExperimentResults> {
  return get<ExperimentResults>(
    `${EXPERIMENTS_BASE}/${encodeURIComponent(experimentId)}/results`,
    params as Record<string, unknown> | undefined
  );
}

/**
 * Get heterogeneous treatment effects by segment.
 *
 * @param experimentId - Experiment ID
 * @param segments - Segment dimensions to analyze
 * @returns Treatment effects by segment
 *
 * @example
 * ```typescript
 * const hte = await getSegmentResults('exp_abc123', ['region', 'specialty']);
 * console.log(`Analyzed segments: ${hte.segments_analyzed.join(', ')}`);
 * ```
 */
export async function getSegmentResults(
  experimentId: string,
  segments?: string[]
): Promise<SegmentResultsResponse> {
  return get<SegmentResultsResponse>(
    `${EXPERIMENTS_BASE}/${encodeURIComponent(experimentId)}/results/segments`,
    { segments }
  );
}

// =============================================================================
// SRM ENDPOINTS
// =============================================================================

/**
 * Get SRM check history for an experiment.
 *
 * @param experimentId - Experiment ID
 * @param limit - Maximum checks to return (default: 10)
 * @returns SRM check history
 *
 * @example
 * ```typescript
 * const history = await getSRMChecks('exp_abc123');
 * if (history.srm_detected_count > 0) {
 *   console.warn(`SRM detected ${history.srm_detected_count} times!`);
 * }
 * ```
 */
export async function getSRMChecks(
  experimentId: string,
  limit: number = 10
): Promise<SRMChecksListResponse> {
  return get<SRMChecksListResponse>(
    `${EXPERIMENTS_BASE}/${encodeURIComponent(experimentId)}/srm-checks`,
    { limit }
  );
}

/**
 * Run an SRM check for an experiment.
 *
 * @param experimentId - Experiment ID
 * @returns SRM check result
 *
 * @example
 * ```typescript
 * const check = await runSRMCheck('exp_abc123');
 * if (check.is_srm_detected) {
 *   console.error(`SRM detected! Chi²=${check.chi_squared_statistic}, p=${check.p_value}`);
 * }
 * ```
 */
export async function runSRMCheck(
  experimentId: string
): Promise<SRMCheckResult> {
  return post<SRMCheckResult, Record<string, never>>(
    `${EXPERIMENTS_BASE}/${encodeURIComponent(experimentId)}/srm-check`,
    {}
  );
}

// =============================================================================
// FIDELITY ENDPOINTS
// =============================================================================

/**
 * Get Digital Twin fidelity comparisons for an experiment.
 *
 * @param experimentId - Experiment ID
 * @param limit - Maximum comparisons to return (default: 10)
 * @returns Fidelity comparison history
 *
 * @example
 * ```typescript
 * const fidelity = await getFidelityComparisons('exp_abc123');
 * console.log(`Average fidelity: ${fidelity.average_fidelity_score}`);
 * ```
 */
export async function getFidelityComparisons(
  experimentId: string,
  limit: number = 10
): Promise<FidelityComparisonsResponse> {
  return get<FidelityComparisonsResponse>(
    `${EXPERIMENTS_BASE}/${encodeURIComponent(experimentId)}/fidelity`,
    { limit }
  );
}

/**
 * Update fidelity comparison with latest experiment results.
 *
 * @param experimentId - Experiment ID
 * @param twinSimulationId - Digital Twin simulation ID
 * @returns Updated fidelity comparison
 *
 * @example
 * ```typescript
 * const comparison = await updateFidelityComparison('exp_abc123', 'sim_xyz789');
 * console.log(`Predicted: ${comparison.predicted_effect}, Actual: ${comparison.actual_effect}`);
 * console.log(`Fidelity score: ${comparison.fidelity_score}`);
 * ```
 */
export async function updateFidelityComparison(
  experimentId: string,
  twinSimulationId: string
): Promise<FidelityComparison> {
  return post<FidelityComparison, Record<string, never>>(
    `${EXPERIMENTS_BASE}/${encodeURIComponent(experimentId)}/fidelity/${encodeURIComponent(twinSimulationId)}`,
    {}
  );
}

// =============================================================================
// MONITORING ENDPOINTS
// =============================================================================

/**
 * Trigger experiment monitoring sweep.
 *
 * Checks all active experiments (or specified ones) for health issues.
 *
 * @param request - Monitoring parameters
 * @param asyncMode - If true, runs asynchronously (default: false)
 * @returns Monitoring results with alerts
 *
 * @example
 * ```typescript
 * const monitor = await triggerMonitoring({
 *   check_srm: true,
 *   check_enrollment: true,
 *   srm_threshold: 0.001,
 * });
 *
 * console.log(`${monitor.healthy_count} healthy, ${monitor.critical_count} critical`);
 * monitor.alerts.forEach(alert => {
 *   console.log(`[${alert.severity}] ${alert.experiment_name}: ${alert.message}`);
 * });
 * ```
 */
export async function triggerMonitoring(
  request: TriggerMonitorRequest,
  asyncMode: boolean = false
): Promise<MonitorResponse> {
  return post<MonitorResponse, TriggerMonitorRequest>(
    `${EXPERIMENTS_BASE}/monitor`,
    request,
    { params: { async_mode: asyncMode } }
  );
}

/**
 * Get health status for a single experiment.
 *
 * @param experimentId - Experiment ID
 * @returns Experiment health summary
 *
 * @example
 * ```typescript
 * const health = await getExperimentHealth('exp_abc123');
 * if (health.health_status === ExperimentHealthStatus.CRITICAL) {
 *   console.error(`Experiment ${health.experiment_name} is critical!`);
 * }
 * ```
 */
export async function getExperimentHealth(
  experimentId: string
): Promise<ExperimentHealthSummary> {
  return get<ExperimentHealthSummary>(
    `${EXPERIMENTS_BASE}/${encodeURIComponent(experimentId)}/health`
  );
}

/**
 * Get alerts for an experiment.
 *
 * @param experimentId - Experiment ID
 * @param severity - Optional severity filter
 * @param limit - Maximum alerts to return (default: 50)
 * @returns List of alerts
 *
 * @example
 * ```typescript
 * const alerts = await getExperimentAlerts('exp_abc123', AlertSeverity.CRITICAL);
 * if (alerts.critical_count > 0) {
 *   alerts.alerts.forEach(a => console.error(a.message));
 * }
 * ```
 */
export async function getExperimentAlerts(
  experimentId: string,
  severity?: AlertSeverity,
  limit: number = 50
): Promise<ExperimentAlertsResponse> {
  return get<ExperimentAlertsResponse>(
    `${EXPERIMENTS_BASE}/${encodeURIComponent(experimentId)}/alerts`,
    { severity, limit }
  );
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * Randomize and enroll units in a single operation.
 *
 * Convenience function that randomizes units and then enrolls them.
 *
 * @param experimentId - Experiment ID
 * @param request - Randomization parameters
 * @returns Array of enrollment results
 *
 * @example
 * ```typescript
 * const enrollments = await randomizeAndEnroll('exp_abc123', {
 *   units: [
 *     { unit_id: 'hcp_001', unit_type: 'hcp' },
 *     { unit_id: 'hcp_002', unit_type: 'hcp' },
 *   ],
 * });
 *
 * console.log(`Enrolled ${enrollments.length} units`);
 * ```
 */
export async function randomizeAndEnroll(
  experimentId: string,
  request: RandomizeRequest
): Promise<EnrollmentResult[]> {
  // First randomize
  const randomizeResult = await randomizeUnits(experimentId, request);

  // Then enroll each unit
  const enrollments: EnrollmentResult[] = [];
  for (const assignment of randomizeResult.assignments) {
    const enrollment = await enrollUnit(experimentId, {
      unit_id: assignment.unit_id,
      unit_type: assignment.unit_type,
    });
    enrollments.push(enrollment);
  }

  return enrollments;
}

/**
 * Get comprehensive experiment status.
 *
 * Fetches health, enrollment stats, and recent alerts in parallel.
 *
 * @param experimentId - Experiment ID
 * @returns Combined status object
 *
 * @example
 * ```typescript
 * const status = await getComprehensiveStatus('exp_abc123');
 * console.log(`Health: ${status.health.health_status}`);
 * console.log(`Enrolled: ${status.enrollment.total_enrolled}`);
 * console.log(`Critical alerts: ${status.alerts.critical_count}`);
 * ```
 */
export async function getComprehensiveStatus(experimentId: string): Promise<{
  health: ExperimentHealthSummary;
  enrollment: EnrollmentStatsResponse;
  alerts: ExperimentAlertsResponse;
  srm: SRMChecksListResponse;
}> {
  const [health, enrollment, alerts, srm] = await Promise.all([
    getExperimentHealth(experimentId),
    getEnrollmentStats(experimentId),
    getExperimentAlerts(experimentId),
    getSRMChecks(experimentId, 5),
  ]);

  return { health, enrollment, alerts, srm };
}

/**
 * Check if experiment should stop early.
 *
 * Runs interim analysis and returns stopping recommendation.
 *
 * @param experimentId - Experiment ID
 * @returns Stopping decision and analysis result
 *
 * @example
 * ```typescript
 * const check = await checkEarlyStopping('exp_abc123');
 * if (check.shouldStop) {
 *   console.log(`Recommend stopping: ${check.decision} (p=${check.pValue})`);
 * }
 * ```
 */
export async function checkEarlyStopping(experimentId: string): Promise<{
  shouldStop: boolean;
  decision: string;
  pValue: number;
  adjustedAlpha: number;
  conditionalPower: number;
}> {
  const result = await triggerInterimAnalysis(experimentId, {});

  return {
    shouldStop:
      result.decision === 'stop_efficacy' || result.decision === 'stop_futility',
    decision: result.decision,
    pValue: result.p_value,
    adjustedAlpha: result.adjusted_alpha,
    conditionalPower: result.conditional_power,
  };
}

/**
 * Monitor all active experiments.
 *
 * Convenience function to check all active experiments with default settings.
 *
 * @returns Monitoring results
 *
 * @example
 * ```typescript
 * const monitor = await monitorAllExperiments();
 * if (monitor.critical_count > 0) {
 *   console.error(`${monitor.critical_count} experiments need attention!`);
 * }
 * ```
 */
export async function monitorAllExperiments(): Promise<MonitorResponse> {
  return triggerMonitoring({
    check_srm: true,
    check_enrollment: true,
    check_fidelity: true,
    srm_threshold: 0.001,
  });
}
