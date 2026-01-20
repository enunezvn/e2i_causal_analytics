/**
 * Segment Analysis API Client
 * ===========================
 *
 * TypeScript API client functions for the E2I Segment Analysis endpoints.
 * Uses the shared apiClient for consistent error handling and interceptors.
 *
 * Endpoints:
 * - Segment analysis execution (CATE estimation)
 * - Policy recommendations
 * - Service health
 *
 * @module api/segments
 */

import { get, post } from '@/lib/api-client';
import type {
  ListPoliciesParams,
  PolicyListResponse,
  RunSegmentAnalysisRequest,
  SegmentAnalysisResponse,
  SegmentHealthResponse,
} from '@/types/segments';

// =============================================================================
// SEGMENT ANALYSIS API ENDPOINTS
// =============================================================================

const SEGMENTS_BASE = '/segments';

// =============================================================================
// ANALYSIS ENDPOINTS
// =============================================================================

/**
 * Run segment analysis for treatment effect heterogeneity.
 *
 * Invokes the Heterogeneous Optimizer agent (Tier 2) to estimate CATE,
 * identify high/low responder segments, and generate targeting recommendations.
 *
 * @param request - Segment analysis parameters
 * @param asyncMode - If true, returns immediately with analysis ID (default: true)
 * @returns Segment analysis results or pending status
 *
 * @example
 * ```typescript
 * const result = await runSegmentAnalysis({
 *   query: 'Which HCP segments respond best to rep visits?',
 *   treatment_var: 'rep_visits',
 *   outcome_var: 'trx',
 *   segment_vars: ['region', 'specialty'],
 *   effect_modifiers: ['practice_size', 'years_experience'],
 * });
 *
 * if (result.status === SegmentAnalysisStatus.COMPLETED) {
 *   console.log(`Overall ATE: ${result.overall_ate}`);
 *   console.log(`Heterogeneity: ${result.heterogeneity_score}`);
 *   result.high_responders.forEach(seg => {
 *     console.log(`${seg.segment_id}: CATE ${seg.cate_estimate}`);
 *   });
 * }
 * ```
 */
export async function runSegmentAnalysis(
  request: RunSegmentAnalysisRequest,
  asyncMode: boolean = true
): Promise<SegmentAnalysisResponse> {
  return post<SegmentAnalysisResponse, RunSegmentAnalysisRequest>(
    `${SEGMENTS_BASE}/analyze`,
    request,
    { params: { async_mode: asyncMode } }
  );
}

/**
 * Get segment analysis results by ID.
 *
 * Use this to poll for results when running analysis asynchronously.
 *
 * @param analysisId - Unique analysis identifier
 * @returns Segment analysis results
 *
 * @example
 * ```typescript
 * const analysis = await getSegmentAnalysis('seg_abc123');
 * if (analysis.status === SegmentAnalysisStatus.COMPLETED) {
 *   console.log(analysis.executive_summary);
 *   analysis.policy_recommendations.forEach(policy => {
 *     console.log(`${policy.segment}: +${policy.expected_incremental_outcome}`);
 *   });
 * }
 * ```
 */
export async function getSegmentAnalysis(
  analysisId: string
): Promise<SegmentAnalysisResponse> {
  return get<SegmentAnalysisResponse>(
    `${SEGMENTS_BASE}/${encodeURIComponent(analysisId)}`
  );
}

// =============================================================================
// POLICY ENDPOINTS
// =============================================================================

/**
 * List targeting policy recommendations.
 *
 * Returns policy recommendations across all completed analyses.
 *
 * @param params - Optional filters for lift, confidence, and limit
 * @returns List of policy recommendations
 *
 * @example
 * ```typescript
 * // Get high-confidence recommendations
 * const policies = await listPolicies({
 *   min_confidence: 0.8,
 *   min_lift: 10.0,
 *   limit: 10,
 * });
 *
 * policies.recommendations.forEach(policy => {
 *   console.log(`${policy.segment}: ${policy.current_treatment_rate} -> ${policy.recommended_treatment_rate}`);
 * });
 * console.log(`Total expected lift: ${policies.expected_total_lift}`);
 * ```
 */
export async function listPolicies(
  params?: ListPoliciesParams
): Promise<PolicyListResponse> {
  return get<PolicyListResponse>(`${SEGMENTS_BASE}/policies`, {
    min_lift: params?.min_lift,
    min_confidence: params?.min_confidence,
    limit: params?.limit,
  });
}

// =============================================================================
// HEALTH ENDPOINTS
// =============================================================================

/**
 * Get health status of segment analysis service.
 *
 * Checks agent and causal library availability.
 *
 * @returns Service health information
 *
 * @example
 * ```typescript
 * const health = await getSegmentHealth();
 * if (health.agent_available && health.econml_available) {
 *   console.log(`Service healthy, ${health.analyses_24h} analyses in last 24h`);
 * } else {
 *   console.warn('Some components unavailable');
 * }
 * ```
 */
export async function getSegmentHealth(): Promise<SegmentHealthResponse> {
  return get<SegmentHealthResponse>(`${SEGMENTS_BASE}/health`);
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * Run segment analysis and poll until complete.
 *
 * Convenience function that handles async polling automatically.
 *
 * @param request - Segment analysis parameters
 * @param pollIntervalMs - Polling interval in milliseconds (default: 2000)
 * @param maxWaitMs - Maximum wait time in milliseconds (default: 120000)
 * @returns Completed segment analysis results
 * @throws Error if analysis fails or times out
 *
 * @example
 * ```typescript
 * try {
 *   const result = await runSegmentAnalysisAndWait({
 *     query: 'Identify best segments for targeting',
 *     treatment_var: 'email_campaigns',
 *     outcome_var: 'conversion',
 *     segment_vars: ['industry', 'company_size'],
 *   });
 *   console.log(result.executive_summary);
 *   console.log(`${result.high_responders.length} high responder segments found`);
 * } catch (error) {
 *   console.error('Analysis failed:', error);
 * }
 * ```
 */
export async function runSegmentAnalysisAndWait(
  request: RunSegmentAnalysisRequest,
  pollIntervalMs: number = 2000,
  maxWaitMs: number = 120000
): Promise<SegmentAnalysisResponse> {
  // Start analysis asynchronously
  const initial = await runSegmentAnalysis(request, true);

  // If already complete, return immediately
  if (initial.status === 'completed' || initial.status === 'failed') {
    return initial;
  }

  // Poll until complete or timeout
  const startTime = Date.now();
  const analysisId = initial.analysis_id;

  while (Date.now() - startTime < maxWaitMs) {
    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));

    const result = await getSegmentAnalysis(analysisId);

    if (result.status === 'completed') {
      return result;
    }

    if (result.status === 'failed') {
      throw new Error(
        `Segment analysis failed: ${result.warnings.join(', ') || 'Unknown error'}`
      );
    }
  }

  throw new Error(`Segment analysis timed out after ${maxWaitMs}ms`);
}

/**
 * Get high responder segments for a treatment.
 *
 * Convenience function to quickly analyze which segments respond best.
 *
 * @param treatmentVar - Treatment variable to analyze
 * @param outcomeVar - Outcome variable to measure
 * @param segmentVars - Variables to segment by
 * @param topCount - Number of top segments to return (default: 5)
 * @returns Segment analysis with high responders
 *
 * @example
 * ```typescript
 * const result = await getHighResponders(
 *   'rep_visits',
 *   'trx',
 *   ['region', 'specialty'],
 *   5
 * );
 * result.high_responders.forEach(seg => {
 *   console.log(`${seg.segment_id}: CATE ${seg.cate_estimate}, ${seg.recommendation}`);
 * });
 * ```
 */
export async function getHighResponders(
  treatmentVar: string,
  outcomeVar: string,
  segmentVars: string[],
  topCount: number = 5
): Promise<SegmentAnalysisResponse> {
  return runSegmentAnalysisAndWait({
    query: `Identify top ${topCount} high responder segments for ${treatmentVar}`,
    treatment_var: treatmentVar,
    outcome_var: outcomeVar,
    segment_vars: segmentVars,
    top_segments_count: topCount,
  });
}

/**
 * Get optimal targeting policy for a treatment.
 *
 * Convenience function to get targeting recommendations.
 *
 * @param treatmentVar - Treatment variable to optimize
 * @param outcomeVar - Outcome variable to maximize
 * @param segmentVars - Variables to segment by
 * @returns Segment analysis with policy recommendations
 *
 * @example
 * ```typescript
 * const result = await getOptimalPolicy(
 *   'marketing_spend',
 *   'revenue',
 *   ['customer_segment', 'region']
 * );
 * console.log(`Expected total lift: ${result.expected_total_lift}`);
 * console.log(result.optimal_allocation_summary);
 * ```
 */
export async function getOptimalPolicy(
  treatmentVar: string,
  outcomeVar: string,
  segmentVars: string[]
): Promise<SegmentAnalysisResponse> {
  return runSegmentAnalysisAndWait({
    query: `Generate optimal targeting policy for ${treatmentVar} to maximize ${outcomeVar}`,
    treatment_var: treatmentVar,
    outcome_var: outcomeVar,
    segment_vars: segmentVars,
    question_type: 'targeting' as never,
  });
}
