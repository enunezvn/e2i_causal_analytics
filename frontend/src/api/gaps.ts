/**
 * Gap Analysis API Client
 * =======================
 *
 * TypeScript API client functions for the E2I Gap Analysis endpoints.
 * Uses the shared apiClient for consistent error handling and interceptors.
 *
 * Endpoints:
 * - Gap analysis execution
 * - Opportunity listing
 * - Service health
 *
 * @module api/gaps
 */

import { get, post } from '@/lib/api-client';
import type {
  GapAnalysisResponse,
  GapHealthResponse,
  ListOpportunitiesParams,
  OpportunityListResponse,
  RunGapAnalysisRequest,
} from '@/types/gaps';

// =============================================================================
// GAP ANALYSIS API ENDPOINTS
// =============================================================================

const GAPS_BASE = '/gaps';

// =============================================================================
// ANALYSIS ENDPOINTS
// =============================================================================

/**
 * Run gap analysis for a brand.
 *
 * Invokes the Gap Analyzer agent (Tier 2) to detect performance gaps,
 * calculate ROI for closing each gap, and prioritize opportunities.
 *
 * @param request - Gap analysis parameters
 * @param asyncMode - If true, returns immediately with analysis ID (default: true)
 * @returns Gap analysis results or pending status
 *
 * @example
 * ```typescript
 * const result = await runGapAnalysis({
 *   query: 'Identify performance gaps for Kisqali in Q4',
 *   brand: 'kisqali',
 *   metrics: ['trx', 'market_share'],
 *   segments: ['region', 'specialty'],
 * });
 *
 * if (result.status === AnalysisStatus.COMPLETED) {
 *   console.log(`Found ${result.prioritized_opportunities.length} opportunities`);
 *   console.log(`Total value: $${result.total_addressable_value}`);
 * }
 * ```
 */
export async function runGapAnalysis(
  request: RunGapAnalysisRequest,
  asyncMode: boolean = true
): Promise<GapAnalysisResponse> {
  return post<GapAnalysisResponse, RunGapAnalysisRequest>(
    `${GAPS_BASE}/analyze`,
    request,
    { params: { async_mode: asyncMode } }
  );
}

/**
 * Get gap analysis results by ID.
 *
 * Use this to poll for results when running analysis asynchronously.
 *
 * @param analysisId - Unique analysis identifier
 * @returns Gap analysis results
 *
 * @example
 * ```typescript
 * const analysis = await getGapAnalysis('gap_abc123');
 * if (analysis.status === AnalysisStatus.COMPLETED) {
 *   console.log(analysis.executive_summary);
 *   analysis.quick_wins.forEach(opp => {
 *     console.log(`${opp.rank}. ${opp.recommended_action}`);
 *   });
 * }
 * ```
 */
export async function getGapAnalysis(
  analysisId: string
): Promise<GapAnalysisResponse> {
  return get<GapAnalysisResponse>(
    `${GAPS_BASE}/${encodeURIComponent(analysisId)}`
  );
}

// =============================================================================
// OPPORTUNITY ENDPOINTS
// =============================================================================

/**
 * List prioritized opportunities across all analyses.
 *
 * Returns opportunities filtered and sorted by ROI.
 *
 * @param params - Optional filters for brand, ROI, and difficulty
 * @returns List of prioritized opportunities
 *
 * @example
 * ```typescript
 * // Get all quick wins
 * const quickWins = await listOpportunities({
 *   difficulty: ImplementationDifficulty.LOW,
 *   min_roi: 2.0,
 *   limit: 10,
 * });
 *
 * // Get opportunities for specific brand
 * const kisqaliOpps = await listOpportunities({
 *   brand: 'kisqali',
 * });
 * ```
 */
export async function listOpportunities(
  params?: ListOpportunitiesParams
): Promise<OpportunityListResponse> {
  return get<OpportunityListResponse>(`${GAPS_BASE}/opportunities`, {
    brand: params?.brand,
    min_roi: params?.min_roi,
    difficulty: params?.difficulty,
    limit: params?.limit,
  });
}

// =============================================================================
// HEALTH ENDPOINTS
// =============================================================================

/**
 * Get health status of gap analysis service.
 *
 * Checks agent availability and returns service metrics.
 *
 * @returns Service health information
 *
 * @example
 * ```typescript
 * const health = await getGapHealth();
 * if (health.agent_available) {
 *   console.log(`Service healthy, ${health.analyses_24h} analyses in last 24h`);
 * } else {
 *   console.warn('Gap Analyzer agent unavailable');
 * }
 * ```
 */
export async function getGapHealth(): Promise<GapHealthResponse> {
  return get<GapHealthResponse>(`${GAPS_BASE}/health`);
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * Run gap analysis and poll until complete.
 *
 * Convenience function that handles async polling automatically.
 *
 * @param request - Gap analysis parameters
 * @param pollIntervalMs - Polling interval in milliseconds (default: 2000)
 * @param maxWaitMs - Maximum wait time in milliseconds (default: 60000)
 * @returns Completed gap analysis results
 * @throws Error if analysis fails or times out
 *
 * @example
 * ```typescript
 * try {
 *   const result = await runGapAnalysisAndWait({
 *     query: 'Find gaps in Northeast region',
 *     brand: 'kisqali',
 *     metrics: ['trx'],
 *     segments: ['region'],
 *   });
 *   console.log(result.executive_summary);
 * } catch (error) {
 *   console.error('Analysis failed:', error);
 * }
 * ```
 */
export async function runGapAnalysisAndWait(
  request: RunGapAnalysisRequest,
  pollIntervalMs: number = 2000,
  maxWaitMs: number = 60000
): Promise<GapAnalysisResponse> {
  // Start analysis asynchronously
  const initial = await runGapAnalysis(request, true);

  // If already complete, return immediately
  if (initial.status === 'completed' || initial.status === 'failed') {
    return initial;
  }

  // Poll until complete or timeout
  const startTime = Date.now();
  const analysisId = initial.analysis_id;

  while (Date.now() - startTime < maxWaitMs) {
    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));

    const result = await getGapAnalysis(analysisId);

    if (result.status === 'completed') {
      return result;
    }

    if (result.status === 'failed') {
      throw new Error(
        `Gap analysis failed: ${result.warnings.join(', ') || 'Unknown error'}`
      );
    }
  }

  throw new Error(`Gap analysis timed out after ${maxWaitMs}ms`);
}

/**
 * Get quick win opportunities for a brand.
 *
 * Convenience function to get low-difficulty, high-ROI opportunities.
 *
 * @param brand - Brand identifier
 * @param limit - Maximum number of results (default: 5)
 * @returns List of quick win opportunities
 *
 * @example
 * ```typescript
 * const quickWins = await getQuickWins('kisqali', 3);
 * quickWins.forEach(opp => {
 *   console.log(`${opp.recommended_action} - ROI: ${opp.roi_estimate.expected_roi}x`);
 * });
 * ```
 */
export async function getQuickWins(
  brand: string,
  limit: number = 5
): Promise<OpportunityListResponse> {
  return listOpportunities({
    brand,
    difficulty: 'low' as never, // ImplementationDifficulty.LOW
    limit,
  });
}

/**
 * Get strategic bet opportunities for a brand.
 *
 * Convenience function to get high-impact, high-difficulty opportunities.
 *
 * @param brand - Brand identifier
 * @param limit - Maximum number of results (default: 5)
 * @returns List of strategic bet opportunities
 *
 * @example
 * ```typescript
 * const strategicBets = await getStrategicBets('kisqali', 3);
 * strategicBets.forEach(opp => {
 *   console.log(`${opp.recommended_action} - Value: $${opp.roi_estimate.estimated_revenue_impact}`);
 * });
 * ```
 */
export async function getStrategicBets(
  brand: string,
  limit: number = 5
): Promise<OpportunityListResponse> {
  return listOpportunities({
    brand,
    difficulty: 'high' as never, // ImplementationDifficulty.HIGH
    limit,
  });
}
