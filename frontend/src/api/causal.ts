/**
 * Causal Inference API Client
 * ===========================
 *
 * TypeScript API client functions for the E2I Causal Inference endpoints.
 * Uses the shared apiClient for consistent error handling and interceptors.
 *
 * Endpoints:
 * - Hierarchical CATE analysis
 * - Library routing
 * - Sequential and parallel pipelines
 * - Cross-library validation
 * - Estimator listing
 * - Service health
 *
 * @module api/causal
 */

import { get, post } from '@/lib/api-client';
import type {
  CausalLibrary,
  CrossValidationRequest,
  CrossValidationResponse,
  EstimatorListResponse,
  HierarchicalAnalysisRequest,
  HierarchicalAnalysisResponse,
  ParallelPipelineRequest,
  ParallelPipelineResponse,
  RouteQueryRequest,
  RouteQueryResponse,
  SequentialPipelineRequest,
  SequentialPipelineResponse,
  CausalHealthResponse,
} from '@/types/causal';

// =============================================================================
// CAUSAL API ENDPOINTS
// =============================================================================

const CAUSAL_BASE = '/causal';

// =============================================================================
// HIERARCHICAL ANALYSIS ENDPOINTS
// =============================================================================

/**
 * Run hierarchical CATE analysis.
 *
 * Performs segment-level CATE estimation using EconML within CausalML segments,
 * then aggregates results using nested confidence interval methodology.
 *
 * @param request - Hierarchical analysis parameters
 * @param asyncMode - If true, returns immediately with analysis ID (default: true)
 * @returns Analysis results or pending status
 *
 * @example
 * ```typescript
 * const result = await runHierarchicalAnalysis({
 *   treatment_var: 'rep_visits',
 *   outcome_var: 'trx_count',
 *   effect_modifiers: ['age', 'region', 'specialty'],
 *   n_segments: 3,
 *   segmentation_method: SegmentationMethod.QUANTILE,
 *   estimator_type: EstimatorType.CAUSAL_FOREST,
 * });
 *
 * if (result.status === CausalAnalysisStatus.COMPLETED) {
 *   console.log(`Overall ATE: ${result.overall_ate}`);
 *   result.segment_results.forEach(seg => {
 *     console.log(`${seg.segment_name}: CATE ${seg.cate_mean}`);
 *   });
 * }
 * ```
 */
export async function runHierarchicalAnalysis(
  request: HierarchicalAnalysisRequest,
  asyncMode: boolean = true
): Promise<HierarchicalAnalysisResponse> {
  return post<HierarchicalAnalysisResponse, HierarchicalAnalysisRequest>(
    `${CAUSAL_BASE}/hierarchical/analyze`,
    request,
    { params: { async_mode: asyncMode } }
  );
}

/**
 * Get hierarchical analysis results by ID.
 *
 * Use this to poll for results when running analysis asynchronously.
 *
 * @param analysisId - Unique analysis identifier
 * @returns Analysis results
 *
 * @example
 * ```typescript
 * const result = await getHierarchicalAnalysis('ha_abc123');
 * if (result.status === CausalAnalysisStatus.COMPLETED) {
 *   console.log(result.nested_ci?.aggregate_ate);
 * }
 * ```
 */
export async function getHierarchicalAnalysis(
  analysisId: string
): Promise<HierarchicalAnalysisResponse> {
  return get<HierarchicalAnalysisResponse>(
    `${CAUSAL_BASE}/hierarchical/${encodeURIComponent(analysisId)}`
  );
}

// =============================================================================
// LIBRARY ROUTING ENDPOINTS
// =============================================================================

/**
 * Route a causal query to the appropriate library.
 *
 * Analyzes the query to determine the best causal inference library
 * based on the question type (effect, heterogeneity, targeting, etc.).
 *
 * @param request - Query routing parameters
 * @returns Routing recommendation with confidence score
 *
 * @example
 * ```typescript
 * const routing = await routeQuery({
 *   query: 'Does increasing sales rep visits cause higher TRx?',
 *   treatment_var: 'rep_visits',
 *   outcome_var: 'trx_count',
 * });
 *
 * console.log(`Primary library: ${routing.primary_library}`);
 * console.log(`Confidence: ${routing.routing_confidence}`);
 * console.log(`Rationale: ${routing.routing_rationale}`);
 * ```
 */
export async function routeQuery(
  request: RouteQueryRequest
): Promise<RouteQueryResponse> {
  return post<RouteQueryResponse, RouteQueryRequest>(
    `${CAUSAL_BASE}/route`,
    request
  );
}

// =============================================================================
// PIPELINE ENDPOINTS
// =============================================================================

/**
 * Run sequential multi-library pipeline.
 *
 * Executes causal analysis through multiple libraries in sequence,
 * with state propagation between stages for refined estimates.
 *
 * @param request - Sequential pipeline configuration
 * @param asyncMode - If true, returns immediately with pipeline ID (default: true)
 * @returns Pipeline results or pending status
 *
 * @example
 * ```typescript
 * const result = await runSequentialPipeline({
 *   treatment_var: 'treatment',
 *   outcome_var: 'outcome',
 *   covariates: ['age', 'income'],
 *   stages: [
 *     { library: CausalLibrary.NETWORKX, parameters: {} },
 *     { library: CausalLibrary.DOWHY, estimator: 'propensity_score_matching' },
 *     { library: CausalLibrary.ECONML, estimator: 'causal_forest' },
 *   ],
 *   propagate_state: true,
 * });
 *
 * console.log(`Consensus effect: ${result.consensus_effect}`);
 * console.log(`Library agreement: ${result.library_agreement_score}`);
 * ```
 */
export async function runSequentialPipeline(
  request: SequentialPipelineRequest,
  asyncMode: boolean = true
): Promise<SequentialPipelineResponse> {
  return post<SequentialPipelineResponse, SequentialPipelineRequest>(
    `${CAUSAL_BASE}/pipeline/sequential`,
    request,
    { params: { async_mode: asyncMode } }
  );
}

/**
 * Run parallel multi-library analysis.
 *
 * Executes causal analysis using multiple libraries simultaneously,
 * then computes consensus estimates.
 *
 * @param request - Parallel pipeline configuration
 * @param asyncMode - If true, returns immediately with pipeline ID (default: true)
 * @returns Pipeline results or pending status
 *
 * @example
 * ```typescript
 * const result = await runParallelPipeline({
 *   treatment_var: 'treatment',
 *   outcome_var: 'outcome',
 *   libraries: [CausalLibrary.DOWHY, CausalLibrary.ECONML, CausalLibrary.CAUSALML],
 *   estimators: {
 *     econml: 'causal_forest',
 *     causalml: 'uplift_random_forest',
 *   },
 * });
 *
 * console.log(`Libraries succeeded: ${result.libraries_succeeded.join(', ')}`);
 * console.log(`Consensus: ${result.consensus_effect} [${result.consensus_ci_lower}, ${result.consensus_ci_upper}]`);
 * ```
 */
export async function runParallelPipeline(
  request: ParallelPipelineRequest,
  asyncMode: boolean = true
): Promise<ParallelPipelineResponse> {
  return post<ParallelPipelineResponse, ParallelPipelineRequest>(
    `${CAUSAL_BASE}/pipeline/parallel`,
    request,
    { params: { async_mode: asyncMode } }
  );
}

// =============================================================================
// VALIDATION ENDPOINTS
// =============================================================================

/**
 * Run cross-library validation.
 *
 * Validates causal estimates by comparing results between two libraries,
 * computing agreement metrics and confidence interval overlap.
 *
 * @param request - Cross-validation configuration
 * @returns Validation results with agreement metrics
 *
 * @example
 * ```typescript
 * const validation = await runCrossValidation({
 *   treatment_var: 'treatment',
 *   outcome_var: 'outcome',
 *   primary_library: CausalLibrary.ECONML,
 *   validation_library: CausalLibrary.CAUSALML,
 *   agreement_threshold: 0.85,
 * });
 *
 * if (validation.validation_passed) {
 *   console.log(`Validated! Agreement: ${validation.agreement_score}`);
 * } else {
 *   console.warn(`Validation failed: ${validation.recommendations.join(', ')}`);
 * }
 * ```
 */
export async function runCrossValidation(
  request: CrossValidationRequest
): Promise<CrossValidationResponse> {
  return post<CrossValidationResponse, CrossValidationRequest>(
    `${CAUSAL_BASE}/validate`,
    request
  );
}

// =============================================================================
// ESTIMATOR ENDPOINTS
// =============================================================================

/**
 * List available causal estimators.
 *
 * Returns all available estimators with their capabilities and parameters.
 *
 * @param library - Optional library filter
 * @returns List of estimators grouped by library
 *
 * @example
 * ```typescript
 * // Get all estimators
 * const all = await listEstimators();
 * console.log(`Total estimators: ${all.total}`);
 *
 * // Get EconML estimators only
 * const econml = await listEstimators(CausalLibrary.ECONML);
 * econml.estimators.forEach(e => {
 *   console.log(`${e.name}: ${e.description}`);
 * });
 * ```
 */
export async function listEstimators(
  library?: CausalLibrary
): Promise<EstimatorListResponse> {
  return get<EstimatorListResponse>(`${CAUSAL_BASE}/estimators`, {
    library,
  });
}

// =============================================================================
// HEALTH ENDPOINTS
// =============================================================================

/**
 * Get health status of causal inference service.
 *
 * Checks library availability, estimator loading, and component readiness.
 *
 * @returns Service health information
 *
 * @example
 * ```typescript
 * const health = await getCausalHealth();
 * if (health.status === 'healthy') {
 *   console.log(`Libraries: ${Object.entries(health.libraries_available)
 *     .filter(([_, v]) => v).map(([k]) => k).join(', ')}`);
 *   console.log(`${health.analysis_count_24h} analyses in last 24h`);
 * } else {
 *   console.warn(`Causal engine ${health.status}: ${health.error}`);
 * }
 * ```
 */
export async function getCausalHealth(): Promise<CausalHealthResponse> {
  return get<CausalHealthResponse>(`${CAUSAL_BASE}/health`);
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * Run hierarchical analysis and poll until complete.
 *
 * Convenience function that handles async polling automatically.
 *
 * @param request - Analysis parameters
 * @param pollIntervalMs - Polling interval in milliseconds (default: 2000)
 * @param maxWaitMs - Maximum wait time in milliseconds (default: 180000)
 * @returns Completed analysis results
 * @throws Error if analysis fails or times out
 *
 * @example
 * ```typescript
 * try {
 *   const result = await runHierarchicalAnalysisAndWait({
 *     treatment_var: 'treatment',
 *     outcome_var: 'outcome',
 *     n_segments: 4,
 *   });
 *   console.log(`Heterogeneity I²: ${result.segment_heterogeneity}`);
 * } catch (error) {
 *   console.error('Analysis failed:', error);
 * }
 * ```
 */
export async function runHierarchicalAnalysisAndWait(
  request: HierarchicalAnalysisRequest,
  pollIntervalMs: number = 2000,
  maxWaitMs: number = 180000
): Promise<HierarchicalAnalysisResponse> {
  // Start analysis asynchronously
  const initial = await runHierarchicalAnalysis(request, true);

  // If already complete, return immediately
  if (initial.status === 'completed' || initial.status === 'failed') {
    if (initial.status === 'failed') {
      throw new Error(
        `Analysis failed: ${initial.errors.join(', ') || 'Unknown error'}`
      );
    }
    return initial;
  }

  // Poll until complete or timeout
  const startTime = Date.now();
  const analysisId = initial.analysis_id;

  while (Date.now() - startTime < maxWaitMs) {
    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));

    const result = await getHierarchicalAnalysis(analysisId);

    if (result.status === 'completed') {
      return result;
    }

    if (result.status === 'failed') {
      throw new Error(
        `Analysis failed: ${result.errors.join(', ') || 'Unknown error'}`
      );
    }
  }

  throw new Error(`Analysis timed out after ${maxWaitMs}ms`);
}

/**
 * Route query and run analysis with recommended library.
 *
 * First routes the query, then executes analysis using the recommended approach.
 *
 * @param query - Natural language causal question
 * @param treatmentVar - Treatment variable
 * @param outcomeVar - Outcome variable
 * @param covariates - Optional covariate variables
 * @returns Pipeline response from recommended approach
 *
 * @example
 * ```typescript
 * const result = await routeAndRunAnalysis(
 *   'How does the effect of rep visits on TRx vary by region?',
 *   'rep_visits',
 *   'trx_count',
 *   ['age', 'specialty']
 * );
 * console.log(`Consensus effect: ${result.consensus_effect}`);
 * ```
 */
export async function routeAndRunAnalysis(
  query: string,
  treatmentVar: string,
  outcomeVar: string,
  covariates?: string[]
): Promise<ParallelPipelineResponse> {
  // First route the query
  const routing = await routeQuery({
    query,
    treatment_var: treatmentVar,
    outcome_var: outcomeVar,
  });

  // Build library list from routing recommendation
  const libraries = [
    routing.primary_library,
    ...routing.secondary_libraries.slice(0, 2),
  ];

  // Run parallel analysis with recommended libraries
  return runParallelPipeline(
    {
      treatment_var: treatmentVar,
      outcome_var: outcomeVar,
      covariates: covariates ?? [],
      libraries,
    },
    false // Run synchronously
  );
}

/**
 * Quick effect estimation using DoWhy.
 *
 * Simplified interface for basic causal effect questions.
 *
 * @param treatmentVar - Treatment variable
 * @param outcomeVar - Outcome variable
 * @param covariates - Covariate variables
 * @returns Effect estimate with confidence interval
 *
 * @example
 * ```typescript
 * const effect = await quickEffectEstimate('treatment', 'outcome', ['age']);
 * console.log(`Effect: ${effect.consensus_effect}`);
 * ```
 */
export async function quickEffectEstimate(
  treatmentVar: string,
  outcomeVar: string,
  covariates?: string[]
): Promise<ParallelPipelineResponse> {
  return runParallelPipeline(
    {
      treatment_var: treatmentVar,
      outcome_var: outcomeVar,
      covariates: covariates ?? [],
      libraries: ['dowhy' as CausalLibrary, 'econml' as CausalLibrary],
      consensus_method: 'variance_weighted',
      timeout_seconds: 60,
    },
    false
  );
}

/**
 * Full causal analysis using all libraries.
 *
 * Comprehensive analysis running all available libraries in parallel.
 *
 * @param treatmentVar - Treatment variable
 * @param outcomeVar - Outcome variable
 * @param covariates - Covariate variables
 * @returns Comprehensive pipeline response
 *
 * @example
 * ```typescript
 * const result = await fullCausalAnalysis('treatment', 'outcome', ['x1', 'x2']);
 * console.log(`Agreement: ${result.library_agreement_score}`);
 * Object.entries(result.library_results).forEach(([lib, res]) => {
 *   console.log(`${lib}: ${(res as { effect?: number }).effect}`);
 * });
 * ```
 */
export async function fullCausalAnalysis(
  treatmentVar: string,
  outcomeVar: string,
  covariates?: string[]
): Promise<ParallelPipelineResponse> {
  return runParallelPipeline(
    {
      treatment_var: treatmentVar,
      outcome_var: outcomeVar,
      covariates: covariates ?? [],
      libraries: [
        'dowhy' as CausalLibrary,
        'econml' as CausalLibrary,
        'causalml' as CausalLibrary,
        'networkx' as CausalLibrary,
      ],
      consensus_method: 'variance_weighted',
      timeout_seconds: 180,
    },
    false
  );
}
