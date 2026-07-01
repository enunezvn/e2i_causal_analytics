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
import {
  CausalAnalysisHistoryResponseWireSchema,
  CausalHealthResponseWireSchema,
  EstimatorListResponseWireSchema,
} from '@/lib/api-schemas';
import type {
  AgentCausalAnalysisRequest,
  AgentCausalAnalysisResponse,
  CausalAnalysisHistoryResponse,
  CausalBrandsResponse,
  CausalLibrary,
  CausalVariablesResponse,
  ClinicalContext,
  DiscoverEffectsResponse,
  ProposeQuestionsResponse,
  CrossValidationRequest,
  CrossValidationResponse,
  EstimationDataResponse,
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
  TreatmentEffectResponse,
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
 * Run the causal_impact agent end-to-end.
 *
 * Leverages the agent (NOT the manual hierarchical/pipeline knobs): it builds the
 * causal DAG, selects an estimator data-drivenly via the energy-score router
 * (or the forced `estimator`), estimates the treatment->outcome effect, and runs
 * refutation + sensitivity. Real data is loaded server-side from the gold-standard
 * dataset; the agent fails closed (no fabricated ATE) when it cannot estimate.
 *
 * @example
 * ```typescript
 * const result = await runCausalAgentAnalysis({
 *   treatment_var: 'treatment_arm',
 *   outcome_var: 'persistent_180d',
 *   // estimator omitted => Auto (agent picks from the registry)
 * });
 * console.log(result.ate, result.selected_estimator, result.dag.edges);
 * ```
 */
export async function runCausalAgentAnalysis(
  request: AgentCausalAnalysisRequest
): Promise<AgentCausalAnalysisResponse> {
  return post<AgentCausalAnalysisResponse, AgentCausalAnalysisRequest>(
    `${CAUSAL_BASE}/agent-analyze`,
    request
  );
}

/** Poll a submitted agent run by id. */
export async function getCausalAgentAnalysis(
  analysisId: string
): Promise<AgentCausalAnalysisResponse> {
  return get<AgentCausalAnalysisResponse>(
    `${CAUSAL_BASE}/agent-analyze/${encodeURIComponent(analysisId)}`
  );
}

/**
 * Submit an agent run and poll until it finishes.
 *
 * The agent's energy-score selection + refutation takes minutes, so the run is
 * async (submit -> poll). Resolves with the final response for ANY terminal
 * status — including `failed` (the page renders the honest fail-closed result
 * with its warnings); only a network error or a poll timeout throws.
 */
export async function runCausalAgentAnalysisAndWait(
  request: AgentCausalAnalysisRequest,
  pollIntervalMs: number = 2500,
  maxWaitMs: number = 900000
): Promise<AgentCausalAnalysisResponse> {
  const isTerminal = (s: string) =>
    s === 'completed' || s === 'needs_review' || s === 'failed';

  const initial = await runCausalAgentAnalysis(request);
  if (isTerminal(initial.status)) return initial;

  const startTime = Date.now();
  const analysisId = initial.analysis_id;
  while (Date.now() - startTime < maxWaitMs) {
    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
    const result = await getCausalAgentAnalysis(analysisId);
    if (isTerminal(result.status)) return result;
  }
  throw new Error(`Causal agent analysis timed out after ${maxWaitMs}ms`);
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
// DATASET / VARIABLE DISCOVERY ENDPOINTS
// =============================================================================

/**
 * List candidate treatment / outcome / covariate variables for a dataset.
 *
 * Powers the page's variable selectors so they only offer columns that exist
 * in the real estimation frame (no fictional `rep_visits` / `trx_count`
 * defaults).
 *
 * @param dataset - Dataset to introspect (default: 'patient_journeys')
 * @returns Candidate variable lists and the full column set
 *
 * @example
 * ```typescript
 * const vars = await getCausalVariables('patient_journeys');
 * console.log(vars.treatment_candidates); // ['treatment_arm', 'treatment_initiated']
 * ```
 */
export async function getCausalVariables(
  dataset: string = 'patient_journeys'
): Promise<CausalVariablesResponse> {
  // `get(endpoint, params)` takes a FLAT params object and wraps it for axios
  // itself. Passing `{ params: {...} }` here double-wraps it into
  // `params[dataset]=...`, which the backend ignores (dataset has a default).
  return get<CausalVariablesResponse>(`${CAUSAL_BASE}/variables`, {
    dataset,
  });
}

/**
 * Fetch agent-proposed, data-ranked candidate causal questions for a dataset.
 *
 * The agent ranks candidate treatment->outcome pairs by a data-driven screening
 * signal (adjusted association strength) so the analyst confirms a question
 * rather than guessing. NOT a validated effect — run the analysis for that.
 */
export async function proposeCausalQuestions(
  dataset: string = 'patient_journeys'
): Promise<ProposeQuestionsResponse> {
  return get<ProposeQuestionsResponse>(`${CAUSAL_BASE}/propose-questions`, {
    dataset,
  });
}

/**
 * Submit a discover-effects job: the agent validates each candidate question
 * (DAG + estimator + refutation gate) and ranks the effects. Heavy/async →
 * returns a pending job; poll {@link getDiscoverEffects}.
 *
 * @param brand - optional brand to scope the cohort to (a row subset). Omit /
 *   null = all brands.
 */
export async function discoverCausalEffects(
  dataset: string = 'patient_journeys',
  brand?: string | null
): Promise<DiscoverEffectsResponse> {
  // `dataset` (and optional `brand`) are query params (no request body).
  const params = new URLSearchParams({ dataset });
  if (brand) params.set('brand', brand);
  return post<DiscoverEffectsResponse, Record<string, never>>(
    `${CAUSAL_BASE}/discover-effects?${params.toString()}`,
    {}
  );
}

/** List the brands present in a dataset's cohort (drives the brand dropdown). */
export async function getCausalBrands(
  dataset: string = 'patient_journeys'
): Promise<CausalBrandsResponse> {
  return get<CausalBrandsResponse>(
    `${CAUSAL_BASE}/brands?dataset=${encodeURIComponent(dataset)}`
  );
}

/**
 * Fetch the brand-faithful, sourced clinical context for a discovered effect
 * (drug + mechanism of action, the disease's real pivotal endpoints, a
 * real-world-evidence citation). Additive narrative; never changes the estimate.
 */
export async function getClinicalContext(
  brand: string,
  outcome: string
): Promise<ClinicalContext> {
  // Flat params: `get(endpoint, params)` wraps them for axios (see api-client).
  return get<ClinicalContext>(`${CAUSAL_BASE}/clinical-context`, { brand, outcome });
}

/** Poll a discover-effects job by id (ranked effects fill in progressively). */
export async function getDiscoverCausalEffects(
  jobId: string
): Promise<DiscoverEffectsResponse> {
  return get<DiscoverEffectsResponse>(
    `${CAUSAL_BASE}/discover-effects/${encodeURIComponent(jobId)}`
  );
}

/**
 * Fetch real estimation-ready records for the chosen variables.
 *
 * The returned `estimation_data_records` are passed verbatim into the parallel
 * pipeline request's `filters` so the libraries estimate effects on real rows.
 *
 * @param args - Dataset + treatment / outcome / covariates + row limit
 * @returns Estimation-ready records and their columns
 *
 * @example
 * ```typescript
 * const data = await getCausalEstimationData({
 *   treatment_var: 'treatment_arm',
 *   outcome_var: 'persistent_180d',
 *   covariates: ['disease_severity', 'engagement_score', 'age_at_diagnosis'],
 * });
 * console.log(`${data.n_rows} rows ready for estimation`);
 * ```
 */
export async function getCausalEstimationData(args: {
  dataset?: string;
  treatment_var: string;
  outcome_var: string;
  covariates?: string[];
  limit?: number;
}): Promise<EstimationDataResponse> {
  // Flat params: `get()` wraps them for axios. The previous `{ params: {...} }`
  // double-wrap serialized `params[treatment_var]=...`, so the backend's
  // REQUIRED treatment_var/outcome_var arrived missing → 422 (the live bug that
  // made "Run parallel pipeline" fail with "Could not load estimation data").
  return get<EstimationDataResponse>(`${CAUSAL_BASE}/estimation-data`, {
    dataset: args.dataset ?? 'patient_journeys',
    treatment_var: args.treatment_var,
    outcome_var: args.outcome_var,
    covariates: (args.covariates ?? []).join(','),
    limit: args.limit ?? 4000,
  });
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
  return get<EstimatorListResponse>(
    `${CAUSAL_BASE}/estimators`,
    {
      library,
    },
    { schema: EstimatorListResponseWireSchema }
  );
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
  return get<CausalHealthResponse>(`${CAUSAL_BASE}/health`, undefined, {
    schema: CausalHealthResponseWireSchema,
  });
}

/**
 * Get recent completed causal analyses for the Analysis History tab.
 *
 * Returns REAL `causal_analysis_completed` episodic events (newest first),
 * or an honest empty list when none exist.
 *
 * @param limit - Maximum history items to return (1-100, default 20)
 * @returns Recent completed causal analyses
 *
 * @example
 * ```typescript
 * const history = await getCausalAnalysisHistory(20);
 * console.log(`${history.total} recent analyses`);
 * ```
 */
export async function getCausalAnalysisHistory(
  limit: number = 20
): Promise<CausalAnalysisHistoryResponse> {
  return get<CausalAnalysisHistoryResponse>(
    `${CAUSAL_BASE}/history`,
    { limit },
    { schema: CausalAnalysisHistoryResponseWireSchema }
  );
}

// =============================================================================
// TREATMENT EFFECTS ENDPOINTS
// =============================================================================

/**
 * Estimate the treatment effect for one (cohort, brand) cell.
 *
 * Loads a confounded cohort frame from the DB and runs the live DoWhy+EconML
 * sequential pipeline to recover a de-confounded ATE + CI + p_value + n. This is
 * a HEAVY synchronous compute (~10-90s); call it only when both cohort and brand
 * are chosen (ideally behind an explicit Run affordance).
 *
 * @param cohort - initiation | persistence | discontinuation | hcp_adoption
 * @param brand - Remibrutinib | Fabhalta | Kisqali
 * @returns Real treatment-effect estimate for the cell
 *
 * @example
 * ```typescript
 * const te = await getTreatmentEffects('hcp_adoption', 'Fabhalta');
 * console.log(`ATE: ${te.ate} [${te.ci_lower}, ${te.ci_upper}]`);
 * ```
 */
export async function getTreatmentEffects(
  cohort: string,
  brand: string
): Promise<TreatmentEffectResponse> {
  return get<TreatmentEffectResponse>(
    `${CAUSAL_BASE}/treatment-effects`,
    { cohort, brand },
    // The DoWhy+EconML fit is a heavy synchronous compute (~40s measured for
    // hcp_adoption/Remibrutinib; backend budgets 90s via _TE_TIMEOUT_SECONDS,
    // nginx allows 120s). Override the 30s client default so we wait for the
    // real estimate instead of aborting mid-fit. A genuine 90s-cap hit still
    // returns the backend's clean 408, which the UI renders as "unavailable".
    { timeout: 95000 }
  );
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
