/**
 * Causal Inference React Query Hooks
 * ===================================
 *
 * TanStack Query hooks for the Causal Inference API endpoints.
 * Provides typed query and mutation hooks for causal analysis.
 *
 * @module hooks/api/use-causal
 */

import { useEffect, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import { ApiError } from '@/lib/api-client';
import {
  runHierarchicalAnalysis,
  runCausalAgentAnalysisAndWait,
  getHierarchicalAnalysis,
  getCausalVariables,
  proposeCausalQuestions,
  discoverCausalEffects,
  getDiscoverCausalEffects,
  getCausalBrands,
  getClinicalContext,
  routeQuery,
  runSequentialPipeline,
  runParallelPipeline,
  runCrossValidation,
  listEstimators,
  getCausalHealth,
  getCausalAnalysisHistory,
  getTreatmentEffects,
  runHierarchicalAnalysisAndWait,
  routeAndRunAnalysis,
  quickEffectEstimate,
  fullCausalAnalysis,
} from '@/api/causal';
import type {
  AgentCausalAnalysisRequest,
  AgentCausalAnalysisResponse,
  CausalAnalysisHistoryResponse,
  CausalBrandsResponse,
  CausalLibrary,
  CausalVariablesResponse,
  ClinicalContext,
  ProposeQuestionsResponse,
  DiscoverEffectsResponse,
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
  TreatmentEffectResponse,
} from '@/types/causal';

// =============================================================================
// QUERY HOOKS
// =============================================================================

/**
 * Hook to fetch a hierarchical analysis by ID.
 *
 * @param analysisId - The unique analysis identifier
 * @param options - Additional query options
 * @returns Query result with hierarchical analysis data
 *
 * @example
 * ```tsx
 * const { data, isLoading } = useHierarchicalAnalysis('ha_abc123');
 * if (data?.status === 'completed') {
 *   console.log(`Overall ATE: ${data.overall_ate}`);
 * }
 * ```
 */
export function useHierarchicalAnalysis(
  analysisId: string,
  options?: Omit<UseQueryOptions<HierarchicalAnalysisResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<HierarchicalAnalysisResponse, ApiError>({
    queryKey: queryKeys.causal.hierarchicalAnalysis(analysisId),
    queryFn: () => getHierarchicalAnalysis(analysisId),
    enabled: !!analysisId,
    ...options,
  });
}

/**
 * Hook to list candidate treatment / outcome / covariate variables for a
 * dataset.
 *
 * Feeds the Causal Discovery page's variable selectors so they only offer
 * columns that exist in the real estimation frame.
 *
 * @param dataset - Dataset to introspect (default: 'patient_journeys')
 * @param brand - Brand the analysis is scoped to; covariate candidates are
 *   brand-scoped server-side. null/undefined = all brands (universals only).
 * @param options - Additional query options
 * @returns Query result with candidate variable lists
 *
 * @example
 * ```tsx
 * const { data: variables } = useCausalVariables('patient_journeys', 'Fabhalta');
 * variables?.covariate_candidates.forEach((c) => console.log(c));
 * ```
 */
export function useCausalVariables(
  dataset: string = 'patient_journeys',
  brand?: string | null,
  options?: Omit<UseQueryOptions<CausalVariablesResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<CausalVariablesResponse, ApiError>({
    // brand is part of the key: a brand switch must refetch, not serve the
    // previous brand's candidates from the 5-minute cache.
    queryKey: queryKeys.causal.variables(dataset, brand ?? null),
    queryFn: () => getCausalVariables(dataset, brand),
    staleTime: 5 * 60 * 1000, // 5 minutes - dataset schema rarely changes
    ...options,
  });
}

/**
 * Hook for agent-proposed, data-ranked candidate causal questions.
 *
 * The agent ranks candidate treatment->outcome pairs by a data-driven screening
 * signal so the analyst confirms a question instead of guessing. Screening
 * signal only — the analysis run validates it.
 */
export function useProposeQuestions(
  dataset: string = 'patient_journeys',
  options?: Omit<UseQueryOptions<ProposeQuestionsResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<ProposeQuestionsResponse, ApiError>({
    queryKey: ['causal', 'propose-questions', dataset],
    queryFn: () => proposeCausalQuestions(dataset),
    staleTime: 5 * 60 * 1000,
    ...options,
  });
}

/**
 * Discover-effects leaderboard: submit a job, then poll it (every 3s) until the
 * agent has validated every candidate question. The ranked effects fill in
 * progressively. ``start()`` submits; ``job`` is the latest (submit or poll)
 * state.
 *
 * @param brand - optional brand to scope the cohort to (a row subset). Omit /
 *   null = all brands.
 */
export function useDiscoverEffects(
  dataset: string = 'patient_journeys',
  brand?: string | null
) {
  const brandKey = brand ?? null;
  // The active job is TAGGED with the (dataset, brand) it was discovered for, so
  // a job started for one scope can never surface under another — not via a
  // lingering poll AND not via a submit that resolves AFTER a grain/brand switch
  // (TanStack reset() does not suppress an in-flight mutation's onSuccess, so a
  // bare setJobId there would re-adopt the stale job). `initial` preserves the
  // immediate submit response so the leaderboard renders without waiting a poll.
  const [active, setActive] = useState<{
    jobId: string;
    dataset: string;
    brand: string | null;
    initial: DiscoverEffectsResponse;
  } | null>(null);

  const {
    mutate: startMutate,
    isPending: isStarting,
    error: startError,
    reset: resetSubmit,
  } = useMutation<{ job: DiscoverEffectsResponse; dataset: string; brand: string | null }, ApiError>(
    {
      mutationFn: async () => {
        const job = await discoverCausalEffects(dataset, brand);
        return { job, dataset, brand: brandKey };
      },
      onSuccess: ({ job, dataset: ds, brand: br }) =>
        setActive({ jobId: job.job_id, dataset: ds, brand: br, initial: job }),
    }
  );

  // Drop the active job (and any retained submit state) whenever the scope
  // changes, so the previous scope's leaderboard returns to the honest empty
  // state instead of e.g. showing a Patient-grain leaderboard under HCP.
  useEffect(() => {
    setActive(null);
    resetSubmit();
  }, [dataset, brandKey, resetSubmit]);

  const inScope = active !== null && active.dataset === dataset && active.brand === brandKey;
  const jobId = inScope ? active.jobId : null;

  const poll = useQuery<DiscoverEffectsResponse, ApiError>({
    queryKey: ['causal', 'discover-effects', jobId],
    queryFn: () => getDiscoverCausalEffects(jobId as string),
    enabled: !!jobId,
    // Poll until the job completes; then stop.
    refetchInterval: (query) =>
      query.state.data && query.state.data.status === 'completed' ? false : 3000,
  });

  return {
    start: startMutate,
    isStarting,
    startError,
    job: inScope ? poll.data ?? active.initial : null,
    isPolling: poll.isFetching,
  };
}

/**
 * List the brands present in a dataset's cohort — drives the discovery page's
 * brand dropdown (data-driven: only brands with real rows are offered).
 */
export function useCausalBrands(
  dataset: string = 'patient_journeys',
  options?: Omit<UseQueryOptions<CausalBrandsResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<CausalBrandsResponse, ApiError>({
    queryKey: ['causal', 'brands', dataset],
    queryFn: () => getCausalBrands(dataset),
    staleTime: 5 * 60 * 1000,
    ...options,
  });
}

/**
 * Fetch the clinical context (drug + MoA, real pivotal endpoints, RWE citation)
 * for a discovered effect's brand + outcome, and — when `treatment` is given — for
 * that specific (treatment -> outcome) analysis. Additive narrative — does not touch
 * the causal estimate. Disabled until both `brand` and `outcome` are present;
 * `treatment` is optional so the brand-level view keeps working unchanged.
 */
export function useClinicalContext(
  brand: string | null | undefined,
  outcome: string | null | undefined,
  treatment?: string | null,
  options?: Omit<UseQueryOptions<ClinicalContext, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<ClinicalContext, ApiError>({
    // treatment is part of the key: two analyses of one brand+outcome differ in
    // their framing and their literature citation.
    queryKey: ['causal', 'clinical-context', brand, outcome, treatment ?? null],
    queryFn: () => getClinicalContext(brand as string, outcome as string, treatment),
    // Real biomedical facts change slowly, but a DEGRADED response (an upstream
    // that was unreachable) self-heals server-side after 10 minutes — holding the
    // client copy fresh for 30 would show a stale outage for 20 minutes after the
    // backend had recovered. Match the backend's self-heal window; a fully-live
    // result is cached server-side anyway, so the refetch is nearly free.
    staleTime: 10 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
    retry: false,
    ...options,
    enabled: Boolean(brand) && Boolean(outcome) && (options?.enabled ?? true),
  });
}

/**
 * Hook to list available causal estimators.
 *
 * @param library - Optional library filter
 * @param options - Additional query options
 * @returns Query result with estimator list
 *
 * @example
 * ```tsx
 * const { data } = useEstimators('econml');
 * data?.estimators.forEach(e => console.log(e.name));
 * ```
 */
export function useEstimators(
  library?: CausalLibrary,
  options?: Omit<UseQueryOptions<EstimatorListResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<EstimatorListResponse, ApiError>({
    queryKey: queryKeys.causal.estimators(library),
    queryFn: () => listEstimators(library),
    staleTime: 5 * 60 * 1000, // 5 minutes - estimators don't change often
    ...options,
  });
}

/**
 * Hook to get causal inference service health.
 *
 * @param options - Additional query options
 * @returns Query result with service health status
 *
 * @example
 * ```tsx
 * const { data: health } = useCausalHealth();
 * if (health?.status === 'healthy') {
 *   console.log('Causal engine is ready');
 * }
 * ```
 */
export function useCausalHealth(
  options?: Omit<UseQueryOptions<CausalHealthResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<CausalHealthResponse, ApiError>({
    queryKey: queryKeys.causal.health(),
    queryFn: () => getCausalHealth(),
    staleTime: 30 * 1000,
    ...options,
  });
}

/**
 * Hook to fetch recent completed causal analyses for the Analysis History tab.
 *
 * Returns REAL `causal_analysis_completed` episodic events (newest first), or an
 * honest empty list when none exist.
 *
 * @param limit - Maximum history items to return (1-100, default 20)
 * @param options - Additional query options
 * @returns Query result with recent causal analyses
 *
 * @example
 * ```tsx
 * const { data: history } = useCausalAnalysisHistory();
 * console.log(`${history?.total ?? 0} recent analyses`);
 * ```
 */
export function useCausalAnalysisHistory(
  limit: number = 20,
  options?: Omit<UseQueryOptions<CausalAnalysisHistoryResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<CausalAnalysisHistoryResponse, ApiError>({
    queryKey: queryKeys.causal.history(limit),
    queryFn: () => getCausalAnalysisHistory(limit),
    staleTime: 30 * 1000,
    ...options,
  });
}

/**
 * Hook to estimate the treatment effect for one (cohort, brand) cell.
 *
 * Runs the live DoWhy+EconML sequential pipeline server-side (~5-30s heavy
 * compute). DISABLED by default until both `cohort` and `brand` are set AND
 * `enabled` is true — the page gates it behind an explicit Run button so the
 * heavy endpoint is not spammed on every selector change.
 *
 * @param cohort - Cohort name (initiation/persistence/discontinuation/hcp_adoption)
 * @param brand - Brand (Remibrutinib/Fabhalta/Kisqali)
 * @param options - Additional query options (commonly `{ enabled }`)
 * @returns Query result with the real treatment-effect estimate
 *
 * @example
 * ```tsx
 * const { data, isFetching } = useTreatmentEffects(cohort, brand, { enabled: run });
 * ```
 */
export function useTreatmentEffects(
  cohort: string | null,
  brand: string | null,
  options?: Omit<UseQueryOptions<TreatmentEffectResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<TreatmentEffectResponse, ApiError>({
    queryKey: queryKeys.causal.treatmentEffects(cohort ?? undefined, brand ?? undefined),
    queryFn: () => getTreatmentEffects(cohort as string, brand as string),
    // Heavy compute: cache aggressively and never auto-refetch in the
    // background; re-running is an explicit user action.
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    retry: false,
    refetchOnWindowFocus: false,
    ...options,
    // Caller-supplied enabled is ANDed with both selections being present so the
    // long query never fires on a half-specified cell.
    enabled: Boolean(cohort) && Boolean(brand) && (options?.enabled ?? true),
  });
}

// =============================================================================
// MUTATION HOOKS
// =============================================================================

/**
 * Hook to run hierarchical CATE analysis.
 *
 * @param options - Mutation options
 * @returns Mutation object for triggering analysis
 *
 * @example
 * ```tsx
 * const { mutate: runAnalysis, isPending } = useRunHierarchicalAnalysis();
 *
 * runAnalysis({
 *   request: {
 *     treatment_var: 'rep_visits',
 *     outcome_var: 'trx_count',
 *     effect_modifiers: ['age', 'region'],
 *   },
 *   asyncMode: true,
 * });
 * ```
 */
export function useRunHierarchicalAnalysis(
  options?: Omit<
    UseMutationOptions<
      HierarchicalAnalysisResponse,
      ApiError,
      { request: HierarchicalAnalysisRequest; asyncMode?: boolean }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    HierarchicalAnalysisResponse,
    ApiError,
    { request: HierarchicalAnalysisRequest; asyncMode?: boolean }
  >({
    mutationFn: ({ request, asyncMode = true }) => runHierarchicalAnalysis(request, asyncMode),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.causal.hierarchicalAnalysis(data.analysis_id), data);
    },
    ...options,
  });
}

/**
 * Hook to run hierarchical analysis and wait for completion.
 *
 * @param options - Mutation options
 * @returns Mutation object for running analysis with polling
 */
export function useRunHierarchicalAnalysisAndWait(
  options?: Omit<
    UseMutationOptions<
      HierarchicalAnalysisResponse,
      ApiError,
      { request: HierarchicalAnalysisRequest; pollIntervalMs?: number; maxWaitMs?: number }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    HierarchicalAnalysisResponse,
    ApiError,
    { request: HierarchicalAnalysisRequest; pollIntervalMs?: number; maxWaitMs?: number }
  >({
    mutationFn: ({ request, pollIntervalMs, maxWaitMs }) =>
      runHierarchicalAnalysisAndWait(request, pollIntervalMs, maxWaitMs),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.causal.hierarchicalAnalysis(data.analysis_id), data);
    },
    // The mutationFn is POST + poll-to-completion. The app's QueryClient
    // retried mutations once by default until #1846 (src/lib/query-client.ts,
    // now retry: 0); this stays explicit because under ANY client default a
    // retry — after a poll-ceiling timeout or a transient GET error — submits
    // a SECOND heavy CATE analysis while the first still holds the worker's
    // single heavy-compute slot (#1839; same defect as the segment hook,
    // #1836). The `retry: false` on this module's QUERY hooks does not cover
    // a mutation. Re-running is an explicit user action, never a silent retry.
    retry: false,
    ...options,
  });
}

/**
 * Hook to route a causal query to the appropriate library.
 *
 * @param options - Mutation options
 * @returns Mutation object for query routing
 *
 * @example
 * ```tsx
 * const { mutate: route } = useRouteQuery();
 *
 * route({
 *   query: 'Does increasing rep visits cause higher TRx?',
 *   treatment_var: 'rep_visits',
 *   outcome_var: 'trx_count',
 * });
 * ```
 */
export function useRouteQuery(
  options?: Omit<UseMutationOptions<RouteQueryResponse, ApiError, RouteQueryRequest>, 'mutationFn'>
) {
  return useMutation<RouteQueryResponse, ApiError, RouteQueryRequest>({
    mutationFn: (request) => routeQuery(request),
    ...options,
  });
}

/**
 * Hook to run a sequential multi-library pipeline.
 *
 * @param options - Mutation options
 * @returns Mutation object for sequential pipeline
 */
export function useRunSequentialPipeline(
  options?: Omit<
    UseMutationOptions<
      SequentialPipelineResponse,
      ApiError,
      { request: SequentialPipelineRequest; asyncMode?: boolean }
    >,
    'mutationFn'
  >
) {
  return useMutation<
    SequentialPipelineResponse,
    ApiError,
    { request: SequentialPipelineRequest; asyncMode?: boolean }
  >({
    mutationFn: ({ request, asyncMode = true }) => runSequentialPipeline(request, asyncMode),
    ...options,
  });
}

/**
 * Hook to run a parallel multi-library pipeline.
 *
 * @param options - Mutation options
 * @returns Mutation object for parallel pipeline
 *
 * @example
 * ```tsx
 * const { mutate: runParallel } = useRunParallelPipeline();
 *
 * runParallel({
 *   request: {
 *     treatment_var: 'treatment',
 *     outcome_var: 'outcome',
 *     libraries: ['dowhy', 'econml', 'causalml'],
 *   },
 *   asyncMode: false,
 * });
 * ```
 */
export function useRunParallelPipeline(
  options?: Omit<
    UseMutationOptions<
      ParallelPipelineResponse,
      ApiError,
      { request: ParallelPipelineRequest; asyncMode?: boolean }
    >,
    'mutationFn'
  >
) {
  return useMutation<
    ParallelPipelineResponse,
    ApiError,
    { request: ParallelPipelineRequest; asyncMode?: boolean }
  >({
    mutationFn: ({ request, asyncMode = true }) => runParallelPipeline(request, asyncMode),
    ...options,
  });
}

/**
 * Run the causal_impact agent end-to-end (DAG + treatment->outcome effect +
 * refutation). Synchronous: resolves with the full analysis. The agent picks an
 * estimator data-drivenly unless `estimator` is set.
 *
 * @example
 * ```typescript
 * const { mutateAsync: runAgent, isPending } = useRunCausalAgentAnalysis();
 * const result = await runAgent({ treatment_var: 'treatment_arm', outcome_var: 'persistent_180d' });
 * ```
 */
export function useRunCausalAgentAnalysis(
  options?: Omit<
    UseMutationOptions<AgentCausalAnalysisResponse, ApiError, AgentCausalAnalysisRequest>,
    'mutationFn'
  >
) {
  return useMutation<AgentCausalAnalysisResponse, ApiError, AgentCausalAnalysisRequest>({
    // Submit -> poll: the agent run takes minutes; `isPending` stays true for the
    // whole wait, and `data` is the final response (including honest `failed`).
    mutationFn: (request) => runCausalAgentAnalysisAndWait(request),
    // This is an *AndWait mutation (POST + poll, 900 s ceiling): a mutation
    // retry (the app default until #1846) would re-run the whole mutationFn after
    // a poll-ceiling timeout — a SECOND agent run submitted while the first is
    // still executing (#1839). Re-running is an explicit user action.
    retry: false,
    ...options,
  });
}

/**
 * Hook to run cross-library validation.
 *
 * @param options - Mutation options
 * @returns Mutation object for cross-validation
 */
export function useRunCrossValidation(
  options?: Omit<UseMutationOptions<CrossValidationResponse, ApiError, CrossValidationRequest>, 'mutationFn'>
) {
  return useMutation<CrossValidationResponse, ApiError, CrossValidationRequest>({
    mutationFn: (request) => runCrossValidation(request),
    ...options,
  });
}

/**
 * Hook to route a query and run analysis with recommended libraries.
 *
 * @param options - Mutation options
 * @returns Mutation object for routing and running analysis
 */
export function useRouteAndRunAnalysis(
  options?: Omit<
    UseMutationOptions<
      ParallelPipelineResponse,
      ApiError,
      { query: string; treatmentVar: string; outcomeVar: string; covariates?: string[] }
    >,
    'mutationFn'
  >
) {
  return useMutation<
    ParallelPipelineResponse,
    ApiError,
    { query: string; treatmentVar: string; outcomeVar: string; covariates?: string[] }
  >({
    mutationFn: ({ query, treatmentVar, outcomeVar, covariates }) =>
      routeAndRunAnalysis(query, treatmentVar, outcomeVar, covariates),
    ...options,
  });
}

/**
 * Hook for quick effect estimation using DoWhy.
 *
 * @param options - Mutation options
 * @returns Mutation object for quick estimation
 */
export function useQuickEffectEstimate(
  options?: Omit<
    UseMutationOptions<
      ParallelPipelineResponse,
      ApiError,
      { treatmentVar: string; outcomeVar: string; covariates?: string[] }
    >,
    'mutationFn'
  >
) {
  return useMutation<
    ParallelPipelineResponse,
    ApiError,
    { treatmentVar: string; outcomeVar: string; covariates?: string[] }
  >({
    mutationFn: ({ treatmentVar, outcomeVar, covariates }) =>
      quickEffectEstimate(treatmentVar, outcomeVar, covariates),
    ...options,
  });
}

/**
 * Hook for full causal analysis using all libraries.
 *
 * @param options - Mutation options
 * @returns Mutation object for full analysis
 */
export function useFullCausalAnalysis(
  options?: Omit<
    UseMutationOptions<
      ParallelPipelineResponse,
      ApiError,
      { treatmentVar: string; outcomeVar: string; covariates?: string[] }
    >,
    'mutationFn'
  >
) {
  return useMutation<
    ParallelPipelineResponse,
    ApiError,
    { treatmentVar: string; outcomeVar: string; covariates?: string[] }
  >({
    mutationFn: ({ treatmentVar, outcomeVar, covariates }) =>
      fullCausalAnalysis(treatmentVar, outcomeVar, covariates),
    ...options,
  });
}

// =============================================================================
// POLLING HOOKS
// =============================================================================

/**
 * Hook to poll a hierarchical analysis until completion.
 *
 * @param analysisId - The analysis ID to poll
 * @param options - Query options
 * @returns Query result that updates until completion
 */
export function usePollHierarchicalAnalysis(
  analysisId: string,
  options?: Omit<UseQueryOptions<HierarchicalAnalysisResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<HierarchicalAnalysisResponse, ApiError>({
    queryKey: queryKeys.causal.hierarchicalAnalysis(analysisId),
    queryFn: () => getHierarchicalAnalysis(analysisId),
    enabled: !!analysisId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === 'completed' || status === 'failed') {
        return false;
      }
      return 2000;
    },
    ...options,
  });
}
