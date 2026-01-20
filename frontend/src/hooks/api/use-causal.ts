/**
 * Causal Inference React Query Hooks
 * ===================================
 *
 * TanStack Query hooks for the Causal Inference API endpoints.
 * Provides typed query and mutation hooks for causal analysis.
 *
 * @module hooks/api/use-causal
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import { ApiError } from '@/lib/api-client';
import {
  runHierarchicalAnalysis,
  getHierarchicalAnalysis,
  routeQuery,
  runSequentialPipeline,
  runParallelPipeline,
  runCrossValidation,
  listEstimators,
  getCausalHealth,
  runHierarchicalAnalysisAndWait,
  routeAndRunAnalysis,
  quickEffectEstimate,
  fullCausalAnalysis,
} from '@/api/causal';
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
