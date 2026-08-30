/**
 * Segment Analysis React Query Hooks
 * ===================================
 *
 * TanStack Query hooks for the Segment Analysis API endpoints.
 * Provides typed query and mutation hooks for heterogeneous treatment
 * effect analysis and targeting optimization.
 *
 * @module hooks/api/use-segments
 */

import { useQuery, useMutation, useQueryClient, keepPreviousData } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import { ApiError } from '@/lib/api-client';
import {
  runSegmentAnalysis,
  getSegmentAnalysis,
  listPolicies,
  getSegmentHealth,
  getSegmentDatasets,
  runSegmentAnalysisAndWait,
  waitForSegmentAnalysis,
  getHighResponders,
  getOptimalPolicy,
} from '@/api/segments';
import type {
  ListPoliciesParams,
  PolicyListResponse,
  RunSegmentAnalysisRequest,
  SegmentAnalysisResponse,
  SegmentDatasetsResponse,
  SegmentHealthResponse,
} from '@/types/segments';

// =============================================================================
// QUERY HOOKS
// =============================================================================

/**
 * Hook to fetch a segment analysis by ID.
 *
 * @param analysisId - The unique analysis identifier
 * @param options - Additional query options
 * @returns Query result with segment analysis data
 *
 * @example
 * ```tsx
 * const { data, isLoading } = useSegmentAnalysis('seg_abc123');
 * if (data?.status === 'completed') {
 *   console.log(`Overall ATE: ${data.overall_ate}`);
 *   console.log(`Heterogeneity: ${data.heterogeneity_score}`);
 * }
 * ```
 */
export function useSegmentAnalysis(
  analysisId: string,
  options?: Omit<UseQueryOptions<SegmentAnalysisResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<SegmentAnalysisResponse, ApiError>({
    queryKey: queryKeys.segments.analysis(analysisId),
    queryFn: () => getSegmentAnalysis(analysisId),
    enabled: !!analysisId,
    ...options,
  });
}

/**
 * Hook to list targeting policy recommendations.
 *
 * @param params - Optional filter parameters (min_lift, min_confidence, limit)
 * @param options - Additional query options
 * @returns Query result with policy list
 *
 * @example
 * ```tsx
 * const { data } = usePolicies({ min_confidence: 0.8, min_lift: 10.0, limit: 10 });
 * data?.recommendations.forEach(policy => {
 *   console.log(`${policy.segment}: ${policy.current_treatment_rate} -> ${policy.recommended_treatment_rate}`);
 * });
 * ```
 */
export function usePolicies(
  params?: ListPoliciesParams,
  options?: Omit<UseQueryOptions<PolicyListResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<PolicyListResponse, ApiError>({
    queryKey: [...queryKeys.segments.policies(), params?.min_lift, params?.min_confidence, params?.limit],
    queryFn: () => listPolicies(params),
    staleTime: 60 * 1000,
    ...options,
  });
}

/**
 * Hook to get segment analysis service health.
 *
 * @param options - Additional query options
 * @returns Query result with service health status
 *
 * @example
 * ```tsx
 * const { data: health } = useSegmentHealth();
 * if (health?.agent_available && health?.econml_available) {
 *   console.log('Segment analyzer is ready');
 * }
 * ```
 */
export function useSegmentHealth(
  options?: Omit<UseQueryOptions<SegmentHealthResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<SegmentHealthResponse, ApiError>({
    queryKey: queryKeys.segments.health(),
    queryFn: () => getSegmentHealth(),
    staleTime: 30 * 1000,
    ...options,
  });
}

/**
 * Hook to fetch the curated config options for the Segment Analysis page.
 *
 * Returns the brand-scoped treatment/outcome options (causal_paths SSOT pairs)
 * and the data-driven brand list. Drives the agent-driven config dropdowns
 * (brand / treatment / outcome). Long-lived (registry + cohort brands rarely
 * change); keyed by brand, and the previous brand's options are kept as
 * placeholder while the next scope loads so the dropdowns never flash back to
 * the single curated defaults (which would also reset the user's selection).
 *
 * @param params - `brand` to scope the options to (undefined = all brands)
 * @param options - Additional query options
 * @returns Query result with brand-scoped treatment/outcome options + brands
 *
 * @example
 * ```tsx
 * const { data: datasets } = useSegmentDatasets({ brand: 'Remibrutinib' });
 * // datasets?.treatments, datasets?.outcomes_by_treatment, datasets?.brands
 * ```
 */
export function useSegmentDatasets(
  params?: { brand?: string },
  options?: Omit<UseQueryOptions<SegmentDatasetsResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  const brand = params?.brand;
  return useQuery<SegmentDatasetsResponse, ApiError>({
    queryKey: queryKeys.segments.datasets(brand),
    queryFn: () => getSegmentDatasets(brand),
    staleTime: 5 * 60 * 1000,
    placeholderData: keepPreviousData,
    ...options,
  });
}

// =============================================================================
// MUTATION HOOKS
// =============================================================================

/**
 * Hook to run segment analysis.
 *
 * @param options - Mutation options
 * @returns Mutation object for triggering analysis
 *
 * @example
 * ```tsx
 * const { mutate: analyze, isPending } = useRunSegmentAnalysis();
 *
 * analyze({
 *   request: {
 *     query: 'Which HCP segments respond best to rep visits?',
 *     treatment_var: 'rep_visits',
 *     outcome_var: 'trx',
 *     segment_vars: ['region', 'specialty'],
 *   },
 *   asyncMode: true,
 * });
 * ```
 */
export function useRunSegmentAnalysis(
  options?: Omit<
    UseMutationOptions<
      SegmentAnalysisResponse,
      ApiError,
      { request: RunSegmentAnalysisRequest; asyncMode?: boolean }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    SegmentAnalysisResponse,
    ApiError,
    { request: RunSegmentAnalysisRequest; asyncMode?: boolean }
  >({
    mutationFn: ({ request, asyncMode = true }) => runSegmentAnalysis(request, asyncMode),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.segments.analysis(data.analysis_id), data);
      queryClient.invalidateQueries({ queryKey: queryKeys.segments.policies() });
    },
    ...options,
  });
}

/** Submit a new analysis (POST), then poll its durable record. */
export interface RunSegmentAnalysisAndWaitVariables {
  request: RunSegmentAnalysisRequest;
  pollIntervalMs?: number;
  maxWaitMs?: number;
}

/**
 * Re-attach to an existing analysis by id — GET polling only, never a POST.
 * The page's "Keep waiting" action after a poll-ceiling expiry (#1841).
 */
export interface ResumeSegmentAnalysisVariables {
  resumeAnalysisId: string;
  pollIntervalMs?: number;
  maxWaitMs?: number;
}

export type SegmentAnalysisWaitVariables =
  | RunSegmentAnalysisAndWaitVariables
  | ResumeSegmentAnalysisVariables;

/**
 * Hook to run segment analysis and wait for completion.
 *
 * One mutation, two entry points: `{ request }` POSTs a new analysis and polls
 * it; `{ resumeAnalysisId }` re-attaches to a durable record that is still
 * running after the poll ceiling expired (the mutation error is then a
 * `SegmentAnalysisTimeoutError` carrying that id). Both land the completed
 * record in the same `data` slot, so a resumed completion renders exactly like
 * a normal one, and `error` may be an `ApiError` (transport), a
 * `SegmentAnalysisTimeoutError` (ceiling), or a plain `Error` (failed record).
 *
 * @param options - Mutation options
 * @returns Mutation object for running analysis with polling
 */
export function useRunSegmentAnalysisAndWait(
  options?: Omit<
    UseMutationOptions<SegmentAnalysisResponse, Error, SegmentAnalysisWaitVariables>,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<SegmentAnalysisResponse, Error, SegmentAnalysisWaitVariables>({
    mutationFn: (variables) =>
      'resumeAnalysisId' in variables
        ? waitForSegmentAnalysis(
            variables.resumeAnalysisId,
            variables.pollIntervalMs,
            variables.maxWaitMs
          )
        : runSegmentAnalysisAndWait(
            variables.request,
            variables.pollIntervalMs,
            variables.maxWaitMs
          ),
    // Never let react-query re-run this mutation. The app default retries
    // mutations once (src/lib/query-client.ts), but this mutationFn is "POST a
    // heavy analysis, then poll its durable record": once the POST has landed,
    // a retry — after a poll-ceiling timeout or a transient GET error — submits
    // a SECOND heavy analysis while the first still holds the worker's single
    // heavy-compute slot, and the OOM guard rejects it ("compute capacity
    // saturated; retry later") while the original completes unseen (live
    // 2026-08-30). Re-running is an explicit user action.
    retry: false,
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.segments.analysis(data.analysis_id), data);
      queryClient.invalidateQueries({ queryKey: queryKeys.segments.policies() });
    },
    ...options,
  });
}

/**
 * Hook to get high responder segments.
 *
 * @param options - Mutation options
 * @returns Mutation object for fetching high responders
 *
 * @example
 * ```tsx
 * const { mutate: findHighResponders } = useGetHighResponders();
 *
 * findHighResponders({
 *   treatmentVar: 'rep_visits',
 *   outcomeVar: 'trx',
 *   segmentVars: ['region', 'specialty'],
 *   topCount: 5,
 * });
 * ```
 */
export function useGetHighResponders(
  options?: Omit<
    UseMutationOptions<
      SegmentAnalysisResponse,
      ApiError,
      { treatmentVar: string; outcomeVar: string; segmentVars: string[]; topCount?: number }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    SegmentAnalysisResponse,
    ApiError,
    { treatmentVar: string; outcomeVar: string; segmentVars: string[]; topCount?: number }
  >({
    mutationFn: ({ treatmentVar, outcomeVar, segmentVars, topCount }) =>
      getHighResponders(treatmentVar, outcomeVar, segmentVars, topCount),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.segments.analysis(data.analysis_id), data);
      queryClient.invalidateQueries({ queryKey: queryKeys.segments.policies() });
    },
    ...options,
  });
}

/**
 * Hook to get optimal targeting policy.
 *
 * @param options - Mutation options
 * @returns Mutation object for fetching optimal policy
 *
 * @example
 * ```tsx
 * const { mutate: findOptimalPolicy } = useGetOptimalPolicy();
 *
 * findOptimalPolicy({
 *   treatmentVar: 'marketing_spend',
 *   outcomeVar: 'revenue',
 *   segmentVars: ['customer_segment', 'region'],
 * });
 * ```
 */
export function useGetOptimalPolicy(
  options?: Omit<
    UseMutationOptions<
      SegmentAnalysisResponse,
      ApiError,
      { treatmentVar: string; outcomeVar: string; segmentVars: string[] }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    SegmentAnalysisResponse,
    ApiError,
    { treatmentVar: string; outcomeVar: string; segmentVars: string[] }
  >({
    mutationFn: ({ treatmentVar, outcomeVar, segmentVars }) =>
      getOptimalPolicy(treatmentVar, outcomeVar, segmentVars),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.segments.analysis(data.analysis_id), data);
      queryClient.invalidateQueries({ queryKey: queryKeys.segments.policies() });
    },
    ...options,
  });
}

// =============================================================================
// POLLING HOOKS
// =============================================================================

/**
 * Hook to poll a segment analysis until completion.
 *
 * @param analysisId - The analysis ID to poll
 * @param options - Query options
 * @returns Query result that updates until completion
 */
export function usePollSegmentAnalysis(
  analysisId: string,
  options?: Omit<UseQueryOptions<SegmentAnalysisResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<SegmentAnalysisResponse, ApiError>({
    queryKey: queryKeys.segments.analysis(analysisId),
    queryFn: () => getSegmentAnalysis(analysisId),
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
