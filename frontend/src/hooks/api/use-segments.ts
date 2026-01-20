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

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import { ApiError } from '@/lib/api-client';
import {
  runSegmentAnalysis,
  getSegmentAnalysis,
  listPolicies,
  getSegmentHealth,
  runSegmentAnalysisAndWait,
  getHighResponders,
  getOptimalPolicy,
} from '@/api/segments';
import type {
  ListPoliciesParams,
  PolicyListResponse,
  RunSegmentAnalysisRequest,
  SegmentAnalysisResponse,
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

/**
 * Hook to run segment analysis and wait for completion.
 *
 * @param options - Mutation options
 * @returns Mutation object for running analysis with polling
 */
export function useRunSegmentAnalysisAndWait(
  options?: Omit<
    UseMutationOptions<
      SegmentAnalysisResponse,
      ApiError,
      { request: RunSegmentAnalysisRequest; pollIntervalMs?: number; maxWaitMs?: number }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    SegmentAnalysisResponse,
    ApiError,
    { request: RunSegmentAnalysisRequest; pollIntervalMs?: number; maxWaitMs?: number }
  >({
    mutationFn: ({ request, pollIntervalMs, maxWaitMs }) =>
      runSegmentAnalysisAndWait(request, pollIntervalMs, maxWaitMs),
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
