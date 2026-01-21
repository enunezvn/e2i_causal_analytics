/**
 * Gap Analysis React Query Hooks
 * ==============================
 *
 * TanStack Query hooks for the Gap Analysis API endpoints.
 * Provides typed query and mutation hooks with caching and invalidation.
 *
 * @module hooks/api/use-gaps
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import { ApiError } from '@/lib/api-client';
import {
  runGapAnalysis,
  getGapAnalysis,
  listOpportunities,
  getGapHealth,
  runGapAnalysisAndWait,
  getQuickWins,
  getStrategicBets,
} from '@/api/gaps';
import type {
  RunGapAnalysisRequest,
  GapAnalysisResponse,
  OpportunityListResponse,
  GapHealthResponse,
  ListOpportunitiesParams,
} from '@/types/gaps';

// =============================================================================
// QUERY HOOKS
// =============================================================================

/**
 * Hook to fetch a gap analysis by ID.
 *
 * @param analysisId - The unique analysis identifier
 * @param options - Additional query options
 * @returns Query result with gap analysis data
 *
 * @example
 * ```tsx
 * const { data, isLoading } = useGapAnalysis('gap_abc123');
 * if (data?.status === 'completed') {
 *   console.log(data.total_revenue_gap);
 * }
 * ```
 */
export function useGapAnalysis(
  analysisId: string,
  options?: Omit<UseQueryOptions<GapAnalysisResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<GapAnalysisResponse, ApiError>({
    queryKey: queryKeys.gaps.analysis(analysisId),
    queryFn: () => getGapAnalysis(analysisId),
    enabled: !!analysisId,
    ...options,
  });
}

/**
 * Hook to list gap opportunities.
 *
 * @param params - Optional filter parameters (minRoi, limit)
 * @param options - Additional query options
 * @returns Query result with opportunity list
 *
 * @example
 * ```tsx
 * const { data } = useOpportunities({ min_roi: 1.5, limit: 10 });
 * data?.opportunities.forEach(opp => {
 *   console.log(`${opp.territory}: ${opp.roi}x ROI`);
 * });
 * ```
 */
export function useOpportunities(
  params?: ListOpportunitiesParams,
  options?: Omit<UseQueryOptions<OpportunityListResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<OpportunityListResponse, ApiError>({
    queryKey: [...queryKeys.gaps.opportunities(), params?.min_roi, params?.limit],
    queryFn: () => listOpportunities(params),
    staleTime: 60 * 1000, // 1 minute
    ...options,
  });
}

/**
 * Hook to get gap analysis service health.
 *
 * @param options - Additional query options
 * @returns Query result with service health status
 *
 * @example
 * ```tsx
 * const { data: health } = useGapHealth();
 * if (health?.agent_available) {
 *   console.log('Gap analyzer is ready');
 * }
 * ```
 */
export function useGapHealth(
  options?: Omit<UseQueryOptions<GapHealthResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<GapHealthResponse, ApiError>({
    queryKey: queryKeys.gaps.health(),
    queryFn: () => getGapHealth(),
    staleTime: 30 * 1000, // 30 seconds
    ...options,
  });
}

/**
 * Hook to fetch quick win opportunities.
 *
 * @param brand - Brand to filter opportunities by
 * @param count - Number of quick wins to return (default: 5)
 * @param options - Additional query options
 * @returns Query result with quick win opportunities
 *
 * @example
 * ```tsx
 * const { data } = useQuickWins('kisqali', 10);
 * data?.opportunities.forEach(opp => console.log(opp));
 * ```
 */
export function useQuickWins(
  brand: string,
  count: number = 5,
  options?: Omit<UseQueryOptions<OpportunityListResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<OpportunityListResponse, ApiError>({
    queryKey: [...queryKeys.gaps.opportunities(), 'quick-wins', brand, count],
    queryFn: () => getQuickWins(brand, count),
    enabled: !!brand,
    staleTime: 2 * 60 * 1000, // 2 minutes
    ...options,
  });
}

/**
 * Hook to fetch strategic bet opportunities.
 *
 * @param brand - Brand to filter opportunities by
 * @param count - Number of strategic bets to return (default: 5)
 * @param options - Additional query options
 * @returns Query result with strategic bet opportunities
 *
 * @example
 * ```tsx
 * const { data } = useStrategicBets('kisqali', 5);
 * data?.opportunities.forEach(opp => console.log(opp));
 * ```
 */
export function useStrategicBets(
  brand: string,
  count: number = 5,
  options?: Omit<UseQueryOptions<OpportunityListResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<OpportunityListResponse, ApiError>({
    queryKey: [...queryKeys.gaps.opportunities(), 'strategic-bets', brand, count],
    queryFn: () => getStrategicBets(brand, count),
    enabled: !!brand,
    staleTime: 2 * 60 * 1000, // 2 minutes
    ...options,
  });
}

// =============================================================================
// MUTATION HOOKS
// =============================================================================

/**
 * Hook to run a gap analysis.
 *
 * @param options - Mutation options including onSuccess/onError callbacks
 * @returns Mutation object for triggering gap analysis
 *
 * @example
 * ```tsx
 * const { mutate: runAnalysis, isPending } = useRunGapAnalysis({
 *   onSuccess: (data) => {
 *     console.log(`Analysis ${data.analysis_id} started`);
 *   },
 * });
 *
 * runAnalysis({
 *   request: { query: 'Find gaps in Northeast territory' },
 *   asyncMode: true,
 * });
 * ```
 */
export function useRunGapAnalysis(
  options?: Omit<
    UseMutationOptions<
      GapAnalysisResponse,
      ApiError,
      { request: RunGapAnalysisRequest; asyncMode?: boolean }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    GapAnalysisResponse,
    ApiError,
    { request: RunGapAnalysisRequest; asyncMode?: boolean }
  >({
    mutationFn: ({ request, asyncMode = true }) => runGapAnalysis(request, asyncMode),
    onSuccess: (data) => {
      // Cache the result
      queryClient.setQueryData(queryKeys.gaps.analysis(data.analysis_id), data);
      // Invalidate opportunities list (new data may be available)
      queryClient.invalidateQueries({ queryKey: queryKeys.gaps.opportunities() });
    },
    ...options,
  });
}

/**
 * Hook to run a gap analysis and wait for completion.
 *
 * @param options - Mutation options
 * @returns Mutation object for running analysis with polling
 *
 * @example
 * ```tsx
 * const { mutate: runAndWait, isPending } = useRunGapAnalysisAndWait();
 *
 * runAndWait({
 *   request: { query: 'Find revenue gaps' },
 *   pollIntervalMs: 2000,
 *   maxWaitMs: 120000,
 * });
 * ```
 */
export function useRunGapAnalysisAndWait(
  options?: Omit<
    UseMutationOptions<
      GapAnalysisResponse,
      ApiError,
      { request: RunGapAnalysisRequest; pollIntervalMs?: number; maxWaitMs?: number }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    GapAnalysisResponse,
    ApiError,
    { request: RunGapAnalysisRequest; pollIntervalMs?: number; maxWaitMs?: number }
  >({
    mutationFn: ({ request, pollIntervalMs, maxWaitMs }) =>
      runGapAnalysisAndWait(request, pollIntervalMs, maxWaitMs),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.gaps.analysis(data.analysis_id), data);
      queryClient.invalidateQueries({ queryKey: queryKeys.gaps.opportunities() });
    },
    ...options,
  });
}

// =============================================================================
// POLLING HOOKS
// =============================================================================

/**
 * Hook to poll a gap analysis until completion.
 *
 * @param analysisId - The analysis ID to poll
 * @param options - Query options (use refetchInterval for polling)
 * @returns Query result that updates until completion
 *
 * @example
 * ```tsx
 * const { data, isLoading } = usePollGapAnalysis('gap_abc123', {
 *   refetchInterval: (query) =>
 *     query.state.data?.status === 'completed' ? false : 2000,
 * });
 * ```
 */
export function usePollGapAnalysis(
  analysisId: string,
  options?: Omit<UseQueryOptions<GapAnalysisResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<GapAnalysisResponse, ApiError>({
    queryKey: queryKeys.gaps.analysis(analysisId),
    queryFn: () => getGapAnalysis(analysisId),
    enabled: !!analysisId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      // Stop polling when completed or failed
      if (status === 'completed' || status === 'failed') {
        return false;
      }
      return 2000; // Poll every 2 seconds
    },
    ...options,
  });
}
