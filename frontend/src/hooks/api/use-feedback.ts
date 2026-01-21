/**
 * Feedback Learning React Query Hooks
 * ====================================
 *
 * TanStack Query hooks for the Feedback Learning API endpoints.
 * Provides typed query and mutation hooks for the Feedback Learner agent
 * (Tier 5 self-improvement system).
 *
 * Supports:
 * - Learning cycle execution
 * - Pattern detection and listing
 * - Knowledge updates management
 * - Service health monitoring
 *
 * @module hooks/api/use-feedback
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import { ApiError } from '@/lib/api-client';
import {
  runLearningCycle,
  getLearningResults,
  processFeedback,
  listPatterns,
  listUpdates,
  applyUpdate,
  rollbackUpdate,
  getFeedbackHealth,
  runLearningCycleAndWait,
  getFeedbackStatus,
  quickLearningCycle,
} from '@/api/feedback';
import type {
  RunLearningRequest,
  ProcessFeedbackRequest,
  ListPatternsParams,
  ListUpdatesParams,
  LearningResponse,
  PatternListResponse,
  UpdateListResponse,
  KnowledgeUpdate,
  FeedbackHealthResponse,
} from '@/types/feedback';

// =============================================================================
// QUERY HOOKS
// =============================================================================

/**
 * Hook to fetch learning results by batch ID.
 *
 * @param batchId - The learning batch identifier
 * @param options - Additional query options
 * @returns Query result with learning results
 *
 * @example
 * ```tsx
 * const { data: results } = useLearningResults('fb_abc123');
 * if (results?.status === 'completed') {
 *   console.log(`Found ${results.patterns_detected} patterns`);
 * }
 * ```
 */
export function useLearningResults(
  batchId: string,
  options?: Omit<UseQueryOptions<LearningResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<LearningResponse, ApiError>({
    queryKey: queryKeys.feedback.learning(batchId),
    queryFn: () => getLearningResults(batchId),
    enabled: !!batchId,
    ...options,
  });
}

/**
 * Hook to list detected patterns.
 *
 * @param params - Optional filter parameters (severity, type, agent)
 * @param options - Additional query options
 * @returns Query result with pattern list
 *
 * @example
 * ```tsx
 * const { data: patterns } = usePatterns({ severity: PatternSeverity.CRITICAL });
 * console.log(`${patterns?.critical_count} critical patterns`);
 * ```
 */
export function usePatterns(
  params?: ListPatternsParams,
  options?: Omit<UseQueryOptions<PatternListResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<PatternListResponse, ApiError>({
    queryKey: [...queryKeys.feedback.patterns(), params?.severity, params?.pattern_type, params?.agent, params?.limit],
    queryFn: () => listPatterns(params),
    staleTime: 60 * 1000, // 1 minute
    ...options,
  });
}

/**
 * Hook to list knowledge updates.
 *
 * @param params - Optional filter parameters (status, type, agent)
 * @param options - Additional query options
 * @returns Query result with update list
 *
 * @example
 * ```tsx
 * const { data: updates } = useKnowledgeUpdates({ status: UpdateStatus.PROPOSED });
 * console.log(`${updates?.proposed_count} pending approval`);
 * ```
 */
export function useKnowledgeUpdates(
  params?: ListUpdatesParams,
  options?: Omit<UseQueryOptions<UpdateListResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<UpdateListResponse, ApiError>({
    queryKey: [...queryKeys.feedback.updates(), params?.status, params?.update_type, params?.agent, params?.limit],
    queryFn: () => listUpdates(params),
    staleTime: 30 * 1000, // 30 seconds
    ...options,
  });
}

/**
 * Hook to get feedback learning service health.
 *
 * @param options - Additional query options
 * @returns Query result with service health status
 *
 * @example
 * ```tsx
 * const { data: health } = useFeedbackHealth();
 * if (health?.agent_available) {
 *   console.log(`${health.cycles_24h} cycles in last 24h`);
 * }
 * ```
 */
export function useFeedbackHealth(
  options?: Omit<UseQueryOptions<FeedbackHealthResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<FeedbackHealthResponse, ApiError>({
    queryKey: queryKeys.feedback.health(),
    queryFn: () => getFeedbackHealth(),
    staleTime: 30 * 1000, // 30 seconds
    ...options,
  });
}

// =============================================================================
// COMPOSITE HOOKS
// =============================================================================

/**
 * Combined feedback status type
 */
export interface FeedbackStatusData {
  health: FeedbackHealthResponse;
  patterns: PatternListResponse;
  updates: UpdateListResponse;
}

/**
 * Hook to get comprehensive feedback status.
 *
 * @param options - Additional query options
 * @returns Query result with health, patterns, and updates
 *
 * @example
 * ```tsx
 * const { data: status } = useFeedbackStatus();
 * console.log(`Health: ${status?.health.status}`);
 * console.log(`Critical: ${status?.patterns.critical_count}`);
 * console.log(`Pending: ${status?.updates.proposed_count}`);
 * ```
 */
export function useFeedbackStatus(
  options?: Omit<UseQueryOptions<FeedbackStatusData, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<FeedbackStatusData, ApiError>({
    queryKey: [...queryKeys.feedback.all(), 'status'],
    queryFn: () => getFeedbackStatus(),
    staleTime: 30 * 1000, // 30 seconds
    ...options,
  });
}

/**
 * Dashboard data for feedback learning
 */
export interface FeedbackDashboardData {
  health: FeedbackHealthResponse;
  patterns: PatternListResponse;
  updates: UpdateListResponse;
  criticalPatterns: number;
  pendingUpdates: number;
}

/**
 * Hook for feedback dashboard display.
 *
 * @param options - Additional query options
 * @returns Query result for dashboard
 *
 * @example
 * ```tsx
 * const { data: dashboard } = useFeedbackDashboard();
 * console.log(`${dashboard?.criticalPatterns} critical, ${dashboard?.pendingUpdates} pending`);
 * ```
 */
export function useFeedbackDashboard(
  options?: Omit<UseQueryOptions<FeedbackDashboardData, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<FeedbackDashboardData, ApiError>({
    queryKey: [...queryKeys.feedback.all(), 'dashboard'],
    queryFn: async () => {
      const status = await getFeedbackStatus();
      return {
        ...status,
        criticalPatterns: status.patterns.critical_count,
        pendingUpdates: status.updates.proposed_count,
      };
    },
    staleTime: 30 * 1000, // 30 seconds
    ...options,
  });
}

// =============================================================================
// MUTATION HOOKS
// =============================================================================

/**
 * Hook to run a feedback learning cycle.
 *
 * @param options - Mutation options including onSuccess/onError callbacks
 * @returns Mutation object for triggering learning cycle
 *
 * @example
 * ```tsx
 * const { mutate: runLearning, isPending } = useRunLearningCycle({
 *   onSuccess: (data) => {
 *     console.log(`Learning cycle ${data.batch_id} started`);
 *   },
 * });
 *
 * runLearning({
 *   request: { focus_agents: ['causal_impact'], min_feedback_count: 10 },
 *   asyncMode: true,
 * });
 * ```
 */
export function useRunLearningCycle(
  options?: Omit<
    UseMutationOptions<
      LearningResponse,
      ApiError,
      { request: RunLearningRequest; asyncMode?: boolean }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    LearningResponse,
    ApiError,
    { request: RunLearningRequest; asyncMode?: boolean }
  >({
    mutationFn: ({ request, asyncMode = true }) => runLearningCycle(request, asyncMode),
    onSuccess: (data) => {
      // Cache the result
      queryClient.setQueryData(queryKeys.feedback.learning(data.batch_id), data);
      // Invalidate patterns and updates (new data may be available)
      queryClient.invalidateQueries({ queryKey: queryKeys.feedback.patterns() });
      queryClient.invalidateQueries({ queryKey: queryKeys.feedback.updates() });
    },
    ...options,
  });
}

/**
 * Hook to run a learning cycle and wait for completion.
 *
 * @param options - Mutation options
 * @returns Mutation object with polling
 *
 * @example
 * ```tsx
 * const { mutate: runAndWait, isPending } = useRunLearningCycleAndWait();
 *
 * runAndWait({
 *   request: { focus_agents: ['gap_analyzer'] },
 *   pollIntervalMs: 2000,
 *   maxWaitMs: 120000,
 * });
 * ```
 */
export function useRunLearningCycleAndWait(
  options?: Omit<
    UseMutationOptions<
      LearningResponse,
      ApiError,
      { request: RunLearningRequest; pollIntervalMs?: number; maxWaitMs?: number }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    LearningResponse,
    ApiError,
    { request: RunLearningRequest; pollIntervalMs?: number; maxWaitMs?: number }
  >({
    mutationFn: ({ request, pollIntervalMs, maxWaitMs }) =>
      runLearningCycleAndWait(request, pollIntervalMs, maxWaitMs),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.feedback.learning(data.batch_id), data);
      queryClient.invalidateQueries({ queryKey: queryKeys.feedback.patterns() });
      queryClient.invalidateQueries({ queryKey: queryKeys.feedback.updates() });
    },
    ...options,
  });
}

/**
 * Hook to run a quick learning cycle on recent feedback.
 *
 * @param options - Mutation options
 * @returns Mutation object for quick analysis
 *
 * @example
 * ```tsx
 * const { mutate: quickAnalysis } = useQuickLearningCycle();
 * quickAnalysis(['causal_impact', 'gap_analyzer']);
 * ```
 */
export function useQuickLearningCycle(
  options?: Omit<UseMutationOptions<LearningResponse, ApiError, string[] | undefined>, 'mutationFn'>
) {
  const queryClient = useQueryClient();

  return useMutation<LearningResponse, ApiError, string[] | undefined>({
    mutationFn: (focusAgents) => quickLearningCycle(focusAgents),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.feedback.learning(data.batch_id), data);
      queryClient.invalidateQueries({ queryKey: queryKeys.feedback.patterns() });
      queryClient.invalidateQueries({ queryKey: queryKeys.feedback.updates() });
    },
    ...options,
  });
}

/**
 * Hook to process specific feedback items.
 *
 * @param options - Mutation options
 * @returns Mutation object for feedback processing
 *
 * @example
 * ```tsx
 * const { mutate: process } = useProcessFeedback();
 * process({
 *   items: [{ feedback_type: FeedbackType.RATING, ... }],
 *   detect_patterns: true,
 * });
 * ```
 */
export function useProcessFeedback(
  options?: Omit<UseMutationOptions<LearningResponse, ApiError, ProcessFeedbackRequest>, 'mutationFn'>
) {
  const queryClient = useQueryClient();

  return useMutation<LearningResponse, ApiError, ProcessFeedbackRequest>({
    mutationFn: (request) => processFeedback(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.feedback.patterns() });
    },
    ...options,
  });
}

/**
 * Hook to apply a knowledge update.
 *
 * @param options - Mutation options
 * @returns Mutation object for applying updates
 *
 * @example
 * ```tsx
 * const { mutate: apply } = useApplyUpdate({
 *   onSuccess: (data) => {
 *     console.log(`Update ${data.update_id} applied`);
 *   },
 * });
 * apply({ updateId: 'upd_abc123' });
 * ```
 */
export function useApplyUpdate(
  options?: Omit<
    UseMutationOptions<KnowledgeUpdate, ApiError, { updateId: string; force?: boolean }>,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<KnowledgeUpdate, ApiError, { updateId: string; force?: boolean }>({
    mutationFn: ({ updateId, force }) => applyUpdate(updateId, force),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.feedback.updates() });
      queryClient.invalidateQueries({ queryKey: queryKeys.feedback.health() });
    },
    ...options,
  });
}

/**
 * Hook to rollback a knowledge update.
 *
 * @param options - Mutation options
 * @returns Mutation object for rolling back updates
 *
 * @example
 * ```tsx
 * const { mutate: rollback } = useRollbackUpdate({
 *   onSuccess: (data) => {
 *     console.log(`Update ${data.update_id} rolled back`);
 *   },
 * });
 * rollback('upd_abc123');
 * ```
 */
export function useRollbackUpdate(
  options?: Omit<UseMutationOptions<KnowledgeUpdate, ApiError, string>, 'mutationFn'>
) {
  const queryClient = useQueryClient();

  return useMutation<KnowledgeUpdate, ApiError, string>({
    mutationFn: (updateId) => rollbackUpdate(updateId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.feedback.updates() });
      queryClient.invalidateQueries({ queryKey: queryKeys.feedback.health() });
    },
    ...options,
  });
}

// =============================================================================
// POLLING HOOKS
// =============================================================================

/**
 * Hook to poll learning results until completion.
 *
 * @param batchId - The batch ID to poll
 * @param options - Query options (use refetchInterval for polling)
 * @returns Query result that updates until completion
 *
 * @example
 * ```tsx
 * const { data } = usePollLearningResults('fb_abc123', {
 *   refetchInterval: (query) =>
 *     query.state.data?.status === 'completed' ? false : 2000,
 * });
 * ```
 */
export function usePollLearningResults(
  batchId: string,
  options?: Omit<UseQueryOptions<LearningResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<LearningResponse, ApiError>({
    queryKey: queryKeys.feedback.learning(batchId),
    queryFn: () => getLearningResults(batchId),
    enabled: !!batchId,
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
