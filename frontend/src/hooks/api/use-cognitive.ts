/**
 * Cognitive API Query Hooks
 * =========================
 *
 * TanStack Query hooks for the E2I Cognitive Workflow API.
 * Provides type-safe data fetching, caching, and state management
 * for cognitive operations.
 *
 * Features:
 * - Automatic caching and background refetching
 * - Optimistic updates for mutations
 * - Loading and error states
 * - Query key management via queryKeys
 *
 * @module hooks/api/use-cognitive
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import {
  processCognitiveQuery,
  getCognitiveStatus,
  createSession,
  getSession,
  deleteSession,
  listSessions,
  cognitiveRAGSearch,
} from '@/api/cognitive';
import type {
  CognitiveQueryRequest,
  CognitiveQueryResponse,
  CognitiveRAGRequest,
  CognitiveRAGResponse,
  CreateSessionRequest,
  CreateSessionResponse,
  DeleteSessionResponse,
  SessionResponse,
} from '@/types/cognitive';
import type { ApiError } from '@/lib/api-client';

// =============================================================================
// QUERY HOOKS
// =============================================================================

/**
 * Hook to check cognitive service status.
 *
 * @param options - Additional TanStack Query options
 * @returns Query result with service status
 *
 * @example
 * ```tsx
 * const { data: status } = useCognitiveStatus();
 * console.log(`Agents: ${status?.agents.join(', ')}`);
 * ```
 */
export function useCognitiveStatus(
  options?: Omit<
    UseQueryOptions<
      { status: string; version: string; agents: string[] },
      ApiError
    >,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<{ status: string; version: string; agents: string[] }, ApiError>({
    queryKey: queryKeys.cognitive.status(),
    queryFn: getCognitiveStatus,
    // Status checks should be fresh
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 60 * 1000, // Refetch every minute
    ...options,
  });
}

/**
 * Hook to fetch all active cognitive sessions.
 *
 * @param params - Optional query parameters for filtering
 * @param options - Additional TanStack Query options
 * @returns Query result with session list
 *
 * @example
 * ```tsx
 * const { data } = useSessions({ user_id: 'user_abc' });
 * console.log(`Active sessions: ${data?.total}`);
 * ```
 */
export function useSessions(
  params?: Record<string, unknown>,
  options?: Omit<
    UseQueryOptions<{ sessions: SessionResponse[]; total: number }, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<{ sessions: SessionResponse[]; total: number }, ApiError>({
    queryKey: [...queryKeys.cognitive.sessions(), params],
    queryFn: () => listSessions(params),
    ...options,
  });
}

/**
 * Hook to fetch a single cognitive session by ID.
 *
 * @param sessionId - The session identifier
 * @param options - Additional TanStack Query options
 * @returns Query result with session data
 *
 * @example
 * ```tsx
 * const { data: session, isLoading } = useSession('sess_abc123');
 * ```
 */
export function useSession(
  sessionId: string,
  options?: Omit<
    UseQueryOptions<SessionResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<SessionResponse, ApiError>({
    queryKey: queryKeys.cognitive.session(sessionId),
    queryFn: () => getSession(sessionId),
    enabled: !!sessionId,
    ...options,
  });
}

// =============================================================================
// MUTATION HOOKS
// =============================================================================

/**
 * Hook for processing cognitive queries.
 *
 * Routes queries through the appropriate agents based on detected intent.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: process, data: response, isPending } = useCognitiveQuery();
 *
 * process({
 *   query: 'What factors are driving TRx decline?',
 *   brand: 'Kisqali',
 *   include_evidence: true
 * });
 * ```
 */
export function useCognitiveQuery(
  options?: Omit<
    UseMutationOptions<
      CognitiveQueryResponse,
      ApiError,
      CognitiveQueryRequest,
      unknown
    >,
    'mutationFn' | 'onSuccess'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    CognitiveQueryResponse,
    ApiError,
    CognitiveQueryRequest,
    unknown
  >({
    mutationFn: processCognitiveQuery,
    onSuccess: (_data, variables) => {
      // If we used a session, invalidate it to get fresh state
      if (variables.session_id) {
        void queryClient.invalidateQueries({
          queryKey: queryKeys.cognitive.session(variables.session_id),
        });
      }
    },
    ...options,
  });
}

/**
 * Hook for creating new cognitive sessions.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: create, data: session } = useCreateSession();
 *
 * create({
 *   user_id: 'user_abc',
 *   brand: 'Kisqali',
 *   region: 'northeast'
 * });
 * ```
 */
export function useCreateSession(
  options?: Omit<
    UseMutationOptions<
      CreateSessionResponse,
      ApiError,
      CreateSessionRequest,
      unknown
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    CreateSessionResponse,
    ApiError,
    CreateSessionRequest,
    unknown
  >({
    mutationFn: createSession,
    onSuccess: (...args) => {
      // Invalidate sessions list
      void queryClient.invalidateQueries({
        queryKey: queryKeys.cognitive.sessions(),
      });

      options?.onSuccess?.(...args);
    },
    ...options,
  });
}

/**
 * Hook for deleting cognitive sessions.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: remove } = useDeleteSession();
 * remove('sess_abc123');
 * ```
 */
export function useDeleteSession(
  options?: Omit<
    UseMutationOptions<DeleteSessionResponse, ApiError, string, unknown>,
    'mutationFn' | 'onSuccess'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<DeleteSessionResponse, ApiError, string, unknown>({
    mutationFn: deleteSession,
    onSuccess: (_data, sessionId) => {
      // Invalidate sessions list and remove the deleted session from cache
      void queryClient.invalidateQueries({
        queryKey: queryKeys.cognitive.sessions(),
      });
      queryClient.removeQueries({
        queryKey: queryKeys.cognitive.session(sessionId),
      });
    },
    ...options,
  });
}

/**
 * Hook for DSPy-enhanced cognitive RAG search.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: search, data: result } = useCognitiveRAG();
 *
 * search({
 *   query: 'What is driving TRx trend for Kisqali?',
 *   conversation_id: 'conv_123'
 * });
 * ```
 */
export function useCognitiveRAG(
  options?: Omit<
    UseMutationOptions<CognitiveRAGResponse, ApiError, CognitiveRAGRequest>,
    'mutationFn'
  >
) {
  return useMutation<CognitiveRAGResponse, ApiError, CognitiveRAGRequest>({
    mutationFn: cognitiveRAGSearch,
    ...options,
  });
}

// =============================================================================
// PREFETCH HELPERS
// =============================================================================

/**
 * Prefetch cognitive service status.
 *
 * @example
 * ```tsx
 * const queryClient = useQueryClient();
 * prefetchCognitiveStatus(queryClient);
 * ```
 */
export async function prefetchCognitiveStatus(
  queryClient: ReturnType<typeof useQueryClient>
) {
  await queryClient.prefetchQuery({
    queryKey: queryKeys.cognitive.status(),
    queryFn: getCognitiveStatus,
  });
}

/**
 * Prefetch a specific session for faster navigation.
 *
 * @param sessionId - The session ID to prefetch
 *
 * @example
 * ```tsx
 * prefetchSession(queryClient, 'sess_abc123');
 * ```
 */
export async function prefetchSession(
  queryClient: ReturnType<typeof useQueryClient>,
  sessionId: string
) {
  await queryClient.prefetchQuery({
    queryKey: queryKeys.cognitive.session(sessionId),
    queryFn: () => getSession(sessionId),
  });
}
