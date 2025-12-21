/**
 * Memory API Query Hooks
 * ======================
 *
 * TanStack Query hooks for the E2I Memory System API.
 * Provides type-safe data fetching, caching, and state management
 * for memory operations.
 *
 * Features:
 * - Automatic caching and background refetching
 * - Optimistic updates for mutations
 * - Loading and error states
 * - Query key management via queryKeys
 *
 * @module hooks/api/use-memory
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import {
  searchMemory,
  createEpisodicMemory,
  getEpisodicMemories,
  getEpisodicMemory,
  recordProceduralFeedback,
  querySemanticPaths,
  getMemoryStats,
} from '@/api/memory';
import type {
  EpisodicMemoryInput,
  EpisodicMemoryResponse,
  MemorySearchRequest,
  MemorySearchResponse,
  MemoryStatsResponse,
  ProceduralFeedbackRequest,
  ProceduralFeedbackResponse,
  SemanticPathRequest,
  SemanticPathResponse,
} from '@/types/memory';
import type { ApiError } from '@/lib/api-client';

// =============================================================================
// QUERY HOOKS
// =============================================================================

/**
 * Hook to fetch memory system statistics.
 *
 * @param options - Additional TanStack Query options
 * @returns Query result with memory statistics
 *
 * @example
 * ```tsx
 * const { data: stats } = useMemoryStats();
 * console.log(`Total episodic memories: ${stats?.episodic.total_memories}`);
 * ```
 */
export function useMemoryStats(
  options?: Omit<
    UseQueryOptions<MemoryStatsResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<MemoryStatsResponse, ApiError>({
    queryKey: queryKeys.memory.stats(),
    queryFn: getMemoryStats,
    // Stats can be slightly stale
    staleTime: 10 * 60 * 1000, // 10 minutes
    ...options,
  });
}

/**
 * Hook to fetch episodic memories with optional filters.
 *
 * @param params - Query parameters for filtering episodic memories
 * @param options - Additional TanStack Query options
 * @returns Query result with episodic memories list
 *
 * @example
 * ```tsx
 * const { data: memories } = useEpisodicMemories({ session_id: 'sess_123' });
 * ```
 */
export function useEpisodicMemories(
  params?: Record<string, unknown>,
  options?: Omit<
    UseQueryOptions<EpisodicMemoryResponse[], ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<EpisodicMemoryResponse[], ApiError>({
    queryKey: [...queryKeys.memory.episodic(), params],
    queryFn: () => getEpisodicMemories(params),
    ...options,
  });
}

/**
 * Hook to fetch a single episodic memory by ID.
 *
 * @param memoryId - The unique memory identifier
 * @param options - Additional TanStack Query options
 * @returns Query result with episodic memory data
 *
 * @example
 * ```tsx
 * const { data: memory, isLoading } = useEpisodicMemory('mem_12345');
 * ```
 */
export function useEpisodicMemory(
  memoryId: string,
  options?: Omit<
    UseQueryOptions<EpisodicMemoryResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<EpisodicMemoryResponse, ApiError>({
    queryKey: queryKeys.memory.episodicMemory(memoryId),
    queryFn: () => getEpisodicMemory(memoryId),
    enabled: !!memoryId,
    ...options,
  });
}

// =============================================================================
// MUTATION HOOKS
// =============================================================================

/**
 * Hook for hybrid memory search.
 *
 * Uses mutation pattern since search queries can be expensive
 * and we want explicit control over when they run.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: search, data: results, isPending } = useMemorySearch();
 *
 * search({
 *   query: 'best practices for HCP engagement',
 *   k: 10,
 *   retrieval_method: RetrievalMethod.HYBRID
 * });
 * ```
 */
export function useMemorySearch(
  options?: Omit<
    UseMutationOptions<MemorySearchResponse, ApiError, MemorySearchRequest>,
    'mutationFn'
  >
) {
  return useMutation<MemorySearchResponse, ApiError, MemorySearchRequest>({
    mutationFn: searchMemory,
    ...options,
  });
}

/**
 * Hook for creating episodic memory entries.
 *
 * Automatically invalidates episodic memory queries after success.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: createMemory, isPending } = useCreateEpisodicMemory();
 *
 * createMemory({
 *   content: 'Dr. Smith showed interest in Kisqali.',
 *   event_type: 'hcp_interaction',
 *   brand: 'Kisqali'
 * });
 * ```
 */
export function useCreateEpisodicMemory(
  options?: Omit<
    UseMutationOptions<
      EpisodicMemoryResponse,
      ApiError,
      EpisodicMemoryInput,
      unknown
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    EpisodicMemoryResponse,
    ApiError,
    EpisodicMemoryInput,
    unknown
  >({
    mutationFn: createEpisodicMemory,
    onSuccess: (...args) => {
      // Invalidate episodic memory queries to refetch with new data
      void queryClient.invalidateQueries({
        queryKey: queryKeys.memory.episodic(),
      });
      void queryClient.invalidateQueries({
        queryKey: queryKeys.memory.stats(),
      });

      // Call user's onSuccess if provided
      options?.onSuccess?.(...args);
    },
    ...options,
  });
}

/**
 * Hook for recording procedural feedback.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: recordFeedback, isPending } = useProceduralFeedback();
 *
 * recordFeedback({
 *   procedure_id: 'proc_hcp_outreach',
 *   outcome: 'success',
 *   score: 0.95
 * });
 * ```
 */
export function useProceduralFeedback(
  options?: Omit<
    UseMutationOptions<
      ProceduralFeedbackResponse,
      ApiError,
      ProceduralFeedbackRequest
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    ProceduralFeedbackResponse,
    ApiError,
    ProceduralFeedbackRequest
  >({
    mutationFn: recordProceduralFeedback,
    onSuccess: (...args) => {
      // Invalidate memory stats
      void queryClient.invalidateQueries({
        queryKey: queryKeys.memory.stats(),
      });

      options?.onSuccess?.(...args);
    },
    ...options,
  });
}

/**
 * Hook for querying semantic paths.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: queryPaths, data: paths } = useSemanticPaths();
 *
 * queryPaths({
 *   kpi_name: 'TRx',
 *   max_depth: 3,
 *   min_confidence: 0.6
 * });
 * ```
 */
export function useSemanticPaths(
  options?: Omit<
    UseMutationOptions<SemanticPathResponse, ApiError, SemanticPathRequest>,
    'mutationFn'
  >
) {
  return useMutation<SemanticPathResponse, ApiError, SemanticPathRequest>({
    mutationFn: querySemanticPaths,
    ...options,
  });
}

// =============================================================================
// PREFETCH HELPERS
// =============================================================================

/**
 * Prefetch memory statistics for faster navigation.
 *
 * @example
 * ```tsx
 * const queryClient = useQueryClient();
 * prefetchMemoryStats(queryClient);
 * ```
 */
export async function prefetchMemoryStats(
  queryClient: ReturnType<typeof useQueryClient>
) {
  await queryClient.prefetchQuery({
    queryKey: queryKeys.memory.stats(),
    queryFn: getMemoryStats,
  });
}

/**
 * Prefetch episodic memories for faster navigation.
 *
 * @param params - Optional filter parameters
 *
 * @example
 * ```tsx
 * prefetchEpisodicMemories(queryClient, { limit: 20 });
 * ```
 */
export async function prefetchEpisodicMemories(
  queryClient: ReturnType<typeof useQueryClient>,
  params?: Record<string, unknown>
) {
  await queryClient.prefetchQuery({
    queryKey: [...queryKeys.memory.episodic(), params],
    queryFn: () => getEpisodicMemories(params),
  });
}
