/**
 * RAG API Query Hooks
 * ===================
 *
 * TanStack Query hooks for the E2I Hybrid RAG API.
 * Provides type-safe data fetching, caching, and state management
 * for RAG operations.
 *
 * Features:
 * - Automatic caching and background refetching
 * - Loading and error states
 * - Query key management via queryKeys
 *
 * @module hooks/api/use-rag
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import {
  searchRAG,
  queryRAG,
  extractEntities,
  getCausalSubgraph,
  getCausalPaths,
  getRAGStats,
  getRAGHealth,
} from '@/api/rag';
import type {
  CausalPathResponse,
  CausalSubgraphResponse,
  ExtractedEntities,
  RAGHealthResponse,
  RAGSearchRequest,
  RAGSearchResponse,
  RAGStatsResponse,
} from '@/types/rag';
import type { ApiError } from '@/lib/api-client';

// =============================================================================
// QUERY HOOKS
// =============================================================================

/**
 * Hook to fetch RAG usage statistics.
 *
 * @param periodHours - Time period in hours (default 24)
 * @param options - Additional TanStack Query options
 * @returns Query result with RAG statistics
 *
 * @example
 * ```tsx
 * const { data: stats } = useRAGStats(48);
 * console.log(`Total searches: ${stats?.total_searches}`);
 * ```
 */
export function useRAGStats(
  periodHours?: number,
  options?: Omit<
    UseQueryOptions<RAGStatsResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<RAGStatsResponse, ApiError>({
    queryKey: [...queryKeys.rag.stats(), periodHours],
    queryFn: () => getRAGStats(periodHours),
    // Stats can be slightly stale
    staleTime: 10 * 60 * 1000, // 10 minutes
    ...options,
  });
}

/**
 * Hook to check RAG service health.
 *
 * @param options - Additional TanStack Query options
 * @returns Query result with health status
 *
 * @example
 * ```tsx
 * const { data: health } = useRAGHealth();
 * const isHealthy = health?.status === 'healthy';
 * ```
 */
export function useRAGHealth(
  options?: Omit<
    UseQueryOptions<RAGHealthResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<RAGHealthResponse, ApiError>({
    queryKey: queryKeys.rag.health(),
    queryFn: getRAGHealth,
    // Health checks should be fresh
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 60 * 1000, // Refetch every minute
    ...options,
  });
}

/**
 * Hook to fetch causal subgraph for an entity.
 *
 * @param entity - Center entity identifier
 * @param depth - Traversal depth (default 2)
 * @param options - Additional TanStack Query options
 * @returns Query result with subgraph data
 *
 * @example
 * ```tsx
 * const { data: subgraph } = useCausalSubgraph('kisqali', 3);
 * console.log(`Nodes: ${subgraph?.node_count}`);
 * ```
 */
export function useCausalSubgraph(
  entity: string,
  depth?: number,
  options?: Omit<
    UseQueryOptions<CausalSubgraphResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<CausalSubgraphResponse, ApiError>({
    queryKey: [...queryKeys.rag.subgraph(entity), depth],
    queryFn: () => getCausalSubgraph(entity, depth),
    enabled: !!entity,
    ...options,
  });
}

/**
 * Hook to fetch causal paths between entities.
 *
 * @param source - Source entity identifier
 * @param target - Target entity identifier
 * @param maxDepth - Maximum path depth (default 4)
 * @param options - Additional TanStack Query options
 * @returns Query result with paths data
 *
 * @example
 * ```tsx
 * const { data: paths } = useCausalPaths('hcp_engagement', 'trx', 3);
 * console.log(`Shortest path: ${paths?.shortest_path_length} hops`);
 * ```
 */
export function useCausalPaths(
  source: string,
  target: string,
  maxDepth?: number,
  options?: Omit<
    UseQueryOptions<CausalPathResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<CausalPathResponse, ApiError>({
    queryKey: [...queryKeys.rag.paths(source, target), maxDepth],
    queryFn: () => getCausalPaths(source, target, maxDepth),
    enabled: !!source && !!target,
    ...options,
  });
}

// =============================================================================
// MUTATION HOOKS
// =============================================================================

/**
 * Hook for hybrid RAG search.
 *
 * Uses mutation pattern since search queries can be expensive
 * and we want explicit control over when they run.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: search, data: results, isPending } = useRAGSearch();
 *
 * search({
 *   query: 'What are the key drivers of TRx?',
 *   mode: SearchMode.HYBRID,
 *   top_k: 10
 * });
 * ```
 */
export function useRAGSearch(
  options?: Omit<
    UseMutationOptions<RAGSearchResponse, ApiError, RAGSearchRequest>,
    'mutationFn'
  >
) {
  return useMutation<RAGSearchResponse, ApiError, RAGSearchRequest>({
    mutationFn: searchRAG,
    ...options,
  });
}

/**
 * Hook for simple RAG queries.
 *
 * Simplified interface for quick searches with just a query string.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: query, data: results } = useRAGQuery();
 *
 * query({
 *   query: 'Kisqali efficacy data',
 *   params: { top_k: 5 }
 * });
 * ```
 */
export function useRAGQuery(
  options?: Omit<
    UseMutationOptions<
      RAGSearchResponse,
      ApiError,
      { query: string; params?: Partial<Omit<RAGSearchRequest, 'query'>> }
    >,
    'mutationFn'
  >
) {
  return useMutation<
    RAGSearchResponse,
    ApiError,
    { query: string; params?: Partial<Omit<RAGSearchRequest, 'query'>> }
  >({
    mutationFn: ({ query, params }) => queryRAG(query, params),
    ...options,
  });
}

/**
 * Hook for extracting entities from queries.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: extract, data: entities } = useExtractEntities();
 *
 * extract({ query: 'TRx trend for Kisqali in Northeast' });
 * ```
 */
export function useExtractEntities(
  options?: Omit<
    UseMutationOptions<ExtractedEntities, ApiError, { query: string }>,
    'mutationFn'
  >
) {
  return useMutation<ExtractedEntities, ApiError, { query: string }>({
    mutationFn: extractEntities,
    ...options,
  });
}

// =============================================================================
// PREFETCH HELPERS
// =============================================================================

/**
 * Prefetch RAG statistics.
 *
 * @param periodHours - Optional time period in hours
 *
 * @example
 * ```tsx
 * const queryClient = useQueryClient();
 * prefetchRAGStats(queryClient, 48);
 * ```
 */
export async function prefetchRAGStats(
  queryClient: ReturnType<typeof useQueryClient>,
  periodHours?: number
) {
  await queryClient.prefetchQuery({
    queryKey: [...queryKeys.rag.stats(), periodHours],
    queryFn: () => getRAGStats(periodHours),
  });
}

/**
 * Prefetch causal subgraph for an entity.
 *
 * @param entity - Entity identifier
 * @param depth - Traversal depth
 *
 * @example
 * ```tsx
 * prefetchCausalSubgraph(queryClient, 'kisqali', 3);
 * ```
 */
export async function prefetchCausalSubgraph(
  queryClient: ReturnType<typeof useQueryClient>,
  entity: string,
  depth?: number
) {
  await queryClient.prefetchQuery({
    queryKey: [...queryKeys.rag.subgraph(entity), depth],
    queryFn: () => getCausalSubgraph(entity, depth),
  });
}

/**
 * Prefetch causal paths between entities.
 *
 * @param source - Source entity
 * @param target - Target entity
 * @param maxDepth - Maximum path depth
 *
 * @example
 * ```tsx
 * prefetchCausalPaths(queryClient, 'hcp_engagement', 'trx', 3);
 * ```
 */
export async function prefetchCausalPaths(
  queryClient: ReturnType<typeof useQueryClient>,
  source: string,
  target: string,
  maxDepth?: number
) {
  await queryClient.prefetchQuery({
    queryKey: [...queryKeys.rag.paths(source, target), maxDepth],
    queryFn: () => getCausalPaths(source, target, maxDepth),
  });
}
