/**
 * Graph API Query Hooks
 * =====================
 *
 * TanStack Query hooks for the E2I Knowledge Graph API.
 * Provides type-safe data fetching, caching, and state management
 * for graph operations.
 *
 * Features:
 * - Automatic caching and background refetching
 * - Optimistic updates for mutations
 * - Loading and error states
 * - Query key management via queryKeys
 *
 * @module hooks/api/use-graph
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import {
  listNodes,
  getNode,
  getNodeNetwork,
  listRelationships,
  traverseGraph,
  queryCausalChains,
  executeCypherQuery,
  addEpisode,
  searchGraph,
  getGraphStats,
  getGraphHealth,
} from '@/api/graph';
import type {
  AddEpisodeRequest,
  AddEpisodeResponse,
  CausalChainRequest,
  CausalChainResponse,
  CypherQueryRequest,
  CypherQueryResponse,
  GraphHealthResponse,
  GraphNode,
  GraphStatsResponse,
  ListNodesParams,
  ListNodesResponse,
  ListRelationshipsParams,
  ListRelationshipsResponse,
  NodeNetworkResponse,
  SearchGraphRequest,
  SearchGraphResponse,
  TraverseRequest,
  TraverseResponse,
} from '@/types/graph';
import type { ApiError } from '@/lib/api-client';

// =============================================================================
// QUERY HOOKS
// =============================================================================

/**
 * Hook to fetch a paginated list of graph nodes.
 *
 * @param params - Query parameters for filtering nodes
 * @param options - Additional TanStack Query options
 * @returns Query result with nodes data, loading, and error states
 *
 * @example
 * ```tsx
 * const { data, isLoading, error } = useNodes({
 *   entity_types: 'HCP,Brand',
 *   limit: 20
 * });
 * ```
 */
export function useNodes(
  params?: ListNodesParams,
  options?: Omit<
    UseQueryOptions<ListNodesResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<ListNodesResponse, ApiError>({
    queryKey: [...queryKeys.graph.nodes(), params],
    queryFn: () => listNodes(params),
    ...options,
  });
}

/**
 * Hook to fetch a single graph node by ID.
 *
 * @param nodeId - The unique node identifier
 * @param options - Additional TanStack Query options
 * @returns Query result with node data
 *
 * @example
 * ```tsx
 * const { data: node, isLoading } = useNode('hcp_12345');
 * ```
 */
export function useNode(
  nodeId: string,
  options?: Omit<UseQueryOptions<GraphNode, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<GraphNode, ApiError>({
    queryKey: queryKeys.graph.node(nodeId),
    queryFn: () => getNode(nodeId),
    enabled: !!nodeId,
    ...options,
  });
}

/**
 * Hook to fetch the relationship network around a node.
 *
 * @param nodeId - The central node identifier
 * @param maxDepth - Maximum traversal depth (1-5, default 2)
 * @param options - Additional TanStack Query options
 * @returns Query result with connected nodes grouped by type
 *
 * @example
 * ```tsx
 * const { data: network } = useNodeNetwork('patient_001', 3);
 * ```
 */
export function useNodeNetwork(
  nodeId: string,
  maxDepth?: number,
  options?: Omit<
    UseQueryOptions<NodeNetworkResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<NodeNetworkResponse, ApiError>({
    queryKey: [...queryKeys.graph.nodeNetwork(nodeId), maxDepth],
    queryFn: () => getNodeNetwork(nodeId, maxDepth),
    enabled: !!nodeId,
    ...options,
  });
}

/**
 * Hook to fetch a paginated list of graph relationships.
 *
 * @param params - Query parameters for filtering relationships
 * @param options - Additional TanStack Query options
 * @returns Query result with relationships data
 *
 * @example
 * ```tsx
 * const { data } = useRelationships({
 *   relationship_types: 'CAUSES,IMPACTS',
 *   min_confidence: 0.7
 * });
 * ```
 */
export function useRelationships(
  params?: ListRelationshipsParams,
  options?: Omit<
    UseQueryOptions<ListRelationshipsResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<ListRelationshipsResponse, ApiError>({
    queryKey: [...queryKeys.graph.relationships(), params],
    queryFn: () => listRelationships(params),
    ...options,
  });
}

/**
 * Hook to fetch graph statistics.
 *
 * @param options - Additional TanStack Query options
 * @returns Query result with graph statistics
 *
 * @example
 * ```tsx
 * const { data: stats } = useGraphStats();
 * console.log(`Total nodes: ${stats?.total_nodes}`);
 * ```
 */
export function useGraphStats(
  options?: Omit<
    UseQueryOptions<GraphStatsResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<GraphStatsResponse, ApiError>({
    queryKey: queryKeys.graph.stats(),
    queryFn: getGraphStats,
    // Stats can be slightly stale, so we use a longer stale time
    staleTime: 10 * 60 * 1000, // 10 minutes
    ...options,
  });
}

/**
 * Hook to check graph service health.
 *
 * @param options - Additional TanStack Query options
 * @returns Query result with health status
 *
 * @example
 * ```tsx
 * const { data: health } = useGraphHealth();
 * const isHealthy = health?.status === 'healthy';
 * ```
 */
export function useGraphHealth(
  options?: Omit<
    UseQueryOptions<GraphHealthResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<GraphHealthResponse, ApiError>({
    queryKey: [...queryKeys.graph.all(), 'health'],
    queryFn: getGraphHealth,
    // Health checks should be fresh
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 60 * 1000, // Refetch every minute
    ...options,
  });
}

/**
 * Hook to search the graph using natural language.
 *
 * @param query - The search query text
 * @param request - Additional search parameters
 * @param options - Additional TanStack Query options
 * @returns Query result with search results
 *
 * @example
 * ```tsx
 * const { data: results } = useGraphSearch('What factors impact TRx?', {
 *   entity_types: ['HCP', 'Brand'],
 *   k: 10
 * });
 * ```
 */
export function useGraphSearch(
  query: string,
  request?: Omit<SearchGraphRequest, 'query'>,
  options?: Omit<
    UseQueryOptions<SearchGraphResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  const fullRequest: SearchGraphRequest = {
    query,
    ...request,
  };

  return useQuery<SearchGraphResponse, ApiError>({
    queryKey: queryKeys.graph.search(query),
    queryFn: () => searchGraph(fullRequest),
    enabled: !!query && query.length >= 2,
    ...options,
  });
}

// =============================================================================
// MUTATION HOOKS
// =============================================================================

/**
 * Hook for graph traversal mutation.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: traverse, isPending } = useTraverseGraph();
 *
 * traverse({
 *   start_node_id: 'kpi_trx',
 *   relationship_types: ['CAUSES', 'IMPACTS'],
 *   max_depth: 3
 * });
 * ```
 */
export function useTraverseGraph(
  options?: Omit<
    UseMutationOptions<TraverseResponse, ApiError, TraverseRequest>,
    'mutationFn'
  >
) {
  return useMutation<TraverseResponse, ApiError, TraverseRequest>({
    mutationFn: traverseGraph,
    ...options,
  });
}

/**
 * Hook for querying causal chains.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: queryCausal, data: chains } = useCausalChains();
 *
 * queryCausal({
 *   kpi_name: 'TRx',
 *   min_confidence: 0.6,
 *   max_chain_length: 3
 * });
 * ```
 */
export function useCausalChains(
  options?: Omit<
    UseMutationOptions<CausalChainResponse, ApiError, CausalChainRequest>,
    'mutationFn'
  >
) {
  return useMutation<CausalChainResponse, ApiError, CausalChainRequest>({
    mutationFn: queryCausalChains,
    ...options,
  });
}

/**
 * Hook for executing openCypher queries.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: executeQuery, data: results } = useCypherQuery();
 *
 * executeQuery({
 *   query: 'MATCH (h:HCP) RETURN h LIMIT 10',
 *   read_only: true
 * });
 * ```
 */
export function useCypherQuery(
  options?: Omit<
    UseMutationOptions<CypherQueryResponse, ApiError, CypherQueryRequest>,
    'mutationFn'
  >
) {
  return useMutation<CypherQueryResponse, ApiError, CypherQueryRequest>({
    mutationFn: executeCypherQuery,
    ...options,
  });
}

/**
 * Hook for adding a knowledge episode to the graph.
 *
 * Automatically invalidates the graph queries after a successful mutation
 * to ensure the UI reflects the new data.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: addNewEpisode, isPending } = useAddEpisode();
 *
 * addNewEpisode({
 *   content: 'Dr. Smith prescribed Kisqali for the patient.',
 *   source: 'orchestrator',
 *   session_id: 'sess_abc123'
 * });
 * ```
 */
export function useAddEpisode(
  options?: Omit<
    UseMutationOptions<AddEpisodeResponse, ApiError, AddEpisodeRequest, unknown>,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<AddEpisodeResponse, ApiError, AddEpisodeRequest, unknown>({
    mutationFn: addEpisode,
    onSuccess: (...args) => {
      // Invalidate relevant queries to refetch with new data
      void queryClient.invalidateQueries({ queryKey: queryKeys.graph.nodes() });
      void queryClient.invalidateQueries({
        queryKey: queryKeys.graph.relationships(),
      });
      void queryClient.invalidateQueries({ queryKey: queryKeys.graph.stats() });

      // Call user's onSuccess if provided
      options?.onSuccess?.(...args);
    },
    ...options,
  });
}

// =============================================================================
// PREFETCH HELPERS
// =============================================================================

/**
 * Prefetch graph nodes for faster navigation.
 *
 * @param params - Query parameters for filtering nodes
 *
 * @example
 * ```tsx
 * // In a component or effect
 * const queryClient = useQueryClient();
 * prefetchNodes(queryClient, { entity_types: 'HCP', limit: 50 });
 * ```
 */
export async function prefetchNodes(
  queryClient: ReturnType<typeof useQueryClient>,
  params?: ListNodesParams
) {
  await queryClient.prefetchQuery({
    queryKey: [...queryKeys.graph.nodes(), params],
    queryFn: () => listNodes(params),
  });
}

/**
 * Prefetch a single node for faster navigation.
 *
 * @param nodeId - The node ID to prefetch
 *
 * @example
 * ```tsx
 * // On hover or when anticipating navigation
 * prefetchNode(queryClient, 'hcp_12345');
 * ```
 */
export async function prefetchNode(
  queryClient: ReturnType<typeof useQueryClient>,
  nodeId: string
) {
  await queryClient.prefetchQuery({
    queryKey: queryKeys.graph.node(nodeId),
    queryFn: () => getNode(nodeId),
  });
}

/**
 * Prefetch graph statistics.
 *
 * @example
 * ```tsx
 * // On app initialization or dashboard load
 * prefetchGraphStats(queryClient);
 * ```
 */
export async function prefetchGraphStats(
  queryClient: ReturnType<typeof useQueryClient>
) {
  await queryClient.prefetchQuery({
    queryKey: queryKeys.graph.stats(),
    queryFn: getGraphStats,
  });
}
