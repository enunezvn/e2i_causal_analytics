/**
 * RAG API Client
 * ==============
 *
 * TypeScript API client functions for the E2I Hybrid RAG endpoints.
 * Uses the shared apiClient for consistent error handling and interceptors.
 *
 * Endpoints:
 * - Hybrid search (vector + fulltext + graph)
 * - Entity extraction
 * - Causal subgraph queries
 * - Causal path queries
 * - RAG statistics
 * - Health checks
 *
 * @module api/rag
 */

import { get, post } from '@/lib/api-client';
import type {
  CausalPathResponse,
  CausalSubgraphResponse,
  ExtractedEntities,
  ExtractEntitiesParams,
  RAGHealthResponse,
  RAGSearchRequest,
  RAGSearchResponse,
  RAGStatsResponse,
} from '@/types/rag';

// =============================================================================
// RAG API ENDPOINTS
// =============================================================================

const RAG_BASE = '/rag';

// =============================================================================
// SEARCH ENDPOINTS
// =============================================================================

/**
 * Perform hybrid RAG search across vector, fulltext, and graph backends.
 *
 * Uses Reciprocal Rank Fusion (RRF) to combine results from multiple
 * retrieval methods with configurable modes and filters.
 *
 * @param request - Search parameters including query and mode
 * @returns Ranked search results with extracted entities and statistics
 *
 * @example
 * ```typescript
 * const results = await searchRAG({
 *   query: 'What are the key drivers of TRx in the Northeast?',
 *   mode: SearchMode.HYBRID,
 *   top_k: 10,
 *   include_graph_boost: true
 * });
 * ```
 */
export async function searchRAG(
  request: RAGSearchRequest
): Promise<RAGSearchResponse> {
  return post<RAGSearchResponse, RAGSearchRequest>(
    `${RAG_BASE}/search`,
    request
  );
}

/**
 * Simple query endpoint for quick searches.
 *
 * @param query - Natural language query string
 * @param params - Optional additional parameters
 * @returns Search results
 *
 * @example
 * ```typescript
 * const results = await queryRAG('Kisqali efficacy data', { top_k: 5 });
 * ```
 */
export async function queryRAG(
  query: string,
  params?: Partial<Omit<RAGSearchRequest, 'query'>>
): Promise<RAGSearchResponse> {
  return searchRAG({ query, ...params });
}

// =============================================================================
// ENTITY EXTRACTION ENDPOINTS
// =============================================================================

/**
 * Extract domain entities from a query.
 *
 * Identifies brands, regions, KPIs, agents, journey stages,
 * time references, and HCP segments from natural language.
 *
 * @param params - Query to analyze
 * @returns Extracted domain entities
 *
 * @example
 * ```typescript
 * const entities = await extractEntities({
 *   query: 'What is the TRx trend for Kisqali in the Northeast?'
 * });
 * console.log(`Brands: ${entities.brands.join(', ')}`);
 * ```
 */
export async function extractEntities(
  params: ExtractEntitiesParams
): Promise<ExtractedEntities> {
  return get<ExtractedEntities>(
    `${RAG_BASE}/entities`,
    params as unknown as Record<string, unknown>
  );
}

// =============================================================================
// GRAPH ENDPOINTS
// =============================================================================

/**
 * Get causal subgraph centered on an entity.
 *
 * Returns nodes and edges within the specified depth for visualization.
 *
 * @param entity - Center entity identifier
 * @param depth - Traversal depth (default 2)
 * @returns Subgraph with nodes, edges, and statistics
 *
 * @example
 * ```typescript
 * const subgraph = await getCausalSubgraph('kisqali', 3);
 * console.log(`Nodes: ${subgraph.node_count}, Edges: ${subgraph.edge_count}`);
 * ```
 */
export async function getCausalSubgraph(
  entity: string,
  depth?: number
): Promise<CausalSubgraphResponse> {
  return get<CausalSubgraphResponse>(
    `${RAG_BASE}/graph/subgraph/${encodeURIComponent(entity)}`,
    depth !== undefined ? { depth } : undefined
  );
}

/**
 * Find causal paths between two entities.
 *
 * Discovers all paths connecting source to target with path statistics.
 *
 * @param source - Source entity identifier
 * @param target - Target entity identifier
 * @param maxDepth - Maximum path depth (default 4)
 * @returns Paths connecting source and target
 *
 * @example
 * ```typescript
 * const paths = await getCausalPaths('hcp_engagement', 'trx', 3);
 * console.log(`Shortest path: ${paths.shortest_path_length} hops`);
 * ```
 */
export async function getCausalPaths(
  source: string,
  target: string,
  maxDepth?: number
): Promise<CausalPathResponse> {
  return get<CausalPathResponse>(`${RAG_BASE}/graph/paths`, {
    source,
    target,
    max_depth: maxDepth,
  });
}

// =============================================================================
// STATS ENDPOINT
// =============================================================================

/**
 * Get RAG usage statistics.
 *
 * Returns search counts, latency metrics, top queries, and error rates.
 *
 * @param periodHours - Time period in hours (default 24)
 * @returns RAG usage statistics
 *
 * @example
 * ```typescript
 * const stats = await getRAGStats(48);
 * console.log(`Total searches: ${stats.total_searches}`);
 * ```
 */
export async function getRAGStats(
  periodHours?: number
): Promise<RAGStatsResponse> {
  return get<RAGStatsResponse>(
    `${RAG_BASE}/stats`,
    periodHours !== undefined ? { period_hours: periodHours } : undefined
  );
}

// =============================================================================
// HEALTH ENDPOINT
// =============================================================================

/**
 * Health check for the RAG service.
 *
 * Returns status of vector, fulltext, and graph backends
 * with circuit breaker states.
 *
 * @returns Service health status with backend checks
 *
 * @example
 * ```typescript
 * const health = await getRAGHealth();
 * if (health.status === 'healthy') {
 *   console.log('RAG service is operational');
 * }
 * ```
 */
export async function getRAGHealth(): Promise<RAGHealthResponse> {
  return get<RAGHealthResponse>(`${RAG_BASE}/health`);
}
