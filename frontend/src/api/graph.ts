/**
 * Graph API Client
 * ================
 *
 * TypeScript API client functions for the E2I Knowledge Graph endpoints.
 * Uses the shared apiClient for consistent error handling and interceptors.
 *
 * Endpoints:
 * - Node listing and retrieval
 * - Relationship queries
 * - Graph traversal
 * - Causal chain analysis
 * - openCypher query execution
 * - Natural language search
 * - Graph statistics and health
 *
 * @module api/graph
 */

import { get, post } from '@/lib/api-client';
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

// =============================================================================
// GRAPH API ENDPOINTS
// =============================================================================

const GRAPH_BASE = '/graph';

// =============================================================================
// NODE ENDPOINTS
// =============================================================================

/**
 * List nodes in the knowledge graph with filtering and pagination.
 *
 * @param params - Query parameters for filtering nodes
 * @returns Paginated list of graph nodes
 *
 * @example
 * ```typescript
 * const response = await listNodes({
 *   entity_types: 'HCP,Brand',
 *   search: 'oncology',
 *   limit: 20
 * });
 * ```
 */
export async function listNodes(
  params?: ListNodesParams
): Promise<ListNodesResponse> {
  return get<ListNodesResponse>(
    `${GRAPH_BASE}/nodes`,
    params as Record<string, unknown> | undefined
  );
}

/**
 * Get a specific node by ID.
 *
 * @param nodeId - Unique node identifier
 * @returns Full node details including all properties
 * @throws {ApiError} When node is not found (404)
 *
 * @example
 * ```typescript
 * const node = await getNode('hcp_12345');
 * ```
 */
export async function getNode(nodeId: string): Promise<GraphNode> {
  return get<GraphNode>(`${GRAPH_BASE}/nodes/${encodeURIComponent(nodeId)}`);
}

/**
 * Get the relationship network around a node.
 *
 * Returns all connected nodes within max_depth hops, grouped by type.
 * Supports Patient and HCP nodes with specialized network discovery.
 *
 * @param nodeId - Central node identifier
 * @param maxDepth - Maximum traversal depth (1-5, default 2)
 * @returns Connected nodes grouped by type
 *
 * @example
 * ```typescript
 * const network = await getNodeNetwork('patient_001', 3);
 * ```
 */
export async function getNodeNetwork(
  nodeId: string,
  maxDepth?: number
): Promise<NodeNetworkResponse> {
  return get<NodeNetworkResponse>(
    `${GRAPH_BASE}/nodes/${encodeURIComponent(nodeId)}/network`,
    maxDepth !== undefined ? { max_depth: maxDepth } : undefined
  );
}

// =============================================================================
// RELATIONSHIP ENDPOINTS
// =============================================================================

/**
 * List relationships in the knowledge graph with filtering.
 *
 * Supports filtering by relationship types, source/target nodes, and confidence.
 *
 * @param params - Query parameters for filtering relationships
 * @returns Paginated list of graph relationships
 *
 * @example
 * ```typescript
 * const response = await listRelationships({
 *   relationship_types: 'CAUSES,IMPACTS',
 *   min_confidence: 0.7
 * });
 * ```
 */
export async function listRelationships(
  params?: ListRelationshipsParams
): Promise<ListRelationshipsResponse> {
  return get<ListRelationshipsResponse>(
    `${GRAPH_BASE}/relationships`,
    params as Record<string, unknown> | undefined
  );
}

// =============================================================================
// TRAVERSAL ENDPOINTS
// =============================================================================

/**
 * Traverse the graph from a starting node.
 *
 * Returns a subgraph containing all nodes within max_depth hops,
 * all relationships connecting them, and paths from start to discovered nodes.
 *
 * @param request - Traversal configuration
 * @returns Subgraph with nodes, relationships, and paths
 *
 * @example
 * ```typescript
 * const result = await traverseGraph({
 *   start_node_id: 'kpi_trx',
 *   relationship_types: ['CAUSES', 'IMPACTS'],
 *   direction: 'incoming',
 *   max_depth: 3
 * });
 * ```
 */
export async function traverseGraph(
  request: TraverseRequest
): Promise<TraverseResponse> {
  return post<TraverseResponse, TraverseRequest>(
    `${GRAPH_BASE}/traverse`,
    request
  );
}

// =============================================================================
// CAUSAL CHAIN ENDPOINTS
// =============================================================================

/**
 * Query causal chains in the knowledge graph.
 *
 * Finds chains of CAUSES/IMPACTS relationships connecting entities to KPIs,
 * source to target entities, or upstream drivers to downstream outcomes.
 *
 * @param request - Causal chain query configuration
 * @returns Discovered causal chains with confidence scores
 *
 * @example
 * ```typescript
 * const chains = await queryCausalChains({
 *   kpi_name: 'TRx',
 *   min_confidence: 0.6,
 *   max_chain_length: 3
 * });
 * ```
 */
export async function queryCausalChains(
  request: CausalChainRequest
): Promise<CausalChainResponse> {
  return post<CausalChainResponse, CausalChainRequest>(
    `${GRAPH_BASE}/causal-chains`,
    request
  );
}

// =============================================================================
// CYPHER QUERY ENDPOINT
// =============================================================================

/**
 * Execute an openCypher query against the graph.
 *
 * Supports MATCH, RETURN, WHERE clauses with parameterized queries.
 * Read-only mode is enforced by default.
 *
 * @param request - Cypher query with parameters
 * @returns Query results with column names and row count
 * @throws {ApiError} When query contains write operations in read-only mode (400)
 *
 * @example
 * ```typescript
 * const result = await executeCypherQuery({
 *   query: 'MATCH (h:HCP)-[:PRESCRIBES]->(b:Brand) WHERE b.name = $brand RETURN h LIMIT 10',
 *   parameters: { brand: 'Kisqali' },
 *   read_only: true
 * });
 * ```
 */
export async function executeCypherQuery(
  request: CypherQueryRequest
): Promise<CypherQueryResponse> {
  return post<CypherQueryResponse, CypherQueryRequest>(
    `${GRAPH_BASE}/query`,
    request
  );
}

// =============================================================================
// EPISODE ENDPOINTS
// =============================================================================

/**
 * Add a knowledge episode to the graph.
 *
 * Graphiti automatically extracts entities, discovers relationships,
 * links to existing graph nodes, and tracks temporal validity.
 *
 * @param request - Episode content and metadata
 * @returns Created episode with extracted entities and relationships
 *
 * @example
 * ```typescript
 * const episode = await addEpisode({
 *   content: 'Dr. Smith prescribed Kisqali for the patient with HR+/HER2- breast cancer.',
 *   source: 'orchestrator',
 *   session_id: 'sess_abc123'
 * });
 * ```
 */
export async function addEpisode(
  request: AddEpisodeRequest
): Promise<AddEpisodeResponse> {
  return post<AddEpisodeResponse, AddEpisodeRequest>(
    `${GRAPH_BASE}/episodes`,
    request
  );
}

// =============================================================================
// SEARCH ENDPOINTS
// =============================================================================

/**
 * Natural language search across the knowledge graph.
 *
 * Uses semantic search to find relevant entities, facts, relationships,
 * and historical episodes matching the query.
 *
 * @param request - Search query and filters
 * @returns Ranked search results with relevance scores
 *
 * @example
 * ```typescript
 * const results = await searchGraph({
 *   query: 'What factors impact TRx in the Northeast?',
 *   entity_types: ['HCP', 'Brand', 'KPI'],
 *   k: 10
 * });
 * ```
 */
export async function searchGraph(
  request: SearchGraphRequest
): Promise<SearchGraphResponse> {
  return post<SearchGraphResponse, SearchGraphRequest>(
    `${GRAPH_BASE}/search`,
    request
  );
}

// =============================================================================
// STATS ENDPOINT
// =============================================================================

/**
 * Get knowledge graph statistics.
 *
 * Returns counts for total nodes and relationships, breakdowns by type,
 * and episode/community counts.
 *
 * @returns Graph statistics summary
 *
 * @example
 * ```typescript
 * const stats = await getGraphStats();
 * console.log(`Total nodes: ${stats.total_nodes}`);
 * ```
 */
export async function getGraphStats(): Promise<GraphStatsResponse> {
  return get<GraphStatsResponse>(`${GRAPH_BASE}/stats`);
}

// =============================================================================
// HEALTH ENDPOINT
// =============================================================================

/**
 * Health check for graph services.
 *
 * Returns status of Graphiti service, FalkorDB connection,
 * and WebSocket connections.
 *
 * @returns Service health status
 *
 * @example
 * ```typescript
 * const health = await getGraphHealth();
 * if (health.status === 'healthy') {
 *   console.log('Graph services are operational');
 * }
 * ```
 */
export async function getGraphHealth(): Promise<GraphHealthResponse> {
  return get<GraphHealthResponse>(`${GRAPH_BASE}/health`);
}
