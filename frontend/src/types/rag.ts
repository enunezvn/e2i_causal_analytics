/**
 * Hybrid RAG Types
 * ================
 *
 * TypeScript interfaces for the E2I Hybrid RAG API.
 * Based on src/api/routes/rag.py backend schemas.
 *
 * @module types/rag
 */

import type { TimestampedResponse } from './api';

// =============================================================================
// ENUMS
// =============================================================================

/**
 * Search mode options
 */
export enum SearchMode {
  /** Use all backends with RRF fusion */
  HYBRID = 'hybrid',
  /** Semantic search only */
  VECTOR_ONLY = 'vector',
  /** Keyword search only */
  FULLTEXT_ONLY = 'fulltext',
  /** Graph traversal only */
  GRAPH_ONLY = 'graph',
}

/**
 * Result format options
 */
export enum ResultFormat {
  /** All metadata and scores */
  FULL = 'full',
  /** Essential fields only */
  COMPACT = 'compact',
  /** Just document IDs */
  IDS_ONLY = 'ids',
}

// =============================================================================
// SEARCH MODELS
// =============================================================================

/**
 * Single search result item
 */
export interface SearchResultItem {
  /** Unique document identifier */
  document_id: string;
  /** Document content or snippet */
  content: string;
  /** Relevance score (0-1) */
  score: number;
  /** Source backend (vector/fulltext/graph) */
  source: string;
  /** Additional metadata */
  metadata: Record<string, unknown>;
}

/**
 * Extracted entities from query
 */
export interface ExtractedEntities {
  /** Brand names (Kisqali, Fabhalta, Remibrutinib) */
  brands: string[];
  /** Region names (northeast, south, midwest, west) */
  regions: string[];
  /** KPI names (trx, nrx, conversion_rate, etc.) */
  kpis: string[];
  /** Agent names */
  agents: string[];
  /** Patient journey stages */
  journey_stages: string[];
  /** Time references (Q1, last month, etc.) */
  time_references: string[];
  /** HCP segments */
  hcp_segments: string[];
}

/**
 * Hybrid search request payload
 */
export interface RAGSearchRequest {
  /** Natural language query (1-1000 chars) */
  query: string;
  /** Search mode (default: hybrid) */
  mode?: SearchMode;
  /** Maximum results to return (1-50, default 10) */
  top_k?: number;
  /** Minimum relevance score (0-1, default 0) */
  min_score?: number;
  /** Require results from all backends */
  require_all_sources?: boolean;
  /** Apply graph context boost (default true) */
  include_graph_boost?: boolean;
  /** Optional metadata filters */
  filters?: Record<string, unknown>;
  /** Result format (default: full) */
  format?: ResultFormat;
  /** Session ID for logging */
  session_id?: string;
  /** User ID for logging */
  user_id?: string;
}

/**
 * Hybrid search response payload
 */
export interface RAGSearchResponse extends TimestampedResponse {
  /** Unique search identifier */
  search_id: string;
  /** Original query */
  query: string;
  /** Ranked search results */
  results: SearchResultItem[];
  /** Total number of results */
  total_results: number;
  /** Extracted domain entities */
  entities: ExtractedEntities;
  /** Search statistics per backend */
  stats: {
    vector_count?: number;
    fulltext_count?: number;
    graph_count?: number;
    [key: string]: unknown;
  };
  /** Total search latency in milliseconds */
  latency_ms: number;
}

// =============================================================================
// GRAPH MODELS
// =============================================================================

/**
 * Node in the causal graph (for visualization)
 */
export interface RAGGraphNode {
  /** Node identifier */
  id: string;
  /** Display label */
  label: string;
  /** Node type (brand/kpi/region/agent) */
  type: string;
  /** Node properties */
  properties: Record<string, unknown>;
}

/**
 * Edge in the causal graph
 */
export interface RAGGraphEdge {
  /** Source node ID */
  source: string;
  /** Target node ID */
  target: string;
  /** Relationship type */
  relationship: string;
  /** Edge weight (default 1.0) */
  weight: number;
  /** Edge properties */
  properties: Record<string, unknown>;
}

/**
 * Causal subgraph for visualization
 */
export interface CausalSubgraphResponse {
  /** Center entity of the subgraph */
  entity: string;
  /** Graph nodes */
  nodes: RAGGraphNode[];
  /** Graph edges */
  edges: RAGGraphEdge[];
  /** Traversal depth */
  depth: number;
  /** Total node count */
  node_count: number;
  /** Total edge count */
  edge_count: number;
  /** Query time in milliseconds */
  query_time_ms: number;
}

/**
 * Causal path between two entities
 */
export interface CausalPathResponse {
  /** Source entity */
  source: string;
  /** Target entity */
  target: string;
  /** List of paths (node sequences) */
  paths: string[][];
  /** Length of shortest path */
  shortest_path_length: number;
  /** Number of paths found */
  total_paths: number;
  /** Query time in milliseconds */
  query_time_ms: number;
}

// =============================================================================
// HEALTH MODELS
// =============================================================================

/**
 * Health status for a single backend
 */
export interface BackendHealthStatus {
  /** healthy/degraded/unhealthy/unknown */
  status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  /** Last check latency in ms */
  latency_ms: number;
  /** Last health check time (ISO 8601) */
  last_check: string;
  /** Consecutive failure count */
  consecutive_failures: number;
  /** Circuit breaker state */
  circuit_breaker_state?: 'closed' | 'open' | 'half_open';
  /** Last error message */
  error?: string;
}

/**
 * Overall RAG health response
 */
export interface RAGHealthResponse extends TimestampedResponse {
  /** Overall status: healthy/degraded/unhealthy */
  status: 'healthy' | 'degraded' | 'unhealthy';
  /** Per-backend health */
  backends: {
    vector?: BackendHealthStatus;
    fulltext?: BackendHealthStatus;
    graph?: BackendHealthStatus;
    [key: string]: BackendHealthStatus | undefined;
  };
  /** Whether background monitoring is active */
  monitoring_enabled: boolean;
}

// =============================================================================
// STATS MODELS
// =============================================================================

/**
 * RAG usage statistics
 */
export interface RAGStatsResponse {
  /** Time period in hours */
  period_hours: number;
  /** Total searches in period */
  total_searches: number;
  /** Average latency in ms */
  avg_latency_ms: number;
  /** Top queries */
  top_queries: Array<{
    query: string;
    count: number;
  }>;
  /** Usage by backend */
  backend_usage: {
    vector: number;
    fulltext: number;
    graph: number;
  };
  /** Error rate (0-1) */
  error_rate: number;
  /** Status message */
  message?: string;
}

// =============================================================================
// ENTITY EXTRACTION
// =============================================================================

/**
 * Query parameters for entity extraction endpoint
 */
export interface ExtractEntitiesParams {
  /** Query to analyze (1-1000 chars) */
  query: string;
}

/**
 * Response for entity extraction (same as ExtractedEntities)
 */
export type ExtractEntitiesResponse = ExtractedEntities;
