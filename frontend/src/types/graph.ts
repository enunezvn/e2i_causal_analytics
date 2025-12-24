/**
 * Knowledge Graph Types
 * =====================
 *
 * TypeScript interfaces for the E2I Knowledge Graph API.
 * Based on src/api/models/graph.py backend schemas.
 *
 * @module types/graph
 */

import type {
  PaginatedResponse,
  QueryLatency,
  TimestampedResponse,
  TimestampFields,
  SortOrder,
} from './api';

// =============================================================================
// ENUMS
// =============================================================================

/**
 * Entity types in the E2I knowledge graph
 */
export enum EntityType {
  PATIENT = 'Patient',
  HCP = 'HCP',
  BRAND = 'Brand',
  REGION = 'Region',
  KPI = 'KPI',
  CAUSAL_PATH = 'CausalPath',
  TRIGGER = 'Trigger',
  AGENT = 'Agent',
  EPISODE = 'Episode',
  COMMUNITY = 'Community',
  TREATMENT = 'Treatment',
  PREDICTION = 'Prediction',
  EXPERIMENT = 'Experiment',
  AGENT_ACTIVITY = 'AgentActivity',
}

/**
 * Relationship types in the E2I knowledge graph
 */
export enum RelationshipType {
  TREATED_BY = 'TREATED_BY',
  PRESCRIBED = 'PRESCRIBED',
  PRESCRIBES = 'PRESCRIBES',
  CAUSES = 'CAUSES',
  IMPACTS = 'IMPACTS',
  INFLUENCES = 'INFLUENCES',
  DISCOVERED = 'DISCOVERED',
  GENERATED = 'GENERATED',
  MENTIONS = 'MENTIONS',
  MEMBER_OF = 'MEMBER_OF',
  RELATES_TO = 'RELATES_TO',
  RECEIVED = 'RECEIVED',
  LOCATED_IN = 'LOCATED_IN',
  PRACTICES_IN = 'PRACTICES_IN',
  MEASURED_IN = 'MEASURED_IN',
}

/**
 * Fields to sort nodes by
 */
export enum NodeSortField {
  CREATED_AT = 'created_at',
  UPDATED_AT = 'updated_at',
  NAME = 'name',
  TYPE = 'type',
}

// =============================================================================
// BASE MODELS
// =============================================================================

/**
 * A node in the knowledge graph
 */
export interface GraphNode extends TimestampFields {
  /** Unique node identifier */
  id: string;
  /** Node entity type */
  type: EntityType;
  /** Node display name */
  name: string;
  /** Additional node properties */
  properties: Record<string, unknown>;
}

/**
 * A relationship (edge) in the knowledge graph
 */
export interface GraphRelationship {
  /** Unique relationship identifier */
  id: string;
  /** Relationship type */
  type: RelationshipType;
  /** Source node ID */
  source_id: string;
  /** Target node ID */
  target_id: string;
  /** Additional relationship properties */
  properties: Record<string, unknown>;
  /** Confidence score (0-1) */
  confidence?: number;
  /** Creation timestamp */
  created_at?: string;
}

/**
 * A path through the graph (sequence of nodes and relationships)
 */
export interface GraphPath {
  /** Nodes in the path */
  nodes: GraphNode[];
  /** Relationships connecting nodes */
  relationships: GraphRelationship[];
  /** Combined path confidence */
  total_confidence?: number;
  /** Number of hops in the path */
  path_length: number;
}

// =============================================================================
// REQUEST MODELS
// =============================================================================

/**
 * Request for listing graph nodes (query params)
 */
export interface ListNodesParams {
  /** Comma-separated entity types to filter by */
  entity_types?: string;
  /** Text search in node names/properties */
  search?: string;
  /** Maximum results (1-500, default 50) */
  limit?: number;
  /** Pagination offset */
  offset?: number;
  /** Sort field */
  sort_by?: NodeSortField;
  /** Sort order */
  sort_order?: SortOrder;
}

/**
 * Request for listing graph relationships (query params)
 */
export interface ListRelationshipsParams {
  /** Comma-separated relationship types */
  relationship_types?: string;
  /** Filter by source node ID */
  source_id?: string;
  /** Filter by target node ID */
  target_id?: string;
  /** Minimum confidence threshold (0-1) */
  min_confidence?: number;
  /** Maximum results (1-500, default 50) */
  limit?: number;
  /** Pagination offset */
  offset?: number;
}

/**
 * Request for graph traversal
 */
export interface TraverseRequest {
  /** Starting node ID */
  start_node_id: string;
  /** Relationship types to follow */
  relationship_types?: RelationshipType[];
  /** Traversal direction: outgoing, incoming, both */
  direction?: 'outgoing' | 'incoming' | 'both';
  /** Maximum traversal depth (1-5, default 2) */
  max_depth?: number;
  /** Minimum edge confidence (0-1, default 0) */
  min_confidence?: number;
  /** Include node/edge properties */
  include_properties?: boolean;
}

/**
 * Request for causal chain queries
 */
export interface CausalChainRequest {
  /** Target KPI name */
  kpi_name?: string;
  /** Source entity for chain */
  source_entity_id?: string;
  /** Target entity for chain */
  target_entity_id?: string;
  /** Minimum causal effect size */
  min_effect_size?: number;
  /** Minimum confidence (0-1, default 0.5) */
  min_confidence?: number;
  /** Maximum chain length (1-10, default 4) */
  max_chain_length?: number;
}

/**
 * Request for executing openCypher queries
 */
export interface CypherQueryRequest {
  /** openCypher query string */
  query: string;
  /** Query parameters */
  parameters?: Record<string, unknown>;
  /** Query timeout in seconds (1-120, default 30) */
  timeout_seconds?: number;
  /** Enforce read-only query (default true) */
  read_only?: boolean;
}

/**
 * Request for adding a knowledge episode
 */
export interface AddEpisodeRequest {
  /** Episode content text */
  content: string;
  /** Source of the episode (agent name or 'user') */
  source: string;
  /** Session ID for context */
  session_id?: string;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
  /** Automatically extract entities/relationships (default true) */
  extract_entities?: boolean;
}

/**
 * Request for natural language graph search
 */
export interface SearchGraphRequest {
  /** Natural language search query */
  query: string;
  /** Filter results by entity types */
  entity_types?: EntityType[];
  /** Session ID for context */
  session_id?: string;
  /** Number of results (1-50, default 10) */
  k?: number;
  /** Minimum relevance score (0-1, default 0) */
  min_score?: number;
}

// =============================================================================
// RESPONSE MODELS
// =============================================================================

/**
 * Response for listing graph nodes
 */
export interface ListNodesResponse
  extends PaginatedResponse,
    QueryLatency,
    TimestampedResponse {
  /** List of nodes */
  nodes: GraphNode[];
}

/**
 * Response for listing graph relationships
 */
export interface ListRelationshipsResponse
  extends PaginatedResponse,
    QueryLatency,
    TimestampedResponse {
  /** List of relationships */
  relationships: GraphRelationship[];
}

/**
 * Response for graph traversal
 */
export interface TraverseResponse extends QueryLatency, TimestampedResponse {
  /** Subgraph with nodes and edges */
  subgraph: {
    nodes: GraphNode[];
    relationships: GraphRelationship[];
  };
  /** All traversed nodes */
  nodes: GraphNode[];
  /** All traversed relationships */
  relationships: GraphRelationship[];
  /** Discovered paths */
  paths: GraphPath[];
  /** Actual depth reached */
  max_depth_reached: number;
}

/**
 * Response for causal chain queries
 */
export interface CausalChainResponse extends QueryLatency, TimestampedResponse {
  /** Discovered causal chains */
  chains: GraphPath[];
  /** Number of chains found */
  total_chains: number;
  /** Highest confidence chain */
  strongest_chain?: GraphPath;
  /** Aggregate causal effect */
  aggregate_effect?: number;
}

/**
 * Response for openCypher query execution
 */
export interface CypherQueryResponse extends QueryLatency, TimestampedResponse {
  /** Query results */
  results: Record<string, unknown>[];
  /** Result column names */
  columns: string[];
  /** Number of result rows */
  row_count: number;
  /** Whether query was read-only */
  read_only: boolean;
}

/**
 * Response for adding a knowledge episode
 */
export interface AddEpisodeResponse extends TimestampedResponse {
  /** Created episode ID */
  episode_id: string;
  /** Entities extracted from content */
  extracted_entities: Array<{
    type: string;
    name: string;
    confidence: number;
  }>;
  /** Relationships extracted from content */
  extracted_relationships: Array<{
    type: string;
    source_id: string;
    target_id: string;
    confidence: number;
  }>;
  /** Brief content summary */
  content_summary?: string;
  /** Processing latency in milliseconds */
  processing_latency_ms: number;
}

/**
 * Response for natural language graph search
 */
export interface SearchGraphResponse extends QueryLatency, TimestampedResponse {
  /** Search results */
  results: Array<{
    id: string;
    name: string;
    type: string;
    score: number;
    properties?: Record<string, unknown>;
    relationships?: Array<{
      type: string;
      target_id: string;
    }>;
  }>;
  /** Number of results */
  total_results: number;
  /** Original query */
  query: string;
}

/**
 * Response for graph statistics
 */
export interface GraphStatsResponse extends TimestampedResponse {
  /** Total node count */
  total_nodes: number;
  /** Total relationship count */
  total_relationships: number;
  /** Node counts by entity type */
  nodes_by_type: Record<string, number>;
  /** Relationship counts by type */
  relationships_by_type: Record<string, number>;
  /** Total episodes in graph */
  total_episodes: number;
  /** Total communities */
  total_communities: number;
  /** Last graph update time */
  last_updated?: string;
}

/**
 * Response for node network queries
 */
export interface NodeNetworkResponse extends QueryLatency, TimestampedResponse {
  /** Central node ID */
  node_id: string;
  /** Central node type */
  node_type: EntityType;
  /** Connected nodes grouped by type */
  connected_nodes: Record<
    string,
    Array<{
      id: string;
      properties: Record<string, unknown>;
    }>
  >;
  /** Total connected nodes */
  total_connections: number;
  /** Traversal depth used */
  max_depth: number;
}

// =============================================================================
// WEBSOCKET MODELS
// =============================================================================

/**
 * Message format for WebSocket graph streaming
 */
export interface GraphStreamMessage {
  /** Event type: node_added, edge_added, update, etc. */
  event_type: string;
  /** Event payload data */
  payload: Record<string, unknown>;
  /** Timestamp (ISO 8601) */
  timestamp: string;
  /** Associated session ID */
  session_id?: string;
}

/**
 * Subscription configuration for graph streaming
 */
export interface GraphSubscription {
  /** Entity types to subscribe to */
  entity_types?: EntityType[];
  /** Relationship types to subscribe to */
  relationship_types?: RelationshipType[];
  /** Session IDs to filter events */
  session_ids?: string[];
  /** Include full properties */
  include_properties?: boolean;
}

// =============================================================================
// HEALTH CHECK
// =============================================================================

/**
 * Graph service health check response
 */
export interface GraphHealthResponse {
  /** Overall status */
  status: 'healthy' | 'degraded';
  /** Graphiti service status */
  graphiti: 'connected' | 'unavailable';
  /** FalkorDB status */
  falkordb: 'connected' | 'unavailable';
  /** Active WebSocket connections */
  websocket_connections: number;
  /** Timestamp (ISO 8601) */
  timestamp: string;
}
