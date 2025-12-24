/**
 * Memory System Types
 * ===================
 *
 * TypeScript interfaces for the E2I Memory System API.
 * Based on src/api/routes/memory.py backend schemas.
 *
 * @module types/memory
 */

import type {
  QueryLatency,
  TimestampedResponse,
  Metadata,
} from './api';

// =============================================================================
// ENUMS
// =============================================================================

/**
 * Memory types for filtering search
 */
export enum MemoryType {
  EPISODIC = 'episodic',
  PROCEDURAL = 'procedural',
  SEMANTIC = 'semantic',
  ALL = 'all',
}

/**
 * Retrieval methods for hybrid search
 */
export enum RetrievalMethod {
  DENSE = 'dense',
  SPARSE = 'sparse',
  GRAPH = 'graph',
  HYBRID = 'hybrid',
}

// =============================================================================
// SEARCH MODELS
// =============================================================================

/**
 * Request for hybrid memory search
 */
export interface MemorySearchRequest {
  /** Search query text (1-2000 chars) */
  query: string;
  /** Number of results to return (1-100, default 10) */
  k?: number;
  /** Memory types to search (all if not specified) */
  memory_types?: MemoryType[];
  /** Retrieval method (default: hybrid) */
  retrieval_method?: RetrievalMethod;
  /** Entity IDs for graph traversal */
  entities?: string[];
  /** KPI name for targeted graph traversal */
  kpi_name?: string;
  /** Additional filters (brand, region, agent_name) */
  filters?: Record<string, unknown>;
  /** Custom weights for hybrid retrieval */
  weights?: Record<string, number>;
  /** Minimum score threshold (0-1, default 0) */
  min_score?: number;
}

/**
 * Single search result
 */
export interface MemorySearchResult {
  /** Retrieved content */
  content: string;
  /** Source memory type or table */
  source: string;
  /** Source record ID */
  source_id: string;
  /** Relevance score (0-1) */
  score: number;
  /** Method used to retrieve */
  retrieval_method: string;
  /** Additional metadata */
  metadata: Record<string, unknown>;
}

/**
 * Response for memory search
 */
export interface MemorySearchResponse extends TimestampedResponse {
  /** Original query */
  query: string;
  /** Search results */
  results: MemorySearchResult[];
  /** Number of results returned */
  total_results: number;
  /** Method used */
  retrieval_method: string;
  /** Search latency in milliseconds */
  search_latency_ms: number;
}

// =============================================================================
// EPISODIC MEMORY MODELS
// =============================================================================

/**
 * Input for creating episodic memory
 */
export interface EpisodicMemoryInput extends Metadata {
  /** Memory content (1-10000 chars) */
  content: string;
  /** Type of event (query, response, action) */
  event_type: string;
  /** Session ID */
  session_id?: string;
  /** Agent that created this memory */
  agent_name?: string;
  /** Brand context */
  brand?: string;
  /** Region context */
  region?: string;
  /** Associated HCP ID */
  hcp_id?: string;
  /** Associated patient ID */
  patient_id?: string;
}

/**
 * Response for episodic memory operations
 */
export interface EpisodicMemoryResponse extends Metadata {
  /** Memory ID */
  id: string;
  /** Memory content */
  content: string;
  /** Event type */
  event_type: string;
  /** Session ID */
  session_id?: string;
  /** Agent name */
  agent_name?: string;
  /** Brand context */
  brand?: string;
  /** Region context */
  region?: string;
  /** Creation timestamp */
  created_at: string;
}

// =============================================================================
// PROCEDURAL MEMORY MODELS
// =============================================================================

/**
 * Request to record procedural outcome feedback
 */
export interface ProceduralFeedbackRequest {
  /** ID of the procedure */
  procedure_id: string;
  /** Outcome: success, partial, failure */
  outcome: 'success' | 'partial' | 'failure';
  /** Outcome score (0-1) */
  score: number;
  /** Optional feedback text */
  feedback_text?: string;
  /** Session context */
  session_id?: string;
  /** Agent providing feedback */
  agent_name?: string;
}

/**
 * Response for procedural feedback recording
 */
export interface ProceduralFeedbackResponse extends TimestampedResponse {
  /** Procedure ID */
  procedure_id: string;
  /** Whether feedback was recorded */
  feedback_recorded: boolean;
  /** New success rate after update */
  new_success_rate?: number;
  /** Status message */
  message: string;
}

// =============================================================================
// SEMANTIC MEMORY MODELS
// =============================================================================

/**
 * Request for semantic graph path queries
 */
export interface SemanticPathRequest {
  /** Starting entity ID */
  start_entity_id?: string;
  /** Ending entity ID */
  end_entity_id?: string;
  /** KPI name for causal paths */
  kpi_name?: string;
  /** Type of relationship to follow (default: causal_path) */
  relationship_type?: string;
  /** Maximum traversal depth (1-10, default 3) */
  max_depth?: number;
  /** Minimum confidence (0-1, default 0.5) */
  min_confidence?: number;
}

/**
 * Response for semantic path queries
 */
export interface SemanticPathResponse extends QueryLatency, TimestampedResponse {
  /** Found paths */
  paths: Array<Record<string, unknown>>;
  /** Number of paths found */
  total_paths: number;
  /** Depth searched */
  max_depth_searched: number;
}

// =============================================================================
// STATS MODELS
// =============================================================================

/**
 * Memory system statistics
 */
export interface MemoryStatsResponse {
  /** Episodic memory stats */
  episodic: {
    /** Total memories */
    total_memories: number;
    /** Memories in last 24 hours */
    recent_24h: number;
  };
  /** Procedural memory stats */
  procedural: {
    /** Total procedures */
    total_procedures: number;
    /** Average success rate */
    average_success_rate: number;
  };
  /** Semantic memory stats */
  semantic: {
    /** Total entities */
    total_entities: number;
    /** Total relationships */
    total_relationships: number;
  };
  /** Last update timestamp (ISO 8601) */
  last_updated: string;
}
