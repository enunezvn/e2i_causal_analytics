/**
 * Memory API Client
 * =================
 *
 * TypeScript API client functions for the E2I Memory System endpoints.
 * Uses the shared apiClient for consistent error handling and interceptors.
 *
 * Endpoints:
 * - Hybrid memory search
 * - Episodic memory CRUD
 * - Procedural feedback recording
 * - Semantic path queries
 * - Memory statistics
 *
 * @module api/memory
 */

import { get, post } from '@/lib/api-client';
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

// =============================================================================
// MEMORY API ENDPOINTS
// =============================================================================

const MEMORY_BASE = '/memory';

// =============================================================================
// SEARCH ENDPOINTS
// =============================================================================

/**
 * Perform hybrid memory search across episodic, procedural, and semantic memory.
 *
 * Supports multiple retrieval methods: dense, sparse, graph, or hybrid.
 *
 * @param request - Search parameters including query and filters
 * @returns Ranked search results with relevance scores
 *
 * @example
 * ```typescript
 * const results = await searchMemory({
 *   query: 'What are the best practices for HCP engagement?',
 *   k: 10,
 *   memory_types: [MemoryType.EPISODIC, MemoryType.PROCEDURAL],
 *   retrieval_method: RetrievalMethod.HYBRID
 * });
 * ```
 */
export async function searchMemory(
  request: MemorySearchRequest
): Promise<MemorySearchResponse> {
  return post<MemorySearchResponse, MemorySearchRequest>(
    `${MEMORY_BASE}/search`,
    request
  );
}

// =============================================================================
// EPISODIC MEMORY ENDPOINTS
// =============================================================================

/**
 * Create a new episodic memory entry.
 *
 * Episodic memories capture specific events, interactions, or observations
 * with optional context like session, agent, brand, and region.
 *
 * @param input - Episodic memory content and metadata
 * @returns Created episodic memory with generated ID
 *
 * @example
 * ```typescript
 * const memory = await createEpisodicMemory({
 *   content: 'Dr. Smith expressed interest in Kisqali efficacy data.',
 *   event_type: 'hcp_interaction',
 *   session_id: 'sess_abc123',
 *   brand: 'Kisqali',
 *   region: 'northeast'
 * });
 * ```
 */
export async function createEpisodicMemory(
  input: EpisodicMemoryInput
): Promise<EpisodicMemoryResponse> {
  return post<EpisodicMemoryResponse, EpisodicMemoryInput>(
    `${MEMORY_BASE}/episodic`,
    input
  );
}

/**
 * Get episodic memories with optional filters.
 *
 * @param params - Query parameters for filtering episodic memories
 * @returns List of episodic memories
 *
 * @example
 * ```typescript
 * const memories = await getEpisodicMemories({
 *   session_id: 'sess_abc123',
 *   limit: 20
 * });
 * ```
 */
export async function getEpisodicMemories(
  params?: Record<string, unknown>
): Promise<EpisodicMemoryResponse[]> {
  return get<EpisodicMemoryResponse[]>(`${MEMORY_BASE}/episodic`, params);
}

/**
 * Get a specific episodic memory by ID.
 *
 * @param memoryId - Unique episodic memory identifier
 * @returns Episodic memory details
 * @throws {ApiError} When memory is not found (404)
 *
 * @example
 * ```typescript
 * const memory = await getEpisodicMemory('mem_12345');
 * ```
 */
export async function getEpisodicMemory(
  memoryId: string
): Promise<EpisodicMemoryResponse> {
  return get<EpisodicMemoryResponse>(
    `${MEMORY_BASE}/episodic/${encodeURIComponent(memoryId)}`
  );
}

// =============================================================================
// PROCEDURAL MEMORY ENDPOINTS
// =============================================================================

/**
 * Record feedback for a procedural memory (success/failure outcome).
 *
 * Updates the procedure's success rate based on the outcome.
 *
 * @param request - Feedback including procedure ID, outcome, and score
 * @returns Updated procedural memory with new success rate
 *
 * @example
 * ```typescript
 * const result = await recordProceduralFeedback({
 *   procedure_id: 'proc_hcp_outreach',
 *   outcome: 'success',
 *   score: 0.95,
 *   feedback_text: 'HCP responded positively'
 * });
 * ```
 */
export async function recordProceduralFeedback(
  request: ProceduralFeedbackRequest
): Promise<ProceduralFeedbackResponse> {
  return post<ProceduralFeedbackResponse, ProceduralFeedbackRequest>(
    `${MEMORY_BASE}/procedural/feedback`,
    request
  );
}

// =============================================================================
// SEMANTIC MEMORY ENDPOINTS
// =============================================================================

/**
 * Query causal paths in the semantic memory graph.
 *
 * Finds relationship paths between entities or from entities to KPIs.
 *
 * @param request - Path query parameters
 * @returns Discovered paths with traversal statistics
 *
 * @example
 * ```typescript
 * const paths = await querySemanticPaths({
 *   kpi_name: 'TRx',
 *   max_depth: 3,
 *   min_confidence: 0.6
 * });
 * ```
 */
export async function querySemanticPaths(
  request: SemanticPathRequest
): Promise<SemanticPathResponse> {
  return post<SemanticPathResponse, SemanticPathRequest>(
    `${MEMORY_BASE}/semantic/paths`,
    request
  );
}

// =============================================================================
// STATS ENDPOINT
// =============================================================================

/**
 * Get memory system statistics.
 *
 * Returns counts and metrics for episodic, procedural, and semantic memory.
 *
 * @returns Memory statistics summary
 *
 * @example
 * ```typescript
 * const stats = await getMemoryStats();
 * console.log(`Total episodic memories: ${stats.episodic.total_memories}`);
 * ```
 */
export async function getMemoryStats(): Promise<MemoryStatsResponse> {
  return get<MemoryStatsResponse>(`${MEMORY_BASE}/stats`);
}
