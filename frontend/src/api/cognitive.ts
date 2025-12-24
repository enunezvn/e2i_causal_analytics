/**
 * Cognitive API Client
 * ====================
 *
 * TypeScript API client functions for the E2I Cognitive Workflow endpoints.
 * Uses the shared apiClient for consistent error handling and interceptors.
 *
 * Endpoints:
 * - Cognitive query processing
 * - Session management
 * - Cognitive RAG search
 *
 * @module api/cognitive
 */

import { get, post, del } from '@/lib/api-client';
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

// =============================================================================
// COGNITIVE API ENDPOINTS
// =============================================================================

const COGNITIVE_BASE = '/cognitive';

// =============================================================================
// QUERY ENDPOINTS
// =============================================================================

/**
 * Process a cognitive query through the full workflow.
 *
 * Routes the query through appropriate agents (causal, prediction,
 * optimization, monitoring, explanation) based on detected intent.
 *
 * @param request - Query with context and options
 * @returns Response with evidence trail and processing details
 *
 * @example
 * ```typescript
 * const response = await processCognitiveQuery({
 *   query: 'What factors are driving TRx decline in the Northeast?',
 *   brand: 'Kisqali',
 *   include_evidence: true
 * });
 * ```
 */
export async function processCognitiveQuery(
  request: CognitiveQueryRequest
): Promise<CognitiveQueryResponse> {
  return post<CognitiveQueryResponse, CognitiveQueryRequest>(
    `${COGNITIVE_BASE}/query`,
    request
  );
}

/**
 * Get the status of the cognitive service.
 *
 * @returns Service status and version information
 *
 * @example
 * ```typescript
 * const status = await getCognitiveStatus();
 * console.log(`Cognitive service version: ${status.version}`);
 * ```
 */
export async function getCognitiveStatus(): Promise<{
  status: string;
  version: string;
  agents: string[];
}> {
  return get<{
    status: string;
    version: string;
    agents: string[];
  }>(`${COGNITIVE_BASE}/status`);
}

// =============================================================================
// SESSION ENDPOINTS
// =============================================================================

/**
 * Create a new cognitive session.
 *
 * Sessions maintain conversation context and working memory
 * across multiple query interactions.
 *
 * @param request - Session initialization parameters
 * @returns Created session with ID and expiration
 *
 * @example
 * ```typescript
 * const session = await createSession({
 *   user_id: 'user_abc',
 *   brand: 'Kisqali',
 *   region: 'northeast'
 * });
 * ```
 */
export async function createSession(
  request: CreateSessionRequest
): Promise<CreateSessionResponse> {
  return post<CreateSessionResponse, CreateSessionRequest>(
    `${COGNITIVE_BASE}/sessions`,
    request
  );
}

/**
 * Get the current state of a cognitive session.
 *
 * Returns session context, message history, and evidence trail.
 *
 * @param sessionId - Session identifier
 * @returns Full session state
 * @throws {ApiError} When session is not found (404)
 *
 * @example
 * ```typescript
 * const session = await getSession('sess_abc123');
 * console.log(`Messages: ${session.messages.length}`);
 * ```
 */
export async function getSession(sessionId: string): Promise<SessionResponse> {
  return get<SessionResponse>(
    `${COGNITIVE_BASE}/sessions/${encodeURIComponent(sessionId)}`
  );
}

/**
 * Delete a cognitive session.
 *
 * Cleans up session state and releases resources.
 *
 * @param sessionId - Session identifier to delete
 * @returns Deletion confirmation
 * @throws {ApiError} When session is not found (404)
 *
 * @example
 * ```typescript
 * const result = await deleteSession('sess_abc123');
 * console.log(`Deleted: ${result.deleted}`);
 * ```
 */
export async function deleteSession(
  sessionId: string
): Promise<DeleteSessionResponse> {
  return del<DeleteSessionResponse>(
    `${COGNITIVE_BASE}/sessions/${encodeURIComponent(sessionId)}`
  );
}

/**
 * List all active cognitive sessions.
 *
 * @param params - Optional query parameters for filtering
 * @returns List of active sessions
 *
 * @example
 * ```typescript
 * const sessions = await listSessions({ user_id: 'user_abc' });
 * ```
 */
export async function listSessions(
  params?: Record<string, unknown>
): Promise<{ sessions: SessionResponse[]; total: number }> {
  return get<{ sessions: SessionResponse[]; total: number }>(
    `${COGNITIVE_BASE}/sessions`,
    params
  );
}

// =============================================================================
// COGNITIVE RAG ENDPOINTS
// =============================================================================

/**
 * Perform DSPy-enhanced cognitive RAG search.
 *
 * Uses multi-hop retrieval with automatic query rewriting and
 * agent routing based on detected intent.
 *
 * @param request - RAG query with optional conversation context
 * @returns Synthesized response with evidence and routing info
 *
 * @example
 * ```typescript
 * const result = await cognitiveRAGSearch({
 *   query: 'What is driving the TRx trend for Kisqali?',
 *   conversation_id: 'conv_123'
 * });
 * console.log(`Routed to agents: ${result.routed_agents.join(', ')}`);
 * ```
 */
export async function cognitiveRAGSearch(
  request: CognitiveRAGRequest
): Promise<CognitiveRAGResponse> {
  return post<CognitiveRAGResponse, CognitiveRAGRequest>(
    `${COGNITIVE_BASE}/rag`,
    request
  );
}
