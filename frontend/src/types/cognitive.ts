/**
 * Cognitive Workflow Types
 * ========================
 *
 * TypeScript interfaces for the E2I Cognitive Workflow API.
 * Based on src/api/routes/cognitive.py backend schemas.
 *
 * @module types/cognitive
 */

import type {
  Metadata,
  TimestampedResponse,
} from './api';

// =============================================================================
// ENUMS
// =============================================================================

/**
 * Types of cognitive queries
 */
export enum QueryType {
  /** Causal inference questions */
  CAUSAL = 'causal',
  /** ML prediction requests */
  PREDICTION = 'prediction',
  /** Resource optimization */
  OPTIMIZATION = 'optimization',
  /** Health/drift monitoring */
  MONITORING = 'monitoring',
  /** Explainability requests */
  EXPLANATION = 'explanation',
  /** General analytics */
  GENERAL = 'general',
}

/**
 * Session states
 */
export enum SessionState {
  ACTIVE = 'active',
  COMPLETED = 'completed',
  FAILED = 'failed',
  TIMEOUT = 'timeout',
}

/**
 * Phases of the cognitive workflow
 */
export enum CognitivePhase {
  SUMMARIZE = 'summarize',
  INVESTIGATE = 'investigate',
  EXECUTE = 'execute',
  REFLECT = 'reflect',
  COMPLETE = 'complete',
}

// =============================================================================
// QUERY MODELS
// =============================================================================

/**
 * Request for full cognitive query processing
 */
export interface CognitiveQueryRequest extends Metadata {
  /** User query (1-5000 chars) */
  query: string;
  /** Existing session ID to continue */
  session_id?: string;
  /** User identifier */
  user_id?: string;
  /** Brand context (Kisqali, Fabhalta, Remibrutinib) */
  brand?: string;
  /** Region context */
  region?: string;
  /** Type of query (auto-detected if not specified) */
  query_type?: QueryType;
  /** Include evidence trail in response (default true) */
  include_evidence?: boolean;
  /** Max memory results to retrieve (1-50, default 10) */
  max_memory_results?: number;
}

/**
 * Single piece of evidence from memory retrieval
 */
export interface EvidenceItem {
  /** Evidence content */
  content: string;
  /** Memory source */
  source: string;
  /** Relevance score (0-1) */
  relevance_score: number;
  /** How it was retrieved */
  retrieval_method: string;
}

/**
 * Response from cognitive query processing
 */
export interface CognitiveQueryResponse extends Metadata, TimestampedResponse {
  /** Session identifier */
  session_id: string;
  /** Original query */
  query: string;
  /** Generated response */
  response: string;
  /** Detected or specified query type */
  query_type: QueryType;
  /** Response confidence (0-1) */
  confidence: number;
  /** Primary agent that handled the query */
  agent_used: string;
  /** Evidence trail */
  evidence?: EvidenceItem[];
  /** Workflow phases completed */
  phases_completed: CognitivePhase[];
  /** Total processing time in ms */
  processing_time_ms: number;
}

// =============================================================================
// SESSION MODELS
// =============================================================================

/**
 * Current session context
 */
export interface SessionContext {
  /** Session identifier */
  session_id: string;
  /** User identifier */
  user_id?: string;
  /** Brand context */
  brand?: string;
  /** Region context */
  region?: string;
  /** Current state */
  state: SessionState;
  /** Creation timestamp (ISO 8601) */
  created_at: string;
  /** Last activity timestamp (ISO 8601) */
  last_activity: string;
  /** Number of messages */
  message_count: number;
  /** Current workflow phase */
  current_phase?: CognitivePhase;
}

/**
 * Message in session history
 */
export interface SessionMessage extends Metadata {
  /** Message role (user, assistant, system) */
  role: 'user' | 'assistant' | 'system';
  /** Message content */
  content: string;
  /** Timestamp (ISO 8601) */
  timestamp: string;
  /** Agent name (for assistant messages) */
  agent_name?: string;
}

/**
 * Full session state response
 */
export interface SessionResponse {
  /** Session context */
  context: SessionContext;
  /** Message history */
  messages: SessionMessage[];
  /** Accumulated evidence */
  evidence_trail: EvidenceItem[];
  /** Memory retrieval stats */
  memory_stats: Record<string, unknown>;
}

/**
 * Request to create a new cognitive session
 */
export interface CreateSessionRequest {
  /** User identifier */
  user_id?: string;
  /** Brand context */
  brand?: string;
  /** Region context */
  region?: string;
  /** Initial context */
  initial_context?: Record<string, unknown>;
}

/**
 * Response for session creation
 */
export interface CreateSessionResponse {
  /** Created session ID */
  session_id: string;
  /** Initial state */
  state: SessionState;
  /** Creation timestamp (ISO 8601) */
  created_at: string;
  /** Session expiration time (ISO 8601) */
  expires_at: string;
}

/**
 * Response for session deletion
 */
export interface DeleteSessionResponse {
  /** Session ID that was deleted */
  session_id: string;
  /** Whether deletion was successful */
  deleted: boolean;
  /** Timestamp (ISO 8601) */
  timestamp: string;
}

// =============================================================================
// COGNITIVE RAG MODELS
// =============================================================================

/**
 * Request for DSPy-enhanced cognitive RAG search
 */
export interface CognitiveRAGRequest {
  /** Natural language query (1-5000 chars) */
  query: string;
  /** Conversation/session ID for context continuity */
  conversation_id?: string;
  /** Compressed conversation history */
  conversation_history?: string;
}

/**
 * DSPy training signal
 */
export interface DSPySignal {
  /** Signal type */
  type: string;
  /** Signal value */
  value: unknown;
  /** Additional context */
  context?: Record<string, unknown>;
}

/**
 * Response from DSPy-enhanced cognitive RAG search
 */
export interface CognitiveRAGResponse {
  /** Synthesized natural language response */
  response: string;
  /** Evidence pieces gathered */
  evidence: Array<{
    content: string;
    source: string;
    score?: number;
    metadata?: Record<string, unknown>;
  }>;
  /** Number of retrieval hops performed */
  hop_count: number;
  /** Chart configuration if applicable */
  visualization_config: Record<string, unknown>;
  /** Agents recommended for further processing */
  routed_agents: string[];
  /** Extracted entities */
  entities: string[];
  /** Detected query intent */
  intent: string;
  /** DSPy-optimized query rewrite */
  rewritten_query: string;
  /** Training signals for optimization */
  dspy_signals: DSPySignal[];
  /** Whether this exchange should be stored in long-term memory */
  worth_remembering: boolean;
  /** Total processing time in milliseconds */
  latency_ms: number;
  /** Error message if processing failed */
  error?: string;
}
