/**
 * Audit Chain API Types
 * =====================
 *
 * TypeScript interfaces for the E2I Audit Chain API.
 * Based on src/api/routes/audit.py backend schemas.
 *
 * Supports:
 * - Audit chain entry retrieval
 * - Chain integrity verification
 * - Workflow summary aggregation
 * - Recent workflow listing
 *
 * The audit chain provides cryptographic proof of agent execution
 * sequence and data integrity using hash chains.
 *
 * @module types/audit
 */

// =============================================================================
// AUDIT ENTRY TYPES
// =============================================================================

/**
 * Single audit chain entry response
 */
export interface AuditEntryResponse {
  /** Unique entry identifier (UUID) */
  entry_id: string;
  /** Workflow this entry belongs to (UUID) */
  workflow_id: string;
  /** Sequential position in the chain (1-indexed) */
  sequence_number: number;
  /** Name of the agent that created this entry */
  agent_name: string;
  /** Agent tier (0-5) */
  agent_tier: number;
  /** Type of action performed */
  action_type: string;
  /** Entry creation timestamp (ISO format) */
  created_at: string;
  /** Processing duration in milliseconds */
  duration_ms?: number;

  // Validation & confidence
  /** Whether validation checks passed */
  validation_passed?: boolean;
  /** Confidence score (0-1) */
  confidence_score?: number;
  /** Results from refutation testing */
  refutation_results?: Record<string, unknown>;

  // Hash chain fields
  /** Previous entry in the chain (UUID) */
  previous_entry_id?: string;
  /** Hash of the previous entry */
  previous_hash?: string;
  /** Cryptographic hash of this entry */
  entry_hash: string;

  // Context
  /** User who initiated the workflow */
  user_id?: string;
  /** Session identifier (UUID) */
  session_id?: string;
  /** Brand context (Remibrutinib, Fabhalta, Kisqali) */
  brand?: string;
}

// =============================================================================
// VERIFICATION TYPES
// =============================================================================

/**
 * Chain verification result
 */
export interface ChainVerificationResponse {
  /** Workflow that was verified (UUID) */
  workflow_id: string;
  /** Whether the entire chain is valid */
  is_valid: boolean;
  /** Number of entries checked */
  entries_checked: number;
  /** First invalid entry if chain broken (UUID) */
  first_invalid_entry?: string;
  /** Error message if verification failed */
  error_message?: string;
  /** Timestamp of verification (ISO format) */
  verified_at: string;
}

// =============================================================================
// SUMMARY TYPES
// =============================================================================

/**
 * Workflow summary response
 */
export interface WorkflowSummaryResponse {
  /** Workflow identifier (UUID) */
  workflow_id: string;
  /** Total entries in the workflow */
  total_entries: number;
  /** First entry timestamp (ISO format) */
  first_entry_at?: string;
  /** Last entry timestamp (ISO format) */
  last_entry_at?: string;
  /** List of agents that participated */
  agents_involved: string[];
  /** List of agent tiers involved */
  tiers_involved: number[];
  /** Whether the chain passed verification */
  chain_verified: boolean;
  /** Brand context */
  brand?: string;

  // Aggregated metrics
  /** Total processing time across all entries */
  total_duration_ms: number;
  /** Average confidence score across entries */
  avg_confidence_score?: number;
  /** Count of entries that passed validation */
  validation_passed_count: number;
  /** Count of entries that failed validation */
  validation_failed_count: number;
}

// =============================================================================
// RECENT WORKFLOWS TYPES
// =============================================================================

/**
 * Recent workflow listing item
 */
export interface RecentWorkflowResponse {
  /** Workflow identifier (UUID) */
  workflow_id: string;
  /** Workflow start timestamp (ISO format) */
  started_at: string;
  /** Number of entries in the workflow */
  entry_count: number;
  /** Name of the first agent in the chain */
  first_agent: string;
  /** Name of the last agent in the chain */
  last_agent: string;
  /** Brand context */
  brand?: string;
}

// =============================================================================
// REQUEST/QUERY TYPES
// =============================================================================

/**
 * Parameters for listing workflow entries
 */
export interface ListWorkflowEntriesParams {
  /** Maximum entries to return (1-1000) */
  limit?: number;
  /** Number of entries to skip */
  offset?: number;
}

/**
 * Parameters for listing recent workflows
 */
export interface ListRecentWorkflowsParams {
  /** Maximum workflows to return (1-100) */
  limit?: number;
  /** Filter by brand */
  brand?: string;
  /** Filter by agent name */
  agent_name?: string;
}
