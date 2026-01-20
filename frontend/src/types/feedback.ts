/**
 * Feedback Learning API Types
 * ===========================
 *
 * TypeScript interfaces for the E2I Feedback Learning API.
 * Based on src/api/routes/feedback.py backend schemas.
 *
 * Supports:
 * - Feedback collection and processing
 * - Pattern detection from user feedback
 * - Learning recommendations generation
 * - Knowledge updates management
 *
 * @module types/feedback
 */

// =============================================================================
// ENUMS
// =============================================================================

/**
 * Types of user feedback
 */
export enum FeedbackType {
  RATING = 'rating',
  CORRECTION = 'correction',
  OUTCOME = 'outcome',
  EXPLICIT = 'explicit',
}

/**
 * Types of patterns that can be detected
 */
export enum PatternType {
  ACCURACY_ISSUE = 'accuracy_issue',
  LATENCY_ISSUE = 'latency_issue',
  RELEVANCE_ISSUE = 'relevance_issue',
  FORMAT_ISSUE = 'format_issue',
  COVERAGE_GAP = 'coverage_gap',
}

/**
 * Severity levels for detected patterns
 */
export enum PatternSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical',
}

/**
 * Types of knowledge updates
 */
export enum UpdateType {
  PROMPT_REFINEMENT = 'prompt_refinement',
  EXAMPLE_ADDITION = 'example_addition',
  RULE_MODIFICATION = 'rule_modification',
  PARAMETER_TUNING = 'parameter_tuning',
  INDEX_UPDATE = 'index_update',
}

/**
 * Status of knowledge updates
 */
export enum UpdateStatus {
  PROPOSED = 'proposed',
  APPROVED = 'approved',
  APPLIED = 'applied',
  ROLLED_BACK = 'rolled_back',
}

/**
 * Status of a learning cycle
 */
export enum LearningStatus {
  PENDING = 'pending',
  COLLECTING = 'collecting',
  ANALYZING = 'analyzing',
  EXTRACTING = 'extracting',
  UPDATING = 'updating',
  COMPLETED = 'completed',
  FAILED = 'failed',
}

// =============================================================================
// REQUEST TYPES
// =============================================================================

/**
 * Individual feedback item to process
 */
export interface FeedbackItem {
  /** Unique feedback identifier (auto-generated if not provided) */
  feedback_id?: string;
  /** Feedback timestamp (ISO format) */
  timestamp?: string;
  /** Type of feedback */
  feedback_type: FeedbackType;
  /** Agent that generated the original response */
  source_agent: string;
  /** Original user query */
  query: string;
  /** Agent's response to the query */
  agent_response: string;
  /** User's feedback (rating, correction, outcome, etc.) */
  user_feedback: unknown;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Request to run a feedback learning cycle
 */
export interface RunLearningRequest {
  /** Start of time range (ISO format, defaults to last 24h) */
  time_range_start?: string;
  /** End of time range (ISO format, defaults to now) */
  time_range_end?: string;
  /** Specific agents to focus on (all if not specified) */
  focus_agents?: string[];
  /** Minimum feedback items required to proceed */
  min_feedback_count?: number;
  /** Minimum frequency for pattern detection (0-1) */
  pattern_threshold?: number;
  /** Automatically apply approved updates */
  auto_apply?: boolean;
}

/**
 * Request to process specific feedback items
 */
export interface ProcessFeedbackRequest {
  /** Feedback items to process */
  items: FeedbackItem[];
  /** Whether to detect patterns */
  detect_patterns?: boolean;
  /** Whether to generate recommendations */
  generate_recommendations?: boolean;
}

/**
 * Request to apply a knowledge update
 */
export interface ApplyUpdateRequest {
  /** Update identifier to apply */
  update_id: string;
  /** Force apply even if not approved */
  force?: boolean;
}

/**
 * Parameters for listing patterns
 */
export interface ListPatternsParams {
  /** Filter by severity */
  severity?: PatternSeverity;
  /** Filter by type */
  pattern_type?: PatternType;
  /** Filter by affected agent */
  agent?: string;
  /** Maximum results */
  limit?: number;
}

/**
 * Parameters for listing updates
 */
export interface ListUpdatesParams {
  /** Filter by status */
  status?: UpdateStatus;
  /** Filter by type */
  update_type?: UpdateType;
  /** Filter by target agent */
  agent?: string;
  /** Maximum results */
  limit?: number;
}

// =============================================================================
// RESPONSE TYPES
// =============================================================================

/**
 * Pattern detected from feedback analysis
 */
export interface DetectedPattern {
  /** Unique pattern identifier */
  pattern_id: string;
  /** Type of pattern */
  pattern_type: PatternType;
  /** Human-readable description */
  description: string;
  /** Number of occurrences */
  frequency: number;
  /** Impact severity */
  severity: PatternSeverity;
  /** Agents affected by this pattern */
  affected_agents: string[];
  /** Example feedback IDs */
  example_feedback_ids: string[];
  /** Hypothesized root cause */
  root_cause_hypothesis: string;
  /** Detection confidence (0-1) */
  confidence: number;
}

/**
 * Recommendation for system improvement
 */
export interface LearningRecommendation {
  /** Unique recommendation identifier */
  recommendation_id: string;
  /** Pattern this addresses */
  pattern_id: string;
  /** Priority rank (1=highest) */
  priority: number;
  /** Type of recommendation */
  recommendation_type: string;
  /** What should be changed */
  description: string;
  /** Expected improvement */
  expected_impact: string;
  /** Low/Medium/High */
  implementation_effort: string;
  /** Agents to modify */
  affected_agents: string[];
}

/**
 * Proposed or applied knowledge update
 */
export interface KnowledgeUpdate {
  /** Unique update identifier */
  update_id: string;
  /** Type of update */
  update_type: UpdateType;
  /** Current status */
  status: UpdateStatus;
  /** Agent to update */
  target_agent: string;
  /** Component being updated */
  target_component: string;
  /** Current configuration */
  current_value?: string;
  /** Proposed new configuration */
  proposed_value: string;
  /** Why this update is needed */
  rationale: string;
  /** Expected impact */
  expected_improvement: string;
  /** When proposed */
  created_at: string;
  /** When applied */
  applied_at?: string;
}

/**
 * Summary statistics from feedback analysis
 */
export interface FeedbackSummary {
  /** Total items processed */
  total_feedback_items: number;
  /** Count by feedback type */
  by_type: Record<string, number>;
  /** Count by source agent */
  by_agent: Record<string, number>;
  /** Average rating (if applicable) */
  average_rating?: number;
  /** Ratio of positive feedback */
  positive_ratio: number;
  /** Analysis start time */
  time_range_start: string;
  /** Analysis end time */
  time_range_end: string;
}

/**
 * Response from feedback learning cycle
 */
export interface LearningResponse {
  /** Unique batch identifier */
  batch_id: string;
  /** Learning status */
  status: LearningStatus;

  // Results
  /** Patterns detected from feedback */
  detected_patterns: DetectedPattern[];
  /** Improvement recommendations */
  learning_recommendations: LearningRecommendation[];
  /** Top priority items */
  priority_improvements: string[];
  /** Proposed knowledge updates */
  proposed_updates: KnowledgeUpdate[];
  /** Updates that were applied */
  applied_updates: KnowledgeUpdate[];

  // Summary
  /** Executive summary */
  learning_summary: string;
  /** Feedback statistics */
  feedback_summary?: FeedbackSummary;

  // Metrics
  /** Number of patterns found */
  patterns_detected: number;
  /** Number of recommendations */
  recommendations_generated: number;
  /** Number of updates proposed */
  updates_proposed: number;
  /** Number of updates applied */
  updates_applied: number;

  // Metadata
  /** Feedback collection time */
  collection_latency_ms: number;
  /** Analysis time */
  analysis_latency_ms: number;
  /** Total processing time */
  total_latency_ms: number;
  /** Completion timestamp */
  timestamp: string;
  /** Any errors encountered */
  errors: string[];
  /** Warnings */
  warnings: string[];
}

/**
 * Response for listing patterns
 */
export interface PatternListResponse {
  /** Total patterns */
  total_count: number;
  /** Critical severity count */
  critical_count: number;
  /** High severity count */
  high_count: number;
  /** List of patterns */
  patterns: DetectedPattern[];
}

/**
 * Response for listing knowledge updates
 */
export interface UpdateListResponse {
  /** Total updates */
  total_count: number;
  /** Pending approval */
  proposed_count: number;
  /** Already applied */
  applied_count: number;
  /** List of updates */
  updates: KnowledgeUpdate[];
}

/**
 * Health check response for feedback learning service
 */
export interface FeedbackHealthResponse {
  /** Service status */
  status: string;
  /** Feedback Learner agent status */
  agent_available: boolean;
  /** Last learning cycle timestamp */
  last_learning_cycle?: string;
  /** Learning cycles in last 24 hours */
  cycles_24h: number;
  /** Active patterns being tracked */
  patterns_active: number;
  /** Updates pending approval */
  pending_updates: number;
}
