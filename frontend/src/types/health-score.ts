/**
 * Health Score API Types
 * ======================
 *
 * TypeScript interfaces for the E2I Health Score API.
 * Based on src/api/routes/health_score.py backend schemas.
 *
 * Supports:
 * - System health monitoring
 * - Component, model, pipeline, and agent health
 * - Health scoring and grading
 * - Historical health trends
 *
 * @module types/health-score
 */

// =============================================================================
// ENUMS
// =============================================================================

/**
 * Scope of health check
 */
export enum CheckScope {
  FULL = 'full',
  QUICK = 'quick',
  MODELS = 'models',
  PIPELINES = 'pipelines',
  AGENTS = 'agents',
}

/**
 * Status of a system component
 */
export enum ComponentStatus {
  HEALTHY = 'healthy',
  DEGRADED = 'degraded',
  UNHEALTHY = 'unhealthy',
  UNKNOWN = 'unknown',
}

/**
 * Status of a model
 */
export enum ModelStatus {
  HEALTHY = 'healthy',
  DEGRADED = 'degraded',
  UNHEALTHY = 'unhealthy',
}

/**
 * Status of a data pipeline
 */
export enum PipelineStatus {
  HEALTHY = 'healthy',
  STALE = 'stale',
  FAILED = 'failed',
}

/**
 * Health letter grade
 */
export enum HealthGrade {
  A = 'A',
  B = 'B',
  C = 'C',
  D = 'D',
  F = 'F',
}

// =============================================================================
// COMPONENT TYPES
// =============================================================================

/**
 * Status of a system component
 */
export interface ComponentHealth {
  /** Component identifier */
  component_name: string;
  /** Component status */
  status: ComponentStatus;
  /** Check latency in ms */
  latency_ms?: number;
  /** Last check timestamp */
  last_check: string;
  /** Error message if unhealthy */
  error_message?: string;
  /** Additional details */
  details?: Record<string, unknown>;
}

/**
 * Model performance metrics
 */
export interface ModelHealth {
  /** Model identifier */
  model_id: string;
  /** Model display name */
  model_name: string;
  /** Model accuracy */
  accuracy?: number;
  /** Model precision */
  precision?: number;
  /** Model recall */
  recall?: number;
  /** Model F1 score */
  f1_score?: number;
  /** AUC-ROC score */
  auc_roc?: number;
  /** 50th percentile prediction latency */
  prediction_latency_p50_ms?: number;
  /** 99th percentile prediction latency */
  prediction_latency_p99_ms?: number;
  /** Predictions in last 24 hours; null when unmeasured */
  predictions_last_24h: number | null;
  /** Error rate (0-1); null when unmeasured */
  error_rate: number | null;
  /** Model health status */
  status: ModelStatus;
}

/**
 * Data pipeline status
 */
export interface PipelineHealth {
  /** Pipeline identifier */
  pipeline_name: string;
  /** Last run timestamp */
  last_run: string;
  /** Last successful run timestamp */
  last_success: string;
  /** Rows processed in last run */
  rows_processed: number;
  /** Data freshness in hours */
  freshness_hours: number;
  /** Pipeline status */
  status: PipelineStatus;
}

/**
 * Agent availability status
 */
export interface AgentHealth {
  /** Agent identifier */
  agent_name: string;
  /** Agent tier (0-5) */
  tier: number;
  /** Whether agent is available */
  available: boolean;
  /** Average response latency; null when unmeasured (provenance "partial") */
  avg_latency_ms: number | null;
  /** Success rate (0-1); null when unmeasured (provenance "partial") */
  success_rate: number | null;
  /** Last invocation timestamp */
  last_invocation?: string;
  /** Invocations in last 24 hours */
  invocations_24h: number;
}

// =============================================================================
// RESPONSE TYPES
// =============================================================================

/**
 * Response from health check
 */
export interface HealthScoreResponse {
  /** Unique check identifier */
  check_id: string;
  /** Scope of this check */
  check_scope: CheckScope;

  // Overall score
  /** Overall health score (0-100) */
  overall_health_score: number;
  /** Letter grade (A-F) */
  health_grade: HealthGrade;

  // Component scores (0-1)
  /** Component health score */
  component_health_score: number;
  /** Model health score */
  model_health_score: number;
  /** Pipeline health score */
  pipeline_health_score: number;
  /** Agent health score */
  agent_health_score: number;

  // Details (included based on scope)
  /** Component status details */
  component_statuses?: ComponentHealth[];
  /** Model health details */
  model_metrics?: ModelHealth[];
  /** Pipeline status details */
  pipeline_statuses?: PipelineHealth[];
  /** Agent status details */
  agent_statuses?: AgentHealth[];

  // Issues
  /** Critical issues requiring attention */
  critical_issues: string[];
  /** Non-critical warnings */
  warnings: string[];
  /** Recommended actions */
  recommendations: string[];

  // Summary
  /** Human-readable health summary */
  health_summary: string;

  // Metadata
  /** Check duration in ms */
  check_latency_ms: number;
  /** Check timestamp */
  timestamp: string;
  /** Provenance: measured | partial | unknown | placeholder */
  data_provenance?: string;
}

/**
 * Response for component health check
 */
export interface ComponentHealthResponse {
  /** Aggregate score (0-1) */
  component_health_score: number;
  /** Total components checked */
  total_components: number;
  /** Healthy component count */
  healthy_count: number;
  /** Degraded component count */
  degraded_count: number;
  /** Unhealthy component count */
  unhealthy_count: number;
  /** Component details */
  components: ComponentHealth[];
  /** Check duration */
  check_latency_ms: number;
  /** Provenance: measured | partial | unknown | placeholder */
  data_provenance?: string;
}

/**
 * Response for model health check
 */
export interface ModelHealthResponse {
  /** Aggregate score (0-1) */
  model_health_score: number;
  /** Total models checked */
  total_models: number;
  /** Healthy model count */
  healthy_count: number;
  /** Degraded model count */
  degraded_count: number;
  /** Unhealthy model count */
  unhealthy_count: number;
  /** Model details */
  models: ModelHealth[];
  /** Check duration */
  check_latency_ms: number;
  /** Provenance: measured | partial | unknown | placeholder */
  data_provenance?: string;
}

/**
 * Response for pipeline health check
 */
export interface PipelineHealthResponse {
  /** Aggregate score (0-1) */
  pipeline_health_score: number;
  /** Total pipelines checked */
  total_pipelines: number;
  /** Healthy pipeline count */
  healthy_count: number;
  /** Stale pipeline count */
  stale_count: number;
  /** Failed pipeline count */
  failed_count: number;
  /** Pipeline details */
  pipelines: PipelineHealth[];
  /** Check duration */
  check_latency_ms: number;
  /** Provenance: measured | partial | unknown | placeholder */
  data_provenance?: string;
}

/**
 * Response for agent health check
 */
export interface AgentHealthResponse {
  /** Aggregate score (0-1) */
  agent_health_score: number;
  /** Total agents checked */
  total_agents: number;
  /** Available agent count */
  available_count: number;
  /** Unavailable agent count */
  unavailable_count: number;
  /** Agent details */
  agents: AgentHealth[];
  /** Agent count by tier */
  by_tier: Record<string, number>;
  /** Check duration */
  check_latency_ms: number;
  /** Provenance: measured | partial | unknown | placeholder */
  data_provenance?: string;
}

/**
 * Historical health check record
 */
export interface HealthHistoryItem {
  /** Check identifier */
  check_id: string;
  /** Check timestamp */
  timestamp: string;
  /** Score at time of check */
  overall_health_score: number;
  /** Grade at time of check */
  health_grade: HealthGrade;
  /** Number of critical issues */
  critical_issues_count: number;
  /** Provenance of the recorded score: measured | partial | unknown | placeholder */
  data_provenance?: string;
}

/**
 * One UTC-day aggregate of recorded full health checks (durable history)
 */
export interface HealthHistoryDailyPoint {
  /** UTC day (YYYY-MM-DD) */
  date: string;
  /** Mean overall health score that day */
  avg_score: number;
  /** Lowest overall health score that day */
  min_score: number;
  /** Highest overall health score that day */
  max_score: number;
  /** Recorded checks contributing to the day */
  checks_count: number;
  /** 'measured' only when every contributing check was measured, else 'partial' */
  data_provenance?: string;
}

/**
 * Response for health check history
 */
export interface HealthHistoryResponse {
  /** Total checks in history */
  total_checks: number;
  /** Historical records */
  checks: HealthHistoryItem[];
  /** Average health score */
  avg_health_score: number | null;
  /** Trend direction; 'unknown' when there are too few checks to compute one */
  trend: 'improving' | 'stable' | 'declining' | 'unknown';
  /**
   * Per-UTC-day aggregates over the requested window, ascending. Empty until
   * durable history accumulates (it starts recording at rollout — there is no
   * retroactive backfill).
   */
  daily?: HealthHistoryDailyPoint[];
  /** Days window served from durable history; null on the in-memory fallback */
  window_days?: number | null;
}

/**
 * Service status response
 */
export interface HealthServiceStatus {
  /** Service status */
  status: string;
  /** Health Score agent available */
  agent_available: boolean;
  /** Last health check */
  last_check?: string;
  /** Checks in last 24 hours */
  checks_24h: number;
  /** Average check latency */
  avg_check_latency_ms: number;
}
