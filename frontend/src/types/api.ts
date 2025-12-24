/**
 * Common API Types
 * =================
 *
 * Shared types and interfaces used across all API modules.
 * Includes common response patterns, pagination, and utility types.
 *
 * @module types/api
 */

// =============================================================================
// COMMON ENUMS
// =============================================================================

/**
 * Sort order for list queries
 */
export enum SortOrder {
  ASC = 'asc',
  DESC = 'desc',
}

// =============================================================================
// PAGINATION TYPES
// =============================================================================

/**
 * Standard pagination parameters for list requests
 */
export interface PaginationParams {
  /** Number of items to return (1-500, default 50) */
  limit?: number;
  /** Number of items to skip */
  offset?: number;
}

/**
 * Standard paginated response metadata
 */
export interface PaginatedResponse {
  /** Applied limit */
  limit: number;
  /** Applied offset */
  offset: number;
  /** Total number of matching items */
  total_count: number;
  /** Whether more results exist beyond current page */
  has_more: boolean;
}

// =============================================================================
// TIMESTAMP TYPES
// =============================================================================

/**
 * Common timestamp fields for entities
 */
export interface TimestampFields {
  /** Creation timestamp (ISO 8601) */
  created_at?: string;
  /** Last update timestamp (ISO 8601) */
  updated_at?: string;
}

/**
 * Response with timestamp
 */
export interface TimestampedResponse {
  /** Response timestamp (ISO 8601) */
  timestamp: string;
}

// =============================================================================
// QUERY PERFORMANCE TYPES
// =============================================================================

/**
 * Query performance metadata
 */
export interface QueryLatency {
  /** Query execution time in milliseconds */
  query_latency_ms: number;
}

/**
 * Processing performance metadata
 */
export interface ProcessingLatency {
  /** Total processing time in milliseconds */
  processing_time_ms: number;
}

// =============================================================================
// ERROR TYPES
// =============================================================================

/**
 * Standard API error response from backend
 */
export interface ApiErrorDetail {
  /** Error message */
  detail: string;
}

/**
 * Validation error item
 */
export interface ValidationError {
  /** Location of the error (e.g., ["body", "field_name"]) */
  loc: (string | number)[];
  /** Error message */
  msg: string;
  /** Error type */
  type: string;
}

/**
 * Validation error response (FastAPI 422)
 */
export interface ValidationErrorResponse {
  detail: ValidationError[];
}

// =============================================================================
// FILTER TYPES
// =============================================================================

/**
 * Generic filter parameters
 */
export interface FilterParams {
  /** Key-value filters */
  filters?: Record<string, unknown>;
}

/**
 * Search parameters
 */
export interface SearchParams {
  /** Text search query */
  search?: string;
  /** Minimum score threshold (0-1) */
  min_score?: number;
}

// =============================================================================
// METADATA TYPES
// =============================================================================

/**
 * Generic metadata container
 */
export interface Metadata {
  /** Additional key-value metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Properties container (for graph entities)
 */
export interface Properties {
  /** Entity properties */
  properties: Record<string, unknown>;
}

// =============================================================================
// HEALTH CHECK TYPES
// =============================================================================

/**
 * Service health status
 */
export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy' | 'unknown';

/**
 * Generic health check response
 */
export interface HealthCheckResponse {
  /** Overall status */
  status: HealthStatus;
  /** Timestamp of health check */
  timestamp: string;
  /** Service-specific details */
  [key: string]: unknown;
}

// =============================================================================
// UTILITY TYPES
// =============================================================================

/**
 * Make specific properties optional
 */
export type PartialBy<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

/**
 * Make specific properties required
 */
export type RequiredBy<T, K extends keyof T> = Omit<T, K> & Required<Pick<T, K>>;

/**
 * Extract request type from an endpoint
 */
export type RequestOf<T> = T extends (req: infer R) => unknown ? R : never;

/**
 * Extract response type from an endpoint
 */
export type ResponseOf<T> = T extends () => Promise<infer R> ? R : never;
