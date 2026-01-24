/**
 * KPI System Types
 * ================
 *
 * TypeScript interfaces for the E2I KPI API.
 * Based on src/api/schemas/kpi.py and src/kpi/models.py backend schemas.
 *
 * @module types/kpi
 */

import { CausalLibrary } from './causal';

// Re-export for consumers that import from kpi
export { CausalLibrary };

// =============================================================================
// ENUMS
// =============================================================================

/**
 * KPI workstreams
 */
export enum Workstream {
  WS1_DATA_QUALITY = 'ws1_data_quality',
  WS1_MODEL_PERFORMANCE = 'ws1_model_performance',
  WS2_TRIGGERS = 'ws2_triggers',
  WS3_BUSINESS = 'ws3_business',
  BRAND_SPECIFIC = 'brand_specific',
  CAUSAL_METRICS = 'causal_metrics',
}

/**
 * How the KPI is calculated
 */
export enum CalculationType {
  DIRECT = 'direct',
  DERIVED = 'derived',
}

/**
 * Status of KPI against thresholds
 */
export enum KPIStatus {
  GOOD = 'good',
  WARNING = 'warning',
  CRITICAL = 'critical',
  UNKNOWN = 'unknown',
}

// =============================================================================
// REQUEST MODELS
// =============================================================================

/**
 * Context for KPI calculation
 */
export interface KPICalculationContext {
  /** Filter by brand (remibrutinib, fabhalta, kisqali) */
  brand?: string;
  /** Start date for time-based calculations (ISO 8601) */
  start_date?: string;
  /** End date for time-based calculations (ISO 8601) */
  end_date?: string;
  /** Territory filter */
  territory?: string;
  /** Customer segment filter */
  segment?: string;
  /** Additional context parameters */
  extra?: Record<string, unknown>;
}

/**
 * Request schema for calculating a single KPI
 */
export interface KPICalculationRequest {
  /** KPI identifier (e.g., WS1-DQ-001) */
  kpi_id: string;
  /** Whether to use cached results if available (default: true) */
  use_cache?: boolean;
  /** Force recalculation even if cached (default: false) */
  force_refresh?: boolean;
  /** Calculation context (filters, date range, etc.) */
  context?: KPICalculationContext;
}

/**
 * Request schema for batch KPI calculation
 */
export interface BatchKPICalculationRequest {
  /** List of specific KPI IDs to calculate */
  kpi_ids?: string[];
  /** Calculate all KPIs for this workstream */
  workstream?: string;
  /** Whether to use cached results if available (default: true) */
  use_cache?: boolean;
  /** Calculation context for all KPIs */
  context?: KPICalculationContext;
}

/**
 * Request schema for cache invalidation
 */
export interface CacheInvalidationRequest {
  /** Specific KPI ID to invalidate */
  kpi_id?: string;
  /** Invalidate all KPIs for this workstream */
  workstream?: string;
  /** Invalidate all cached KPIs (use with caution) */
  invalidate_all?: boolean;
}

/**
 * Parameters for listing KPIs
 */
export interface KPIListParams {
  /** Filter by workstream */
  workstream?: Workstream | string;
  /** Filter by causal library */
  causal_library?: CausalLibrary | string;
}

// =============================================================================
// RESPONSE MODELS
// =============================================================================

/**
 * KPI threshold configuration
 */
export interface KPIThreshold {
  /** Target threshold value */
  target?: number;
  /** Warning threshold value */
  warning?: number;
  /** Critical threshold value */
  critical?: number;
}

/**
 * KPI metadata/definition
 */
export interface KPIMetadata {
  /** KPI identifier */
  id: string;
  /** Human-readable KPI name */
  name: string;
  /** KPI definition/description */
  definition: string;
  /** Calculation formula */
  formula: string;
  /** Direct or derived calculation type */
  calculation_type: string;
  /** Workstream this KPI belongs to */
  workstream: string;
  /** Source database tables */
  tables: string[];
  /** Source columns */
  columns: string[];
  /** Database view name if applicable */
  view?: string;
  /** Threshold configuration */
  threshold?: KPIThreshold;
  /** Unit of measurement */
  unit?: string;
  /** Calculation frequency (e.g., 'daily') */
  frequency: string;
  /** Primary causal library for this KPI */
  primary_causal_library: string;
  /** Brand filter if applicable */
  brand?: string;
  /** Additional notes */
  note?: string;
}

/**
 * Result of a single KPI calculation
 */
export interface KPIResult {
  /** KPI identifier */
  kpi_id: string;
  /** Calculated KPI value */
  value?: number;
  /** Status against thresholds */
  status: KPIStatus | string;
  /** Calculation timestamp (ISO 8601) */
  calculated_at: string;
  /** Whether result was from cache */
  cached: boolean;
  /** When cache entry expires (ISO 8601) */
  cache_expires_at?: string;
  /** Error message if calculation failed */
  error?: string;
  /** Causal library used for calculation */
  causal_library_used?: string;
  /** 95% confidence interval [lower, upper] */
  confidence_interval?: [number, number];
  /** Statistical p-value */
  p_value?: number;
  /** Effect size if applicable */
  effect_size?: number;
  /** Additional calculation metadata */
  metadata: Record<string, unknown>;
}

/**
 * Response for batch KPI calculation
 */
export interface BatchKPICalculationResponse {
  /** Workstream if specified */
  workstream?: string;
  /** List of KPI results */
  results: KPIResult[];
  /** Batch calculation timestamp (ISO 8601) */
  calculated_at: string;
  /** Total number of KPIs requested */
  total_kpis: number;
  /** Number of successful calculations */
  successful: number;
  /** Number of failed calculations */
  failed: number;
}

/**
 * Response for listing KPIs
 */
export interface KPIListResponse {
  /** List of KPI metadata */
  kpis: KPIMetadata[];
  /** Total number of KPIs */
  total: number;
  /** Filtered workstream if any */
  workstream?: string;
  /** Filtered causal library if any */
  causal_library?: string;
}

/**
 * Information about a workstream
 */
export interface WorkstreamInfo {
  /** Workstream identifier */
  id: string;
  /** Human-readable workstream name */
  name: string;
  /** Number of KPIs in this workstream */
  kpi_count: number;
  /** Workstream description */
  description?: string;
}

/**
 * Response for listing workstreams
 */
export interface WorkstreamListResponse {
  /** List of workstreams */
  workstreams: WorkstreamInfo[];
  /** Total number of workstreams */
  total: number;
}

/**
 * Response for cache invalidation
 */
export interface CacheInvalidationResponse {
  /** Number of cache entries invalidated */
  invalidated_count: number;
  /** Status message */
  message: string;
}

/**
 * Response for KPI system health
 */
export interface KPIHealthResponse {
  /** Overall health status: healthy, degraded, unhealthy */
  status: 'healthy' | 'degraded' | 'unhealthy';
  /** Whether KPI registry is loaded */
  registry_loaded: boolean;
  /** Total KPIs in registry */
  total_kpis: number;
  /** Whether caching is enabled */
  cache_enabled: boolean;
  /** Current cache size */
  cache_size: number;
  /** Whether database is connected */
  database_connected: boolean;
  /** Available workstreams */
  workstreams_available: string[];
  /** Timestamp of last calculation (ISO 8601) */
  last_calculation?: string;
  /** Error message if unhealthy */
  error?: string;
}
