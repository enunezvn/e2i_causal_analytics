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
  /** Could not evaluate (missing data or calculation error) */
  UNKNOWN = 'unknown',
  /** No target by design — volume/causal metrics tracked for trend/context only */
  INFORMATIONAL = 'informational',
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
  /** Filter by geographic region (northeast/south/midwest/west; matched
   *  case-insensitively). Routes KPIs to their region-scoped query variants. */
  region?: string;
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
  /** Target threshold value (monotone mode) */
  target?: number;
  /** Warning threshold value (monotone mode) */
  warning?: number;
  /** Critical threshold value (monotone mode) */
  critical?: number;
  /** Ideal value for deviation-from-ideal KPIs (band mode, e.g. calibration slope) */
  ideal?: number;
  /** GOOD when abs(value - ideal) <= this (band mode) */
  good_tolerance?: number;
  /** WARNING when abs(value - ideal) <= this; CRITICAL beyond (band mode) */
  warning_tolerance?: number;
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
  /** Display-format hint: 'percent' = value is a 0-1 ratio rendered as NN.N%. */
  value_format?: string;
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
  /** Provenance: 'database' = real (synthetic-excluded) rows; 'synthetic' =
   *  computed over synthetic-gold rows in demo/review mode (badged in the UI). */
  data_source?: string;
  /** Region the caller asked for (#1538); absent when none was requested. */
  region_requested?: string | null;
  /** Region the value was actually computed for; null/absent when the
   *  calculator has no region variant (the value is global/portfolio). */
  region_applied?: string | null;
  /** 'default' = no region requested; 'applied' = a region-scoped variant
   *  computed this value; 'not_applicable' = region requested but NOT applied
   *  — the value is global and must not be captioned with the region.
   *  Absent entirely on pre-#1538 backends. */
  region_status?: string;
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

// =============================================================================
// KPI HISTORY (time-series KPI-history view)
// =============================================================================

/** One materialized monthly KPI value. */
export interface KPIHistoryPoint {
  /** Month (YYYY-MM-DD, first of month). */
  metric_date: string;
  value: number;
  status?: string | null;
}

/** Date-ordered KPI history for one KPI (empty when no real series exists). */
export interface KPIHistoryResponse {
  kpi_id: string;
  /** '' = global / all brands. */
  brand: string;
  /** '' = all regions. */
  region: string;
  count: number;
  points: KPIHistoryPoint[];
}

/** Patient axis a segmented history can be split by. */
export type KPIHistoryAxis = 'segment' | 'therapy_line';

/** One axis bucket's monthly series (e.g. the high-severity tier). */
export interface KPIHistorySegmentSeries {
  /** Bucket key (e.g. 'high_severity', or '2' for LOT). */
  key: string;
  /** Display label (e.g. 'High severity', '2 prior lines'). */
  label: string;
  count: number;
  points: KPIHistoryPoint[];
}

/**
 * Per-axis-bucket monthly history for one KPI, computed live (migration 110).
 * Bucket series partition the headline series month by month; only the
 * Rx-volume family (TRx/NRx/NBRx) supports axes.
 */
export interface KPISegmentedHistoryResponse {
  kpi_id: string;
  /** '' = global / all brands. */
  brand: string;
  axis: KPIHistoryAxis;
  /** Latest prescription event date backing the series (frontier). */
  data_through?: string | null;
  count: number;
  series: KPIHistorySegmentSeries[];
}

// =============================================================================
// KPI CLAIMS-LAG NOWCAST (backlog #45 — Rx-volume family only)
// =============================================================================

/**
 * One monthly point of the claims-lag nowcast overlay.
 *
 * Mirrors the generated `KPINowcastPoint` schema (types/generated/api.ts) —
 * a DEDICATED model server-side so the shared `/history` point stays
 * byte-untouched.
 */
export interface KPINowcastPoint {
  /** Service month (YYYY-MM-DD, first of month). */
  metric_date: string;
  /** The base KPI value over ALL events (the eventual truth; matches /history). */
  mature_value: number;
  /** Events whose claim_available_date <= frontier (the as-of under-count). */
  provisional_value: number;
  /** True while the month's claims are still maturing (not fully arrived). */
  provisional: boolean;
  /** Estimated fraction of the month's claims arrived as of the frontier
   *  (empirical chain-ladder CF; null when younger than the lag support). */
  completion_factor?: number | null;
  /** provisional_value / completion_factor (the grossed-up estimate). */
  nowcast_value?: number | null;
  /** Bootstrap CI lower bound (provisional months only). */
  nowcast_ci_lower?: number | null;
  /** Bootstrap CI upper bound (provisional months only). */
  nowcast_ci_upper?: number | null;
}

/**
 * Claims-lag provisional/nowcast monthly series for one Rx-volume KPI
 * (WS3-BI-005 TRx / WS3-BI-006 NRx / WS3-BI-007 NBRx; other KPIs 422).
 *
 * When the completion curve cannot be estimated honestly,
 * `insufficient_maturity` is true, `reason` says why and `points` is EMPTY —
 * never a fabricated fallback completion factor.
 */
export interface KPINowcastHistoryResponse {
  kpi_id: string;
  /** '' = global / all brands. */
  brand: string;
  /** Prescription frontier (max event_date) backing the as-of view. */
  data_through?: string | null;
  /** True when no honest completion curve could be estimated (see reason). */
  insufficient_maturity: boolean;
  /** no_data | arrival_plane_not_populated | arrival_plane_partial: <months>
   *  | insufficient_mature_months | no_arrived_claims | null. */
  reason?: string | null;
  /** Mature service months backing the completion curve. */
  mature_months_used: number;
  /** Frontier month excluded from estimation and output (anchor-cap pile-up). */
  anchor_cap_month?: string | null;
  /** Share of events carrying claim_available_date (1.0 = fully stamped). */
  arrival_plane_coverage?: number | null;
  /** Nominal bootstrap CI level. */
  ci_level: number;
  count: number;
  points: KPINowcastPoint[];
}

/** One (brand, region) scope of a KPI's history (migration 126 lattice). */
export interface KPIHistoryScopeEntry {
  /** '' = global / all brands. */
  brand: string;
  /** '' = all regions. */
  region: string;
  points: number;
  first_date?: string | null;
  last_date?: string | null;
}

/** History coverage for one KPI: which scopes have a real series.
 *
 * `brands`/`points`/dates describe the BRAND axis (region='' rows only —
 * unchanged semantics); `scopes` is the full (brand, region) lattice — the
 * authority on which region series exist per brand (#1536). */
export interface KPIHistoryCoverageEntry {
  kpi_id: string;
  /** Brand scopes with points; '' = global. Per-brand-only KPIs have no ''. */
  brands: string[];
  points: number;
  first_date?: string | null;
  last_date?: string | null;
  /** Full (brand, region) scope lattice, sorted by (brand, region). */
  scopes?: KPIHistoryScopeEntry[];
}

/** Coverage map for the registry — KPIs absent here have NO history. */
export interface KPIHistoryCoverageResponse {
  coverage: KPIHistoryCoverageEntry[];
  total: number;
}
