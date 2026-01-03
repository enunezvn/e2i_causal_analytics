/**
 * KPI API Client
 * ==============
 *
 * TypeScript API client functions for the E2I KPI endpoints.
 * Uses the shared apiClient for consistent error handling and interceptors.
 *
 * Endpoints:
 * - List KPIs and workstreams
 * - Calculate individual KPIs
 * - Batch KPI calculations
 * - Cache management
 * - System health
 *
 * @module api/kpi
 */

import { get, post } from '@/lib/api-client';
import type {
  BatchKPICalculationRequest,
  BatchKPICalculationResponse,
  CacheInvalidationRequest,
  CacheInvalidationResponse,
  KPICalculationRequest,
  KPIHealthResponse,
  KPIListParams,
  KPIListResponse,
  KPIMetadata,
  KPIResult,
  WorkstreamListResponse,
} from '@/types/kpi';

// =============================================================================
// KPI API ENDPOINTS
// =============================================================================

const KPI_BASE = '/kpis';

// =============================================================================
// LIST ENDPOINTS
// =============================================================================

/**
 * List all available KPIs with optional filtering.
 *
 * @param params - Optional filters for workstream and causal library
 * @returns List of KPI metadata
 *
 * @example
 * ```typescript
 * const kpis = await listKPIs({ workstream: Workstream.WS1_DATA_QUALITY });
 * console.log(`Found ${kpis.total} KPIs`);
 * ```
 */
export async function listKPIs(params?: KPIListParams): Promise<KPIListResponse> {
  return get<KPIListResponse>(KPI_BASE, {
    workstream: params?.workstream,
    causal_library: params?.causal_library,
  });
}

/**
 * Get list of available workstreams with KPI counts.
 *
 * @returns List of workstreams
 *
 * @example
 * ```typescript
 * const workstreams = await getWorkstreams();
 * workstreams.workstreams.forEach(ws => {
 *   console.log(`${ws.name}: ${ws.kpi_count} KPIs`);
 * });
 * ```
 */
export async function getWorkstreams(): Promise<WorkstreamListResponse> {
  return get<WorkstreamListResponse>(`${KPI_BASE}/workstreams`);
}

// =============================================================================
// METADATA ENDPOINTS
// =============================================================================

/**
 * Get metadata for a specific KPI.
 *
 * @param kpiId - KPI identifier (e.g., WS1-DQ-001)
 * @returns KPI metadata/definition
 *
 * @example
 * ```typescript
 * const metadata = await getKPIMetadata('WS1-DQ-001');
 * console.log(`${metadata.name}: ${metadata.definition}`);
 * ```
 */
export async function getKPIMetadata(kpiId: string): Promise<KPIMetadata> {
  return get<KPIMetadata>(`${KPI_BASE}/${encodeURIComponent(kpiId)}/metadata`);
}

// =============================================================================
// CALCULATION ENDPOINTS
// =============================================================================

/**
 * Get a calculated KPI value.
 *
 * This endpoint returns the calculated value for a KPI with optional filters.
 *
 * @param kpiId - KPI identifier
 * @param brand - Optional brand filter
 * @param useCache - Whether to use cached result (default: true)
 * @returns Calculated KPI result
 *
 * @example
 * ```typescript
 * const result = await getKPIValue('WS1-DQ-001', 'remibrutinib');
 * console.log(`Value: ${result.value}, Status: ${result.status}`);
 * ```
 */
export async function getKPIValue(
  kpiId: string,
  brand?: string,
  useCache: boolean = true
): Promise<KPIResult> {
  return get<KPIResult>(`${KPI_BASE}/${encodeURIComponent(kpiId)}`, {
    brand,
    use_cache: useCache,
  });
}

/**
 * Calculate a specific KPI with full context.
 *
 * @param request - Calculation request with KPI ID and context
 * @returns Calculated KPI result
 *
 * @example
 * ```typescript
 * const result = await calculateKPI({
 *   kpi_id: 'WS1-DQ-001',
 *   context: {
 *     brand: 'remibrutinib',
 *     start_date: '2024-01-01',
 *     end_date: '2024-12-31',
 *   },
 *   force_refresh: true,
 * });
 * ```
 */
export async function calculateKPI(
  request: KPICalculationRequest
): Promise<KPIResult> {
  return post<KPIResult, KPICalculationRequest>(`${KPI_BASE}/calculate`, request);
}

/**
 * Calculate multiple KPIs in batch.
 *
 * Can specify either a list of KPI IDs or a workstream to calculate all KPIs.
 *
 * @param request - Batch calculation request
 * @returns Batch calculation response with all results
 *
 * @example
 * ```typescript
 * // Calculate specific KPIs
 * const result = await batchCalculateKPIs({
 *   kpi_ids: ['WS1-DQ-001', 'WS1-DQ-002', 'WS1-MP-001'],
 *   context: { brand: 'remibrutinib' },
 * });
 *
 * // Calculate all KPIs for a workstream
 * const wsResult = await batchCalculateKPIs({
 *   workstream: 'ws1_data_quality',
 * });
 * console.log(`Success: ${wsResult.successful}/${wsResult.total_kpis}`);
 * ```
 */
export async function batchCalculateKPIs(
  request: BatchKPICalculationRequest
): Promise<BatchKPICalculationResponse> {
  return post<BatchKPICalculationResponse, BatchKPICalculationRequest>(
    `${KPI_BASE}/batch`,
    request
  );
}

// =============================================================================
// CACHE MANAGEMENT ENDPOINTS
// =============================================================================

/**
 * Invalidate cached KPI values.
 *
 * Can invalidate a specific KPI, all KPIs in a workstream, or all KPIs.
 *
 * @param request - Cache invalidation request
 * @returns Invalidation result
 *
 * @example
 * ```typescript
 * // Invalidate specific KPI
 * await invalidateKPICache({ kpi_id: 'WS1-DQ-001' });
 *
 * // Invalidate workstream
 * await invalidateKPICache({ workstream: 'ws1_data_quality' });
 *
 * // Invalidate all (use with caution)
 * await invalidateKPICache({ invalidate_all: true });
 * ```
 */
export async function invalidateKPICache(
  request: CacheInvalidationRequest
): Promise<CacheInvalidationResponse> {
  return post<CacheInvalidationResponse, CacheInvalidationRequest>(
    `${KPI_BASE}/invalidate`,
    request
  );
}

// =============================================================================
// HEALTH ENDPOINTS
// =============================================================================

/**
 * Get KPI system health status.
 *
 * @returns Health status including registry, cache, and database status
 *
 * @example
 * ```typescript
 * const health = await getKPIHealth();
 * if (health.status !== 'healthy') {
 *   console.warn('KPI system issues:', health.error);
 * }
 * ```
 */
export async function getKPIHealth(): Promise<KPIHealthResponse> {
  return get<KPIHealthResponse>(`${KPI_BASE}/health`);
}
