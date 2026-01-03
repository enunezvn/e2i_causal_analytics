/**
 * KPI React Query Hooks
 * =====================
 *
 * TanStack Query hooks for interacting with the E2I KPI API.
 * Provides caching, deduplication, and optimistic updates.
 *
 * Features:
 * - List KPIs with filtering
 * - Calculate individual and batch KPIs
 * - Workstream management
 * - Cache invalidation
 * - System health monitoring
 *
 * @module hooks/api/use-kpi
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import {
  listKPIs,
  getWorkstreams,
  getKPIMetadata,
  getKPIValue,
  calculateKPI,
  batchCalculateKPIs,
  invalidateKPICache,
  getKPIHealth,
} from '@/api/kpi';
import { queryKeys, queryClient as globalQueryClient } from '@/lib/query-client';
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
// TYPE DEFINITIONS
// =============================================================================

type KPIQueryKey = ReturnType<typeof queryKeys.kpi.list>;

// =============================================================================
// QUERY HOOKS - LIST OPERATIONS
// =============================================================================

/**
 * Hook to fetch list of KPIs with optional filtering.
 *
 * @param params - Optional filters for workstream and causal library
 * @param options - Additional React Query options
 * @returns Query result with KPI list
 *
 * @example
 * ```tsx
 * const { data, isLoading } = useKPIList({ workstream: Workstream.WS1_DATA_QUALITY });
 *
 * return (
 *   <div>
 *     {data?.kpis.map(kpi => (
 *       <div key={kpi.id}>{kpi.name}</div>
 *     ))}
 *   </div>
 * );
 * ```
 */
export function useKPIList(
  params?: KPIListParams,
  options?: Omit<UseQueryOptions<KPIListResponse, Error, KPIListResponse, KPIQueryKey>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.kpi.list(),
    queryFn: () => listKPIs(params),
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
}

/**
 * Hook to fetch available workstreams.
 *
 * @param options - Additional React Query options
 * @returns Query result with workstream list
 *
 * @example
 * ```tsx
 * const { data } = useWorkstreams();
 *
 * return (
 *   <select>
 *     {data?.workstreams.map(ws => (
 *       <option key={ws.id} value={ws.id}>{ws.name}</option>
 *     ))}
 *   </select>
 * );
 * ```
 */
export function useWorkstreams(
  options?: Omit<UseQueryOptions<WorkstreamListResponse, Error>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.kpi.workstreams(),
    queryFn: getWorkstreams,
    staleTime: 10 * 60 * 1000, // 10 minutes - workstreams rarely change
    ...options,
  });
}

/**
 * Hook to fetch KPI system health.
 *
 * @param options - Additional React Query options
 * @returns Query result with health status
 *
 * @example
 * ```tsx
 * const { data: health } = useKPIHealth();
 *
 * if (health?.status !== 'healthy') {
 *   return <Alert severity="warning">{health?.error}</Alert>;
 * }
 * ```
 */
export function useKPIHealth(
  options?: Omit<UseQueryOptions<KPIHealthResponse, Error>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.kpi.health(),
    queryFn: getKPIHealth,
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 60 * 1000, // Auto-refresh every minute
    ...options,
  });
}

// =============================================================================
// QUERY HOOKS - KPI DETAILS
// =============================================================================

/**
 * Hook to fetch KPI metadata/definition.
 *
 * @param kpiId - KPI identifier
 * @param options - Additional React Query options
 * @returns Query result with KPI metadata
 *
 * @example
 * ```tsx
 * const { data: metadata } = useKPIMetadata('WS1-DQ-001');
 *
 * return (
 *   <div>
 *     <h2>{metadata?.name}</h2>
 *     <p>{metadata?.definition}</p>
 *     <code>{metadata?.formula}</code>
 *   </div>
 * );
 * ```
 */
export function useKPIMetadata(
  kpiId: string,
  options?: Omit<UseQueryOptions<KPIMetadata, Error>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.kpi.detail(kpiId),
    queryFn: () => getKPIMetadata(kpiId),
    staleTime: 30 * 60 * 1000, // 30 minutes - metadata rarely changes
    enabled: !!kpiId,
    ...options,
  });
}

/**
 * Hook to fetch a calculated KPI value.
 *
 * @param kpiId - KPI identifier
 * @param brand - Optional brand filter
 * @param options - Additional React Query options
 * @returns Query result with KPI value
 *
 * @example
 * ```tsx
 * const { data: result, isLoading } = useKPIValue('WS1-DQ-001', 'remibrutinib');
 *
 * return (
 *   <KPICard
 *     value={result?.value}
 *     status={result?.status}
 *     loading={isLoading}
 *   />
 * );
 * ```
 */
export function useKPIValue(
  kpiId: string,
  brand?: string,
  options?: Omit<UseQueryOptions<KPIResult, Error>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: [...queryKeys.kpi.detail(kpiId), 'value', brand] as const,
    queryFn: () => getKPIValue(kpiId, brand),
    staleTime: 2 * 60 * 1000, // 2 minutes
    enabled: !!kpiId,
    ...options,
  });
}

// =============================================================================
// MUTATION HOOKS - CALCULATIONS
// =============================================================================

/**
 * Hook to calculate a single KPI with full context.
 *
 * @param options - Additional React Query mutation options
 * @returns Mutation object for KPI calculation
 *
 * @example
 * ```tsx
 * const { mutate: calculate, isPending } = useCalculateKPI({
 *   onSuccess: (result) => {
 *     toast.success(`KPI calculated: ${result.value}`);
 *   }
 * });
 *
 * const handleCalculate = () => {
 *   calculate({
 *     kpi_id: 'WS1-DQ-001',
 *     context: { brand: 'remibrutinib' },
 *     force_refresh: true,
 *   });
 * };
 * ```
 */
export function useCalculateKPI(
  options?: Omit<UseMutationOptions<KPIResult, Error, KPICalculationRequest>, 'mutationFn'>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: calculateKPI,
    onSuccess: (result, variables) => {
      // Update the cached value for this KPI
      queryClient.setQueryData(
        [...queryKeys.kpi.detail(variables.kpi_id), 'value', variables.context?.brand] as const,
        result
      );
    },
    ...options,
  });
}

/**
 * Hook to calculate multiple KPIs in batch.
 *
 * @param options - Additional React Query mutation options
 * @returns Mutation object for batch calculation
 *
 * @example
 * ```tsx
 * const { mutate: batchCalculate, isPending } = useBatchCalculateKPIs({
 *   onSuccess: (result) => {
 *     toast.success(`Calculated ${result.successful}/${result.total_kpis} KPIs`);
 *   }
 * });
 *
 * const handleBatchCalculate = () => {
 *   batchCalculate({
 *     workstream: 'ws1_data_quality',
 *     context: { brand: selectedBrand },
 *   });
 * };
 * ```
 */
export function useBatchCalculateKPIs(
  options?: Omit<UseMutationOptions<BatchKPICalculationResponse, Error, BatchKPICalculationRequest>, 'mutationFn'>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: batchCalculateKPIs,
    onSuccess: () => {
      // Invalidate all KPI-related queries
      queryClient.invalidateQueries({ queryKey: queryKeys.kpi.all() });
    },
    ...options,
  });
}

// =============================================================================
// MUTATION HOOKS - CACHE MANAGEMENT
// =============================================================================

/**
 * Hook to invalidate KPI cache.
 *
 * @param options - Additional React Query mutation options
 * @returns Mutation object for cache invalidation
 *
 * @example
 * ```tsx
 * const { mutate: invalidateCache } = useInvalidateKPICache({
 *   onSuccess: (result) => {
 *     toast.info(`Invalidated ${result.invalidated_count} cache entries`);
 *   }
 * });
 *
 * // Invalidate specific KPI
 * invalidateCache({ kpi_id: 'WS1-DQ-001' });
 *
 * // Invalidate workstream
 * invalidateCache({ workstream: 'ws1_data_quality' });
 * ```
 */
export function useInvalidateKPICache(
  options?: Omit<UseMutationOptions<CacheInvalidationResponse, Error, CacheInvalidationRequest>, 'mutationFn'>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: invalidateKPICache,
    onSuccess: (_, variables) => {
      // Invalidate React Query cache based on what was invalidated
      if (variables.invalidate_all) {
        queryClient.invalidateQueries({ queryKey: queryKeys.kpi.all() });
      } else if (variables.kpi_id) {
        queryClient.invalidateQueries({ queryKey: queryKeys.kpi.detail(variables.kpi_id) });
      } else {
        // Workstream invalidation - refresh all
        queryClient.invalidateQueries({ queryKey: queryKeys.kpi.all() });
      }
    },
    ...options,
  });
}

// =============================================================================
// PREFETCH HELPERS
// =============================================================================

/**
 * Prefetch KPI list for faster navigation.
 *
 * @param params - Optional filter parameters
 *
 * @example
 * ```tsx
 * // Prefetch on hover
 * <Link
 *   to="/kpis"
 *   onMouseEnter={() => prefetchKPIList()}
 * >
 *   View KPIs
 * </Link>
 * ```
 */
export function prefetchKPIList(params?: KPIListParams): Promise<void> {
  return globalQueryClient.prefetchQuery({
    queryKey: queryKeys.kpi.list(),
    queryFn: () => listKPIs(params),
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Prefetch workstreams list.
 *
 * @example
 * ```tsx
 * useEffect(() => {
 *   prefetchWorkstreams();
 * }, []);
 * ```
 */
export function prefetchWorkstreams(): Promise<void> {
  return globalQueryClient.prefetchQuery({
    queryKey: queryKeys.kpi.workstreams(),
    queryFn: getWorkstreams,
    staleTime: 10 * 60 * 1000,
  });
}

/**
 * Prefetch KPI metadata for faster detail view loading.
 *
 * @param kpiId - KPI identifier to prefetch
 *
 * @example
 * ```tsx
 * // Prefetch on hover over KPI row
 * <tr onMouseEnter={() => prefetchKPIMetadata(kpi.id)}>
 *   <td>{kpi.name}</td>
 * </tr>
 * ```
 */
export function prefetchKPIMetadata(kpiId: string): Promise<void> {
  return globalQueryClient.prefetchQuery({
    queryKey: queryKeys.kpi.detail(kpiId),
    queryFn: () => getKPIMetadata(kpiId),
    staleTime: 30 * 60 * 1000,
  });
}

/**
 * Prefetch KPI health for dashboard.
 *
 * @example
 * ```tsx
 * useEffect(() => {
 *   prefetchKPIHealth();
 * }, []);
 * ```
 */
export function prefetchKPIHealth(): Promise<void> {
  return globalQueryClient.prefetchQuery({
    queryKey: queryKeys.kpi.health(),
    queryFn: getKPIHealth,
    staleTime: 30 * 1000,
  });
}

// =============================================================================
// COMBINED HOOKS
// =============================================================================

/**
 * Hook to get complete KPI info (metadata + current value).
 *
 * Combines metadata and value queries for a complete KPI view.
 *
 * @param kpiId - KPI identifier
 * @param brand - Optional brand filter
 * @returns Combined query results for metadata and value
 *
 * @example
 * ```tsx
 * const { metadata, value, isLoading, error } = useKPIDetail('WS1-DQ-001', 'remibrutinib');
 *
 * if (isLoading) return <Skeleton />;
 *
 * return (
 *   <KPIDetailCard
 *     name={metadata?.name}
 *     definition={metadata?.definition}
 *     value={value?.value}
 *     status={value?.status}
 *     threshold={metadata?.threshold}
 *   />
 * );
 * ```
 */
export function useKPIDetail(kpiId: string, brand?: string) {
  const metadataQuery = useKPIMetadata(kpiId);
  const valueQuery = useKPIValue(kpiId, brand);

  return {
    metadata: metadataQuery.data,
    value: valueQuery.data,
    isLoading: metadataQuery.isLoading || valueQuery.isLoading,
    error: metadataQuery.error || valueQuery.error,
    isMetadataLoading: metadataQuery.isLoading,
    isValueLoading: valueQuery.isLoading,
    refetch: () => {
      metadataQuery.refetch();
      valueQuery.refetch();
    },
  };
}
