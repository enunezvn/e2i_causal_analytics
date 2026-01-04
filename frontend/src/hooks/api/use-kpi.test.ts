/**
 * KPI React Query Hooks Tests
 * ===========================
 *
 * Tests for TanStack Query hooks for the E2I KPI API.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';

// Mock the API functions
vi.mock('@/api/kpi', () => ({
  listKPIs: vi.fn(),
  getWorkstreams: vi.fn(),
  getKPIMetadata: vi.fn(),
  getKPIValue: vi.fn(),
  calculateKPI: vi.fn(),
  batchCalculateKPIs: vi.fn(),
  invalidateKPICache: vi.fn(),
  getKPIHealth: vi.fn(),
}));

// Mock query-client
vi.mock('@/lib/query-client', () => ({
  queryKeys: {
    all: ['e2i'] as const,
    kpi: {
      all: () => ['e2i', 'kpi'] as const,
      list: () => ['e2i', 'kpi', 'list'] as const,
      workstreams: () => ['e2i', 'kpi', 'workstreams'] as const,
      health: () => ['e2i', 'kpi', 'health'] as const,
      detail: (id: string) => ['e2i', 'kpi', 'detail', id] as const,
    },
  },
  queryClient: new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0 },
      mutations: { retry: false },
    },
  }),
}));

import {
  useKPIList,
  useWorkstreams,
  useKPIHealth,
  useKPIMetadata,
  useKPIValue,
  useCalculateKPI,
  useBatchCalculateKPIs,
  useInvalidateKPICache,
  useKPIDetail,
  prefetchKPIList,
  prefetchWorkstreams,
  prefetchKPIMetadata,
  prefetchKPIHealth,
} from './use-kpi';
import * as kpiApi from '@/api/kpi';
import { queryClient as globalQueryClient } from '@/lib/query-client';

// =============================================================================
// TEST UTILITIES
// =============================================================================

function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });
}

function createWrapper() {
  const queryClient = createTestQueryClient();
  return {
    queryClient,
    wrapper: ({ children }: { children: React.ReactNode }) =>
      React.createElement(QueryClientProvider, { client: queryClient }, children),
  };
}

// =============================================================================
// TYPE IMPORTS
// =============================================================================

import type {
  KPIListResponse,
  WorkstreamListResponse,
  KPIHealthResponse,
  KPIMetadata,
  KPIResult,
  BatchKPICalculationResponse,
  CacheInvalidationResponse,
  WorkstreamInfo,
} from '@/types/kpi';

// =============================================================================
// MOCK DATA
// =============================================================================

const mockKPIMetadata1: KPIMetadata = {
  id: 'WS1-DQ-001',
  name: 'Data Freshness',
  definition: 'Measures how current the data is',
  formula: 'current_date - last_update_date',
  calculation_type: 'direct',
  workstream: 'ws1_data_quality',
  tables: ['data_sources'],
  columns: ['last_update_date'],
  frequency: 'daily',
  primary_causal_library: 'none',
  threshold: { warning: 24, critical: 48 },
  unit: 'hours',
};

const mockKPIMetadata2: KPIMetadata = {
  id: 'WS2-TRX-001',
  name: 'TRx Volume',
  definition: 'Total prescription volume',
  formula: 'SUM(trx_count)',
  calculation_type: 'direct',
  workstream: 'ws2_triggers',
  tables: ['prescriptions'],
  columns: ['trx_count'],
  frequency: 'weekly',
  primary_causal_library: 'dowhy',
};

const mockKPIListResponse: KPIListResponse = {
  kpis: [mockKPIMetadata1, mockKPIMetadata2],
  total: 2,
};

const mockWorkstreamInfo1: WorkstreamInfo = {
  id: 'ws1_data_quality',
  name: 'Data Quality',
  kpi_count: 10,
  description: 'Data quality metrics',
};

const mockWorkstreamInfo2: WorkstreamInfo = {
  id: 'ws2_triggers',
  name: 'Trigger Metrics',
  kpi_count: 8,
  description: 'Trigger-related metrics',
};

const mockWorkstreamsResponse: WorkstreamListResponse = {
  workstreams: [mockWorkstreamInfo1, mockWorkstreamInfo2],
  total: 2,
};

const mockKPIHealthResponse: KPIHealthResponse = {
  status: 'healthy',
  registry_loaded: true,
  total_kpis: 46,
  cache_enabled: true,
  cache_size: 150,
  database_connected: true,
  workstreams_available: ['ws1_data_quality', 'ws2_triggers', 'ws3_business'],
  last_calculation: '2024-01-15T10:00:00Z',
};

const mockKPIMetadata: KPIMetadata = {
  id: 'WS1-DQ-001',
  name: 'Data Freshness',
  definition: 'Measures how current the data is',
  formula: 'current_date - last_update_date',
  calculation_type: 'direct',
  workstream: 'ws1_data_quality',
  tables: ['data_sources'],
  columns: ['last_update_date'],
  frequency: 'daily',
  primary_causal_library: 'none',
  threshold: { warning: 24, critical: 48 },
  unit: 'hours',
};

const mockKPIResult: KPIResult = {
  kpi_id: 'WS1-DQ-001',
  value: 12.5,
  status: 'good',
  calculated_at: '2024-01-15T12:00:00Z',
  cached: false,
  metadata: { brand: 'remibrutinib' },
};

const mockBatchCalculationResponse: BatchKPICalculationResponse = {
  results: [mockKPIResult],
  calculated_at: '2024-01-15T12:00:00Z',
  total_kpis: 1,
  successful: 1,
  failed: 0,
};

const mockCacheInvalidationResponse: CacheInvalidationResponse = {
  invalidated_count: 5,
  message: 'Successfully invalidated 5 cache entries',
};

// =============================================================================
// QUERY HOOK TESTS
// =============================================================================

describe('useKPIList', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches KPI list successfully', async () => {
    vi.mocked(kpiApi.listKPIs).mockResolvedValueOnce(mockKPIListResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useKPIList(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockKPIListResponse);
    expect(kpiApi.listKPIs).toHaveBeenCalledWith(undefined);
  });

  it('passes params to API', async () => {
    vi.mocked(kpiApi.listKPIs).mockResolvedValueOnce(mockKPIListResponse);
    const { wrapper } = createWrapper();
    const params = { workstream: 'ws1_data_quality' };

    const { result } = renderHook(() => useKPIList(params), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(kpiApi.listKPIs).toHaveBeenCalledWith(params);
  });

  it('handles error state', async () => {
    const error = new Error('Service unavailable');
    vi.mocked(kpiApi.listKPIs).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useKPIList(), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('respects custom options', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useKPIList(undefined, { enabled: false }), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(kpiApi.listKPIs).not.toHaveBeenCalled();
  });
});

describe('useWorkstreams', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches workstreams successfully', async () => {
    vi.mocked(kpiApi.getWorkstreams).mockResolvedValueOnce(mockWorkstreamsResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useWorkstreams(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockWorkstreamsResponse);
    expect(kpiApi.getWorkstreams).toHaveBeenCalled();
  });

  it('handles empty workstreams', async () => {
    const emptyResponse: WorkstreamListResponse = { workstreams: [], total: 0 };
    vi.mocked(kpiApi.getWorkstreams).mockResolvedValueOnce(emptyResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useWorkstreams(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.workstreams).toHaveLength(0);
  });
});

describe('useKPIHealth', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches KPI health successfully', async () => {
    vi.mocked(kpiApi.getKPIHealth).mockResolvedValueOnce(mockKPIHealthResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useKPIHealth(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockKPIHealthResponse);
    expect(kpiApi.getKPIHealth).toHaveBeenCalled();
  });

  it('handles unhealthy status', async () => {
    const unhealthyResponse: KPIHealthResponse = {
      ...mockKPIHealthResponse,
      status: 'degraded',
      error: 'Cache service unavailable',
    };
    vi.mocked(kpiApi.getKPIHealth).mockResolvedValueOnce(unhealthyResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useKPIHealth(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.status).toBe('degraded');
  });
});

describe('useKPIMetadata', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches KPI metadata successfully', async () => {
    vi.mocked(kpiApi.getKPIMetadata).mockResolvedValueOnce(mockKPIMetadata);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useKPIMetadata('WS1-DQ-001'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockKPIMetadata);
    expect(kpiApi.getKPIMetadata).toHaveBeenCalledWith('WS1-DQ-001');
  });

  it('is disabled when kpiId is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useKPIMetadata(''), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(kpiApi.getKPIMetadata).not.toHaveBeenCalled();
  });

  it('handles 404 error', async () => {
    const error = { status: 404, message: 'KPI not found' };
    vi.mocked(kpiApi.getKPIMetadata).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useKPIMetadata('nonexistent'), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useKPIValue', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches KPI value successfully', async () => {
    vi.mocked(kpiApi.getKPIValue).mockResolvedValueOnce(mockKPIResult);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useKPIValue('WS1-DQ-001'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockKPIResult);
    expect(kpiApi.getKPIValue).toHaveBeenCalledWith('WS1-DQ-001', undefined);
  });

  it('passes brand to API', async () => {
    vi.mocked(kpiApi.getKPIValue).mockResolvedValueOnce(mockKPIResult);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useKPIValue('WS1-DQ-001', 'remibrutinib'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(kpiApi.getKPIValue).toHaveBeenCalledWith('WS1-DQ-001', 'remibrutinib');
  });

  it('is disabled when kpiId is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useKPIValue(''), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(kpiApi.getKPIValue).not.toHaveBeenCalled();
  });
});

// =============================================================================
// MUTATION HOOK TESTS
// =============================================================================

describe('useCalculateKPI', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('calculates KPI successfully', async () => {
    vi.mocked(kpiApi.calculateKPI).mockResolvedValueOnce(mockKPIResult);
    const { wrapper, queryClient } = createWrapper();
    const setQueryDataSpy = vi.spyOn(queryClient, 'setQueryData');

    const { result } = renderHook(() => useCalculateKPI(), { wrapper });

    const request = {
      kpi_id: 'WS1-DQ-001',
      context: { brand: 'remibrutinib' },
      force_refresh: true,
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockKPIResult);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(kpiApi.calculateKPI).toHaveBeenCalledWith(request, expect.anything());
    expect(setQueryDataSpy).toHaveBeenCalled();
  });

  it('calls onSuccess callback', async () => {
    vi.mocked(kpiApi.calculateKPI).mockResolvedValueOnce(mockKPIResult);
    const { wrapper } = createWrapper();
    const onSuccess = vi.fn();

    const { result } = renderHook(() => useCalculateKPI({ onSuccess }), { wrapper });

    result.current.mutate({ kpi_id: 'WS1-DQ-001' });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(onSuccess).toHaveBeenCalled();
  });

  it('handles calculation error', async () => {
    const error = new Error('Calculation failed');
    vi.mocked(kpiApi.calculateKPI).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCalculateKPI(), { wrapper });

    result.current.mutate({ kpi_id: 'invalid' });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useBatchCalculateKPIs', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('batch calculates KPIs successfully', async () => {
    vi.mocked(kpiApi.batchCalculateKPIs).mockResolvedValueOnce(mockBatchCalculationResponse);
    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useBatchCalculateKPIs(), { wrapper });

    const request = {
      workstream: 'ws1_data_quality',
      context: { brand: 'remibrutinib' },
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockBatchCalculationResponse);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(kpiApi.batchCalculateKPIs).toHaveBeenCalledWith(request, expect.anything());
    expect(invalidateSpy).toHaveBeenCalled();
  });

  it('handles partial failures', async () => {
    const partialResponse = { ...mockBatchCalculationResponse, failed: 2 };
    vi.mocked(kpiApi.batchCalculateKPIs).mockResolvedValueOnce(partialResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useBatchCalculateKPIs(), { wrapper });

    result.current.mutate({ workstream: 'ws1_data_quality' });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.failed).toBe(2);
  });
});

describe('useInvalidateKPICache', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('invalidates specific KPI cache', async () => {
    vi.mocked(kpiApi.invalidateKPICache).mockResolvedValueOnce(mockCacheInvalidationResponse);
    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useInvalidateKPICache(), { wrapper });

    result.current.mutate({ kpi_id: 'WS1-DQ-001' });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockCacheInvalidationResponse);
    expect(invalidateSpy).toHaveBeenCalled();
  });

  it('invalidates workstream cache', async () => {
    vi.mocked(kpiApi.invalidateKPICache).mockResolvedValueOnce(mockCacheInvalidationResponse);
    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useInvalidateKPICache(), { wrapper });

    result.current.mutate({ workstream: 'ws1_data_quality' });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(invalidateSpy).toHaveBeenCalled();
  });

  it('invalidates all cache', async () => {
    vi.mocked(kpiApi.invalidateKPICache).mockResolvedValueOnce(mockCacheInvalidationResponse);
    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useInvalidateKPICache(), { wrapper });

    result.current.mutate({ invalidate_all: true });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(invalidateSpy).toHaveBeenCalled();
  });
});

// =============================================================================
// COMBINED HOOK TESTS
// =============================================================================

describe('useKPIDetail', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches both metadata and value', async () => {
    vi.mocked(kpiApi.getKPIMetadata).mockResolvedValueOnce(mockKPIMetadata);
    vi.mocked(kpiApi.getKPIValue).mockResolvedValueOnce(mockKPIResult);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useKPIDetail('WS1-DQ-001'), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.metadata).toEqual(mockKPIMetadata);
    expect(result.current.value).toEqual(mockKPIResult);
  });

  it('passes brand to value query', async () => {
    vi.mocked(kpiApi.getKPIMetadata).mockResolvedValueOnce(mockKPIMetadata);
    vi.mocked(kpiApi.getKPIValue).mockResolvedValueOnce(mockKPIResult);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useKPIDetail('WS1-DQ-001', 'remibrutinib'), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(kpiApi.getKPIValue).toHaveBeenCalledWith('WS1-DQ-001', 'remibrutinib');
  });

  it('reports loading state correctly', async () => {
    vi.mocked(kpiApi.getKPIMetadata).mockImplementation(
      () => new Promise((resolve) => setTimeout(() => resolve(mockKPIMetadata), 100))
    );
    vi.mocked(kpiApi.getKPIValue).mockResolvedValueOnce(mockKPIResult);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useKPIDetail('WS1-DQ-001'), { wrapper });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => expect(result.current.isLoading).toBe(false));
  });

  it('provides refetch function', async () => {
    vi.mocked(kpiApi.getKPIMetadata).mockResolvedValue(mockKPIMetadata);
    vi.mocked(kpiApi.getKPIValue).mockResolvedValue(mockKPIResult);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useKPIDetail('WS1-DQ-001'), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    // Call refetch
    result.current.refetch();

    await waitFor(() => expect(kpiApi.getKPIMetadata).toHaveBeenCalledTimes(2));
  });
});

// =============================================================================
// PREFETCH HELPER TESTS
// =============================================================================

describe('prefetchKPIList', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Clear the global query client cache to ensure fresh prefetch calls
    globalQueryClient.clear();
  });

  it('prefetches KPI list', async () => {
    vi.mocked(kpiApi.listKPIs).mockResolvedValueOnce(mockKPIListResponse);

    await prefetchKPIList();

    expect(kpiApi.listKPIs).toHaveBeenCalledWith(undefined);
  });

  it('prefetches with params', async () => {
    vi.mocked(kpiApi.listKPIs).mockResolvedValueOnce(mockKPIListResponse);
    const params = { workstream: 'ws1_data_quality' };

    await prefetchKPIList(params);

    expect(kpiApi.listKPIs).toHaveBeenCalledWith(params);
  });
});

describe('prefetchWorkstreams', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches workstreams', async () => {
    vi.mocked(kpiApi.getWorkstreams).mockResolvedValueOnce(mockWorkstreamsResponse);

    await prefetchWorkstreams();

    expect(kpiApi.getWorkstreams).toHaveBeenCalled();
  });
});

describe('prefetchKPIMetadata', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches KPI metadata', async () => {
    vi.mocked(kpiApi.getKPIMetadata).mockResolvedValueOnce(mockKPIMetadata);

    await prefetchKPIMetadata('WS1-DQ-001');

    expect(kpiApi.getKPIMetadata).toHaveBeenCalledWith('WS1-DQ-001');
  });
});

describe('prefetchKPIHealth', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches KPI health', async () => {
    vi.mocked(kpiApi.getKPIHealth).mockResolvedValueOnce(mockKPIHealthResponse);

    await prefetchKPIHealth();

    expect(kpiApi.getKPIHealth).toHaveBeenCalled();
  });
});
