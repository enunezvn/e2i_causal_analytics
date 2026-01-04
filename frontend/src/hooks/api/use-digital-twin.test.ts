/**
 * Digital Twin API Query Hooks Tests
 * ===================================
 *
 * Tests for TanStack Query hooks for the E2I Digital Twin simulation API.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';

// Mock the API functions
vi.mock('@/api/digital-twin', () => ({
  runSimulation: vi.fn(),
  compareScenarios: vi.fn(),
  getSimulation: vi.fn(),
  getSimulationHistory: vi.fn(),
  getDigitalTwinHealth: vi.fn(),
}));

// Mock query-client
vi.mock('@/lib/query-client', () => ({
  queryKeys: {
    all: ['e2i'] as const,
    digitalTwin: {
      all: () => ['e2i', 'digitalTwin'] as const,
      simulation: (id: string) => ['e2i', 'digitalTwin', 'simulation', id] as const,
      history: (brand?: string) => ['e2i', 'digitalTwin', 'history', brand] as const,
      health: () => ['e2i', 'digitalTwin', 'health'] as const,
    },
  },
}));

import {
  useSimulation,
  useSimulationHistory,
  useDigitalTwinHealth,
  useRunSimulation,
  useCompareScenarios,
  prefetchSimulationHistory,
} from './use-digital-twin';
import * as digitalTwinApi from '@/api/digital-twin';

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
// MOCK DATA
// =============================================================================

const mockSimulationResponse = {
  simulation_id: 'sim_abc123',
  intervention_type: 'hcp_engagement',
  brand: 'Remibrutinib',
  status: 'completed',
  outcomes: {
    ate: { estimate: 0.15, std_error: 0.02, p_value: 0.001 },
    confidence_interval: [0.11, 0.19],
  },
  sample_size: 1000,
  duration_days: 90,
  created_at: '2024-01-15T10:00:00Z',
  completed_at: '2024-01-15T10:05:00Z',
};

const mockSimulationHistoryResponse = {
  simulations: [mockSimulationResponse],
  total: 1,
  page: 1,
  page_size: 10,
};

const mockHealthResponse = {
  status: 'healthy',
  model_version: '1.2.0',
  last_calibration: '2024-01-10T00:00:00Z',
};

const mockScenarioComparisonResult = {
  base_result: mockSimulationResponse,
  alternative_results: [
    { ...mockSimulationResponse, simulation_id: 'sim_alt1', intervention_type: 'digital_outreach' },
  ],
  comparison: {
    best_scenario: 'hcp_engagement',
    relative_improvement: 0.25,
  },
};

// =============================================================================
// QUERY HOOK TESTS
// =============================================================================

describe('useSimulation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches simulation successfully', async () => {
    vi.mocked(digitalTwinApi.getSimulation).mockResolvedValueOnce(mockSimulationResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useSimulation('sim_abc123'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockSimulationResponse);
    expect(digitalTwinApi.getSimulation).toHaveBeenCalledWith('sim_abc123');
  });

  it('is disabled when simulationId is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useSimulation(''), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(digitalTwinApi.getSimulation).not.toHaveBeenCalled();
  });

  it('handles 404 error', async () => {
    const error = { status: 404, message: 'Simulation not found' };
    vi.mocked(digitalTwinApi.getSimulation).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useSimulation('nonexistent'), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('respects custom options', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(
      () => useSimulation('sim_abc123', { enabled: false }),
      { wrapper }
    );

    expect(result.current.fetchStatus).toBe('idle');
    expect(digitalTwinApi.getSimulation).not.toHaveBeenCalled();
  });
});

describe('useSimulationHistory', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches simulation history successfully', async () => {
    vi.mocked(digitalTwinApi.getSimulationHistory).mockResolvedValueOnce(mockSimulationHistoryResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useSimulationHistory(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockSimulationHistoryResponse);
    expect(digitalTwinApi.getSimulationHistory).toHaveBeenCalledWith(undefined);
  });

  it('passes params to API', async () => {
    vi.mocked(digitalTwinApi.getSimulationHistory).mockResolvedValueOnce(mockSimulationHistoryResponse);
    const { wrapper } = createWrapper();
    const params = { brand: 'Remibrutinib', limit: 10 };

    const { result } = renderHook(() => useSimulationHistory(params), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(digitalTwinApi.getSimulationHistory).toHaveBeenCalledWith(params);
  });

  it('handles empty history', async () => {
    const emptyResponse = { simulations: [], total: 0, page: 1, page_size: 10 };
    vi.mocked(digitalTwinApi.getSimulationHistory).mockResolvedValueOnce(emptyResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useSimulationHistory(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.total).toBe(0);
  });

  it('filters by intervention type', async () => {
    vi.mocked(digitalTwinApi.getSimulationHistory).mockResolvedValueOnce(mockSimulationHistoryResponse);
    const { wrapper } = createWrapper();
    const params = { intervention_type: 'hcp_engagement', limit: 5 };

    const { result } = renderHook(() => useSimulationHistory(params), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(digitalTwinApi.getSimulationHistory).toHaveBeenCalledWith(params);
  });
});

describe('useDigitalTwinHealth', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches digital twin health successfully', async () => {
    vi.mocked(digitalTwinApi.getDigitalTwinHealth).mockResolvedValueOnce(mockHealthResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useDigitalTwinHealth(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockHealthResponse);
    expect(digitalTwinApi.getDigitalTwinHealth).toHaveBeenCalled();
  });

  it('handles unhealthy status', async () => {
    const unhealthyResponse = { ...mockHealthResponse, status: 'degraded' };
    vi.mocked(digitalTwinApi.getDigitalTwinHealth).mockResolvedValueOnce(unhealthyResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useDigitalTwinHealth(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.status).toBe('degraded');
  });

  it('handles error state', async () => {
    const error = new Error('Service unavailable');
    vi.mocked(digitalTwinApi.getDigitalTwinHealth).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useDigitalTwinHealth(), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

// =============================================================================
// MUTATION HOOK TESTS
// =============================================================================

describe('useRunSimulation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('runs simulation successfully', async () => {
    vi.mocked(digitalTwinApi.runSimulation).mockResolvedValueOnce(mockSimulationResponse);
    const { wrapper, queryClient } = createWrapper();
    const setQueryDataSpy = vi.spyOn(queryClient, 'setQueryData');
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useRunSimulation(), { wrapper });

    const request = {
      intervention_type: 'hcp_engagement',
      brand: 'Remibrutinib',
      sample_size: 1000,
      duration_days: 90,
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockSimulationResponse);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(digitalTwinApi.runSimulation).toHaveBeenCalledWith(request, expect.anything());
    expect(setQueryDataSpy).toHaveBeenCalled();
    expect(invalidateSpy).toHaveBeenCalled();
  });

  it('handles simulation error', async () => {
    const error = new Error('Simulation failed');
    vi.mocked(digitalTwinApi.runSimulation).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useRunSimulation(), { wrapper });

    result.current.mutate({
      intervention_type: 'hcp_engagement',
      brand: 'Unknown',
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('calls onSuccess callback', async () => {
    vi.mocked(digitalTwinApi.runSimulation).mockResolvedValueOnce(mockSimulationResponse);
    const { wrapper } = createWrapper();
    const onSuccess = vi.fn();

    const { result } = renderHook(() => useRunSimulation({ onSuccess }), { wrapper });

    result.current.mutate({
      intervention_type: 'hcp_engagement',
      brand: 'Remibrutinib',
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(onSuccess).toHaveBeenCalled();
  });

  it('calls onError callback', async () => {
    const error = new Error('Simulation failed');
    vi.mocked(digitalTwinApi.runSimulation).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();
    const onError = vi.fn();

    const { result } = renderHook(() => useRunSimulation({ onError }), { wrapper });

    result.current.mutate({
      intervention_type: 'hcp_engagement',
      brand: 'Unknown',
    });

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(onError).toHaveBeenCalled();
  });
});

describe('useCompareScenarios', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('compares scenarios successfully', async () => {
    vi.mocked(digitalTwinApi.compareScenarios).mockResolvedValueOnce(mockScenarioComparisonResult);
    const { wrapper, queryClient } = createWrapper();
    const setQueryDataSpy = vi.spyOn(queryClient, 'setQueryData');

    const { result } = renderHook(() => useCompareScenarios(), { wrapper });

    const request = {
      base_scenario: {
        intervention_type: 'hcp_engagement',
        brand: 'Remibrutinib',
      },
      alternative_scenarios: [
        { intervention_type: 'digital_outreach', brand: 'Remibrutinib' },
      ],
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockScenarioComparisonResult);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(digitalTwinApi.compareScenarios).toHaveBeenCalledWith(request, expect.anything());
    // Should cache base and alternative results
    expect(setQueryDataSpy).toHaveBeenCalled();
  });

  it('handles comparison error', async () => {
    const error = new Error('Comparison failed');
    vi.mocked(digitalTwinApi.compareScenarios).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCompareScenarios(), { wrapper });

    result.current.mutate({
      base_scenario: { intervention_type: 'hcp_engagement', brand: 'Unknown' },
      alternative_scenarios: [],
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('calls onSuccess callback', async () => {
    vi.mocked(digitalTwinApi.compareScenarios).mockResolvedValueOnce(mockScenarioComparisonResult);
    const { wrapper } = createWrapper();
    const onSuccess = vi.fn();

    const { result } = renderHook(() => useCompareScenarios({ onSuccess }), { wrapper });

    result.current.mutate({
      base_scenario: { intervention_type: 'hcp_engagement', brand: 'Remibrutinib' },
      alternative_scenarios: [],
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(onSuccess).toHaveBeenCalled();
  });

  it('caches all simulation results', async () => {
    vi.mocked(digitalTwinApi.compareScenarios).mockResolvedValueOnce(mockScenarioComparisonResult);
    const { wrapper, queryClient } = createWrapper();
    const setQueryDataSpy = vi.spyOn(queryClient, 'setQueryData');

    const { result } = renderHook(() => useCompareScenarios(), { wrapper });

    result.current.mutate({
      base_scenario: { intervention_type: 'hcp_engagement', brand: 'Remibrutinib' },
      alternative_scenarios: [{ intervention_type: 'digital_outreach', brand: 'Remibrutinib' }],
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    // Should cache base result and each alternative result
    expect(setQueryDataSpy).toHaveBeenCalledTimes(2);
  });
});

// =============================================================================
// PREFETCH HELPER TESTS
// =============================================================================

describe('prefetchSimulationHistory', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches simulation history', async () => {
    vi.mocked(digitalTwinApi.getSimulationHistory).mockResolvedValueOnce(mockSimulationHistoryResponse);
    const queryClient = createTestQueryClient();

    await prefetchSimulationHistory(queryClient);

    expect(digitalTwinApi.getSimulationHistory).toHaveBeenCalledWith({ brand: undefined, limit: 10 });
  });

  it('prefetches with brand filter', async () => {
    vi.mocked(digitalTwinApi.getSimulationHistory).mockResolvedValueOnce(mockSimulationHistoryResponse);
    const queryClient = createTestQueryClient();

    await prefetchSimulationHistory(queryClient, 'Remibrutinib');

    expect(digitalTwinApi.getSimulationHistory).toHaveBeenCalledWith({ brand: 'Remibrutinib', limit: 10 });
  });
});
