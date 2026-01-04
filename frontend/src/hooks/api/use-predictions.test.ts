/**
 * Model Predictions React Query Hooks Tests
 * =========================================
 *
 * Tests for TanStack Query hooks for the E2I Model Predictions API.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';

// Mock the API functions
vi.mock('@/api/predictions', () => ({
  predict: vi.fn(),
  predictBatch: vi.fn(),
  getModelHealth: vi.fn(),
  getModelInfo: vi.fn(),
  getModelsStatus: vi.fn(),
}));

// Mock query-client
vi.mock('@/lib/query-client', () => ({
  queryKeys: {
    all: ['e2i'] as const,
    predictions: {
      all: () => ['e2i', 'predictions'] as const,
      modelHealth: (modelName: string) => ['e2i', 'predictions', 'health', modelName] as const,
      modelInfo: (modelName: string) => ['e2i', 'predictions', 'info', modelName] as const,
      modelsStatus: () => ['e2i', 'predictions', 'status'] as const,
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
  useModelHealth,
  useModelInfo,
  useModelsStatus,
  usePredict,
  useBatchPredict,
  useModelDetail,
  useInvalidateModelCache,
  prefetchModelHealth,
  prefetchModelInfo,
  prefetchModelsStatus,
} from './use-predictions';
import * as predictionsApi from '@/api/predictions';

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

const mockModelHealthResponse = {
  model_name: 'churn_model',
  status: 'healthy',
  last_inference_time: '2024-01-15T12:00:00Z',
  error: null,
};

const mockModelInfoResponse = {
  name: 'churn_model',
  version: '2.1.0',
  type: 'classification',
  metrics: { accuracy: 0.92, f1_score: 0.89 },
  features: ['hcp_id', 'territory', 'engagement_score'],
  created_at: '2024-01-01T00:00:00Z',
};

const mockModelsStatusResponse = {
  total_models: 5,
  healthy_count: 4,
  unhealthy_count: 1,
  models: [
    { name: 'churn_model', status: 'healthy' },
    { name: 'propensity_model', status: 'healthy' },
    { name: 'conversion_model', status: 'unhealthy' },
  ],
};

const mockPredictionResponse = {
  prediction: 0.85,
  probabilities: { churn: 0.85, retain: 0.15 },
  model_version: '2.1.0',
  inference_time_ms: 15,
};

const mockBatchPredictionResponse = {
  predictions: [
    { id: 'HCP001', prediction: 0.85, probabilities: { churn: 0.85, retain: 0.15 } },
    { id: 'HCP002', prediction: 0.32, probabilities: { churn: 0.32, retain: 0.68 } },
  ],
  total_count: 2,
  success_count: 2,
  error_count: 0,
  processing_time_ms: 45,
};

// =============================================================================
// QUERY HOOK TESTS
// =============================================================================

describe('useModelHealth', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches model health successfully', async () => {
    vi.mocked(predictionsApi.getModelHealth).mockResolvedValueOnce(mockModelHealthResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useModelHealth('churn_model'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockModelHealthResponse);
    expect(predictionsApi.getModelHealth).toHaveBeenCalledWith('churn_model');
  });

  it('is disabled when modelName is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useModelHealth(''), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(predictionsApi.getModelHealth).not.toHaveBeenCalled();
  });

  it('handles error state', async () => {
    const error = new Error('Model not found');
    vi.mocked(predictionsApi.getModelHealth).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useModelHealth('unknown_model'), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('respects custom options', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(
      () => useModelHealth('churn_model', { enabled: false }),
      { wrapper }
    );

    expect(result.current.fetchStatus).toBe('idle');
    expect(predictionsApi.getModelHealth).not.toHaveBeenCalled();
  });
});

describe('useModelInfo', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches model info successfully', async () => {
    vi.mocked(predictionsApi.getModelInfo).mockResolvedValueOnce(mockModelInfoResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useModelInfo('churn_model'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockModelInfoResponse);
    expect(predictionsApi.getModelInfo).toHaveBeenCalledWith('churn_model');
  });

  it('is disabled when modelName is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useModelInfo(''), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(predictionsApi.getModelInfo).not.toHaveBeenCalled();
  });

  it('handles error state', async () => {
    const error = new Error('Model not found');
    vi.mocked(predictionsApi.getModelInfo).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useModelInfo('unknown_model'), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useModelsStatus', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches all models status successfully', async () => {
    vi.mocked(predictionsApi.getModelsStatus).mockResolvedValueOnce(mockModelsStatusResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useModelsStatus(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockModelsStatusResponse);
    expect(predictionsApi.getModelsStatus).toHaveBeenCalledWith(undefined);
  });

  it('passes model filter to API', async () => {
    vi.mocked(predictionsApi.getModelsStatus).mockResolvedValueOnce(mockModelsStatusResponse);
    const { wrapper } = createWrapper();
    const models = ['churn_model', 'propensity_model'];

    const { result } = renderHook(() => useModelsStatus(models), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(predictionsApi.getModelsStatus).toHaveBeenCalledWith(models);
  });

  it('handles empty models list', async () => {
    const emptyResponse = { ...mockModelsStatusResponse, models: [], total_models: 0 };
    vi.mocked(predictionsApi.getModelsStatus).mockResolvedValueOnce(emptyResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useModelsStatus(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.total_models).toBe(0);
  });
});

// =============================================================================
// MUTATION HOOK TESTS
// =============================================================================

describe('usePredict', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('makes prediction successfully', async () => {
    vi.mocked(predictionsApi.predict).mockResolvedValueOnce(mockPredictionResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => usePredict(), { wrapper });

    result.current.mutate({
      modelName: 'churn_model',
      request: {
        features: { hcp_id: 'HCP001', territory: 'Northeast' },
        return_probabilities: true,
      },
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockPredictionResponse);
    expect(predictionsApi.predict).toHaveBeenCalledWith('churn_model', {
      features: { hcp_id: 'HCP001', territory: 'Northeast' },
      return_probabilities: true,
    });
  });

  it('handles prediction error', async () => {
    const error = new Error('Prediction failed');
    vi.mocked(predictionsApi.predict).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => usePredict(), { wrapper });

    result.current.mutate({
      modelName: 'churn_model',
      request: { features: {} },
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('calls onSuccess callback', async () => {
    vi.mocked(predictionsApi.predict).mockResolvedValueOnce(mockPredictionResponse);
    const { wrapper } = createWrapper();
    const onSuccess = vi.fn();

    const { result } = renderHook(() => usePredict({ onSuccess }), { wrapper });

    result.current.mutate({
      modelName: 'churn_model',
      request: { features: {} },
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(onSuccess).toHaveBeenCalled();
  });

  it('calls onError callback', async () => {
    const error = new Error('Prediction failed');
    vi.mocked(predictionsApi.predict).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();
    const onError = vi.fn();

    const { result } = renderHook(() => usePredict({ onError }), { wrapper });

    result.current.mutate({
      modelName: 'churn_model',
      request: { features: {} },
    });

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(onError).toHaveBeenCalled();
  });
});

describe('useBatchPredict', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('makes batch predictions successfully', async () => {
    vi.mocked(predictionsApi.predictBatch).mockResolvedValueOnce(mockBatchPredictionResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useBatchPredict(), { wrapper });

    result.current.mutate({
      modelName: 'churn_model',
      request: {
        instances: [
          { features: { hcp_id: 'HCP001', territory: 'Northeast' } },
          { features: { hcp_id: 'HCP002', territory: 'Southeast' } },
        ],
      },
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockBatchPredictionResponse);
    expect(predictionsApi.predictBatch).toHaveBeenCalledWith('churn_model', {
      instances: [
        { features: { hcp_id: 'HCP001', territory: 'Northeast' } },
        { features: { hcp_id: 'HCP002', territory: 'Southeast' } },
      ],
    });
  });

  it('handles batch prediction error', async () => {
    const error = new Error('Batch prediction failed');
    vi.mocked(predictionsApi.predictBatch).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useBatchPredict(), { wrapper });

    result.current.mutate({
      modelName: 'churn_model',
      request: { instances: [] },
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('handles partial failures', async () => {
    const partialResponse = { ...mockBatchPredictionResponse, error_count: 1, success_count: 1 };
    vi.mocked(predictionsApi.predictBatch).mockResolvedValueOnce(partialResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useBatchPredict(), { wrapper });

    result.current.mutate({
      modelName: 'churn_model',
      request: { instances: [] },
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.error_count).toBe(1);
  });
});

// =============================================================================
// COMBINED HOOK TESTS
// =============================================================================

describe('useModelDetail', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches both health and info', async () => {
    vi.mocked(predictionsApi.getModelHealth).mockResolvedValueOnce(mockModelHealthResponse);
    vi.mocked(predictionsApi.getModelInfo).mockResolvedValueOnce(mockModelInfoResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useModelDetail('churn_model'), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.health).toEqual(mockModelHealthResponse);
    expect(result.current.info).toEqual(mockModelInfoResponse);
  });

  it('reports loading state correctly', async () => {
    vi.mocked(predictionsApi.getModelHealth).mockImplementation(
      () => new Promise((resolve) => setTimeout(() => resolve(mockModelHealthResponse), 100))
    );
    vi.mocked(predictionsApi.getModelInfo).mockResolvedValueOnce(mockModelInfoResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useModelDetail('churn_model'), { wrapper });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => expect(result.current.isLoading).toBe(false));
  });

  it('provides individual loading states', async () => {
    vi.mocked(predictionsApi.getModelHealth).mockResolvedValueOnce(mockModelHealthResponse);
    vi.mocked(predictionsApi.getModelInfo).mockResolvedValueOnce(mockModelInfoResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useModelDetail('churn_model'), { wrapper });

    // Initially both are loading
    expect(result.current.isHealthLoading).toBe(true);
    expect(result.current.isInfoLoading).toBe(true);

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.isHealthLoading).toBe(false);
    expect(result.current.isInfoLoading).toBe(false);
  });

  it('provides refetch function', async () => {
    vi.mocked(predictionsApi.getModelHealth).mockResolvedValue(mockModelHealthResponse);
    vi.mocked(predictionsApi.getModelInfo).mockResolvedValue(mockModelInfoResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useModelDetail('churn_model'), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    // Call refetch
    result.current.refetch();

    await waitFor(() => expect(predictionsApi.getModelHealth).toHaveBeenCalledTimes(2));
  });
});

describe('useInvalidateModelCache', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('invalidates specific model cache', async () => {
    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useInvalidateModelCache(), { wrapper });

    result.current('churn_model');

    expect(invalidateSpy).toHaveBeenCalledTimes(2); // health and info
  });

  it('invalidates all predictions cache when no model specified', async () => {
    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useInvalidateModelCache(), { wrapper });

    result.current();

    expect(invalidateSpy).toHaveBeenCalled();
  });
});

// =============================================================================
// PREFETCH HELPER TESTS
// =============================================================================

describe('prefetchModelHealth', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches model health', async () => {
    vi.mocked(predictionsApi.getModelHealth).mockResolvedValueOnce(mockModelHealthResponse);

    await prefetchModelHealth('churn_model');

    expect(predictionsApi.getModelHealth).toHaveBeenCalledWith('churn_model');
  });
});

describe('prefetchModelInfo', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches model info', async () => {
    vi.mocked(predictionsApi.getModelInfo).mockResolvedValueOnce(mockModelInfoResponse);

    await prefetchModelInfo('churn_model');

    expect(predictionsApi.getModelInfo).toHaveBeenCalledWith('churn_model');
  });
});

describe('prefetchModelsStatus', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches models status', async () => {
    vi.mocked(predictionsApi.getModelsStatus).mockResolvedValueOnce(mockModelsStatusResponse);

    await prefetchModelsStatus();

    expect(predictionsApi.getModelsStatus).toHaveBeenCalled();
  });
});
