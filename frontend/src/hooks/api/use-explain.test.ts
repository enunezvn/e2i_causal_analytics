/**
 * Explain API Query Hooks Tests
 * =============================
 *
 * Tests for TanStack Query hooks for the E2I Model Interpretability API.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';
import type {
  ListExplainableModelsResponse,
  ExplanationHistoryResponse,
  ExplainHealthResponse,
  ExplainResponse,
  BatchExplainResponse,
} from '@/types/explain';
import { ModelType } from '@/types/explain';

// Mock the API functions
vi.mock('@/api/explain', () => ({
  getExplanation: vi.fn(),
  getBatchExplanations: vi.fn(),
  getExplanationHistory: vi.fn(),
  listExplainableModels: vi.fn(),
  getExplainHealth: vi.fn(),
}));

// Mock query-client
vi.mock('@/lib/query-client', () => ({
  queryKeys: {
    all: ['e2i'] as const,
    explain: {
      all: () => ['e2i', 'explain'] as const,
      models: () => ['e2i', 'explain', 'models'] as const,
      history: (patientId: string) => ['e2i', 'explain', 'history', patientId] as const,
      health: () => ['e2i', 'explain', 'health'] as const,
    },
  },
}));

import {
  useExplainableModels,
  useExplanationHistory,
  useExplainHealth,
  useExplain,
  useBatchExplain,
  prefetchExplainableModels,
  prefetchExplanationHistory,
} from './use-explain';
import * as explainApi from '@/api/explain';

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

const mockModelsResponse: ListExplainableModelsResponse = {
  supported_models: [
    { model_type: ModelType.PROPENSITY, latest_version: '1.0.0', explainer_type: 'TreeExplainer', avg_latency_ms: 50 },
    { model_type: ModelType.CHURN_PREDICTION, latest_version: '1.0.0', explainer_type: 'TreeExplainer', avg_latency_ms: 45 },
  ],
  total_models: 2,
};

const mockExplanationResponse: ExplainResponse = {
  explanation_id: 'exp_new001',
  request_timestamp: '2024-01-15T10:00:00Z',
  patient_id: 'patient_123',
  model_type: ModelType.PROPENSITY,
  model_version_id: 'propensity_v1',
  prediction_class: 'high_propensity',
  prediction_probability: 0.85,
  base_value: 0.45,
  top_features: [
    { feature_name: 'visit_frequency', feature_value: 12, shap_value: 0.15, contribution_direction: 'positive', contribution_rank: 1 },
    { feature_name: 'engagement_score', feature_value: 0.8, shap_value: 0.12, contribution_direction: 'positive', contribution_rank: 2 },
  ],
  shap_sum: 0.27,
  computation_time_ms: 250,
  audit_stored: true,
};

const mockHistoryResponse: ExplanationHistoryResponse = {
  patient_id: 'patient_123',
  total_explanations: 1,
  explanations: [mockExplanationResponse],
};

const mockHealthResponse: ExplainHealthResponse = {
  status: 'healthy',
  service: 'explain-service',
  version: '1.0.0',
  timestamp: '2024-01-15T10:00:00Z',
  dependencies: {
    bentoml: 'connected',
    feast: 'connected',
    shap_explainer: 'loaded',
    ml_shap_analyses_db: 'connected',
  },
};

const mockBatchExplanationResponse: BatchExplainResponse = {
  batch_id: 'batch_001',
  total_requests: 1,
  successful: 1,
  failed: 0,
  explanations: [mockExplanationResponse],
  errors: [],
  total_time_ms: 500,
};

// =============================================================================
// QUERY HOOK TESTS
// =============================================================================

describe('useExplainableModels', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches explainable models successfully', async () => {
    vi.mocked(explainApi.listExplainableModels).mockResolvedValueOnce(mockModelsResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useExplainableModels(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockModelsResponse);
    expect(explainApi.listExplainableModels).toHaveBeenCalled();
  });

  it('handles error state', async () => {
    const error = new Error('Service unavailable');
    vi.mocked(explainApi.listExplainableModels).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useExplainableModels(), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('respects custom options', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useExplainableModels({ enabled: false }), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(explainApi.listExplainableModels).not.toHaveBeenCalled();
  });
});

describe('useExplanationHistory', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches explanation history successfully', async () => {
    vi.mocked(explainApi.getExplanationHistory).mockResolvedValueOnce(mockHistoryResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useExplanationHistory('patient_123'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockHistoryResponse);
    expect(explainApi.getExplanationHistory).toHaveBeenCalledWith({
      patient_id: 'patient_123',
      model_type: undefined,
      limit: undefined,
    });
  });

  it('passes model type and limit to API call', async () => {
    vi.mocked(explainApi.getExplanationHistory).mockResolvedValueOnce(mockHistoryResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(
      () => useExplanationHistory('patient_123', 'propensity' as any, 5),
      { wrapper }
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(explainApi.getExplanationHistory).toHaveBeenCalledWith({
      patient_id: 'patient_123',
      model_type: ModelType.PROPENSITY,
      limit: 5,
    });
  });

  it('is disabled when patientId is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useExplanationHistory(''), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(explainApi.getExplanationHistory).not.toHaveBeenCalled();
  });

  it('handles empty history', async () => {
    const emptyHistoryResponse: ExplanationHistoryResponse = {
      patient_id: 'patient_123',
      total_explanations: 0,
      explanations: [],
    };
    vi.mocked(explainApi.getExplanationHistory).mockResolvedValueOnce(emptyHistoryResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useExplanationHistory('patient_123'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.total_explanations).toBe(0);
  });
});

describe('useExplainHealth', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches explain health successfully', async () => {
    vi.mocked(explainApi.getExplainHealth).mockResolvedValueOnce(mockHealthResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useExplainHealth(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockHealthResponse);
    expect(explainApi.getExplainHealth).toHaveBeenCalled();
  });

  it('handles service unavailable', async () => {
    const error = { status: 503, message: 'Service Unavailable' };
    vi.mocked(explainApi.getExplainHealth).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useExplainHealth(), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

// =============================================================================
// MUTATION HOOK TESTS
// =============================================================================

describe('useExplain', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('gets explanation successfully', async () => {
    vi.mocked(explainApi.getExplanation).mockResolvedValueOnce(mockExplanationResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useExplain(), { wrapper });

    const request = {
      patient_id: 'patient_123',
      model_type: ModelType.PROPENSITY,
      top_k: 5,
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockExplanationResponse);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(explainApi.getExplanation).toHaveBeenCalledWith(request, expect.anything());
  });

  it('handles explanation error', async () => {
    const error = new Error('Model not found');
    vi.mocked(explainApi.getExplanation).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useExplain(), { wrapper });

    result.current.mutate({ patient_id: 'invalid', model_type: ModelType.PROPENSITY });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('calls onSuccess callback', async () => {
    vi.mocked(explainApi.getExplanation).mockResolvedValueOnce(mockExplanationResponse);
    const { wrapper } = createWrapper();
    const onSuccess = vi.fn();

    const { result } = renderHook(() => useExplain({ onSuccess }), { wrapper });

    result.current.mutate({ patient_id: 'patient_123', model_type: ModelType.PROPENSITY });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(onSuccess).toHaveBeenCalled();
  });

  it('calls onError callback', async () => {
    const error = new Error('Explanation failed');
    vi.mocked(explainApi.getExplanation).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();
    const onError = vi.fn();

    const { result } = renderHook(() => useExplain({ onError }), { wrapper });

    result.current.mutate({ patient_id: 'patient_123', model_type: ModelType.PROPENSITY });

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(onError).toHaveBeenCalled();
  });
});

describe('useBatchExplain', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('gets batch explanations successfully', async () => {
    vi.mocked(explainApi.getBatchExplanations).mockResolvedValueOnce(mockBatchExplanationResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useBatchExplain(), { wrapper });

    const request = {
      requests: [
        { patient_id: 'p1', model_type: ModelType.PROPENSITY },
        { patient_id: 'p2', model_type: ModelType.PROPENSITY },
      ],
      parallel: true,
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockBatchExplanationResponse);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(explainApi.getBatchExplanations).toHaveBeenCalledWith(request, expect.anything());
  });

  it('handles batch error', async () => {
    const error = new Error('Batch processing failed');
    vi.mocked(explainApi.getBatchExplanations).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useBatchExplain(), { wrapper });

    result.current.mutate({ requests: [] });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('handles partial failures', async () => {
    const partialResponse: BatchExplainResponse = {
      batch_id: 'batch_partial',
      total_requests: 2,
      successful: 1,
      failed: 1,
      explanations: [mockExplanationResponse],
      errors: [{ patient_id: 'p2', error: 'Model unavailable' }],
      total_time_ms: 500,
    };
    vi.mocked(explainApi.getBatchExplanations).mockResolvedValueOnce(partialResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useBatchExplain(), { wrapper });

    result.current.mutate({
      requests: [
        { patient_id: 'p1', model_type: ModelType.PROPENSITY },
        { patient_id: 'p2', model_type: ModelType.PROPENSITY },
      ],
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.failed).toBe(1);
  });
});

// =============================================================================
// PREFETCH HELPER TESTS
// =============================================================================

describe('prefetchExplainableModels', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches explainable models', async () => {
    vi.mocked(explainApi.listExplainableModels).mockResolvedValueOnce(mockModelsResponse);
    const queryClient = createTestQueryClient();

    await prefetchExplainableModels(queryClient);

    expect(explainApi.listExplainableModels).toHaveBeenCalled();
  });
});

describe('prefetchExplanationHistory', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches explanation history', async () => {
    vi.mocked(explainApi.getExplanationHistory).mockResolvedValueOnce(mockHistoryResponse);
    const queryClient = createTestQueryClient();

    await prefetchExplanationHistory(queryClient, 'patient_123');

    expect(explainApi.getExplanationHistory).toHaveBeenCalledWith({
      patient_id: 'patient_123',
      model_type: undefined,
      limit: undefined,
    });
  });

  it('prefetches with model type and limit', async () => {
    vi.mocked(explainApi.getExplanationHistory).mockResolvedValueOnce(mockHistoryResponse);
    const queryClient = createTestQueryClient();

    await prefetchExplanationHistory(queryClient, 'patient_123', 'propensity' as any, 10);

    expect(explainApi.getExplanationHistory).toHaveBeenCalledWith({
      patient_id: 'patient_123',
      model_type: 'propensity',
      limit: 10,
    });
  });
});
