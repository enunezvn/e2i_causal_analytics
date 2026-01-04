/**
 * RAG API Query Hooks Tests
 * =========================
 *
 * Tests for TanStack Query hooks for the E2I Hybrid RAG API.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';

// Mock the API functions
vi.mock('@/api/rag', () => ({
  searchRAG: vi.fn(),
  queryRAG: vi.fn(),
  extractEntities: vi.fn(),
  getCausalSubgraph: vi.fn(),
  getCausalPaths: vi.fn(),
  getRAGStats: vi.fn(),
  getRAGHealth: vi.fn(),
}));

// Mock query-client
vi.mock('@/lib/query-client', () => ({
  queryKeys: {
    all: ['e2i'] as const,
    rag: {
      all: () => ['e2i', 'rag'] as const,
      stats: () => ['e2i', 'rag', 'stats'] as const,
      health: () => ['e2i', 'rag', 'health'] as const,
      subgraph: (entity: string) => ['e2i', 'rag', 'subgraph', entity] as const,
      paths: (source: string, target: string) => ['e2i', 'rag', 'paths', source, target] as const,
    },
  },
}));

import {
  useRAGStats,
  useRAGHealth,
  useCausalSubgraph,
  useCausalPaths,
  useRAGSearch,
  useRAGQuery,
  useExtractEntities,
  prefetchRAGStats,
  prefetchCausalSubgraph,
  prefetchCausalPaths,
} from './use-rag';
import * as ragApi from '@/api/rag';

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

const mockStatsResponse = {
  total_searches: 1500,
  avg_response_time_ms: 250,
  cache_hit_rate: 0.75,
  search_by_mode: { vector: 800, sparse: 400, hybrid: 300 },
  period_hours: 24,
};

const mockHealthResponse = {
  status: 'healthy',
  vector_store: 'healthy',
  sparse_index: 'healthy',
  graph_store: 'healthy',
  embedding_model: 'loaded',
};

const mockSubgraphResponse = {
  entity: 'kisqali',
  nodes: [
    { id: 'kisqali', type: 'brand', label: 'Kisqali' },
    { id: 'trx', type: 'kpi', label: 'TRx' },
  ],
  edges: [{ source: 'kisqali', target: 'trx', relationship: 'IMPACTS' }],
  node_count: 2,
  edge_count: 1,
  depth: 2,
};

const mockPathsResponse = {
  source: 'hcp_engagement',
  target: 'trx',
  paths: [
    {
      nodes: ['hcp_engagement', 'conversion_rate', 'trx'],
      relationships: ['CAUSES', 'DRIVES'],
      confidence: 0.85,
    },
  ],
  total_paths: 1,
  shortest_path_length: 3,
};

const mockSearchResponse = {
  results: [
    {
      id: 'doc_001',
      content: 'TRx trends for Kisqali show positive growth',
      score: 0.92,
      metadata: { source: 'analytics_report', date: '2024-01-15' },
    },
  ],
  total: 1,
  query: 'Kisqali TRx',
  search_time_ms: 150,
};

const mockEntitiesResponse = {
  brands: ['Kisqali'],
  kpis: ['TRx'],
  regions: ['Northeast'],
  time_references: ['Q4 2024'],
};

// =============================================================================
// QUERY HOOK TESTS
// =============================================================================

describe('useRAGStats', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches RAG stats successfully', async () => {
    vi.mocked(ragApi.getRAGStats).mockResolvedValueOnce(mockStatsResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useRAGStats(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockStatsResponse);
    expect(ragApi.getRAGStats).toHaveBeenCalledWith(undefined);
  });

  it('passes period hours to API', async () => {
    vi.mocked(ragApi.getRAGStats).mockResolvedValueOnce(mockStatsResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useRAGStats(48), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(ragApi.getRAGStats).toHaveBeenCalledWith(48);
  });

  it('handles error state', async () => {
    const error = new Error('Service unavailable');
    vi.mocked(ragApi.getRAGStats).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useRAGStats(), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('respects custom options', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useRAGStats(24, { enabled: false }), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(ragApi.getRAGStats).not.toHaveBeenCalled();
  });
});

describe('useRAGHealth', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches RAG health successfully', async () => {
    vi.mocked(ragApi.getRAGHealth).mockResolvedValueOnce(mockHealthResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useRAGHealth(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockHealthResponse);
    expect(ragApi.getRAGHealth).toHaveBeenCalled();
  });

  it('handles unhealthy status', async () => {
    const unhealthyResponse = { ...mockHealthResponse, status: 'degraded', vector_store: 'unhealthy' };
    vi.mocked(ragApi.getRAGHealth).mockResolvedValueOnce(unhealthyResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useRAGHealth(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.status).toBe('degraded');
  });
});

describe('useCausalSubgraph', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches causal subgraph successfully', async () => {
    vi.mocked(ragApi.getCausalSubgraph).mockResolvedValueOnce(mockSubgraphResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCausalSubgraph('kisqali'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockSubgraphResponse);
    expect(ragApi.getCausalSubgraph).toHaveBeenCalledWith('kisqali', undefined);
  });

  it('passes depth to API', async () => {
    vi.mocked(ragApi.getCausalSubgraph).mockResolvedValueOnce(mockSubgraphResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCausalSubgraph('kisqali', 3), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(ragApi.getCausalSubgraph).toHaveBeenCalledWith('kisqali', 3);
  });

  it('is disabled when entity is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCausalSubgraph(''), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(ragApi.getCausalSubgraph).not.toHaveBeenCalled();
  });
});

describe('useCausalPaths', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches causal paths successfully', async () => {
    vi.mocked(ragApi.getCausalPaths).mockResolvedValueOnce(mockPathsResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCausalPaths('hcp_engagement', 'trx'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockPathsResponse);
    expect(ragApi.getCausalPaths).toHaveBeenCalledWith('hcp_engagement', 'trx', undefined);
  });

  it('passes maxDepth to API', async () => {
    vi.mocked(ragApi.getCausalPaths).mockResolvedValueOnce(mockPathsResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCausalPaths('hcp_engagement', 'trx', 5), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(ragApi.getCausalPaths).toHaveBeenCalledWith('hcp_engagement', 'trx', 5);
  });

  it('is disabled when source is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCausalPaths('', 'trx'), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(ragApi.getCausalPaths).not.toHaveBeenCalled();
  });

  it('is disabled when target is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCausalPaths('hcp_engagement', ''), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(ragApi.getCausalPaths).not.toHaveBeenCalled();
  });

  it('handles no paths found', async () => {
    const emptyPathsResponse = { ...mockPathsResponse, paths: [], total_paths: 0 };
    vi.mocked(ragApi.getCausalPaths).mockResolvedValueOnce(emptyPathsResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCausalPaths('unknown', 'entity'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.total_paths).toBe(0);
  });
});

// =============================================================================
// MUTATION HOOK TESTS
// =============================================================================

describe('useRAGSearch', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('performs RAG search successfully', async () => {
    vi.mocked(ragApi.searchRAG).mockResolvedValueOnce(mockSearchResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useRAGSearch(), { wrapper });

    const request = {
      query: 'Kisqali TRx trends',
      mode: 'hybrid',
      top_k: 10,
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockSearchResponse);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(ragApi.searchRAG).toHaveBeenCalledWith(request, expect.anything());
  });

  it('handles search error', async () => {
    const error = new Error('Search failed');
    vi.mocked(ragApi.searchRAG).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useRAGSearch(), { wrapper });

    result.current.mutate({ query: 'test' });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('calls onSuccess callback', async () => {
    vi.mocked(ragApi.searchRAG).mockResolvedValueOnce(mockSearchResponse);
    const { wrapper } = createWrapper();
    const onSuccess = vi.fn();

    const { result } = renderHook(() => useRAGSearch({ onSuccess }), { wrapper });

    result.current.mutate({ query: 'test' });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(onSuccess).toHaveBeenCalled();
  });
});

describe('useRAGQuery', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('performs simple RAG query successfully', async () => {
    vi.mocked(ragApi.queryRAG).mockResolvedValueOnce(mockSearchResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useRAGQuery(), { wrapper });

    result.current.mutate({
      query: 'Kisqali efficacy data',
      params: { top_k: 5 },
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockSearchResponse);
    expect(ragApi.queryRAG).toHaveBeenCalledWith('Kisqali efficacy data', { top_k: 5 });
  });

  it('works without params', async () => {
    vi.mocked(ragApi.queryRAG).mockResolvedValueOnce(mockSearchResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useRAGQuery(), { wrapper });

    result.current.mutate({ query: 'test query' });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(ragApi.queryRAG).toHaveBeenCalledWith('test query', undefined);
  });

  it('handles query error', async () => {
    const error = new Error('Query failed');
    vi.mocked(ragApi.queryRAG).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useRAGQuery(), { wrapper });

    result.current.mutate({ query: 'test' });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useExtractEntities', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('extracts entities successfully', async () => {
    vi.mocked(ragApi.extractEntities).mockResolvedValueOnce(mockEntitiesResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useExtractEntities(), { wrapper });

    result.current.mutate({ query: 'TRx trend for Kisqali in Northeast' });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockEntitiesResponse);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(ragApi.extractEntities).toHaveBeenCalledWith(
      { query: 'TRx trend for Kisqali in Northeast' },
      expect.anything()
    );
  });

  it('handles extraction error', async () => {
    const error = new Error('Extraction failed');
    vi.mocked(ragApi.extractEntities).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useExtractEntities(), { wrapper });

    result.current.mutate({ query: 'test' });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('handles empty entities', async () => {
    const emptyResponse = { brands: [], kpis: [], regions: [], time_references: [] };
    vi.mocked(ragApi.extractEntities).mockResolvedValueOnce(emptyResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useExtractEntities(), { wrapper });

    result.current.mutate({ query: 'random text with no entities' });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.brands).toHaveLength(0);
  });
});

// =============================================================================
// PREFETCH HELPER TESTS
// =============================================================================

describe('prefetchRAGStats', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches RAG stats', async () => {
    vi.mocked(ragApi.getRAGStats).mockResolvedValueOnce(mockStatsResponse);
    const queryClient = createTestQueryClient();

    await prefetchRAGStats(queryClient);

    expect(ragApi.getRAGStats).toHaveBeenCalledWith(undefined);
  });

  it('prefetches with period hours', async () => {
    vi.mocked(ragApi.getRAGStats).mockResolvedValueOnce(mockStatsResponse);
    const queryClient = createTestQueryClient();

    await prefetchRAGStats(queryClient, 48);

    expect(ragApi.getRAGStats).toHaveBeenCalledWith(48);
  });
});

describe('prefetchCausalSubgraph', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches causal subgraph', async () => {
    vi.mocked(ragApi.getCausalSubgraph).mockResolvedValueOnce(mockSubgraphResponse);
    const queryClient = createTestQueryClient();

    await prefetchCausalSubgraph(queryClient, 'kisqali');

    expect(ragApi.getCausalSubgraph).toHaveBeenCalledWith('kisqali', undefined);
  });

  it('prefetches with depth', async () => {
    vi.mocked(ragApi.getCausalSubgraph).mockResolvedValueOnce(mockSubgraphResponse);
    const queryClient = createTestQueryClient();

    await prefetchCausalSubgraph(queryClient, 'kisqali', 3);

    expect(ragApi.getCausalSubgraph).toHaveBeenCalledWith('kisqali', 3);
  });
});

describe('prefetchCausalPaths', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches causal paths', async () => {
    vi.mocked(ragApi.getCausalPaths).mockResolvedValueOnce(mockPathsResponse);
    const queryClient = createTestQueryClient();

    await prefetchCausalPaths(queryClient, 'hcp_engagement', 'trx');

    expect(ragApi.getCausalPaths).toHaveBeenCalledWith('hcp_engagement', 'trx', undefined);
  });

  it('prefetches with maxDepth', async () => {
    vi.mocked(ragApi.getCausalPaths).mockResolvedValueOnce(mockPathsResponse);
    const queryClient = createTestQueryClient();

    await prefetchCausalPaths(queryClient, 'hcp_engagement', 'trx', 5);

    expect(ragApi.getCausalPaths).toHaveBeenCalledWith('hcp_engagement', 'trx', 5);
  });
});
