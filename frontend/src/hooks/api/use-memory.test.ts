/**
 * Memory API Query Hooks Tests
 * ============================
 *
 * Tests for TanStack Query hooks for the E2I Memory System API.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';

// Mock the API functions
vi.mock('@/api/memory', () => ({
  searchMemory: vi.fn(),
  createEpisodicMemory: vi.fn(),
  getEpisodicMemories: vi.fn(),
  getEpisodicMemory: vi.fn(),
  recordProceduralFeedback: vi.fn(),
  querySemanticPaths: vi.fn(),
  getMemoryStats: vi.fn(),
}));

// Mock query-client
vi.mock('@/lib/query-client', () => ({
  queryKeys: {
    all: ['e2i'] as const,
    memory: {
      all: () => ['e2i', 'memory'] as const,
      stats: () => ['e2i', 'memory', 'stats'] as const,
      episodic: () => ['e2i', 'memory', 'episodic'] as const,
      episodicMemory: (id: string) => ['e2i', 'memory', 'episodic', id] as const,
    },
  },
}));

import {
  useMemoryStats,
  useEpisodicMemories,
  useEpisodicMemory,
  useMemorySearch,
  useCreateEpisodicMemory,
  useProceduralFeedback,
  useSemanticPaths,
  prefetchMemoryStats,
  prefetchEpisodicMemories,
} from './use-memory';
import * as memoryApi from '@/api/memory';

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
  MemoryStatsResponse,
  EpisodicMemoryResponse,
  MemorySearchResponse,
  MemorySearchResult,
  ProceduralFeedbackResponse,
  SemanticPathResponse,
  EpisodicMemoryInput,
  ProceduralFeedbackRequest,
} from '@/types/memory';

// =============================================================================
// MOCK DATA
// =============================================================================

const mockStatsResponse: MemoryStatsResponse = {
  episodic: {
    total_memories: 1500,
    recent_24h: 45,
  },
  semantic: {
    total_entities: 250,
    total_relationships: 1200,
  },
  procedural: {
    total_procedures: 45,
    average_success_rate: 0.87,
  },
  last_updated: '2024-01-15T10:00:00Z',
};

const mockEpisodicMemory: EpisodicMemoryResponse = {
  id: 'mem_abc123',
  content: 'Dr. Smith showed interest in Kisqali.',
  event_type: 'hcp_interaction',
  brand: 'Kisqali',
  session_id: 'sess_123',
  created_at: '2024-01-15T10:00:00Z',
};

const mockEpisodicMemories: EpisodicMemoryResponse[] = [mockEpisodicMemory];

const mockSearchResult: MemorySearchResult = {
  content: 'Dr. Smith showed interest in Kisqali.',
  source: 'episodic',
  source_id: 'mem_abc123',
  score: 0.95,
  retrieval_method: 'hybrid',
  metadata: { brand: 'Kisqali' },
};

const mockSearchResponse: MemorySearchResponse = {
  query: 'HCP interaction Kisqali',
  results: [mockSearchResult],
  total_results: 1,
  retrieval_method: 'hybrid',
  search_latency_ms: 50,
  timestamp: '2024-01-15T10:00:00Z',
};

const mockProceduralFeedbackResponse: ProceduralFeedbackResponse = {
  procedure_id: 'proc_hcp_outreach',
  feedback_recorded: true,
  new_success_rate: 0.92,
  message: 'Feedback recorded successfully',
  timestamp: '2024-01-15T10:00:00Z',
};

const mockSemanticPathResponse: SemanticPathResponse = {
  paths: [
    {
      nodes: ['TRx', 'HCP_Targeting', 'Brand_Awareness'],
      relationships: ['CAUSES', 'IMPACTS'],
      confidence: 0.85,
    },
  ],
  total_paths: 1,
  max_depth_searched: 3,
  query_latency_ms: 25,
  timestamp: '2024-01-15T10:00:00Z',
};

// =============================================================================
// QUERY HOOK TESTS
// =============================================================================

describe('useMemoryStats', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches memory stats successfully', async () => {
    vi.mocked(memoryApi.getMemoryStats).mockResolvedValueOnce(mockStatsResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useMemoryStats(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockStatsResponse);
    expect(memoryApi.getMemoryStats).toHaveBeenCalled();
  });

  it('handles error state', async () => {
    const error = new Error('Service unavailable');
    vi.mocked(memoryApi.getMemoryStats).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useMemoryStats(), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('respects custom options', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useMemoryStats({ enabled: false }), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(memoryApi.getMemoryStats).not.toHaveBeenCalled();
  });
});

describe('useEpisodicMemories', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches episodic memories successfully', async () => {
    vi.mocked(memoryApi.getEpisodicMemories).mockResolvedValueOnce(mockEpisodicMemories);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useEpisodicMemories(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockEpisodicMemories);
    expect(memoryApi.getEpisodicMemories).toHaveBeenCalledWith(undefined);
  });

  it('passes params to API call', async () => {
    vi.mocked(memoryApi.getEpisodicMemories).mockResolvedValueOnce(mockEpisodicMemories);
    const { wrapper } = createWrapper();
    const params = { session_id: 'sess_123', limit: 20 };

    const { result } = renderHook(() => useEpisodicMemories(params), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(memoryApi.getEpisodicMemories).toHaveBeenCalledWith(params);
  });

  it('handles empty list', async () => {
    vi.mocked(memoryApi.getEpisodicMemories).mockResolvedValueOnce([]);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useEpisodicMemories(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual([]);
  });
});

describe('useEpisodicMemory', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches single episodic memory successfully', async () => {
    vi.mocked(memoryApi.getEpisodicMemory).mockResolvedValueOnce(mockEpisodicMemory);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useEpisodicMemory('mem_abc123'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockEpisodicMemory);
    expect(memoryApi.getEpisodicMemory).toHaveBeenCalledWith('mem_abc123');
  });

  it('is disabled when memoryId is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useEpisodicMemory(''), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(memoryApi.getEpisodicMemory).not.toHaveBeenCalled();
  });

  it('handles 404 error', async () => {
    const error = { status: 404, message: 'Memory not found' };
    vi.mocked(memoryApi.getEpisodicMemory).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useEpisodicMemory('nonexistent'), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

// =============================================================================
// MUTATION HOOK TESTS
// =============================================================================

describe('useMemorySearch', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('searches memory successfully', async () => {
    vi.mocked(memoryApi.searchMemory).mockResolvedValueOnce(mockSearchResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useMemorySearch(), { wrapper });

    const request = {
      query: 'HCP interaction Kisqali',
      k: 10,
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockSearchResponse);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(memoryApi.searchMemory).toHaveBeenCalledWith(request, expect.anything());
  });

  it('handles search error', async () => {
    const error = new Error('Search failed');
    vi.mocked(memoryApi.searchMemory).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useMemorySearch(), { wrapper });

    result.current.mutate({ query: 'test' });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('calls onSuccess callback', async () => {
    vi.mocked(memoryApi.searchMemory).mockResolvedValueOnce(mockSearchResponse);
    const { wrapper } = createWrapper();
    const onSuccess = vi.fn();

    const { result } = renderHook(() => useMemorySearch({ onSuccess }), { wrapper });

    result.current.mutate({ query: 'test' });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(onSuccess).toHaveBeenCalled();
  });
});

describe('useCreateEpisodicMemory', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('creates episodic memory successfully', async () => {
    vi.mocked(memoryApi.createEpisodicMemory).mockResolvedValueOnce(mockEpisodicMemory);
    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useCreateEpisodicMemory(), { wrapper });

    const request = {
      content: 'Dr. Smith showed interest in Kisqali.',
      event_type: 'hcp_interaction',
      brand: 'Kisqali',
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockEpisodicMemory);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(memoryApi.createEpisodicMemory).toHaveBeenCalledWith(request, expect.anything());
    expect(invalidateSpy).toHaveBeenCalled();
  });

  it('calls user onSuccess callback', async () => {
    vi.mocked(memoryApi.createEpisodicMemory).mockResolvedValueOnce(mockEpisodicMemory);
    const { wrapper } = createWrapper();
    const onSuccess = vi.fn();

    const { result } = renderHook(() => useCreateEpisodicMemory({ onSuccess }), { wrapper });

    const input: EpisodicMemoryInput = { content: 'Test content', event_type: 'test' };
    result.current.mutate(input);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(onSuccess).toHaveBeenCalled();
  });

  it('handles creation error', async () => {
    const error = new Error('Creation failed');
    vi.mocked(memoryApi.createEpisodicMemory).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCreateEpisodicMemory(), { wrapper });

    const input: EpisodicMemoryInput = { content: 'Test content', event_type: 'test' };
    result.current.mutate(input);

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useProceduralFeedback', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('records procedural feedback successfully', async () => {
    vi.mocked(memoryApi.recordProceduralFeedback).mockResolvedValueOnce(mockProceduralFeedbackResponse);
    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useProceduralFeedback(), { wrapper });

    const request: ProceduralFeedbackRequest = {
      procedure_id: 'proc_hcp_outreach',
      outcome: 'success',
      score: 0.95,
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockProceduralFeedbackResponse);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(memoryApi.recordProceduralFeedback).toHaveBeenCalledWith(request, expect.anything());
    expect(invalidateSpy).toHaveBeenCalled();
  });

  it('calls user onSuccess callback', async () => {
    vi.mocked(memoryApi.recordProceduralFeedback).mockResolvedValueOnce(mockProceduralFeedbackResponse);
    const { wrapper } = createWrapper();
    const onSuccess = vi.fn();

    const { result } = renderHook(() => useProceduralFeedback({ onSuccess }), { wrapper });

    const request: ProceduralFeedbackRequest = {
      procedure_id: 'proc_test',
      outcome: 'success',
      score: 0.85,
    };
    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(onSuccess).toHaveBeenCalled();
  });
});

describe('useSemanticPaths', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('queries semantic paths successfully', async () => {
    vi.mocked(memoryApi.querySemanticPaths).mockResolvedValueOnce(mockSemanticPathResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useSemanticPaths(), { wrapper });

    const request = {
      kpi_name: 'TRx',
      max_depth: 3,
      min_confidence: 0.6,
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockSemanticPathResponse);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(memoryApi.querySemanticPaths).toHaveBeenCalledWith(request, expect.anything());
  });

  it('handles query error', async () => {
    const error = new Error('Path query failed');
    vi.mocked(memoryApi.querySemanticPaths).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useSemanticPaths(), { wrapper });

    result.current.mutate({ kpi_name: 'TRx' });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

// =============================================================================
// PREFETCH HELPER TESTS
// =============================================================================

describe('prefetchMemoryStats', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches memory stats', async () => {
    vi.mocked(memoryApi.getMemoryStats).mockResolvedValueOnce(mockStatsResponse);
    const queryClient = createTestQueryClient();

    await prefetchMemoryStats(queryClient);

    expect(memoryApi.getMemoryStats).toHaveBeenCalled();
  });
});

describe('prefetchEpisodicMemories', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches episodic memories', async () => {
    vi.mocked(memoryApi.getEpisodicMemories).mockResolvedValueOnce(mockEpisodicMemories);
    const queryClient = createTestQueryClient();

    await prefetchEpisodicMemories(queryClient);

    expect(memoryApi.getEpisodicMemories).toHaveBeenCalledWith(undefined);
  });

  it('prefetches with params', async () => {
    vi.mocked(memoryApi.getEpisodicMemories).mockResolvedValueOnce(mockEpisodicMemories);
    const queryClient = createTestQueryClient();
    const params = { limit: 20 };

    await prefetchEpisodicMemories(queryClient, params);

    expect(memoryApi.getEpisodicMemories).toHaveBeenCalledWith(params);
  });
});
