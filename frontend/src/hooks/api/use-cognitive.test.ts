/**
 * Cognitive API Query Hooks Tests
 * ================================
 *
 * Tests for TanStack Query hooks for the E2I Cognitive Workflow API.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';
import type {
  SessionResponse,
  CognitiveQueryResponse,
  CreateSessionResponse,
  DeleteSessionResponse,
  CognitiveRAGResponse,
} from '@/types/cognitive';
import {
  QueryType,
  SessionState,
  CognitivePhase,
} from '@/types/cognitive';

// Mock the API functions
vi.mock('@/api/cognitive', () => ({
  processCognitiveQuery: vi.fn(),
  getCognitiveStatus: vi.fn(),
  createSession: vi.fn(),
  getSession: vi.fn(),
  deleteSession: vi.fn(),
  listSessions: vi.fn(),
  cognitiveRAGSearch: vi.fn(),
}));

// Mock query-client
vi.mock('@/lib/query-client', () => ({
  queryKeys: {
    all: ['e2i'] as const,
    cognitive: {
      all: () => ['e2i', 'cognitive'] as const,
      status: () => ['e2i', 'cognitive', 'status'] as const,
      sessions: () => ['e2i', 'cognitive', 'sessions'] as const,
      session: (id: string) => ['e2i', 'cognitive', 'session', id] as const,
    },
  },
}));

import {
  useCognitiveStatus,
  useSessions,
  useSession,
  useCognitiveQuery,
  useCreateSession,
  useDeleteSession,
  useCognitiveRAG,
  prefetchCognitiveStatus,
  prefetchSession,
} from './use-cognitive';
import * as cognitiveApi from '@/api/cognitive';

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

const mockStatusResponse = {
  status: 'healthy',
  version: '1.0.0',
  agents: ['orchestrator', 'causal_impact', 'gap_analyzer'],
};

const mockSession: SessionResponse = {
  context: {
    session_id: 'sess_abc123',
    user_id: 'user_001',
    brand: 'Kisqali',
    state: SessionState.ACTIVE,
    created_at: '2024-01-15T10:00:00Z',
    last_activity: '2024-01-15T12:00:00Z',
    message_count: 5,
  },
  messages: [
    {
      role: 'user',
      content: 'What factors are driving TRx decline?',
      timestamp: '2024-01-15T10:00:00Z',
    },
  ],
  evidence_trail: [],
  memory_stats: {},
};

const mockSessionsResponse: { sessions: SessionResponse[]; total: number } = {
  sessions: [mockSession],
  total: 1,
};

const mockQueryResponse: CognitiveQueryResponse = {
  session_id: 'sess_abc123',
  query: 'What factors are driving TRx decline?',
  response: 'TRx is declining due to increased competition and market saturation.',
  query_type: QueryType.CAUSAL,
  confidence: 0.85,
  agent_used: 'causal_impact',
  evidence: [
    {
      content: 'Market analysis shows increased competition',
      source: 'trend_analysis',
      relevance_score: 0.9,
      retrieval_method: 'semantic_search',
    },
  ],
  phases_completed: [CognitivePhase.SUMMARIZE, CognitivePhase.EXECUTE],
  processing_time_ms: 1500,
  timestamp: '2024-01-15T12:00:00Z',
};

const mockCreateSessionResponse: CreateSessionResponse = {
  session_id: 'sess_new123',
  state: SessionState.ACTIVE,
  created_at: '2024-01-16T10:00:00Z',
  expires_at: '2024-01-17T10:00:00Z',
};

const mockDeleteSessionResponse: DeleteSessionResponse = {
  session_id: 'sess_abc123',
  deleted: true,
  timestamp: '2024-01-16T11:00:00Z',
};

const mockRAGResponse: CognitiveRAGResponse = {
  response: 'TRx trends show a 15% increase in Q4.',
  evidence: [
    {
      content: 'Q4 analysis shows growth',
      source: 'doc1',
      score: 0.92,
    },
  ],
  hop_count: 2,
  visualization_config: {},
  routed_agents: ['causal_impact'],
  entities: ['TRx', 'Q4'],
  intent: 'trend_analysis',
  rewritten_query: 'TRx trend Kisqali Q4 2024',
  dspy_signals: [],
  worth_remembering: true,
  latency_ms: 500,
};

// =============================================================================
// QUERY HOOK TESTS
// =============================================================================

describe('useCognitiveStatus', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches cognitive status successfully', async () => {
    vi.mocked(cognitiveApi.getCognitiveStatus).mockResolvedValueOnce(mockStatusResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCognitiveStatus(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockStatusResponse);
    expect(cognitiveApi.getCognitiveStatus).toHaveBeenCalled();
  });

  it('handles error state', async () => {
    const error = new Error('Service unavailable');
    vi.mocked(cognitiveApi.getCognitiveStatus).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCognitiveStatus(), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('respects custom options', async () => {
    vi.mocked(cognitiveApi.getCognitiveStatus).mockResolvedValueOnce(mockStatusResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCognitiveStatus({ enabled: false }), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(cognitiveApi.getCognitiveStatus).not.toHaveBeenCalled();
  });
});

describe('useSessions', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches sessions list successfully', async () => {
    vi.mocked(cognitiveApi.listSessions).mockResolvedValueOnce(mockSessionsResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useSessions(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockSessionsResponse);
    expect(cognitiveApi.listSessions).toHaveBeenCalledWith(undefined);
  });

  it('passes params to API call', async () => {
    vi.mocked(cognitiveApi.listSessions).mockResolvedValueOnce(mockSessionsResponse);
    const { wrapper } = createWrapper();
    const params = { user_id: 'user_001' };

    const { result } = renderHook(() => useSessions(params), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(cognitiveApi.listSessions).toHaveBeenCalledWith(params);
  });

  it('handles empty sessions list', async () => {
    vi.mocked(cognitiveApi.listSessions).mockResolvedValueOnce({ sessions: [], total: 0 });
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useSessions(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.total).toBe(0);
  });
});

describe('useSession', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches single session successfully', async () => {
    vi.mocked(cognitiveApi.getSession).mockResolvedValueOnce(mockSession);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useSession('sess_abc123'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockSession);
    expect(cognitiveApi.getSession).toHaveBeenCalledWith('sess_abc123');
  });

  it('is disabled when sessionId is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useSession(''), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(cognitiveApi.getSession).not.toHaveBeenCalled();
  });

  it('handles 404 error', async () => {
    const error = { status: 404, message: 'Session not found' };
    vi.mocked(cognitiveApi.getSession).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useSession('nonexistent'), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

// =============================================================================
// MUTATION HOOK TESTS
// =============================================================================

describe('useCognitiveQuery', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('processes cognitive query successfully', async () => {
    vi.mocked(cognitiveApi.processCognitiveQuery).mockResolvedValueOnce(mockQueryResponse);
    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useCognitiveQuery(), { wrapper });

    const request = {
      query: 'What factors are driving TRx decline?',
      brand: 'Kisqali',
      include_evidence: true,
      session_id: 'sess_abc123',
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockQueryResponse);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(cognitiveApi.processCognitiveQuery).toHaveBeenCalledWith(request, expect.anything());

    // Should invalidate session when session_id provided
    expect(invalidateSpy).toHaveBeenCalled();
  });

  it('does not invalidate when no session_id', async () => {
    vi.mocked(cognitiveApi.processCognitiveQuery).mockResolvedValueOnce(mockQueryResponse);
    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useCognitiveQuery(), { wrapper });

    const request = {
      query: 'What factors are driving TRx decline?',
      brand: 'Kisqali',
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    // Should not invalidate when no session_id
    expect(invalidateSpy).not.toHaveBeenCalled();
  });

  it('handles mutation error', async () => {
    const error = new Error('Query processing failed');
    vi.mocked(cognitiveApi.processCognitiveQuery).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCognitiveQuery(), { wrapper });

    result.current.mutate({ query: 'test' });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useCreateSession', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('creates session successfully', async () => {
    vi.mocked(cognitiveApi.createSession).mockResolvedValueOnce(mockCreateSessionResponse);
    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useCreateSession(), { wrapper });

    const request = {
      user_id: 'user_001',
      brand: 'Kisqali',
      region: 'northeast',
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockCreateSessionResponse);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(cognitiveApi.createSession).toHaveBeenCalledWith(request, expect.anything());
    expect(invalidateSpy).toHaveBeenCalled();
  });

  it('calls user onSuccess callback', async () => {
    vi.mocked(cognitiveApi.createSession).mockResolvedValueOnce(mockCreateSessionResponse);
    const { wrapper } = createWrapper();
    const onSuccess = vi.fn();

    const { result } = renderHook(() => useCreateSession({ onSuccess }), { wrapper });

    result.current.mutate({ user_id: 'user_001' });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(onSuccess).toHaveBeenCalled();
  });
});

describe('useDeleteSession', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('deletes session successfully', async () => {
    vi.mocked(cognitiveApi.deleteSession).mockResolvedValueOnce(mockDeleteSessionResponse);
    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');
    const removeSpy = vi.spyOn(queryClient, 'removeQueries');

    const { result } = renderHook(() => useDeleteSession(), { wrapper });

    result.current.mutate('sess_abc123');

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockDeleteSessionResponse);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(cognitiveApi.deleteSession).toHaveBeenCalledWith('sess_abc123', expect.anything());
    expect(invalidateSpy).toHaveBeenCalled();
    expect(removeSpy).toHaveBeenCalled();
  });

  it('handles delete error', async () => {
    const error = { status: 404, message: 'Session not found' };
    vi.mocked(cognitiveApi.deleteSession).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useDeleteSession(), { wrapper });

    result.current.mutate('nonexistent');

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useCognitiveRAG', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('performs RAG search successfully', async () => {
    vi.mocked(cognitiveApi.cognitiveRAGSearch).mockResolvedValueOnce(mockRAGResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCognitiveRAG(), { wrapper });

    const request = {
      query: 'What is driving TRx trend for Kisqali?',
      conversation_id: 'conv_123',
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockRAGResponse);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(cognitiveApi.cognitiveRAGSearch).toHaveBeenCalledWith(request, expect.anything());
  });

  it('handles RAG error', async () => {
    const error = new Error('RAG search failed');
    vi.mocked(cognitiveApi.cognitiveRAGSearch).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCognitiveRAG(), { wrapper });

    result.current.mutate({ query: 'test' });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('calls onSuccess callback', async () => {
    vi.mocked(cognitiveApi.cognitiveRAGSearch).mockResolvedValueOnce(mockRAGResponse);
    const { wrapper } = createWrapper();
    const onSuccess = vi.fn();

    const { result } = renderHook(() => useCognitiveRAG({ onSuccess }), { wrapper });

    result.current.mutate({ query: 'test' });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(onSuccess).toHaveBeenCalled();
  });
});

// =============================================================================
// PREFETCH HELPER TESTS
// =============================================================================

describe('prefetchCognitiveStatus', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches cognitive status', async () => {
    vi.mocked(cognitiveApi.getCognitiveStatus).mockResolvedValueOnce(mockStatusResponse);
    const queryClient = createTestQueryClient();

    await prefetchCognitiveStatus(queryClient);

    expect(cognitiveApi.getCognitiveStatus).toHaveBeenCalled();
  });
});

describe('prefetchSession', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches single session', async () => {
    vi.mocked(cognitiveApi.getSession).mockResolvedValueOnce(mockSession);
    const queryClient = createTestQueryClient();

    await prefetchSession(queryClient, 'sess_abc123');

    expect(cognitiveApi.getSession).toHaveBeenCalledWith('sess_abc123');
  });
});
