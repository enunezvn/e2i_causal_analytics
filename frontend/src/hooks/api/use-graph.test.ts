/**
 * Graph API Query Hooks Tests
 * ===========================
 *
 * Tests for TanStack Query hooks for the E2I Knowledge Graph API.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';

// Mock the API functions
vi.mock('@/api/graph', () => ({
  listNodes: vi.fn(),
  getNode: vi.fn(),
  getNodeNetwork: vi.fn(),
  listRelationships: vi.fn(),
  traverseGraph: vi.fn(),
  queryCausalChains: vi.fn(),
  executeCypherQuery: vi.fn(),
  addEpisode: vi.fn(),
  searchGraph: vi.fn(),
  getGraphStats: vi.fn(),
  getGraphHealth: vi.fn(),
}));

// Mock query-client
vi.mock('@/lib/query-client', () => ({
  queryKeys: {
    all: ['e2i'] as const,
    graph: {
      all: () => ['e2i', 'graph'] as const,
      nodes: () => ['e2i', 'graph', 'nodes'] as const,
      node: (id: string) => ['e2i', 'graph', 'node', id] as const,
      nodeNetwork: (id: string) => ['e2i', 'graph', 'nodeNetwork', id] as const,
      relationships: () => ['e2i', 'graph', 'relationships'] as const,
      stats: () => ['e2i', 'graph', 'stats'] as const,
      search: (query: string) => ['e2i', 'graph', 'search', query] as const,
    },
  },
}));

import {
  useNodes,
  useNode,
  useNodeNetwork,
  useRelationships,
  useGraphStats,
  useGraphHealth,
  useGraphSearch,
  useTraverseGraph,
  useCausalChains,
  useCypherQuery,
  useAddEpisode,
  prefetchNodes,
  prefetchNode,
  prefetchGraphStats,
} from './use-graph';
import * as graphApi from '@/api/graph';

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

const mockNodesResponse = {
  nodes: [
    { id: 'node1', entity_type: 'HCP', name: 'Dr. Smith', properties: {} },
    { id: 'node2', entity_type: 'Brand', name: 'Kisqali', properties: {} },
  ],
  total: 2,
  page: 1,
  limit: 20,
};

const mockNode = {
  id: 'node1',
  entity_type: 'HCP',
  name: 'Dr. Smith',
  properties: { specialty: 'Oncology' },
};

const mockNetworkResponse = {
  central_node: mockNode,
  connected_nodes: {
    HCP: [],
    Brand: [{ id: 'brand1', name: 'Kisqali' }],
  },
  relationships: [],
  total_connections: 1,
};

const mockRelationshipsResponse = {
  relationships: [
    { id: 'rel1', source_id: 'node1', target_id: 'node2', type: 'PRESCRIBES', confidence: 0.9 },
  ],
  total: 1,
  page: 1,
  limit: 20,
};

const mockStatsResponse = {
  total_nodes: 1000,
  total_relationships: 5000,
  node_types: { HCP: 500, Brand: 100, Patient: 400 },
  relationship_types: { PRESCRIBES: 2000, CAUSES: 3000 },
  total_episodes: 150,
  total_communities: 25,
};

const mockHealthResponse = {
  status: 'healthy',
  graphiti_available: true,
  falkordb_connected: true,
  websocket_clients: 3,
};

const mockSearchResponse = {
  results: [
    { id: 'node1', entity_type: 'HCP', name: 'Dr. Smith', score: 0.95 },
  ],
  total: 1,
  query: 'oncology specialists',
};

const mockTraverseResponse = {
  nodes: [mockNode],
  relationships: [],
  paths: [],
  total_nodes: 1,
  total_relationships: 0,
};

const mockCausalChainResponse = {
  chains: [
    {
      id: 'chain1',
      nodes: ['kpi_trx', 'hcp_targeting'],
      confidence: 0.85,
      chain_length: 2,
    },
  ],
  total: 1,
};

const mockCypherResponse = {
  results: [{ h: { id: 'hcp1', name: 'Dr. Smith' } }],
  columns: ['h'],
  row_count: 1,
  execution_time_ms: 15,
};

const mockEpisodeResponse = {
  episode_id: 'ep_123',
  entities_extracted: 3,
  relationships_created: 2,
  content_summary: 'Prescription event recorded',
};

// =============================================================================
// QUERY HOOK TESTS
// =============================================================================

describe('useNodes', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches nodes successfully', async () => {
    vi.mocked(graphApi.listNodes).mockResolvedValueOnce(mockNodesResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useNodes(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockNodesResponse);
    expect(graphApi.listNodes).toHaveBeenCalledWith(undefined);
  });

  it('passes params to API call', async () => {
    vi.mocked(graphApi.listNodes).mockResolvedValueOnce(mockNodesResponse);
    const { wrapper } = createWrapper();
    const params = { entity_types: 'HCP,Brand', limit: 50 };

    const { result } = renderHook(() => useNodes(params), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(graphApi.listNodes).toHaveBeenCalledWith(params);
  });

  it('handles error state', async () => {
    const error = new Error('Network error');
    vi.mocked(graphApi.listNodes).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useNodes(), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(result.current.error).toBeDefined();
  });

  it('respects custom options', async () => {
    vi.mocked(graphApi.listNodes).mockResolvedValueOnce(mockNodesResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useNodes(undefined, { enabled: false }), { wrapper });

    // Query should not run when disabled
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(graphApi.listNodes).not.toHaveBeenCalled();
  });
});

describe('useNode', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches single node successfully', async () => {
    vi.mocked(graphApi.getNode).mockResolvedValueOnce(mockNode);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useNode('node1'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockNode);
    expect(graphApi.getNode).toHaveBeenCalledWith('node1');
  });

  it('is disabled when nodeId is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useNode(''), { wrapper });

    // Query should not run for empty nodeId
    expect(result.current.fetchStatus).toBe('idle');
    expect(graphApi.getNode).not.toHaveBeenCalled();
  });

  it('handles 404 error', async () => {
    const error = { status: 404, message: 'Node not found' };
    vi.mocked(graphApi.getNode).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useNode('nonexistent'), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useNodeNetwork', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches node network successfully', async () => {
    vi.mocked(graphApi.getNodeNetwork).mockResolvedValueOnce(mockNetworkResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useNodeNetwork('node1'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockNetworkResponse);
    expect(graphApi.getNodeNetwork).toHaveBeenCalledWith('node1', undefined);
  });

  it('passes maxDepth to API call', async () => {
    vi.mocked(graphApi.getNodeNetwork).mockResolvedValueOnce(mockNetworkResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useNodeNetwork('node1', 3), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(graphApi.getNodeNetwork).toHaveBeenCalledWith('node1', 3);
  });

  it('is disabled when nodeId is empty', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useNodeNetwork(''), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(graphApi.getNodeNetwork).not.toHaveBeenCalled();
  });
});

describe('useRelationships', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches relationships successfully', async () => {
    vi.mocked(graphApi.listRelationships).mockResolvedValueOnce(mockRelationshipsResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useRelationships(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockRelationshipsResponse);
  });

  it('passes params to API call', async () => {
    vi.mocked(graphApi.listRelationships).mockResolvedValueOnce(mockRelationshipsResponse);
    const { wrapper } = createWrapper();
    const params = { relationship_types: 'CAUSES', min_confidence: 0.7 };

    const { result } = renderHook(() => useRelationships(params), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(graphApi.listRelationships).toHaveBeenCalledWith(params);
  });
});

describe('useGraphStats', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches graph stats successfully', async () => {
    vi.mocked(graphApi.getGraphStats).mockResolvedValueOnce(mockStatsResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useGraphStats(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockStatsResponse);
    expect(graphApi.getGraphStats).toHaveBeenCalled();
  });
});

describe('useGraphHealth', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches graph health successfully', async () => {
    vi.mocked(graphApi.getGraphHealth).mockResolvedValueOnce(mockHealthResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useGraphHealth(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockHealthResponse);
    expect(graphApi.getGraphHealth).toHaveBeenCalled();
  });
});

describe('useGraphSearch', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('searches graph successfully', async () => {
    vi.mocked(graphApi.searchGraph).mockResolvedValueOnce(mockSearchResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useGraphSearch('oncology specialists'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockSearchResponse);
    expect(graphApi.searchGraph).toHaveBeenCalledWith({
      query: 'oncology specialists',
    });
  });

  it('passes additional request params', async () => {
    vi.mocked(graphApi.searchGraph).mockResolvedValueOnce(mockSearchResponse);
    const { wrapper } = createWrapper();
    const request = { entity_types: ['HCP'], k: 10 };

    const { result } = renderHook(() => useGraphSearch('oncology', request), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(graphApi.searchGraph).toHaveBeenCalledWith({
      query: 'oncology',
      entity_types: ['HCP'],
      k: 10,
    });
  });

  it('is disabled for empty query', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useGraphSearch(''), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(graphApi.searchGraph).not.toHaveBeenCalled();
  });

  it('is disabled for short query (less than 2 chars)', async () => {
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useGraphSearch('a'), { wrapper });

    expect(result.current.fetchStatus).toBe('idle');
    expect(graphApi.searchGraph).not.toHaveBeenCalled();
  });

  it('is enabled for query with 2+ chars', async () => {
    vi.mocked(graphApi.searchGraph).mockResolvedValueOnce(mockSearchResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useGraphSearch('ab'), { wrapper });

    await waitFor(() => expect(graphApi.searchGraph).toHaveBeenCalled());
  });
});

// =============================================================================
// MUTATION HOOK TESTS
// =============================================================================

describe('useTraverseGraph', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('traverses graph successfully', async () => {
    vi.mocked(graphApi.traverseGraph).mockResolvedValueOnce(mockTraverseResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useTraverseGraph(), { wrapper });

    const request = {
      start_node_id: 'kpi_trx',
      relationship_types: ['CAUSES', 'IMPACTS'],
      max_depth: 3,
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockTraverseResponse);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(graphApi.traverseGraph).toHaveBeenCalledWith(request, expect.anything());
  });

  it('handles mutation error', async () => {
    const error = new Error('Traversal failed');
    vi.mocked(graphApi.traverseGraph).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useTraverseGraph(), { wrapper });

    result.current.mutate({ start_node_id: 'invalid' });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('calls onSuccess callback', async () => {
    vi.mocked(graphApi.traverseGraph).mockResolvedValueOnce(mockTraverseResponse);
    const { wrapper } = createWrapper();
    const onSuccess = vi.fn();

    const { result } = renderHook(() => useTraverseGraph({ onSuccess }), { wrapper });

    result.current.mutate({ start_node_id: 'kpi_trx' });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(onSuccess).toHaveBeenCalled();
  });
});

describe('useCausalChains', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('queries causal chains successfully', async () => {
    vi.mocked(graphApi.queryCausalChains).mockResolvedValueOnce(mockCausalChainResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCausalChains(), { wrapper });

    const request = {
      kpi_name: 'TRx',
      min_confidence: 0.6,
      max_chain_length: 3,
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockCausalChainResponse);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(graphApi.queryCausalChains).toHaveBeenCalledWith(request, expect.anything());
  });
});

describe('useCypherQuery', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('executes cypher query successfully', async () => {
    vi.mocked(graphApi.executeCypherQuery).mockResolvedValueOnce(mockCypherResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCypherQuery(), { wrapper });

    const request = {
      query: 'MATCH (h:HCP) RETURN h LIMIT 10',
      read_only: true,
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockCypherResponse);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(graphApi.executeCypherQuery).toHaveBeenCalledWith(request, expect.anything());
  });

  it('handles query error', async () => {
    const error = { status: 400, message: 'Invalid query syntax' };
    vi.mocked(graphApi.executeCypherQuery).mockRejectedValueOnce(error);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useCypherQuery(), { wrapper });

    result.current.mutate({ query: 'INVALID QUERY' });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useAddEpisode', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('adds episode successfully', async () => {
    vi.mocked(graphApi.addEpisode).mockResolvedValueOnce(mockEpisodeResponse);
    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useAddEpisode(), { wrapper });

    const request = {
      content: 'Dr. Smith prescribed Kisqali for the patient.',
      source: 'orchestrator',
      session_id: 'sess_abc123',
    };

    result.current.mutate(request);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockEpisodeResponse);
    // TanStack Query passes (variables, mutationContext) to mutationFn
    expect(graphApi.addEpisode).toHaveBeenCalledWith(request, expect.anything());

    // Should invalidate related queries
    expect(invalidateSpy).toHaveBeenCalled();
  });

  it('calls user onSuccess callback', async () => {
    vi.mocked(graphApi.addEpisode).mockResolvedValueOnce(mockEpisodeResponse);
    const { wrapper } = createWrapper();
    const onSuccess = vi.fn();

    const { result } = renderHook(() => useAddEpisode({ onSuccess }), { wrapper });

    result.current.mutate({
      content: 'Test episode',
      source: 'test',
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(onSuccess).toHaveBeenCalled();
  });
});

// =============================================================================
// PREFETCH HELPER TESTS
// =============================================================================

describe('prefetchNodes', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches nodes', async () => {
    vi.mocked(graphApi.listNodes).mockResolvedValueOnce(mockNodesResponse);
    const queryClient = createTestQueryClient();

    await prefetchNodes(queryClient);

    expect(graphApi.listNodes).toHaveBeenCalledWith(undefined);
  });

  it('prefetches nodes with params', async () => {
    vi.mocked(graphApi.listNodes).mockResolvedValueOnce(mockNodesResponse);
    const queryClient = createTestQueryClient();
    const params = { entity_types: 'HCP', limit: 50 };

    await prefetchNodes(queryClient, params);

    expect(graphApi.listNodes).toHaveBeenCalledWith(params);
  });
});

describe('prefetchNode', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches single node', async () => {
    vi.mocked(graphApi.getNode).mockResolvedValueOnce(mockNode);
    const queryClient = createTestQueryClient();

    await prefetchNode(queryClient, 'node1');

    expect(graphApi.getNode).toHaveBeenCalledWith('node1');
  });
});

describe('prefetchGraphStats', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('prefetches graph stats', async () => {
    vi.mocked(graphApi.getGraphStats).mockResolvedValueOnce(mockStatsResponse);
    const queryClient = createTestQueryClient();

    await prefetchGraphStats(queryClient);

    expect(graphApi.getGraphStats).toHaveBeenCalled();
  });
});
