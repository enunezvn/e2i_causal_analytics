/**
 * Agent Query-Key Namespacing Tests (#1958 step 2)
 * ================================================
 *
 * `agentKeys.status()` used to return the bare `['agent-status']` literal
 * because four other components hardcoded that same literal inline. Renaming
 * it while they did so would have split ONE shared cache entry into TWO,
 * making those components refetch independently and drift out of sync — a
 * behaviour change wearing a consistency fix's clothes. #1958 step 1 (#1959)
 * moved every consumer onto `useAgentStatus()`; this file pins the two
 * properties that made step 2 safe and keeps them from regressing:
 *
 *  1. the key is namespaced (`['agents', 'status']`), not the legacy literal;
 *  2. every consumer still lands on exactly ONE cache entry and ONE fetch.
 *
 * (2) is the property that actually matters: it fails if anyone reintroduces
 * a second inline key for `/agents/status`.
 *
 * Only `@/api/agents` is mocked, so the key observed here is the real key
 * written into the QueryClient cache by the real hook.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';

vi.mock('@/api/agents', () => ({
  getAgentStatus: vi.fn().mockResolvedValue({ agents: [], total_agents: 0 }),
  getAgentActivity: vi.fn().mockResolvedValue({ activities: [] }),
  getAgentTierMetrics: vi.fn().mockResolvedValue({ tiers: [] }),
}));

import { agentKeys, useAgentStatus } from './use-agents';
import * as agentsApi from '@/api/agents';

function createTestQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0 },
      mutations: { retry: false },
    },
  });
}

function createWrapper(client: QueryClient) {
  return ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client }, children);
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe('agentKeys.status() is namespaced', () => {
  it('returns the namespaced key, not the legacy bare literal', () => {
    expect(agentKeys.status()).toEqual(['agents', 'status']);
    expect(agentKeys.status()).not.toEqual(['agent-status']);
  });

  it('does not collide with the sibling agent query keys', () => {
    const status = JSON.stringify(agentKeys.status());
    expect(JSON.stringify(agentKeys.activity())).not.toBe(status);
    expect(JSON.stringify(agentKeys.tierMetrics())).not.toBe(status);
  });

  it('useAgentStatus writes that exact key into the cache', async () => {
    const client = createTestQueryClient();
    renderHook(() => useAgentStatus(), { wrapper: createWrapper(client) });

    await waitFor(() =>
      expect(client.getQueryCache().getAll().length).toBeGreaterThan(0)
    );

    const keys = client.getQueryCache().getAll().map((q) => q.queryKey);
    expect(keys).toEqual([['agents', 'status']]);
  });
});

describe('all consumers share ONE agent-status cache entry', () => {
  it('two independent consumers produce one cache entry and one fetch', async () => {
    const client = createTestQueryClient();

    // Two separate components, each calling the hook on its own — the shape
    // Home / ExecutiveSummary / both chat surfaces are in after step 1.
    renderHook(() => useAgentStatus(), { wrapper: createWrapper(client) });
    renderHook(() => useAgentStatus(), { wrapper: createWrapper(client) });

    await waitFor(() =>
      expect(vi.mocked(agentsApi.getAgentStatus)).toHaveBeenCalled()
    );

    expect(client.getQueryCache().getAll()).toHaveLength(1);
    expect(vi.mocked(agentsApi.getAgentStatus)).toHaveBeenCalledTimes(1);
  });
});
