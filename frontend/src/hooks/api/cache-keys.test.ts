/**
 * React Query Cache-Key Correctness Tests
 * =======================================
 *
 * Dedicated regression tests for the "disputed-findings" sweep covering
 * result-affecting parameters that must be folded into TanStack Query cache
 * keys (and into the matching prefetch / invalidation keys).
 *
 * IMPORTANT: This file intentionally does NOT mock `@/lib/query-client`.
 * It exercises the REAL `queryKeys` factories and the REAL hooks so that the
 * key actually written into the QueryClient cache is observed. Only the
 * `@/api/*` modules are mocked so no network traffic occurs.
 *
 * Each test asserts the bug it guards against: two calls with different
 * result-affecting params MUST produce DIFFERENT cache keys, and a prefetch
 * key MUST EQUAL the hook's read key.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { QueryKey } from '@tanstack/react-query';
import * as React from 'react';

// ---------------------------------------------------------------------------
// API module mocks (network isolation only)
// ---------------------------------------------------------------------------

vi.mock('@/api/digital-twin', () => ({
  runSimulation: vi.fn(),
  compareScenarios: vi.fn(),
  getSimulation: vi.fn(),
  getSimulationHistory: vi.fn().mockResolvedValue({ simulations: [], total: 0 }),
  getDigitalTwinHealth: vi.fn(),
}));

vi.mock('@/api/graph', () => ({
  searchGraph: vi.fn().mockResolvedValue({ results: [], total: 0 }),
  getNodes: vi.fn(),
  getNode: vi.fn(),
  getNodeNetwork: vi.fn(),
  getRelationships: vi.fn(),
  getGraphStats: vi.fn(),
  traverseGraph: vi.fn(),
  queryCausalChains: vi.fn(),
}));

vi.mock('@/api/monitoring', () => ({
  getLatestDriftStatus: vi.fn().mockResolvedValue({ drift_results: [] }),
  getDriftHistory: vi.fn(),
  getDriftStatus: vi.fn(),
  triggerDriftDetection: vi.fn(),
}));

vi.mock('@/api/kpi', () => ({
  listKPIs: vi.fn().mockResolvedValue({ kpis: [], total: 0 }),
}));

vi.mock('@/api/experiments', () => ({
  getSegmentResults: vi.fn().mockResolvedValue({ segments: [] }),
  triggerMonitoring: vi.fn().mockResolvedValue({ monitored: [] }),
}));

import { queryKeys } from '@/lib/query-client';
import { useSimulationHistory, prefetchSimulationHistory, useRunSimulation } from './use-digital-twin';
import { useGraphSearch } from './use-graph';
import { useLatestDriftStatus, prefetchLatestDriftStatus } from './use-monitoring';
import { prefetchKPIList, useKPIList } from './use-kpi';
import { useSegmentResults, useTriggerMonitoring } from './use-experiments';

import * as digitalTwinApi from '@/api/digital-twin';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

/** Collect all queryKeys currently in a client's cache. */
function cacheKeys(client: QueryClient): QueryKey[] {
  return client.getQueryCache().getAll().map((q) => q.queryKey);
}

/** Render a query hook, wait until its query is registered in cache, return that key. */
async function keyForQueryHook(
  hookFn: () => unknown,
  client: QueryClient
): Promise<QueryKey> {
  renderHook(hookFn, { wrapper: createWrapper(client) });
  await waitFor(() => expect(client.getQueryCache().getAll().length).toBeGreaterThan(0));
  const keys = cacheKeys(client);
  // Each render uses a fresh client, so exactly one query is expected.
  expect(keys).toHaveLength(1);
  return keys[0];
}

beforeEach(() => {
  vi.clearAllMocks();
});

// ===========================================================================
// FINDING 1: useSimulationHistory key omits limit/offset
// ===========================================================================

describe('Finding 1: digitalTwin history cache key includes limit/offset', () => {
  it('factory varies with limit and offset', () => {
    const a = queryKeys.digitalTwin.history({ limit: 10, offset: 0 });
    const b = queryKeys.digitalTwin.history({ limit: 50, offset: 0 });
    const c = queryKeys.digitalTwin.history({ limit: 10, offset: 50 });
    expect(a).not.toEqual(b);
    expect(a).not.toEqual(c);
    expect(b).not.toEqual(c);
  });

  it('useSimulationHistory produces DIFFERENT cache keys for different limit', async () => {
    const k1 = await keyForQueryHook(
      () => useSimulationHistory({ limit: 10, offset: 0 }),
      createTestQueryClient()
    );
    const k2 = await keyForQueryHook(
      () => useSimulationHistory({ limit: 50, offset: 0 }),
      createTestQueryClient()
    );
    expect(k1).not.toEqual(k2);
  });

  it('prefetchSimulationHistory key EQUALS the hook read key for the same params', async () => {
    const params = { limit: 25, offset: 75 };
    const readKey = await keyForQueryHook(
      () => useSimulationHistory(params),
      createTestQueryClient()
    );
    const prefetchClient = createTestQueryClient();
    await prefetchSimulationHistory(prefetchClient, params);
    expect(cacheKeys(prefetchClient)).toContainEqual(readKey);
  });

  it('run-simulation invalidation matches the parameterised history keys', async () => {
    const client = createTestQueryClient();
    // Seed two history queries with different params.
    await keyForQueryHook(() => useSimulationHistory({ limit: 10 }), client).catch(() => undefined);
    // (single client this time so both register)
    renderHook(() => useSimulationHistory({ limit: 10 }), { wrapper: createWrapper(client) });
    renderHook(() => useSimulationHistory({ limit: 50 }), { wrapper: createWrapper(client) });
    await waitFor(() => expect(cacheKeys(client).length).toBeGreaterThanOrEqual(2));

    const invalidateSpy = vi.spyOn(client, 'invalidateQueries');
    vi.mocked(digitalTwinApi.runSimulation).mockResolvedValueOnce({
      simulation_id: 'sim-1',
    } as never);

    const { result } = renderHook(() => useRunSimulation(), { wrapper: createWrapper(client) });
    result.current.mutate({} as never);
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    // The invalidation must use a partial (prefix) match on history so it
    // matches keys that carry limit/offset suffixes.
    expect(invalidateSpy).toHaveBeenCalled();
    const historyInvalidation = invalidateSpy.mock.calls.find((call) => {
      const key = (call[0] as { queryKey?: QueryKey } | undefined)?.queryKey;
      return Array.isArray(key) && key.includes('history');
    });
    expect(historyInvalidation).toBeDefined();
    const invKey = (historyInvalidation![0] as { queryKey: QueryKey }).queryKey as unknown[];
    // Prefix-only key (no trailing limit/offset) so it matches all variants.
    expect(invKey[invKey.length - 1]).toBe('history');
  });
});

// ===========================================================================
// FINDING 2: useGraphSearch key omits entity_types/k/min_score
// ===========================================================================

describe('Finding 2: graph search cache key includes entity_types/k/min_score', () => {
  it('keys differ for different k', async () => {
    const k1 = await keyForQueryHook(
      () => useGraphSearch('trx drivers', { k: 5 }),
      createTestQueryClient()
    );
    const k2 = await keyForQueryHook(
      () => useGraphSearch('trx drivers', { k: 25 }),
      createTestQueryClient()
    );
    expect(k1).not.toEqual(k2);
  });

  it('keys differ for different entity_types', async () => {
    const k1 = await keyForQueryHook(
      () => useGraphSearch('trx drivers', { entity_types: ['HCP'] as never }),
      createTestQueryClient()
    );
    const k2 = await keyForQueryHook(
      () => useGraphSearch('trx drivers', { entity_types: ['Brand'] as never }),
      createTestQueryClient()
    );
    expect(k1).not.toEqual(k2);
  });

  it('keys differ for different min_score', async () => {
    const k1 = await keyForQueryHook(
      () => useGraphSearch('trx drivers', { min_score: 0.1 }),
      createTestQueryClient()
    );
    const k2 = await keyForQueryHook(
      () => useGraphSearch('trx drivers', { min_score: 0.9 }),
      createTestQueryClient()
    );
    expect(k1).not.toEqual(k2);
  });

  it('keys do NOT vary with session_id (not result-affecting)', async () => {
    const k1 = await keyForQueryHook(
      () => useGraphSearch('trx drivers', { k: 10, session_id: 'sess-a' }),
      createTestQueryClient()
    );
    const k2 = await keyForQueryHook(
      () => useGraphSearch('trx drivers', { k: 10, session_id: 'sess-b' }),
      createTestQueryClient()
    );
    expect(k1).toEqual(k2);
  });
});

// ===========================================================================
// FINDING 3: useLatestDriftStatus key omits limit
// ===========================================================================

describe('Finding 3: latest drift status cache key includes limit', () => {
  it('useLatestDriftStatus produces different keys for different limit', async () => {
    const k1 = await keyForQueryHook(
      () => useLatestDriftStatus('model-1', 5),
      createTestQueryClient()
    );
    const k2 = await keyForQueryHook(
      () => useLatestDriftStatus('model-1', 25),
      createTestQueryClient()
    );
    expect(k1).not.toEqual(k2);
  });

  it('prefetchLatestDriftStatus key EQUALS the hook read key for same model+limit', async () => {
    const readKey = await keyForQueryHook(
      () => useLatestDriftStatus('model-1', 15),
      createTestQueryClient()
    );
    const prefetchClient = createTestQueryClient();
    await prefetchLatestDriftStatus(prefetchClient, 'model-1', 15);
    expect(cacheKeys(prefetchClient)).toContainEqual(readKey);
  });
});

// ===========================================================================
// FINDING 4: prefetchKPIList key != useKPIList read key
// ===========================================================================

describe('Finding 4: prefetchKPIList key equals useKPIList read key', () => {
  it('prefetch and read keys match for the same params', async () => {
    const params = { workstream: 'ws1_data_quality', causal_library: 'dowhy' } as never;
    const readKey = await keyForQueryHook(
      () => useKPIList(params),
      createTestQueryClient()
    );
    // prefetchKPIList writes into the module-global query client; assert the
    // key it WOULD write equals the read key by reading the global cache.
    const { queryClient: globalClient } = await import('@/lib/query-client');
    globalClient.clear();
    await prefetchKPIList(params);
    expect(cacheKeys(globalClient)).toContainEqual(readKey);
  });

  it('prefetch keys differ between distinct params (no cache collision)', async () => {
    const { queryClient: globalClient } = await import('@/lib/query-client');
    globalClient.clear();
    await prefetchKPIList({ workstream: 'ws1_data_quality' } as never);
    await prefetchKPIList({ workstream: 'ws2_triggers' } as never);
    const keys = cacheKeys(globalClient);
    expect(keys.length).toBeGreaterThanOrEqual(2);
  });
});

// ===========================================================================
// FINDING 5: useSegmentResults key order-sensitive on segments
// ===========================================================================

describe('Finding 5: segment results cache key is order-insensitive', () => {
  it('produces the SAME key regardless of segment array order', async () => {
    const k1 = await keyForQueryHook(
      () => useSegmentResults('exp-1', ['region', 'specialty']),
      createTestQueryClient()
    );
    const k2 = await keyForQueryHook(
      () => useSegmentResults('exp-1', ['specialty', 'region']),
      createTestQueryClient()
    );
    expect(k1).toEqual(k2);
  });

  it('still distinguishes genuinely different segment sets', async () => {
    const k1 = await keyForQueryHook(
      () => useSegmentResults('exp-1', ['region']),
      createTestQueryClient()
    );
    const k2 = await keyForQueryHook(
      () => useSegmentResults('exp-1', ['specialty']),
      createTestQueryClient()
    );
    expect(k1).not.toEqual(k2);
  });
});

// ===========================================================================
// FINDING 6: useTriggerMonitoring over-invalidates whole experiments namespace
// ===========================================================================

describe('Finding 6: trigger monitoring narrows invalidation to affected sub-keys', () => {
  it('invalidates monitoring-affected sub-keys per experiment id, not the whole namespace', async () => {
    const client = createTestQueryClient();
    const invalidateSpy = vi.spyOn(client, 'invalidateQueries');

    const { result } = renderHook(() => useTriggerMonitoring(), { wrapper: createWrapper(client) });
    result.current.mutate({ experiment_ids: ['exp-1', 'exp-2'] } as never);
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const invalidatedKeys = invalidateSpy.mock.calls
      .map((call) => (call[0] as { queryKey?: QueryKey } | undefined)?.queryKey)
      .filter((k): k is QueryKey => Array.isArray(k));

    // Must NOT invalidate the entire experiments namespace when ids are given.
    const namespaceKey = queryKeys.experiments.all();
    expect(invalidatedKeys).not.toContainEqual(namespaceKey as unknown as QueryKey);

    // The monitoring sweep refreshes health, alerts, SRM, enrollment, and
    // fidelity data server-side (check_srm/check_enrollment/check_fidelity),
    // so all of those sub-keys must be invalidated for each requested id.
    for (const id of ['exp-1', 'exp-2']) {
      expect(invalidatedKeys).toContainEqual(queryKeys.experiments.health(id) as unknown as QueryKey);
      expect(invalidatedKeys).toContainEqual(queryKeys.experiments.alerts(id) as unknown as QueryKey);
      expect(invalidatedKeys).toContainEqual(queryKeys.experiments.srmChecks(id) as unknown as QueryKey);
      expect(invalidatedKeys).toContainEqual(
        queryKeys.experiments.enrollmentStats(id) as unknown as QueryKey
      );
      expect(invalidatedKeys).toContainEqual(
        queryKeys.experiments.fidelityComparisons(id) as unknown as QueryKey
      );
    }
  });

  it('falls back to the experiments namespace when no ids are given', async () => {
    const client = createTestQueryClient();
    const invalidateSpy = vi.spyOn(client, 'invalidateQueries');

    const { result } = renderHook(() => useTriggerMonitoring(), { wrapper: createWrapper(client) });
    result.current.mutate({} as never);
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const invalidatedKeys = invalidateSpy.mock.calls
      .map((call) => (call[0] as { queryKey?: QueryKey } | undefined)?.queryKey)
      .filter((k): k is QueryKey => Array.isArray(k));

    expect(invalidatedKeys).toContainEqual(queryKeys.experiments.all() as unknown as QueryKey);
  });
});
