/**
 * useRunOptimizationAndWait — no client-side retry
 * ================================================
 *
 * Sibling of the `useRunSegmentAnalysisAndWait` fix (#1836): the mutationFn is
 * "POST /resources/optimize, then poll the durable record". The app's
 * QueryClient retries mutations once by default (src/lib/query-client.ts,
 * `mutations.retry = 1`), so a poll-ceiling timeout on a still-running
 * optimization would make react-query silently re-run the whole mutation — a
 * SECOND heavy optimization submitted while the first still holds the worker's
 * single heavy-compute slot (#1839). A timed-out poll is not a transport
 * failure; re-running is an explicit user action.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';

vi.mock('@/api/resources', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/api/resources')>()),
  runOptimizationAndWait: vi.fn(),
}));

import { useRunOptimizationAndWait } from './use-resources';
import { runOptimizationAndWait } from '@/api/resources';
import { ResourceType } from '@/types/resources';

/**
 * Mirror the PRODUCTION mutation default (retry once). With `retry: false`
 * here the test would be vacuous — it must fail on the unfixed hook.
 */
function createAppLikeWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0 },
      mutations: { retry: 1, retryDelay: 0 },
    },
  });
  return ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
}

describe('useRunOptimizationAndWait — a timed-out poll must not re-submit the optimization', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('submits exactly once when the poll ceiling expires (no react-query mutation retry)', async () => {
    (runOptimizationAndWait as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('Optimization timed out after 120000ms')
    );

    const { result } = renderHook(() => useRunOptimizationAndWait(), {
      wrapper: createAppLikeWrapper(),
    });

    act(() => {
      result.current.mutate({
        request: {
          query: 'Optimize rep_time allocation to maximize roi',
          resource_type: ResourceType.REP_TIME,
          allocation_targets: [],
        },
        maxWaitMs: 1,
      });
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    // One POST+poll. A second call here is a duplicate heavy optimization.
    expect(runOptimizationAndWait).toHaveBeenCalledTimes(1);
  });
});
