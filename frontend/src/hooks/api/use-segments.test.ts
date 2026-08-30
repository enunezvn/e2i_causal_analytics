/**
 * useRunSegmentAnalysisAndWait — no client-side retry
 * ===================================================
 *
 * The mutationFn is "POST /segments/analyze, then poll the durable record".
 * The app's QueryClient retries mutations once by default
 * (src/lib/query-client.ts, `mutations.retry = 1`), so when the poll ceiling
 * expired on a still-running analysis react-query silently re-ran the whole
 * mutation: a SECOND heavy analysis was submitted while the first still held
 * the worker's single heavy-compute slot, and the OOM guard rejected it with
 * "compute capacity saturated; retry later" — the error the user saw, while
 * the first run completed fine 30–90 s later, unseen (live 2026-08-30, three
 * re-POSTs each exactly ~122 s after the original). A timed-out poll is not a
 * transport failure; re-running the mutation is never correct once the POST
 * has landed. Re-running is an explicit user action.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';

vi.mock('@/api/segments', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/api/segments')>()),
  runSegmentAnalysisAndWait: vi.fn(),
}));

import { useRunSegmentAnalysisAndWait } from './use-segments';
import { runSegmentAnalysisAndWait } from '@/api/segments';

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

describe('useRunSegmentAnalysisAndWait — a timed-out poll must not re-submit the analysis', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('submits exactly once when the poll ceiling expires (no react-query mutation retry)', async () => {
    (runSegmentAnalysisAndWait as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('Segment analysis timed out after 120000ms')
    );

    const { result } = renderHook(() => useRunSegmentAnalysisAndWait(), {
      wrapper: createAppLikeWrapper(),
    });

    act(() => {
      result.current.mutate({
        request: { query: 'HTE of copay_support on persistent_180d', brand: 'Fabhalta' },
        maxWaitMs: 1,
      });
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    // One POST+poll. A second call here is a duplicate heavy analysis.
    expect(runSegmentAnalysisAndWait).toHaveBeenCalledTimes(1);
  });
});
