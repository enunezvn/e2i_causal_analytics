/**
 * useRunLearningCycleAndWait — no client-side retry
 * =================================================
 *
 * Sibling of the `useRunSegmentAnalysisAndWait` fix (#1836): the mutationFn is
 * "POST /feedback/learn, then poll the durable batch". The app's QueryClient
 * retries mutations once by default (src/lib/query-client.ts,
 * `mutations.retry = 1`), so a poll-ceiling timeout on a still-running cycle
 * would make react-query silently re-run the whole mutation — a SECOND learning
 * cycle queued behind the first (#1839). A timed-out poll is not a transport
 * failure; re-running is an explicit user action.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';

vi.mock('@/api/feedback', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/api/feedback')>()),
  runLearningCycleAndWait: vi.fn(),
}));

import { useRunLearningCycleAndWait } from './use-feedback';
import { runLearningCycleAndWait } from '@/api/feedback';

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

describe('useRunLearningCycleAndWait — a timed-out poll must not re-submit the cycle', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('submits exactly once when the poll ceiling expires (no react-query mutation retry)', async () => {
    (runLearningCycleAndWait as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('Learning cycle timed out after 120000ms')
    );

    const { result } = renderHook(() => useRunLearningCycleAndWait(), {
      wrapper: createAppLikeWrapper(),
    });

    act(() => {
      result.current.mutate({
        request: { focus_agents: ['gap_analyzer'] },
        maxWaitMs: 1,
      });
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    // One POST+poll. A second call here is a duplicate learning cycle.
    expect(runLearningCycleAndWait).toHaveBeenCalledTimes(1);
  });
});
