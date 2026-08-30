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
  quickLearningCycle: vi.fn(),
}));

import { useRunLearningCycleAndWait, useQuickLearningCycle } from './use-feedback';
import { runLearningCycleAndWait, quickLearningCycle } from '@/api/feedback';

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

/**
 * useQuickLearningCycle is the hook the Feedback Learning page actually runs
 * (FeedbackLearning.tsx `runQuickCycle`). It is NOT a poll: it is a single
 * synchronous `POST /feedback/learn?async_mode=false` that the backend
 * executes inline — new batch_id, patterns/updates persisted, then the
 * response. The UI-driven cycles measured 18.8 s and 23.1 s on prod
 * (`feedback_learning_batches`, 2026-07-29 / 2026-08-26) against the
 * api-client's 30 s axios timeout, and a failing cycle is a 500 after the
 * FAILED batch was persisted. Either way the request rejects AFTER the server
 * started a batch, so a react-query retry is a second full cycle with a
 * second batch and duplicated artifacts.
 */
describe('useQuickLearningCycle — a rejected synchronous cycle must not be re-POSTed', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('POSTs exactly once when the request times out client-side (no react-query mutation retry)', async () => {
    (quickLearningCycle as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('timeout of 30000ms exceeded')
    );

    const { result } = renderHook(() => useQuickLearningCycle(), {
      wrapper: createAppLikeWrapper(),
    });

    act(() => {
      result.current.mutate(undefined);
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    // The server already started (or finished) a batch for the first POST.
    expect(quickLearningCycle).toHaveBeenCalledTimes(1);
  });
});
