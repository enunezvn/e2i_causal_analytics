/**
 * useRunSegmentAnalysisAndWait — no client-side retry
 * ===================================================
 *
 * The mutationFn is "POST /segments/analyze, then poll the durable record".
 * The app's QueryClient retried mutations once by default until #1846
 * (src/lib/query-client.ts, `mutations.retry = 1`; now `0`), so when the poll ceiling
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
  waitForSegmentAnalysis: vi.fn(),
}));

import { useRunSegmentAnalysisAndWait } from './use-segments';
import {
  runSegmentAnalysisAndWait,
  waitForSegmentAnalysis,
  SegmentAnalysisTimeoutError,
} from '@/api/segments';
import type { SegmentAnalysisResponse } from '@/types/segments';

/**
 * Mirror the pre-#1846 PRODUCTION mutation default (retry once). The app
 * default is `retry: 0` since #1846, but the hook's own `retry: false` must
 * hold under ANY client default; with `retry: false` here the test would be
 * vacuous — it must fail on the unfixed hook.
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

/**
 * #1841 — a ceiling expiry is not the end of the run. The record is durable and
 * usually still computing, so the mutation must (a) surface the typed timeout
 * with its analysis_id and (b) accept a `resumeAnalysisId` variant that
 * re-attaches with GET polling only. The resumed completion lands in the same
 * `data` slot a normal completion does, so the page renders it identically.
 */
describe('useRunSegmentAnalysisAndWait — Keep waiting resumes the same analysis_id (#1841)', () => {
  const completed = { analysis_id: 'seg_1841', status: 'completed' } as SegmentAnalysisResponse;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('exposes the typed timeout (with analysis_id) as the mutation error', async () => {
    (runSegmentAnalysisAndWait as ReturnType<typeof vi.fn>).mockRejectedValue(
      new SegmentAnalysisTimeoutError('seg_1841', 300_000)
    );
    const { result } = renderHook(() => useRunSegmentAnalysisAndWait(), {
      wrapper: createAppLikeWrapper(),
    });

    act(() => {
      result.current.mutate({ request: { query: 'q', brand: 'Fabhalta' }, maxWaitMs: 300_000 });
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.error).toBeInstanceOf(SegmentAnalysisTimeoutError);
    expect((result.current.error as SegmentAnalysisTimeoutError).analysisId).toBe('seg_1841');
    expect(runSegmentAnalysisAndWait).toHaveBeenCalledTimes(1);
  });

  it('resumeAnalysisId polls the same id via waitForSegmentAnalysis and never re-POSTs', async () => {
    (runSegmentAnalysisAndWait as ReturnType<typeof vi.fn>).mockRejectedValue(
      new SegmentAnalysisTimeoutError('seg_1841', 300_000)
    );
    (waitForSegmentAnalysis as ReturnType<typeof vi.fn>).mockResolvedValue(completed);
    const { result } = renderHook(() => useRunSegmentAnalysisAndWait(), {
      wrapper: createAppLikeWrapper(),
    });

    act(() => {
      result.current.mutate({ request: { query: 'q', brand: 'Fabhalta' }, maxWaitMs: 300_000 });
    });
    await waitFor(() => expect(result.current.isError).toBe(true));

    act(() => {
      result.current.mutate({ resumeAnalysisId: 'seg_1841', pollIntervalMs: 2_000, maxWaitMs: 300_000 });
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual(completed);
    expect(result.current.error).toBeNull();
    expect(waitForSegmentAnalysis).toHaveBeenCalledTimes(1);
    expect(waitForSegmentAnalysis).toHaveBeenCalledWith('seg_1841', 2_000, 300_000);
    // Still exactly one POST+poll across the whole expire → resume cycle.
    expect(runSegmentAnalysisAndWait).toHaveBeenCalledTimes(1);
  });

  it('a failed record during resume surfaces as the ordinary failure error', async () => {
    (waitForSegmentAnalysis as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('Segment analysis failed: estimator blew up')
    );
    const { result } = renderHook(() => useRunSegmentAnalysisAndWait(), {
      wrapper: createAppLikeWrapper(),
    });

    act(() => {
      result.current.mutate({ resumeAnalysisId: 'seg_1841' });
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.error).not.toBeInstanceOf(SegmentAnalysisTimeoutError);
    expect(result.current.error?.message).toBe('Segment analysis failed: estimator blew up');
    // retry: false applies to the resume variant too (no duplicate re-attach).
    expect(waitForSegmentAnalysis).toHaveBeenCalledTimes(1);
    expect(runSegmentAnalysisAndWait).not.toHaveBeenCalled();
  });
});
