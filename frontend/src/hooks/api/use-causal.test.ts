/**
 * useDiscoverEffects — scope-reset tests
 * ======================================
 *
 * A discovered leaderboard is scoped to exactly (dataset, brand). When the user
 * changes grain or brand, the previous job's effects no longer describe the new
 * scope, so the hook must drop them (both the polled job AND the retained
 * mutation result) — otherwise e.g. a Patient-grain leaderboard would leak under
 * the HCP grain as plausible-but-wrong results. (codex Finding 1, HIGH.)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';

vi.mock('@/api/causal', () => ({
  discoverCausalEffects: vi.fn(),
  getDiscoverCausalEffects: vi.fn(),
  getDiscoverQuestions: vi.fn(),
  cancelDiscoverCausalEffects: vi.fn(),
  runHierarchicalAnalysisAndWait: vi.fn(),
  runCausalAgentAnalysisAndWait: vi.fn(),
}));

import {
  useDiscoverEffects,
  useRunHierarchicalAnalysisAndWait,
  useRunCausalAgentAnalysis,
} from './use-causal';
import {
  discoverCausalEffects,
  getDiscoverCausalEffects,
  cancelDiscoverCausalEffects,
  runHierarchicalAnalysisAndWait,
  runCausalAgentAnalysisAndWait,
} from '@/api/causal';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0 },
      mutations: { retry: false },
    },
  });
  return ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
}

const PATIENT_JOB = {
  job_id: 'j1',
  status: 'completed',
  dataset: 'patient_journeys',
  brand: null,
  total: 1,
  completed: 1,
  effects: [
    {
      treatment: 'treatment_arm',
      outcome: 'persistent_180d',
      status: 'completed',
      analysis_id: 'a1',
    },
  ],
  note: '',
};

describe('useDiscoverEffects — scope reset', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (discoverCausalEffects as ReturnType<typeof vi.fn>).mockResolvedValue(PATIENT_JOB);
    (getDiscoverCausalEffects as ReturnType<typeof vi.fn>).mockResolvedValue(PATIENT_JOB);
  });

  it('drops the previous job when the dataset (grain) changes', async () => {
    const { result, rerender } = renderHook(
      ({ dataset, brand }: { dataset: string; brand: string | null }) =>
        useDiscoverEffects(dataset, brand),
      {
        wrapper: createWrapper(),
        initialProps: { dataset: 'patient_journeys', brand: null as string | null },
      }
    );

    act(() => {
      result.current.start();
    });
    await waitFor(() => expect(result.current.job).not.toBeNull());
    expect(result.current.job?.dataset).toBe('patient_journeys');

    // Switch grain → no stale cross-dataset leaderboard.
    rerender({ dataset: 'hcp_adoption', brand: null });
    await waitFor(() => expect(result.current.job).toBeNull());
  });

  it('drops the previous job when the brand changes', async () => {
    const { result, rerender } = renderHook(
      ({ dataset, brand }: { dataset: string; brand: string | null }) =>
        useDiscoverEffects(dataset, brand),
      {
        wrapper: createWrapper(),
        initialProps: { dataset: 'patient_journeys', brand: null as string | null },
      }
    );

    act(() => {
      result.current.start();
    });
    await waitFor(() => expect(result.current.job).not.toBeNull());

    rerender({ dataset: 'patient_journeys', brand: 'Kisqali' });
    await waitFor(() => expect(result.current.job).toBeNull());
  });

  it('does not adopt a job whose scope changed before its submit resolved (race)', async () => {
    // The submit is in-flight when the user switches grain; reset() does not
    // suppress its onSuccess, so the job is TAGGED with the scope it was started
    // for and must not surface under the new scope when it resolves late.
    let resolveSubmit!: (v: typeof PATIENT_JOB) => void;
    (discoverCausalEffects as ReturnType<typeof vi.fn>).mockImplementation(
      () => new Promise((res) => { resolveSubmit = res as (v: typeof PATIENT_JOB) => void; })
    );
    const { result, rerender } = renderHook(
      ({ dataset, brand }: { dataset: string; brand: string | null }) =>
        useDiscoverEffects(dataset, brand),
      {
        wrapper: createWrapper(),
        initialProps: { dataset: 'patient_journeys', brand: null as string | null },
      }
    );

    act(() => {
      result.current.start(); // Patient submit, in-flight
    });
    // Wait until the mutationFn has actually started (resolveSubmit captured).
    await waitFor(() => expect(discoverCausalEffects).toHaveBeenCalled());
    rerender({ dataset: 'hcp_adoption', brand: null }); // switch scope mid-flight
    await act(async () => {
      resolveSubmit(PATIENT_JOB); // Patient submit resolves AFTER the switch
      await Promise.resolve();
    });

    // The Patient job must NOT surface under the HCP grain.
    expect(result.current.job).toBeNull();
  });
});

/**
 * useRunHierarchicalAnalysisAndWait — no client-side retry
 * ========================================================
 *
 * Sibling of the `useRunSegmentAnalysisAndWait` fix (#1836): the mutationFn is
 * "POST /causal/hierarchical, then poll the record". The app's QueryClient
 * retried mutations once by default until #1846 (src/lib/query-client.ts,
 * `mutations.retry = 1`; now `0`), so a poll-ceiling timeout on a still-running
 * analysis would make react-query silently re-run the whole mutation — a
 * SECOND heavy CATE analysis submitted while the first still holds the
 * worker's single heavy-compute slot (#1839). The `retry: false` on the
 * clinical-context / treatment-effects QUERIES in this module does not cover
 * this MUTATION. A timed-out poll is not a transport failure; re-running is an
 * explicit user action.
 */

/**
 * Mirror the pre-#1846 PRODUCTION mutation default (retry once). The app
 * default is `retry: 0` since #1846, but the hook's own `retry: false` must
 * hold under ANY client default. `createWrapper` above uses
 * `mutations: { retry: false }`, which would make this test vacuous — it
 * must fail on the unfixed hook.
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

describe('useRunHierarchicalAnalysisAndWait — a timed-out poll must not re-submit the analysis', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('submits exactly once when the poll ceiling expires (no react-query mutation retry)', async () => {
    (runHierarchicalAnalysisAndWait as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('Analysis timed out after 180000ms')
    );

    const { result } = renderHook(() => useRunHierarchicalAnalysisAndWait(), {
      wrapper: createAppLikeWrapper(),
    });

    act(() => {
      result.current.mutate({
        request: {
          treatment_var: 'copay_support',
          outcome_var: 'persistent_180d',
        },
        maxWaitMs: 1,
      });
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    // One POST+poll. A second call here is a duplicate heavy analysis.
    expect(runHierarchicalAnalysisAndWait).toHaveBeenCalledTimes(1);
  });
});

/**
 * useRunCausalAgentAnalysis is an *AndWait hook in everything but name: its
 * mutationFn is `runCausalAgentAnalysisAndWait` (POST /causal/agent-analyze,
 * then poll for up to 900 s — "the agent run takes minutes"), and it is the
 * hook the Causal Analysis page runs (CausalAnalysis.tsx). Same re-submit
 * defect as the named siblings (#1839).
 */
describe('useRunCausalAgentAnalysis — a timed-out poll must not re-submit the agent run', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('submits exactly once when the poll ceiling expires (no react-query mutation retry)', async () => {
    (runCausalAgentAnalysisAndWait as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('Causal agent analysis timed out after 900000ms')
    );

    const { result } = renderHook(() => useRunCausalAgentAnalysis(), {
      wrapper: createAppLikeWrapper(),
    });

    act(() => {
      result.current.mutate({
        treatment_var: 'copay_support',
        outcome_var: 'persistent_180d',
        brand: 'Fabhalta',
      });
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(runCausalAgentAnalysisAndWait).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// Question subset + cooperative cancel (CAUSAL-DISC-UX)
// ---------------------------------------------------------------------------

const RUNNING_JOB = {
  ...PATIENT_JOB,
  status: 'running',
  total: 2,
  completed: 0,
  cancel_requested: false,
};

describe('useDiscoverEffects — question subset + cancel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (discoverCausalEffects as ReturnType<typeof vi.fn>).mockResolvedValue(RUNNING_JOB);
    (getDiscoverCausalEffects as ReturnType<typeof vi.fn>).mockResolvedValue(RUNNING_JOB);
    (cancelDiscoverCausalEffects as ReturnType<typeof vi.fn>).mockResolvedValue({
      ...RUNNING_JOB,
      cancel_requested: true,
    });
  });

  it('passes the selected questions to the submit, and nothing when every candidate runs', async () => {
    const subset = [{ treatment: 'treatment_arm', outcome: 'persistent_180d', brand: 'Kisqali' }];
    const { result } = renderHook(() => useDiscoverEffects('patient_journeys', 'Kisqali'), {
      wrapper: createWrapper(),
    });
    act(() => {
      result.current.start(subset);
    });
    await waitFor(() =>
      expect(discoverCausalEffects).toHaveBeenCalledWith('patient_journeys', 'Kisqali', subset)
    );
    act(() => {
      result.current.start();
    });
    await waitFor(() =>
      expect(discoverCausalEffects).toHaveBeenLastCalledWith('patient_journeys', 'Kisqali', undefined)
    );
  });

  it('cancel() targets the active job and holds cancelRequested until that job ends', async () => {
    const { result } = renderHook(() => useDiscoverEffects('patient_journeys', null), {
      wrapper: createWrapper(),
    });
    expect(result.current.cancelRequested).toBe(false);
    act(() => {
      result.current.cancel(); // no active job → no request
    });
    expect(cancelDiscoverCausalEffects).not.toHaveBeenCalled();

    act(() => {
      result.current.start();
    });
    await waitFor(() => expect(result.current.job?.status).toBe('running'));
    act(() => {
      result.current.cancel();
    });
    await waitFor(() => expect(cancelDiscoverCausalEffects).toHaveBeenCalledWith('j1'));
    await waitFor(() => expect(result.current.cancelRequested).toBe(true));
    // A poll that has not yet observed the marker (the task's own publish can
    // briefly overwrite the row flag) must NOT flip the button back.
    expect(result.current.job?.cancel_requested).toBe(false);
    expect(result.current.cancelRequested).toBe(true);

    // The job ends `cancelled` → the hook reports it as the terminal job.
    (getDiscoverCausalEffects as ReturnType<typeof vi.fn>).mockResolvedValue({
      ...RUNNING_JOB,
      status: 'cancelled',
      completed: 1,
      cancel_requested: true,
    });
    await waitFor(() => expect(result.current.job?.status).toBe('cancelled'), { timeout: 5000 });
  }, 10000);

  it('a fresh run in the same scope does not inherit the previous cancel', async () => {
    const { result } = renderHook(() => useDiscoverEffects('patient_journeys', null), {
      wrapper: createWrapper(),
    });
    act(() => {
      result.current.start();
    });
    await waitFor(() => expect(result.current.job?.status).toBe('running'));
    act(() => {
      result.current.cancel();
    });
    await waitFor(() => expect(result.current.cancelRequested).toBe(true));

    (discoverCausalEffects as ReturnType<typeof vi.fn>).mockResolvedValue({
      ...RUNNING_JOB,
      job_id: 'j2',
    });
    (getDiscoverCausalEffects as ReturnType<typeof vi.fn>).mockResolvedValue({
      ...RUNNING_JOB,
      job_id: 'j2',
    });
    act(() => {
      result.current.start();
    });
    await waitFor(() => expect(result.current.job?.job_id).toBe('j2'));
    expect(result.current.cancelRequested).toBe(false);
  });

  it('drops cancel state with the job on a scope switch', async () => {
    const { result, rerender } = renderHook(
      ({ dataset, brand }: { dataset: string; brand: string | null }) =>
        useDiscoverEffects(dataset, brand),
      {
        wrapper: createWrapper(),
        initialProps: { dataset: 'patient_journeys', brand: null as string | null },
      }
    );
    act(() => {
      result.current.start();
    });
    await waitFor(() => expect(result.current.job?.status).toBe('running'));
    act(() => {
      result.current.cancel();
    });
    await waitFor(() => expect(result.current.cancelRequested).toBe(true));
    rerender({ dataset: 'hcp_adoption', brand: null });
    await waitFor(() => expect(result.current.job).toBeNull());
    expect(result.current.cancelRequested).toBe(false);
  });
});
