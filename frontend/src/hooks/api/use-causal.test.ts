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
}));

import { useDiscoverEffects } from './use-causal';
import { discoverCausalEffects, getDiscoverCausalEffects } from '@/api/causal';

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
});
