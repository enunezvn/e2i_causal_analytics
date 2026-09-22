/**
 * Proposed-experiments hooks (#2206, owner item C).
 *
 * useProposedExperiments reads the twin simulations that propose an experiment;
 * useCreateDraftExperiment creates a linked `draft` and invalidates the caches
 * whose contents it changed (the proposal leaves the list; history rows show the
 * link; the simulation detail carries it).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';
import type {
  DraftExperimentResponse,
  ProposedExperimentsResponse,
} from '@/types/digital-twin';
import { FidelityStatus } from '@/types/digital-twin';

vi.mock('@/api/digital-twin', () => ({
  listProposedExperiments: vi.fn(),
  createDraftExperiment: vi.fn(),
}));

import { useCreateDraftExperiment, useProposedExperiments } from './use-digital-twin';
import * as digitalTwinApi from '@/api/digital-twin';
import { queryKeys } from '@/lib/query-client';

function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } },
  });
}

function createWrapper(queryClient: QueryClient) {
  return ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
}

const response: ProposedExperimentsResponse = {
  proposals: [
    {
      simulation_id: 'sim-1',
      model_id: 'model-1',
      brand: 'Kisqali',
      intervention_type: 'email_campaign',
      intervention_config: {},
      simulated_ate: 0.1,
      simulated_ci_lower: 0.05,
      simulated_ci_upper: 0.15,
      recommendation: 'deploy',
      recommendation_rationale: 'ok',
      recommended_sample_size: 100,
      recommended_duration_weeks: 8,
      simulation_confidence: 0.7,
      data_provenance: 'cohort_estimated_synthetic_gold_v1',
      fidelity_status: FidelityStatus.UNVALIDATED,
      created_at: '2026-09-22T00:00:00Z',
      proposal_basis: 'twin_simulation',
      outcome_column: 'cohort_conversion_outcome',
      effect_scale: 'absolute',
    },
  ],
  outcome_column: 'cohort_conversion_outcome',
  outcome_measurable_in_real_mode: false,
  total_proposed: 1,
  total_linked: 0,
  real_experiments_running: 0,
};

const draft: DraftExperimentResponse = {
  experiment_id: 'exp-1',
  simulation_id: 'sim-1',
  experiment_name: 'twin_proposal_Kisqali_email_campaign_sim-1',
  status: 'draft',
  brand: 'Kisqali',
  intervention_channel: 'email_campaign',
  prediction_target: 'cohort_conversion_outcome',
  target_enrollment: 100,
  planned_duration_days: 56,
  created_by: 'ops@example.com',
  outcome_column: 'cohort_conversion_outcome',
  outcome_measurable_in_real_mode: false,
  linked: true,
  next_step: 'promote',
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe('useProposedExperiments', () => {
  it('fetches the proposals under a brand-aware key', async () => {
    vi.mocked(digitalTwinApi.listProposedExperiments).mockResolvedValue(response);
    const queryClient = createTestQueryClient();
    const { result } = renderHook(() => useProposedExperiments({ brand: 'Kisqali' }), {
      wrapper: createWrapper(queryClient),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual(response);
    expect(digitalTwinApi.listProposedExperiments).toHaveBeenCalledWith({ brand: 'Kisqali' });
    expect(
      queryClient.getQueryData(queryKeys.digitalTwin.proposedExperiments({ brand: 'Kisqali' }))
    ).toEqual(response);
    // A brand filter and the all-brands read never share a cache entry.
    expect(queryKeys.digitalTwin.proposedExperiments({ brand: 'Kisqali' })).not.toEqual(
      queryKeys.digitalTwin.proposedExperiments()
    );
  });

  it('surfaces a failed request as an error state', async () => {
    vi.mocked(digitalTwinApi.listProposedExperiments).mockRejectedValue(new Error('boom'));
    const queryClient = createTestQueryClient();
    const { result } = renderHook(() => useProposedExperiments(), {
      wrapper: createWrapper(queryClient),
    });
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.error?.message).toBe('boom');
  });
});

describe('useCreateDraftExperiment', () => {
  it('creates the draft for the simulation and invalidates proposals, history and the detail', async () => {
    vi.mocked(digitalTwinApi.createDraftExperiment).mockResolvedValue(draft);
    const queryClient = createTestQueryClient();
    const invalidate = vi.spyOn(queryClient, 'invalidateQueries');
    const { result } = renderHook(() => useCreateDraftExperiment(), {
      wrapper: createWrapper(queryClient),
    });

    result.current.mutate('sim-1');
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(digitalTwinApi.createDraftExperiment).toHaveBeenCalledWith('sim-1');
    expect(result.current.data).toEqual(draft);
    const invalidated = invalidate.mock.calls.map((c) => c[0]?.queryKey);
    expect(invalidated).toContainEqual([...queryKeys.digitalTwin.all(), 'proposed-experiments']);
    expect(invalidated).toContainEqual([...queryKeys.digitalTwin.all(), 'history']);
    expect(invalidated).toContainEqual(queryKeys.digitalTwin.simulation('sim-1'));
  });

  it('surfaces a 409 (already linked) as the mutation error', async () => {
    vi.mocked(digitalTwinApi.createDraftExperiment).mockRejectedValue(
      new Error('Simulation sim-1 is already linked to experiment exp-0.')
    );
    const queryClient = createTestQueryClient();
    const { result } = renderHook(() => useCreateDraftExperiment(), {
      wrapper: createWrapper(queryClient),
    });
    result.current.mutate('sim-1');
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.error?.message).toMatch(/already linked/);
  });
});
