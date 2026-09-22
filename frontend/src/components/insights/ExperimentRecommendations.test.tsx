/**
 * ExperimentRecommendations Tests — live experiment health monitor wiring plus the
 * proposed-experiments feed (#2206).
 *
 * The monitor half is a MONITORING feed (health, enrollment, information fraction,
 * SRM, open alerts) ranked worst-first: no fabricated Digital-Twin scores for a
 * monitoring row. The proposals half is a second real feed: completed twin
 * simulations recommending deploy/refine, not yet linked to an experiment, each
 * with the twin's own numbers; an admin can create a linked `draft` experiment.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

vi.mock('@/hooks/api', () => ({ useTriggerMonitoring: vi.fn() }));
vi.mock('@/hooks/api/use-digital-twin', () => ({
  useProposedExperiments: vi.fn(),
  useCreateDraftExperiment: vi.fn(),
}));
const mockUseAuth = vi.fn();
vi.mock('@/hooks/use-auth', () => ({ useAuth: () => mockUseAuth() }));
const mockToast = vi.fn();
vi.mock('@/hooks/use-toast', () => ({ toast: (...args: unknown[]) => mockToast(...args) }));

import { useTriggerMonitoring } from '@/hooks/api';
import { useCreateDraftExperiment, useProposedExperiments } from '@/hooks/api/use-digital-twin';
import { ExperimentRecommendations } from './ExperimentRecommendations';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </MemoryRouter>
  );
}

function summary(overrides: Record<string, unknown> = {}) {
  return {
    experiment_id: `exp_${Math.random().toString(36).slice(2, 10)}`,
    experiment_name: 'Kisqali Outreach',
    health_status: 'healthy',
    total_enrolled: 1200,
    enrollment_rate_per_day: 25,
    current_information_fraction: 0.6,
    has_srm: false,
    active_alerts: 0,
    last_checked: '2026-06-01T00:00:00Z',
    ...overrides,
  };
}

function monitorData(overrides: Record<string, unknown> = {}) {
  return {
    experiments_checked: 1,
    healthy_count: 1,
    warning_count: 0,
    critical_count: 0,
    experiments: [summary()],
    alerts: [],
    monitor_summary: '',
    recommended_actions: [],
    check_latency_ms: 10,
    timestamp: '2026-06-01T00:00:00Z',
    ...overrides,
  };
}

function mockMonitoring(data: unknown, mutate = vi.fn()) {
  (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue({
    data,
    isPending: false,
    mutate,
  });
  return mutate;
}

function proposal(overrides: Record<string, unknown> = {}) {
  return {
    simulation_id: 'sim-prop-1',
    model_id: 'model-1',
    brand: 'Remibrutinib',
    intervention_type: 'digital_engagement',
    intervention_config: { duration_weeks: 12 },
    simulated_ate: 0.3999,
    simulated_ci_lower: 0.3105,
    simulated_ci_upper: 0.4893,
    recommendation: 'deploy',
    recommendation_rationale: 'Effect is positive and the 95% CI excludes zero.',
    recommended_sample_size: 13,
    recommended_duration_weeks: 13,
    simulation_confidence: 0.72,
    data_provenance: 'cohort_estimated_synthetic_gold_v1',
    fidelity_status: 'unvalidated',
    created_at: '2026-09-22T10:00:00Z',
    proposal_basis: 'twin_simulation',
    ...overrides,
  };
}

function mockProposals(
  data: unknown,
  state: { isPending?: boolean; isError?: boolean; error?: { message: string } } = {}
) {
  (useProposedExperiments as ReturnType<typeof vi.fn>).mockReturnValue({
    data,
    isPending: state.isPending ?? false,
    isError: state.isError ?? false,
    error: state.error,
  });
}

function mockDraft(mutate = vi.fn(), state: { isPending?: boolean; variables?: string } = {}) {
  (useCreateDraftExperiment as ReturnType<typeof vi.fn>).mockReturnValue({
    mutate,
    isPending: state.isPending ?? false,
    variables: state.variables,
  });
  return mutate;
}

beforeEach(() => {
  vi.clearAllMocks();
  mockMonitoring(undefined);
  mockProposals({ proposals: [], total_proposed: 0, total_linked: 0, real_experiments_running: 0 });
  mockDraft();
  mockUseAuth.mockReturnValue({ isAdmin: false });
});

describe('ExperimentRecommendations (Experiment Health Monitor)', () => {
  it('triggers a monitoring sweep on mount and shows an empty state with no data', () => {
    const mutate = mockMonitoring(undefined);
    render(<ExperimentRecommendations />, { wrapper: createWrapper() });
    expect(mutate).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/No running experiments/i)).toBeInTheDocument();
    // The fabricated sample title must NOT appear.
    expect(screen.queryByText('Increased Call Frequency - NE Region')).not.toBeInTheDocument();
  });

  it('renders live experiment summaries with the monitor framing, not a recommendation pipeline', () => {
    mockMonitoring(monitorData());
    render(<ExperimentRecommendations />, { wrapper: createWrapper() });

    expect(screen.getByText('Experiment Health Monitor')).toBeInTheDocument();
    expect(screen.getByText('1 Monitored')).toBeInTheDocument();
    expect(screen.getByText('Kisqali Outreach')).toBeInTheDocument();
    expect(screen.getByText('Healthy')).toBeInTheDocument();
    // Dead pipeline states/actions from the old widget must NOT render.
    expect(screen.queryByText('Recommended')).not.toBeInTheDocument();
    expect(screen.queryByText('Simulated')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /simulate/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /approve/i })).not.toBeInTheDocument();
  });

  it('shows REAL health fields and never fabricates a Digital-Twin score or lift for a monitored row', () => {
    mockMonitoring(
      monitorData({
        healthy_count: 0,
        warning_count: 1,
        experiments: [
          summary({ experiment_name: 'Fabhalta Adoption', health_status: 'warning' }),
        ],
      })
    );
    render(<ExperimentRecommendations />, { wrapper: createWrapper() });

    // Real, sourced fields render. Compute the formatted value the same way the
    // component does (toLocaleString) so the assertion is locale-agnostic.
    expect(screen.getByText('Enrolled')).toBeInTheDocument();
    expect(screen.getAllByText((1200).toLocaleString()).length).toBeGreaterThan(0);
    expect(screen.getByText('Information fraction')).toBeInTheDocument();
    expect(screen.getByText('60%')).toBeInTheDocument();
    expect(screen.getByText('Open Alerts')).toBeInTheDocument();
    expect(screen.getByText('Warning')).toBeInTheDocument();

    // Fabricated Digital-Twin metrics / claims must NOT render on a monitored row.
    expect(screen.queryByText('Digital Twin Score')).not.toBeInTheDocument();
    expect(screen.queryByText('Expected Lift')).not.toBeInTheDocument();
    // The old "not yet wired" disclaimer is retired: proposals are a real feed now (#2206).
    expect(screen.queryByText(/not yet wired/i)).not.toBeInTheDocument();
    expect(screen.getByText(/nothing runs until a draft is promoted/i)).toBeInTheDocument();
  });

  it('ranks worst-first, caps the list, and links to /experiments for the rest', () => {
    const experiments = [
      ...Array.from({ length: 5 }, (_, i) =>
        summary({ experiment_id: `exp_h_${i}`, experiment_name: `Healthy ${i}` })
      ),
      summary({
        experiment_id: 'exp_warn',
        experiment_name: 'Warning Exp',
        health_status: 'warning',
        active_alerts: 1,
      }),
      summary({
        experiment_id: 'exp_crit',
        experiment_name: 'Critical Exp',
        health_status: 'critical',
        active_alerts: 3,
        has_srm: true,
      }),
    ];
    mockMonitoring(
      monitorData({
        experiments_checked: 7,
        healthy_count: 5,
        warning_count: 1,
        critical_count: 1,
        experiments,
      })
    );
    render(<ExperimentRecommendations />, { wrapper: createWrapper() });

    expect(screen.getByText('7 Monitored')).toBeInTheDocument();
    // Worst-first: critical and warning always make the top-5 cut.
    expect(screen.getByText('Critical Exp')).toBeInTheDocument();
    expect(screen.getByText('Warning Exp')).toBeInTheDocument();
    expect(screen.getByText('SRM detected')).toBeInTheDocument();
    // Only 5 cards render; the overflow goes through the /experiments link.
    expect(screen.queryByText('Healthy 3')).not.toBeInTheDocument();
    expect(screen.queryByText('Healthy 4')).not.toBeInTheDocument();
    const link = screen.getByRole('link', { name: /view all 7 monitored experiments/i });
    expect(link).toHaveAttribute('href', '/experiments');
  });

  it('surfaces the sweep-level recommended actions from the monitor agent', () => {
    mockMonitoring(
      monitorData({
        recommended_actions: [
          'URGENT: Investigate SRM in Exp A',
          'Review enrollment for Exp B',
        ],
      })
    );
    render(<ExperimentRecommendations />, { wrapper: createWrapper() });

    expect(screen.getByText('Recommended actions')).toBeInTheDocument();
    expect(screen.getByText('URGENT: Investigate SRM in Exp A')).toBeInTheDocument();
    expect(screen.getByText('Review enrollment for Exp B')).toBeInTheDocument();
  });

  it('discloses the synthetic-gold substrate when the deployment forces it', () => {
    mockMonitoring(monitorData({ synthetic_data_forced: true }));
    render(<ExperimentRecommendations />, { wrapper: createWrapper() });
    expect(screen.getByText(/synthetic-gold substrate/i)).toBeInTheDocument();
  });

  it('shows a labeled error state instead of "no experiments" when the monitor crashed', () => {
    mockMonitoring(monitorData({ experiments: [], errors: ['db node unreachable'] }));
    render(<ExperimentRecommendations />, { wrapper: createWrapper() });
    expect(screen.getByText(/Couldn’t load experiments/i)).toBeInTheDocument();
    expect(screen.queryByText(/No running experiments/i)).not.toBeInTheDocument();
  });
});

describe('ExperimentRecommendations (Proposed experiments, #2206)', () => {
  it('lists each proposal with the twin’s own numbers, fidelity state and provenance', () => {
    mockProposals({
      proposals: [proposal(), proposal({ simulation_id: 'sim-prop-2', brand: 'Kisqali', intervention_type: 'email_campaign', recommendation: 'refine', recommendation_rationale: 'The interval straddles the minimum effect.', simulated_ate: 0.021, simulated_ci_lower: -0.004, simulated_ci_upper: 0.046, recommended_sample_size: 4200, recommended_duration_weeks: 10 })],
      total_proposed: 2,
      total_linked: 0,
      real_experiments_running: 0,
    });
    render(<ExperimentRecommendations />, { wrapper: createWrapper() });

    expect(screen.getByText('Proposed experiments')).toBeInTheDocument();
    expect(screen.getByText('2 proposed')).toBeInTheDocument();
    expect(screen.getByText('Remibrutinib · Digital engagement')).toBeInTheDocument();
    expect(screen.getByText('Kisqali · Email campaign')).toBeInTheDocument();
    expect(screen.getByText('Twin says deploy')).toBeInTheDocument();
    expect(screen.getByText('Twin says refine')).toBeInTheDocument();
    // Predicted lift with its interval, exactly the twin's numbers (no invented score).
    expect(screen.getByText('+40.0% [+31.1%, +48.9%]')).toBeInTheDocument();
    expect(screen.getByText('+2.1% [-0.4%, +4.6%]')).toBeInTheDocument();
    expect(screen.getByText(/n=13 · 13 weeks · estimated on the brand cohort/)).toBeInTheDocument();
    expect(screen.getByText(/n=4,200 · 10 weeks/)).toBeInTheDocument();
    expect(screen.getAllByText('Model unvalidated')).toHaveLength(2);
    expect(screen.getByText('Effect is positive and the 95% CI excludes zero.')).toBeInTheDocument();
    expect(screen.getByText('The interval straddles the minimum effect.')).toBeInTheDocument();
  });

  it('states the honest envelope: proposals, linked, real running, model state', () => {
    mockProposals({
      proposals: [proposal()],
      total_proposed: 1,
      total_linked: 3,
      real_experiments_running: 0,
    });
    render(<ExperimentRecommendations />, { wrapper: createWrapper() });
    expect(screen.getByTestId('proposals-envelope')).toHaveTextContent(
      '1 proposal from twin simulations · 3 linked · 0 real experiments running · models unvalidated'
    );
  });

  it('explains an empty list: nothing proposed yet, or everything already linked', () => {
    mockProposals({ proposals: [], total_proposed: 0, total_linked: 0, real_experiments_running: 0 });
    const { unmount } = render(<ExperimentRecommendations />, { wrapper: createWrapper() });
    expect(screen.getByText('No proposed experiments')).toBeInTheDocument();
    expect(screen.getByText(/recommends deploy or refine yet/i)).toBeInTheDocument();
    unmount();

    mockProposals({ proposals: [], total_proposed: 0, total_linked: 4, real_experiments_running: 0 });
    render(<ExperimentRecommendations />, { wrapper: createWrapper() });
    expect(screen.getByText(/already linked to an experiment \(4 linked\)/i)).toBeInTheDocument();
  });

  it('shows a labeled error state when the proposals request failed', () => {
    mockProposals(undefined, { isError: true, error: { message: 'boom' } });
    render(<ExperimentRecommendations />, { wrapper: createWrapper() });
    expect(screen.getByText(/Couldn’t load proposed experiments/i)).toBeInTheDocument();
    expect(screen.getByText('boom')).toBeInTheDocument();
  });

  it('offers "Create draft experiment" to an admin only', () => {
    mockProposals({ proposals: [proposal()], total_proposed: 1, total_linked: 0, real_experiments_running: 0 });
    const { unmount } = render(<ExperimentRecommendations />, { wrapper: createWrapper() });
    expect(screen.queryByRole('button', { name: /create draft experiment/i })).not.toBeInTheDocument();
    unmount();

    mockUseAuth.mockReturnValue({ isAdmin: true });
    render(<ExperimentRecommendations />, { wrapper: createWrapper() });
    expect(screen.getByRole('button', { name: /create draft experiment/i })).toBeInTheDocument();
  });

  it('confirms, creates the draft for that simulation, and shows the experiment id as a draft', () => {
    mockUseAuth.mockReturnValue({ isAdmin: true });
    mockProposals({ proposals: [proposal()], total_proposed: 1, total_linked: 0, real_experiments_running: 0 });
    const mutate = mockDraft(
      vi.fn((simulationId: string, opts?: { onSuccess?: (d: unknown) => void }) => {
        opts?.onSuccess?.({
          experiment_id: 'exp-draft-001',
          simulation_id: simulationId,
          experiment_name: 'twin_proposal_Remibrutinib_digital_engagement_sim-prop',
          status: 'draft',
          brand: 'Remibrutinib',
          intervention_channel: 'digital_engagement',
          prediction_target: 'cohort_conversion_outcome',
          target_enrollment: 13,
          planned_duration_days: 91,
          linked: true,
          next_step: 'promote',
        });
      })
    );
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);

    render(<ExperimentRecommendations />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByRole('button', { name: /create draft experiment/i }));

    expect(confirmSpy).toHaveBeenCalledTimes(1);
    expect(confirmSpy.mock.calls[0][0]).toMatch(/status "draft"/);
    expect(mutate).toHaveBeenCalledTimes(1);
    expect(mutate.mock.calls[0][0]).toBe('sim-prop-1');
    expect(screen.getByText('exp-draft-001')).toBeInTheDocument();
    expect(screen.getByText(/stays a draft until promoted/i)).toBeInTheDocument();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'Draft experiment created' })
    );
    // The action is spent for this row.
    expect(screen.queryByRole('button', { name: /create draft experiment/i })).not.toBeInTheDocument();
    confirmSpy.mockRestore();
  });

  it('does nothing when the confirmation is declined', () => {
    mockUseAuth.mockReturnValue({ isAdmin: true });
    mockProposals({ proposals: [proposal()], total_proposed: 1, total_linked: 0, real_experiments_running: 0 });
    const mutate = mockDraft();
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);

    render(<ExperimentRecommendations />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByRole('button', { name: /create draft experiment/i }));

    expect(mutate).not.toHaveBeenCalled();
    confirmSpy.mockRestore();
  });

  it('reports a failed draft creation as a destructive toast, not a silent no-op', () => {
    mockUseAuth.mockReturnValue({ isAdmin: true });
    mockProposals({ proposals: [proposal()], total_proposed: 1, total_linked: 0, real_experiments_running: 0 });
    mockDraft(
      vi.fn((_id: string, opts?: { onError?: (e: { message: string }) => void }) => {
        opts?.onError?.({ message: 'Simulation sim-prop-1 is already linked to experiment X.' });
      })
    );
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);

    render(<ExperimentRecommendations />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByRole('button', { name: /create draft experiment/i }));

    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'Could not create the draft experiment', variant: 'destructive' })
    );
    confirmSpy.mockRestore();
  });
});
