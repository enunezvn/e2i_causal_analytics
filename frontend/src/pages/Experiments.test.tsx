/**
 * Experiments Page Tests — interim/fidelity live wiring (H2)
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, act, within, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Hooks are mocked so the page never hits the network in unit tests.
vi.mock('@/hooks/api', () => ({
  useTriggerMonitoring: vi.fn(),
  useInterimAnalyses: vi.fn(),
  useFidelityComparisons: vi.fn(),
  useExperimentsInsight: vi.fn(),
}));

import {
  useTriggerMonitoring,
  useInterimAnalyses,
  useFidelityComparisons,
  useExperimentsInsight,
} from '@/hooks/api';
import Experiments from './Experiments';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

const idle = { data: undefined, isLoading: false, isError: false, error: null };

beforeEach(() => {
  vi.clearAllMocks();
  (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    isPending: false,
    mutate: vi.fn(),
  });
  (useInterimAnalyses as ReturnType<typeof vi.fn>).mockReturnValue(idle);
  (useFidelityComparisons as ReturnType<typeof vi.fn>).mockReturnValue(idle);
  (useExperimentsInsight as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    isPending: false,
    error: null,
    mutate: vi.fn(),
  });
});

describe('Experiments — interim analyses (H2)', () => {
  it('renders an empty state (not sample rows) when no experiment is selected', async () => {
    const user = userEvent.setup();
    render(<Experiments />, { wrapper: createWrapper() });
    await act(async () => {
      await user.click(screen.getByRole('tab', { name: 'Analytics' }));
    });
    await waitFor(() => {
      expect(screen.getByText(/Select an experiment to view interim analyses/i)).toBeInTheDocument();
    });
    // The fabricated sample experiment ids must NOT appear.
    expect(screen.queryByText('exp_multi_brand_001')).not.toBeInTheDocument();
  });

  it('renders live interim-analysis rows for the selected experiment', async () => {
    // Live monitor roster so a real experiment card exists to select.
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue({
      isPending: false,
      mutate: vi.fn(),
      data: {
        experiments: [
          {
            experiment_id: 'exp_live_42',
            experiment_name: 'Live Experiment 42',
            health_status: 'healthy',
            total_enrolled: 100,
            enrollment_rate_per_day: 10,
            current_information_fraction: 0.5,
            has_srm: false,
            active_alerts: 0,
            last_checked: '2026-06-01T00:00:00Z',
          },
        ],
        alerts: [],
      },
    });
    (useInterimAnalyses as ReturnType<typeof vi.fn>).mockReturnValue({
      ...idle,
      data: {
        experiment_id: 'exp_live_42',
        total_analyses: 1,
        analyses: [
          {
            analysis_id: 'ia_live_1',
            analysis_number: 7,
            performed_at: '2026-06-01T00:00:00Z',
            information_fraction: 0.5,
            p_value: 0.0321,
            decision: 'continue',
          },
        ],
      },
    });
    const user = userEvent.setup();
    render(<Experiments />, { wrapper: createWrapper() });
    // Select the live experiment card (sets selectedExperiment).
    await act(async () => {
      await user.click(screen.getByText('Live Experiment 42'));
    });
    // Switch to the Analytics tab where interim analyses render.
    await act(async () => {
      await user.click(screen.getByRole('tab', { name: 'Analytics' }));
    });
    await waitFor(() => {
      // Live analysis number + decision render from the hook payload.
      expect(screen.getByText('0.0321')).toBeInTheDocument();
    });
    expect(screen.getByText('continue')).toBeInTheDocument();
    expect(screen.getByText('7')).toBeInTheDocument();
  });
});

describe('Experiments — enrollment honesty + synthetic opt-in', () => {
  it('does NOT render the fabricated hardcoded enrollment series', async () => {
    const user = userEvent.setup();
    render(<Experiments />, { wrapper: createWrapper() });
    await act(async () => {
      await user.click(screen.getByRole('tab', { name: 'Analytics' }));
    });
    // The demo weekly enrollment values (450/920/1380/...) and the chart title
    // were hardcoded scaffolding — they must be gone.
    expect(screen.queryByText('Enrollment Progress')).not.toBeInTheDocument();
    expect(screen.queryByText('450')).not.toBeInTheDocument();
    expect(screen.queryByText('1380')).not.toBeInTheDocument();
    expect(screen.queryByText('2520')).not.toBeInTheDocument();
  });

  it('passes include_synthetic=true when the synthetic opt-in is enabled', async () => {
    const mutate = vi.fn();
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isPending: false,
      mutate,
    });
    const user = userEvent.setup();
    render(<Experiments />, { wrapper: createWrapper() });

    // Toggle the synthetic opt-in, then run monitoring.
    await act(async () => {
      await user.click(screen.getByLabelText(/include synthetic/i));
    });
    await act(async () => {
      // Header button (index 0); the empty-state CTA renders a second
      // identically-named button when no experiments are loaded.
      await user.click(screen.getAllByRole('button', { name: /run monitoring/i })[0]);
    });

    expect(mutate).toHaveBeenCalledWith(
      expect.objectContaining({ include_synthetic: true }),
    );
  });

  it('defaults to real-mode (include_synthetic=false) when the opt-in is off', async () => {
    const mutate = vi.fn();
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isPending: false,
      mutate,
    });
    const user = userEvent.setup();
    render(<Experiments />, { wrapper: createWrapper() });
    await act(async () => {
      // Header button (index 0); the empty-state CTA renders a second
      // identically-named button when no experiments are loaded.
      await user.click(screen.getAllByRole('button', { name: /run monitoring/i })[0]);
    });
    expect(mutate).toHaveBeenCalledWith(
      expect.objectContaining({ include_synthetic: false }),
    );
  });
});

describe('Experiments — fidelity (H2)', () => {
  it('renders an empty state on the fidelity tab when no experiment is selected', async () => {
    const user = userEvent.setup();
    render(<Experiments />, { wrapper: createWrapper() });
    await act(async () => {
      await user.click(screen.getByRole('tab', { name: 'Digital Twin' }));
    });
    await waitFor(() => {
      expect(
        screen.getByText(/Select an experiment to view fidelity tracking/i),
      ).toBeInTheDocument();
    });
    // The fabricated "92% / 74%" summary numbers must NOT be present.
    expect(screen.queryByText('92%')).not.toBeInTheDocument();
    expect(screen.queryByText('74%')).not.toBeInTheDocument();
  });

  it('renders live fidelity summary cards for the selected experiment', async () => {
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue({
      isPending: false,
      mutate: vi.fn(),
      data: {
        experiments: [
          {
            experiment_id: 'exp_live_77',
            experiment_name: 'Fidelity Experiment 77',
            health_status: 'healthy',
            total_enrolled: 100,
            enrollment_rate_per_day: 10,
            current_information_fraction: 0.5,
            has_srm: false,
            active_alerts: 0,
            last_checked: '2026-06-01T00:00:00Z',
          },
        ],
        alerts: [],
      },
    });
    (useFidelityComparisons as ReturnType<typeof vi.fn>).mockReturnValue({
      ...idle,
      data: {
        experiment_id: 'exp_live_77',
        total_comparisons: 2,
        average_fidelity_score: 0.8,
        comparisons: [
          {
            comparison_id: 'c1',
            twin_simulation_id: 't1',
            timestamp: '2026-05-01T00:00:00Z',
            predicted_effect: 0.1,
            actual_effect: 0.09,
            prediction_error: 0.01,
            fidelity_score: 0.9,
          },
          {
            comparison_id: 'c2',
            twin_simulation_id: 't2',
            timestamp: '2026-06-01T00:00:00Z',
            predicted_effect: 0.1,
            actual_effect: 0.07,
            prediction_error: 0.03,
            fidelity_score: 0.7,
          },
        ],
      },
    });
    const user = userEvent.setup();
    render(<Experiments />, { wrapper: createWrapper() });
    await act(async () => {
      await user.click(screen.getByText('Fidelity Experiment 77'));
    });
    await act(async () => {
      await user.click(screen.getByRole('tab', { name: 'Digital Twin' }));
    });
    await waitFor(() => {
      // Initial fidelity 90% (first point) and current 70% (last point) — derived live.
      expect(screen.getByText('90%')).toBeInTheDocument();
    });
    expect(screen.getByText('70%')).toBeInTheDocument();
  });
});

// A monitor payload whose substrate is synthetic-gold, with a stale-data alert.
// Overrides merge into `data` so a test can flip synthetic_data_included/forced.
function monitorWith(overrides: Record<string, unknown>) {
  return {
    isPending: false,
    mutate: vi.fn(),
    data: {
      experiments_checked: 1,
      healthy_count: 0,
      warning_count: 1,
      critical_count: 1,
      experiments: [
        {
          experiment_id: 'exp_synth_1',
          experiment_name: 'Synthetic Experiment 1',
          health_status: 'warning',
          total_enrolled: 0,
          enrollment_rate_per_day: 0,
          current_information_fraction: 0,
          has_srm: false,
          active_alerts: 1,
          last_checked: '2026-06-01T00:00:00Z',
          is_synthetic: true,
        },
      ],
      alerts: [
        {
          alert_id: 'a1',
          alert_type: 'stale_data',
          severity: 'critical',
          experiment_id: 'exp_synth_1',
          experiment_name: 'Synthetic Experiment 1',
          message:
            "Data staleness detected in 'Synthetic Experiment 1': no new data for 5.8 days (threshold: 24.0h)",
          details: {},
          recommended_action: 'Check data pipeline.',
          timestamp: '2026-06-01T00:00:00Z',
        },
      ],
      monitor_summary: 'ok',
      recommended_actions: [],
      check_latency_ms: 10,
      timestamp: '2026-06-01T00:00:00Z',
      ...overrides,
    },
  };
}

describe('Experiments — synthetic-substrate honesty + load CTA', () => {
  it('shows a Run Monitoring CTA in the empty state that triggers a sweep', async () => {
    const mutate = vi.fn();
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isPending: false,
      mutate,
    });
    const user = userEvent.setup();
    render(<Experiments />, { wrapper: createWrapper() });
    // The empty state itself carries the call-to-action button.
    const empty = screen.getByTestId('empty-state');
    await act(async () => {
      await user.click(within(empty).getByRole('button', { name: /run monitoring/i }));
    });
    expect(mutate).toHaveBeenCalled();
  });

  it('surfaces the in-silico substrate context banner when the substrate is synthetic', () => {
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue(
      monitorWith({ synthetic_data_included: true, synthetic_data_forced: false }),
    );
    render(<Experiments />, { wrapper: createWrapper() });
    expect(
      screen.getByText(/In-silico A\/B testing on the synthetic-gold substrate/i),
    ).toBeInTheDocument();
    // The banner explains the weekly refresh cadence behind the 8-day
    // staleness threshold instead of the old "static dataset" framing.
    expect(screen.getByText(/refreshes weekly/i)).toBeInTheDocument();
  });

  it('does NOT show the synthetic banner when the substrate is not synthetic', () => {
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue(
      monitorWith({ synthetic_data_included: false }),
    );
    render(<Experiments />, { wrapper: createWrapper() });
    expect(
      screen.queryByText(/In-silico A\/B testing on the synthetic-gold substrate/i),
    ).not.toBeInTheDocument();
  });

  it('disables and relabels the synthetic checkbox when the deployment forces inclusion', () => {
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue(
      monitorWith({ synthetic_data_included: true, synthetic_data_forced: true }),
    );
    render(<Experiments />, { wrapper: createWrapper() });
    const checkbox = screen.getByLabelText(/always included in this deployment/i);
    expect(checkbox).toBeDisabled();
  });

  it('shows the substrate banner on a forced synthetic-gold deployment even when no row is is_synthetic', () => {
    // The goldstd experiments are tagged is_synthetic=False (kept servable in
    // real-mode), so synthetic_data_included can be false even though the whole
    // deployment is a synthetic-gold showcase. The deployment flag
    // (synthetic_data_forced) must drive the banner, not the per-row flag.
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue(
      monitorWith({ synthetic_data_forced: true, synthetic_data_included: false }),
    );
    render(<Experiments />, { wrapper: createWrapper() });
    expect(
      screen.getByText(/In-silico A\/B testing on the synthetic-gold substrate/i),
    ).toBeInTheDocument();
  });

  it('adds a weekly-refresh provenance note on the Alerts tab (consistent with the banner)', async () => {
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue(
      monitorWith({ synthetic_data_included: true, synthetic_data_forced: false }),
    );
    const user = userEvent.setup();
    render(<Experiments />, { wrapper: createWrapper() });
    await act(async () => {
      await user.click(screen.getByRole('tab', { name: /alerts/i }));
    });
    expect(
      screen.getByText(/computed on the weekly-refreshed synthetic-gold/i),
    ).toBeInTheDocument();
    // The old "static ... no live feed" story contradicted the weekly-refresh
    // banner on the same page — it must stay gone.
    expect(screen.queryByText(/static synthetic-gold data/i)).not.toBeInTheDocument();
  });
});

describe('Experiments — brand filter + explainability + portfolio insight (2026-07-11)', () => {
  it('never renders NaN in the KPI grid before the first monitoring run (0/0 guard)', () => {
    // Default mock: data undefined — the on-mount state before Run Monitoring.
    render(<Experiments />, { wrapper: createWrapper() });
    expect(screen.getByText('Avg Enrollment/Day')).toBeInTheDocument();
    expect(screen.queryByText('NaN')).not.toBeInTheDocument();
  });

  it('renders the brand selector with All Brands + the three platform brands', () => {
    render(<Experiments />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByRole('combobox', { name: /brand/i }));
    expect(screen.getByRole('option', { name: 'All Brands' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Remibrutinib' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Kisqali' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Fabhalta' })).toBeInTheDocument();
  });

  it('passes the selected brand and the weekly-cadence staleness threshold to the sweep', async () => {
    const mutate = vi.fn();
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isPending: false,
      mutate,
    });
    const user = userEvent.setup();
    render(<Experiments />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByRole('combobox', { name: /brand/i }));
    fireEvent.click(screen.getByRole('option', { name: 'Kisqali' }));
    await act(async () => {
      await user.click(screen.getAllByRole('button', { name: /run monitoring/i })[0]);
    });
    expect(mutate).toHaveBeenCalledWith(
      expect.objectContaining({ brand: 'Kisqali', stale_data_threshold_hours: 192 }),
    );
  });

  it('omits brand from the request when All Brands is selected (server interleaves)', async () => {
    const mutate = vi.fn();
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isPending: false,
      mutate,
    });
    const user = userEvent.setup();
    render(<Experiments />, { wrapper: createWrapper() });
    await act(async () => {
      await user.click(screen.getAllByRole('button', { name: /run monitoring/i })[0]);
    });
    expect(mutate).toHaveBeenCalledWith(
      expect.objectContaining({ brand: undefined, stale_data_threshold_hours: 192 }),
    );
  });

  it('renders the experiment description, brand and intervention badges on the card', () => {
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue({
      isPending: false,
      mutate: vi.fn(),
      data: {
        experiments: [
          {
            experiment_id: 'exp_meaningful_1',
            experiment_name:
              'Fabhalta: Speaker Program Invitation → PNH therapy persistence — lapsed prescribers, west (#002)',
            health_status: 'healthy',
            total_enrolled: 300,
            enrollment_rate_per_day: 8,
            current_information_fraction: 0.3,
            has_srm: false,
            active_alerts: 0,
            last_checked: '2026-07-01T00:00:00Z',
            brand: 'Fabhalta',
            intervention_channel: 'speaker_program_invitation',
            description:
              'In-silico A/B test: does speaker program invitation increase PNH therapy persistence among lapsed prescribers…',
          },
        ],
        alerts: [],
      },
    });
    render(<Experiments />, { wrapper: createWrapper() });
    expect(screen.getByText(/In-silico A\/B test: does speaker program/i)).toBeInTheDocument();
    expect(screen.getByText('Fabhalta')).toBeInTheDocument();
    // Channel badge renders the humanized taxonomy label
    expect(screen.getByText('Speaker Program Invitation')).toBeInTheDocument();
    // No raw-id fallback when a description exists
    expect(screen.queryByText('exp_meaningful_1')).not.toBeInTheDocument();
  });

  it('falls back to the experiment id when the row predates the explainability metadata', () => {
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue(
      monitorWith({}),
    );
    render(<Experiments />, { wrapper: createWrapper() });
    // monitorWith's roster row has no description -> honest id fallback.
    expect(screen.getByText('exp_synth_1')).toBeInTheDocument();
  });

  it('renders the portfolio insight card and generates with the current scope', async () => {
    const insightMutate = vi.fn();
    (useExperimentsInsight as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isPending: false,
      error: null,
      mutate: insightMutate,
    });
    const user = userEvent.setup();
    render(<Experiments />, { wrapper: createWrapper() });
    expect(screen.getByText('Portfolio Strategic Read')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('combobox', { name: /brand/i }));
    fireEvent.click(screen.getByRole('option', { name: 'Fabhalta' }));
    await act(async () => {
      await user.click(screen.getByRole('button', { name: /generate/i }));
    });
    expect(insightMutate).toHaveBeenCalledWith(
      expect.objectContaining({ brand: 'Fabhalta', include_synthetic: false }),
    );
  });
});

describe('Experiments — health honesty (2026-07-11 follow-up)', () => {
  // Roster with one experiment carrying a REAL enrollment plan; overrides let
  // each test strip the plan or attach a health reason.
  const rosterWith = (exp: Record<string, unknown>) => ({
    isPending: false,
    mutate: vi.fn(),
    data: {
      experiments_checked: 1,
      total_running: 955,
      healthy_count: 0,
      warning_count: 1,
      critical_count: 0,
      experiments: [
        {
          experiment_id: 'exp_h1',
          experiment_name: 'Health Experiment 1',
          health_status: 'warning',
          total_enrolled: 500,
          enrollment_rate_per_day: 7.1,
          current_information_fraction: 0.83,
          target_enrollment: 600,
          has_srm: false,
          active_alerts: 0,
          last_checked: '2026-07-11T00:00:00Z',
          ...exp,
        },
      ],
      alerts: [],
      monitor_summary: 'ok',
      recommended_actions: [],
      check_latency_ms: 10,
      timestamp: '2026-07-11T00:00:00Z',
    },
  });

  it('shows the running-portfolio size, not the monitored-roster cap, as Active Experiments', async () => {
    // Live incident: "25 active experiments seems to be hard coded" — the KPI
    // card showed the 25-roster cap for every scope. It must show the exact
    // running count, with the monitored-roster size in the card's info tooltip.
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue(rosterWith({}));
    const user = userEvent.setup();
    render(<Experiments />, { wrapper: createWrapper() });
    expect(screen.getByText('955')).toBeInTheDocument();
    // KPICard renders `description` behind its info icon — hover it.
    const header = screen.getByText('Active Experiments').closest('div')!.parentElement!;
    const infoIcon = header.querySelector('.cursor-help');
    expect(infoIcon).not.toBeNull();
    await act(async () => {
      await user.hover(infoIcon as Element);
    });
    const notes = await screen.findAllByText(/1 newest monitored this sweep/i);
    expect(notes.length).toBeGreaterThan(0);
  });

  it('explains the health flag on hover via the server-computed reason', async () => {
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue(
      rosterWith({
        health_reason:
          'Past planned duration (day 70 of 60) at 83% of the 600 enrollment target',
      }),
    );
    const user = userEvent.setup();
    render(<Experiments />, { wrapper: createWrapper() });
    await act(async () => {
      await user.hover(screen.getByText('warning'));
    });
    const explanations = await screen.findAllByText(/Past planned duration \(day 70 of 60\)/i);
    expect(explanations.length).toBeGreaterThan(0);
  });

  it('renders plan progress as a percentage of the recorded target', () => {
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue(rosterWith({}));
    render(<Experiments />, { wrapper: createWrapper() });
    expect(screen.getByText('Plan Progress')).toBeInTheDocument();
    expect(screen.getByText('83% of 600')).toBeInTheDocument();
  });

  it('renders an honest dash (never a fabricated 0%) when no plan is recorded', () => {
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue(
      rosterWith({ current_information_fraction: null, target_enrollment: null }),
    );
    render(<Experiments />, { wrapper: createWrapper() });
    expect(screen.getByText('—')).toBeInTheDocument();
    expect(screen.queryByText('0%')).not.toBeInTheDocument();
  });
});
