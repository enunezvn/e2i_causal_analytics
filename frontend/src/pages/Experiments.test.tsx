/**
 * Experiments Page Tests — interim/fidelity live wiring (H2)
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, act, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Hooks are mocked so the page never hits the network in unit tests.
vi.mock('@/hooks/api', () => ({
  useTriggerMonitoring: vi.fn(),
  useInterimAnalyses: vi.fn(),
  useFidelityComparisons: vi.fn(),
}));

import {
  useTriggerMonitoring,
  useInterimAnalyses,
  useFidelityComparisons,
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

  it('surfaces the static-synthetic context banner when the substrate is synthetic', () => {
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue(
      monitorWith({ synthetic_data_included: true, synthetic_data_forced: false }),
    );
    render(<Experiments />, { wrapper: createWrapper() });
    expect(screen.getByText(/Static synthetic-gold substrate/i)).toBeInTheDocument();
    expect(screen.getByText(/no live feed/i)).toBeInTheDocument();
  });

  it('does NOT show the synthetic banner when the substrate is not synthetic', () => {
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue(
      monitorWith({ synthetic_data_included: false }),
    );
    render(<Experiments />, { wrapper: createWrapper() });
    expect(screen.queryByText(/Static synthetic-gold substrate/i)).not.toBeInTheDocument();
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
    expect(screen.getByText(/Static synthetic-gold substrate/i)).toBeInTheDocument();
  });

  it('adds a static-synthetic note on the Alerts tab', async () => {
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue(
      monitorWith({ synthetic_data_included: true, synthetic_data_forced: false }),
    );
    const user = userEvent.setup();
    render(<Experiments />, { wrapper: createWrapper() });
    await act(async () => {
      await user.click(screen.getByRole('tab', { name: /alerts/i }));
    });
    expect(
      screen.getByText(/computed on static synthetic-gold data/i),
    ).toBeInTheDocument();
  });
});
