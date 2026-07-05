/**
 * ExperimentRecommendations Tests — live experiment health monitor wiring.
 *
 * The card is a MONITORING feed (health, enrollment, information fraction,
 * SRM, open alerts) ranked worst-first: no fabricated Digital-Twin scores,
 * no invented Recommended/Simulated/Approved pipeline states.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

vi.mock('@/hooks/api', () => ({ useTriggerMonitoring: vi.fn() }));

import { useTriggerMonitoring } from '@/hooks/api';
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

beforeEach(() => {
  vi.clearAllMocks();
  mockMonitoring(undefined);
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

  it('shows REAL health fields and never fabricates a Digital-Twin score or lift', () => {
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

    // Fabricated Digital-Twin metrics / claims must NOT render.
    expect(screen.queryByText('Digital Twin Score')).not.toBeInTheDocument();
    expect(screen.queryByText('Expected Lift')).not.toBeInTheDocument();
    // The honest disclosure that twin pre-screening is not wired IS present.
    expect(screen.getByText(/not yet wired/i)).toBeInTheDocument();
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
