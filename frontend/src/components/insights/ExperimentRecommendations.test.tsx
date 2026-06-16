/**
 * ExperimentRecommendations Tests — live experiments wiring (H3)
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

vi.mock('@/hooks/api', () => ({ useTriggerMonitoring: vi.fn() }));

import { useTriggerMonitoring } from '@/hooks/api';
import { ExperimentRecommendations } from './ExperimentRecommendations';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    isPending: false,
    mutate: vi.fn(),
  });
});

describe('ExperimentRecommendations (H3)', () => {
  it('triggers a monitoring sweep on mount and shows an empty state with no data', () => {
    const mutate = vi.fn();
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isPending: false,
      mutate,
    });
    render(<ExperimentRecommendations />, { wrapper: createWrapper() });
    expect(mutate).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/No experiments to recommend/i)).toBeInTheDocument();
    // The fabricated sample title must NOT appear.
    expect(screen.queryByText('Increased Call Frequency - NE Region')).not.toBeInTheDocument();
  });

  it('renders live experiment summaries when monitoring returns data', () => {
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue({
      isPending: false,
      mutate: vi.fn(),
      data: {
        experiments_checked: 1,
        healthy_count: 1,
        warning_count: 0,
        critical_count: 0,
        experiments: [
          {
            experiment_id: 'exp_live_1',
            experiment_name: 'Kisqali Outreach',
            health_status: 'healthy',
            total_enrolled: 1200,
            enrollment_rate_per_day: 25,
            current_information_fraction: 0.6,
            has_srm: false,
            active_alerts: 0,
            last_checked: '2026-06-01T00:00:00Z',
          },
        ],
        alerts: [],
        monitor_summary: '',
        recommended_actions: [],
        check_latency_ms: 10,
        timestamp: '2026-06-01T00:00:00Z',
      },
    });
    render(<ExperimentRecommendations />, { wrapper: createWrapper() });
    expect(screen.getByText('Kisqali Outreach')).toBeInTheDocument();
  });

  it('shows REAL health fields and never fabricates a Digital-Twin score or lift', () => {
    (useTriggerMonitoring as ReturnType<typeof vi.fn>).mockReturnValue({
      isPending: false,
      mutate: vi.fn(),
      data: {
        experiments_checked: 1,
        healthy_count: 0,
        warning_count: 1,
        critical_count: 0,
        experiments: [
          {
            experiment_id: 'exp_live_2',
            experiment_name: 'Fabhalta Adoption',
            health_status: 'warning',
            total_enrolled: 1200,
            enrollment_rate_per_day: 25,
            current_information_fraction: 0.6,
            has_srm: false,
            active_alerts: 0,
            last_checked: '2026-06-01T00:00:00Z',
          },
        ],
        alerts: [],
        monitor_summary: '',
        recommended_actions: [],
        check_latency_ms: 10,
        timestamp: '2026-06-01T00:00:00Z',
      },
    });
    render(<ExperimentRecommendations />, { wrapper: createWrapper() });

    // Real, sourced fields render.
    expect(screen.getByText('Enrolled')).toBeInTheDocument();
    expect(screen.getByText('1,200')).toBeInTheDocument();
    expect(screen.getByText('Information fraction')).toBeInTheDocument();
    expect(screen.getByText('60%')).toBeInTheDocument();
    expect(screen.getByText('Open Alerts')).toBeInTheDocument();

    // Fabricated Digital-Twin metrics / claims must NOT render.
    expect(screen.queryByText('Digital Twin Score')).not.toBeInTheDocument();
    expect(screen.queryByText('Expected Lift')).not.toBeInTheDocument();
    expect(
      screen.queryByText(/simulated using our Digital Twin environment/i),
    ).not.toBeInTheDocument();
    // The honest disclosure that twin pre-screening is not wired IS present.
    expect(screen.getByText(/not yet wired/i)).toBeInTheDocument();
  });
});
