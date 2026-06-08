/**
 * ResourceOptimization Page Tests — pre-run empty state + live scenarios (H4)
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

vi.mock('@/hooks/api', () => ({
  useResourceHealth: vi.fn(),
  useRunOptimization: vi.fn(),
  useScenarios: vi.fn(),
}));

import { useResourceHealth, useRunOptimization, useScenarios } from '@/hooks/api';
import ResourceOptimization from './ResourceOptimization';

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
  (useResourceHealth as ReturnType<typeof vi.fn>).mockReturnValue({ data: { agent_available: true, scipy_available: true }, isLoading: false });
  (useScenarios as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined });
  (useRunOptimization as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined, isPending: false, mutate: vi.fn() });
});

describe('ResourceOptimization (H4)', () => {
  it('shows a pre-run prompt (not sample data) before an optimization is run', () => {
    render(<ResourceOptimization />, { wrapper: createWrapper() });
    expect(screen.getByText(/Run an optimization to see results/i)).toBeInTheDocument();
    // The fabricated sample optimization id must NOT render.
    expect(screen.queryByText('opt_abc123')).not.toBeInTheDocument();
  });

  it('renders the live optimization result after a run', () => {
    (useRunOptimization as ReturnType<typeof vi.fn>).mockReturnValue({
      isPending: false,
      mutate: vi.fn(),
      data: {
        optimization_id: 'opt_live_9',
        status: 'completed',
        resource_type: 'budget',
        objective: 'maximize_roi',
        optimal_allocations: [],
        objective_value: 1,
        solver_status: 'optimal',
        solve_time_ms: 5,
        scenarios: [],
        sensitivity_analysis: {},
        projected_total_outcome: 1000,
        projected_roi: 1.5,
        impact_by_segment: {},
        optimization_summary: 'done',
        recommendations: [],
        formulation_latency_ms: 1,
        optimization_latency_ms: 2,
        total_latency_ms: 3,
        timestamp: '2026-06-01T00:00:00Z',
        warnings: [],
      },
    });
    render(<ResourceOptimization />, { wrapper: createWrapper() });
    // The live KPI summary renders the result's projected ROI (1.50x) — the
    // pre-run prompt is gone and the sample optimization id is never shown.
    expect(screen.getByText('1.50x')).toBeInTheDocument();
    expect(screen.queryByText(/Run an optimization to see results/i)).not.toBeInTheDocument();
    expect(screen.queryByText('opt_abc123')).not.toBeInTheDocument();
  });
});
