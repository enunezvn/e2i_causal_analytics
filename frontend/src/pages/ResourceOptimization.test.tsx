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

  it('does NOT fabricate an allocation trend; renders an honest empty state', () => {
    // A completed result WITH allocations (entity has distinct current vs
    // optimized). The old AllocationTrendChart invented "Q1/Q2/Q3/Q4" series
    // from hardcoded 0.9/0.95 multipliers of current_allocation. After the
    // anti-fabrication fix, NO fabricated quarter series may render — instead
    // the honest "no trend data" state appears.
    (useRunOptimization as ReturnType<typeof vi.fn>).mockReturnValue({
      isPending: false,
      mutate: vi.fn(),
      data: {
        optimization_id: 'opt_live_trend',
        status: 'completed',
        resource_type: 'budget',
        objective: 'maximize_roi',
        optimal_allocations: [
          {
            entity_id: 'territory_northeast',
            entity_type: 'territory',
            current_allocation: 50000,
            optimized_allocation: 60000,
            change: 10000,
            change_percentage: 20,
            expected_impact: 1.3,
          },
        ],
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

    // Honest empty state present.
    expect(screen.getByText(/No allocation trend data/i)).toBeInTheDocument();

    // The fabricated quarter series must NOT render anywhere on the page.
    expect(screen.queryByText('Q1')).not.toBeInTheDocument();
    expect(screen.queryByText('Q2')).not.toBeInTheDocument();
    expect(screen.queryByText(/Q3 \(Current\)/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Q4 \(Optimized\)/i)).not.toBeInTheDocument();

    // The misleading "Historical and projected" card description is gone.
    expect(screen.queryByText(/Historical and projected/i)).not.toBeInTheDocument();
  });

  it('surfaces degraded storage honestly (cross-worker 404 risk)', () => {
    (useResourceHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        agent_available: true,
        scipy_available: true,
        status: 'degraded',
        storage_mode: 'degraded',
        optimizations_24h: 0,
      },
      isLoading: false,
    });
    render(<ResourceOptimization />, { wrapper: createWrapper() });
    expect(screen.getByText(/Storage Degraded/i)).toBeInTheDocument();
  });

  it('does NOT show the degraded-storage badge when storage is durable', () => {
    (useResourceHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        agent_available: true,
        scipy_available: true,
        status: 'healthy',
        storage_mode: 'durable',
        optimizations_24h: 0,
      },
      isLoading: false,
    });
    render(<ResourceOptimization />, { wrapper: createWrapper() });
    expect(screen.queryByText(/Storage Degraded/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Solver Ready/i)).toBeInTheDocument();
  });
});
