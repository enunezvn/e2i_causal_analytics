/**
 * ResourceOptimization Page Tests
 * ================================
 * Covers: pre-run empty state, live result rendering, the async run-and-WAIT
 * wiring (empty synthetic targets + polling options), honest failure surface,
 * synthetic-data provenance banner, running indicator, and the no-fabricated-
 * trend guarantee.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

vi.mock('@/hooks/api', () => ({
  useResourceHealth: vi.fn(),
  useRunOptimizationAndWait: vi.fn(),
  useScenarios: vi.fn(),
}));

import { useResourceHealth, useRunOptimizationAndWait, useScenarios } from '@/hooks/api';
import ResourceOptimization from './ResourceOptimization';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

const mockRun = (overrides: Record<string, unknown> = {}) => {
  (useRunOptimizationAndWait as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    error: null,
    isPending: false,
    mutate: vi.fn(),
    ...overrides,
  });
};

function completedResult(overrides: Record<string, unknown> = {}) {
  return {
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
    projected_roi: 0.0166,
    impact_by_segment: {},
    optimization_summary: 'done',
    recommendations: [],
    formulation_latency_ms: 1,
    optimization_latency_ms: 2,
    total_latency_ms: 3,
    timestamp: '2026-06-01T00:00:00Z',
    warnings: [],
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  (useResourceHealth as ReturnType<typeof vi.fn>).mockReturnValue({
    data: { agent_available: true, scipy_available: true },
    isLoading: false,
  });
  (useScenarios as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined });
  mockRun();
});

describe('ResourceOptimization', () => {
  it('shows a pre-run prompt (not sample data) before an optimization is run', () => {
    render(<ResourceOptimization />, { wrapper: createWrapper() });
    expect(screen.getByText(/Run an optimization to see results/i)).toBeInTheDocument();
    // The fabricated sample optimization id must NOT render.
    expect(screen.queryByText('opt_abc123')).not.toBeInTheDocument();
  });

  it('always surfaces the illustrative / synthetic-data badge', () => {
    render(<ResourceOptimization />, { wrapper: createWrapper() });
    expect(screen.getByText(/Illustrative · synthetic data/i)).toBeInTheDocument();
  });

  it('runs via run-and-wait: sends empty targets + polling options (no chicken-and-egg seed)', () => {
    const mutate = vi.fn();
    mockRun({ mutate });
    render(<ResourceOptimization />, { wrapper: createWrapper() });

    fireEvent.click(screen.getByRole('button', { name: /Run Optimization/i }));

    expect(mutate).toHaveBeenCalledTimes(1);
    const arg = mutate.mock.calls[0][0];
    // Backend seeds synthetic targets — the page must NOT re-send prior allocations.
    expect(arg.request.allocation_targets).toEqual([]);
    // Polling must be configured (fire-and-forget was the original "does nothing" bug).
    expect(arg.pollIntervalMs).toBeGreaterThan(0);
    expect(arg.maxWaitMs).toBeGreaterThan(0);
  });

  it('shows a running indicator while the optimization is pending', () => {
    mockRun({ isPending: true });
    render(<ResourceOptimization />, { wrapper: createWrapper() });
    expect(screen.getByText(/Running optimization for/i)).toBeInTheDocument();
  });

  it('surfaces an honest failure banner when the run errors', () => {
    mockRun({ error: new Error('Optimization timed out after 120000ms') });
    render(<ResourceOptimization />, { wrapper: createWrapper() });
    expect(screen.getByText(/Optimization failed/i)).toBeInTheDocument();
    expect(screen.getByText(/timed out after 120000ms/i)).toBeInTheDocument();
  });

  it('renders the live optimization result and the honest outcome-lift KPI', () => {
    mockRun({ data: completedResult() });
    render(<ResourceOptimization />, { wrapper: createWrapper() });
    // projected_roi 0.0166 is an incremental ratio -> shown as "+1.7%", NOT a
    // misleading "0.02x" multiple.
    expect(screen.getByText('+1.7%')).toBeInTheDocument();
    expect(screen.getByText(/Projected Outcome Lift/i)).toBeInTheDocument();
    expect(screen.queryByText(/Run an optimization to see results/i)).not.toBeInTheDocument();
    expect(screen.queryByText('opt_abc123')).not.toBeInTheDocument();
  });

  it('surfaces the synthetic-data provenance banner on the result', () => {
    mockRun({
      data: completedResult({
        warnings: [
          'SYNTHETIC DATA: no real per-entity budget source is wired, so this optimization ran on 10 territories seeded from synthetic territory_metrics.',
        ],
      }),
    });
    render(<ResourceOptimization />, { wrapper: createWrapper() });
    expect(screen.getByText(/Illustrative result — synthetic data/i)).toBeInTheDocument();
    expect(screen.getByText(/seeded from synthetic territory_metrics/i)).toBeInTheDocument();
  });

  it('renders each warning exactly once, even on the Recommendations tab', () => {
    // The always-visible provenance banner AND a per-tab "Warnings" card under
    // Recommendations both rendered optimizationResult.warnings -> the same line
    // appeared twice once that tab was opened. The card was removed; warnings are
    // surfaced once, up front. (The 32x backend duplication is fixed separately.)
    mockRun({
      data: completedResult({
        warnings: [
          'SYNTHETIC DATA: no real per-entity budget source is wired, so this optimization ran on 10 territories seeded from synthetic territory_metrics.',
        ],
      }),
    });
    render(<ResourceOptimization />, { wrapper: createWrapper() });
    // Default (allocations) tab: surfaced once by the top banner.
    expect(
      screen.getAllByText(/seeded from synthetic territory_metrics/i)
    ).toHaveLength(1);
    // Open the Recommendations tab, where the duplicate "Warnings" card lived.
    fireEvent.click(screen.getByRole('tab', { name: /Recommendations/i }));
    expect(
      screen.getAllByText(/seeded from synthetic territory_metrics/i)
    ).toHaveLength(1);
  });

  it('does NOT fabricate an allocation trend; renders an honest empty state', () => {
    mockRun({
      data: completedResult({
        optimization_id: 'opt_live_trend',
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
      }),
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
