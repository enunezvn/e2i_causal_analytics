/**
 * ResourceOptimization Page Tests
 * ================================
 * Covers: pre-run empty state, live result rendering, the async run-and-WAIT
 * wiring (empty synthetic targets + polling options), honest failure surface,
 * synthetic-data provenance banner, running indicator, brand selection, the
 * removed allocation-trend card, honest outcome units, and marginal-return
 * sensitivity copy.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

vi.mock('@/hooks/api', () => ({
  useResourceHealth: vi.fn(),
  useRunOptimizationAndWait: vi.fn(),
  useResourceOptimizationInsight: vi.fn(),
  useScenarios: vi.fn(),
}));

import {
  useResourceHealth,
  useRunOptimizationAndWait,
  useResourceOptimizationInsight,
  useScenarios,
} from '@/hooks/api';
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
  (useResourceOptimizationInsight as ReturnType<typeof vi.fn>).mockReturnValue({
    mutate: vi.fn(),
    isPending: false,
    error: null,
    data: undefined,
  });
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

  it('always renders the Strategic Interpretation insight card header (even pre-run)', async () => {
    render(<ResourceOptimization />, { wrapper: createWrapper() });
    expect(await screen.findByText(/strategic interpretation/i)).toBeInTheDocument();
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

  it('renders NO allocation-trend card at all (no fabrication, no dead placeholder)', () => {
    mockRun({
      data: completedResult({
        optimization_id: 'opt_live_trend',
        optimal_allocations: [
          {
            entity_id: 'south-T01',
            entity_type: 'territory',
            current_allocation: 50000,
            optimized_allocation: 60000,
            change: 10000,
            change_percentage: 20,
            expected_impact: 320,
          },
        ],
      }),
    });
    render(<ResourceOptimization />, { wrapper: createWrapper() });

    // The card (fabricated once, then a permanently-empty placeholder) is gone.
    expect(screen.queryByText(/Allocation Trend/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/No allocation trend data/i)).not.toBeInTheDocument();

    // The fabricated quarter series must NOT render anywhere on the page.
    expect(screen.queryByText('Q1')).not.toBeInTheDocument();
    expect(screen.queryByText('Q2')).not.toBeInTheDocument();
    expect(screen.queryByText(/Historical and projected/i)).not.toBeInTheDocument();
  });

  it('offers a brand selector and sends the selection (null for All Brands)', () => {
    const mutate = vi.fn();
    mockRun({ mutate });
    render(<ResourceOptimization />, { wrapper: createWrapper() });

    // Default: All Brands -> brand: null in the request.
    fireEvent.click(screen.getByRole('button', { name: /Run Optimization/i }));
    expect(mutate.mock.calls[0][0].request.brand).toBeNull();

    // Pick a brand -> its name goes to the backend seeder.
    const brandSelect = screen.getAllByRole('combobox')[0];
    fireEvent.change(brandSelect, { target: { value: 'Remibrutinib' } });
    fireEvent.click(screen.getByRole('button', { name: /Run Optimization/i }));
    expect(mutate.mock.calls[1][0].request.brand).toBe('Remibrutinib');
  });

  it('renders a null change_percentage as "New" (never 0%) for zero-current allocations', async () => {
    mockRun({
      data: completedResult({
        optimal_allocations: [
          {
            entity_id: 'south-T09',
            entity_type: 'territory',
            current_allocation: 0,
            optimized_allocation: 50000,
            change: 50000,
            change_percentage: null,
            expected_impact: 320,
          },
          {
            entity_id: 'south-T01',
            entity_type: 'territory',
            current_allocation: 50000,
            optimized_allocation: 60000,
            change: 10000,
            change_percentage: 20,
            expected_impact: 320,
          },
        ],
      }),
    });
    render(<ResourceOptimization />, { wrapper: createWrapper() });
    await userEvent.click(screen.getByRole('tab', { name: /Allocations/i }));

    // The new allocation must read as a move, not "0.0%".
    expect(screen.getByText('New')).toBeInTheDocument();
    expect(screen.getByText('+20.0%')).toBeInTheDocument();
    expect(screen.queryByText('0.0%')).not.toBeInTheDocument();
  });

  it('sends the actual deployed spend alongside the budget in the insight request', async () => {
    const insightMutate = vi.fn();
    (useResourceOptimizationInsight as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: insightMutate,
      isPending: false,
      error: null,
      data: undefined,
    });
    // maximize_roi underspend: budget (current) 100K, deployed only 80K.
    mockRun({
      data: completedResult({
        optimal_allocations: [
          {
            entity_id: 'south-T01',
            entity_type: 'territory',
            current_allocation: 60000,
            optimized_allocation: 50000,
            change: -10000,
            change_percentage: -16.7,
            expected_impact: 320,
          },
          {
            entity_id: 'west-T01',
            entity_type: 'territory',
            current_allocation: 40000,
            optimized_allocation: 30000,
            change: -10000,
            change_percentage: -25,
            expected_impact: 120,
          },
        ],
      }),
    });
    render(<ResourceOptimization />, { wrapper: createWrapper() });
    await userEvent.click(
      screen.getByRole('button', { name: /Generate strategic insight/i })
    );

    const payload = insightMutate.mock.calls[0][0];
    expect(payload.total_budget).toBe(100000);
    expect(payload.total_spend).toBe(80000);
  });

  it('sends a genuine $0 deployed spend as 0, not null (insight must narrate it)', async () => {
    const insightMutate = vi.fn();
    (useResourceOptimizationInsight as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: insightMutate,
      isPending: false,
      error: null,
      data: undefined,
    });
    // Extreme underspend: the optimizer recommends deploying nothing of a
    // nonzero budget. Dropping this to null would flip the insight back to
    // "total budget under optimization" and hide the Deployed $0 chip.
    mockRun({
      data: completedResult({
        optimal_allocations: [
          {
            entity_id: 'south-T01',
            entity_type: 'territory',
            current_allocation: 60000,
            optimized_allocation: 0,
            change: -60000,
            change_percentage: -100,
            expected_impact: 0,
          },
        ],
      }),
    });
    render(<ResourceOptimization />, { wrapper: createWrapper() });
    await userEvent.click(
      screen.getByRole('button', { name: /Generate strategic insight/i })
    );

    const payload = insightMutate.mock.calls[0][0];
    expect(payload.total_budget).toBe(60000);
    expect(payload.total_spend).toBe(0);
  });

  it('renders impact shares as rounded percentages by region', () => {
    mockRun({
      data: completedResult({
        impact_by_segment: { south: 38.1, northeast: 23.7, midwest: 20.3, west: 17.9 },
      }),
    });
    render(<ResourceOptimization />, { wrapper: createWrapper() });
    expect(screen.getByText(/Impact by Region/i)).toBeInTheDocument();
    expect(screen.getByText(/Share of projected outcome by region/i)).toBeInTheDocument();
  });

  it('shows outcome units (not $K) and marginal-return copy on the sensitivity tab', async () => {
    mockRun({
      data: completedResult({
        projected_total_outcome: 12345,
        sensitivity_analysis: { 'south-T01': 0.0011, 'west-T03': 0.0009 },
      }),
    });
    render(<ResourceOptimization />, { wrapper: createWrapper() });

    // KPI shows plain outcome units (12,345), not a dollar figure ($12K).
    expect(screen.getByText('12,345')).toBeInTheDocument();
    expect(screen.queryByText('$12K')).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('tab', { name: /Sensitivity/i }));
    expect(screen.getByText(/Marginal Returns/i)).toBeInTheDocument();
    // The fabricated relaxation claim is gone.
    expect(screen.queryByText(/A 10% relaxation would improve/i)).not.toBeInTheDocument();
    // Fallback (no current series, e.g. an older cached response): single value.
    expect(screen.getByText(/\+1\.10 outcome units per additional \$1K/i)).toBeInTheDocument();
  });

  it('renders the before->after equalization when a current-marginal series is present', async () => {
    mockRun({
      data: completedResult({
        // Optimized marginals equalized (~5.00 per $1K); current marginals
        // dispersed by productivity (south grew, west was cut).
        sensitivity_analysis: { 'south-T01': 0.005, 'west-T03': 0.005 },
        sensitivity_analysis_current: { 'south-T01': 0.0055, 'west-T03': 0.0041 },
      }),
    });
    render(<ResourceOptimization />, { wrapper: createWrapper() });

    await userEvent.click(screen.getByRole('tab', { name: /Sensitivity/i }));

    // Per-territory detail shows the current -> optimized transition, not a
    // single flat number.
    expect(
      screen.getByText(/\+5\.50 → \+5\.00 outcome units per additional \$1K/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/\+4\.10 → \+5\.00 outcome units per additional \$1K/i)
    ).toBeInTheDocument();
    // Direction reads off the concave relationship: current > optimized => grown.
    expect(screen.getByText('Funded up')).toBeInTheDocument();
    expect(screen.getByText('Funded down')).toBeInTheDocument();
    // The misleading "Above/At-below median" ranking of equalized values is gone.
    expect(screen.queryByText(/median/i)).not.toBeInTheDocument();
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
