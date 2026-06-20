/**
 * GapAnalysis Page — Run-Analysis wiring + Warning Banner coverage
 * ================================================================
 *
 * The "Run Analysis" button previously used a fire-and-forget async call with
 * no polling: it refetched the opportunities list immediately, racing ahead of
 * the ~8s background job, so the page never showed the new result and the
 * button appeared to "do nothing". These tests pin the corrected wiring — the
 * page now uses `useRunGapAnalysisAndWait` (poll-to-completion):
 *
 *  - clicking Run Analysis fires the mutation with poll options (NOT asyncMode);
 *  - while pending, an explicit running indicator renders and the button is
 *    disabled;
 *  - a genuine failure/timeout surfaces in a labeled banner;
 *  - completed-analysis `warnings[]` still render (F-010-frontend).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import GapAnalysis from './GapAnalysis';

vi.mock('recharts', async () => {
  const actual = await vi.importActual('recharts');
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container">{children}</div>
    ),
  };
});

vi.mock('@/hooks/api', () => ({
  useOpportunities: vi.fn(),
  useGapHealth: vi.fn(),
  useRunGapAnalysisAndWait: vi.fn(),
}));

import {
  useOpportunities,
  useGapHealth,
  useRunGapAnalysisAndWait,
} from '@/hooks/api';

type MockFn = ReturnType<typeof vi.fn>;

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

/** Default the two query hooks to an empty, healthy state. */
function mockQueries() {
  (useOpportunities as MockFn).mockReturnValue({
    data: { opportunities: [], total_addressable_value: 0, quick_wins_count: 0, strategic_bets_count: 0 },
    isLoading: false,
    refetch: vi.fn().mockResolvedValue({}),
  });
  (useGapHealth as MockFn).mockReturnValue({
    data: { agent_available: true, analyses_24h: 3 },
    isLoading: false,
  });
}

function mockRun(overrides: Record<string, unknown> = {}) {
  const mutate = vi.fn();
  (useRunGapAnalysisAndWait as MockFn).mockReturnValue({
    data: undefined,
    mutate,
    isPending: false,
    error: null,
    ...overrides,
  });
  return mutate;
}

/**
 * Build a minimal PrioritizedOpportunity carrying a curated `category` plus the
 * fields the card reads. `category` is set by the LIST endpoint (membership in
 * the prioritizer's quick_wins/strategic_bets lists) — it is NOT derived from
 * `implementation_difficulty` (deliberately mismatched here to prove the page
 * badges by category, not by effort).
 */
function makeOpp(opts: {
  id: string;
  rank: number;
  category: 'quick_win' | 'strategic_bet' | 'other';
  difficulty: 'low' | 'medium' | 'high';
  action?: string;
  roi?: number;
}) {
  return {
    rank: opts.rank,
    gap: {
      gap_id: opts.id,
      metric: 'trx',
      segment: 'region',
      segment_value: 'Northeast',
      current_value: 100,
      target_value: 150,
      gap_size: 50,
      gap_percentage: 33.3,
      gap_type: 'vs_target',
    },
    roi_estimate: {
      gap_id: opts.id,
      estimated_revenue_impact: 1_000_000,
      estimated_cost_to_close: 100_000,
      expected_roi: opts.roi ?? 5,
      risk_adjusted_roi: opts.roi ?? 5,
      payback_period_months: 6,
      attribution_level: 'high',
      attribution_rate: 0.8,
      confidence: 0.85,
    },
    recommended_action: opts.action ?? `Action ${opts.id}`,
    implementation_difficulty: opts.difficulty,
    time_to_impact: '3 months',
    category: opts.category,
  };
}

/**
 * Load the opportunities query with a curated mix. `strategic_bets_count` /
 * `quick_wins_count` are the AUTHORITATIVE headline counts and MUST equal the
 * number of opportunities tagged with the matching category (the no-phantom
 * invariant the page must preserve in the default "All" view).
 */
function mockLoadedOpportunities() {
  const opportunities = [
    // category=strategic_bet but LOW difficulty — proves the badge follows the
    // curated category, not the effort attribute.
    makeOpp({ id: 'sb1', rank: 1, category: 'strategic_bet', difficulty: 'low', action: 'Expand specialty coverage', roi: 7 }),
    makeOpp({ id: 'sb2', rank: 2, category: 'strategic_bet', difficulty: 'high', action: 'Launch new payer program', roi: 6 }),
    // category=quick_win but HIGH difficulty — same point in reverse.
    makeOpp({ id: 'qw1', rank: 3, category: 'quick_win', difficulty: 'high', action: 'Optimize call cadence', roi: 5 }),
    makeOpp({ id: 'ot1', rank: 4, category: 'other', difficulty: 'medium', action: 'Refresh sample mix', roi: 3 }),
  ];
  (useOpportunities as MockFn).mockReturnValue({
    data: {
      opportunities,
      total_addressable_value: 4_000_000,
      quick_wins_count: 1,
      strategic_bets_count: 2,
    },
    isLoading: false,
    refetch: vi.fn().mockResolvedValue({}),
  });
  return { opportunities, quick_wins_count: 1, strategic_bets_count: 2 };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockQueries();
});

describe('GapAnalysis — F-002 empty state', () => {
  it('renders empty state when no opportunities loaded (F-002)', () => {
    mockRun();
    render(<GapAnalysis />, { wrapper: createWrapper() });

    expect(screen.getByText(/No gap opportunities available/)).toBeInTheDocument();
    // Former SAMPLE_OPPORTUNITIES strings must not appear.
    expect(screen.queryByText(/Add 2 field reps/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Northeast/)).not.toBeInTheDocument();
  });
});

describe('GapAnalysis — Run Analysis wiring (poll-to-completion)', () => {
  it('fires the analysis mutation with poll options (not fire-and-forget) when Run Analysis is clicked', async () => {
    const mutate = mockRun();
    const user = userEvent.setup();
    render(<GapAnalysis />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('button', { name: /run analysis/i }));

    expect(mutate).toHaveBeenCalledTimes(1);
    const arg = mutate.mock.calls[0][0] as {
      request: { brand: string; metrics: string[]; segments: string[] };
      asyncMode?: boolean;
      pollIntervalMs?: number;
      maxWaitMs?: number;
    };
    // Default brand is Kisqali; request carries the real metric/segment dims.
    expect(arg.request.brand).toBe('Kisqali');
    expect(arg.request.metrics).toContain('trx');
    expect(arg.request.segments).toContain('region');
    // The fix: poll-to-completion, NOT the old fire-and-forget async shape.
    expect(arg.pollIntervalMs).toBeGreaterThan(0);
    expect(arg.maxWaitMs).toBeGreaterThan(0);
    expect(arg.asyncMode).toBeUndefined();
  });

  it('shows an explicit running indicator and disables the button while pending', () => {
    mockRun({ isPending: true });
    render(<GapAnalysis />, { wrapper: createWrapper() });

    expect(screen.getByText(/Running gap analysis for Kisqali/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /run analysis/i })).toBeDisabled();
  });

  it('surfaces a genuine failure/timeout in a labeled banner', () => {
    mockRun({ error: new Error('Gap analysis timed out after 120000ms') });
    render(<GapAnalysis />, { wrapper: createWrapper() });

    expect(screen.getByText(/Gap analysis failed/i)).toBeInTheDocument();
    expect(screen.getByText(/timed out after 120000ms/)).toBeInTheDocument();
  });

  it('does not render the running indicator or error banner in the idle state', () => {
    mockRun();
    render(<GapAnalysis />, { wrapper: createWrapper() });

    expect(screen.queryByText(/Running gap analysis/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Gap analysis failed/i)).not.toBeInTheDocument();
  });
});

describe('GapAnalysis — brand selection wiring (All Brands)', () => {
  it('passes the selected brand to useOpportunities by default (Kisqali)', () => {
    mockRun();
    render(<GapAnalysis />, { wrapper: createWrapper() });

    expect(useOpportunities).toHaveBeenCalledWith(
      expect.objectContaining({ brand: 'Kisqali', limit: 50 }),
    );
  });

  it('maps "All Brands" to no brand filter (undefined) and disables Run Analysis', async () => {
    const user = userEvent.setup();
    mockRun();
    render(<GapAnalysis />, { wrapper: createWrapper() });

    // Empty state renders only the brand Select (the difficulty Select lives in
    // the Tabs, which are hidden when there are no opportunities).
    await user.click(screen.getByRole('combobox'));
    await user.click(await screen.findByRole('option', { name: /all brands/i }));

    // 'all' must reach the API as an ABSENT brand filter, not the literal string
    // "all" (which would match no brand and empty the page).
    await waitFor(() => {
      expect(useOpportunities).toHaveBeenLastCalledWith(
        expect.objectContaining({ brand: undefined, limit: 50 }),
      );
    });
    // Run Analysis needs a concrete brand → disabled for the cross-brand view.
    expect(screen.getByRole('button', { name: /run analysis/i })).toBeDisabled();
  });
});

describe('GapAnalysis — Quick Win/Strategic Bet framework (effort folded in)', () => {
  it('badges each card by its CURATED category (Quick Win / Strategic Bet / Other), not by effort', () => {
    mockLoadedOpportunities();
    mockRun();
    render(<GapAnalysis />, { wrapper: createWrapper() });

    // Primary framework badges render.
    expect(screen.getAllByText('Strategic Bet').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Quick Win').length).toBeGreaterThan(0);
  });

  it('preserves the no-phantom invariant: #cards badged "Strategic Bet" === strategic_bets_count (same for Quick Wins)', () => {
    const { strategic_bets_count, quick_wins_count } = mockLoadedOpportunities();
    mockRun();
    render(<GapAnalysis />, { wrapper: createWrapper() });

    // The Opportunities tab is the default; count the primary category badges in
    // the card list (the table lives in the Charts tab and is not rendered yet).
    expect(screen.getAllByText('Strategic Bet')).toHaveLength(strategic_bets_count);
    expect(screen.getAllByText('Quick Win')).toHaveLength(quick_wins_count);
  });

  it('shows a folded-in effort sub-badge on each card (secondary attribute, prefixed "Effort:")', () => {
    const { opportunities } = mockLoadedOpportunities();
    mockRun();
    render(<GapAnalysis />, { wrapper: createWrapper() });

    const effortBadges = screen.getAllByText(/Effort:/i);
    // One folded-in effort badge per visible card.
    expect(effortBadges).toHaveLength(opportunities.length);
    // The old primary-label form must be gone.
    expect(screen.queryByText('High Effort')).not.toBeInTheDocument();
    expect(screen.queryByText('Low Effort')).not.toBeInTheDocument();
    expect(screen.queryByText('Medium Effort')).not.toBeInTheDocument();
  });

  it('offers the framework (positive) options in the opportunity-type dropdown and NOT "High Effort"', async () => {
    mockLoadedOpportunities();
    mockRun();
    const user = userEvent.setup();
    render(<GapAnalysis />, { wrapper: createWrapper() });

    // Two comboboxes now: brand (first) + opportunity type (second).
    const combos = screen.getAllByRole('combobox');
    expect(combos.length).toBeGreaterThanOrEqual(2);
    await user.click(combos[1]);

    expect(await screen.findByRole('option', { name: 'All Opportunities' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Quick Wins' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Strategic Bets' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Other' })).toBeInTheDocument();
    // The negative-reading effort labels must NOT be options.
    expect(screen.queryByRole('option', { name: /High Effort/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: /Low Effort/i })).not.toBeInTheDocument();
  });

  it('narrows the visible cards to category === strategic_bet when "Strategic Bets" is selected', async () => {
    const { strategic_bets_count } = mockLoadedOpportunities();
    mockRun();
    const user = userEvent.setup();
    render(<GapAnalysis />, { wrapper: createWrapper() });

    const combos = screen.getAllByRole('combobox');
    await user.click(combos[1]);
    await user.click(await screen.findByRole('option', { name: 'Strategic Bets' }));

    await waitFor(() => {
      // Only strategic-bet cards remain → no Quick Win / Other badges.
      expect(screen.queryByText('Quick Win')).not.toBeInTheDocument();
    });
    expect(screen.queryByText('Other')).not.toBeInTheDocument();
    // The strategic-bet cards are still all present.
    expect(screen.getAllByText('Strategic Bet')).toHaveLength(strategic_bets_count);
    expect(screen.getByText('Expand specialty coverage')).toBeInTheDocument();
    expect(screen.queryByText('Optimize call cadence')).not.toBeInTheDocument();
  });
});

/** One fully-formed opportunity; competitor density overridable per test. */
function opportunity(roiOverrides: Record<string, unknown> = {}) {
  return {
    rank: 1,
    gap: {
      gap_id: 'region_Northeast_trx',
      metric: 'trx',
      segment: 'region',
      segment_value: 'Northeast',
      current_value: 85,
      target_value: 100,
      gap_size: 15,
      gap_percentage: 15,
      gap_type: 'vs_target',
    },
    roi_estimate: {
      gap_id: 'region_Northeast_trx',
      estimated_revenue_impact: 500000,
      estimated_cost_to_close: 100000,
      expected_roi: 5,
      risk_adjusted_roi: 4,
      payback_period_months: 6,
      attribution_level: 'partial',
      attribution_rate: 0.7,
      confidence: 0.8,
      ...roiOverrides,
    },
    recommended_action: 'Increase field coverage',
    implementation_difficulty: 'low',
    time_to_impact: '3-6 months',
    category: 'quick_win',
  };
}

function mockOpportunities(list: unknown[]) {
  (useOpportunities as MockFn).mockReturnValue({
    data: {
      opportunities: list,
      total_addressable_value: 500000,
      quick_wins_count: list.length,
      strategic_bets_count: 0,
    },
    isLoading: false,
    refetch: vi.fn().mockResolvedValue({}),
  });
}

describe('GapAnalysis — competitor density surfacing (#1056)', () => {
  it('renders the market-landscape competitor density on a bet that carries it', () => {
    mockRun();
    mockOpportunities([
      opportunity({
        competitor_products_count: 3,
        competitor_density_label: 'moderate',
        competitor_drug_names: ['Verzenio', 'Ibrance', 'Kisqali'],
      }),
    ]);

    render(<GapAnalysis />, { wrapper: createWrapper() });

    expect(screen.getByText(/Market landscape \(3 rivals\)/i)).toBeInTheDocument();
    expect(screen.getByText('moderate')).toBeInTheDocument();
    expect(screen.getByText('Verzenio')).toBeInTheDocument();
  });

  it('omits the badge when a bet has no competitor density (honest empty state)', () => {
    mockRun();
    mockOpportunities([
      opportunity({
        competitor_products_count: 0,
        competitor_density_label: 'unknown',
        competitor_drug_names: [],
      }),
    ]);

    render(<GapAnalysis />, { wrapper: createWrapper() });

    // The bet still renders, but no market-landscape badge appears.
    expect(screen.getByText('Increase field coverage')).toBeInTheDocument();
    expect(screen.queryByText(/Market landscape/i)).not.toBeInTheDocument();
  });
});

describe('GapAnalysis — warnings rendering (F-010-frontend)', () => {
  it('does not render WarningBanner before mutation runs', () => {
    mockRun();
    render(<GapAnalysis />, { wrapper: createWrapper() });
    expect(screen.queryByTestId('warning-banner')).not.toBeInTheDocument();
  });

  it('renders WarningBanner when the completed analysis includes warnings[]', () => {
    mockRun({
      data: {
        analysis_id: 'gap_001',
        status: 'completed',
        warnings: ['ROI estimates degraded due to missing brand data'],
        opportunities: [],
        total_addressable_value: 0,
        quick_wins_count: 0,
        strategic_bets_count: 0,
      },
    });

    render(<GapAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByTestId('warning-banner')).toBeInTheDocument();
    expect(
      screen.getByText('ROI estimates degraded due to missing brand data'),
    ).toBeInTheDocument();
  });
});
