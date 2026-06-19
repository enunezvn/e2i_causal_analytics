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
