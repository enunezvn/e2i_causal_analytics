/**
 * PriorityActionsROI Tests — live opportunities wiring (H3)
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

vi.mock('@/hooks/api', () => ({ useOpportunities: vi.fn() }));

// Spy useNavigate (also provides the Router context the component now needs).
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>();
  return { ...actual, useNavigate: () => mockNavigate };
});

import { useOpportunities } from '@/hooks/api';
import { PriorityActionsROI } from './PriorityActionsROI';

const SAMPLE_OPP = {
  rank: 1,
  gap: { gap_id: 'g1', metric: 'trx', segment: 'region', segment_value: 'NE', current_value: 1, target_value: 2, gap_size: 1, gap_percentage: 50, gap_type: 'vs_target' },
  roi_estimate: { gap_id: 'g1', estimated_revenue_impact: 2_400_000, estimated_cost_to_close: 100000, expected_roi: 3.1, risk_adjusted_roi: 2.8, payback_period_months: 4, attribution_level: 'territory', attribution_rate: 0.6, confidence: 0.9 },
  recommended_action: 'Add one rep call/month in NE',
  implementation_difficulty: 'low',
  time_to_impact: '4-6 weeks',
};

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
  (useOpportunities as ReturnType<typeof vi.fn>).mockReturnValue(idle);
});

describe('PriorityActionsROI (H3)', () => {
  it('shows an empty state (not SAMPLE_ACTIONS) when there are no opportunities', () => {
    (useOpportunities as ReturnType<typeof vi.fn>).mockReturnValue({
      ...idle,
      data: { total_count: 0, quick_wins_count: 0, strategic_bets_count: 0, opportunities: [], total_addressable_value: 0 },
    });
    render(<PriorityActionsROI />, { wrapper: createWrapper() });
    expect(screen.getByText(/No prioritized opportunities/i)).toBeInTheDocument();
    expect(screen.queryByText('Increase NE Region Call Frequency')).not.toBeInTheDocument();
  });

  it('renders a live opportunity row', () => {
    (useOpportunities as ReturnType<typeof vi.fn>).mockReturnValue({
      ...idle,
      data: {
        total_count: 1,
        quick_wins_count: 1,
        strategic_bets_count: 0,
        total_addressable_value: 2_400_000,
        opportunities: [
          {
            rank: 1,
            gap: { gap_id: 'g1', metric: 'trx', segment: 'region', segment_value: 'NE', current_value: 1, target_value: 2, gap_size: 1, gap_percentage: 50, gap_type: 'vs_target' },
            roi_estimate: { gap_id: 'g1', estimated_revenue_impact: 2_400_000, estimated_cost_to_close: 100000, expected_roi: 3.1, risk_adjusted_roi: 2.8, payback_period_months: 4, attribution_level: 'territory', attribution_rate: 0.6, confidence: 0.9 },
            recommended_action: 'Add one rep call/month in NE',
            implementation_difficulty: 'low',
            time_to_impact: '4-6 weeks',
          },
        ],
      },
    });
    render(<PriorityActionsROI brand="remibrutinib" />, { wrapper: createWrapper() });
    expect(screen.getByText('Add one rep call/month in NE')).toBeInTheDocument();
    // The live $2.4M ROI value renders both in the header total and the row.
    // (fixture deliberately uses the same value for total_addressable_value and
    // the single opportunity's estimated_revenue_impact).
    expect(screen.getAllByText('$2.4M').length).toBeGreaterThanOrEqual(1);
    expect(useOpportunities).toHaveBeenCalledWith(
      expect.objectContaining({ brand: 'remibrutinib' }),
    );
  });

  it('shows an error state when the opportunities feed fails (e.g. C1 shadow before shard 01)', () => {
    (useOpportunities as ReturnType<typeof vi.fn>).mockReturnValue({
      ...idle,
      isError: true,
      error: { message: 'Gap analysis opportunities not found' },
    });
    render(<PriorityActionsROI />, { wrapper: createWrapper() });
    expect(screen.getByText(/Could not load opportunities/i)).toBeInTheDocument();
  });

  it('opens the shared drill-down dialog when an opportunity card is clicked (T7b)', async () => {
    (useOpportunities as ReturnType<typeof vi.fn>).mockReturnValue({
      ...idle,
      data: {
        total_count: 1,
        quick_wins_count: 1,
        strategic_bets_count: 0,
        total_addressable_value: 2_400_000,
        opportunities: [SAMPLE_OPP],
      },
    });
    const user = userEvent.setup();
    render(<PriorityActionsROI />, { wrapper: createWrapper() });

    // The card is a button labeled by its recommended action — clicking it
    // opens the same "why" drawer the Gap-Analysis page uses.
    await user.click(
      screen.getByRole('button', { name: /Add one rep call\/month in NE/i })
    );

    expect(await screen.findByText('Why this rank')).toBeInTheDocument();
    expect(screen.getByText('Why this timeline')).toBeInTheDocument();
    expect(screen.getByText('ROI breakdown')).toBeInTheDocument();
  });

  it('navigates to /gap-analysis when "View All Recommendations" is clicked', async () => {
    (useOpportunities as ReturnType<typeof vi.fn>).mockReturnValue({
      ...idle,
      data: { total_count: 1, quick_wins_count: 1, strategic_bets_count: 0, total_addressable_value: 2_400_000, opportunities: [SAMPLE_OPP] },
    });
    const user = userEvent.setup();
    render(<PriorityActionsROI />, { wrapper: createWrapper() });

    const viewAll = screen.getByRole('button', { name: /view all recommendations/i });
    await user.click(viewAll);
    expect(mockNavigate).toHaveBeenCalledWith('/gap-analysis');
  });
});
