/**
 * GapAnalysis Page — Warning Banner Coverage
 * ==========================================
 *
 * Focused regression tests asserting that F-010-frontend wiring renders
 * API-reported `warnings[]` from the gap-analysis mutation.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
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
  useRunGapAnalysis: vi.fn(),
}));

import {
  useOpportunities,
  useGapHealth,
  useRunGapAnalysis,
} from '@/hooks/api';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe('GapAnalysis — warnings rendering (F-010-frontend)', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    (useOpportunities as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { opportunities: [], total_addressable_value: 0, quick_wins_count: 0, strategic_bets_count: 0 },
      isLoading: false,
      refetch: vi.fn().mockResolvedValue({}),
    });

    (useGapHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { agent_available: true, analyses_24h: 3 },
      isLoading: false,
    });
  });

  it('does not render WarningBanner before mutation runs', () => {
    (useRunGapAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<GapAnalysis />, { wrapper: createWrapper() });
    expect(screen.queryByTestId('warning-banner')).not.toBeInTheDocument();
  });

  it('renders WarningBanner when API response includes warnings[]', () => {
    (useRunGapAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        analysis_id: 'gap_001',
        warnings: ['ROI estimates degraded due to missing brand data'],
        opportunities: [],
        total_addressable_value: 0,
        quick_wins_count: 0,
        strategic_bets_count: 0,
      },
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<GapAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByTestId('warning-banner')).toBeInTheDocument();
    expect(
      screen.getByText('ROI estimates degraded due to missing brand data'),
    ).toBeInTheDocument();
  });
});
