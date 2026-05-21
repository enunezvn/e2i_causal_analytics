/**
 * CausalAnalysis Page — Empty State Coverage
 * ==========================================
 *
 * Regression tests for F-002: when API hooks return no data, the page
 * must render an explicit empty state, NOT a hardcoded SAMPLE_HIERARCHICAL_RESULT.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import CausalAnalysis from './CausalAnalysis';

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
  useCausalHealth: vi.fn(),
  useRunHierarchicalAnalysis: vi.fn(),
}));

import {
  useCausalHealth,
  useRunHierarchicalAnalysis,
} from '@/hooks/api';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe('CausalAnalysis — F-002 empty state', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (useCausalHealth as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined });
    (useRunHierarchicalAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      mutateAsync: vi.fn(),
      isPending: false,
    });
  });

  it('renders empty state on hierarchical tab when no analysis result', () => {
    render(<CausalAnalysis />, { wrapper: createWrapper() });

    // Page renders empty state for hierarchical results.
    expect(
      screen.getByText(/No hierarchical CATE analysis available/),
    ).toBeInTheDocument();
  }, 20000);

  it('does NOT render hardcoded SAMPLE_HIERARCHICAL_RESULT values (0.245 ATE)', () => {
    render(<CausalAnalysis />, { wrapper: createWrapper() });

    // The former SAMPLE_HIERARCHICAL_RESULT.overall_ate was 0.245; assert
    // the page no longer surfaces that fabricated value when no analysis
    // result exists.
    expect(screen.queryByText(/0\.245/)).not.toBeInTheDocument();
    // Fabricated segment heterogeneity I² = 42.5%
    expect(screen.queryByText(/42\.5%/)).not.toBeInTheDocument();
    // Fabricated segment name "High Uplift"
    expect(screen.queryByText('High Uplift')).not.toBeInTheDocument();
  }, 20000);
});
