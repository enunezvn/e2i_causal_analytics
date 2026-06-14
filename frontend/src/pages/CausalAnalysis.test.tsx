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
  // Added in PR #947 (51ab0de6) to wire the History tab to real episodic
  // data; the component calls this at render time, so the mock must export it
  // or every render throws "No useCausalAnalysisHistory export … on the mock".
  useCausalAnalysisHistory: vi.fn(),
}));

import {
  useCausalHealth,
  useRunHierarchicalAnalysis,
  useCausalAnalysisHistory,
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
    // No history loaded — matches the F-002 empty-state theme. The component
    // reads historyData?.total / historyData.items behind a truthiness guard,
    // so `data: undefined` renders the History tab's empty branch (no crash).
    (useCausalAnalysisHistory as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false,
    });
  });

  it('renders empty state on hierarchical tab when no analysis result', () => {
    render(<CausalAnalysis />, { wrapper: createWrapper() });

    // Page renders empty state for hierarchical results.
    expect(
      screen.getByText(/No hierarchical CATE analysis available/),
    ).toBeInTheDocument();
  }, 20000);

  it('does not claim a 95% confidence level the schema never reports', () => {
    // HierarchicalAnalysisResponse exposes raw overall_ci_lower/upper with NO
    // confidence-level field — the UI must render "CI:" without inventing 95%.
    (useRunHierarchicalAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        overall_ate: 0.18,
        overall_ci_lower: 0.12,
        overall_ci_upper: 0.24,
        status: 'completed',
        segment_results: [],
        segment_heterogeneity: null,
        nested_ci: null,
        n_segments_analyzed: 0,
        segmentation_method: 'tree',
        estimator_type: 'dml',
        latency_ms: 1200,
      },
      mutateAsync: vi.fn(),
      isPending: false,
    });
    render(<CausalAnalysis />, { wrapper: createWrapper() });

    expect(screen.queryByText(/95% CI/)).not.toBeInTheDocument();
    expect(screen.getByText(/^CI:/)).toBeInTheDocument();
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
