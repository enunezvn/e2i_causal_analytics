/**
 * SegmentAnalysis Page — Warning Banner Coverage
 * ==============================================
 *
 * Focused regression tests asserting that F-010-frontend wiring renders
 * API-reported `warnings[]` on the page.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import SegmentAnalysis from './SegmentAnalysis';

// Mock recharts to keep render light + avoid SVG/canvas churn.
vi.mock('recharts', async () => {
  const actual = await vi.importActual('recharts');
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container">{children}</div>
    ),
  };
});

// Mock API hooks so we can drive `warnings[]` directly. The page uses the
// polling mutation (useRunSegmentAnalysisAndWait) so async results resolve to
// COMPLETED before render; its mutation API surface (data/mutate/isPending/
// error) is identical to useRunSegmentAnalysis.
vi.mock('@/hooks/api', () => ({
  useSegmentHealth: vi.fn(),
  useRunSegmentAnalysisAndWait: vi.fn(),
  usePolicies: vi.fn(),
}));

vi.mock('@/hooks/use-query-error', () => ({
  useQueryErrorToast: vi.fn(),
  useMutationError: vi.fn(() => vi.fn()),
}));

import {
  useSegmentHealth,
  useRunSegmentAnalysisAndWait,
  usePolicies,
} from '@/hooks/api';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe('SegmentAnalysis — F-002 empty state + F-010 warnings', () => {
  it('renders empty state when no analysis result (F-002)', () => {
    (useSegmentHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { agent_available: true, econml_available: true, causalml_available: true, analyses_24h: 0 },
      isLoading: false,
      error: null,
      refetch: vi.fn(),
      isRefetching: false,
    });
    (usePolicies as ReturnType<typeof vi.fn>).mockReturnValue({ data: { policies: [] }, error: null });
    (useRunSegmentAnalysisAndWait as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });

    expect(
      screen.getByText(/No segment analysis available/),
    ).toBeInTheDocument();
    // Former sampleAnalysisResult values must not be in the DOM.
    expect(screen.queryByText('Cardiology')).not.toBeInTheDocument();
    expect(screen.queryByText('Northeast')).not.toBeInTheDocument();
    expect(screen.queryByText('0.28')).not.toBeInTheDocument();
  });

});

describe('SegmentAnalysis — warnings rendering (F-010-frontend)', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    (useSegmentHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { agent_available: true, econml_available: true, causalml_available: true, analyses_24h: 5 },
      isLoading: false,
      error: null,
      refetch: vi.fn(),
      isRefetching: false,
    });

    (usePolicies as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { policies: [] },
      error: null,
    });
  });

  it('does not render a WarningBanner when no API response has been received', () => {
    (useRunSegmentAnalysisAndWait as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });

    expect(screen.queryByTestId('warning-banner')).not.toBeInTheDocument();
  });

  it('renders WarningBanner with each warning string when API returns warnings[]', () => {
    (useRunSegmentAnalysisAndWait as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        analysis_id: 'a1',
        warnings: ['Using mock data', 'CATE bounds approximate'],
        cate_by_segment: {},
        overall_ate: 0.3,
        heterogeneity_score: 0.5,
        feature_importance: {},
        uplift_metrics: { overall_auuc: 0.5, overall_qini: 0.3, targeting_efficiency: 0.6, model_type_used: 'causal_forest' },
        high_responders: [],
        low_responders: [],
        policy_recommendations: [],
        expected_total_lift: 0,
        optimal_allocation_summary: '',
        executive_summary: '',
        key_insights: [],
        libraries_used: [],
        library_agreement_score: 0.8,
        validation_passed: true,
        estimation_latency_ms: 100,
        analysis_latency_ms: 100,
        total_latency_ms: 200,
        timestamp: new Date().toISOString(),
        confidence: 0.85,
      },
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });

    expect(screen.getByTestId('warning-banner')).toBeInTheDocument();
    expect(screen.getByText('Using mock data')).toBeInTheDocument();
    expect(screen.getByText('CATE bounds approximate')).toBeInTheDocument();
  });

  it('does not render WarningBanner when API returns empty warnings[]', () => {
    (useRunSegmentAnalysisAndWait as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        analysis_id: 'a2',
        warnings: [],
        cate_by_segment: {},
        overall_ate: 0.3,
        heterogeneity_score: 0.5,
        feature_importance: {},
        uplift_metrics: { overall_auuc: 0.5, overall_qini: 0.3, targeting_efficiency: 0.6, model_type_used: 'causal_forest' },
        high_responders: [],
        low_responders: [],
        policy_recommendations: [],
        expected_total_lift: 0,
        optimal_allocation_summary: '',
        executive_summary: '',
        key_insights: [],
        libraries_used: [],
        library_agreement_score: 0.8,
        validation_passed: true,
        estimation_latency_ms: 100,
        analysis_latency_ms: 100,
        total_latency_ms: 200,
        timestamp: new Date().toISOString(),
        confidence: 0.85,
      },
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });
    expect(screen.queryByTestId('warning-banner')).not.toBeInTheDocument();
  });
});
