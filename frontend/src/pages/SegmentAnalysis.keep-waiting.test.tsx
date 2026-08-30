/**
 * SegmentAnalysis — poll-ceiling expiry offers "Keep waiting" (#1841)
 * ==================================================================
 *
 * The analysis is a durable server-side record that is usually still running
 * (or already finished) when the page's poll ceiling expires. The page must
 * keep its `analysis_id`, show a non-destructive "Still running" state with a
 * Keep waiting action that re-attaches with GET polling on the SAME id (never
 * a second POST), and render a completion reached that way exactly like a
 * normal completion. A record that turns out `failed` during the resume
 * surfaces as the ordinary failure.
 *
 * Faithful seams: the REAL `useRunSegmentAnalysisAndWait` mutation (real
 * react-query state machine under a `retry: 1` client default — the app
 * default until #1846, kept here so the hook's own `retry: false` is what is
 * exercised) drives the page;
 * only the API module's POST+poll / poll-only functions are mocked, and the
 * poll loop itself is covered by src/api/segments.wait.test.ts.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import SegmentAnalysis from './SegmentAnalysis';
import type { SegmentAnalysisResponse } from '@/types/segments';
import { SegmentAnalysisStatus } from '@/types/segments';

vi.mock('recharts', async () => {
  const actual = await vi.importActual('recharts');
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container">{children}</div>
    ),
  };
});

// Keep the real segment mutation hook; stub only the data hooks the page reads.
vi.mock('@/hooks/api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/hooks/api')>()),
  useSegmentHealth: vi.fn(),
  useSegmentDatasets: vi.fn(),
  usePolicies: vi.fn(),
  useCausalVariables: vi.fn(),
}));

// The hook's two API seams: POST+poll and poll-only re-attach.
vi.mock('@/api/segments', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/api/segments')>()),
  runSegmentAnalysisAndWait: vi.fn(),
  waitForSegmentAnalysis: vi.fn(),
}));

vi.mock('@/hooks/use-toast', () => ({
  toast: vi.fn(),
  useToast: vi.fn(() => ({ toasts: [], toast: vi.fn(), dismiss: vi.fn() })),
}));

import {
  useSegmentHealth,
  useSegmentDatasets,
  usePolicies,
  useCausalVariables,
} from '@/hooks/api';
import {
  runSegmentAnalysisAndWait,
  waitForSegmentAnalysis,
  SegmentAnalysisTimeoutError,
} from '@/api/segments';
import { toast } from '@/hooks/use-toast';

const ANALYSIS_ID = 'seg_1841_durable';
const SINGLE_BRAND_CEILING_MS = 300_000;

const mockFn = (fn: unknown) => fn as ReturnType<typeof vi.fn>;

/** Mirror the pre-#1846 PRODUCTION mutation default (retry once; the app default is now `retry: 0`) so the hook's `retry: false` is exercised. */
function createAppLikeWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0 },
      mutations: { retry: 1, retryDelay: 0 },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

function primeDataHooks() {
  mockFn(useSegmentHealth).mockReturnValue({
    data: { agent_available: true, econml_available: true, causalml_available: true, analyses_24h: 5 },
    isLoading: false,
    error: null,
    refetch: vi.fn(),
    isRefetching: false,
  });
  mockFn(useSegmentDatasets).mockReturnValue({
    data: {
      treatments: ['treatment_arm'],
      outcomes: ['persistent_180d'],
      brands: ['Fabhalta'],
    },
    isLoading: false,
    error: null,
  });
  mockFn(usePolicies).mockReturnValue({ data: { recommendations: [] }, error: null });
  mockFn(useCausalVariables).mockReturnValue({
    data: {
      dataset: 'patient_journeys',
      treatment_candidates: [],
      outcome_candidates: [],
      covariate_candidates: [],
      columns: [],
      clinical_biomarkers: [],
    },
  });
}

function completedResponse(): SegmentAnalysisResponse {
  return {
    analysis_id: ANALYSIS_ID,
    status: SegmentAnalysisStatus.COMPLETED,
    cate_by_segment: {},
    overall_ate: 0.084,
    confidence_level: 0.95,
    heterogeneity_score: 0.42,
    feature_importance: {},
    uplift_metrics: undefined,
    high_responders: [],
    mid_responders: [],
    low_responders: [],
    policy_recommendations: [],
    expected_total_lift: 0,
    optimal_allocation_summary: '',
    executive_summary: 'Copay support lifts persistence most in the severe band.',
    strategic_interpretation: undefined,
    key_insights: [],
    libraries_used: ['econml'],
    library_agreement_score: 0.8,
    validation_passed: true,
    estimation_latency_ms: 100,
    analysis_latency_ms: 100,
    total_latency_ms: 200,
    timestamp: '2026-08-30T13:12:16Z',
    warnings: [],
    confidence: 0.85,
  };
}

function runAndExpire() {
  mockFn(runSegmentAnalysisAndWait).mockRejectedValue(
    new SegmentAnalysisTimeoutError(ANALYSIS_ID, SINGLE_BRAND_CEILING_MS)
  );
  fireEvent.click(screen.getByRole('button', { name: /Run Analysis/i }));
}

beforeEach(() => {
  vi.clearAllMocks();
  primeDataHooks();
});

describe('SegmentAnalysis — poll-ceiling expiry keeps the analysis_id (#1841)', () => {
  it('shows a non-destructive Still running state with Keep waiting, one POST, no error toast', async () => {
    render(<SegmentAnalysis />, { wrapper: createAppLikeWrapper() });

    runAndExpire();

    expect(await screen.findByText(/still running/i)).toBeInTheDocument();
    expect(screen.getByText(ANALYSIS_ID)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /keep waiting/i })).toBeInTheDocument();

    // Not a failure: no destructive error card, no destructive toast.
    expect(screen.queryByText('Analysis failed')).not.toBeInTheDocument();
    expect(screen.queryByText(/timed out/i)).not.toBeInTheDocument();
    expect(toast).not.toHaveBeenCalled();

    // Exactly one POST+poll; nothing re-attached yet.
    expect(runSegmentAnalysisAndWait).toHaveBeenCalledTimes(1);
    expect(waitForSegmentAnalysis).not.toHaveBeenCalled();
  });

  it('Keep waiting resumes GET polling on the same id and renders the completion like a normal one', async () => {
    // Reference: a normal completion of the same record.
    mockFn(runSegmentAnalysisAndWait).mockResolvedValue(completedResponse());
    const normal = render(<SegmentAnalysis />, { wrapper: createAppLikeWrapper() });
    fireEvent.click(screen.getByRole('button', { name: /Run Analysis/i }));
    await screen.findByText('0.084'); // Overall ATE KPI on the default tab
    const normalText = normal.container.textContent;
    normal.unmount();
    vi.clearAllMocks();
    primeDataHooks();

    // Expiry → Keep waiting → completed via the resume path.
    const resumed = render(<SegmentAnalysis />, { wrapper: createAppLikeWrapper() });
    runAndExpire();
    const keepWaiting = await screen.findByRole('button', { name: /keep waiting/i });

    mockFn(waitForSegmentAnalysis).mockResolvedValue(completedResponse());
    fireEvent.click(keepWaiting);

    await waitFor(() =>
      expect(waitForSegmentAnalysis).toHaveBeenCalledWith(
        ANALYSIS_ID,
        undefined,
        SINGLE_BRAND_CEILING_MS
      )
    );
    expect(await screen.findByText('0.084')).toBeInTheDocument();
    expect(screen.queryByText(/still running/i)).not.toBeInTheDocument();
    expect(screen.queryByText('No segment analysis available')).not.toBeInTheDocument();

    // No second POST across expire → resume, and the rendered page is the
    // same as a normal completion of the same record.
    expect(runSegmentAnalysisAndWait).toHaveBeenCalledTimes(1);
    expect(waitForSegmentAnalysis).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: /Run Analysis/i })).toBeEnabled();
    expect(resumed.container.textContent).toBe(normalText);
  });

  it('a record that fails during Keep waiting surfaces as the ordinary failure', async () => {
    render(<SegmentAnalysis />, { wrapper: createAppLikeWrapper() });
    runAndExpire();
    const keepWaiting = await screen.findByRole('button', { name: /keep waiting/i });

    mockFn(waitForSegmentAnalysis).mockRejectedValue(
      new Error('Segment analysis failed: estimator blew up')
    );
    fireEvent.click(keepWaiting);

    expect(await screen.findByText('Analysis failed')).toBeInTheDocument();
    expect(screen.getByText(/estimator blew up/)).toBeInTheDocument();
    expect(screen.queryByText(/still running/i)).not.toBeInTheDocument();
    expect(toast).toHaveBeenCalledWith(expect.objectContaining({ variant: 'destructive' }));
    expect(runSegmentAnalysisAndWait).toHaveBeenCalledTimes(1);
    expect(waitForSegmentAnalysis).toHaveBeenCalledTimes(1);
  });
});
