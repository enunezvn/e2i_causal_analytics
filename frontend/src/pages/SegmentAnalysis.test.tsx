/**
 * SegmentAnalysis Page — Clinical HTE rebuild coverage
 * ====================================================
 *
 * Covers:
 *  - F-002 empty state (no fabricated fallback) + F-010 warnings rendering
 *  - Mid responders column (renders when present; honest empty-state when not)
 *  - Responder drill-down dialog (opens + shows defining features)
 *  - Strategic Interpretation rendering in the Insights tab
 *  - Agent-driven config sends { query, brand, treatment_var, outcome_var }
 *    (NOT the old segment_vars/effect_modifiers/data_source hardcode)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import SegmentAnalysis from './SegmentAnalysis';
import type { SegmentAnalysisResponse, SegmentProfile } from '@/types/segments';
import { ResponderType, SegmentAnalysisStatus } from '@/types/segments';

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

// Mock API hooks so we can drive responses directly. The page uses the polling
// mutation (useRunSegmentAnalysisAndWait) so async results resolve to COMPLETED
// before render; its mutation API surface (data/mutate/isPending/error) is
// identical to useRunSegmentAnalysis. useSegmentDatasets drives the dropdowns.
vi.mock('@/hooks/api', () => ({
  useSegmentHealth: vi.fn(),
  useSegmentDatasets: vi.fn(),
  useRunSegmentAnalysisAndWait: vi.fn(),
  usePolicies: vi.fn(),
}));

vi.mock('@/hooks/use-query-error', () => ({
  useQueryErrorToast: vi.fn(),
  useMutationError: vi.fn(() => vi.fn()),
}));

import {
  useSegmentHealth,
  useSegmentDatasets,
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

const mockHook = (hook: unknown) => hook as ReturnType<typeof vi.fn>;

/** Minimal healthy + datasets defaults shared by most tests. */
function primeBaseHooks() {
  mockHook(useSegmentHealth).mockReturnValue({
    data: { agent_available: true, econml_available: true, causalml_available: true, analyses_24h: 5 },
    isLoading: false,
    error: null,
    refetch: vi.fn(),
    isRefetching: false,
  });
  mockHook(useSegmentDatasets).mockReturnValue({
    data: {
      treatments: ['treatment_arm', 'treatment_initiated'],
      outcomes: ['persistent_180d', 'discontinued_180d'],
      brands: ['Kisqali', 'Cosentyx'],
    },
    isLoading: false,
    error: null,
  });
  mockHook(usePolicies).mockReturnValue({ data: { recommendations: [] }, error: null });
}

const profile = (over: Partial<SegmentProfile> = {}): SegmentProfile => ({
  segment_id: 'seg-1',
  responder_type: ResponderType.AVERAGE,
  cate_estimate: 0.12,
  defining_features: [{ disease_severity_band: 'med' }],
  size: 100,
  size_percentage: 25,
  recommendation: 'Maintain standard cadence.',
  ...over,
});

/** A COMPLETED response with only the fields a test cares about populated. */
function completedResponse(over: Partial<SegmentAnalysisResponse> = {}): SegmentAnalysisResponse {
  return {
    analysis_id: 'a1',
    status: SegmentAnalysisStatus.COMPLETED,
    cate_by_segment: {},
    overall_ate: 0.3,
    confidence_level: 0.95,
    heterogeneity_score: 0.5,
    feature_importance: {},
    uplift_metrics: undefined,
    high_responders: [],
    mid_responders: [],
    low_responders: [],
    policy_recommendations: [],
    expected_total_lift: 0,
    optimal_allocation_summary: '',
    executive_summary: '',
    strategic_interpretation: undefined,
    key_insights: [],
    libraries_used: [],
    library_agreement_score: 0.8,
    validation_passed: true,
    estimation_latency_ms: 100,
    analysis_latency_ms: 100,
    total_latency_ms: 200,
    timestamp: new Date().toISOString(),
    warnings: [],
    confidence: 0.85,
    ...over,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe('SegmentAnalysis — F-002 empty state + F-010 warnings', () => {
  it('renders empty state when no analysis result (F-002)', () => {
    primeBaseHooks();
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: undefined,
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });

    expect(screen.getByText(/No segment analysis available/)).toBeInTheDocument();
    // Former sampleAnalysisResult values must not be in the DOM.
    expect(screen.queryByText('Cardiology')).not.toBeInTheDocument();
    expect(screen.queryByText('Northeast')).not.toBeInTheDocument();
    expect(screen.queryByText('0.28')).not.toBeInTheDocument();
  });

  it('does not render a WarningBanner when no API response has been received', () => {
    primeBaseHooks();
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: undefined,
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });

    expect(screen.queryByTestId('warning-banner')).not.toBeInTheDocument();
  });

  it('renders WarningBanner with each warning string when API returns warnings[]', () => {
    primeBaseHooks();
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: completedResponse({ warnings: ['Using mock data', 'CATE bounds approximate'] }),
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
    primeBaseHooks();
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: completedResponse({ warnings: [] }),
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });
    expect(screen.queryByTestId('warning-banner')).not.toBeInTheDocument();
  });
});

describe('SegmentAnalysis — Mid responders column', () => {
  it('renders the Mid responder column with a card when mid_responders are present', async () => {
    const user = userEvent.setup();
    primeBaseHooks();
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: completedResponse({
        mid_responders: [
          profile({
            segment_id: 'mid-seg-A',
            responder_type: ResponderType.AVERAGE,
            cate_estimate: 0.15,
            defining_features: [{ disease_severity_band: 'med' }],
          }),
        ],
      }),
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });

    // Switch to the Responders tab.
    await user.click(screen.getByRole('tab', { name: /Responders/i }));

    await waitFor(() => {
      // Exact match disambiguates the mid column title from the "Above-Average
      // Responders" KPI/high-column titles that also contain "Average Responders".
      expect(screen.getByText('Average Responders')).toBeInTheDocument();
    });
    // The mid card is present (its segment id is rendered in the card label).
    expect(screen.getByLabelText(/View details for mid-seg-A/i)).toBeInTheDocument();
    // The "Average Responder" badge is rendered on the card.
    expect(screen.getByText('Average Responder')).toBeInTheDocument();
  });

  it('renders an honest empty-state for the Mid column when no mid_responders', async () => {
    const user = userEvent.setup();
    primeBaseHooks();
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: completedResponse({ mid_responders: [] }),
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Responders/i }));

    await waitFor(() => {
      expect(screen.getByText(/No average-band segments for this run/i)).toBeInTheDocument();
    });
  });
});

describe('SegmentAnalysis — responder drill-down', () => {
  it('opens a drill-down dialog showing defining features when a card is clicked', async () => {
    const user = userEvent.setup();
    primeBaseHooks();
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: completedResponse({
        high_responders: [
          profile({
            segment_id: 'high-seg-X',
            responder_type: ResponderType.HIGH,
            cate_estimate: 0.42,
            defining_features: [{ disease_severity_band: 'high' }, { ecog_performance_status: '2' }],
            recommendation: 'Prioritize outreach to this segment.',
          }),
        ],
      }),
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Responders/i }));

    // No dialog before click.
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();

    await user.click(await screen.findByLabelText(/View details for high-seg-X/i));

    const dialog = await screen.findByRole('dialog');
    expect(dialog).toBeInTheDocument();
    // Defining features rendered as key:value badges inside the dialog.
    expect(within(dialog).getByText('disease_severity_band: high')).toBeInTheDocument();
    expect(within(dialog).getByText('ecog_performance_status: 2')).toBeInTheDocument();
    // Recommendation text is shown.
    expect(within(dialog).getByText('Prioritize outreach to this segment.')).toBeInTheDocument();
  });
});

describe('SegmentAnalysis — strategic interpretation', () => {
  it('renders the Strategic Interpretation card in the Insights tab', async () => {
    const user = userEvent.setup();
    primeBaseHooks();
    const narrative = 'Tier 1: high-severity patients respond strongly.\nTier 2: moderate response in mid severity.';
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: completedResponse({
        strategic_interpretation: narrative,
        segment_heterogeneity: 62.5,
        n_segments_analyzed: 3,
        segmentation_method_used: 'quantile',
      }),
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Insights/i }));

    await waitFor(() => {
      expect(screen.getByText('Strategic Interpretation')).toBeInTheDocument();
    });
    // Multi-paragraph narrative preserved (matched by a substring).
    expect(screen.getByText(/high-severity patients respond strongly/i)).toBeInTheDocument();
    // Heterogeneity panel surfaces the I^2 label + method.
    expect(screen.getByText(/Between-segment heterogeneity \(I²\)/i)).toBeInTheDocument();
    expect(screen.getByText('62.5%')).toBeInTheDocument();
    expect(screen.getByText('Quantile')).toBeInTheDocument();
  });
});

describe('SegmentAnalysis — agent-driven config request', () => {
  it('sends only { query, brand, treatment_var, outcome_var } (not the old hardcode)', () => {
    primeBaseHooks();
    const mutate = vi.fn();
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: undefined,
      mutate,
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });

    // Run with the default cohort (All brands) + default curated pair.
    fireEvent.click(screen.getByRole('button', { name: /Run Analysis/i }));

    expect(mutate).toHaveBeenCalledTimes(1);
    const arg = mutate.mock.calls[0][0] as { request: Record<string, unknown> };
    const req = arg.request;

    // Defaults: All brands -> brand undefined; curated defaults sent.
    expect(req.brand).toBeUndefined();
    expect(req.treatment_var).toBe('treatment_arm');
    expect(req.outcome_var).toBe('persistent_180d');
    expect(typeof req.query).toBe('string');

    // The old hardcoded substrate fields must NOT be sent — the backend fixes
    // them server-side now.
    expect(req).not.toHaveProperty('segment_vars');
    expect(req).not.toHaveProperty('effect_modifiers');
    expect(req).not.toHaveProperty('confounders');
    expect(req).not.toHaveProperty('data_source');
    expect(req).not.toHaveProperty('filters');
  });
});

describe('SegmentAnalysis — T2: robust options + durable-run timeout', () => {
  it('runs with a poll ceiling above the old 120s default (durable record; no premature timeout)', () => {
    primeBaseHooks();
    const mutate = vi.fn();
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: undefined,
      mutate,
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByRole('button', { name: /Run Analysis/i }));

    expect(mutate).toHaveBeenCalledTimes(1);
    const arg = mutate.mock.calls[0][0] as { maxWaitMs?: number };
    // All-brands runs scan the full cohort (~90s+); the old 120s FE cap raced a
    // still-running, server-side-durable analysis and threw "timed out" on a run
    // that actually completed. The page must give the poll a generous ceiling.
    expect(arg.maxWaitMs).toBeGreaterThan(120000);
  });

  it('surfaces a degraded-options notice when GET /segments/datasets fails (no silent single-defaults)', () => {
    primeBaseHooks();
    // /datasets errors -> the page must NOT silently masquerade the single curated
    // defaults as the full option set; it must tell the user options are incomplete.
    mockHook(useSegmentDatasets).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error('segment service unavailable'),
    });
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: undefined,
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });

    expect(screen.getByTestId('datasets-degraded-notice')).toBeInTheDocument();
  });

  it('renders backend display labels in the Outcome dropdown (not raw title-case)', async () => {
    // REGRESSION: the dropdowns title-cased the raw column name, so the curated
    // labels GET /segments/datasets returns (e.g. low_gap_180d -> "Low refill gap
    // (≤30d)") never reached the user. They must render the label.
    const user = userEvent.setup();
    primeBaseHooks();
    mockHook(useSegmentDatasets).mockReturnValue({
      data: {
        treatments: ['treatment_arm'],
        outcomes: ['adherent_180d', 'low_gap_180d'],
        brands: ['Kisqali'],
        labels: {
          treatment_arm: 'Treatment arm',
          adherent_180d: 'Adherent at 180d',
          low_gap_180d: 'Low refill gap (≤30d)',
        },
      },
      isLoading: false,
      error: null,
    });
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: undefined,
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });

    // Outcome dropdown renders curated labels.
    await user.click(screen.getByRole('combobox', { name: 'Outcome variable' }));
    expect(
      await screen.findByRole('option', { name: 'Low refill gap (≤30d)' })
    ).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Adherent at 180d' })).toBeInTheDocument();
    // The raw title-cased form must NOT be what's shown.
    expect(screen.queryByRole('option', { name: 'Low Gap 180d' })).not.toBeInTheDocument();

    // Close, then assert the Treatment dropdown uses the SAME labelFor path.
    await user.keyboard('{Escape}');
    await user.click(screen.getByRole('combobox', { name: 'Treatment variable' }));
    expect(await screen.findByRole('option', { name: 'Treatment arm' })).toBeInTheDocument();
  });
});
