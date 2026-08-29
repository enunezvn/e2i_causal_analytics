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
 *  - Library Validation honest-null (no fabricated "0% / Failed" from nulls)
 *  - Policy Details: 2-decimal lift with units, zero-lift maintain cards hidden
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
  useCausalVariables: vi.fn(),
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
  useCausalVariables,
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
  // Biomarker union for the feature-importance display grouping (mirrors the
  // backend's clinical_biomarkers field on GET /causal/variables).
  mockHook(useCausalVariables).mockReturnValue({
    data: {
      dataset: 'patient_journeys',
      treatment_candidates: [],
      outcome_candidates: [],
      covariate_candidates: [],
      columns: [],
      clinical_biomarkers: [
        'ecog_performance_status',
        'egfr',
        'ldh_ratio',
        'proteinuria_g_day',
        'urticaria_severity_uas7',
      ],
    },
  });
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

describe('SegmentAnalysis — feature-importance biomarker grouping', () => {
  it('labels indication-specific biomarkers apart from generic confounders', () => {
    primeBaseHooks();
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: completedResponse({
        feature_importance: {
          disease_severity: 0.5,
          urticaria_severity_uas7: 0.3,
          age_at_diagnosis: 0.2,
        },
      }),
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });

    // The card explains the split, and the legend appears because a biomarker
    // (UAS7) is present among the modifiers.
    expect(screen.getByText('Feature Importance for CATE')).toBeInTheDocument();
    expect(screen.getByText(/indication-specific biomarkers/i)).toBeInTheDocument();
    expect(screen.getByText(/Indication-specific biomarker/)).toBeInTheDocument();
    expect(screen.getByText(/Generic confounder \(all brands\)/)).toBeInTheDocument();
  });

  it('renders no grouping legend when all modifiers are generic (all-brands run)', () => {
    primeBaseHooks();
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: completedResponse({
        feature_importance: { disease_severity: 0.6, age_at_diagnosis: 0.4 },
      }),
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });

    expect(screen.getByText('Feature Importance for CATE')).toBeInTheDocument();
    expect(screen.queryByText(/Indication-specific biomarker$/)).not.toBeInTheDocument();
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

describe('SegmentAnalysis — Library Validation honest-null', () => {
  it('renders "Not computed" / "Not run" when validation fields are null (no fabricated 0%/Failed)', async () => {
    const user = userEvent.setup();
    primeBaseHooks();
    // Legitimately-not-computed run: e.g. uplift degraded or <3 paired segments.
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: completedResponse({
        libraries_used: [],
        library_agreement_score: undefined,
        validation_passed: undefined,
      }),
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Uplift Metrics/i }));

    await waitFor(() => {
      expect(screen.getByText('Not computed')).toBeInTheDocument();
    });
    expect(screen.getByText('Not run')).toBeInTheDocument();
    expect(screen.getByText('Not reported')).toBeInTheDocument();
    // The old null-coerced fabrications must be gone.
    expect(screen.queryByText('0%')).not.toBeInTheDocument();
    expect(screen.queryByText('Failed')).not.toBeInTheDocument();
  });

  it('renders the real score and verdict when validation was computed', async () => {
    const user = userEvent.setup();
    primeBaseHooks();
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: completedResponse({
        libraries_used: ['econml', 'causalml'],
        library_agreement_score: 0.756,
        validation_passed: true,
      }),
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Uplift Metrics/i }));

    await waitFor(() => {
      expect(screen.getByText('76%')).toBeInTheDocument();
    });
    expect(screen.getByText('Passed')).toBeInTheDocument();
    expect(screen.getByText('econml')).toBeInTheDocument();
    expect(screen.getByText('causalml')).toBeInTheDocument();
  });
});

describe('SegmentAnalysis — Policy Details formatting + zero-lift filtering', () => {
  const policy = (
    segment: string,
    lift: number,
    recommendedRate = 0.7,
  ) => ({
    segment,
    current_treatment_rate: 0.5,
    recommended_treatment_rate: recommendedRate,
    expected_incremental_outcome: lift,
    confidence: 0.9,
  });

  it('rounds lift to 2 decimals with units and hides zero-lift maintain segments', async () => {
    const user = userEvent.setup();
    primeBaseHooks();
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: completedResponse({
        policy_recommendations: [
          // Real values from run seg_8b4bc09251c7 — the raw float the card used
          // to print verbatim.
          policy('ecog_performance_status=1.0', 204.60681722776272),
          policy('ecog_performance_status=0.0', 0, 0.5),
          policy('age_band=65+', 0, 0.5),
        ],
      }),
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Policies/i }));

    await waitFor(() => {
      expect(screen.getByText('+204.61 outcomes')).toBeInTheDocument();
    });
    // Raw unrounded float must not appear anywhere.
    expect(screen.queryByText(/204\.60681722776272/)).not.toBeInTheDocument();
    // Zero-lift maintain segments render no card...
    expect(screen.queryByText('age_band=65+')).not.toBeInTheDocument();
    // ...but are honestly accounted for in the footnote.
    expect(screen.getByText(/2 other segments showed no statistically significant/i)).toBeInTheDocument();
  });

  it('shows an honest empty note when every segment is maintained (all zero-lift)', async () => {
    const user = userEvent.setup();
    primeBaseHooks();
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: completedResponse({
        policy_recommendations: [
          policy('ecog_performance_status=0.0', 0, 0.5),
          policy('age_band=65+', 0, 0.5),
        ],
      }),
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Policies/i }));

    await waitFor(() => {
      expect(
        screen.getByText(/No segment shows a treatment effect significantly above average/i),
      ).toBeInTheDocument();
    });
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
  // ---------------------------------------------------------------------------
  // Brand-scoped, SSOT-derived options (2026-08-29 /segment-analysis review).
  // The flat allowlists offered Fabhalta's complement_inhibitor_status on a
  // Remibrutinib cohort (503 "No usable rows"), listed treatment_initiated in
  // BOTH dropdowns, and let a user pose a pair with no modeled causal edge
  // (treatment_initiated -> persistent_180d) whose cross-library check then
  // honestly FAILED. The page must ask the backend for options scoped to the
  // selected brand and scope the Outcome dropdown to the selected treatment.
  // ---------------------------------------------------------------------------

  const scopedDatasets = {
    treatments: ['treatment_arm', 'rep_detailing_high', 'urticaria_severity_uas7'],
    outcomes: ['persistent_180d', 'discontinued_180d', 'treatment_initiated'],
    outcomes_by_treatment: {
      treatment_arm: ['persistent_180d', 'discontinued_180d', 'treatment_initiated'],
      rep_detailing_high: ['treatment_initiated'],
      urticaria_severity_uas7: ['persistent_180d'],
    },
    brands: ['Remibrutinib', 'Kisqali'],
    brand: null,
    options_source: 'causal_paths',
    labels: {
      treatment_arm: 'Treatment arm',
      rep_detailing_high: 'High rep detailing',
      urticaria_severity_uas7: 'Uncontrolled CSU (UAS7 ≥ 28)',
      persistent_180d: 'Persistent at 180d',
      discontinued_180d: 'Discontinued at 180d',
      treatment_initiated: 'Treatment initiated',
    },
  };

  it('requests options scoped to the selected brand (re-queries /segments/datasets with brand)', async () => {
    const user = userEvent.setup();
    primeBaseHooks();
    mockHook(useSegmentDatasets).mockReturnValue({ data: scopedDatasets, isLoading: false, error: null });
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: undefined,
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });

    // Default cohort: all brands -> no brand scope.
    expect(mockHook(useSegmentDatasets)).toHaveBeenLastCalledWith({ brand: undefined });

    await user.click(screen.getByRole('combobox', { name: 'Brand' }));
    await user.click(await screen.findByRole('option', { name: 'Remibrutinib' }));

    await waitFor(() =>
      expect(mockHook(useSegmentDatasets)).toHaveBeenLastCalledWith({ brand: 'Remibrutinib' })
    );
  });

  it('scopes the Outcome dropdown to the selected treatment and never lists the treatment itself', async () => {
    const user = userEvent.setup();
    primeBaseHooks();
    mockHook(useSegmentDatasets).mockReturnValue({ data: scopedDatasets, isLoading: false, error: null });
    const mutate = vi.fn();
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: undefined,
      mutate,
      isPending: false,
      error: null,
    });

    render(<SegmentAnalysis />, { wrapper: createWrapper() });

    // Default treatment_arm -> its three modeled outcomes.
    await user.click(screen.getByRole('combobox', { name: 'Outcome variable' }));
    expect(await screen.findByRole('option', { name: 'Persistent at 180d' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Treatment initiated' })).toBeInTheDocument();
    await user.keyboard('{Escape}');

    // Switch to a treatment whose only modeled outcome is treatment_initiated.
    await user.click(screen.getByRole('combobox', { name: 'Treatment variable' }));
    await user.click(await screen.findByRole('option', { name: 'High rep detailing' }));

    await user.click(screen.getByRole('combobox', { name: 'Outcome variable' }));
    expect(await screen.findByRole('option', { name: 'Treatment initiated' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Persistent at 180d' })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'High rep detailing' })).not.toBeInTheDocument();
    await user.keyboard('{Escape}');

    // The previously selected outcome (persistent_180d) is no longer valid for
    // this treatment, so the run must send the scoped outcome, not a stale one.
    fireEvent.click(screen.getByRole('button', { name: /Run Analysis/i }));
    expect(mutate).toHaveBeenCalledTimes(1);
    const arg = mutate.mock.calls[0][0] as {
      request: { treatment_var?: string; outcome_var?: string };
    };
    expect(arg.request.treatment_var).toBe('rep_detailing_high');
    expect(arg.request.outcome_var).toBe('treatment_initiated');
  });

  it('resets a selected treatment that the newly scoped options no longer offer', async () => {
    const user = userEvent.setup();
    primeBaseHooks();
    // All-brands options include Fabhalta's axis for this test's purposes.
    const allBrands = {
      ...scopedDatasets,
      treatments: ['treatment_arm', 'complement_inhibitor_status'],
      outcomes_by_treatment: {
        treatment_arm: ['persistent_180d'],
        complement_inhibitor_status: ['persistent_180d'],
      },
      labels: { ...scopedDatasets.labels, complement_inhibitor_status: 'Prior C5-inhibitor (switch)' },
    };
    mockHook(useSegmentDatasets).mockReturnValue({ data: allBrands, isLoading: false, error: null });
    const mutate = vi.fn();
    mockHook(useRunSegmentAnalysisAndWait).mockReturnValue({
      data: undefined,
      mutate,
      isPending: false,
      error: null,
    });

    const { rerender } = render(<SegmentAnalysis />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('combobox', { name: 'Treatment variable' }));
    await user.click(await screen.findByRole('option', { name: 'Prior C5-inhibitor (switch)' }));
    expect(screen.getByRole('combobox', { name: 'Treatment variable' })).toHaveTextContent(
      'Prior C5-inhibitor (switch)'
    );

    // The backend now returns Remibrutinib-scoped options (no Fabhalta axis).
    mockHook(useSegmentDatasets).mockReturnValue({
      data: { ...scopedDatasets, brand: 'Remibrutinib' },
      isLoading: false,
      error: null,
    });
    rerender(<SegmentAnalysis />);

    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Treatment variable' })).toHaveTextContent(
        'Treatment arm'
      )
    );
    fireEvent.click(screen.getByRole('button', { name: /Run Analysis/i }));
    const arg = mutate.mock.calls[0][0] as { request: { treatment_var?: string } };
    expect(arg.request.treatment_var).toBe('treatment_arm');
  });

  it('tells the user when options fell back to the flat curated lists (not brand/pair-scoped)', () => {
    primeBaseHooks();
    mockHook(useSegmentDatasets).mockReturnValue({
      data: { ...scopedDatasets, outcomes_by_treatment: {}, options_source: 'curated_fallback' },
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

    expect(screen.getByTestId('datasets-fallback-notice')).toBeInTheDocument();
  });
});
