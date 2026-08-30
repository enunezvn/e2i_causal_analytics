/**
 * HeterogeneousTreatmentEffects Tests — real segment CATE wiring.
 *
 * The card runs POST /api/segments/analyze (Heterogeneous Optimizer / EconML
 * CausalForestDML over the patient_journeys clinical substrate — the same
 * contract as /segment-analysis) and renders real per-segment CATE from
 * `cate_by_segment`. Honest states only: empty -> run action, completed ->
 * real numbers, failed -> labeled error. No fabricated segments, no invented
 * p-values. The segment set, effect modifiers, confounders, and data source
 * are FIXED server-side; the card must send only the caller-selectable fields.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HeterogeneousTreatmentEffects } from './HeterogeneousTreatmentEffects';
import * as useSegments from '@/hooks/api/use-segments';
import * as useInsights from '@/hooks/api/use-insights';
import { SegmentAnalysisTimeoutError } from '@/api/segments';

vi.mock('@/hooks/api/use-segments');
vi.mock('@/hooks/api/use-insights');

type Mutation = ReturnType<typeof useSegments.useRunSegmentAnalysisAndWait>;
type InsightMutation = ReturnType<typeof useInsights.useHTEInsight>;

function mockSegments(overrides: Partial<Mutation> = {}) {
  vi.mocked(useSegments.useRunSegmentAnalysisAndWait).mockReturnValue({
    mutate: vi.fn(),
    data: undefined,
    error: null,
    isPending: false,
    ...overrides,
  } as unknown as Mutation);
}

function mockInsight(overrides: Partial<InsightMutation> = {}) {
  const mutate = vi.fn();
  vi.mocked(useInsights.useHTEInsight).mockReturnValue({
    mutate,
    data: undefined,
    error: null,
    isPending: false,
    variables: undefined,
    ...overrides,
  } as unknown as InsightMutation);
  return mutate;
}

// Faithful to a real /api/segments/analyze completed response on the clinical
// patient_journeys contract: pp-scale binary-outcome effects (persistent_180d)
// broken down by a server-fixed clinical dimension.
const COMPLETED = {
  analysis_id: 'seg_test_1',
  status: 'completed',
  cate_by_segment: {
    disease_severity: [
      {
        segment_name: 'disease_severity',
        segment_value: 'severe',
        cate_estimate: 0.017,
        cate_ci_lower: 0.009,
        cate_ci_upper: 0.025,
        sample_size: 4023,
        statistical_significance: true,
      },
      {
        segment_name: 'disease_severity',
        segment_value: 'moderate',
        cate_estimate: 0.012,
        cate_ci_lower: 0.004,
        cate_ci_upper: 0.02,
        sample_size: 3975,
        statistical_significance: true,
      },
    ],
  },
  overall_ate: 0.014,
  heterogeneity_score: 0.067,
  high_responders: [],
  low_responders: [],
  policy_recommendations: [],
  key_insights: [],
  estimation_latency_ms: 1,
  analysis_latency_ms: 1,
  total_latency_ms: 2,
  timestamp: '2026-06-16T00:00:00Z',
  warnings: [],
  confidence: 0.9,
};

beforeEach(() => {
  vi.clearAllMocks();
  mockSegments();
  mockInsight();
});

describe('HeterogeneousTreatmentEffects — real segment CATE', () => {
  it('shows an honest empty state with a run action before any analysis', () => {
    render(<HeterogeneousTreatmentEffects />);
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /run cate analysis/i })).toBeInTheDocument();
    // No fabricated segment names from the old fake-data widget.
    expect(screen.queryByText('High-Volume Specialists')).not.toBeInTheDocument();
  });

  it('sends the clinical contract: allowlisted defaults, no server-fixed fields', async () => {
    const mutate = vi.fn();
    mockSegments({ mutate } as unknown as Partial<Mutation>);
    const user = userEvent.setup();
    render(<HeterogeneousTreatmentEffects />);

    await user.click(screen.getByRole('button', { name: /run cate analysis/i }));

    expect(mutate).toHaveBeenCalledTimes(1);
    const arg = mutate.mock.calls[0][0];
    // patient_journeys allowlist defaults (the old business_metrics pair —
    // engagement_score -> conversion_rate — 422s against the rebuilt route).
    expect(arg.request.treatment_var).toBe('treatment_arm');
    expect(arg.request.outcome_var).toBe('persistent_180d');
    // Segment set / modifiers / data source are fixed server-side — the card
    // must NOT send them (a stale data_source was the live failure mode).
    expect(arg.request.segment_vars).toBeUndefined();
    expect(arg.request.effect_modifiers).toBeUndefined();
    expect(arg.request.data_source).toBeUndefined();
    expect(arg.request.filters).toBeUndefined();
  });

  it('passes the brand row-filter through when provided', async () => {
    const mutate = vi.fn();
    mockSegments({ mutate } as unknown as Partial<Mutation>);
    const user = userEvent.setup();
    render(<HeterogeneousTreatmentEffects brand="Remibrutinib" />);

    await user.click(screen.getByRole('button', { name: /run cate analysis/i }));

    expect(mutate.mock.calls[0][0].request.brand).toBe('Remibrutinib');
  });

  it('renders real per-segment CATE in percentage points from cate_by_segment', () => {
    mockSegments({ data: COMPLETED } as unknown as Partial<Mutation>);
    render(<HeterogeneousTreatmentEffects />);

    expect(screen.getByText('severe')).toBeInTheDocument();
    expect(screen.getByText('moderate')).toBeInTheDocument();
    // Binary outcome => percentage points, matching /segment-analysis.
    expect(screen.getByText(/\+1\.7 pp/)).toBeInTheDocument(); // severe CATE
    expect(screen.getByText(/\+1\.2 pp/)).toBeInTheDocument(); // moderate CATE
    expect(screen.getByText('2/2 significant')).toBeInTheDocument();
    expect(screen.queryByText('High-Volume Specialists')).not.toBeInTheDocument();
  });

  it('never labels a non-binary outcome in percentage points (raw effect instead)', () => {
    mockSegments({ data: COMPLETED } as unknown as Partial<Mutation>);
    // engagement_score is a continuous covariate the backend's union allowlist
    // accepts as outcome_var — pp would be a lie, so the card must fall back
    // to the raw effect scale.
    render(<HeterogeneousTreatmentEffects outcomeVar="engagement_score" />);

    expect(screen.queryByText(/\+1\.7 pp/)).not.toBeInTheDocument();
    expect(screen.queryByText(/\+1\.4 pp/)).not.toBeInTheDocument();
    expect(screen.getByText(/\+0\.017/)).toBeInTheDocument(); // severe CATE, raw
    expect(screen.getByText(/\+0\.014/)).toBeInTheDocument(); // overall ATE, raw
  });

  it('shows a loading state while the analysis is pending', () => {
    mockSegments({ isPending: true } as unknown as Partial<Mutation>);
    render(<HeterogeneousTreatmentEffects />);
    expect(screen.getByText(/analyzing segment effects/i)).toBeInTheDocument();
  });

  it('shows a labeled error when the analysis throws (no silent fallback)', () => {
    mockSegments({ error: new Error('CATE estimation failed') } as unknown as Partial<Mutation>);
    render(<HeterogeneousTreatmentEffects />);
    expect(screen.getByText(/cate analysis failed/i)).toBeInTheDocument();
    expect(screen.queryByText('severe')).not.toBeInTheDocument();
  });

  it('treats a failed-status response as an error, never as real numbers', () => {
    mockSegments({
      data: { ...COMPLETED, status: 'failed', cate_by_segment: {}, warnings: ['internal error'] },
    } as unknown as Partial<Mutation>);
    render(<HeterogeneousTreatmentEffects />);
    expect(screen.getByText(/cate analysis failed/i)).toBeInTheDocument();
    expect(screen.queryByText('severe')).not.toBeInTheDocument();
  });

  it('never claims significance alone is a targeting mandate (info banner honesty)', () => {
    render(<HeterogeneousTreatmentEffects />);
    expect(screen.getByText(/not by itself a targeting mandate/i)).toBeInTheDocument();
    expect(screen.queryByText(/a reliable targeting opportunity/i)).not.toBeInTheDocument();
  });
});

describe('HeterogeneousTreatmentEffects — dedicated HTE strategic insight', () => {
  it('auto-generates the insight with an analysis_id-ONLY payload on completion', () => {
    const mutate = mockInsight();
    mockSegments({ data: COMPLETED } as unknown as Partial<Mutation>);
    render(<HeterogeneousTreatmentEffects />);

    expect(mutate).toHaveBeenCalledTimes(1);
    // Server-derived trust boundary: the card must never post figures.
    expect(mutate).toHaveBeenCalledWith({ analysis_id: 'seg_test_1' });
  });

  it('does not fire the insight for a failed run', () => {
    const mutate = mockInsight();
    mockSegments({
      data: { ...COMPLETED, status: 'failed', cate_by_segment: {} },
    } as unknown as Partial<Mutation>);
    render(<HeterogeneousTreatmentEffects />);
    expect(mutate).not.toHaveBeenCalled();
  });

  it('renders the insight generated for THIS run', () => {
    mockInsight({
      data: {
        insight: 'Severe patients respond strongest; differential targeting is not warranted.',
        key_takeaways: ['2 of 2 segments clear zero'],
        grounding: [{ label: 'Overall ATE', value: '+1.4pp' }],
        is_fallback: false,
        provenance: 'Segment-level CATE analysis (server-derived)',
        generated_at: '2026-07-05T00:00:00Z',
      },
      variables: { analysis_id: 'seg_test_1' },
    } as unknown as Partial<InsightMutation>);
    mockSegments({ data: COMPLETED } as unknown as Partial<Mutation>);
    render(<HeterogeneousTreatmentEffects />);

    expect(screen.getByText(/severe patients respond strongest/i)).toBeInTheDocument();
    expect(screen.getByText('2 of 2 segments clear zero')).toBeInTheDocument();
    expect(screen.getByText(/server-derived/)).toBeInTheDocument();
  });

  it('suppresses insight output attributed to a DIFFERENT run (late resolve)', () => {
    mockInsight({
      data: {
        insight: 'Stale interpretation from a previous run.',
        key_takeaways: [],
        grounding: [],
        is_fallback: false,
        provenance: 'Segment-level CATE analysis (server-derived)',
        generated_at: '2026-07-05T00:00:00Z',
      },
      variables: { analysis_id: 'seg_previous_run' },
    } as unknown as Partial<InsightMutation>);
    mockSegments({ data: COMPLETED } as unknown as Partial<Mutation>);
    render(<HeterogeneousTreatmentEffects />);

    expect(screen.queryByText(/stale interpretation/i)).not.toBeInTheDocument();
  });
});

// -----------------------------------------------------------------------------
// #1841 — poll-ceiling expiry on this card. The card passes maxWaitMs 300 s and
// its default (no brand) request is the all-brands workload, which
// /segment-analysis gives 600 s (single-brand runs measured 153–208 s), so the
// ceiling is reachable. The record is durable and still running: the card must
// NOT render the rose "failed" card (whose only action, Refresh, is a NEW POST)
// but a non-destructive still-running state with Keep waiting that re-attaches
// to the SAME analysis_id via the GET-only resume branch.
// -----------------------------------------------------------------------------
describe('HeterogeneousTreatmentEffects — poll-ceiling expiry keeps the analysis_id (#1841)', () => {
  const timeout = () => new SegmentAnalysisTimeoutError('seg_hte_1841', 300_000);

  it('renders a non-destructive still-running state, not "CATE analysis failed"', () => {
    mockSegments({ error: timeout() } as unknown as Partial<Mutation>);
    render(<HeterogeneousTreatmentEffects />);

    expect(screen.queryByText(/cate analysis failed/i)).not.toBeInTheDocument();
    expect(screen.getByText(/still running/i)).toBeInTheDocument();
    expect(screen.getByText('seg_hte_1841')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /keep waiting/i })).toBeInTheDocument();
    // Still no fabricated numbers.
    expect(screen.queryByText('severe')).not.toBeInTheDocument();
  });

  it('Keep waiting resumes the SAME id via the GET-only branch — never a second POST', async () => {
    const mutate = vi.fn();
    mockSegments({ mutate, error: timeout() } as unknown as Partial<Mutation>);
    const user = userEvent.setup();
    render(<HeterogeneousTreatmentEffects />);

    await user.click(screen.getByRole('button', { name: /keep waiting/i }));

    expect(mutate).toHaveBeenCalledTimes(1);
    const arg = mutate.mock.calls[0][0];
    expect(arg).toMatchObject({ resumeAnalysisId: 'seg_hte_1841', maxWaitMs: 300_000 });
    expect(arg.request).toBeUndefined();
  });

  it('keeps Refresh as a genuine re-run (new POST) for real failures', async () => {
    const mutate = vi.fn();
    mockSegments({ mutate, error: new Error('CATE estimation failed') } as unknown as Partial<Mutation>);
    const user = userEvent.setup();
    render(<HeterogeneousTreatmentEffects />);

    expect(screen.getByText(/cate analysis failed/i)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /keep waiting/i })).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /refresh segment analysis/i }));
    expect(mutate.mock.calls[0][0].request).toBeDefined();
  });
});
