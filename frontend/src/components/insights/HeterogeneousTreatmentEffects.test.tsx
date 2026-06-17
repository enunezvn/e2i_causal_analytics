/**
 * HeterogeneousTreatmentEffects Tests — real segment CATE wiring.
 *
 * The card runs POST /api/segments/analyze (Heterogeneous Optimizer / EconML
 * CausalForestDML over the live synthetic cohort) and renders real per-segment
 * CATE from `cate_by_segment`. Honest states only: empty -> run action,
 * completed -> real numbers, failed -> labeled error. No fabricated segments,
 * no invented p-values.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HeterogeneousTreatmentEffects } from './HeterogeneousTreatmentEffects';
import * as useSegments from '@/hooks/api/use-segments';

vi.mock('@/hooks/api/use-segments');

type Mutation = ReturnType<typeof useSegments.useRunSegmentAnalysisAndWait>;

function mockSegments(overrides: Partial<Mutation> = {}) {
  vi.mocked(useSegments.useRunSegmentAnalysisAndWait).mockReturnValue({
    mutate: vi.fn(),
    data: undefined,
    error: null,
    isPending: false,
    ...overrides,
  } as unknown as Mutation);
}

// Faithful to a real /api/segments/analyze completed response (the values match
// the in-container verification: per-region CATE with tight CIs).
const COMPLETED = {
  analysis_id: 'seg_test_1',
  status: 'completed',
  cate_by_segment: {
    region: [
      {
        segment_name: 'region',
        segment_value: 'northeast',
        cate_estimate: 0.287,
        cate_ci_lower: 0.286,
        cate_ci_upper: 0.288,
        sample_size: 4023,
        statistical_significance: true,
      },
      {
        segment_name: 'region',
        segment_value: 'midwest',
        cate_estimate: 0.244,
        cate_ci_lower: 0.242,
        cate_ci_upper: 0.245,
        sample_size: 3975,
        statistical_significance: true,
      },
    ],
  },
  overall_ate: 0.268,
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
});

describe('HeterogeneousTreatmentEffects — real segment CATE', () => {
  it('shows an honest empty state with a run action before any analysis', () => {
    render(<HeterogeneousTreatmentEffects />);
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /run cate analysis/i })).toBeInTheDocument();
    // No fabricated segment names from the old fake-data widget.
    expect(screen.queryByText('High-Volume Specialists')).not.toBeInTheDocument();
  });

  it('fires segment analysis over the cohort when run is clicked', async () => {
    const mutate = vi.fn();
    mockSegments({ mutate } as unknown as Partial<Mutation>);
    const user = userEvent.setup();
    render(<HeterogeneousTreatmentEffects />);

    await user.click(screen.getByRole('button', { name: /run cate analysis/i }));

    expect(mutate).toHaveBeenCalledTimes(1);
    const arg = mutate.mock.calls[0][0];
    expect(arg.request.treatment_var).toBeTruthy();
    expect(arg.request.outcome_var).toBeTruthy();
    expect(arg.request.segment_vars.length).toBeGreaterThan(0);
    expect(arg.request.data_source).toBe('business_metrics');
  });

  it('renders real per-segment CATE from cate_by_segment', () => {
    mockSegments({ data: COMPLETED } as unknown as Partial<Mutation>);
    render(<HeterogeneousTreatmentEffects />);

    expect(screen.getByText('northeast')).toBeInTheDocument();
    expect(screen.getByText('midwest')).toBeInTheDocument();
    expect(screen.getByText(/\+28\.7%/)).toBeInTheDocument(); // northeast CATE
    expect(screen.getByText(/\+24\.4%/)).toBeInTheDocument(); // midwest CATE
    expect(screen.getByText('2/2 significant')).toBeInTheDocument();
    expect(screen.queryByText('High-Volume Specialists')).not.toBeInTheDocument();
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
    expect(screen.queryByText('northeast')).not.toBeInTheDocument();
  });

  it('treats a failed-status response as an error, never as real numbers', () => {
    mockSegments({
      data: { ...COMPLETED, status: 'failed', cate_by_segment: {}, warnings: ['internal error'] },
    } as unknown as Partial<Mutation>);
    render(<HeterogeneousTreatmentEffects />);
    expect(screen.getByText(/cate analysis failed/i)).toBeInTheDocument();
    expect(screen.queryByText('northeast')).not.toBeInTheDocument();
  });
});
