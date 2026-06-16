/**
 * HeterogeneousTreatmentEffects Tests
 * ===================================
 *
 * Red-first guard for the fake-CATE finding: the widget formerly booted
 * `useState(SAMPLE_SEGMENTS)` (fabricated segments: "High-Volume
 * Specialists" CATE +23%, p=0.001 ...) with a no-op transform effect, so
 * every render showed fake treatment effects regardless of API state.
 *
 * Desired behavior (three honest states only):
 * - no analysis yet  -> explicit empty state with a "Run CATE analysis" action
 * - analysis success -> real segment_results from /causal/hierarchical/analyze
 * - analysis failure / demo placeholder -> labeled error / empty, never fake
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HeterogeneousTreatmentEffects } from './HeterogeneousTreatmentEffects';
import * as useCausal from '@/hooks/api/use-causal';
import * as useExplain from '@/hooks/api/use-explain';
import type { HierarchicalAnalysisResponse } from '@/types/causal';
import { CausalAnalysisStatus } from '@/types/causal';

vi.mock('@/hooks/api/use-causal');
vi.mock('@/hooks/api/use-explain');

type HierMutation = ReturnType<typeof useCausal.useRunHierarchicalAnalysisAndWait>;

function mockHierarchical(overrides: Partial<HierMutation> = {}) {
  vi.mocked(useCausal.useRunHierarchicalAnalysisAndWait).mockReturnValue({
    mutate: vi.fn(),
    data: undefined,
    error: null,
    isPending: false,
    ...overrides,
  } as unknown as HierMutation);
}

const COMPLETED_RESPONSE: HierarchicalAnalysisResponse = {
  analysis_id: 'ha_test_1',
  status: CausalAnalysisStatus.COMPLETED,
  segment_results: [
    {
      segment_id: 0,
      segment_name: 'high_uplift',
      n_samples: 412,
      uplift_range: [0.4, 0.9],
      cate_mean: 0.31,
      cate_std: 0.05,
      cate_ci_lower: 0.21,
      cate_ci_upper: 0.41,
      success: true,
    },
    {
      segment_id: 1,
      segment_name: 'low_uplift',
      n_samples: 1024,
      uplift_range: [0.0, 0.4],
      cate_mean: 0.02,
      cate_std: 0.04,
      cate_ci_lower: -0.06,
      cate_ci_upper: 0.1,
      success: true,
    },
  ],
  overall_ate: 0.11,
  overall_ci_lower: 0.04,
  overall_ci_upper: 0.18,
  segment_heterogeneity: 0.62,
  n_segments_analyzed: 2,
  segmentation_method: 'quantile',
  estimator_type: 'causal_forest',
  latency_ms: 1234,
  created_at: '2026-06-12T00:00:00Z',
  warnings: [],
  errors: [],
};

beforeEach(() => {
  vi.clearAllMocks();
  mockHierarchical();
  // Legacy hook the widget formerly mis-used; harmless default if imported.
  if (useExplain.useBatchExplain) {
    vi.mocked(useExplain.useBatchExplain).mockReturnValue({
      mutate: vi.fn(),
      data: undefined,
      isPending: false,
    } as unknown as ReturnType<typeof useExplain.useBatchExplain>);
  }
});

describe('HeterogeneousTreatmentEffects — no fabricated segments', () => {
  it('renders an honest empty state (not SAMPLE_SEGMENTS) before any analysis has run', () => {
    render(<HeterogeneousTreatmentEffects />);

    // Fabricated SAMPLE_SEGMENTS values must never render.
    expect(screen.queryByText('High-Volume Specialists')).not.toBeInTheDocument();
    expect(screen.queryByText('Academic Medical Centers')).not.toBeInTheDocument();
    expect(screen.queryByText('Early Adopters')).not.toBeInTheDocument();
    expect(screen.queryByText(/p-value: 0\.001/)).not.toBeInTheDocument();

    // Honest empty state with an explicit run action instead.
    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /run cate analysis/i })
    ).toBeInTheDocument();
  });

  it('fires the hierarchical CATE analysis when the user clicks run', async () => {
    const mutate = vi.fn();
    mockHierarchical({ mutate } as unknown as Partial<HierMutation>);
    const user = userEvent.setup();
    render(<HeterogeneousTreatmentEffects />);

    await user.click(screen.getByRole('button', { name: /run cate analysis/i }));

    expect(mutate).toHaveBeenCalledTimes(1);
    const arg = mutate.mock.calls[0][0];
    expect(arg.request.treatment_var).toBeTruthy();
    expect(arg.request.outcome_var).toBeTruthy();
  });

  it('renders real segment results from a completed analysis', () => {
    mockHierarchical({ data: COMPLETED_RESPONSE } as unknown as Partial<HierMutation>);
    render(<HeterogeneousTreatmentEffects />);

    expect(screen.getByText('high_uplift')).toBeInTheDocument();
    expect(screen.getByText('low_uplift')).toBeInTheDocument();
    // Real CATE from the API (+31.0%), not the fabricated +23%.
    expect(screen.getByText(/\+31\.0%/)).toBeInTheDocument();
    expect(screen.queryByText('High-Volume Specialists')).not.toBeInTheDocument();
  });

  it('shows a loading state while the analysis is pending', () => {
    mockHierarchical({ isPending: true } as unknown as Partial<HierMutation>);
    render(<HeterogeneousTreatmentEffects />);
    expect(screen.getByText(/analyzing segment effects/i)).toBeInTheDocument();
    expect(screen.queryByText('High-Volume Specialists')).not.toBeInTheDocument();
  });

  it('shows a labeled error state when the analysis fails (no silent fallback)', () => {
    mockHierarchical({
      error: new Error('CATE estimation failed'),
    } as unknown as Partial<HierMutation>);
    render(<HeterogeneousTreatmentEffects />);

    expect(screen.getByText(/cate estimation failed/i)).toBeInTheDocument();
    expect(screen.queryByText('High-Volume Specialists')).not.toBeInTheDocument();
  });

  it('shows an honest "data not wired" state for a 503 no-data-backend error (not a red failure)', () => {
    mockHierarchical({
      error: Object.assign(
        new Error(
          'Causal pipeline endpoints have no real data backend wired. ' +
            'Pass demo_mode=true to get a clearly-labeled pinned-zero placeholder.',
        ),
        { status: 503 },
      ),
    } as unknown as Partial<HierMutation>);
    render(<HeterogeneousTreatmentEffects />);

    // Honest informational state, not the red "CATE analysis failed" alarm.
    expect(screen.getByText(/Live CATE data isn.t wired yet/i)).toBeInTheDocument();
    expect(screen.queryByText(/CATE analysis failed/i)).not.toBeInTheDocument();
    // The dev-facing "pass demo_mode=true" detail is not leaked to the user here.
    expect(screen.queryByText(/demo_mode=true/i)).not.toBeInTheDocument();
  });

  it('still shows a red error for a genuine non-503 failure', () => {
    mockHierarchical({
      error: Object.assign(new Error('CATE estimation failed'), { status: 500 }),
    } as unknown as Partial<HierMutation>);
    render(<HeterogeneousTreatmentEffects />);
    expect(screen.getByText(/cate analysis failed/i)).toBeInTheDocument();
    expect(screen.queryByText(/Live CATE data isn.t wired yet/i)).not.toBeInTheDocument();
  });

  it('treats a demo-mode placeholder response as not-real (no fake numbers)', () => {
    mockHierarchical({
      data: { ...COMPLETED_RESPONSE, is_demo: true } as HierarchicalAnalysisResponse,
    } as unknown as Partial<HierMutation>);
    render(<HeterogeneousTreatmentEffects />);

    // Pinned-zero demo placeholders must not masquerade as real analysis.
    expect(screen.queryByText(/\+31\.0%/)).not.toBeInTheDocument();
    expect(screen.getByText('Demo-mode placeholder response')).toBeInTheDocument();
  });
});
