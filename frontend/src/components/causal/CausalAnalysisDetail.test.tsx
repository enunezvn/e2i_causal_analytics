// frontend/src/components/causal/CausalAnalysisDetail.test.tsx
import { StrictMode } from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderWithProviders, screen } from '@/test/utils';
import { CausalAnalysisDetail } from './CausalAnalysisDetail';
import { useClinicalContext, useClinicalNarrativeInsight } from '@/hooks/api';
import type { AgentCausalAnalysisResponse, ClinicalContext } from '@/types/causal';

// The detail panel uses two hooks from @/hooks/api. Mock both so the clinical
// context query and the narrative mutation are observable: without a `brand`
// prop the context query stays disabled, which is exactly what
// `{ data: undefined }` reproduces for the rest of this suite; the narrative
// mutation is spied via mockNarrativeMutate/mockNarrativeReset.
const mockNarrativeMutate = vi.fn();
const mockNarrativeReset = vi.fn();
vi.mock('@/hooks/api', () => ({
  useClinicalContext: vi.fn(() => ({ data: undefined })),
  useClinicalNarrativeInsight: vi.fn(() => ({
    data: undefined,
    isPending: false,
    mutate: mockNarrativeMutate,
    reset: mockNarrativeReset,
  })),
}));

// Stub the heavy DAG viz — assert the detail feeds it the agent's graph + refutation.
vi.mock('@/components/visualizations/CausalDiscovery', () => ({
  CausalDiscovery: ({
    nodes,
    edges,
    refutationResults,
  }: {
    nodes: unknown[];
    edges: unknown[];
    refutationResults?: unknown[];
  }) => (
    <div
      data-testid="causal-dag"
      data-nodes={nodes.length}
      data-edges={edges.length}
      data-refutations={refutationResults?.length ?? 0}
      data-refutation-warnings={
        (refutationResults as Array<{ status?: string | null }> | undefined)?.filter(
          (r) => r.status === 'warning'
        ).length ?? 0
      }
    />
  ),
}));

const RESULT: AgentCausalAnalysisResponse = {
  analysis_id: 'a1',
  status: 'completed',
  treatment_var: 'treatment_arm',
  outcome_var: 'persistent_180d',
  dataset: 'patient_journeys',
  n_rows: 1500,
  data_source: 'synthetic',
  dag: {
    nodes: ['treatment_arm', 'persistent_180d', 'disease_severity'],
    edges: [
      ['treatment_arm', 'persistent_180d'],
      ['disease_severity', 'persistent_180d'],
    ],
    treatment_nodes: ['treatment_arm'],
    outcome_nodes: ['persistent_180d'],
    adjustment_sets: [['disease_severity']],
    dag_dot: null,
  },
  discovered_confounders: ['disease_severity'],
  ate: 0.0875,
  ate_ci_lower: 0.0867,
  ate_ci_upper: 0.0884,
  p_value: 0,
  statistical_significance: true,
  selected_estimator: 'LinearDML',
  estimator_comparison: {
    candidates: [
      { estimator: 'causal_forest', success: true, energy_score: 0.51, ate: 0.1, error: null, is_selected: false },
      { estimator: 'linear_dml', success: true, energy_score: 0.48, ate: 0.0875, error: null, is_selected: true },
    ],
    selection_reason: 'confounding-robust preferred over OLS',
    energy_score_gap: 0.03,
    n_evaluated: 2,
    n_succeeded: 2,
    quality_tier: 'good',
    requires_review: false,
  },
  refutation: {
    gate_decision: 'proceed',
    passed: true,
    needs_review: false,
    tests_passed: 2,
    tests_total: 3,
    sensitivity_e_value: 1.6,
    tests: [
      { test_name: 'placebo_treatment', passed: true, original_effect: 0.0875, new_effect: 0.001, p_value: 0.6 },
      { test_name: 'random_common_cause', passed: true, original_effect: 0.0875, new_effect: 0.086, p_value: 0.9 },
      { test_name: 'unobserved_common_cause', passed: true, original_effect: 0.0875, new_effect: 0.0875, p_value: 0 },
    ],
  },
  narrative: 'Treatment raises persistence.',
  executive_summary: 'Positive, robust effect.',
  recommendations: ['Monitor outcomes closely'],
  key_insights: ['Estimated causal effect: 0.09 (small)'],
  warnings: [],
  latency_ms: 4200,
};

const CLINICAL: ClinicalContext = {
  brand: 'Remibrutinib',
  drug_name: 'remibrutinib',
  disease: 'Chronic spontaneous urticaria',
  our_outcome: 'persistent_180d',
  mapped_endpoint: null,
  mechanism: { mechanism_of_action: 'BTK inhibitor', source: 'chembl' },
  pivotal_endpoints: { endpoints: [], source: 'clinicaltrials.gov' },
  real_world_evidence: null,
  approved_indications: {
    indications: [], limitations_of_use: null, boxed_warning: null, source: 'openfda',
  },
  competitor_landscape: { competitors: [], count: 0, source: 'curated' },
  honesty_label: 'Effect estimate = a SYNTHETIC patient cohort.',
};

describe('CausalAnalysisDetail', () => {
  beforeEach(() => {
    mockNarrativeMutate.mockClear();
    mockNarrativeReset.mockClear();
  });

  it('renders the effect, estimator, gate, and discovered confounders', () => {
    renderWithProviders(<CausalAnalysisDetail result={RESULT} />);
    // ATE renders in the headline AND the selected-estimator comparison row (same value by design) → getAllByText.
    expect(screen.getAllByText('0.0875').length).toBeGreaterThan(0);
    expect(screen.getByText(/Linear dml/i)).toBeInTheDocument();
    expect(screen.getByText('Proceed')).toBeInTheDocument();
    expect(screen.getByText(/disease_severity/)).toBeInTheDocument();
  });

  it('feeds the DAG (nodes + edges) and per-test refutation into the viz', () => {
    renderWithProviders(<CausalAnalysisDetail result={RESULT} />);
    const dag = screen.getByTestId('causal-dag');
    expect(dag).toHaveAttribute('data-edges', '2');
    expect(dag).toHaveAttribute('data-refutations', '3');
  });

  it('maps the three-state refutation status through to the viz (#1867)', () => {
    const withWarning: AgentCausalAnalysisResponse = {
      ...RESULT,
      refutation: {
        ...RESULT.refutation!,
        tests: [
          { test_name: 'placebo_treatment', passed: true, status: 'passed', original_effect: 0.0875, new_effect: 0.001, p_value: 0.6 },
          // The prod contradiction: E-value in the warning band arrives with
          // passed:false but status:'warning' — the warning must survive mapping.
          { test_name: 'unobserved_common_cause', passed: false, status: 'warning', original_effect: 0.0875, new_effect: 0.0875, p_value: 0 },
        ],
      },
    };
    renderWithProviders(<CausalAnalysisDetail result={withWarning} />);
    expect(screen.getByTestId('causal-dag')).toHaveAttribute('data-refutation-warnings', '1');
  });

  it('renders the estimator-comparison panel (the #1030 data-driven evaluation)', () => {
    renderWithProviders(<CausalAnalysisDetail result={RESULT} />);
    expect(screen.getByText('Estimator selection (data-driven)')).toBeInTheDocument();
    expect(screen.getByText(/2\/2 applicable estimators fit/)).toBeInTheDocument();
    expect(screen.getByText(/confounding-robust preferred over OLS/)).toBeInTheDocument();
  });

  it('renders empty-backdoor estimators as "Not applicable", not a fit failure', () => {
    // A zero-covariate (randomized) question: only OLS applies; the covariate-based
    // estimators are skipped with an honest not-applicable reason, never a raw
    // sklearn "0 feature(s)" traceback.
    const rct: AgentCausalAnalysisResponse = {
      ...RESULT,
      selected_estimator: 'ols',
      estimator_comparison: {
        candidates: [
          { estimator: 'ols', success: true, skipped: false, energy_score: 0.42, ate: 0.43, error: null, is_selected: true },
          { estimator: 'causal_forest', success: false, skipped: true, energy_score: null, ate: null, error: 'not applicable: no covariates to adjust for (randomized / empty-backdoor design).', is_selected: false },
          { estimator: 'linear_dml', success: false, skipped: true, energy_score: null, ate: null, error: 'not applicable: no covariates to adjust for (randomized / empty-backdoor design).', is_selected: false },
        ],
        selection_reason: 'No covariates to adjust for (randomized / empty-backdoor design), so ols is correct.',
        energy_score_gap: 0,
        n_evaluated: 3,
        n_succeeded: 1,
        quality_tier: 'good',
        requires_review: false,
      },
    };
    renderWithProviders(<CausalAnalysisDetail result={rct} />);
    // Header distinguishes fit vs not-applicable, never implying the skipped ones failed.
    expect(screen.getByText(/1\/1 applicable estimator fit/)).toBeInTheDocument();
    expect(screen.getByText(/2 not applicable \(no covariates\)/)).toBeInTheDocument();
    // The skipped estimators are badged "Not applicable" and never show the raw
    // sklearn "0 feature(s)" traceback text.
    expect(screen.getAllByText('Not applicable').length).toBe(2);
    expect(screen.queryByText(/0 feature/)).toBeNull();
  });

  it('renders interpretation: key insights + recommendations', () => {
    renderWithProviders(<CausalAnalysisDetail result={RESULT} />);
    expect(screen.getByText('Positive, robust effect.')).toBeInTheDocument();
    expect(screen.getByText('Key insights')).toBeInTheDocument();
    expect(screen.getByText('Recommended actions')).toBeInTheDocument();
    expect(screen.getByText('Monitor outcomes closely')).toBeInTheDocument();
  });

  it('shows an honest empty-state when no DAG was produced', () => {
    renderWithProviders(<CausalAnalysisDetail result={{ ...RESULT, dag: { ...RESULT.dag, nodes: [], edges: [] } }} />);
    expect(screen.getByText('No DAG produced')).toBeInTheDocument();
  });

  it('surfaces the naive (unadjusted) ATE and the confounding bias adjustment removed', () => {
    renderWithProviders(
      <CausalAnalysisDetail
        result={{
          ...RESULT,
          ate: 0.185,
          naive_ate: 0.2815,
          naive_ate_ci_lower: 0.26,
          naive_ate_ci_upper: 0.3,
          confounding_bias_removed: 0.0965,
        }}
      />
    );
    // Both the naive estimate and the bias-removed delta are shown to the analyst.
    expect(screen.getByText(/naive \(unadjusted\)/i)).toBeInTheDocument();
    expect(screen.getByText('0.2815')).toBeInTheDocument();
    expect(screen.getByText(/0\.0965/)).toBeInTheDocument();
    expect(screen.getByText(/bias removed|overstated/i)).toBeInTheDocument();
  });

  it('marks the naive contrast not-applicable for a non-binary treatment', () => {
    renderWithProviders(
      <CausalAnalysisDetail
        result={{ ...RESULT, naive_ate: null, naive_ate_ci_lower: null, naive_ate_ci_upper: null, confounding_bias_removed: null }}
      />
    );
    expect(screen.getByText(/not applicable/i)).toBeInTheDocument();
  });

  // ── #1188: RCT baseline adjustment must be framed as VARIANCE REDUCTION ──
  const EFFICIENCY_RESULT: AgentCausalAnalysisResponse = {
    ...RESULT,
    treatment_var: 'control_group_flag',
    outcome_var: 'action_taken',
    dataset: 'nba_triggers',
    ate: 0.0752,
    ate_ci_lower: 0.0657,
    ate_ci_upper: 0.0848,
    naive_ate: 0.0819,
    naive_ate_ci_lower: 0.0716,
    naive_ate_ci_upper: 0.0921,
    confounding_bias_removed: 0.0067,
    adjustment_type: 'efficiency',
    baseline_covariates: ['disease_severity', 'age_at_diagnosis'],
    selected_estimator: 'linear_dml',
    estimator_comparison: {
      candidates: [
        { estimator: 'linear_dml', success: true, energy_score: 0.44, ate: 0.0752, error: null, is_selected: true },
        { estimator: 'ols', success: true, energy_score: 0.46, ate: 0.0819, error: null, is_selected: false },
      ],
      selection_reason:
        'Randomized design: baseline covariates enter only for variance reduction.',
      energy_score_gap: 0.02,
      n_evaluated: 2,
      n_succeeded: 2,
      quality_tier: 'good',
      requires_review: false,
    },
  };

  it('frames an efficiency run as precision adjustment, never confounding removal', () => {
    renderWithProviders(<CausalAnalysisDetail result={EFFICIENCY_RESULT} />);
    // Panel header switches to the precision framing.
    expect(screen.getByText(/precision adjustment/i)).toBeInTheDocument();
    expect(screen.queryByText(/^Confounding adjustment$/)).toBeNull();
    // Copy explains variance reduction + unbiasedness; the confounding-bias
    // prose ('overstated'/'bias removed') must NOT appear for an RCT.
    expect(screen.getAllByText(/variance reduction/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/unbiased/i).length).toBeGreaterThan(0);
    expect(screen.queryByText(/overstated|bias removed/i)).toBeNull();
  });

  it('shows the unadjusted anchor with BOTH intervals so the tightening is visible', () => {
    renderWithProviders(<CausalAnalysisDetail result={EFFICIENCY_RESULT} />);
    // Anchor labeled as the unadjusted reference, with its CI; adjusted CI too.
    expect(screen.getByText(/unadjusted \(anchor\)/i)).toBeInTheDocument();
    expect(screen.getAllByText('0.0819').length).toBeGreaterThan(0);
    // formatCI renders 3 decimals.
    expect(screen.getAllByText(/\[0\.072, 0\.092\]/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/\[0\.066, 0\.085\]/).length).toBeGreaterThan(0);
  });

  it('lists the baseline covariates the run adjusted for', () => {
    renderWithProviders(<CausalAnalysisDetail result={EFFICIENCY_RESULT} />);
    expect(screen.getByText(/disease_severity, age_at_diagnosis/)).toBeInTheDocument();
  });

  it('badges OLS as the unbiased anchor in the estimator comparison', () => {
    renderWithProviders(<CausalAnalysisDetail result={EFFICIENCY_RESULT} />);
    expect(screen.getByText('Unbiased anchor')).toBeInTheDocument();
  });
});

// #1763: every detail view IS a (treatment -> outcome) analysis; the clinical context
// it requests must say so, or the panel answers a question nobody asked.
describe('CausalAnalysisDetail — clinical context follows the analysis (#1763)', () => {
  it('asks for context for this analysis: brand, outcome AND treatment', () => {
    renderWithProviders(<CausalAnalysisDetail result={RESULT} brand="Kisqali" />);
    expect(vi.mocked(useClinicalContext)).toHaveBeenCalledWith(
      'Kisqali',
      'persistent_180d',
      'treatment_arm'
    );
  });
});

// The narrative auto-fires exactly once per distinct analysis, gated on both the
// clinical context having loaded and a brand being in scope (the narrative is
// brand-scoped, same as the clinical context itself).
describe('CausalAnalysisDetail — auto-fires the clinical narrative (Task 8)', () => {
  beforeEach(() => {
    mockNarrativeMutate.mockClear();
    mockNarrativeReset.mockClear();
    // A per-test `mockReturnValue` on useClinicalNarrativeInsight (see the
    // panel-passthrough test below) otherwise LEAKS into every later test in
    // this file — `mockClear()` only resets call history, not the return-value
    // implementation. Restore the module factory's default here so each test
    // starts from the same "no narrative yet" state and opts in explicitly.
    vi.mocked(useClinicalNarrativeInsight).mockReturnValue({
      data: undefined,
      isPending: false,
      mutate: mockNarrativeMutate,
      reset: mockNarrativeReset,
    } as never);
  });

  it('does NOT fire the narrative before the clinical context has loaded', () => {
    vi.mocked(useClinicalContext).mockReturnValue({ data: undefined } as never);
    renderWithProviders(<CausalAnalysisDetail result={RESULT} brand="Remibrutinib" />);
    expect(mockNarrativeMutate).not.toHaveBeenCalled();
  });

  it('fires the narrative exactly once when context + result are both ready', () => {
    vi.mocked(useClinicalContext).mockReturnValue({ data: CLINICAL } as never);
    const { rerender } = renderWithProviders(
      <CausalAnalysisDetail result={RESULT} brand="Remibrutinib" />
    );
    expect(mockNarrativeMutate).toHaveBeenCalledTimes(1);
    expect(mockNarrativeMutate).toHaveBeenCalledWith({
      brand: 'Remibrutinib',
      grain: 'patient',
      treatment: 'treatment_arm',
      outcome: 'persistent_180d',
      ate: 0.0875,
      ate_ci_lower: 0.0867,
      ate_ci_upper: 0.0884,
      gate_decision: RESULT.refutation.gate_decision ?? null,
      // #1868: per-test verdicts ride along so the narrative can name a
      // warning honestly instead of claiming "survived all robustness checks".
      refutation_tests: [
        { test_name: 'placebo_treatment', passed: true, status: null, details: null },
        { test_name: 'random_common_cause', passed: true, status: null, details: null },
        { test_name: 'unobserved_common_cause', passed: true, status: null, details: null },
      ],
    });
    // A re-render with the same result must not re-fire (keyed auto-fire).
    rerender(<CausalAnalysisDetail result={RESULT} brand="Remibrutinib" />);
    expect(mockNarrativeMutate).toHaveBeenCalledTimes(1);
  });

  it('does NOT fire without a brand (the narrative is brand-scoped)', () => {
    vi.mocked(useClinicalContext).mockReturnValue({ data: CLINICAL } as never);
    renderWithProviders(<CausalAnalysisDetail result={RESULT} />);
    expect(mockNarrativeMutate).not.toHaveBeenCalled();
  });

  it('passes the in-scope narrative through to the clinical context panel', () => {
    vi.mocked(useClinicalContext).mockReturnValue({ data: CLINICAL } as never);
    vi.mocked(useClinicalNarrativeInsight).mockReturnValue({
      data: {
        insight: 'DISTINCTIVE NARRATIVE TEXT for the wiring test',
        key_takeaways: [],
        grounding: [],
        is_fallback: false,
        generated_at: '2026-08-25T00:00:00Z',
        provenance: 'LLM synthesis of the labeled clinical-context sources; facts drawn only from them.',
      },
      isPending: false,
      mutate: mockNarrativeMutate,
      reset: mockNarrativeReset,
    } as never);
    renderWithProviders(<CausalAnalysisDetail result={RESULT} brand="Remibrutinib" />);
    expect(screen.getByText(/DISTINCTIVE NARRATIVE TEXT/)).toBeInTheDocument();
  });

  it('re-fires the narrative when the analysis key changes', () => {
    vi.mocked(useClinicalContext).mockReturnValue({ data: CLINICAL } as never);
    const { rerender } = renderWithProviders(
      <CausalAnalysisDetail result={RESULT} brand="Remibrutinib" />
    );
    expect(mockNarrativeMutate).toHaveBeenCalledTimes(1);
    rerender(<CausalAnalysisDetail result={{ ...RESULT, ate: 0.2 }} brand="Remibrutinib" />);
    expect(mockNarrativeMutate).toHaveBeenCalledTimes(2);
    expect(mockNarrativeMutate).toHaveBeenLastCalledWith(
      expect.objectContaining({ ate: 0.2 })
    );
    expect(mockNarrativeReset).toHaveBeenCalledTimes(2);
  });

  it('fires exactly once under React StrictMode double-invoked effects', () => {
    vi.mocked(useClinicalContext).mockReturnValue({ data: CLINICAL } as never);
    renderWithProviders(
      <StrictMode>
        <CausalAnalysisDetail result={RESULT} brand="Remibrutinib" />
      </StrictMode>
    );
    expect(mockNarrativeMutate).toHaveBeenCalledTimes(1);
  });
});
