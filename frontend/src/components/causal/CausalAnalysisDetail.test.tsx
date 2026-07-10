// frontend/src/components/causal/CausalAnalysisDetail.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { renderWithProviders, screen } from '@/test/utils';
import { CausalAnalysisDetail } from './CausalAnalysisDetail';
import type { AgentCausalAnalysisResponse } from '@/types/causal';

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

describe('CausalAnalysisDetail', () => {
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
});
