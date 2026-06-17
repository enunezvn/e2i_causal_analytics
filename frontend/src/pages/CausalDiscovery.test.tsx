/**
 * CausalDiscovery Page — agent-driven coverage
 * ============================================
 *
 * The page is now one-click agent discovery: pick treatment/outcome, the
 * causal_impact agent LEARNS the DAG from data (guided structure discovery),
 * estimates the effect data-drivenly, and refutes. These tests lock: the honest
 * empty/error states, that the LEARNED-FROM-DATA provenance + data-identified
 * confounders are surfaced, and the rendered result (effect, estimator, gate,
 * DAG). Radix <Select> is not interacted with (not reliably testable in jsdom).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import CausalDiscovery from './CausalDiscovery';

// Stub the heavy DAG viz — assert the page feeds it the agent's learned graph,
// not its internal SVG rendering.
vi.mock('@/components/visualizations/CausalDiscovery', () => ({
  CausalDiscovery: ({ nodes, edges }: { nodes: unknown[]; edges: unknown[] }) => (
    <div data-testid="causal-dag" data-nodes={nodes.length} data-edges={edges.length} />
  ),
}));

vi.mock('@/hooks/api', () => ({
  useCausalVariables: vi.fn(),
  useRunCausalAgentAnalysis: vi.fn(),
}));

import { useCausalVariables, useRunCausalAgentAnalysis } from '@/hooks/api';

const VARIABLES = {
  dataset: 'patient_journeys',
  treatment_candidates: ['treatment_arm', 'treatment_initiated'],
  outcome_candidates: ['persistent_180d', 'discontinued_180d'],
  covariate_candidates: ['disease_severity', 'engagement_score'],
  columns: [],
};

const RESULT = {
  analysis_id: 'r1',
  status: 'completed',
  treatment_var: 'treatment_arm',
  outcome_var: 'persistent_180d',
  dataset: 'patient_journeys',
  n_rows: 1200,
  data_source: 'synthetic',
  dag: {
    nodes: ['treatment_arm', 'persistent_180d', 'disease_severity', 'engagement_score'],
    edges: [
      ['treatment_arm', 'persistent_180d'],
      ['disease_severity', 'treatment_arm'],
      ['disease_severity', 'persistent_180d'],
      ['engagement_score', 'treatment_arm'],
      ['engagement_score', 'persistent_180d'],
    ],
    treatment_nodes: ['treatment_arm'],
    outcome_nodes: ['persistent_180d'],
    adjustment_sets: [['disease_severity', 'engagement_score']],
    dag_dot: null,
  },
  dag_source: 'discovered',
  discovered_confounders: ['disease_severity', 'engagement_score'],
  ate: 0.0875,
  ate_ci_lower: 0.0867,
  ate_ci_upper: 0.0884,
  standard_error: 0.0004,
  p_value: 0.0,
  statistical_significance: true,
  selected_estimator: 'LinearDML',
  confidence: 0.81,
  refutation: {
    gate_decision: 'proceed',
    passed: true,
    needs_review: false,
    tests_passed: 1,
    tests_total: 3,
    sensitivity_e_value: 1.6,
  },
  narrative: 'Treatment raises persistence.',
  executive_summary: 'Positive, robust effect.',
  recommendations: ['Prioritize adherence support'],
  key_insights: [],
  warnings: [],
  latency_ms: 4200,
};

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe('CausalDiscovery — agent-driven', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (useCausalVariables as ReturnType<typeof vi.fn>).mockReturnValue({ data: VARIABLES });
    (useRunCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      mutateAsync: vi.fn(),
      isPending: false,
      isError: false,
      error: null,
    });
  });

  it('renders an honest empty state before any run', () => {
    render(<CausalDiscovery />, { wrapper: createWrapper() });
    expect(screen.getByText(/No discovery run yet/i)).toBeInTheDocument();
  }, 20000);

  it('surfaces the learned-from-data provenance and data-identified confounders', () => {
    (useRunCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: RESULT,
      mutateAsync: vi.fn(),
      isPending: false,
      isError: false,
      error: null,
    });
    render(<CausalDiscovery />, { wrapper: createWrapper() });
    // The DAG is reported as learned from the data, not a hardcoded model.
    expect(screen.getByText(/Learned from data/i)).toBeInTheDocument();
    // The data-identified confounders (adjustment set) are surfaced.
    expect(screen.getByText(/disease_severity, engagement_score/)).toBeInTheDocument();
  }, 20000);

  it('renders the agent result: effect, estimator used, robustness gate, DAG', () => {
    (useRunCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: RESULT,
      mutateAsync: vi.fn(),
      isPending: false,
      isError: false,
      error: null,
    });
    render(<CausalDiscovery />, { wrapper: createWrapper() });
    expect(screen.getByText('0.0875')).toBeInTheDocument();
    expect(screen.getByText('LinearDML')).toBeInTheDocument();
    expect(screen.getByText('Proceed')).toBeInTheDocument();
    // The learned DAG (5 edges) is fed to the viz.
    expect(screen.getByTestId('causal-dag')).toHaveAttribute('data-edges', '5');
    expect(screen.getByText('Treatment raises persistence.')).toBeInTheDocument();
  }, 20000);

  it('surfaces a run failure honestly (no fabricated effect)', () => {
    (useRunCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      mutateAsync: vi.fn(),
      isPending: false,
      isError: true,
      error: { message: 'No usable estimation rows.' },
    });
    render(<CausalDiscovery />, { wrapper: createWrapper() });
    expect(screen.getByText(/Discovery could not run/i)).toBeInTheDocument();
  }, 20000);

  it('shows a needs-review estimate honestly rather than as a clean result', () => {
    (useRunCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { ...RESULT, status: 'needs_review', warnings: ['Gate flagged for review.'] },
      mutateAsync: vi.fn(),
      isPending: false,
      isError: false,
      error: null,
    });
    render(<CausalDiscovery />, { wrapper: createWrapper() });
    expect(screen.getByText(/needs expert review/i)).toBeInTheDocument();
  }, 20000);
});
