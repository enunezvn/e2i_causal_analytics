/**
 * CausalAnalysis Page — agent-driven coverage
 * ===========================================
 *
 * The page leverages the causal_impact agent: pick treatment/outcome (real
 * dropdowns from /causal/variables), Run -> the agent builds the DAG, estimates
 * the treatment->outcome effect data-drivenly, and refutes. These tests lock:
 * the honest empty/error states, the data-driven config (confounders from the
 * live variables endpoint), and the rendered result (effect, estimator used,
 * robustness gate, DAG). Radix <Select> is not interacted with (not reliably
 * testable in jsdom); assertions use plain text / the result / Tabs.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import CausalAnalysis from './CausalAnalysis';

// Stub the heavy DAG viz — we assert the page feeds it the agent's graph, not
// its internal SVG rendering.
vi.mock('@/components/visualizations/CausalDiscovery', () => ({
  CausalDiscovery: ({ nodes, edges }: { nodes: unknown[]; edges: unknown[] }) => (
    <div data-testid="causal-dag" data-nodes={nodes.length} data-edges={edges.length} />
  ),
}));

vi.mock('@/hooks/api', () => ({
  useCausalHealth: vi.fn(),
  useCausalAnalysisHistory: vi.fn(),
  useCausalVariables: vi.fn(),
  useCausalBrands: vi.fn(),
  useRunCausalAgentAnalysis: vi.fn(),
  useEstimators: vi.fn(),
}));

import {
  useCausalHealth,
  useCausalAnalysisHistory,
  useCausalVariables,
  useCausalBrands,
  useRunCausalAgentAnalysis,
  useEstimators,
} from '@/hooks/api';

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
  ate: 0.12,
  ate_ci_lower: 0.05,
  ate_ci_upper: 0.19,
  standard_error: 0.03,
  p_value: 0.001,
  statistical_significance: true,
  selected_estimator: 'CausalForestDML',
  confidence: 0.81,
  refutation: {
    gate_decision: 'proceed',
    passed: true,
    needs_review: false,
    tests_passed: 3,
    tests_total: 3,
    sensitivity_e_value: 1.8,
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

describe('CausalAnalysis — agent-driven', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (useCausalHealth as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined });
    (useCausalAnalysisHistory as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false,
    });
    (useCausalVariables as ReturnType<typeof vi.fn>).mockReturnValue({ data: VARIABLES });
    (useCausalBrands as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { dataset: 'patient_journeys', brands: ['Remibrutinib', 'Kisqali', 'Fabhalta'] },
      isLoading: false,
      error: null,
    });
    (useEstimators as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false,
    });
    (useRunCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      mutateAsync: vi.fn(),
      isPending: false,
      isError: false,
      error: null,
    });
  });

  it('renders an honest empty state before any run', () => {
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByText(/No analysis run yet/i)).toBeInTheDocument();
  }, 20000);

  it('shows data-driven confounders from the live variables endpoint', () => {
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    // The confounders the agent controls for come from /causal/variables —
    // proving the config is data-driven, not hardcoded rep_visits/trx_count.
    expect(screen.getByText(/disease_severity, engagement_score/)).toBeInTheDocument();
  }, 20000);

  it('offers a brand scope dropdown (defaults to all brands) and runs with it', () => {
    const mutateAsync = vi.fn().mockResolvedValue(RESULT);
    (useRunCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      mutateAsync,
      isPending: false,
      isError: false,
      error: null,
    });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    // The brand-scope control renders, defaulting to "All brands".
    expect(screen.getByLabelText('Brand')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Run Analysis/i }));
    // Default scope = all brands -> brand omitted (undefined) from the payload.
    expect(mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        treatment_var: 'treatment_arm',
        outcome_var: 'persistent_180d',
        dataset: 'patient_journeys',
        brand: undefined,
      })
    );
  }, 20000);

  it('renders the agent result: effect, estimator used, robustness gate, DAG', () => {
    (useRunCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: RESULT,
      mutateAsync: vi.fn(),
      isPending: false,
      isError: false,
      error: null,
    });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    // Treatment->outcome effect.
    expect(screen.getByText('0.120')).toBeInTheDocument();
    // The estimator the agent actually used (data-driven selection surfaced).
    expect(screen.getByText('CausalForestDML')).toBeInTheDocument();
    // Robustness gate from the real refutation.
    expect(screen.getByText('Proceed')).toBeInTheDocument();
    // DAG fed to the viz (2 edges from the fixture).
    const dag = screen.getByTestId('causal-dag');
    expect(dag).toHaveAttribute('data-edges', '2');
    // Interpretation surfaced.
    expect(screen.getByText('Treatment raises persistence.')).toBeInTheDocument();
  }, 20000);

  it('surfaces a run failure honestly (fail-closed, no fabricated effect)', () => {
    (useRunCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      mutateAsync: vi.fn(),
      isPending: false,
      isError: true,
      error: { message: 'No usable estimation rows.' },
    });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByText(/Analysis could not run/i)).toBeInTheDocument();
    expect(screen.getByText(/fail-closed/i)).toBeInTheDocument();
  }, 20000);

  it('shows the live estimator-registry total on the overview card', () => {
    (useEstimators as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { estimators: [], total: 12, by_library: {} },
      isLoading: false,
      isError: false,
    });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByText('12')).toBeInTheDocument();
  }, 20000);

  it('renders estimators from the live registry (Estimators tab)', async () => {
    const user = userEvent.setup();
    (useEstimators as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        estimators: [
          {
            name: 'ortho_forest',
            library: 'econml',
            estimator_type: 'CATE',
            description: 'Orthogonal Random Forest for CATE',
            best_for: [],
            parameters: [],
            supports_confidence_intervals: true,
            supports_heterogeneous_effects: true,
          },
        ],
        total: 12,
        by_library: { econml: ['ortho_forest'] },
      },
      isLoading: false,
      isError: false,
    });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /estimators/i }));
    expect(await screen.findByText(/ortho forest/i)).toBeInTheDocument();
  }, 20000);
});
