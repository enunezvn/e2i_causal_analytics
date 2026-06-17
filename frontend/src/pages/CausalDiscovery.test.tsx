/**
 * CausalDiscovery Page — validated-effects leaderboard
 * ====================================================
 *
 * The page surfaces the agent's VALIDATED causal effects ranked by confidence +
 * impact (discover-effects job), and drills into any validated row's DAG +
 * refutation. These tests lock: the honest empty/running states, the ranked
 * leaderboard rendering, kicking off discovery, and the drill-down detail.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import CausalDiscovery from './CausalDiscovery';

// Stub the heavy DAG viz — assert the page feeds it the agent's graph.
vi.mock('@/components/visualizations/CausalDiscovery', () => ({
  CausalDiscovery: ({ nodes, edges }: { nodes: unknown[]; edges: unknown[] }) => (
    <div data-testid="causal-dag" data-nodes={nodes.length} data-edges={edges.length} />
  ),
}));

vi.mock('@/hooks/api', () => ({
  useDiscoverEffects: vi.fn(),
}));

vi.mock('@/api/causal', () => ({
  getCausalAgentAnalysis: vi.fn(),
}));

import { useDiscoverEffects } from '@/hooks/api';
import { getCausalAgentAnalysis } from '@/api/causal';

const EFFECTS = [
  {
    treatment: 'treatment_arm',
    outcome: 'persistent_180d',
    status: 'completed',
    ate: 0.0875,
    ate_ci_lower: 0.0867,
    ate_ci_upper: 0.0884,
    p_value: 0,
    statistical_significance: true,
    selected_estimator: 'LinearDML',
    gate_decision: 'proceed',
    confidence_score: 0.9,
    impact: 0.0875,
    n_rows: 1500,
    analysis_id: 'a1',
  },
  {
    treatment: 'treatment_arm',
    outcome: 'treatment_initiated',
    status: 'blocked',
    ate: -0.006,
    ate_ci_lower: -0.02,
    ate_ci_upper: 0.008,
    p_value: 0.01,
    statistical_significance: true,
    selected_estimator: 'LinearDML',
    gate_decision: 'block',
    confidence_score: 0.4,
    impact: 0.006,
    n_rows: 1500,
    analysis_id: 'a3',
  },
  {
    treatment: 'treatment_arm',
    outcome: 'discontinued_180d',
    status: 'failed',
    statistical_significance: false,
    confidence_score: 0,
    n_rows: 0,
    analysis_id: null,
  },
];

const COMPLETED_JOB = {
  job_id: 'j1',
  status: 'completed',
  dataset: 'patient_journeys',
  total: 3,
  completed: 3,
  effects: EFFECTS,
  note: 'ranked',
};

const DETAIL = {
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
  dag_source: 'discovered',
  discovered_confounders: ['disease_severity'],
  ate: 0.0875,
  ate_ci_lower: 0.0867,
  ate_ci_upper: 0.0884,
  p_value: 0,
  statistical_significance: true,
  selected_estimator: 'LinearDML',
  refutation: { gate_decision: 'proceed', passed: true, needs_review: false, tests_passed: 1, tests_total: 3, sensitivity_e_value: 1.6 },
  narrative: 'Treatment raises persistence.',
  executive_summary: 'Positive, robust effect.',
  recommendations: [],
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

function mockHook(overrides: Record<string, unknown> = {}) {
  (useDiscoverEffects as ReturnType<typeof vi.fn>).mockReturnValue({
    start: vi.fn(),
    isStarting: false,
    startError: null,
    job: null,
    ...overrides,
  });
}

describe('CausalDiscovery — validated-effects leaderboard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockHook();
    (getCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockResolvedValue(DETAIL);
  });

  it('shows an honest empty state before any discovery run', () => {
    render(<CausalDiscovery />, { wrapper: createWrapper() });
    expect(screen.getByText(/No discovery run yet/i)).toBeInTheDocument();
  }, 20000);

  it('renders the ranked leaderboard of validated effects', () => {
    mockHook({ job: COMPLETED_JOB });
    render(<CausalDiscovery />, { wrapper: createWrapper() });
    // Each candidate question (unique outcomes), and the three honest verdicts:
    // proceed (validated), blocked (computed but failed robustness), failed (no run).
    expect(screen.getByText('persistent_180d')).toBeInTheDocument();
    expect(screen.getByText('treatment_initiated')).toBeInTheDocument();
    expect(screen.getByText('discontinued_180d')).toBeInTheDocument();
    expect(screen.getByText('Proceed')).toBeInTheDocument();
    expect(screen.getByText('Blocked')).toBeInTheDocument();
    expect(screen.getByText('Failed')).toBeInTheDocument();
    expect(screen.getByText('0.0875')).toBeInTheDocument();
  }, 20000);

  it('shows progress while the agent is validating', () => {
    mockHook({ job: { ...COMPLETED_JOB, status: 'running', completed: 1 } });
    render(<CausalDiscovery />, { wrapper: createWrapper() });
    expect(screen.getByText(/Validating… \(1\/3\)/)).toBeInTheDocument();
  }, 20000);

  it('starts discovery when the button is clicked', () => {
    const start = vi.fn();
    mockHook({ start });
    render(<CausalDiscovery />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByRole('button', { name: /Discover causal effects/i }));
    expect(start).toHaveBeenCalled();
  }, 20000);

  it('drills into a validated row: shows its DAG + estimator + gate', async () => {
    mockHook({ job: COMPLETED_JOB });
    render(<CausalDiscovery />, { wrapper: createWrapper() });
    // Click the completed row (its question cell).
    fireEvent.click(screen.getByText('persistent_180d'));
    // The full validated detail loads (mocked getCausalAgentAnalysis) -> DAG fed.
    const dag = await screen.findByTestId('causal-dag');
    expect(dag).toHaveAttribute('data-edges', '2');
    expect(screen.getByText(/Treatment raises persistence/)).toBeInTheDocument();
    expect(getCausalAgentAnalysis).toHaveBeenCalledWith('a1');
  }, 20000);
});
