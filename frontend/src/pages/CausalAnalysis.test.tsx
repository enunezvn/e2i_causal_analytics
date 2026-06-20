// frontend/src/pages/CausalAnalysis.test.tsx
/**
 * CausalAnalysis Page — unified agent-led page
 * ============================================
 *
 * The page LANDS on the validated-effects leaderboard (discover-effects job),
 * faceted by grain + brand, each row surfacing its brand + plain-language
 * summary and drilling into the deep view (DAG + refutation + estimator
 * comparison). A secondary "Pose your own question" panel keeps the manual
 * treatment/outcome path sourced from /causal/variables. These tests lock the
 * honest empty/running states, the ranked leaderboard (brand + summary), the
 * facets, the drill-down, and the manual run.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import CausalAnalysis from './CausalAnalysis';

// Stub the shared deep view — assert the page mounts it for the selected row /
// manual result (its internals are covered by CausalAnalysisDetail.test.tsx).
vi.mock('@/components/causal/CausalAnalysisDetail', () => ({
  CausalAnalysisDetail: ({ result }: { result: { analysis_id: string } }) => (
    <div data-testid="causal-detail" data-analysis-id={result.analysis_id} />
  ),
}));

vi.mock('@/hooks/api', () => ({
  useCausalHealth: vi.fn(),
  useCausalAnalysisHistory: vi.fn(),
  useCausalVariables: vi.fn(),
  useCausalBrands: vi.fn(),
  useDiscoverEffects: vi.fn(),
  useRunCausalAgentAnalysis: vi.fn(),
  useEstimators: vi.fn(),
  useClinicalContext: vi.fn(),
}));

vi.mock('@/api/causal', () => ({
  getCausalAgentAnalysis: vi.fn(),
}));

import {
  useCausalHealth,
  useCausalAnalysisHistory,
  useCausalVariables,
  useCausalBrands,
  useDiscoverEffects,
  useRunCausalAgentAnalysis,
  useEstimators,
  useClinicalContext,
} from '@/hooks/api';
import { getCausalAgentAnalysis } from '@/api/causal';

const VARIABLES = {
  dataset: 'patient_journeys',
  treatment_candidates: ['treatment_arm', 'treatment_initiated'],
  outcome_candidates: ['persistent_180d', 'discontinued_180d'],
  covariate_candidates: ['disease_severity', 'engagement_score'],
  columns: [],
};

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
    brand: 'Kisqali',
    summary: 'treatment_arm raises persistent_180d by +0.088 — survived all robustness checks.',
    analysis_id: 'a1',
  },
  {
    treatment: 'treatment_arm',
    outcome: 'treatment_initiated',
    status: 'blocked',
    ate: -0.006,
    statistical_significance: true,
    selected_estimator: 'LinearDML',
    gate_decision: 'block',
    confidence_score: 0.4,
    impact: 0.006,
    n_rows: 1500,
    brand: 'Fabhalta',
    analysis_id: 'a3',
  },
];

const COMPLETED_JOB = {
  job_id: 'j1',
  status: 'completed',
  dataset: 'patient_journeys',
  brand: null,
  total: 2,
  completed: 2,
  effects: EFFECTS,
  note: 'ranked',
};

const DETAIL = { analysis_id: 'a1', status: 'completed' };

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

function mockDiscover(overrides: Record<string, unknown> = {}) {
  (useDiscoverEffects as ReturnType<typeof vi.fn>).mockReturnValue({
    start: vi.fn(),
    isStarting: false,
    startError: null,
    job: null,
    ...overrides,
  });
}

describe('CausalAnalysis — unified agent-led page', () => {
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
    (useClinicalContext as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined });
    (getCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockResolvedValue(DETAIL);
    mockDiscover();
  });

  it('lands on the leaderboard with an honest empty state before any run', () => {
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByText(/No discovery run yet/i)).toBeInTheDocument();
  }, 20000);

  it('offers grain + brand facets; brand defaults to all (null) for the patient grain', () => {
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByLabelText('Grain')).toBeInTheDocument();
    expect(screen.getByLabelText('Brand')).toBeInTheDocument();
    // Patient grain (patient_journeys) is the default; brand null = all brands.
    expect(useDiscoverEffects).toHaveBeenCalledWith('patient_journeys', null);
  }, 20000);

  it('renders the ranked leaderboard with the brand column and per-row summary', () => {
    mockDiscover({ job: COMPLETED_JOB });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByText('persistent_180d')).toBeInTheDocument();
    expect(screen.getByText('treatment_initiated')).toBeInTheDocument();
    // Brand surfaced per row (SSOT-derived scope).
    expect(screen.getByText('Kisqali')).toBeInTheDocument();
    expect(screen.getByText('Fabhalta')).toBeInTheDocument();
    // Plain-language summary surfaced.
    expect(screen.getByText(/raises persistent_180d by \+0\.088/)).toBeInTheDocument();
    // Honest verdicts.
    expect(screen.getByText('Proceed')).toBeInTheDocument();
    expect(screen.getByText('Blocked')).toBeInTheDocument();
  }, 20000);

  it('shows progress while the agent is validating', () => {
    mockDiscover({ job: { ...COMPLETED_JOB, status: 'running', completed: 1 } });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByText(/Validating… \(1\/2\)/)).toBeInTheDocument();
  }, 20000);

  it('drills a validated row into the shared deep view', async () => {
    mockDiscover({ job: COMPLETED_JOB });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByText('persistent_180d'));
    const detail = await screen.findByTestId('causal-detail');
    expect(detail).toHaveAttribute('data-analysis-id', 'a1');
    expect(getCausalAgentAnalysis).toHaveBeenCalledWith('a1');
  }, 20000);

  it('shows an honest error (not an infinite spinner) when the drill-down fetch fails', async () => {
    (getCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('not found'));
    mockDiscover({ job: COMPLETED_JOB });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByText('persistent_180d'));
    expect(await screen.findByText(/Could not load this analysis/i)).toBeInTheDocument();
    expect(screen.queryByTestId('causal-detail')).not.toBeInTheDocument();
  }, 20000);

  it('keeps a "Pose your own question" panel and runs the manual agent path with it', () => {
    const mutateAsync = vi.fn().mockResolvedValue({ analysis_id: 'm1', status: 'completed' });
    (useRunCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      mutateAsync,
      isPending: false,
      isError: false,
      error: null,
    });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    // The secondary manual panel is present (its trigger), defaulting collapsed.
    expect(screen.getByRole('button', { name: /Pose your own question/i })).toBeInTheDocument();
    // Expand it, then run the manual analysis with the data-driven defaults.
    fireEvent.click(screen.getByRole('button', { name: /Pose your own question/i }));
    expect(screen.getByLabelText('Treatment variable')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Run analysis/i }));
    expect(mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        treatment_var: 'treatment_arm',
        outcome_var: 'persistent_180d',
        dataset: 'patient_journeys',
        brand: undefined,
      })
    );
  }, 20000);

  it('explains why the candidate-question set is the size it is', () => {
    mockDiscover({ job: COMPLETED_JOB });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByText(/Why these 2 questions\?/)).toBeInTheDocument();
  }, 20000);

  it('renders the live estimator-registry total on the overview card', () => {
    (useEstimators as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { estimators: [], total: 12, by_library: {} },
      isLoading: false,
      isError: false,
    });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByText('12')).toBeInTheDocument();
  }, 20000);
});
