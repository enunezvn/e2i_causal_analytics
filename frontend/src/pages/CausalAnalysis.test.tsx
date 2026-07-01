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
import userEvent from '@testing-library/user-event';
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
  useCausalDiscoveryInsight: vi.fn(),
  useRunCausalAgentAnalysis: vi.fn(),
  useEstimators: vi.fn(),
  useClinicalContext: vi.fn(),
  useTreatmentEffects: vi.fn(),
  useTreatmentEffectInsight: vi.fn(),
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
  useCausalDiscoveryInsight,
  useRunCausalAgentAnalysis,
  useEstimators,
  useClinicalContext,
  useTreatmentEffects,
  useTreatmentEffectInsight,
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
      reset: vi.fn(),
      isPending: false,
      isError: false,
      error: null,
    });
    (useCausalDiscoveryInsight as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
      error: null,
      data: undefined,
    });
    (useClinicalContext as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined });
    (useTreatmentEffects as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isFetching: false,
      isError: false,
      error: null,
    });
    (useTreatmentEffectInsight as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
      error: null,
      data: undefined,
    });
    (getCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockResolvedValue(DETAIL);
    mockDiscover();
  });

  it('lands on the leaderboard with an honest empty state before any run', () => {
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByText(/No discovery run yet/i)).toBeInTheDocument();
  }, 20000);

  it('renders the strategic interpretation card on the (default) leaderboard tab', async () => {
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    // The shared StrategicInsightCard always renders its "Strategic Interpretation"
    // header on the landing (Leaderboard) tab, even before a discovery run.
    expect(await screen.findByText(/strategic interpretation/i)).toBeInTheDocument();
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
      reset: vi.fn(),
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

  // ── T4: HCP + Trigger grains are now unlocked (backend specs + causal_paths
  // rows shipped; the FE gate that hid them is stale). Selecting either grain
  // must route the discover-effects job to its dataset, and no grain may render
  // a "(coming soon)" / disabled affordance.
  it('unlocks the HCP grain — selecting it discovers effects on hcp_adoption', async () => {
    const user = userEvent.setup();
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('combobox', { name: 'Grain' }));
    await user.click(await screen.findByRole('option', { name: 'HCP' }));
    expect(useDiscoverEffects).toHaveBeenCalledWith('hcp_adoption', null);
  }, 20000);

  it('unlocks the Trigger grain — selecting it discovers effects on nba_triggers', async () => {
    const user = userEvent.setup();
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('combobox', { name: 'Grain' }));
    await user.click(await screen.findByRole('option', { name: 'Trigger' }));
    expect(useDiscoverEffects).toHaveBeenCalledWith('nba_triggers', null);
  }, 20000);

  it('shows no "(coming soon)" / not-wired note now that every grain is live', async () => {
    const user = userEvent.setup();
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('combobox', { name: 'Grain' }));
    // Every grain option is selectable — no disabled "(coming soon)" affordance.
    expect(screen.queryByText(/coming soon/i)).not.toBeInTheDocument();
    for (const opt of screen.getAllByRole('option')) {
      expect(opt).not.toHaveAttribute('aria-disabled', 'true');
    }
    // The verbose "loader is not wired yet — arrives in a later phase" note is gone.
    expect(screen.queryByText(/not wired yet/i)).not.toBeInTheDocument();
  }, 20000);

  // ── T4 (codex Finding 1): a drilled-into deep view belongs to the (dataset,
  // brand)-scoped leaderboard that produced it; switching grain must close it so
  // a Patient analysis never lingers under the HCP grain. (The leaderboard reset
  // itself lives in useDiscoverEffects — covered by use-causal.test.ts.)
  it('closes an open deep view when the grain changes (no stale cross-grain analysis)', async () => {
    const user = userEvent.setup();
    mockDiscover({ job: COMPLETED_JOB });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByText('persistent_180d'));
    expect(await screen.findByTestId('causal-detail')).toBeInTheDocument();
    await user.click(screen.getByRole('combobox', { name: 'Grain' }));
    await user.click(await screen.findByRole('option', { name: 'HCP' }));
    expect(screen.queryByTestId('causal-detail')).not.toBeInTheDocument();
  }, 20000);

  // ── T4 (codex Finding 2): the manual panel's treatment/outcome default to
  // Patient values; the candidate sets are dataset-specific, so a grain switch
  // must clamp any now-invalid selection to a valid candidate — else the manual
  // run submits a column the backend allowlist rejects (400).
  it('clamps the manual treatment/outcome to the new grain’s candidates on switch', async () => {
    const user = userEvent.setup();
    const TRIGGER_VARIABLES = {
      dataset: 'nba_triggers',
      treatment_candidates: ['control_group_flag', 'acceptance_status'],
      outcome_candidates: ['action_taken', 'conversion_flag'],
      covariate_candidates: [],
      columns: [],
    };
    (useCausalVariables as ReturnType<typeof vi.fn>).mockImplementation((ds: string) => ({
      data: ds === 'nba_triggers' ? TRIGGER_VARIABLES : VARIABLES,
    }));
    const mutateAsync = vi.fn().mockResolvedValue({ analysis_id: 'm1', status: 'completed' });
    (useRunCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      mutateAsync,
      reset: vi.fn(),
      isPending: false,
      isError: false,
      error: null,
    });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    // Switch to the Trigger grain (Patient defaults treatment_arm/persistent_180d
    // are both invalid here).
    await user.click(screen.getByRole('combobox', { name: 'Grain' }));
    await user.click(await screen.findByRole('option', { name: 'Trigger' }));
    // Open the manual panel and run — the payload must carry Trigger-valid columns.
    fireEvent.click(screen.getByRole('button', { name: /Pose your own question/i }));
    fireEvent.click(screen.getByRole('button', { name: /Run analysis/i }));
    expect(mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        dataset: 'nba_triggers',
        treatment_var: 'control_group_flag',
        outcome_var: 'action_taken',
      })
    );
  }, 20000);

  // ── T4 (codex round 2/3, Finding 2): a completed manual analysis is scoped to
  // the (dataset, brand) it was RUN for; switching grain or brand must not leave
  // it rendered (mislabeled with the new facets) under the new scope.
  function mockCompletedManualRun() {
    (useRunCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { analysis_id: 'm1', status: 'completed', dataset: 'patient_journeys' },
      mutateAsync: vi.fn().mockResolvedValue({ analysis_id: 'm1', status: 'completed' }),
      reset: vi.fn(),
      isPending: false,
      isError: false,
      error: null,
    });
  }

  it('drops a completed manual analysis when the grain changes (no stale cross-grain result)', async () => {
    const user = userEvent.setup();
    mockCompletedManualRun();
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    // Run a manual analysis on the Patient grain (tags the submitted scope).
    fireEvent.click(screen.getByRole('button', { name: /Pose your own question/i }));
    fireEvent.click(screen.getByRole('button', { name: /Run analysis/i }));
    expect(await screen.findByTestId('causal-detail')).toHaveAttribute('data-analysis-id', 'm1');
    // Switch grain → the Patient-scoped manual result must not linger under HCP.
    await user.click(screen.getByRole('combobox', { name: 'Grain' }));
    await user.click(await screen.findByRole('option', { name: 'HCP' }));
    expect(screen.queryByTestId('causal-detail')).not.toBeInTheDocument();
  }, 20000);

  it('drops a completed manual analysis when the brand changes (same-dataset scope)', async () => {
    const user = userEvent.setup();
    mockCompletedManualRun();
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    // Run on all-brands (brandArg null), then switch to a specific brand.
    fireEvent.click(screen.getByRole('button', { name: /Pose your own question/i }));
    fireEvent.click(screen.getByRole('button', { name: /Run analysis/i }));
    expect(await screen.findByTestId('causal-detail')).toHaveAttribute('data-analysis-id', 'm1');
    await user.click(screen.getByRole('combobox', { name: 'Brand' }));
    await user.click(await screen.findByRole('option', { name: 'Kisqali' }));
    expect(screen.queryByTestId('causal-detail')).not.toBeInTheDocument();
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

  it('renders a Treatment effects tab with the cohort×brand ATE card', async () => {
    const user = userEvent.setup();
    render(<CausalAnalysis />, { wrapper: createWrapper() });

    const tab = await screen.findByRole('tab', { name: /treatment effects/i });
    await user.click(tab);

    expect(await screen.findByText(/Treatment Effect by Cohort/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/^Cohort$/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/^Brand$/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /run estimate/i })).toBeInTheDocument();
  }, 20000);

  it('auto-generates the treatment-effect strategic insight once a result lands', async () => {
    const mutate = vi.fn();
    (useTreatmentEffectInsight as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate,
      isPending: false,
      error: null,
      data: undefined,
    });
    (useTreatmentEffects as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        cohort: 'hcp_adoption',
        brand: 'Remibrutinib',
        treatment_var: 'treatment_arm',
        outcome_var: 'adopted',
        confounders: ['peer_influence_score', 'influence_network_size'],
        ate: 0.1448,
        ci_lower: 0.1426,
        ci_upper: 0.147,
        p_value: 0.0004,
        std_error: 0.001,
        n: 5000,
        estimator: 'linear_dml',
        method: 'dowhy+econml sequential',
        confidence_level: 0.95,
        latency_ms: 40000,
        is_synthetic: true,
        warnings: ['robustness not validated'],
      },
      isFetching: false,
      isError: false,
      error: null,
    });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    await userEvent.click(screen.getByRole('tab', { name: /Treatment effects/i }));
    expect(mutate).toHaveBeenCalledTimes(1);
    expect(mutate).toHaveBeenCalledWith(
      expect.objectContaining({
        cohort: 'hcp_adoption',
        brand: 'Remibrutinib',
        ate: 0.1448,
        n: 5000,
      })
    );
  }, 20000);
});
