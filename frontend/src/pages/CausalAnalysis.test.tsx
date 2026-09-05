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
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import CausalAnalysis from './CausalAnalysis';
import { ApiError } from '@/lib/api-client';
import { renderWithAllProviders } from '@/test/utils';
import {
  CopilotKitWrapper,
  E2ICopilotProvider,
  useE2ICopilot,
} from '@/providers/E2ICopilotProvider';

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
  useDiscoverQuestions: vi.fn(),
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
  useDiscoverQuestions,
  useDiscoverEffects,
  useCausalDiscoveryInsight,
  useRunCausalAgentAnalysis,
  useEstimators,
  useClinicalContext,
  useTreatmentEffects,
  useTreatmentEffectInsight,
} from '@/hooks/api';
import { getCausalAgentAnalysis } from '@/api/causal';

// Biomarker union mirroring the backend's clinical_biomarkers response field
// (brand-independent; classifies covariates for the display split).
const BIOMARKERS = [
  'ecog_performance_status',
  'egfr',
  'ldh_ratio',
  'proteinuria_g_day',
  'urticaria_severity_uas7',
];

const VARIABLES = {
  dataset: 'patient_journeys',
  treatment_candidates: ['treatment_arm', 'treatment_initiated', 'sample_dropped'],
  outcome_candidates: ['persistent_180d', 'discontinued_180d'],
  covariate_candidates: ['disease_severity', 'engagement_score'],
  columns: [],
  clinical_biomarkers: BIOMARKERS,
  // Curated display labels (causal._COLUMN_LABELS) — the SSOT /segment-analysis
  // already renders; this page must render the same names.
  labels: {
    treatment_arm: 'Treatment arm',
    treatment_initiated: 'Treatment initiated',
    persistent_180d: 'Persistent at 180d',
    discontinued_180d: 'Discontinued at 180d',
    sample_dropped: 'Product samples provided (rep sample drop)',
  },
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
    summary: 'Treatment arm raises Persistent at 180d by +0.088 — survived all robustness checks.',
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

// The scope's SSOT candidate questions (GET /causal/discover-effects/questions),
// already labelled — what the Questions selector lists.
const QUESTIONS = {
  dataset: 'patient_journeys',
  brand: null,
  questions: [
    {
      treatment: 'treatment_arm',
      outcome: 'persistent_180d',
      brand: 'Kisqali',
      treatment_label: 'Treatment arm',
      outcome_label: 'Persistent at 180d',
      adjustment_set: ['disease_severity'],
    },
    {
      treatment: 'treatment_arm',
      outcome: 'treatment_initiated',
      brand: 'Fabhalta',
      treatment_label: 'Treatment arm',
      outcome_label: 'Treatment initiated',
      adjustment_set: [],
    },
    {
      treatment: 'sample_dropped',
      outcome: 'treatment_initiated',
      brand: 'Remibrutinib',
      treatment_label: 'Product samples provided (rep sample drop)',
      outcome_label: 'Treatment initiated',
      adjustment_set: [],
    },
  ],
  note: '',
};

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
    cancel: vi.fn(),
    isCancelling: false,
    cancelError: null,
    cancelRequested: false,
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
    (useDiscoverQuestions as ReturnType<typeof vi.fn>).mockReturnValue({
      data: QUESTIONS,
      isLoading: false,
      isError: false,
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
      reset: vi.fn(),
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

  // Strategic Interpretation gating + reset (2026-07-23 frontend review): the
  // interpretation must be grounded in a completed discovery run, reset on a
  // grain/brand switch (no stale text), and auto-regenerate when a fresh run
  // completes.
  describe('strategic interpretation gating + reset', () => {
    it('disables generation until a discovery run has completed', async () => {
      mockDiscover(); // job: null — nothing discovered yet
      render(<CausalAnalysis />, { wrapper: createWrapper() });
      const btn = await screen.findByRole('button', { name: /generate strategic insight/i });
      expect(btn).toBeDisabled();
      expect(screen.getByText(/run discover causal effects/i)).toBeInTheDocument();
    }, 20000);

    it('auto-generates the interpretation once discovery completes, grounded in the effects', async () => {
      const mutate = vi.fn();
      (useCausalDiscoveryInsight as ReturnType<typeof vi.fn>).mockReturnValue({
        mutate,
        reset: vi.fn(),
        isPending: false,
        error: null,
        data: undefined,
      });
      mockDiscover({ job: COMPLETED_JOB });
      render(<CausalAnalysis />, { wrapper: createWrapper() });
      await waitFor(() => expect(mutate).toHaveBeenCalledTimes(1));
      expect(mutate).toHaveBeenCalledWith(
        expect.objectContaining({
          brand: 'All brands',
          grain: 'patient',
          effects: expect.arrayContaining([
            expect.objectContaining({ treatment: 'treatment_arm', outcome: 'persistent_180d' }),
          ]),
        })
      );
    }, 20000);

    it('resets the interpretation when the brand changes (no stale text)', async () => {
      const user = userEvent.setup();
      const reset = vi.fn();
      (useCausalDiscoveryInsight as ReturnType<typeof vi.fn>).mockReturnValue({
        mutate: vi.fn(),
        reset,
        isPending: false,
        error: null,
        data: { insight: 'stale brand-A read', key_takeaways: [], grounding: [], is_fallback: false },
      });
      mockDiscover({ job: COMPLETED_JOB });
      render(<CausalAnalysis />, { wrapper: createWrapper() });
      reset.mockClear(); // ignore the reset that fires on initial mount
      await user.click(screen.getByRole('combobox', { name: 'Brand' }));
      await user.click(await screen.findByRole('option', { name: 'Kisqali' }));
      await waitFor(() => expect(reset).toHaveBeenCalled());
    }, 20000);

    it('never renders an interpretation whose submitted scope is not the active one', () => {
      // A response sits in the mutation cache but no run was submitted for the
      // ACTIVE scope (e.g. an auto-regenerated scope-A call resolving AFTER the
      // user moved on — reset() cleared local data, but the late onSuccess
      // repopulated it). The scope tag no longer matches, so it must be
      // suppressed, not shown — and the honest disabled state shows instead.
      (useCausalDiscoveryInsight as ReturnType<typeof vi.fn>).mockReturnValue({
        mutate: vi.fn(),
        reset: vi.fn(),
        isPending: false,
        error: null,
        data: {
          insight: 'STALE CROSS-SCOPE READ',
          key_takeaways: [],
          grounding: [],
          is_fallback: false,
        },
      });
      mockDiscover(); // job: null → no run tags the active scope
      render(<CausalAnalysis />, { wrapper: createWrapper() });
      expect(screen.queryByText(/stale cross-scope read/i)).not.toBeInTheDocument();
      expect(
        screen.getByRole('button', { name: /generate strategic insight/i })
      ).toBeDisabled();
    }, 20000);
  });

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
    expect(screen.getByText('Persistent at 180d')).toBeInTheDocument();
    expect(screen.getByText('Treatment initiated')).toBeInTheDocument();
    // Brand surfaced per row (SSOT-derived scope).
    expect(screen.getByText('Kisqali')).toBeInTheDocument();
    expect(screen.getByText('Fabhalta')).toBeInTheDocument();
    // Plain-language summary surfaced verbatim (the backend labels it; the row
    // must not re-expose the raw column names anywhere).
    expect(screen.getByText(/Treatment arm raises Persistent at 180d by \+0\.088/)).toBeInTheDocument();
    expect(screen.queryByText(/persistent_180d/)).not.toBeInTheDocument();
    // Honest verdicts.
    expect(screen.getByText('Proceed')).toBeInTheDocument();
    expect(screen.getByText('Blocked')).toBeInTheDocument();
  }, 20000);

  it('renders the curated column labels — never raw column names — on the leaderboard, drill-down title and manual dropdowns', async () => {
    // 2026-09-05: /segment-analysis relabelled sample_dropped (#1893) but this
    // page still printed the raw column — GET /causal/variables already served
    // the same `labels` map; the page never consumed it.
    const user = userEvent.setup();
    const job = {
      ...COMPLETED_JOB,
      total: 3,
      effects: [
        ...EFFECTS,
        {
          treatment: 'sample_dropped',
          outcome: 'treatment_initiated',
          status: 'pending',
          statistical_significance: false,
          confidence_score: 0,
          n_rows: 0,
          brand: 'Remibrutinib',
        },
      ],
    };
    mockDiscover({ job });
    (getCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockResolvedValue({
      ...DETAIL,
      treatment_var: 'treatment_arm',
      outcome_var: 'persistent_180d',
    });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    // Leaderboard row.
    expect(screen.getByText('Product samples provided (rep sample drop)')).toBeInTheDocument();
    expect(screen.queryByText('sample_dropped')).not.toBeInTheDocument();
    // Drill-down title.
    fireEvent.click(screen.getByText('Persistent at 180d'));
    expect(
      await screen.findByText(/Treatment arm\s*→\s*Persistent at 180d/)
    ).toBeInTheDocument();
    // Manual panel dropdown options read the same labels.
    fireEvent.click(screen.getByRole('button', { name: /Pose your own question/i }));
    await user.click(screen.getByRole('combobox', { name: 'Treatment variable' }));
    expect(
      await screen.findByRole('option', { name: 'Product samples provided (rep sample drop)' })
    ).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'sample_dropped' })).not.toBeInTheDocument();
    await user.keyboard('{Escape}');
    await user.click(screen.getByRole('combobox', { name: 'Outcome variable' }));
    expect(await screen.findByRole('option', { name: 'Persistent at 180d' })).toBeInTheDocument();
  }, 20000);

  it('shows progress while the agent is validating', () => {
    mockDiscover({ job: { ...COMPLETED_JOB, status: 'running', completed: 1 } });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    expect(screen.getByText(/Validating… \(1\/2\)/)).toBeInTheDocument();
  }, 20000);

  // CAUSAL-DISC-UX: run only the questions of interest, and stop a run early.
  describe('question subset + cancel', () => {
    it('runs every candidate by default — no subset on the wire', async () => {
      const start = vi.fn();
      mockDiscover({ start });
      render(<CausalAnalysis />, { wrapper: createWrapper() });
      expect(useDiscoverQuestions).toHaveBeenCalledWith('patient_journeys', null);
      expect(screen.getByRole('combobox', { name: 'Questions to discover' })).toHaveTextContent(
        'All 3 questions'
      );
      fireEvent.click(screen.getByRole('button', { name: /Discover causal effects/i }));
      expect(start).toHaveBeenCalledTimes(1);
      expect(start).toHaveBeenCalledWith();
    }, 20000);

    it('runs only the checked questions, sent as SSOT (treatment, outcome, brand) rows', async () => {
      const user = userEvent.setup();
      const start = vi.fn();
      mockDiscover({ start });
      render(<CausalAnalysis />, { wrapper: createWrapper() });
      await user.click(screen.getByRole('combobox', { name: 'Questions to discover' }));
      // Curated labels in the selector, never the raw column.
      expect(await screen.findByLabelText(/Product samples provided/)).toBeInTheDocument();
      expect(screen.queryByText('sample_dropped')).not.toBeInTheDocument();
      await user.click(screen.getByLabelText(/Product samples provided/));
      await user.keyboard('{Escape}');
      expect(screen.getByRole('combobox', { name: 'Questions to discover' })).toHaveTextContent(
        '2 of 3 questions'
      );
      fireEvent.click(screen.getByRole('button', { name: /Discover causal effects/i }));
      expect(start).toHaveBeenCalledWith([
        { treatment: 'treatment_arm', outcome: 'persistent_180d', brand: 'Kisqali' },
        { treatment: 'treatment_arm', outcome: 'treatment_initiated', brand: 'Fabhalta' },
      ]);
    }, 20000);

    it('will not start a run with no question checked', async () => {
      const user = userEvent.setup();
      const start = vi.fn();
      mockDiscover({ start });
      render(<CausalAnalysis />, { wrapper: createWrapper() });
      await user.click(screen.getByRole('combobox', { name: 'Questions to discover' }));
      await user.click(await screen.findByRole('button', { name: 'Clear' }));
      await user.keyboard('{Escape}');
      expect(screen.getByRole('combobox', { name: 'Questions to discover' })).toHaveTextContent(
        'No questions selected'
      );
      const discover = screen.getByRole('button', { name: /Discover causal effects/i });
      expect(discover).toBeDisabled();
      fireEvent.click(discover);
      expect(start).not.toHaveBeenCalled();
    }, 20000);

    it('holds Discover until the candidate list has loaded (a click must not start a full run the user meant to narrow)', () => {
      // codex iter-1 MEDIUM: with the list still loading, `candidateQuestions` is
      // [] so "all selected" is vacuously true and a click would submit every
      // candidate before the user could pick a subset.
      (useDiscoverQuestions as ReturnType<typeof vi.fn>).mockReturnValue({
        data: undefined,
        isLoading: true,
        isError: false,
      });
      const start = vi.fn();
      mockDiscover({ start });
      render(<CausalAnalysis />, { wrapper: createWrapper() });
      expect(screen.getByRole('combobox', { name: 'Questions to discover' })).toHaveTextContent(
        'Loading questions…'
      );
      const discover = screen.getByRole('button', { name: /Discover causal effects/i });
      expect(discover).toBeDisabled();
      fireEvent.click(discover);
      expect(start).not.toHaveBeenCalled();
    }, 20000);

    it('still lets the run start when the candidate list could not load (every candidate runs)', () => {
      (useDiscoverQuestions as ReturnType<typeof vi.fn>).mockReturnValue({
        data: undefined,
        isLoading: false,
        isError: true,
      });
      const start = vi.fn();
      mockDiscover({ start });
      render(<CausalAnalysis />, { wrapper: createWrapper() });
      expect(screen.getByRole('combobox', { name: 'Questions to discover' })).toBeDisabled();
      fireEvent.click(screen.getByRole('button', { name: /Discover causal effects/i }));
      expect(start).toHaveBeenCalledWith();
    }, 20000);

    it('offers Cancel while validating and reports the request until the run stops', () => {
      const cancel = vi.fn();
      mockDiscover({ job: { ...COMPLETED_JOB, status: 'running', completed: 1 }, cancel });
      const { rerender } = render(<CausalAnalysis />, { wrapper: createWrapper() });
      // The selector is locked while a run is in flight.
      expect(screen.getByRole('combobox', { name: 'Questions to discover' })).toBeDisabled();
      fireEvent.click(screen.getByRole('button', { name: /^Cancel$/ }));
      expect(cancel).toHaveBeenCalledTimes(1);

      mockDiscover({
        job: { ...COMPLETED_JOB, status: 'running', completed: 1 },
        cancel,
        cancelRequested: true,
      });
      rerender(<CausalAnalysis />);
      const stopping = screen.getByRole('button', { name: /Stopping after current question/ });
      expect(stopping).toBeDisabled();
      expect(screen.queryByRole('button', { name: /^Cancel$/ })).not.toBeInTheDocument();
    }, 20000);

    it('renders a cancelled run honestly: kept rows, cancelled rows, no fabricated estimates', () => {
      mockDiscover({
        job: {
          ...COMPLETED_JOB,
          status: 'cancelled',
          completed: 1,
          cancel_requested: true,
          effects: [
            EFFECTS[0],
            {
              treatment: 'treatment_arm',
              outcome: 'treatment_initiated',
              status: 'cancelled',
              statistical_significance: false,
              confidence_score: 0,
              n_rows: 0,
              brand: 'Fabhalta',
            },
          ],
        },
      });
      render(<CausalAnalysis />, { wrapper: createWrapper() });
      expect(screen.getByText(/Stopped early — 1\/2 questions validated/)).toBeInTheDocument();
      expect(screen.getByText('Cancelled')).toBeInTheDocument();
      expect(screen.getByText('Proceed')).toBeInTheDocument();
      // The run is over: no Cancel, and discovery can be re-run.
      expect(screen.queryByRole('button', { name: /Cancel|Stopping/ })).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Re-run discovery/i })).toBeEnabled();
      expect(screen.getByRole('combobox', { name: 'Questions to discover' })).toBeEnabled();
    }, 20000);

    it('surfaces a failed cancel without pretending the run stopped', () => {
      mockDiscover({
        job: { ...COMPLETED_JOB, status: 'running', completed: 1 },
        cancelError: new ApiError({
          message: 'boom',
          response: { status: 500 },
        } as unknown as ConstructorParameters<typeof ApiError>[0]),
      });
      render(<CausalAnalysis />, { wrapper: createWrapper() });
      expect(screen.getByText(/Could not cancel the run/)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /^Cancel$/ })).toBeEnabled();
    }, 20000);
  });

  it('drills a validated row into the shared deep view', async () => {
    mockDiscover({ job: COMPLETED_JOB });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByText('Persistent at 180d'));
    const detail = await screen.findByTestId('causal-detail');
    expect(detail).toHaveAttribute('data-analysis-id', 'a1');
    expect(getCausalAgentAnalysis).toHaveBeenCalledWith('a1');
  }, 20000);

  it('shows a retryable error (not an infinite spinner) when a transient drill-down fetch fails', async () => {
    // A non-404 failure (network blip, 5xx, auth lapse) is transient: the record is
    // likely still cached, so the UI offers a plain Retry instead of sending the user
    // to re-run a multi-minute discovery.
    (getCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('boom'));
    mockDiscover({ job: COMPLETED_JOB });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByText('Persistent at 180d'));
    expect(await screen.findByText(/Could not load this analysis/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Retry/i })).toBeInTheDocument();
    expect(screen.queryByTestId('causal-detail')).not.toBeInTheDocument();
  }, 20000);

  it('tells the user to re-run discovery only when the analysis genuinely expired (404)', async () => {
    // A 404 means the cached detail expired (leaderboard outlived its drill-down
    // record). THAT is the only case where re-running discovery is the right fix.
    const notFound = new ApiError({
      response: { status: 404 },
    } as unknown as ConstructorParameters<typeof ApiError>[0]);
    (getCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockRejectedValue(notFound);
    mockDiscover({ job: COMPLETED_JOB });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByText('Persistent at 180d'));
    expect(await screen.findByText(/no longer available/i)).toBeInTheDocument();
    expect(screen.getByText(/Re-run discovery to regenerate it/i)).toBeInTheDocument();
    // No blind Retry button for a genuine expiry — retrying the same id just 404s again.
    expect(screen.queryByRole('button', { name: /Retry/i })).not.toBeInTheDocument();
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

  // ── #1188: opt-in RCT baseline adjustment toggle ──────────────────────────
  it('offers the baseline-adjustment toggle only when the dataset has baselines, and posts adjust_baselines', async () => {
    const user = userEvent.setup();
    const mutateAsync = vi.fn().mockResolvedValue({ analysis_id: 'b1', status: 'completed' });
    (useRunCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      mutateAsync,
      reset: vi.fn(),
      isPending: false,
      isError: false,
      error: null,
    });
    (useCausalVariables as ReturnType<typeof vi.fn>).mockImplementation((ds: string) => ({
      data:
        ds === 'nba_triggers'
          ? {
              dataset: 'nba_triggers',
              treatment_candidates: ['control_group_flag'],
              outcome_candidates: ['action_taken'],
              covariate_candidates: [],
              baseline_candidates: ['disease_severity', 'age_at_diagnosis'],
              columns: [],
            }
          : VARIABLES,
    }));
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    // Switch grain to the Trigger RCT FIRST (existing-test pattern), then open
    // the manual panel: the toggle must be offered there.
    await user.click(screen.getByRole('combobox', { name: 'Grain' }));
    await user.click(await screen.findByRole('option', { name: 'Trigger' }));
    fireEvent.click(screen.getByRole('button', { name: /Pose your own question/i }));
    const toggle = await screen.findByLabelText(/baseline covariates/i);
    expect(toggle).toBeInTheDocument();
    // Opt in and run: the POST body carries adjust_baselines: true.
    fireEvent.click(toggle);
    fireEvent.click(screen.getByRole('button', { name: /Run analysis/i }));
    expect(mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        dataset: 'nba_triggers',
        adjust_baselines: true,
      })
    );
  }, 20000);

  it('hides the baseline toggle on datasets without a curated baseline role', () => {
    (useRunCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      mutateAsync: vi.fn(),
      reset: vi.fn(),
      isPending: false,
      isError: false,
      error: null,
    });
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    // patient_journeys (no baseline role): toggle absent.
    fireEvent.click(screen.getByRole('button', { name: /Pose your own question/i }));
    expect(screen.queryByLabelText(/baseline covariates/i)).toBeNull();
  }, 20000);

  it('defaults the baseline toggle OFF (unadjusted RCT stays the default)', async () => {
    const user = userEvent.setup();
    const mutateAsync = vi.fn().mockResolvedValue({ analysis_id: 'b2', status: 'completed' });
    (useRunCausalAgentAnalysis as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      mutateAsync,
      reset: vi.fn(),
      isPending: false,
      isError: false,
      error: null,
    });
    (useCausalVariables as ReturnType<typeof vi.fn>).mockImplementation((ds: string) => ({
      data:
        ds === 'nba_triggers'
          ? {
              dataset: 'nba_triggers',
              treatment_candidates: ['control_group_flag'],
              outcome_candidates: ['action_taken'],
              covariate_candidates: [],
              baseline_candidates: ['disease_severity', 'age_at_diagnosis'],
              columns: [],
            }
          : VARIABLES,
    }));
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('combobox', { name: 'Grain' }));
    await user.click(await screen.findByRole('option', { name: 'Trigger' }));
    fireEvent.click(screen.getByRole('button', { name: /Pose your own question/i }));
    fireEvent.click(screen.getByRole('button', { name: /Run analysis/i }));
    expect(mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({ dataset: 'nba_triggers', adjust_baselines: false })
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
    fireEvent.click(screen.getByText('Persistent at 180d'));
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

  // ── Brand-scoped covariate candidates (2026-07-13 clinical-faithfulness fix):
  // the offered adjustment set must match what estimation actually uses. The
  // variables query is keyed on the ACTIVE brand, and the manual panel labels
  // generic cross-brand confounders apart from the brand's own indication
  // biomarkers (a Fabhalta question must never show UAS7).
  it('requests brand-scoped variables and re-requests when the brand changes', async () => {
    const user = userEvent.setup();
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    // All-brands default: the query is made with brand=null (universals only).
    expect(useCausalVariables).toHaveBeenCalledWith('patient_journeys', null);
    await user.click(screen.getByRole('combobox', { name: 'Brand' }));
    await user.click(await screen.findByRole('option', { name: 'Kisqali' }));
    expect(useCausalVariables).toHaveBeenCalledWith('patient_journeys', 'Kisqali');
  }, 20000);

  it('labels the brand’s indication biomarkers apart from the generic confounders', async () => {
    const user = userEvent.setup();
    (useCausalVariables as ReturnType<typeof vi.fn>).mockImplementation(
      (_dataset: string, brand?: string | null) => ({
        data:
          brand === 'Kisqali'
            ? {
                ...VARIABLES,
                // Server-scoped: Kisqali's own biomarker only, never UAS7.
                covariate_candidates: [
                  'disease_severity',
                  'engagement_score',
                  'ecog_performance_status',
                ],
              }
            : VARIABLES,
      })
    );
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('combobox', { name: 'Brand' }));
    await user.click(await screen.findByRole('option', { name: 'Kisqali' }));
    fireEvent.click(screen.getByRole('button', { name: /Pose your own question/i }));
    // Generic universals and the brand's own biomarker are labeled apart.
    expect(screen.getByText(/generic confounders \(all brands\)/)).toBeInTheDocument();
    expect(screen.getByText(/indication-specific biomarkers \(Kisqali\)/)).toBeInTheDocument();
    expect(screen.getByText('ecog_performance_status')).toBeInTheDocument();
    // An off-brand biomarker is never displayed for this brand.
    expect(screen.queryByText(/urticaria_severity_uas7/)).not.toBeInTheDocument();
  }, 20000);

  it('shows only generic confounders (no biomarker group) for the all-brands scope', () => {
    render(<CausalAnalysis />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByRole('button', { name: /Pose your own question/i }));
    expect(screen.getByText(/generic confounders \(all brands\)/)).toBeInTheDocument();
    expect(screen.queryByText(/indication-specific biomarkers/)).not.toBeInTheDocument();
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
    const { rerender } = render(<CausalAnalysis />, { wrapper: createWrapper() });
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
    // teData is pre-loaded, so the effect fires on mount; re-rendering with the
    // same estimate must NOT re-fire the mutation (ref-guard keyed on cohort-brand-ate).
    rerender(<CausalAnalysis />);
    expect(mutate).toHaveBeenCalledTimes(1);
  }, 20000);

  // =========================================================================
  // #1752 — BRAND SELECTION IS SHARED WITH THE COPILOT CHAT (same seam as
  // Home's #1749): the page's Brand filter lived in page-local useState with
  // the '__all__' sentinel while every chat surface (pills, /chat/suggestions,
  // the AgentFiltersBridge CoAgent channel) read the copilot provider's filter
  // state. The provider is the single source of truth; the sentinel exists
  // only at the Radix Select boundary, and brandArg semantics ('All' → null
  // API arg) are unchanged.
  // =========================================================================

  describe('Brand selection ↔ copilot filter context (#1752)', () => {
    /** Reads the copilot context brand; the button simulates the chat's
     *  setBrandFilter action (same setFilters seam the action handler uses). */
    function CopilotBrandProbe() {
      const context = useE2ICopilot();
      return (
        <div>
          <span data-testid="copilot-brand">{context.filters.brand}</span>
          <button
            onClick={() => context.setFilters((prev) => ({ ...prev, brand: 'Kisqali' }))}
          >
            chat-sets-kisqali
          </button>
        </div>
      );
    }

    function renderPageWithCopilot() {
      return renderWithAllProviders(
        <CopilotKitWrapper enabled={false}>
          <E2ICopilotProvider>
            <CausalAnalysis />
            <CopilotBrandProbe />
          </E2ICopilotProvider>
        </CopilotKitWrapper>
      );
    }

    it('writes the page brand selection through to the copilot filter context', async () => {
      renderPageWithCopilot();

      fireEvent.click(screen.getByRole('combobox', { name: 'Brand' }));
      fireEvent.click(await screen.findByText('Kisqali'));

      await waitFor(() =>
        expect(screen.getByTestId('copilot-brand')).toHaveTextContent('Kisqali')
      );
    }, 20000);

    it('reflects a chat-driven brand change (setBrandFilter seam) in the page selector', async () => {
      renderPageWithCopilot();

      expect(screen.getByRole('combobox', { name: 'Brand' })).toHaveTextContent('All brands');

      fireEvent.click(screen.getByText('chat-sets-kisqali'));

      await waitFor(() =>
        expect(screen.getByRole('combobox', { name: 'Brand' })).toHaveTextContent('Kisqali')
      );
    }, 20000);

    it('re-scopes the discover-effects leaderboard from a chat-driven brand change', async () => {
      renderPageWithCopilot();

      fireEvent.click(screen.getByText('chat-sets-kisqali'));

      // brandArg derives from the shared filter: 'Kisqali', not the page-local
      // default null — the functional re-scope, not just the dropdown label.
      await waitFor(() =>
        expect(useDiscoverEffects).toHaveBeenLastCalledWith('patient_journeys', 'Kisqali')
      );
    }, 20000);
  });
});
