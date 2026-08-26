/**
 * PredictiveAnalytics Page Tests
 * ==============================
 *
 * The page is DATA-DRIVEN: score a model's real holdout cohort, rank targets
 * (labeled patients/HCPs by cohort), drill into a target's SHAP, and score
 * hypothetical profiles with the explained what-if tool.
 *
 * Covers:
 * - Model selector from useModelsStatus
 * - "Score holdout cohort" -> useScoreCohort.mutate
 * - Completed cohort -> provenance banner + ranked table + distribution
 * - Entity-kind labeling (patients vs "Entity") from the cohort
 * - Ranked row click -> usePredict.mutate with that entity's raw covariates
 * - Prediction + confidence + SHAP contributions render in the drill-down
 * - Cohort-level drivers (not drill-down SHAP) feed the interpretation
 * - What-if toggle reveals explainer + curated form; runs a custom prediction;
 *   auto-generates the what-if interpretation; prefills from the selected row
 * - Loading/error/empty states
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';
import PredictiveAnalytics from './PredictiveAnalytics';

vi.mock('@/hooks/api/use-predictions', () => ({
  useModelsStatus: vi.fn(),
  useModelInfo: vi.fn(),
  usePredict: vi.fn(),
  useScoreCohort: vi.fn(),
  usePollCohortScore: vi.fn(),
}));

// The Strategic Interpretation cards' hooks come from the `@/hooks/api` barrel;
// mock them so the cards render their idle states deterministically.
// Hoisted spies so tests can assert the page resets/feeds the interpretations.
const { mockInsightMutate, mockInsightReset, mockWhatIfMutate, mockWhatIfReset } = vi.hoisted(
  () => ({
    mockInsightMutate: vi.fn(),
    mockInsightReset: vi.fn(),
    mockWhatIfMutate: vi.fn(),
    mockWhatIfReset: vi.fn(),
  })
);
vi.mock('@/hooks/api', () => ({
  usePredictiveCohortInsight: vi.fn(() => ({
    mutate: mockInsightMutate,
    isPending: false,
    error: null,
    data: undefined,
    reset: mockInsightReset,
  })),
  usePredictiveWhatIfInsight: vi.fn(() => ({
    mutate: mockWhatIfMutate,
    isPending: false,
    error: null,
    data: undefined,
    reset: mockWhatIfReset,
  })),
}));

// AG-UI readable harness (2026-08-26): capture every useCopilotReadable call so
// tests can assert WHAT the page shares with the chat agent, not just what it
// renders. The real hook is a no-op outside <CopilotKit>; the harness records.
type ReadableCall = { description: string; value: unknown; available?: 'enabled' | 'disabled' };
const readableHarness = vi.hoisted(() => ({ calls: [] as ReadableCall[] }));
// No importOriginal: the real package drags katex CSS into jsdom. The provider
// module (imported for usePageChatContext) never renders here, so stubs suffice.
vi.mock('@copilotkit/react-core', () => ({
  useCopilotReadable: (opts: ReadableCall) => {
    readableHarness.calls.push(opts);
    return undefined;
  },
  useCopilotAction: () => undefined,
  useCoAgent: () => ({ state: {}, setState: () => undefined, running: false }),
  useCopilotChat: () => ({}),
  useCopilotContext: () => ({}),
  CopilotKit: ({ children }: { children: React.ReactNode }) => children,
}));

import {
  useModelsStatus,
  useModelInfo,
  usePredict,
  useScoreCohort,
  usePollCohortScore,
} from '@/hooks/api/use-predictions';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0 },
      mutations: { retry: false },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

const mockModelsStatus = {
  total_models: 2,
  healthy_count: 2,
  unhealthy_count: 0,
  models: [
    {
      model_name: 'initiation_kisqali_goldstd_lr_v1',
      status: 'healthy',
      endpoint: 'http://localhost:8080/x',
      last_check: '2026-01-04T10:00:00Z',
    },
    {
      model_name: 'persistence_fabhalta_goldstd_lr_v1',
      status: 'healthy',
      endpoint: 'http://localhost:8080/y',
      last_check: '2026-01-04T10:00:00Z',
    },
  ],
  timestamp: '2026-01-04T10:00:00Z',
};

const mockModelInfo = {
  model_id: 'initiation_kisqali_goldstd_lr_v1:abc',
  version: '1.0.0',
  model_loaded: true,
  feature_columns: [
    'academic_hcp',
    'disease_severity',
    'geographic_region_midwest',
    'geographic_region_northeast',
    'geographic_region_south',
    'geographic_region_west',
  ],
  keep_columns: ['disease_severity', 'academic_hcp', 'geographic_region'],
};

const mockCohort = {
  job_id: 'job-1',
  status: 'completed',
  model_name: 'initiation_kisqali_goldstd_lr_v1',
  cohort: 'initiation',
  brand: 'Kisqali',
  split: 'holdout',
  out_of_sample: true,
  feature_source: 'holdout_synthetic',
  n_scored: 1234,
  top_n: 2,
  top_rows: [
    {
      entity_id: 'patient-001',
      probability: 0.91,
      covariates: { disease_severity: 8, academic_hcp: 0, geographic_region: 'south' },
    },
    {
      entity_id: 'patient-002',
      probability: 0.55,
      covariates: { disease_severity: 5, academic_hcp: 1, geographic_region: 'west' },
    },
  ],
  distribution: {
    n: 1234,
    mean: 0.6,
    bin_edges: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    bin_counts: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
  },
  error: null,
  latency_ms: 120,
};

const mockPredictionResponse = {
  model_name: 'initiation_kisqali_goldstd_lr_v1',
  prediction: 'high_risk',
  confidence: 0.87,
  feature_importance: { disease_severity: 0.4, geographic_region_south: 0.21 },
  latency_ms: 42,
  model_version: '1.0.0',
  timestamp: '2026-01-04T10:00:00Z',
};

describe('PredictiveAnalytics (cohort scoring)', () => {
  const mockPredictMutate = vi.fn();
  const mockScoreMutate = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (useModelsStatus as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockModelsStatus,
      isLoading: false,
      error: null,
    });
    (useModelInfo as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockModelInfo,
      isLoading: false,
      error: null,
    });
    (usePredict as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockPredictMutate,
      data: undefined,
      isPending: false,
      isError: false,
      error: null,
      reset: vi.fn(),
    });
    (useScoreCohort as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockScoreMutate,
      data: undefined,
      isPending: false,
      isError: false,
      error: null,
      reset: vi.fn(),
    });
    (usePollCohortScore as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined });
  });

  it('renders the page title', () => {
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    expect(screen.getByText('Predictive Analytics')).toBeInTheDocument();
  });

  it('populates the model selector from useModelsStatus', async () => {
    const user = userEvent.setup();
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('combobox', { name: /model/i }));
    await waitFor(() => {
      expect(
        screen.getByRole('option', { name: /initiation_kisqali_goldstd_lr_v1/i })
      ).toBeInTheDocument();
    });
  }, 15000);

  it('submits a cohort-scoring job when "Score holdout cohort" is clicked', async () => {
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByRole('button', { name: /score holdout cohort/i }));
    await waitFor(() => expect(mockScoreMutate).toHaveBeenCalledTimes(1));
    expect(mockScoreMutate.mock.calls[0][0]).toEqual({
      modelName: 'initiation_kisqali_goldstd_lr_v1',
      topN: 100,
    });
  });

  it('renders provenance, distribution, and ranked rows for a completed cohort', () => {
    (usePollCohortScore as ReturnType<typeof vi.fn>).mockReturnValue({ data: mockCohort });
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    // Honest provenance — out-of-sample synthetic holdout, with the count.
    expect(screen.getByText(/out-of-sample/i)).toBeInTheDocument();
    expect(screen.getByText(/synthetic data/i)).toBeInTheDocument();
    expect(screen.getByText(/1,234/)).toBeInTheDocument();
    // Ranked entities + probabilities (sorted desc by the backend).
    expect(screen.getByText('patient-001')).toBeInTheDocument();
    expect(screen.getByText('91.0%')).toBeInTheDocument();
    expect(screen.getByText('patient-002')).toBeInTheDocument();
    // Distribution summary.
    expect(screen.getByRole('img', { name: /probability distribution/i })).toBeInTheDocument();
  });

  it('drills into a ranked row -> usePredict with that entity raw covariates', async () => {
    (usePollCohortScore as ReturnType<typeof vi.fn>).mockReturnValue({ data: mockCohort });
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    fireEvent.click(screen.getByText('patient-001'));
    await waitFor(() => expect(mockPredictMutate).toHaveBeenCalledTimes(1));
    const { modelName, request } = mockPredictMutate.mock.calls[0][0];
    expect(modelName).toBe('initiation_kisqali_goldstd_lr_v1');
    expect(request.features).toEqual({
      disease_severity: 8,
      academic_hcp: 0,
      geographic_region: 'south',
    });
    expect(request.return_feature_importance).toBe(true);
    expect(request.return_probabilities).toBe(true);
  });

  it('renders prediction + confidence + SHAP contributions in the drill-down', () => {
    (usePredict as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockPredictMutate,
      data: mockPredictionResponse,
      isPending: false,
      isError: false,
      error: null,
      reset: vi.fn(),
    });
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    expect(screen.getByText(/high_risk/i)).toBeInTheDocument();
    expect(screen.getByText(/87(\.0)?%/)).toBeInTheDocument();
    expect(screen.getByText('Feature Contributions')).toBeInTheDocument();
    // SHAP rendered as RAW signed decimals (log-odds), not percentages.
    expect(screen.getByText('+0.400')).toBeInTheDocument();
    expect(screen.queryByText('+40.0%')).not.toBeInTheDocument();
  });

  it('reveals the curated what-if form behind the toggle, explains it, and runs it', async () => {
    const user = userEvent.setup();
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    // Form is NOT shown by default (cohort scoring is the primary flow).
    expect(screen.queryByLabelText(/disease_severity/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /what-if: score a hypothetical/i }));

    // Inputs/outputs are explained, including the predictive-not-causal caveat.
    expect(screen.getByText(/Inputs/)).toBeInTheDocument();
    expect(screen.getByText(/Output/)).toBeInTheDocument();
    expect(
      screen.getByText(/prediction, not a causal estimate/i)
    ).toBeInTheDocument();

    // Curated raw covariates appear: numeric inputs + categorical select.
    expect(screen.getByLabelText(/disease_severity/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/academic_hcp/i)).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: /geographic_region/i })).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/disease_severity/i), { target: { value: '5.6' } });
    fireEvent.change(screen.getByLabelText(/academic_hcp/i), { target: { value: '1' } });
    await user.click(screen.getByRole('combobox', { name: /geographic_region/i }));
    await user.click(await screen.findByRole('option', { name: /^south$/i }));

    fireEvent.click(screen.getByRole('button', { name: /run what-if/i }));
    await waitFor(() => expect(mockPredictMutate).toHaveBeenCalledTimes(1));
    const { request } = mockPredictMutate.mock.calls[0][0];
    expect(request.features).toEqual({
      disease_severity: 5.6,
      academic_hcp: 1,
      geographic_region: 'south',
    });
    expect(typeof request.features.disease_severity).toBe('number');
  }, 15000);

  it('renders backend numeric guidance (min/max/step, range placeholder, hint) on the what-if form', () => {
    (useModelInfo as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        ...mockModelInfo,
        input_fields: [
          {
            name: 'disease_severity',
            type: 'number',
            min: 0,
            max: 10,
            step: 0.1,
            hint: 'Severity scale 0–10 · cohort median ≈ 5',
          },
          {
            name: 'peer_influence_score',
            type: 'number',
            min: 0,
            step: 0.1,
            hint: 'Log-scale network centrality · observed ≈ 0.3–6.5, median ≈ 3',
          },
          { name: 'geographic_region', type: 'category', choices: ['south', 'west'] },
        ],
      },
      isLoading: false,
      error: null,
    });
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    fireEvent.click(screen.getByRole('button', { name: /what-if: score a hypothetical/i }));

    // Bounded covariate: min/max/step attrs + a range placeholder + the hint.
    const severity = screen.getByLabelText(/disease_severity/i);
    expect(severity).toHaveAttribute('min', '0');
    expect(severity).toHaveAttribute('max', '10');
    expect(severity).toHaveAttribute('step', '0.1');
    expect(severity).toHaveAttribute('placeholder', '0–10');
    expect(screen.getByText(/Severity scale 0–10 · cohort median ≈ 5/)).toBeInTheDocument();

    // Unbounded (log-normal) covariate: floor only, generic placeholder, observed-range hint.
    const influence = screen.getByLabelText(/peer_influence_score/i);
    expect(influence).toHaveAttribute('min', '0');
    expect(influence).not.toHaveAttribute('max');
    expect(influence).toHaveAttribute('placeholder', 'Enter peer_influence_score');
    expect(screen.getByText(/observed ≈ 0.3–6.5, median ≈ 3/)).toBeInTheDocument();
  });

  it('auto-generates the what-if interpretation from the returned score + SHAP', async () => {
    const user = userEvent.setup();
    (usePollCohortScore as ReturnType<typeof vi.fn>).mockReturnValue({ data: mockCohort });
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    fireEvent.click(screen.getByRole('button', { name: /what-if: score a hypothetical/i }));
    fireEvent.change(screen.getByLabelText(/disease_severity/i), { target: { value: '5.6' } });
    fireEvent.change(screen.getByLabelText(/academic_hcp/i), { target: { value: '1' } });
    await user.click(screen.getByRole('combobox', { name: /geographic_region/i }));
    await user.click(await screen.findByRole('option', { name: /^south$/i }));
    fireEvent.click(screen.getByRole('button', { name: /run what-if/i }));
    await waitFor(() => expect(mockPredictMutate).toHaveBeenCalledTimes(1));

    // The page passes an onSuccess handler; simulate the predict result landing.
    const options = mockPredictMutate.mock.calls[0][1];
    options.onSuccess({
      ...mockPredictionResponse,
      probabilities: { positive_class: 0.83 },
      feature_importance: { disease_severity: 0.4, geographic_region_south: -0.1 },
    });

    expect(mockWhatIfMutate).toHaveBeenCalledTimes(1);
    const req = mockWhatIfMutate.mock.calls[0][0];
    expect(req.model_version).toBe('initiation_kisqali_goldstd_lr_v1');
    expect(req.probability).toBe(0.83);
    expect(req.features).toEqual({
      disease_severity: 5.6,
      academic_hcp: 1,
      geographic_region: 'south',
    });
    // Cohort context rides along so the read can compare vs the mean.
    expect(req.cohort_mean).toBe(0.6);
    expect(req.n_scored).toBe(1234);
    expect(req.top_drivers[0]).toEqual({ feature: 'disease_severity', importance: 0.4 });
  }, 15000);

  it('labels targets by entity kind (patients) derived from the cohort', () => {
    (usePollCohortScore as ReturnType<typeof vi.fn>).mockReturnValue({ data: mockCohort });
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    // Ranked table header names the kind, not "Entity".
    expect(screen.getByText(/patient \(top 2\)/i)).toBeInTheDocument();
    expect(screen.queryByText(/entity \(top/i)).not.toBeInTheDocument();
    // Card description names cohort + kind + outcome.
    expect(
      screen.getByText(/patients ranked by probability of starting treatment/i)
    ).toBeInTheDocument();
  });

  it('titles the drill-down by entity kind after selecting a ranked row', async () => {
    (usePollCohortScore as ReturnType<typeof vi.fn>).mockReturnValue({ data: mockCohort });
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByText('patient-001'));
    expect(await screen.findByText(/^Patient patient-001$/)).toBeInTheDocument();
  });

  it('feeds the cohort-level drivers (not drill-down SHAP) to the interpretation', async () => {
    (usePollCohortScore as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        ...mockCohort,
        top_drivers: [
          { feature: 'disease_severity', importance: 1.21, direction: 'increases' },
          { feature: 'insurance_type_commercial', importance: 0.4, direction: 'mixed' },
        ],
        drivers_from_top_n: 2,
      },
    });
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    fireEvent.click(screen.getByRole('button', { name: /generate strategic insight/i }));
    await waitFor(() => expect(mockInsightMutate).toHaveBeenCalledTimes(1));
    expect(mockInsightMutate.mock.calls[0][0].top_drivers).toEqual([
      { feature: 'disease_severity', importance: 1.21 },
      { feature: 'insurance_type_commercial', importance: 0.4 },
    ]);
  });

  it('prefills the what-if form from the selected ranked row', async () => {
    (usePollCohortScore as ReturnType<typeof vi.fn>).mockReturnValue({ data: mockCohort });
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    fireEvent.click(screen.getByText('patient-001'));
    fireEvent.click(screen.getByRole('button', { name: /what-if: score a hypothetical/i }));
    fireEvent.click(screen.getByRole('button', { name: /start from patient-001/i }));

    expect(screen.getByLabelText(/disease_severity/i)).toHaveValue(8);
    expect(screen.getByLabelText(/academic_hcp/i)).toHaveValue(0);
  });

  it('shows a loading indicator while useModelsStatus is loading', () => {
    (useModelsStatus as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    });
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it('renders a QueryErrorState when useModelsStatus errors', () => {
    (useModelsStatus as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('Failed to fetch models'),
      refetch: vi.fn(),
      isFetching: false,
    });
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    expect(screen.getByText('Failed to fetch models')).toBeInTheDocument();
  });

  it('surfaces a failed cohort job honestly', () => {
    (usePollCohortScore as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { ...mockCohort, status: 'failed', error: 'Feature store returned incomplete features', top_rows: [], distribution: null },
    });
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    expect(screen.getByText(/cohort scoring failed/i)).toBeInTheDocument();
    expect(screen.getByText(/incomplete features/i)).toBeInTheDocument();
  });

  it('always renders the Strategic Interpretation card', async () => {
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    expect(await screen.findByText(/strategic interpretation/i)).toBeInTheDocument();
  });

  it('resets the strategic interpretation when the cohort is re-scored', async () => {
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    mockInsightReset.mockClear();
    fireEvent.click(screen.getByRole('button', { name: /score holdout cohort/i }));
    await waitFor(() => expect(mockScoreMutate).toHaveBeenCalledTimes(1));
    expect(mockInsightReset).toHaveBeenCalled();
  });

  it('resets the strategic interpretation when the model changes', async () => {
    const user = userEvent.setup();
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    // Wait for the initial auto-select to settle, then clear the mount-time reset.
    await screen.findAllByText('initiation_kisqali_goldstd_lr_v1');
    mockInsightReset.mockClear();
    await user.click(screen.getByRole('combobox', { name: /model/i }));
    await user.click(
      await screen.findByRole('option', { name: /persistence_fabhalta_goldstd_lr_v1/i })
    );
    await waitFor(() => expect(mockInsightReset).toHaveBeenCalled());
  }, 15000);

  it('renders in-row probability bars with a cohort-mean legend', () => {
    (usePollCohortScore as ReturnType<typeof vi.fn>).mockReturnValue({ data: mockCohort });
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    expect(screen.getByText(/the tick marks the\s+cohort mean \(60\.0%\)/i)).toBeInTheDocument();
  });

  it('renders gracefully when the models list is empty', () => {
    (useModelsStatus as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { ...mockModelsStatus, models: [], total_models: 0 },
      isLoading: false,
      error: null,
    });
    (useModelInfo as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: null,
    });
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    expect(screen.getByText(/no models available/i)).toBeInTheDocument();
  });

  // 2026-08-26 (trace session_1787762049084_4psbqsx): asked "how many ranked
  // targets are above 90%?" then "the data is on the GUI", the chat had no
  // idea — the page published only a 3-line opener-pill summary (to
  // POST /chat/suggestions), never a readable. The AG-UI channel for "what the
  // user is looking at" is useCopilotReadable → agent/run body.context.
  describe('AG-UI cohort readable', () => {
    const lastCohortReadable = () => {
      const cohortCalls = readableHarness.calls.filter((c) => /cohort/i.test(c.description));
      return cohortCalls[cohortCalls.length - 1];
    };

    beforeEach(() => {
      readableHarness.calls.length = 0;
    });

    it('publishes the scored cohort — ranked rows + the FULL-cohort histogram — as a readable', () => {
      (usePollCohortScore as ReturnType<typeof vi.fn>).mockReturnValue({ data: mockCohort });
      render(<PredictiveAnalytics />, { wrapper: createWrapper() });

      const readable = lastCohortReadable();
      expect(readable).toBeDefined();
      expect(readable!.available).toBe('enabled');
      // The description must tell the model the histogram covers ALL scored
      // rows (that is how "how many above 90%" is answerable beyond the table).
      expect(readable!.description).toMatch(/full|all/i);

      const value = readable!.value as Record<string, unknown>;
      expect(value.model_name).toBe('initiation_kisqali_goldstd_lr_v1');
      expect(value.cohort_job_id).toBe('job-1');
      expect(value.brand).toBe('Kisqali');
      expect(value.n_scored).toBe(1234);
      expect(value.top_rows_shown).toBe(2);
      expect(value.distribution).toEqual(mockCohort.distribution);
      expect(value.top_rows).toEqual([
        { rank: 1, entity_id: 'patient-001', probability: 0.91 },
        { rank: 2, entity_id: 'patient-002', probability: 0.55 },
      ]);
      // Raw covariates are per-row drill-down payload — never on the wire.
      expect(JSON.stringify(value)).not.toContain('covariates');
    });

    it('marks the readable unavailable until a cohort is scored', () => {
      render(<PredictiveAnalytics />, { wrapper: createWrapper() });
      const readable = lastCohortReadable();
      expect(readable).toBeDefined();
      expect(readable!.available).toBe('disabled');
    });

    it('does not share a failed job as if it were a scored cohort', () => {
      (usePollCohortScore as ReturnType<typeof vi.fn>).mockReturnValue({
        data: { ...mockCohort, status: 'failed', error: 'boom', top_rows: [], distribution: null },
      });
      render(<PredictiveAnalytics />, { wrapper: createWrapper() });
      expect(lastCohortReadable()!.available).toBe('disabled');
    });
  });
});
