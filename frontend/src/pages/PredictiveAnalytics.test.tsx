/**
 * PredictiveAnalytics Page Tests
 * ==============================
 *
 * The page is DATA-DRIVEN: score a model's real holdout cohort, rank targets,
 * drill into an entity's SHAP, with an "Advanced what-if" custom row preserved.
 *
 * Covers:
 * - Model selector from useModelsStatus
 * - "Score holdout cohort" -> useScoreCohort.mutate
 * - Completed cohort -> provenance banner + ranked table + distribution
 * - Ranked row click -> usePredict.mutate with that entity's raw covariates
 * - Prediction + confidence + SHAP contributions render in the drill-down
 * - Advanced what-if toggle reveals the curated form + runs a custom prediction
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

// The Strategic Interpretation card's hook comes from the `@/hooks/api` barrel;
// mock it so the card renders its idle "generate" state deterministically.
// Hoisted spies so tests can assert the page resets the interpretation.
const { mockInsightMutate, mockInsightReset } = vi.hoisted(() => ({
  mockInsightMutate: vi.fn(),
  mockInsightReset: vi.fn(),
}));
vi.mock('@/hooks/api', () => ({
  usePredictiveCohortInsight: vi.fn(() => ({
    mutate: mockInsightMutate,
    isPending: false,
    error: null,
    data: undefined,
    reset: mockInsightReset,
  })),
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

  it('reveals the curated what-if form behind the Advanced toggle and runs it', async () => {
    const user = userEvent.setup();
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    // Form is NOT shown by default (cohort scoring is the primary flow).
    expect(screen.queryByLabelText(/disease_severity/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /advanced.*what-if/i }));

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
});
