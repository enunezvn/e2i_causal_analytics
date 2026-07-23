/**
 * ModelPerformance Page Tests
 * ===========================
 *
 * Tests for the Model Performance analysis page.
 *
 * Per Issue #298: page must be wired to /api/monitoring/performance/*
 * - Model selector populated from useModelsStatus (live)
 * - Metric trend wired via usePerformanceTrend
 * - Performance alerts via usePerformanceAlerts
 * - Comparison via useModelComparison
 * - Loading/error states via QueryErrorState
 *
 * Hard-coded SAMPLE_MODELS / SAMPLE_METRICS arrays must be removed.
 *
 * Trend-enhancement port (from TimeSeries): the page calls usePerformanceTrend
 * THREE times unconditionally (selected model + two gold-standard sibling
 * brands), varying only `{ enabled }`. Tests must therefore identify the
 * SELECTED model's query via `mock.calls.filter((c) => c[1]?.enabled)` — the
 * "last call" is a sibling query, not the selected model.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// =============================================================================
// HOOK MOCKS — wired BEFORE page import so vi.mock can hoist
// =============================================================================

vi.mock('@/hooks/api/use-monitoring', () => ({
  usePerformanceTrend: vi.fn(),
  usePerformanceAlerts: vi.fn(),
  useModelComparison: vi.fn(),
  useConfusionMatrix: vi.fn(),
  useRocCurve: vi.fn(),
}));

vi.mock('@/hooks/api/use-predictions', () => ({
  useModelsStatus: vi.fn(),
}));

vi.mock('@/hooks/api/use-kpi', () => ({
  useKPIList: vi.fn(),
}));

import ModelPerformance from './ModelPerformance';
import {
  usePerformanceTrend,
  usePerformanceAlerts,
  useModelComparison,
  useConfusionMatrix,
  useRocCurve,
} from '@/hooks/api/use-monitoring';
import { useModelsStatus } from '@/hooks/api/use-predictions';
import { useKPIList } from '@/hooks/api/use-kpi';

// =============================================================================
// FIXTURES
// =============================================================================

const mockCreateObjectURL = vi.fn(() => 'blob:mock-url');
const mockRevokeObjectURL = vi.fn();
global.URL.createObjectURL = mockCreateObjectURL;
global.URL.revokeObjectURL = mockRevokeObjectURL;

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
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
      model_name: 'propensity_v2.1.0',
      status: 'healthy',
      endpoint: '/predict/propensity',
      last_check: '2026-05-17T10:00:00Z',
    },
    {
      model_name: 'churn_v1.5.2',
      status: 'healthy',
      endpoint: '/predict/churn',
      last_check: '2026-05-17T10:00:00Z',
    },
  ],
  timestamp: '2026-05-17T10:00:00Z',
};

const mockTrend = {
  model_id: 'propensity_v2.1.0',
  metric_name: 'accuracy',
  current_value: 0.918,
  baseline_value: 0.9,
  change_percent: 2.0,
  trend: 'improving' as const,
  is_significant: true,
  alert_threshold_breached: false,
  alert_threshold: 0.81,
  history: [
    { metric_name: 'accuracy', metric_value: 0.9, recorded_at: '2026-05-10T00:00:00Z' },
    { metric_name: 'accuracy', metric_value: 0.918, recorded_at: '2026-05-17T00:00:00Z' },
  ],
};

const mockAlerts = {
  model_id: 'propensity_v2.1.0',
  alert_count: 1,
  alerts: [
    {
      metric_name: 'precision',
      current_value: 0.71,
      baseline_value: 0.85,
      change_percent: -16.5,
      trend: 'degrading',
      severity: 'high',
      message: 'Precision dropped below baseline by 16.5%',
    },
  ],
};

const mockComparison = {
  model_id: 'propensity_v2.1.0',
  other_model_id: 'churn_v1.5.2',
  metric_name: 'accuracy',
  model_value: 0.918,
  other_model_value: 0.872,
  difference: 0.046,
  difference_percent: 5.28,
  better_model: 'propensity_v2.1.0',
};

const mockConfusion = {
  model_id: 'propensity_v2.1.0',
  available: true,
  tn: 2946,
  fp: 346,
  fn: 1277,
  tp: 506,
  threshold: 0.5,
  sample_size: 5075,
  measured_at: '2026-06-10T00:00:00Z',
};

// WS1 model-performance KPI metadata — the SAME thresholds kpi_definitions.yaml
// serves to the Home KPI grid (fractions, not percents). The page must status
// its "Current <metric>" card against these so both surfaces tell one story.
const mockKpiList = {
  kpis: [
    {
      id: 'WS1-MP-001',
      name: 'ROC-AUC',
      workstream: 'ws1_model_performance',
      threshold: { target: 0.8, warning: 0.7, critical: 0.6 },
    },
    {
      id: 'WS1-MP-003',
      name: 'F1 Score',
      workstream: 'ws1_model_performance',
      threshold: { target: 0.75, warning: 0.6, critical: 0.45 },
    },
  ],
  total: 2,
  workstream: 'ws1_model_performance',
  causal_library: null,
};

const mockRoc = {
  model_id: 'propensity_v2.1.0',
  available: true,
  points: [
    { fpr: 0.0, tpr: 0.0, threshold: 1.0 },
    { fpr: 0.3, tpr: 0.6, threshold: 0.5 },
    { fpr: 1.0, tpr: 1.0, threshold: 0.0 },
  ],
  auc: 0.671,
  sample_size: 5075,
  measured_at: '2026-06-10T00:00:00Z',
};

// =============================================================================
// HELPERS
// =============================================================================

function setHooksToSuccess() {
  (useModelsStatus as ReturnType<typeof vi.fn>).mockReturnValue({
    data: mockModelsStatus,
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
  });
  (usePerformanceTrend as ReturnType<typeof vi.fn>).mockReturnValue({
    data: mockTrend,
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
  });
  (usePerformanceAlerts as ReturnType<typeof vi.fn>).mockReturnValue({
    data: mockAlerts,
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
  });
  (useModelComparison as ReturnType<typeof vi.fn>).mockReturnValue({
    data: mockComparison,
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
  });
  (useConfusionMatrix as ReturnType<typeof vi.fn>).mockReturnValue({
    data: mockConfusion,
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
  });
  (useRocCurve as ReturnType<typeof vi.fn>).mockReturnValue({
    data: mockRoc,
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
  });
  (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
    data: mockKpiList,
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
  });
}

/** Override the trend hook with a given metric/value/breach state. */
function setTrend(overrides: Partial<typeof mockTrend>) {
  (usePerformanceTrend as ReturnType<typeof vi.fn>).mockReturnValue({
    data: { ...mockTrend, ...overrides },
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
  });
}

// =============================================================================
// TESTS
// =============================================================================

describe('ModelPerformance', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setHooksToSuccess();
  });

  it('renders page header with title', () => {
    render(<ModelPerformance />, { wrapper: createWrapper() });

    expect(screen.getByText('Model Performance')).toBeInTheDocument();
    expect(
      screen.getByText(/View model metrics, confusion matrix, ROC curves/i)
    ).toBeInTheDocument();
  });

  it('issue-298: calls live hooks (useModelsStatus + usePerformanceTrend + usePerformanceAlerts)', () => {
    render(<ModelPerformance />, { wrapper: createWrapper() });

    expect(useModelsStatus).toHaveBeenCalled();
    expect(usePerformanceTrend).toHaveBeenCalled();
    expect(usePerformanceAlerts).toHaveBeenCalled();
  });

  it('renders the confusion matrix + ROC curve from live curve data', async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    render(<ModelPerformance />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Confusion Matrix/i }));
    expect(await screen.findByText('True Positive')).toBeInTheDocument();
    expect(screen.getByText('True Negative')).toBeInTheDocument();

    await user.click(screen.getByRole('tab', { name: /ROC Curve/i }));
    expect(await screen.findByText(/AUC = 0\.671/)).toBeInTheDocument();
  });

  it('shows an honest empty-state when curves are not yet recorded', async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    (useConfusionMatrix as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { model_id: 'm', available: false, tn: 0, fp: 0, fn: 0, tp: 0, threshold: 0.5 },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    render(<ModelPerformance />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Confusion Matrix/i }));
    expect(await screen.findByText(/No confusion matrix recorded/i)).toBeInTheDocument();
  });

  it('issue-298: useModelComparison is called with both ids and disabled until 2nd model picked', () => {
    render(<ModelPerformance />, { wrapper: createWrapper() });

    // Comparison hook is invoked on first render — initial 2nd-model id is ''.
    expect(useModelComparison).toHaveBeenCalled();
    const compareCalls = (useModelComparison as ReturnType<typeof vi.fn>).mock.calls;
    const lastCall = compareCalls[compareCalls.length - 1] ?? [];
    // Args are: (modelId, otherModelId, metricName, options)
    const [firstId, otherId, metric, opts] = lastCall;
    // First id should be the auto-selected model from useModelsStatus
    expect(firstId).toBe('propensity_v2.1.0');
    // Other id is initially empty string (no comparison picked yet)
    expect(otherId).toBe('');
    // Comparison metric is a fixed 'accuracy' (independent of the trend default)
    expect(metric).toBe('accuracy');
    // Query MUST be disabled until comparison id is picked
    expect(opts?.enabled).toBe(false);
  });

  it('issue-298: picking a 2nd model enables useModelComparison with both ids', async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 });

    render(<ModelPerformance />, { wrapper: createWrapper() });

    // Navigate to the Comparison tab
    const compareTab = screen.getByRole('tab', { name: /Comparison/i });
    await user.click(compareTab);

    // The "Compare with:" Select trigger should now be visible
    const compareSelects = screen.getAllByRole('combobox');
    // Two comboboxes are present: top model selector + "Compare with" inside the tab
    expect(compareSelects.length).toBeGreaterThanOrEqual(2);
    const compareWithTrigger = compareSelects[compareSelects.length - 1];

    // Open it and pick churn_v1.5.2
    await user.click(compareWithTrigger);
    const churnOption = await screen.findByRole('option', { name: /churn_v1\.5\.2/ });
    await user.click(churnOption);

    // After picking, useModelComparison must be called with both ids and enabled=true.
    const compareCalls = (useModelComparison as ReturnType<typeof vi.fn>).mock.calls;
    const enabledCall = compareCalls.find((c) => c[3]?.enabled === true);
    expect(enabledCall).toBeDefined();
    expect(enabledCall?.[0]).toBe('propensity_v2.1.0');
    expect(enabledCall?.[1]).toBe('churn_v1.5.2');
    expect(enabledCall?.[2]).toBe('accuracy');
  });

  it('renders the comparison verdict with the better-model handle + significance note', async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    // A non-significant comparison so the disclosure note also renders.
    (useModelComparison as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { ...mockComparison, is_significant: false },
      isLoading: false,
      isError: false,
      error: null,
      isRefetching: false,
      refetch: vi.fn(),
    });

    render(<ModelPerformance />, { wrapper: createWrapper() });

    const compareTab = screen.getByRole('tab', { name: /Comparison/i });
    await user.click(compareTab);

    // Pick the 2nd model so the comparison cards render.
    const compareSelects = screen.getAllByRole('combobox');
    await user.click(compareSelects[compareSelects.length - 1]);
    const churnOption = await screen.findByRole('option', { name: /churn_v1\.5\.2/ });
    await user.click(churnOption);

    // The verdict is its own card (label "Better model"), the winning handle is
    // carried in a `title` attr (so a long name truncates instead of spilling a
    // KPI value), and a non-significant difference is disclosed.
    expect(await screen.findByText('Better model')).toBeInTheDocument();
    expect(screen.getByTitle('propensity_v2.1.0')).toBeInTheDocument();
    expect(screen.getByText(/not significant/i)).toBeInTheDocument();
  });

  it('issue-298: stale compareModelId after models-list shrink -> otherId=="" + enabled=false', async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 });

    const { rerender } = render(<ModelPerformance />, { wrapper: createWrapper() });

    // STEP 1: actually select churn_v1.5.2 as comparison so compareModelId is non-empty
    const compareTab = screen.getByRole('tab', { name: /Comparison/i });
    await user.click(compareTab);

    const compareSelects = screen.getAllByRole('combobox');
    const compareWithTrigger = compareSelects[compareSelects.length - 1];
    await user.click(compareWithTrigger);
    const churnOption = await screen.findByRole('option', { name: /churn_v1\.5\.2/ });
    await user.click(churnOption);

    // Sanity: hook should now be enabled with both ids
    let compareCalls = (useModelComparison as ReturnType<typeof vi.fn>).mock.calls;
    const enabledCall = compareCalls.find((c) => c[3]?.enabled === true);
    expect(enabledCall).toBeDefined();
    expect(enabledCall?.[1]).toBe('churn_v1.5.2');

    // STEP 2: shrink the models list so churn is GONE — the stale
    // compareModelId would normally leak into useModelComparison, but
    // effectiveCompareModelId must clamp it back to ''.
    (useModelsStatus as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        total_models: 1,
        healthy_count: 1,
        unhealthy_count: 0,
        models: [mockModelsStatus.models[0]],
        timestamp: '2026-05-17T10:00:00Z',
      },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    rerender(<ModelPerformance />);

    compareCalls = (useModelComparison as ReturnType<typeof vi.fn>).mock.calls;
    const latest = compareCalls[compareCalls.length - 1] ?? [];
    // Stale comparison id must be clamped to '' and the hook disabled
    expect(latest[1]).toBe('');
    expect(latest[3]?.enabled).toBe(false);
  });

  it('issue-298: does NOT render hard-coded sample model names (churn-v3 / hcp-tier / conversion-v2 / adherence-v1)', () => {
    render(<ModelPerformance />, { wrapper: createWrapper() });

    // The 4 hard-coded SAMPLE_MODELS display names must be gone after the wire-up
    expect(screen.queryByText('Patient Churn Predictor')).not.toBeInTheDocument();
    expect(screen.queryByText('HCP Tier Classifier')).not.toBeInTheDocument();
    expect(screen.queryByText('Conversion Predictor')).not.toBeInTheDocument();
    expect(screen.queryByText('Adherence Risk Model')).not.toBeInTheDocument();
  });

  it('issue-298: does NOT render hard-coded sample metric (15,420 samples for churn-v3)', () => {
    render(<ModelPerformance />, { wrapper: createWrapper() });

    // 15,420 was hard-coded for churn-v3 model in the SAMPLE_METRICS lookup
    expect(screen.queryByText('15,420')).not.toBeInTheDocument();
  });

  it('issue-298: model selector lists models from useModelsStatus', () => {
    render(<ModelPerformance />, { wrapper: createWrapper() });

    // Live data-driven model name surfaces as selector default
    // (it appears in the SelectValue trigger and possibly the info card)
    const propensityMatches = screen.getAllByText(/propensity_v2\.1\.0/);
    expect(propensityMatches.length).toBeGreaterThanOrEqual(1);
  });

  it('issue-298: renders performance trend current_value as live KPI', () => {
    render(<ModelPerformance />, { wrapper: createWrapper() });

    // current_value 0.918 -> 91.8% should appear
    expect(screen.getAllByText(/91\.8/).length).toBeGreaterThanOrEqual(1);
  });

  it('issue-298: shows loading skeleton when hooks loading', () => {
    (useModelsStatus as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });

    const { container } = render(<ModelPerformance />, { wrapper: createWrapper() });

    // Loading state must produce some indicator (skeleton, spinner, or "Loading" text)
    const hasLoading =
      container.querySelector('[data-loading="true"]') ||
      container.querySelector('.animate-pulse') ||
      screen.queryByText(/Loading/i);
    expect(hasLoading).toBeTruthy();
  });

  it('issue-298: shows QueryErrorState when models hook errors', () => {
    (useModelsStatus as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error('Network unreachable'),
      refetch: vi.fn(),
    });

    render(<ModelPerformance />, { wrapper: createWrapper() });

    // QueryErrorState typically surfaces an Alert with "wrong" or specific copy
    const errorIndicator =
      screen.queryByRole('alert') ||
      screen.queryByText(/Something went wrong|Network|Error|Unable to/i);
    expect(errorIndicator).toBeTruthy();
  });

  it('issue-298: when models list is empty, hooks are disabled (no live id leakage)', () => {
    (useModelsStatus as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        total_models: 0,
        healthy_count: 0,
        unhealthy_count: 0,
        models: [],
        timestamp: '2026-05-17T10:00:00Z',
      },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });

    render(<ModelPerformance />, { wrapper: createWrapper() });

    // With empty models, every trend query (selected + both sibling-brand
    // overlay queries) must be invoked with an empty id and disabled.
    const trendCalls = (usePerformanceTrend as ReturnType<typeof vi.fn>).mock.calls;
    expect(trendCalls.length).toBeGreaterThan(0);
    for (const [trendParams, trendOpts] of trendCalls) {
      expect(trendParams?.model_id).toBe('');
      expect(trendOpts?.enabled).toBe(false);
    }
  });

  it('displays visualization tabs', () => {
    render(<ModelPerformance />, { wrapper: createWrapper() });

    expect(screen.getByRole('tab', { name: /Confusion Matrix/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /ROC Curve/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Performance Trend/i })).toBeInTheDocument();
  });

  it('displays export button', () => {
    render(<ModelPerformance />, { wrapper: createWrapper() });

    expect(screen.getByRole('button', { name: /Export/i })).toBeInTheDocument();
  });

  it('renders the StrategicInsightCard (Strategic Interpretation) above the metrics', async () => {
    render(<ModelPerformance />, { wrapper: createWrapper() });

    expect(await screen.findByText(/strategic interpretation/i)).toBeInTheDocument();
  });

  it('renders confusion interpretation verdict and ROC band after clicking tabs', async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 });

    // Override models to use an initiation cohort model
    const initiationModel = {
      model_name: 'initiation_remibrutinib_goldstd_lr_v1',
      status: 'healthy',
      endpoint: '/predict/initiation_remibrutinib',
      last_check: '2026-06-20T10:00:00Z',
    };
    (useModelsStatus as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        total_models: 1,
        healthy_count: 1,
        unhealthy_count: 0,
        models: [initiationModel],
        timestamp: '2026-06-20T10:00:00Z',
      },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });

    const initiationConfusion = {
      model_id: 'initiation_remibrutinib_goldstd_lr_v1',
      available: true,
      tn: 2946,
      fp: 346,
      fn: 1277,
      tp: 506,
      threshold: 0.5,
      sample_size: 5075,
      measured_at: '2026-06-20T00:00:00Z',
    };
    (useConfusionMatrix as ReturnType<typeof vi.fn>).mockReturnValue({
      data: initiationConfusion,
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });

    const initiationRoc = {
      model_id: 'initiation_remibrutinib_goldstd_lr_v1',
      available: true,
      points: [
        { fpr: 0, tpr: 0, threshold: 1 },
        { fpr: 1, tpr: 1, threshold: 0 },
      ],
      auc: 0.671,
      sample_size: 5075,
      measured_at: '2026-06-20T00:00:00Z',
    };
    (useRocCurve as ReturnType<typeof vi.fn>).mockReturnValue({
      data: initiationRoc,
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });

    render(<ModelPerformance />, { wrapper: createWrapper() });

    // Click the Confusion Matrix tab and assert interpretation verdict
    await user.click(screen.getByRole('tab', { name: /Confusion Matrix/i }));
    expect(await screen.findByText(/conservative/i)).toBeInTheDocument();
    expect(screen.getByText(/initiated treatment/i)).toBeInTheDocument();

    // Click the ROC Curve tab and assert AUC band sentence
    await user.click(screen.getByRole('tab', { name: /ROC Curve/i }));
    expect(await screen.findByText(/AUC 0\.671 \(weak\)/i)).toBeInTheDocument();
  });

  // ===========================================================================
  // Trend enhancements ported from TimeSeries (metric select, range select,
  // gold-standard all-brands overlay, provenance note)
  // ===========================================================================

  describe('trend enhancements (ported from TimeSeries)', () => {
    /** The selected model's trend query is the ENABLED usePerformanceTrend call. */
    function enabledTrendCalls() {
      return (usePerformanceTrend as ReturnType<typeof vi.fn>).mock.calls.filter(
        (c) => c[1]?.enabled
      );
    }

    /** Swap the models list for a single gold-standard per-brand model. */
    function setGoldstdModel(modelName = 'initiation_remibrutinib_goldstd_lr_v1') {
      (useModelsStatus as ReturnType<typeof vi.fn>).mockReturnValue({
        data: {
          total_models: 1,
          healthy_count: 1,
          unhealthy_count: 0,
          models: [
            {
              model_name: modelName,
              status: 'healthy',
              endpoint: `/predict/${modelName}`,
              last_check: '2026-07-04T10:00:00Z',
            },
          ],
          timestamp: '2026-07-04T10:00:00Z',
        },
        isLoading: false,
        isError: false,
        error: null,
        refetch: vi.fn(),
      });
    }

    it('defaults: enabled trend query uses metric=auc_roc (WS1-MP-001) and days=365', () => {
      render(<ModelPerformance />, { wrapper: createWrapper() });

      const enabled = enabledTrendCalls();
      expect(enabled.length).toBeGreaterThan(0);
      for (const [params] of enabled) {
        expect(params.model_id).toBe('propensity_v2.1.0');
        expect(params.metric_name).toBe('auc_roc');
        expect(params.days).toBe(365);
      }
      // Sibling-brand overlay queries stay disabled for non-goldstd models.
      const disabled = (usePerformanceTrend as ReturnType<typeof vi.fn>).mock.calls.filter(
        (c) => !c[1]?.enabled
      );
      expect(disabled.length).toBeGreaterThan(0);
      for (const [params] of disabled) {
        expect(params.model_id).toBe('');
      }
    });

    it('metric selector drives the enabled trend query metric_name', async () => {
      const user = userEvent.setup({ pointerEventsCheck: 0 });
      render(<ModelPerformance />, { wrapper: createWrapper() });

      await user.click(screen.getByRole('combobox', { name: 'metric' }));
      await user.click(await screen.findByRole('option', { name: 'Precision' }));

      await waitFor(() => {
        const enabled = enabledTrendCalls();
        const last = enabled[enabled.length - 1];
        expect(last?.[0]?.metric_name).toBe('precision');
        expect(last?.[0]?.model_id).toBe('propensity_v2.1.0');
      });
    });

    it('time-range selector drives the enabled trend query days', async () => {
      const user = userEvent.setup({ pointerEventsCheck: 0 });
      render(<ModelPerformance />, { wrapper: createWrapper() });

      await user.click(screen.getByRole('combobox', { name: 'time range' }));
      await user.click(await screen.findByRole('option', { name: '90 Days' }));

      await waitFor(() => {
        const enabled = enabledTrendCalls();
        const last = enabled[enabled.length - 1];
        expect(last?.[0]?.days).toBe(90);
        expect(last?.[0]?.model_id).toBe('propensity_v2.1.0');
      });
    });

    it('does NOT offer the "Compare all brands" toggle for non-goldstd models', () => {
      render(<ModelPerformance />, { wrapper: createWrapper() });

      expect(
        screen.queryByRole('checkbox', { name: /compare all brands/i })
      ).not.toBeInTheDocument();
    });

    it('offers the toggle for goldstd models; enabling it enables both sibling-brand queries', async () => {
      const user = userEvent.setup({ pointerEventsCheck: 0 });
      setGoldstdModel('initiation_remibrutinib_goldstd_lr_v1');
      render(<ModelPerformance />, { wrapper: createWrapper() });

      const toggle = screen.getByRole('checkbox', { name: /compare all brands/i });
      await user.click(toggle);

      await waitFor(() => {
        const enabledIds = new Set(enabledTrendCalls().map((c) => c[0].model_id));
        expect(enabledIds.has('initiation_remibrutinib_goldstd_lr_v1')).toBe(true);
        expect(enabledIds.has('initiation_fabhalta_goldstd_lr_v1')).toBe(true);
        expect(enabledIds.has('initiation_kisqali_goldstd_lr_v1')).toBe(true);
      });
      // Sibling queries inherit the same metric + window as the selected model.
      for (const [params] of enabledTrendCalls().slice(-3)) {
        expect(params.metric_name).toBe('auc_roc');
        expect(params.days).toBe(365);
      }
    });

    it('renders the walk-forward provenance note on the trend card', () => {
      render(<ModelPerformance />, { wrapper: createWrapper() });

      const note = screen.getByTestId('perf-trend-provenance-note');
      expect(note).toHaveTextContent(/walk-forward backtest/i);
      expect(note).toHaveTextContent(/not the calibrated champion/i);
      // mockTrend's latest history point is 2026-05-17 — the data boundary.
      expect(note).toHaveTextContent(/data coverage through May 17/i);
    });
  });

  // ===========================================================================
  // Threshold-aware statuses (Home KPI-grid parity)
  //
  // The Home "Model Performance" tiles status brand-aggregated holdout metrics
  // against kpi_definitions.yaml targets; this page used to color the same
  // metrics green whenever they weren't DEGRADING (alert/trend semantics), so
  // a brand could read yellow on Home and all-green here for identical data.
  // The Current card must now share the canonical WS1 thresholds (from the
  // KPI list API), and reference-value cards must be neutral, not green.
  // ===========================================================================
  describe('threshold statuses (Home KPI-grid parity)', () => {
    beforeEach(() => {
      setHooksToSuccess();
    });

    const currentCard = () => document.querySelector('.perf-current-card');
    const baselineCard = () => document.querySelector('.perf-baseline-card');

    it('Current auc_roc below the WS1-MP-001 target renders WARNING (amber), not green', () => {
      // Fabhalta discontinuation model's live value: stable (no alert breach)
      // but below the 0.80 KPI target — Home shows yellow, so must this card.
      setTrend({ metric_name: 'auc_roc', current_value: 0.7873, alert_threshold_breached: false });
      render(<ModelPerformance />, { wrapper: createWrapper() });

      expect(currentCard()?.className).toContain('border-l-amber-500');
    });

    it('Current auc_roc at/above target renders healthy (emerald)', () => {
      setTrend({ metric_name: 'auc_roc', current_value: 0.85, alert_threshold_breached: false });
      render(<ModelPerformance />, { wrapper: createWrapper() });

      expect(currentCard()?.className).toContain('border-l-emerald-500');
    });

    it('Current auc_roc below the critical threshold renders critical (rose)', () => {
      setTrend({ metric_name: 'auc_roc', current_value: 0.55, alert_threshold_breached: false });
      render(<ModelPerformance />, { wrapper: createWrapper() });

      expect(currentCard()?.className).toContain('border-l-rose-500');
    });

    it('alert breach still escalates to critical even above the KPI target', () => {
      setTrend({ metric_name: 'auc_roc', current_value: 0.85, alert_threshold_breached: true });
      render(<ModelPerformance />, { wrapper: createWrapper() });

      expect(currentCard()?.className).toContain('border-l-rose-500');
    });

    it('metrics without a canonical WS1 threshold (accuracy) keep alert semantics', () => {
      setTrend({ metric_name: 'accuracy', current_value: 0.918, alert_threshold_breached: false });
      render(<ModelPerformance />, { wrapper: createWrapper() });

      expect(currentCard()?.className).toContain('border-l-emerald-500');
    });

    it('KPI list unavailable -> falls back to alert semantics (never blocks the page)', () => {
      (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
        data: undefined,
        isLoading: true,
        isError: false,
        error: null,
        refetch: vi.fn(),
      });
      setTrend({ metric_name: 'auc_roc', current_value: 0.7873, alert_threshold_breached: false });
      render(<ModelPerformance />, { wrapper: createWrapper() });

      expect(currentCard()?.className).toContain('border-l-emerald-500');
    });

    it('Baseline card is neutral — a reference value, not a health claim', () => {
      render(<ModelPerformance />, { wrapper: createWrapper() });

      expect(baselineCard()?.className).toContain('border-l-gray-400');
    });

    it('comparison value cards are neutral, not hardcoded green', async () => {
      const user = userEvent.setup({ pointerEventsCheck: 0 });
      render(<ModelPerformance />, { wrapper: createWrapper() });

      const compareTab = screen.getByRole('tab', { name: /Comparison/i });
      await user.click(compareTab);

      const compareSelects = screen.getAllByRole('combobox');
      await user.click(compareSelects[compareSelects.length - 1]);
      const churnOption = await screen.findByRole('option', { name: /churn_v1\.5\.2/ });
      await user.click(churnOption);

      await screen.findByText('Better model');
      const compareCards = document.querySelectorAll('.compare-value-card');
      expect(compareCards.length).toBe(3);
      for (const card of compareCards) {
        expect(card.className).toContain('border-l-gray-400');
        expect(card.className).not.toContain('border-l-emerald-500');
      }
    });
  });
});
