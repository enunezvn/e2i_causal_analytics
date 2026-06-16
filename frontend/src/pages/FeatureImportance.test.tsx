/**
 * FeatureImportance Page Tests
 * ============================
 *
 * Tests for the Feature Importance analysis page with SHAP visualizations.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import FeatureImportance from './FeatureImportance';
import { ModelType } from '@/types/explain';

// Mock the explain hooks so we can control returned values
vi.mock('@/hooks/api/use-explain', () => ({
  useExplain: vi.fn(),
  useExplainableModels: vi.fn(),
  useExplanationHistory: vi.fn(),
}));

import {
  useExplain,
  useExplainableModels,
  useExplanationHistory,
} from '@/hooks/api/use-explain';

// Mock URL.createObjectURL and URL.revokeObjectURL for export tests
const mockCreateObjectURL = vi.fn(() => 'blob:mock-url');
const mockRevokeObjectURL = vi.fn();

// QueryClient wrapper required because page uses tanstack-query hooks
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

// =============================================================================
// HOOK MOCK DEFAULTS
// =============================================================================

const mockExplainableModels = {
  supported_models: [
    {
      model_type: ModelType.PROPENSITY,
      latest_version: 'v3.2.1',
      explainer_type: 'TreeExplainer' as const,
      avg_latency_ms: 45,
    },
    {
      model_type: ModelType.CHURN_PREDICTION,
      latest_version: 'v1.5.0',
      explainer_type: 'TreeExplainer' as const,
      avg_latency_ms: 32,
    },
  ],
  total_models: 2,
};

const mockExplainResponse = {
  explanation_id: 'expl_test_123',
  request_timestamp: '2026-05-17T10:00:00Z',
  patient_id: 'patient_42',
  model_type: ModelType.PROPENSITY,
  model_version_id: 'v3.2.1',
  prediction_class: 'high_risk',
  prediction_probability: 0.78,
  base_value: 0.42,
  top_features: [
    {
      feature_name: 'live_feature_alpha',
      feature_value: 99,
      shap_value: 0.51,
      contribution_direction: 'positive' as const,
      contribution_rank: 1,
    },
    {
      feature_name: 'live_feature_beta',
      feature_value: 7,
      shap_value: -0.33,
      contribution_direction: 'negative' as const,
      contribution_rank: 2,
    },
  ],
  shap_sum: 0.18,
  computation_time_ms: 38,
  audit_stored: true,
};

beforeEach(() => {
  vi.clearAllMocks();
  global.URL.createObjectURL = mockCreateObjectURL;
  global.URL.revokeObjectURL = mockRevokeObjectURL;

  // Default: models load, no explanation yet (initial state).
  (useExplainableModels as ReturnType<typeof vi.fn>).mockReturnValue({
    data: mockExplainableModels,
    isLoading: false,
    isError: false,
    error: null,
  });

  (useExplain as ReturnType<typeof vi.fn>).mockReturnValue({
    mutate: vi.fn(),
    data: undefined,
    isPending: false,
    isError: false,
    error: null,
    reset: vi.fn(),
  });

  (useExplanationHistory as ReturnType<typeof vi.fn>).mockReturnValue({
    data: { patient_id: '', total_explanations: 0, explanations: [] },
    isLoading: false,
    isError: false,
    error: null,
  });
});

// Cold-start render of this page (radix tabs + recharts + react-query wrapper)
// can exceed the default 5s when the full suite is running in parallel; give
// each test enough headroom so flakes don't mask real regressions.
vi.setConfig({ testTimeout: 15000 });

describe('FeatureImportance', () => {
  it('renders page header with title and description', () => {
    render(<FeatureImportance />, { wrapper: createWrapper() });

    expect(screen.getByText('Feature Importance')).toBeInTheDocument();
    expect(
      screen.getByText(/SHAP values, feature importance bar charts, beeswarm plots/i)
    ).toBeInTheDocument();
  });

  it('displays model selector dropdown', () => {
    render(<FeatureImportance />, { wrapper: createWrapper() });

    expect(screen.getByRole('combobox')).toBeInTheDocument();
  });

  it('displays visualization tabs', () => {
    render(<FeatureImportance />, { wrapper: createWrapper() });

    expect(screen.getByRole('tab', { name: /Bar Chart/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Beeswarm/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Waterfall/i })).toBeInTheDocument();
  });

  it('has clickable Beeswarm tab', () => {
    render(<FeatureImportance />, { wrapper: createWrapper() });

    const beeswarmTab = screen.getByRole('tab', { name: /Beeswarm/i });
    expect(beeswarmTab).toBeInTheDocument();
    expect(beeswarmTab).not.toBeDisabled();
  });

  it('has clickable Waterfall tab', () => {
    render(<FeatureImportance />, { wrapper: createWrapper() });

    const waterfallTab = screen.getByRole('tab', { name: /Waterfall/i });
    expect(waterfallTab).toBeInTheDocument();
    expect(waterfallTab).not.toBeDisabled();
  });

  it('displays refresh button', () => {
    const { container } = render(<FeatureImportance />, { wrapper: createWrapper() });

    // Refresh button has RefreshCw icon
    const refreshButton = container.querySelector('button svg.lucide-refresh-cw');
    expect(refreshButton).toBeInTheDocument();
  });

  it('displays export button', () => {
    render(<FeatureImportance />, { wrapper: createWrapper() });

    expect(screen.getByRole('button', { name: /Export/i })).toBeInTheDocument();
  });
});

// =============================================================================
// LIVE-DATA WIRING TESTS (Issue #299)
// =============================================================================

describe('FeatureImportance — live data wiring (#299)', () => {
  it('populates model selector from useExplainableModels', async () => {
    render(<FeatureImportance />, { wrapper: createWrapper() });

    // Open the model selector dropdown
    const trigger = screen.getByRole('combobox');
    fireEvent.click(trigger);

    // Both supported models should appear in dropdown
    await waitFor(() => {
      // Model types render with title-cased / formatted names
      const propensityOptions = screen.getAllByText(/propensity/i);
      expect(propensityOptions.length).toBeGreaterThanOrEqual(1);
      const churnOptions = screen.getAllByText(/churn/i);
      expect(churnOptions.length).toBeGreaterThanOrEqual(1);
    });
  });

  it('provides a patient ID input', () => {
    render(<FeatureImportance />, { wrapper: createWrapper() });

    expect(
      screen.getByPlaceholderText(/patient/i)
    ).toBeInTheDocument();
  });

  it('invokes useExplain.mutate with patient_id + model_type when Explain is clicked', () => {
    const mockMutate = vi.fn();
    (useExplain as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      data: undefined,
      isPending: false,
      isError: false,
      error: null,
      reset: vi.fn(),
    });

    render(<FeatureImportance />, { wrapper: createWrapper() });

    const patientInput = screen.getByPlaceholderText(/patient/i);
    fireEvent.change(patientInput, { target: { value: 'patient_42' } });

    const explainBtn = screen.getByRole('button', { name: /^explain$/i });
    fireEvent.click(explainBtn);

    expect(mockMutate).toHaveBeenCalledTimes(1);
    const callArg = mockMutate.mock.calls[0][0];
    expect(callArg).toMatchObject({
      patient_id: 'patient_42',
      model_type: expect.any(String),
      // Format + top_k must be supplied so the backend returns a usable
      // top-K SHAP slice (default is 5 server-side; we ask for more).
      format: 'top_k',
      top_k: 10,
    });
  });

  it('clears stale selectedFeature when explain is re-run', () => {
    // Start with an explanation that has features
    (useExplain as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      data: mockExplainResponse,
      isPending: false,
      isError: false,
      error: null,
      reset: vi.fn(),
    });

    render(<FeatureImportance />, { wrapper: createWrapper() });

    // Click on the first feature row to select it (Feature Details card appears)
    const firstFeatureRow = screen.getByText(/live feature alpha/i, { selector: 'div.font-medium.truncate' });
    fireEvent.click(firstFeatureRow);
    expect(
      screen.getByText(/Feature Details: live feature alpha/i)
    ).toBeInTheDocument();

    // Now run Explain again — selectedFeature must be cleared
    const patientInput = screen.getByPlaceholderText(/patient/i);
    fireEvent.change(patientInput, { target: { value: 'new_patient' } });
    const explainBtn = screen.getByRole('button', { name: /^explain$/i });
    fireEvent.click(explainBtn);

    expect(
      screen.queryByText(/Feature Details: live feature alpha/i)
    ).not.toBeInTheDocument();
  });

  it('renders a falsy prediction_probability in history without crashing', () => {
    // Force-render with an explanation + history that has a malformed row
    (useExplain as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      data: mockExplainResponse,
      isPending: false,
      isError: false,
      error: null,
      reset: vi.fn(),
    });
    (useExplanationHistory as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        patient_id: 'patient_42',
        total_explanations: 1,
        // Legacy ml_shap_analyses row missing several ExplainResponse fields
        explanations: [
          {
            explanation_id: 'legacy_row_1',
            // No request_timestamp, no prediction_probability, no model_version_id
            model_type: 'legacy_model',
            prediction_class: null,
            prediction_probability: null,
          },
        ],
      },
      isLoading: false,
      isError: false,
      error: null,
    });

    render(<FeatureImportance />, { wrapper: createWrapper() });

    // Submit a patient to enable the history hook
    const patientInput = screen.getByPlaceholderText(/patient/i);
    fireEvent.change(patientInput, { target: { value: 'patient_42' } });
    const explainBtn = screen.getByRole('button', { name: /^explain$/i });
    fireEvent.click(explainBtn);

    // History tab must remain accessible — no crash thrown during render
    expect(screen.getByRole('tab', { name: /History/i })).toBeInTheDocument();
  });

  it('renders real SHAP values from useExplain response (not synthetic data)', () => {
    (useExplain as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      data: mockExplainResponse,
      isPending: false,
      isError: false,
      error: null,
      reset: vi.fn(),
    });

    render(<FeatureImportance />, { wrapper: createWrapper() });

    // Real features should be displayed (may appear in both feature list AND chart)
    expect(screen.getAllByText(/live feature alpha/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/live feature beta/i).length).toBeGreaterThanOrEqual(1);

    // Real SHAP values should appear (live_feature_alpha +0.5100, beta -0.3300)
    expect(screen.getByText('+0.5100')).toBeInTheDocument();
    expect(screen.getByText('-0.3300')).toBeInTheDocument();

    // Synthetic data markers from the original hard-coded SAMPLE_FEATURES MUST NOT be present
    expect(screen.queryByText('days since last visit')).not.toBeInTheDocument();
    expect(screen.queryByText('total prescriptions ytd')).not.toBeInTheDocument();
    expect(screen.queryByText('+0.3500')).not.toBeInTheDocument();
    expect(screen.queryByText('-0.2800')).not.toBeInTheDocument();
  });

  it('uses real base_value from response (not synthetic SAMPLE_BASE_VALUES)', () => {
    (useExplain as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      data: mockExplainResponse, // base_value = 0.42
      isPending: false,
      isError: false,
      error: null,
      reset: vi.fn(),
    });

    render(<FeatureImportance />, { wrapper: createWrapper() });

    expect(screen.getByText('0.420')).toBeInTheDocument();
    // Synthetic default base value MUST NOT be present
    expect(screen.queryByText('0.350')).not.toBeInTheDocument();
  });

  it('shows a loading state while useExplain is pending', () => {
    (useExplain as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      data: undefined,
      isPending: true,
      isError: false,
      error: null,
      reset: vi.fn(),
    });

    render(<FeatureImportance />, { wrapper: createWrapper() });

    // Some sort of loading indicator (text or aria-busy element) must be present
    const loadingNodes = screen.queryAllByText(/loading|computing|explaining/i);
    expect(loadingNodes.length).toBeGreaterThanOrEqual(1);
  });

  it('shows an error state when useExplain errors', () => {
    (useExplain as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      data: undefined,
      isPending: false,
      isError: true,
      error: { message: 'patient not found' },
      reset: vi.fn(),
    });

    render(<FeatureImportance />, { wrapper: createWrapper() });

    expect(screen.getByText(/patient not found|error|failed/i)).toBeInTheDocument();
  });

  it('shows empty state when no patient has been explained yet', () => {
    // Default beforeEach mock: data === undefined, no pending, no error
    render(<FeatureImportance />, { wrapper: createWrapper() });

    // Should NOT crash and should show some kind of prompt asking to pick a patient
    expect(screen.getByPlaceholderText(/patient/i)).toBeInTheDocument();
    // And MUST NOT show the legacy synthetic features
    expect(screen.queryByText('days since last visit')).not.toBeInTheDocument();
  });

  it('renders a History tab that invokes useExplanationHistory for the submitted patient', () => {
    // Force-render with an explanation already returned so the history hook is enabled
    (useExplain as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      data: mockExplainResponse,
      isPending: false,
      isError: false,
      error: null,
      reset: vi.fn(),
    });
    const mockHistoryHook = vi.fn().mockReturnValue({
      data: {
        patient_id: 'patient_42',
        total_explanations: 1,
        explanations: [
          {
            ...mockExplainResponse,
            explanation_id: 'expl_hist_1',
            request_timestamp: '2026-05-16T08:00:00Z',
            prediction_class: 'high_risk',
            prediction_probability: 0.81,
          },
        ],
      },
      isLoading: false,
      isError: false,
      error: null,
    });
    (useExplanationHistory as ReturnType<typeof vi.fn>).mockImplementation(mockHistoryHook);

    render(<FeatureImportance />, { wrapper: createWrapper() });

    // History tab trigger must exist
    expect(screen.getByRole('tab', { name: /History/i })).toBeInTheDocument();

    // Submit a patient so the history hook gets a real id
    const patientInput = screen.getByPlaceholderText(/patient/i);
    fireEvent.change(patientInput, { target: { value: 'patient_42' } });
    const explainBtn = screen.getByRole('button', { name: /^explain$/i });
    fireEvent.click(explainBtn);

    // useExplanationHistory must have been called with the submitted patient id
    const callsWithPatient42 = mockHistoryHook.mock.calls.filter(
      (call) => call[0] === 'patient_42'
    );
    expect(callsWithPatient42.length).toBeGreaterThanOrEqual(1);
  });
});

// =============================================================================
// BRAND SELECTOR + PER-COHORT ID LABEL TESTS (Issue #967)
// =============================================================================
//
// The gold-standard per-brand serving exposes 12 models (4 cohorts × 3 brands),
// but this page could only ever exercise the default (Remibrutinib) brand
// because it never offered a brand selector and never threaded `brand` into the
// explain request. These tests pin the fix: a brand selector appears for
// gold-standard cohorts, the chosen brand rides the request, and the HCP-grain
// cohort relabels the ID field (patient_id → hcp_id).

/**
 * Build a `useExplainableModels` payload whose first (default-selected) model is
 * a single gold-standard cohort. `is_gold_standard` mirrors the real
 * `/api/explain/models` response field (src/api/routes/explain.py).
 */
function goldstdModels(cohort: string) {
  return {
    supported_models: [
      {
        model_type: cohort,
        latest_version: 'v1',
        explainer_type: 'KernelExplainer' as const,
        is_gold_standard: true,
      },
    ],
    total_models: 1,
  };
}

function mockMutateHook() {
  const mockMutate = vi.fn();
  (useExplain as ReturnType<typeof vi.fn>).mockReturnValue({
    mutate: mockMutate,
    data: undefined,
    isPending: false,
    isError: false,
    error: null,
    reset: vi.fn(),
  });
  return mockMutate;
}

describe('FeatureImportance — brand selector + per-cohort ID label (#967)', () => {
  it('renders a brand selector offering all three gold-standard brands when a gold-standard cohort is selected', () => {
    (useExplainableModels as ReturnType<typeof vi.fn>).mockReturnValue({
      data: goldstdModels('initiation'),
      isLoading: false,
      isError: false,
      error: null,
    });

    render(<FeatureImportance />, { wrapper: createWrapper() });

    const brandSelect = screen.getByRole('combobox', { name: /brand/i });
    expect(brandSelect).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Remibrutinib' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Fabhalta' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Kisqali' })).toBeInTheDocument();
  });

  it('does NOT render a brand selector for a legacy (non-gold-standard) model', () => {
    // beforeEach default mock = propensity + churn (both legacy, no brand)
    render(<FeatureImportance />, { wrapper: createWrapper() });

    expect(
      screen.queryByRole('combobox', { name: /brand/i })
    ).not.toBeInTheDocument();
  });

  it('threads the selected brand into the explain request for a gold-standard cohort', () => {
    const mockMutate = mockMutateHook();
    (useExplainableModels as ReturnType<typeof vi.fn>).mockReturnValue({
      data: goldstdModels('initiation'),
      isLoading: false,
      isError: false,
      error: null,
    });

    render(<FeatureImportance />, { wrapper: createWrapper() });

    // Pick a non-default brand
    fireEvent.change(screen.getByRole('combobox', { name: /brand/i }), {
      target: { value: 'Kisqali' },
    });

    fireEvent.change(screen.getByPlaceholderText(/patient/i), {
      target: { value: 'patient_42' },
    });
    fireEvent.click(screen.getByRole('button', { name: /^explain$/i }));

    expect(mockMutate).toHaveBeenCalledTimes(1);
    expect(mockMutate.mock.calls[0][0]).toMatchObject({
      patient_id: 'patient_42',
      model_type: 'initiation',
      brand: 'Kisqali',
      format: 'top_k',
      top_k: 10,
    });
  });

  it('defaults the brand to Remibrutinib for a gold-standard cohort', () => {
    const mockMutate = mockMutateHook();
    (useExplainableModels as ReturnType<typeof vi.fn>).mockReturnValue({
      data: goldstdModels('persistence'),
      isLoading: false,
      isError: false,
      error: null,
    });

    render(<FeatureImportance />, { wrapper: createWrapper() });

    fireEvent.change(screen.getByPlaceholderText(/patient/i), {
      target: { value: 'patient_7' },
    });
    fireEvent.click(screen.getByRole('button', { name: /^explain$/i }));

    expect(mockMutate.mock.calls[0][0]).toMatchObject({
      model_type: 'persistence',
      brand: 'Remibrutinib',
    });
  });

  it('relabels the ID input to HCP and sends hcp_id for the hcp_adoption cohort', () => {
    const mockMutate = mockMutateHook();
    (useExplainableModels as ReturnType<typeof vi.fn>).mockReturnValue({
      data: goldstdModels('hcp_adoption'),
      isLoading: false,
      isError: false,
      error: null,
    });

    render(<FeatureImportance />, { wrapper: createWrapper() });

    // The ID field is relabeled to HCP grain — a patient placeholder no longer
    // applies, an HCP one does.
    const hcpInput = screen.getByPlaceholderText(/hcp/i);
    expect(hcpInput).toBeInTheDocument();
    expect(screen.queryByPlaceholderText(/patient id/i)).not.toBeInTheDocument();

    fireEvent.change(hcpInput, { target: { value: 'HCP-NE-5678' } });
    fireEvent.click(screen.getByRole('button', { name: /^explain$/i }));

    expect(mockMutate.mock.calls[0][0]).toMatchObject({
      patient_id: 'HCP-NE-5678',
      hcp_id: 'HCP-NE-5678',
      model_type: 'hcp_adoption',
      brand: 'Remibrutinib',
    });
  });

  it('renders "HCP Adoption" (acronym-cased) for the hcp_adoption cohort', () => {
    (useExplainableModels as ReturnType<typeof vi.fn>).mockReturnValue({
      data: goldstdModels('hcp_adoption'),
      isLoading: false,
      isError: false,
      error: null,
    });
    // An explanation is present so the model-info card (which renders the
    // formatted cohort label) is visible.
    (useExplain as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      data: { ...mockExplainResponse, model_type: 'hcp_adoption' },
      isPending: false,
      isError: false,
      error: null,
      reset: vi.fn(),
    });

    render(<FeatureImportance />, { wrapper: createWrapper() });

    // Rendered in both the model-Select trigger and the model-info card.
    expect(screen.getAllByText('HCP Adoption').length).toBeGreaterThanOrEqual(1);
    // The naive Title-case ("Hcp Adoption") must NOT appear anywhere.
    expect(screen.queryByText('Hcp Adoption')).not.toBeInTheDocument();
  });

  it('omits brand and hcp_id from a legacy-model request', () => {
    const mockMutate = mockMutateHook();
    // beforeEach default mock = propensity (legacy) is first → default-selected

    render(<FeatureImportance />, { wrapper: createWrapper() });

    fireEvent.change(screen.getByPlaceholderText(/patient/i), {
      target: { value: 'patient_42' },
    });
    fireEvent.click(screen.getByRole('button', { name: /^explain$/i }));

    const arg = mockMutate.mock.calls[0][0];
    expect(arg.brand).toBeUndefined();
    expect(arg.hcp_id).toBeUndefined();
  });
});
