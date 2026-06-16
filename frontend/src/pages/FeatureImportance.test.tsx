/**
 * FeatureImportance Page Tests
 * ============================
 *
 * Tests for the Feature Importance page (#39): cohort-level (global) SHAP
 * importance + per-entity drill-down across the gold-standard cohort models.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import FeatureImportance from './FeatureImportance';

// Mock the explain hooks so we can control returned values
vi.mock('@/hooks/api/use-explain', () => ({
  useExplain: vi.fn(),
  useExplainableModels: vi.fn(),
  useExplanationHistory: vi.fn(),
  useGlobalFeatureImportance: vi.fn(),
  useSampleEntities: vi.fn(),
}));

import {
  useExplain,
  useExplainableModels,
  useExplanationHistory,
  useGlobalFeatureImportance,
  useSampleEntities,
} from '@/hooks/api/use-explain';

const mockCreateObjectURL = vi.fn(() => 'blob:mock-url');
const mockRevokeObjectURL = vi.fn();

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

// Gold-standard cohorts + one legacy demo type that MUST be filtered out.
const mockExplainableModels = {
  supported_models: [
    {
      model_type: 'initiation',
      latest_version: '1.0',
      explainer_type: 'LinearExplainer' as const,
      is_gold_standard: true,
      description: 'SHAP explanations for initiation predictions',
    },
    {
      model_type: 'persistence',
      latest_version: '1.0',
      explainer_type: 'LinearExplainer' as const,
      is_gold_standard: true,
    },
    {
      model_type: 'discontinuation',
      latest_version: '1.0',
      explainer_type: 'LinearExplainer' as const,
      is_gold_standard: true,
    },
    {
      model_type: 'hcp_adoption',
      latest_version: null,
      explainer_type: 'LinearExplainer' as const,
      is_gold_standard: true,
    },
    {
      model_type: 'propensity',
      latest_version: null,
      explainer_type: 'TreeExplainer' as const,
      is_gold_standard: false,
    },
  ],
  total_models: 5,
};

const mockGlobal = {
  model_type: 'initiation',
  brand: 'Remibrutinib',
  model_name: 'initiation_remibrutinib_goldstd_lr_v1',
  base_value: -0.72,
  sample_size: 30,
  requested_sample_size: 30,
  computation_method: 'LinearExplainer',
  computed_at: '2026-06-15T22:00:00Z',
  cached: true,
  features: [
    {
      feature_name: 'disease_severity',
      mean_abs_shap: 0.81,
      mean_shap: 0.8,
      mean_feature_value: 5.2,
      contribution_rank: 1,
    },
    {
      feature_name: 'geographic_region_northeast',
      mean_abs_shap: 0.14,
      mean_shap: -0.14,
      mean_feature_value: 0.3,
      contribution_rank: 2,
    },
  ],
  points: [
    { feature_name: 'disease_severity', shap_value: 0.7, feature_value: 5.0 },
    { feature_name: 'disease_severity', shap_value: 0.9, feature_value: 6.0 },
    { feature_name: 'geographic_region_northeast', shap_value: -0.1, feature_value: 0 },
  ],
};

const mockSampleEntities = {
  model_type: 'initiation',
  grain: 'patient',
  id_field: 'patient_id',
  entities: ['scvpt_000000', 'scvpt_000001'],
};

const mockExplainResponse = {
  explanation_id: 'expl_test_123',
  request_timestamp: '2026-05-17T10:00:00Z',
  patient_id: 'scvpt_000000',
  model_type: 'initiation',
  model_version_id: 'initiation_remibrutinib_goldstd_lr_v1',
  prediction_class: 'high_propensity',
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

const mockMutate = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  global.URL.createObjectURL = mockCreateObjectURL;
  global.URL.revokeObjectURL = mockRevokeObjectURL;

  (useExplainableModels as ReturnType<typeof vi.fn>).mockReturnValue({
    data: mockExplainableModels,
    isLoading: false,
    isError: false,
    error: null,
  });

  (useGlobalFeatureImportance as ReturnType<typeof vi.fn>).mockReturnValue({
    data: mockGlobal,
    isLoading: false,
    isFetching: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
  });

  (useSampleEntities as ReturnType<typeof vi.fn>).mockReturnValue({
    data: mockSampleEntities,
    isLoading: false,
    isError: false,
  });

  (useExplain as ReturnType<typeof vi.fn>).mockReturnValue({
    mutate: mockMutate,
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

vi.setConfig({ testTimeout: 15000 });

describe('FeatureImportance — page chrome', () => {
  it('renders header title', () => {
    render(<FeatureImportance />, { wrapper: createWrapper() });
    expect(screen.getByText('Feature Importance')).toBeInTheDocument();
  });

  it('renders Cohort and Individual mode tabs (cohort default)', () => {
    render(<FeatureImportance />, { wrapper: createWrapper() });
    expect(screen.getByRole('tab', { name: /Cohort \(global\)/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Individual/i })).toBeInTheDocument();
  });

  it('renders cohort model + brand selectors', () => {
    render(<FeatureImportance />, { wrapper: createWrapper() });
    // model + brand selects (both comboboxes)
    expect(screen.getAllByRole('combobox').length).toBeGreaterThanOrEqual(2);
    // brand default shown
    expect(screen.getAllByText(/Remibrutinib/i).length).toBeGreaterThanOrEqual(1);
  });

  it('filters out legacy demo models — only gold-standard cohorts are selectable', async () => {
    render(<FeatureImportance />, { wrapper: createWrapper() });
    fireEvent.click(screen.getAllByRole('combobox')[0]);
    await waitFor(() => {
      expect(screen.getAllByText(/Initiation/i).length).toBeGreaterThanOrEqual(1);
    });
    // Propensity (legacy, no deployed model) must NOT appear as an option
    expect(screen.queryByText(/Propensity/i)).not.toBeInTheDocument();
  });

  it('displays export button', () => {
    render(<FeatureImportance />, { wrapper: createWrapper() });
    expect(screen.getByRole('button', { name: /Export/i })).toBeInTheDocument();
  });
});

describe('FeatureImportance — cohort (global) mode', () => {
  it('renders global feature importance with mean |SHAP| and net direction', () => {
    render(<FeatureImportance />, { wrapper: createWrapper() });
    // disease_severity: mean_abs 0.81, net positive -> +0.8100
    expect(screen.getAllByText(/disease severity/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('+0.8100')).toBeInTheDocument();
    // northeast: mean_abs 0.14, net negative -> -0.1400
    expect(screen.getByText('-0.1400')).toBeInTheDocument();
  });

  it('uses the real cohort base value from the response', () => {
    render(<FeatureImportance />, { wrapper: createWrapper() });
    expect(screen.getByText('-0.720')).toBeInTheDocument();
  });

  it('shows sample size (honest n) and model name', () => {
    render(<FeatureImportance />, { wrapper: createWrapper() });
    expect(screen.getByText(/n = 30/i)).toBeInTheDocument();
    expect(screen.getByText('initiation_remibrutinib_goldstd_lr_v1')).toBeInTheDocument();
  });

  it('hides Waterfall and History tabs in cohort mode', () => {
    render(<FeatureImportance />, { wrapper: createWrapper() });
    expect(screen.queryByRole('tab', { name: /Waterfall/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: /History/i })).not.toBeInTheDocument();
  });

  it('shows a loading state while the cohort aggregate is computing', () => {
    (useGlobalFeatureImportance as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: true,
      isFetching: true,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    render(<FeatureImportance />, { wrapper: createWrapper() });
    expect(screen.getByText(/Computing cohort feature importance/i)).toBeInTheDocument();
  });

  it('shows an error state when the cohort aggregate fails', () => {
    (useGlobalFeatureImportance as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      isFetching: false,
      isError: true,
      error: { message: 'serving unavailable' },
      refetch: vi.fn(),
    });
    render(<FeatureImportance />, { wrapper: createWrapper() });
    expect(screen.getByText(/serving unavailable/i)).toBeInTheDocument();
  });
});

describe('FeatureImportance — individual mode', () => {
  it('exposes a real-ID picker and auto-runs the explanation', async () => {
    const user = userEvent.setup();
    render(<FeatureImportance />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Individual/i }));

    // Entity picker labeled "Patient ID" for the patient cohort
    await waitFor(() => {
      expect(screen.getByText('Patient ID')).toBeInTheDocument();
    });

    // Auto-run fired with the first real ID + brand + cohort
    await waitFor(() => {
      expect(mockMutate).toHaveBeenCalled();
    });
    const callArg = mockMutate.mock.calls[mockMutate.mock.calls.length - 1][0];
    expect(callArg).toMatchObject({
      patient_id: 'scvpt_000000',
      model_type: 'initiation',
      brand: 'Remibrutinib',
      format: 'top_k',
      top_k: 10,
    });
  });

  it('renders Waterfall + History tabs in individual mode', async () => {
    const user = userEvent.setup();
    render(<FeatureImportance />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Individual/i }));
    await waitFor(() => {
      expect(screen.getByRole('tab', { name: /Waterfall/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /History/i })).toBeInTheDocument();
    });
  });

  it('renders real SHAP values from the per-entity response (not fabricated)', async () => {
    const user = userEvent.setup();
    (useExplain as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      data: mockExplainResponse,
      isPending: false,
      isError: false,
      error: null,
      reset: vi.fn(),
    });
    render(<FeatureImportance />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Individual/i }));

    await waitFor(() => {
      expect(screen.getAllByText(/live feature alpha/i).length).toBeGreaterThanOrEqual(1);
    });
    expect(screen.getByText('+0.5100')).toBeInTheDocument();
    expect(screen.getByText('-0.3300')).toBeInTheDocument();
    // Legacy hard-coded synthetic markers MUST NOT be present
    expect(screen.queryByText('days since last visit')).not.toBeInTheDocument();
    expect(screen.queryByText('+0.3500')).not.toBeInTheDocument();
  });

  it('shows an error state when the per-entity explanation fails', async () => {
    const user = userEvent.setup();
    (useExplain as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      data: undefined,
      isPending: false,
      isError: true,
      error: { message: 'patient not found' },
      reset: vi.fn(),
    });
    render(<FeatureImportance />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Individual/i }));
    await waitFor(() => {
      expect(screen.getByText(/patient not found/i)).toBeInTheDocument();
    });
  });
});

// =============================================================================
// BRAND SELECTOR + PER-COHORT ID LABEL TESTS (Issue #967, in #985's terms)
// =============================================================================
//
// The gold-standard per-brand serving exposes 12 models (4 cohorts × 3 brands).
// The #985 redesign surfaces all of them: a brand selector picks the per-brand
// model, the chosen brand rides every explain request, and the HCP-grain cohort
// (`hcp_adoption`) relabels the ID picker to "HCP ID" and sends `hcp_id`.
//
// These tests are #967's behaviors translated to the #985 component shape:
//   - brand selector = a shadcn <Select> (role=combobox), NOT a native <select>;
//   - the entity is PICKED from a real-ID dropdown, NOT typed in a free-text box;
//   - the explanation AUTO-RUNS on (entity, cohort, brand) change — there is no
//     "Explain" button to click.
// Each assertion targets the REAL rendered DOM of the redesigned page.

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

/** Real HCP IDs returned by `/api/explain/sample-entities` for the HCP cohort. */
const mockHcpEntities = {
  model_type: 'hcp_adoption',
  grain: 'hcp',
  id_field: 'hcp_id',
  entities: ['HCP-NE-5678', 'HCP-NE-5679'],
};

describe('FeatureImportance — brand selector + per-cohort ID label (#967)', () => {
  it('renders a brand selector offering all three gold-standard brands', async () => {
    (useExplainableModels as ReturnType<typeof vi.fn>).mockReturnValue({
      data: goldstdModels('initiation'),
      isLoading: false,
      isError: false,
      error: null,
    });

    render(<FeatureImportance />, { wrapper: createWrapper() });

    // The redesign renders the brand as a shadcn <Select> (the SECOND combobox,
    // after the cohort selector). Open it and assert all three brand options.
    const comboboxes = screen.getAllByRole('combobox');
    expect(comboboxes.length).toBeGreaterThanOrEqual(2);
    fireEvent.click(comboboxes[1]);
    await waitFor(() => {
      expect(screen.getAllByText('Fabhalta').length).toBeGreaterThanOrEqual(1);
    });
    expect(screen.getAllByText('Remibrutinib').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Kisqali').length).toBeGreaterThanOrEqual(1);
  });

  it('only ever offers gold-standard cohorts (legacy demo types are filtered out)', async () => {
    // beforeEach default mock includes `propensity` (legacy, is_gold_standard:false).
    // The redesign filters cohortModels to gold-standard only, so the brand
    // selector is always applicable — there is no reachable non-gold cohort.
    render(<FeatureImportance />, { wrapper: createWrapper() });

    // Brand selector is present (gold cohort is default-selected).
    expect(screen.getAllByText('Remibrutinib').length).toBeGreaterThanOrEqual(1);

    // The cohort selector exposes ONLY gold cohorts — propensity must be absent.
    fireEvent.click(screen.getAllByRole('combobox')[0]);
    await waitFor(() => {
      expect(screen.getAllByText(/Initiation/i).length).toBeGreaterThanOrEqual(1);
    });
    expect(screen.queryByText(/Propensity/i)).not.toBeInTheDocument();
  });

  it('threads the (default) brand into the auto-run explain request for a gold-standard cohort', async () => {
    const user = userEvent.setup();
    (useExplainableModels as ReturnType<typeof vi.fn>).mockReturnValue({
      data: goldstdModels('persistence'),
      isLoading: false,
      isError: false,
      error: null,
    });

    render(<FeatureImportance />, { wrapper: createWrapper() });

    // Individual mode auto-runs the explanation for the first real entity ID.
    await user.click(screen.getByRole('tab', { name: /Individual/i }));
    await waitFor(() => {
      expect(mockMutate).toHaveBeenCalled();
    });
    const arg = mockMutate.mock.calls[mockMutate.mock.calls.length - 1][0];
    expect(arg).toMatchObject({
      model_type: 'persistence',
      brand: 'Remibrutinib',
      format: 'top_k',
      top_k: 10,
    });
  });

  it('relabels the ID picker to "HCP ID" and sends hcp_id for the hcp_adoption cohort', async () => {
    const user = userEvent.setup();
    (useExplainableModels as ReturnType<typeof vi.fn>).mockReturnValue({
      data: goldstdModels('hcp_adoption'),
      isLoading: false,
      isError: false,
      error: null,
    });
    // The HCP cohort's sample entities come back as real HCP IDs.
    (useSampleEntities as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockHcpEntities,
      isLoading: false,
      isError: false,
    });

    render(<FeatureImportance />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Individual/i }));

    // The ID picker is relabeled to "HCP ID" — not "Patient ID".
    await waitFor(() => {
      expect(screen.getByText('HCP ID')).toBeInTheDocument();
    });
    expect(screen.queryByText('Patient ID')).not.toBeInTheDocument();

    // Auto-run sends the entered value as BOTH patient_id and hcp_id so Feast
    // keys the lookup on hcp_id for this cohort.
    await waitFor(() => {
      expect(mockMutate).toHaveBeenCalled();
    });
    const arg = mockMutate.mock.calls[mockMutate.mock.calls.length - 1][0];
    expect(arg).toMatchObject({
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
    // A cohort aggregate is present so the summary card (which renders the
    // formatted cohort label) is visible.
    (useGlobalFeatureImportance as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { ...mockGlobal, model_type: 'hcp_adoption' },
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });

    render(<FeatureImportance />, { wrapper: createWrapper() });

    // Rendered in both the cohort-Select trigger and the summary card.
    expect(screen.getAllByText('HCP Adoption').length).toBeGreaterThanOrEqual(1);
    // The naive Title-case ("Hcp Adoption") must NOT appear anywhere.
    expect(screen.queryByText('Hcp Adoption')).not.toBeInTheDocument();
  });

  it('keys the patient cohorts on patient_id (no hcp_id) in the explain request', async () => {
    const user = userEvent.setup();
    (useExplainableModels as ReturnType<typeof vi.fn>).mockReturnValue({
      data: goldstdModels('initiation'),
      isLoading: false,
      isError: false,
      error: null,
    });

    render(<FeatureImportance />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Individual/i }));

    await waitFor(() => {
      expect(mockMutate).toHaveBeenCalled();
    });
    const arg = mockMutate.mock.calls[mockMutate.mock.calls.length - 1][0];
    // Patient-grain cohort: patient_id is set, hcp_id is omitted.
    expect(arg.patient_id).toBe('scvpt_000000');
    expect(arg.hcp_id).toBeUndefined();
    expect(arg.model_type).toBe('initiation');
  });
});
