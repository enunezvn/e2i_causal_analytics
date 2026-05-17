/**
 * PredictiveAnalytics Page Tests
 * ==============================
 *
 * Tests for the live-data Predictive Analytics page that wires
 * to /api/models/predict/{model_name} via the predictions hooks.
 *
 * Acceptance criteria (issue #300):
 * - Model selector populated from useModelsStatus
 * - Form fields driven by useModelInfo(modelName).input_schema
 * - Run-prediction button invokes usePredict mutation
 * - Display prediction + confidence + feature contributions
 * - Loading/error states via QueryErrorState
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';
import PredictiveAnalytics from './PredictiveAnalytics';

// Mock predictions hooks
vi.mock('@/hooks/api/use-predictions', () => ({
  useModelsStatus: vi.fn(),
  useModelInfo: vi.fn(),
  usePredict: vi.fn(),
}));

import {
  useModelsStatus,
  useModelInfo,
  usePredict,
} from '@/hooks/api/use-predictions';

// Mock Recharts to skip canvas/SVG rendering
vi.mock('recharts', async () => {
  const actual = await vi.importActual('recharts');
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container" style={{ width: 800, height: 400 }}>
        {children}
      </div>
    ),
  };
});

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

// Sample mock data
const mockModelsStatus = {
  total_models: 2,
  healthy_count: 2,
  unhealthy_count: 0,
  models: [
    {
      model_name: 'churn_model',
      status: 'healthy',
      endpoint: 'http://localhost:8080/predictions/churn_model',
      last_check: '2026-01-04T10:00:00Z',
    },
    {
      model_name: 'conversion_model',
      status: 'healthy',
      endpoint: 'http://localhost:8080/predictions/conversion_model',
      last_check: '2026-01-04T10:00:00Z',
    },
  ],
  timestamp: '2026-01-04T10:00:00Z',
};

const mockModelInfo = {
  name: 'churn_model',
  version: '1.0.0',
  type: 'classification',
  description: 'Churn risk model',
  input_schema: {
    hcp_id: 'string',
    territory: 'string',
    specialty: 'string',
    visits_last_quarter: 'number',
  },
  metadata: {},
};

const mockPredictionResponse = {
  model_name: 'churn_model',
  prediction: 'high_risk',
  confidence: 0.87,
  feature_importance: {
    hcp_id: 0.05,
    territory: 0.21,
    specialty: 0.34,
    visits_last_quarter: 0.4,
  },
  latency_ms: 42,
  model_version: '1.0.0',
  timestamp: '2026-01-04T10:00:00Z',
};

describe('PredictiveAnalytics (live API)', () => {
  const mockMutate = vi.fn();

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
      mutate: mockMutate,
      data: undefined,
      isPending: false,
      isError: false,
      error: null,
      reset: vi.fn(),
    });
  });

  // ===========================================================================
  // AC 1: Model selector from useModelsStatus
  // ===========================================================================

  it('renders the page title', () => {
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    expect(screen.getByText('Predictive Analytics')).toBeInTheDocument();
  });

  it('calls useModelsStatus to populate the model selector', () => {
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    expect(useModelsStatus).toHaveBeenCalled();
  });

  it('shows models from useModelsStatus in the selector dropdown', async () => {
    const user = userEvent.setup();
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    const trigger = screen.getByRole('combobox', { name: /model/i });
    await user.click(trigger);

    await waitFor(() => {
      expect(
        screen.getByRole('option', { name: /churn_model/i })
      ).toBeInTheDocument();
    });
    expect(
      screen.getByRole('option', { name: /conversion_model/i })
    ).toBeInTheDocument();
  }, 15000);

  // ===========================================================================
  // AC 2: Form fields driven by useModelInfo input_schema
  // ===========================================================================

  it('calls useModelInfo with the selected model name', () => {
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    // useModelInfo should be called with churn_model (first healthy model)
    expect(useModelInfo).toHaveBeenCalledWith('churn_model');
  });

  it('renders an input field per feature in input_schema', () => {
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    // Each feature key should map to a labelled input
    expect(screen.getByLabelText(/hcp_id/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/territory/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/specialty/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/visits_last_quarter/i)).toBeInTheDocument();
  });

  // ===========================================================================
  // AC 3: Run-prediction button invokes usePredict
  // ===========================================================================

  it('invokes the usePredict mutation when the Run Prediction button is clicked', async () => {
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    // Fill at least one field so the request has feature data. fireEvent is
    // synchronous and avoids the per-keystroke delay of userEvent.type, which
    // matters because Radix UI re-renders after each change.
    const input = screen.getByLabelText(/hcp_id/i);
    fireEvent.change(input, { target: { value: 'HCP001' } });

    const runButton = screen.getByRole('button', { name: /run prediction/i });
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(mockMutate).toHaveBeenCalledTimes(1);
    });
    const callArgs = mockMutate.mock.calls[0][0];
    expect(callArgs.modelName).toBe('churn_model');
    expect(callArgs.request.features).toMatchObject({ hcp_id: 'HCP001' });
  });

  // ===========================================================================
  // AC 4: Display prediction + confidence + feature contributions
  // ===========================================================================

  it('displays the prediction value from the API response', () => {
    (usePredict as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      data: mockPredictionResponse,
      isPending: false,
      isError: false,
      error: null,
      reset: vi.fn(),
    });

    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    // Prediction value rendered on the page
    expect(screen.getByText(/high_risk/i)).toBeInTheDocument();
  });

  it('displays the confidence value from the API response', () => {
    (usePredict as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      data: mockPredictionResponse,
      isPending: false,
      isError: false,
      error: null,
      reset: vi.fn(),
    });

    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    // Confidence 0.87 -> 87%
    expect(screen.getByText(/87(\.0)?%/)).toBeInTheDocument();
  });

  it('renders feature_importance entries from the API response', () => {
    (usePredict as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      data: mockPredictionResponse,
      isPending: false,
      isError: false,
      error: null,
      reset: vi.fn(),
    });

    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    // "Feature Contributions" section title proves the section rendered
    expect(screen.getByText('Feature Contributions')).toBeInTheDocument();
    // Each feature key from feature_importance also appears as a percentage
    // (the keys themselves also appear as form labels, so we assert the
    // signed percentage strings rendered next to each contribution)
    expect(screen.getByText('+40.0%')).toBeInTheDocument(); // visits_last_quarter 0.4
    expect(screen.getByText('+34.0%')).toBeInTheDocument(); // specialty 0.34
    expect(screen.getByText('+21.0%')).toBeInTheDocument(); // territory 0.21
    expect(screen.getByText('+5.0%')).toBeInTheDocument(); // hcp_id 0.05
  });

  it('does NOT render synthetic risk score entities (Generate sample data removed)', () => {
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    // Sentinels from generateRiskScores() / generateUpliftSegments() / generateRecommendations()
    expect(screen.queryByText('Dr. Sarah Chen')).not.toBeInTheDocument();
    expect(screen.queryByText('Memorial Hospital')).not.toBeInTheDocument();
    expect(screen.queryByText('High-Value Responders')).not.toBeInTheDocument();
    expect(
      screen.queryByText('Focus on High-Value Responders Segment')
    ).not.toBeInTheDocument();
  });

  // ===========================================================================
  // AC 5: Loading/error states
  // ===========================================================================

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
    // QueryErrorState renders error.message in the description
    expect(screen.getByText('Failed to fetch models')).toBeInTheDocument();
  });

  it('renders a prediction error message when usePredict errors', () => {
    (usePredict as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      data: undefined,
      isPending: false,
      isError: true,
      error: new Error('Prediction service unavailable'),
      reset: vi.fn(),
    });

    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    // QueryErrorState renders error.message in the description
    expect(screen.getByText('Prediction service unavailable')).toBeInTheDocument();
  });

  it('disables the Run Prediction button while the mutation is pending', () => {
    (usePredict as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      data: undefined,
      isPending: true,
      isError: false,
      error: null,
      reset: vi.fn(),
    });

    render(<PredictiveAnalytics />, { wrapper: createWrapper() });
    const runButton = screen.getByRole('button', { name: /running|run prediction/i });
    expect(runButton).toBeDisabled();
  });

  // ===========================================================================
  // Empty / edge states
  // ===========================================================================

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
    expect(
      screen.getByText(/no models available|no models/i)
    ).toBeInTheDocument();
  });

  it('switches to a different model when the user picks another option', async () => {
    const user = userEvent.setup();
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    const trigger = screen.getByRole('combobox', { name: /model/i });
    await user.click(trigger);
    await waitFor(() => {
      expect(
        screen.getByRole('option', { name: /conversion_model/i })
      ).toBeInTheDocument();
    });
    const option = screen.getByRole('option', { name: /conversion_model/i });
    await user.click(option);

    await waitFor(() => {
      expect(useModelInfo).toHaveBeenCalledWith('conversion_model');
    });
  }, 15000);

  // ===========================================================================
  // Backend schema-shape fallback coverage (codex iter-1 LOW)
  // ===========================================================================

  it('falls back to metadata.feature_names when input_schema is absent', () => {
    (useModelInfo as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        name: 'churn_model',
        // No input_schema; only metadata.feature_names
        metadata: { feature_names: ['alpha', 'beta', 'gamma'] },
      },
      isLoading: false,
      error: null,
    });

    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    expect(screen.getByLabelText(/alpha/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/beta/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/gamma/i)).toBeInTheDocument();
  });

  it('falls back to metadata.input_schema with typed fields', () => {
    (useModelInfo as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        name: 'churn_model',
        metadata: {
          input_schema: {
            visits: 'number',
            territory: 'string',
          },
        },
      },
      isLoading: false,
      error: null,
    });

    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    expect(screen.getByLabelText(/visits/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/territory/i)).toBeInTheDocument();
  });

  // ===========================================================================
  // Type-aware coercion (codex iter-1 MED — string-typed id stays a string)
  // ===========================================================================

  it('does NOT coerce string-typed features to numbers when submitting', async () => {
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    // hcp_id is declared as 'string' in mockModelInfo.input_schema; a
    // numeric-looking input must stay a string in the mutation payload
    const input = screen.getByLabelText(/hcp_id/i);
    fireEvent.change(input, { target: { value: '12345' } });

    const runButton = screen.getByRole('button', { name: /run prediction/i });
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(mockMutate).toHaveBeenCalledTimes(1);
    });
    const features = mockMutate.mock.calls[0][0].request.features;
    expect(features.hcp_id).toBe('12345');
    expect(typeof features.hcp_id).toBe('string');
  });

  it('coerces number-typed features to numbers when submitting', async () => {
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    // visits_last_quarter is declared as 'number' — input must come
    // through as a number on the wire
    const input = screen.getByLabelText(/visits_last_quarter/i);
    fireEvent.change(input, { target: { value: '7' } });

    const runButton = screen.getByRole('button', { name: /run prediction/i });
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(mockMutate).toHaveBeenCalledTimes(1);
    });
    const features = mockMutate.mock.calls[0][0].request.features;
    expect(features.visits_last_quarter).toBe(7);
    expect(typeof features.visits_last_quarter).toBe('number');
  });

  // ===========================================================================
  // Stale-prediction reset on model change (codex iter-1 MED)
  // ===========================================================================

  it('resets the previous prediction when the user switches models', async () => {
    const reset = vi.fn();
    (usePredict as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      data: mockPredictionResponse,
      isPending: false,
      isError: false,
      error: null,
      reset,
    });

    const user = userEvent.setup();
    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    // Wait for initial mount to settle (reset() also fires on mount when
    // selectedModel becomes 'churn_model' via the useEffect default).
    await waitFor(() => {
      expect(reset).toHaveBeenCalled();
    });
    const callsBeforeSwitch = reset.mock.calls.length;

    const trigger = screen.getByRole('combobox', { name: /model/i });
    await user.click(trigger);
    await waitFor(() => {
      expect(
        screen.getByRole('option', { name: /conversion_model/i })
      ).toBeInTheDocument();
    });
    await user.click(screen.getByRole('option', { name: /conversion_model/i }));

    // The switch must trigger at least one additional reset() call;
    // weaker implementations that only reset on mount would NOT increase
    // the count.
    await waitFor(() => {
      expect(reset.mock.calls.length).toBeGreaterThan(callsBeforeSwitch);
    });
  }, 15000);

  // ===========================================================================
  // Stale prediction must NOT be shown alongside a fresh error (codex iter-2)
  // ===========================================================================

  it('hides stale prediction data when a retry errors', () => {
    (usePredict as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      // React Query keeps `data` from the last successful call; if a retry
      // errors we get both `data` (stale) and `isError: true` simultaneously
      data: mockPredictionResponse,
      isPending: false,
      isError: true,
      error: new Error('Prediction service unavailable'),
      reset: vi.fn(),
    });

    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    // Error banner is shown
    expect(screen.getByText('Prediction service unavailable')).toBeInTheDocument();
    // Stale prediction value must NOT also be on the page
    expect(screen.queryByText(/high_risk/i)).not.toBeInTheDocument();
    expect(screen.queryByText('Feature Contributions')).not.toBeInTheDocument();
  });

  // ===========================================================================
  // Top-level info.features fallback (codex iter-2 MED)
  // ===========================================================================

  it('falls back to top-level info.features when input_schema is absent', () => {
    (useModelInfo as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        name: 'churn_model',
        // No input_schema; only top-level info.features (matches backend
        // fixture in tests/api/test_predictions_endpoints.py)
        features: ['feature_a', 'feature_b', 'feature_c'],
      },
      isLoading: false,
      error: null,
    });

    render(<PredictiveAnalytics />, { wrapper: createWrapper() });

    expect(screen.getByLabelText(/feature_a/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/feature_b/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/feature_c/i)).toBeInTheDocument();
  });
});
