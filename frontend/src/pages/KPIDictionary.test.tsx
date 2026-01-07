/**
 * KPIDictionary Page Tests
 * ========================
 *
 * Tests for the KPI Dictionary reference page.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import KPIDictionary from './KPIDictionary';

// Mock the KPI hooks
vi.mock('@/hooks/api/use-kpi', () => ({
  useKPIList: vi.fn(),
  useWorkstreams: vi.fn(),
  useKPIHealth: vi.fn(),
}));

import { useKPIList, useWorkstreams, useKPIHealth } from '@/hooks/api/use-kpi';

// Create wrapper with QueryClientProvider
function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

// Sample KPI data for testing
const mockKPIs = [
  {
    id: 'WS1-DQ-001',
    name: 'Source Coverage - Patients',
    definition: 'Percentage of eligible patients present in source vs reference universe',
    formula: 'covered_patients / reference_patients',
    calculation_type: 'direct',
    workstream: 'ws1_data_quality',
    tables: ['patient_journeys', 'reference_universe'],
    columns: ['patient_id', 'coverage_status'],
    threshold: { target: 85, warning: 70, critical: 50 },
    unit: '%',
    frequency: 'daily',
    primary_causal_library: 'none',
  },
  {
    id: 'WS1-MP-001',
    name: 'ROC-AUC',
    definition: 'Area Under the ROC Curve',
    formula: '∫TPR d(FPR)',
    calculation_type: 'direct',
    workstream: 'ws1_model_performance',
    tables: ['ml_predictions'],
    columns: ['model_auc'],
    threshold: { target: 0.80, warning: 0.70, critical: 0.60 },
    frequency: 'daily',
    primary_causal_library: 'none',
  },
  {
    id: 'WS2-TR-001',
    name: 'Trigger Precision',
    definition: 'Percentage of fired triggers resulting in positive outcome',
    formula: 'true_positives / (true_positives + false_positives)',
    calculation_type: 'direct',
    workstream: 'ws2_triggers',
    tables: ['triggers'],
    columns: ['trigger_status'],
    threshold: { target: 70, warning: 55, critical: 40 },
    unit: '%',
    frequency: 'daily',
    primary_causal_library: 'dowhy',
  },
  {
    id: 'CM-001',
    name: 'Average Treatment Effect (ATE)',
    definition: 'Average causal effect of treatment on outcome',
    formula: 'E[Y(1) - Y(0)]',
    calculation_type: 'derived',
    workstream: 'causal_metrics',
    tables: ['ml_predictions'],
    columns: ['treatment_effect_estimate'],
    frequency: 'weekly',
    primary_causal_library: 'dowhy',
  },
];

const mockWorkstreams = [
  { id: 'ws1_data_quality', name: 'WS1: Data Quality', kpi_count: 1 },
  { id: 'ws1_model_performance', name: 'WS1: Model Performance', kpi_count: 1 },
  { id: 'ws2_triggers', name: 'WS2: Trigger Performance', kpi_count: 1 },
  { id: 'causal_metrics', name: 'Causal Metrics', kpi_count: 1 },
];

describe('KPIDictionary', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Default mock implementations
    (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { kpis: mockKPIs },
      isLoading: false,
    });
    (useWorkstreams as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { workstreams: mockWorkstreams },
    });
    (useKPIHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { status: 'healthy', registry_loaded: true },
    });
  });

  it('renders page header with title and description', () => {
    render(<KPIDictionary />, { wrapper: createWrapper() });

    expect(screen.getByText('KPI Dictionary')).toBeInTheDocument();
    expect(screen.getByText(/Complete reference of all/)).toBeInTheDocument();
  });

  it('displays stats cards with correct data', () => {
    render(<KPIDictionary />, { wrapper: createWrapper() });

    expect(screen.getByText('Total KPIs')).toBeInTheDocument();
    expect(screen.getByText('Workstreams')).toBeInTheDocument();
    expect(screen.getByText('Causal KPIs')).toBeInTheDocument();
    expect(screen.getByText('System Status')).toBeInTheDocument();
  });

  it('shows correct total KPI count', () => {
    render(<KPIDictionary />, { wrapper: createWrapper() });

    // Component uses SAMPLE_KPIS (46 KPIs) when API returns fewer than 46
    // Traverse up through parent divs to get the full card
    const totalKPIsLabel = screen.getByText('Total KPIs');
    const cardContainer = totalKPIsLabel.parentElement?.parentElement;
    expect(cardContainer).toHaveTextContent('46');
    expect(cardContainer).toHaveTextContent('Across all workstreams');
  });

  it('shows healthy system status', () => {
    render(<KPIDictionary />, { wrapper: createWrapper() });

    expect(screen.getByText('Healthy')).toBeInTheDocument();
    expect(screen.getByText('Registry: Loaded')).toBeInTheDocument();
  });

  it('renders workstream tabs', () => {
    render(<KPIDictionary />, { wrapper: createWrapper() });

    expect(screen.getByRole('tab', { name: /All KPIs/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /WS1: Data Quality/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /WS1: Model Performance/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /WS2: Trigger Performance/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Causal Metrics/i })).toBeInTheDocument();
  });

  it('displays all KPI cards initially', () => {
    render(<KPIDictionary />, { wrapper: createWrapper() });

    expect(screen.getByText('Source Coverage - Patients')).toBeInTheDocument();
    expect(screen.getByText('ROC-AUC')).toBeInTheDocument();
    expect(screen.getByText('Trigger Precision')).toBeInTheDocument();
    expect(screen.getByText('Average Treatment Effect (ATE)')).toBeInTheDocument();
  });

  it('shows KPI ID badges', () => {
    render(<KPIDictionary />, { wrapper: createWrapper() });

    expect(screen.getByText('WS1-DQ-001')).toBeInTheDocument();
    expect(screen.getByText('WS1-MP-001')).toBeInTheDocument();
    expect(screen.getByText('WS2-TR-001')).toBeInTheDocument();
    expect(screen.getByText('CM-001')).toBeInTheDocument();
  });

  it('displays KPI formulas', () => {
    render(<KPIDictionary />, { wrapper: createWrapper() });

    expect(screen.getByText('covered_patients / reference_patients')).toBeInTheDocument();
    expect(screen.getByText('∫TPR d(FPR)')).toBeInTheDocument();
    expect(screen.getByText('E[Y(1) - Y(0)]')).toBeInTheDocument();
  });

  it('shows threshold values', () => {
    render(<KPIDictionary />, { wrapper: createWrapper() });

    // Check for threshold labels
    expect(screen.getAllByText('Target:').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Warning:').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Critical:').length).toBeGreaterThan(0);
  });

  it('filters KPIs by search query', async () => {
    const user = userEvent.setup();
    render(<KPIDictionary />, { wrapper: createWrapper() });

    const searchInput = screen.getByPlaceholderText(/Search KPIs/i);
    await act(async () => {
      await user.type(searchInput, 'ROC');
    });

    await waitFor(() => {
      expect(screen.getByText('ROC-AUC')).toBeInTheDocument();
      expect(screen.queryByText('Source Coverage - Patients')).not.toBeInTheDocument();
    }, { timeout: 5000 });
  });

  it('filters KPIs by workstream tab', async () => {
    const user = userEvent.setup();
    render(<KPIDictionary />, { wrapper: createWrapper() });

    const causalTab = screen.getByRole('tab', { name: /Causal Metrics/i });
    await act(async () => {
      await user.click(causalTab);
    });

    await waitFor(() => {
      expect(screen.getByText('Average Treatment Effect (ATE)')).toBeInTheDocument();
    }, { timeout: 5000 });

    // Other KPIs should not be visible
    expect(screen.queryByText('Source Coverage - Patients')).not.toBeInTheDocument();
    expect(screen.queryByText('ROC-AUC')).not.toBeInTheDocument();
  });

  it('shows showing count in filter info', () => {
    render(<KPIDictionary />, { wrapper: createWrapper() });

    // Component uses SAMPLE_KPIS (46 KPIs) when API returns fewer than 46
    expect(screen.getByText(/Showing 46 of 46 KPIs/i)).toBeInTheDocument();
  });

  it('updates count when search filters results', async () => {
    const user = userEvent.setup();
    render(<KPIDictionary />, { wrapper: createWrapper() });

    const searchInput = screen.getByPlaceholderText(/Search KPIs/i);
    await act(async () => {
      await user.type(searchInput, 'ROC-AUC');
    });

    // Component uses SAMPLE_KPIS (46 KPIs) - searching for specific KPI shows fewer
    await waitFor(() => {
      expect(screen.getByText(/Showing \d+ of 46 KPIs/i)).toBeInTheDocument();
      // Should show fewer than 46 after filtering
      expect(screen.queryByText(/Showing 46 of 46 KPIs/i)).not.toBeInTheDocument();
    }, { timeout: 5000 });
  });

  it('shows empty state when no KPIs match search', async () => {
    const user = userEvent.setup();
    render(<KPIDictionary />, { wrapper: createWrapper() });

    const searchInput = screen.getByPlaceholderText(/Search KPIs/i);
    await act(async () => {
      await user.type(searchInput, 'nonexistentkpi123');
    });

    await waitFor(() => {
      expect(screen.getByText('No KPIs found')).toBeInTheDocument();
      expect(screen.getByText('Try adjusting your search or filter criteria')).toBeInTheDocument();
    }, { timeout: 5000 });
  });

  it('shows loading state when fetching KPIs', () => {
    (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: true,
    });

    render(<KPIDictionary />, { wrapper: createWrapper() });

    // Look for loading spinner
    const spinner = document.querySelector('.animate-spin');
    expect(spinner).toBeInTheDocument();
  });

  it('uses sample data when API returns nothing', () => {
    (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
    });
    (useWorkstreams as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
    });

    render(<KPIDictionary />, { wrapper: createWrapper() });

    // Should still show content from sample KPIs
    expect(screen.getByText('KPI Dictionary')).toBeInTheDocument();
    // Sample data has 20 KPIs
    expect(screen.getByText(/Showing.*of.*KPIs/i)).toBeInTheDocument();
  });

  it('shows causal library badge on KPIs using DoWhy/EconML', () => {
    render(<KPIDictionary />, { wrapper: createWrapper() });

    // Check for library mentions in the KPI cards
    expect(screen.getAllByText('dowhy').length).toBeGreaterThan(0);
  });

  it('displays table information in KPI cards', () => {
    render(<KPIDictionary />, { wrapper: createWrapper() });

    // Multiple KPIs may use the same tables, use getAllByText
    expect(screen.getAllByText(/patient_journeys/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/ml_predictions/).length).toBeGreaterThan(0);
  });

  it('shows frequency information', () => {
    render(<KPIDictionary />, { wrapper: createWrapper() });

    // Component uses SAMPLE_KPIS (46 KPIs) with various frequencies
    expect(screen.getAllByText('daily').length).toBeGreaterThan(0);
    expect(screen.getAllByText('weekly').length).toBeGreaterThan(0);
  });

  it('renders footer info about thresholds', () => {
    render(<KPIDictionary />, { wrapper: createWrapper() });

    expect(screen.getByText('About KPI Thresholds')).toBeInTheDocument();
    expect(screen.getByText(/Each KPI has configurable thresholds/)).toBeInTheDocument();
  });

  it('counts causal-enabled KPIs correctly', () => {
    render(<KPIDictionary />, { wrapper: createWrapper() });

    // Component uses SAMPLE_KPIS (46 KPIs) - 19 use dowhy/econml
    // Find the Causal KPIs stat card by traversing up to the card container
    const causalLabel = screen.getByText('Causal KPIs');
    // Get the parent card container (goes up: span -> div.flex -> div.card)
    const causalCard = causalLabel.closest('.bg-\\[var\\(--color-card\\)\\]');
    expect(causalCard).toHaveTextContent('19');
    expect(causalCard).toHaveTextContent('Using DoWhy/EconML');
  });

  it('handles unknown system status gracefully', () => {
    (useKPIHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
    });

    render(<KPIDictionary />, { wrapper: createWrapper() });

    expect(screen.getByText('Unknown')).toBeInTheDocument();
    expect(screen.getByText('Registry: Unknown')).toBeInTheDocument();
  });

  it('shows degraded status with warning variant', () => {
    (useKPIHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { status: 'degraded', registry_loaded: true },
    });

    render(<KPIDictionary />, { wrapper: createWrapper() });

    expect(screen.getByText('degraded')).toBeInTheDocument();
  });

  it('clears search when clearing input', async () => {
    const user = userEvent.setup();
    render(<KPIDictionary />, { wrapper: createWrapper() });

    const searchInput = screen.getByPlaceholderText(/Search KPIs/i);

    // Type to filter
    await act(async () => {
      await user.type(searchInput, 'ROC');
    });

    await waitFor(() => {
      expect(screen.queryByText('Source Coverage - Patients')).not.toBeInTheDocument();
    }, { timeout: 5000 });

    // Clear the search
    await act(async () => {
      await user.clear(searchInput);
    });

    await waitFor(() => {
      expect(screen.getByText('Source Coverage - Patients')).toBeInTheDocument();
      expect(screen.getByText('ROC-AUC')).toBeInTheDocument();
    }, { timeout: 5000 });
  });

  it('searches by KPI definition', async () => {
    const user = userEvent.setup();
    render(<KPIDictionary />, { wrapper: createWrapper() });

    const searchInput = screen.getByPlaceholderText(/Search KPIs/i);
    await act(async () => {
      await user.type(searchInput, 'causal effect');
    });

    await waitFor(() => {
      expect(screen.getByText('Average Treatment Effect (ATE)')).toBeInTheDocument();
      expect(screen.queryByText('ROC-AUC')).not.toBeInTheDocument();
    }, { timeout: 5000 });
  });

  it('displays All KPIs tab as active by default', () => {
    render(<KPIDictionary />, { wrapper: createWrapper() });

    const allTab = screen.getByRole('tab', { name: /All KPIs/i });
    expect(allTab).toHaveAttribute('data-state', 'active');
  });
});
