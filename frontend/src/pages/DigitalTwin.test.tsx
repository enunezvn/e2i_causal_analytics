/**
 * DigitalTwin Page Tests
 * ======================
 *
 * Tests for the Digital Twin simulation page.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import DigitalTwin from './DigitalTwin';
import { InterventionType, RecommendationType, ConfidenceLevel } from '@/types/digital-twin';

// Mock the digital twin hooks
vi.mock('@/hooks/api/use-digital-twin', () => ({
  useDigitalTwinHealth: vi.fn(),
  useSimulationHistory: vi.fn(),
  useRunSimulation: vi.fn(),
}));

import {
  useDigitalTwinHealth,
  useSimulationHistory,
  useRunSimulation,
} from '@/hooks/api/use-digital-twin';

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

// Sample data
const mockHealth = {
  status: 'healthy',
  model_version: '2.1.0',
  last_calibration: '2026-01-01',
};

const mockHistory = {
  simulations: [
    {
      simulation_id: 'sim-001',
      created_at: '2026-01-04T10:00:00Z',
      intervention_type: InterventionType.HCP_ENGAGEMENT,
      brand: 'Remibrutinib',
      ate_estimate: 0.18,
      recommendation_type: RecommendationType.DEPLOY,
    },
    {
      simulation_id: 'sim-002',
      created_at: '2026-01-03T14:30:00Z',
      intervention_type: InterventionType.DIGITAL_MARKETING,
      brand: 'Fabhalta',
      ate_estimate: 0.09,
      recommendation_type: RecommendationType.REFINE,
    },
  ],
  total: 2,
  offset: 0,
  limit: 10,
};

describe('DigitalTwin', () => {
  const mockMutate = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();

    // Default mock implementations
    (useDigitalTwinHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockHealth,
      isLoading: false,
    });
    (useSimulationHistory as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockHistory,
      isLoading: false,
    });
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
    });
  });

  it('renders page header with title and description', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByText('Digital Twin')).toBeInTheDocument();
    expect(screen.getByText('Intervention pre-screening and scenario analysis')).toBeInTheDocument();
  });

  it('displays system health status', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByText('Healthy')).toBeInTheDocument();
    expect(screen.getByText('Model v2.1.0')).toBeInTheDocument();
  });

  it('shows stat cards with metrics', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByText('Simulations Today')).toBeInTheDocument();
    expect(screen.getByText('Avg. Execution Time')).toBeInTheDocument();
    expect(screen.getByText('Deploy Rate')).toBeInTheDocument();
    // Model Fidelity appears in both stat cards and fidelity section
    expect(screen.getAllByText('Model Fidelity').length).toBeGreaterThan(0);
  });

  it('renders simulation configuration form', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByText('Configure Simulation')).toBeInTheDocument();
    expect(screen.getByText('Intervention Type')).toBeInTheDocument();
    expect(screen.getByText('Brand')).toBeInTheDocument();
    expect(screen.getByText('Sample Size')).toBeInTheDocument();
    expect(screen.getByText('Duration (days)')).toBeInTheDocument();
  });

  it('has run simulation button', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByRole('button', { name: /Run Simulation/i })).toBeInTheDocument();
  });

  it('displays Results and History tabs', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByRole('button', { name: /Results/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /History/i })).toBeInTheDocument();
  });

  it('shows sample simulation results by default', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });

    // Check for recommendation badge
    expect(screen.getByText('Deploy')).toBeInTheDocument();
    // Check for simulation outcomes section
    expect(screen.getByText('Simulation Outcomes')).toBeInTheDocument();
    expect(screen.getByText('ATE')).toBeInTheDocument();
    expect(screen.getByText('TRx Lift')).toBeInTheDocument();
    expect(screen.getByText('NRx Lift')).toBeInTheDocument();
    expect(screen.getByText('ROI')).toBeInTheDocument();
  });

  it('displays fidelity metrics', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });

    // Model Fidelity appears in both stat cards and fidelity section
    expect(screen.getAllByText('Model Fidelity').length).toBeGreaterThan(0);
    expect(screen.getByText('Data Coverage')).toBeInTheDocument();
    expect(screen.getByText('Calibration')).toBeInTheDocument();
    expect(screen.getByText('Temporal Alignment')).toBeInTheDocument();
    expect(screen.getByText('Feature Completeness')).toBeInTheDocument();
  });

  it('shows supporting evidence and risk factors', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByText('Supporting Evidence')).toBeInTheDocument();
    expect(screen.getByText('Risk Factors')).toBeInTheDocument();
  });

  it('switches to history tab and shows simulation history', async () => {
    const user = userEvent.setup();
    render(<DigitalTwin />, { wrapper: createWrapper() });

    const historyTab = screen.getByRole('button', { name: /History/i });
    await act(async () => {
      await user.click(historyTab);
    });

    // Wait for tab switch - look for history items from sample data
    // History shows intervention_type formatted (like "Hcp Engagement") and ATE values
    await waitFor(() => {
      // After switching to history tab, we should see more Remibrutinib references
      // (one in select dropdown + ones in history list)
      const remibElements = screen.getAllByText(/Remibrutinib/i);
      expect(remibElements.length).toBeGreaterThan(1);
    }, { timeout: 3000 });
  });

  it('allows selecting intervention type', async () => {
    const user = userEvent.setup();
    render(<DigitalTwin />, { wrapper: createWrapper() });

    // Get all comboboxes - first one is intervention type
    const selects = screen.getAllByRole('combobox');
    const interventionSelect = selects[0];
    await act(async () => {
      await user.selectOptions(interventionSelect, 'patient_support');
    });

    expect(interventionSelect).toHaveValue('patient_support');
  });

  it('allows selecting brand', async () => {
    const user = userEvent.setup();
    render(<DigitalTwin />, { wrapper: createWrapper() });

    const selects = screen.getAllByRole('combobox');
    const brandSelect = selects[1]; // Second select is brand
    await act(async () => {
      await user.selectOptions(brandSelect, 'Fabhalta');
    });

    expect(brandSelect).toHaveValue('Fabhalta');
  });

  it('allows entering sample size', async () => {
    const user = userEvent.setup();
    render(<DigitalTwin />, { wrapper: createWrapper() });

    // Get all spinbuttons - first one is sample size, second is duration
    const spinbuttons = screen.getAllByRole('spinbutton');
    const sampleSizeInput = spinbuttons[0];
    await act(async () => {
      await user.clear(sampleSizeInput);
      await user.type(sampleSizeInput, '2000');
    });

    expect(sampleSizeInput).toHaveValue(2000);
  });

  it('allows entering duration', async () => {
    const user = userEvent.setup();
    render(<DigitalTwin />, { wrapper: createWrapper() });

    // Get all spinbuttons - first one is sample size, second is duration
    const spinbuttons = screen.getAllByRole('spinbutton');
    const durationInput = spinbuttons[1];
    await act(async () => {
      await user.clear(durationInput);
      await user.type(durationInput, '60');
    });

    expect(durationInput).toHaveValue(60);
  });

  it('calls runSimulation when form is submitted', async () => {
    const user = userEvent.setup();
    render(<DigitalTwin />, { wrapper: createWrapper() });

    const submitButton = screen.getByRole('button', { name: /Run Simulation/i });
    await act(async () => {
      await user.click(submitButton);
    });

    expect(mockMutate).toHaveBeenCalledWith({
      intervention_type: InterventionType.HCP_ENGAGEMENT,
      brand: 'Remibrutinib',
      sample_size: 1000,
      duration_days: 90,
    });
  });

  it('shows loading state when simulation is running', () => {
    (useRunSimulation as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mockMutate,
      isPending: true,
    });

    render(<DigitalTwin />, { wrapper: createWrapper() });

    const button = screen.getByRole('button', { name: /Run Simulation/i });
    expect(button).toBeDisabled();
    expect(button.querySelector('.animate-spin')).toBeInTheDocument();
  });

  it('displays confidence intervals correctly', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });

    // Check for 95% CI format
    expect(screen.getAllByText(/95% CI:/i).length).toBeGreaterThan(0);
  });

  it('shows execution time in results', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByText(/Executed in.*ms/i)).toBeInTheDocument();
  });

  it('shows simulation ID in results', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByText(/Simulation ID:/i)).toBeInTheDocument();
  });

  it('renders about section with intervention types', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByText('About the Digital Twin')).toBeInTheDocument();
    expect(screen.getByText('Intervention Types')).toBeInTheDocument();
    expect(screen.getByText('How It Works')).toBeInTheDocument();
  });

  it('shows last calibration date', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByText(/Last model calibration:/i)).toBeInTheDocument();
  });

  it('handles unknown health status', () => {
    (useDigitalTwinHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
    });

    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByText('Unknown')).toBeInTheDocument();
  });

  it('displays recommendation badge with correct type', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });

    // Sample simulation has DEPLOY recommendation
    expect(screen.getByText('Deploy')).toBeInTheDocument();
  });

  it('shows expected value when available', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByText(/Expected Value:/i)).toBeInTheDocument();
  });

  it('displays rationale for recommendation', () => {
    render(<DigitalTwin />, { wrapper: createWrapper() });

    expect(screen.getByText(/Simulation indicates strong positive ATE/i)).toBeInTheDocument();
  });
});
