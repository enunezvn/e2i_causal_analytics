/**
 * SimulationPanel Component Tests
 * ================================
 *
 * Tests for the digital twin simulation configuration panel.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SimulationPanel, type SimulationPanelProps } from './SimulationPanel';
import { InterventionType, type SimulationRequest } from '@/types/digital-twin';

// Mock Radix UI components that need portal
vi.mock('@radix-ui/react-select', async () => {
  const actual = await vi.importActual('@radix-ui/react-select');
  return {
    ...actual,
    Portal: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  };
});

describe('SimulationPanel', () => {
  const mockOnSimulate = vi.fn();

  const defaultProps: SimulationPanelProps = {
    onSimulate: mockOnSimulate,
    isSimulating: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders the simulation configuration card', () => {
      render(<SimulationPanel {...defaultProps} />);

      expect(screen.getByText('Simulation Configuration')).toBeInTheDocument();
      expect(
        screen.getByText(/Configure and run digital twin simulations/)
      ).toBeInTheDocument();
    });

    it('renders intervention type selector', () => {
      render(<SimulationPanel {...defaultProps} />);

      expect(screen.getByText('Intervention Type')).toBeInTheDocument();
    });

    it('renders brand selector', () => {
      render(<SimulationPanel {...defaultProps} />);

      expect(screen.getByText('Target Brand')).toBeInTheDocument();
    });

    it('renders sample size slider with default value', () => {
      render(<SimulationPanel {...defaultProps} />);

      expect(screen.getByText(/Sample Size:/)).toBeInTheDocument();
      expect(screen.getByText('1,000')).toBeInTheDocument();
    });

    it('renders duration slider with default value', () => {
      render(<SimulationPanel {...defaultProps} />);

      expect(screen.getByText(/Duration:/)).toBeInTheDocument();
      expect(screen.getByText('90 days')).toBeInTheDocument();
    });

    it('renders run simulation button', () => {
      render(<SimulationPanel {...defaultProps} />);

      const button = screen.getByRole('button', { name: /Run Simulation/i });
      expect(button).toBeInTheDocument();
      expect(button).not.toBeDisabled();
    });

    it('renders with custom initial brand', () => {
      render(<SimulationPanel {...defaultProps} initialBrand="Kisqali" />);

      // Check that Kisqali is available as a brand option
      expect(screen.getByText('Target Brand')).toBeInTheDocument();
    });

    it('renders with custom brands list', () => {
      const customBrands = ['BrandA', 'BrandB', 'BrandC'];
      render(<SimulationPanel {...defaultProps} brands={customBrands} />);

      expect(screen.getByText('Target Brand')).toBeInTheDocument();
    });
  });

  describe('Loading State', () => {
    it('shows loading state when simulating', () => {
      render(<SimulationPanel {...defaultProps} isSimulating={true} />);

      const button = screen.getByRole('button', { name: /Running Simulation/i });
      expect(button).toBeInTheDocument();
      expect(button).toBeDisabled();
    });

    it('disables form submission when simulating', () => {
      render(<SimulationPanel {...defaultProps} isSimulating={true} />);

      const button = screen.getByRole('button', { name: /Running Simulation/i });
      expect(button).toBeDisabled();
    });
  });

  describe('Advanced Settings Toggle', () => {
    it('hides advanced settings by default', () => {
      render(<SimulationPanel {...defaultProps} />);

      expect(screen.queryByText('Budget (Optional)')).not.toBeInTheDocument();
      expect(screen.queryByText('Target Regions')).not.toBeInTheDocument();
      expect(screen.queryByText('HCP Segments')).not.toBeInTheDocument();
    });

    it('shows advanced settings when toggle clicked', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      const advancedButton = screen.getByRole('button', { name: /Advanced/i });
      await user.click(advancedButton);

      expect(screen.getByText('Budget (Optional)')).toBeInTheDocument();
      expect(screen.getByText('Target Regions')).toBeInTheDocument();
      expect(screen.getByText('HCP Segments')).toBeInTheDocument();
    });

    it('toggles back to basic view', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      const advancedButton = screen.getByRole('button', { name: /Advanced/i });
      await user.click(advancedButton);

      expect(screen.getByText('Budget (Optional)')).toBeInTheDocument();

      const basicButton = screen.getByRole('button', { name: /Basic/i });
      await user.click(basicButton);

      expect(screen.queryByText('Budget (Optional)')).not.toBeInTheDocument();
    });
  });

  describe('Form Submission', () => {
    it('calls onSimulate with correct request on form submission', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      const submitButton = screen.getByRole('button', { name: /Run Simulation/i });
      await user.click(submitButton);

      expect(mockOnSimulate).toHaveBeenCalledTimes(1);
      expect(mockOnSimulate).toHaveBeenCalledWith(
        expect.objectContaining({
          intervention_type: InterventionType.HCP_ENGAGEMENT,
          brand: 'Remibrutinib',
          sample_size: 1000,
          duration_days: 90,
        })
      );
    });

    it('does not include empty optional fields in request', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      const submitButton = screen.getByRole('button', { name: /Run Simulation/i });
      await user.click(submitButton);

      const callArg = mockOnSimulate.mock.calls[0][0] as SimulationRequest;
      expect(callArg.target_regions).toBeUndefined();
      expect(callArg.target_segments).toBeUndefined();
      expect(callArg.budget).toBeUndefined();
    });
  });

  describe('Accessibility', () => {
    it('has accessible form labels', () => {
      render(<SimulationPanel {...defaultProps} />);

      expect(screen.getByLabelText(/Intervention Type/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Target Brand/i)).toBeInTheDocument();
    });

    it('applies custom className', () => {
      const { container } = render(
        <SimulationPanel {...defaultProps} className="custom-class" />
      );

      expect(container.querySelector('.custom-class')).toBeInTheDocument();
    });
  });
});
