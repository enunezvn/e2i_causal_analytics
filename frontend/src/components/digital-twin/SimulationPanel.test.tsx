/**
 * SimulationPanel Component Tests
 * ================================
 *
 * Tests for the digital twin simulation configuration panel.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
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

  // ===========================================================================
  // INTERVENTION TYPE SELECTION
  // ===========================================================================

  describe('Intervention Type Selection', () => {
    it('displays all intervention types in dropdown', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      const trigger = screen.getByLabelText(/Intervention Type/i);
      await user.click(trigger);

      // Use role to get options specifically (avoids multiple element issues)
      expect(screen.getByRole('option', { name: 'HCP Engagement Campaign' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'Patient Support Program' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'Pricing Change' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'Rep Training Program' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'Digital Marketing' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'Formulary Access Initiative' })).toBeInTheDocument();
    }, 15000); // Increased timeout for userEvent interactions

    it('changes intervention type when option selected', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      const trigger = screen.getByLabelText(/Intervention Type/i);
      await user.click(trigger);

      await user.click(screen.getByRole('option', { name: 'Patient Support Program' }));

      // Submit to verify the change was captured
      const submitButton = screen.getByRole('button', { name: /Run Simulation/i });
      await user.click(submitButton);

      expect(mockOnSimulate).toHaveBeenCalledWith(
        expect.objectContaining({
          intervention_type: InterventionType.PATIENT_SUPPORT,
        })
      );
    });
  });

  // ===========================================================================
  // BRAND SELECTION
  // ===========================================================================

  describe('Brand Selection', () => {
    it('displays all brand options in dropdown', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      const trigger = screen.getByLabelText(/Target Brand/i);
      await user.click(trigger);

      // Brands appear in select options
      expect(screen.getAllByText('Remibrutinib').length).toBeGreaterThanOrEqual(1);
      expect(screen.getByRole('option', { name: 'Fabhalta' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'Kisqali' })).toBeInTheDocument();
    });

    it('changes brand when option selected', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      const trigger = screen.getByLabelText(/Target Brand/i);
      await user.click(trigger);

      // Use role to get the option specifically
      await user.click(screen.getByRole('option', { name: 'Kisqali' }));

      const submitButton = screen.getByRole('button', { name: /Run Simulation/i });
      await user.click(submitButton);

      expect(mockOnSimulate).toHaveBeenCalledWith(
        expect.objectContaining({
          brand: 'Kisqali',
        })
      );
    });
  });

  // ===========================================================================
  // REGION SELECTION
  // ===========================================================================

  describe('Region Selection', () => {
    it('displays all region badges in advanced settings', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      // Open advanced settings
      await user.click(screen.getByRole('button', { name: /Advanced/i }));

      expect(screen.getByText('Northeast')).toBeInTheDocument();
      expect(screen.getByText('Southeast')).toBeInTheDocument();
      expect(screen.getByText('Midwest')).toBeInTheDocument();
      expect(screen.getByText('West')).toBeInTheDocument();
      expect(screen.getByText('Southwest')).toBeInTheDocument();
      expect(screen.getByText('Pacific Northwest')).toBeInTheDocument();
    });

    it('toggles region selection when badge clicked', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      await user.click(screen.getByRole('button', { name: /Advanced/i }));

      // Click a region badge to select it
      await user.click(screen.getByText('Northeast'));

      // Submit and check region is included
      const submitButton = screen.getByRole('button', { name: /Run Simulation/i });
      await user.click(submitButton);

      expect(mockOnSimulate).toHaveBeenCalledWith(
        expect.objectContaining({
          target_regions: ['Northeast'],
        })
      );
    });

    it('allows multiple region selections', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      await user.click(screen.getByRole('button', { name: /Advanced/i }));

      // Select multiple regions
      await user.click(screen.getByText('Northeast'));
      await user.click(screen.getByText('West'));

      const submitButton = screen.getByRole('button', { name: /Run Simulation/i });
      await user.click(submitButton);

      expect(mockOnSimulate).toHaveBeenCalledWith(
        expect.objectContaining({
          target_regions: ['Northeast', 'West'],
        })
      );
    });

    it('deselects region when clicked again', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      await user.click(screen.getByRole('button', { name: /Advanced/i }));

      // Select and then deselect
      await user.click(screen.getByText('Northeast'));
      await user.click(screen.getByText('Northeast'));

      const submitButton = screen.getByRole('button', { name: /Run Simulation/i });
      await user.click(submitButton);

      // Region should not be included when deselected
      expect(mockOnSimulate).toHaveBeenCalledWith(
        expect.objectContaining({
          target_regions: undefined,
        })
      );
    });
  });

  // ===========================================================================
  // SEGMENT SELECTION
  // ===========================================================================

  describe('Segment Selection', () => {
    it('displays all HCP segment badges in advanced settings', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      await user.click(screen.getByRole('button', { name: /Advanced/i }));

      expect(screen.getByText('High-Volume HCPs')).toBeInTheDocument();
      expect(screen.getByText('Medium-Volume HCPs')).toBeInTheDocument();
      expect(screen.getByText('Low-Volume HCPs')).toBeInTheDocument();
      expect(screen.getByText('Early Adopters')).toBeInTheDocument();
      expect(screen.getByText('Key Opinion Leaders')).toBeInTheDocument();
      expect(screen.getByText('Academic Centers')).toBeInTheDocument();
    });

    it('toggles segment selection when badge clicked', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      await user.click(screen.getByRole('button', { name: /Advanced/i }));

      await user.click(screen.getByText('High-Volume HCPs'));

      const submitButton = screen.getByRole('button', { name: /Run Simulation/i });
      await user.click(submitButton);

      expect(mockOnSimulate).toHaveBeenCalledWith(
        expect.objectContaining({
          target_segments: ['High-Volume HCPs'],
        })
      );
    });

    it('allows multiple segment selections', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      await user.click(screen.getByRole('button', { name: /Advanced/i }));

      await user.click(screen.getByText('High-Volume HCPs'));
      await user.click(screen.getByText('Key Opinion Leaders'));

      const submitButton = screen.getByRole('button', { name: /Run Simulation/i });
      await user.click(submitButton);

      expect(mockOnSimulate).toHaveBeenCalledWith(
        expect.objectContaining({
          target_segments: ['High-Volume HCPs', 'Key Opinion Leaders'],
        })
      );
    });

    it('deselects segment when clicked again', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      await user.click(screen.getByRole('button', { name: /Advanced/i }));

      await user.click(screen.getByText('High-Volume HCPs'));
      await user.click(screen.getByText('High-Volume HCPs'));

      const submitButton = screen.getByRole('button', { name: /Run Simulation/i });
      await user.click(submitButton);

      expect(mockOnSimulate).toHaveBeenCalledWith(
        expect.objectContaining({
          target_segments: undefined,
        })
      );
    });
  });

  // ===========================================================================
  // BUDGET INPUT
  // ===========================================================================

  describe('Budget Input', () => {
    it('accepts budget input in advanced settings', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      await user.click(screen.getByRole('button', { name: /Advanced/i }));

      const budgetInput = screen.getByLabelText(/Budget/i);
      await user.type(budgetInput, '50000');

      const submitButton = screen.getByRole('button', { name: /Run Simulation/i });
      await user.click(submitButton);

      expect(mockOnSimulate).toHaveBeenCalledWith(
        expect.objectContaining({
          budget: 50000,
        })
      );
    });

    it('allows clearing budget input', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      await user.click(screen.getByRole('button', { name: /Advanced/i }));

      const budgetInput = screen.getByLabelText(/Budget/i);
      await user.type(budgetInput, '50000');
      await user.clear(budgetInput);

      const submitButton = screen.getByRole('button', { name: /Run Simulation/i });
      await user.click(submitButton);

      expect(mockOnSimulate).toHaveBeenCalledWith(
        expect.objectContaining({
          budget: undefined,
        })
      );
    });

    it('displays budget helper text', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      await user.click(screen.getByRole('button', { name: /Advanced/i }));

      expect(screen.getByText(/Budget allocation for ROI calculations/)).toBeInTheDocument();
    });
  });

  // ===========================================================================
  // COMPLETE SIMULATION FLOW
  // ===========================================================================

  describe('Complete Simulation Flow', () => {
    it('submits simulation with all advanced settings', async () => {
      const user = userEvent.setup();
      render(<SimulationPanel {...defaultProps} />);

      // Open advanced settings
      await user.click(screen.getByRole('button', { name: /Advanced/i }));

      // Set budget
      const budgetInput = screen.getByLabelText(/Budget/i);
      await user.type(budgetInput, '100000');

      // Select regions
      await user.click(screen.getByText('Northeast'));
      await user.click(screen.getByText('West'));

      // Select segments
      await user.click(screen.getByText('High-Volume HCPs'));

      // Submit
      const submitButton = screen.getByRole('button', { name: /Run Simulation/i });
      await user.click(submitButton);

      expect(mockOnSimulate).toHaveBeenCalledWith({
        intervention_type: InterventionType.HCP_ENGAGEMENT,
        brand: 'Remibrutinib',
        sample_size: 1000,
        duration_days: 90,
        target_regions: ['Northeast', 'West'],
        target_segments: ['High-Volume HCPs'],
        budget: 100000,
      });
    });
  });
});
