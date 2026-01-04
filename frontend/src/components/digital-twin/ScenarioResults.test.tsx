/**
 * ScenarioResults Component Tests
 * ================================
 *
 * Tests for the digital twin simulation results display.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ScenarioResults, type ScenarioResultsProps } from './ScenarioResults';
import {
  type SimulationResponse,
  type SimulationOutcomes,
  type FidelityMetrics,
  type SensitivityResult,
  type ProjectionDataPoint,
  type SimulationRecommendation,
  InterventionType,
  RecommendationType,
  ConfidenceLevel,
} from '@/types/digital-twin';

// Mock Recharts to avoid SVG rendering issues in tests
vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  ComposedChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="composed-chart">{children}</div>
  ),
  LineChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="line-chart">{children}</div>
  ),
  Line: () => <div data-testid="line" />,
  Area: () => <div data-testid="area" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
  ReferenceLine: () => <div data-testid="reference-line" />,
}));

// Helper to create mock simulation results
function createMockSimulationResponse(
  overrides: Partial<SimulationResponse> = {}
): SimulationResponse {
  const outcomes: SimulationOutcomes = {
    ate: { lower: 0.08, estimate: 0.12, upper: 0.16 },
    trx_lift: { lower: 80, estimate: 120, upper: 160 },
    nrx_lift: { lower: 30, estimate: 45, upper: 60 },
    market_share_change: { lower: 0.5, estimate: 1.2, upper: 1.9 },
    roi: { lower: 1.5, estimate: 2.3, upper: 3.1 },
    nnt: 25,
    cate_by_segment: {
      'High-Value': { lower: 0.15, estimate: 0.22, upper: 0.29 },
    },
  };

  const fidelity: FidelityMetrics = {
    overall_score: 0.85,
    data_coverage: 0.92,
    calibration: 0.81,
    temporal_alignment: 0.88,
    feature_completeness: 0.95,
    confidence_level: ConfidenceLevel.HIGH,
    warnings: ['Limited historical data for new HCPs'],
  };

  const sensitivity: SensitivityResult[] = [
    {
      parameter: 'Sample Size',
      base_value: 1000,
      low_value: 500,
      high_value: 2000,
      ate_at_low: 0.10,
      ate_at_high: 0.14,
      sensitivity_score: 0.45,
    },
    {
      parameter: 'Duration',
      base_value: 90,
      low_value: 60,
      high_value: 120,
      ate_at_low: 0.09,
      ate_at_high: 0.15,
      sensitivity_score: 0.60,
    },
  ];

  const recommendation: SimulationRecommendation = {
    type: RecommendationType.DEPLOY,
    confidence: ConfidenceLevel.HIGH,
    rationale: 'Strong positive effect with high confidence.',
    evidence: ['ATE is significant', 'ROI exceeds 2x'],
    risk_factors: ['Seasonal variation possible'],
    expected_value: 150000,
  };

  const projections: ProjectionDataPoint[] = [
    {
      date: '2024-01-01',
      with_intervention: 1000,
      without_intervention: 900,
      lower_bound: 950,
      upper_bound: 1050,
    },
    {
      date: '2024-02-01',
      with_intervention: 1100,
      without_intervention: 920,
      lower_bound: 1020,
      upper_bound: 1180,
    },
  ];

  return {
    simulation_id: 'sim_test_123',
    created_at: '2024-01-01T12:00:00Z',
    request: {
      intervention_type: InterventionType.HCP_ENGAGEMENT,
      brand: 'Remibrutinib',
      sample_size: 1000,
      duration_days: 90,
    },
    outcomes,
    fidelity,
    sensitivity,
    recommendation,
    projections,
    execution_time_ms: 1500,
    ...overrides,
  };
}

describe('ScenarioResults', () => {
  const defaultProps: ScenarioResultsProps = {
    results: null,
    isLoading: false,
  };

  describe('Empty State', () => {
    it('renders empty state when no results', () => {
      render(<ScenarioResults {...defaultProps} />);

      expect(screen.getByText('No Simulation Results')).toBeInTheDocument();
      expect(
        screen.getByText(/Configure and run a simulation/)
      ).toBeInTheDocument();
    });

    it('renders empty state icon', () => {
      render(<ScenarioResults {...defaultProps} />);

      // Activity icon should be present
      expect(screen.getByText('No Simulation Results')).toBeInTheDocument();
    });
  });

  describe('Loading State', () => {
    it('renders loading state when isLoading is true', () => {
      render(<ScenarioResults {...defaultProps} isLoading={true} />);

      expect(screen.getByText('Running simulation...')).toBeInTheDocument();
      expect(
        screen.getByText('This may take a few moments')
      ).toBeInTheDocument();
    });

    it('shows spinner during loading', () => {
      const { container } = render(
        <ScenarioResults {...defaultProps} isLoading={true} />
      );

      expect(container.querySelector('.animate-spin')).toBeInTheDocument();
    });
  });

  describe('Results Display', () => {
    it('renders outcome cards with results', () => {
      const results = createMockSimulationResponse();
      render(<ScenarioResults results={results} />);

      expect(screen.getByText('Average Treatment Effect')).toBeInTheDocument();
      expect(screen.getByText('TRx Lift')).toBeInTheDocument();
      expect(screen.getByText('Market Share Change')).toBeInTheDocument();
      expect(screen.getByText('ROI Projection')).toBeInTheDocument();
    });

    it('displays ATE value correctly', () => {
      const results = createMockSimulationResponse();
      render(<ScenarioResults results={results} />);

      // ATE estimate of 0.12 should be displayed
      expect(screen.getByText('+0.1')).toBeInTheDocument();
    });

    it('displays confidence intervals', () => {
      const results = createMockSimulationResponse();
      render(<ScenarioResults results={results} />);

      // Check for 95% CI labels
      const ciLabels = screen.getAllByText(/95% CI:/);
      expect(ciLabels.length).toBeGreaterThan(0);
    });

    it('renders projections chart section', () => {
      const results = createMockSimulationResponse();
      render(<ScenarioResults results={results} />);

      expect(screen.getByText('Projected Outcomes')).toBeInTheDocument();
      expect(
        screen.getByText(/Time series projection/)
      ).toBeInTheDocument();
    });

    it('displays execution time', () => {
      const results = createMockSimulationResponse({ execution_time_ms: 2500 });
      render(<ScenarioResults results={results} />);

      expect(screen.getByText(/Computed in 2,500ms/)).toBeInTheDocument();
    });
  });

  describe('Fidelity Metrics', () => {
    it('renders fidelity metrics section', () => {
      const results = createMockSimulationResponse();
      render(<ScenarioResults results={results} />);

      expect(screen.getByText('Model Fidelity')).toBeInTheDocument();
      expect(
        screen.getByText('Confidence in simulation accuracy')
      ).toBeInTheDocument();
    });

    it('displays overall fidelity score', () => {
      const results = createMockSimulationResponse({
        fidelity: {
          ...createMockSimulationResponse().fidelity,
          overall_score: 0.85,
        },
      });
      render(<ScenarioResults results={results} />);

      expect(screen.getByText('85%')).toBeInTheDocument();
      expect(screen.getByText('Overall Fidelity Score')).toBeInTheDocument();
    });

    it('displays individual fidelity metrics', () => {
      const results = createMockSimulationResponse();
      render(<ScenarioResults results={results} />);

      expect(screen.getByText('Data Coverage')).toBeInTheDocument();
      expect(screen.getByText('Model Calibration')).toBeInTheDocument();
      expect(screen.getByText('Temporal Alignment')).toBeInTheDocument();
      expect(screen.getByText('Feature Completeness')).toBeInTheDocument();
    });

    it('displays confidence level badge', () => {
      const results = createMockSimulationResponse();
      render(<ScenarioResults results={results} />);

      expect(screen.getByText('high confidence')).toBeInTheDocument();
    });

    it('displays fidelity warnings when present', () => {
      const results = createMockSimulationResponse();
      render(<ScenarioResults results={results} />);

      expect(screen.getByText('Limitations')).toBeInTheDocument();
      expect(
        screen.getByText(/Limited historical data for new HCPs/)
      ).toBeInTheDocument();
    });
  });

  describe('Sensitivity Analysis', () => {
    it('renders sensitivity analysis section', () => {
      const results = createMockSimulationResponse();
      render(<ScenarioResults results={results} />);

      expect(screen.getByText('Sensitivity Analysis')).toBeInTheDocument();
      expect(
        screen.getByText('How parameter changes affect outcomes')
      ).toBeInTheDocument();
    });

    it('displays sensitivity parameters', () => {
      const results = createMockSimulationResponse();
      render(<ScenarioResults results={results} />);

      expect(screen.getByText('Sample Size')).toBeInTheDocument();
      expect(screen.getByText('Duration')).toBeInTheDocument();
    });

    it('displays sensitivity scores', () => {
      const results = createMockSimulationResponse();
      render(<ScenarioResults results={results} />);

      expect(screen.getByText('Sensitivity: 45%')).toBeInTheDocument();
      expect(screen.getByText('Sensitivity: 60%')).toBeInTheDocument();
    });
  });

  describe('Styling', () => {
    it('applies custom className', () => {
      const { container } = render(
        <ScenarioResults
          results={createMockSimulationResponse()}
          className="custom-class"
        />
      );

      expect(container.querySelector('.custom-class')).toBeInTheDocument();
    });

    it('applies positive color for positive ATE', () => {
      const results = createMockSimulationResponse();
      const { container } = render(<ScenarioResults results={results} />);

      // Check for text-emerald-600 class on positive values
      expect(container.querySelector('.text-emerald-600')).toBeInTheDocument();
    });
  });

  // ===========================================================================
  // CONFIDENCE LEVEL BADGE COLORS
  // ===========================================================================

  describe('Confidence Level Badges', () => {
    it('displays high confidence badge with emerald color', () => {
      const results = createMockSimulationResponse({
        fidelity: {
          ...createMockSimulationResponse().fidelity,
          confidence_level: ConfidenceLevel.HIGH,
        },
      });
      const { container } = render(<ScenarioResults results={results} />);

      // High confidence should use emerald colors
      expect(screen.getByText('high confidence')).toBeInTheDocument();
      expect(container.querySelector('.bg-emerald-100')).toBeInTheDocument();
    });

    it('displays medium confidence badge with amber color', () => {
      const results = createMockSimulationResponse({
        fidelity: {
          ...createMockSimulationResponse().fidelity,
          confidence_level: ConfidenceLevel.MEDIUM,
        },
      });
      const { container } = render(<ScenarioResults results={results} />);

      // Medium confidence should use amber colors
      expect(screen.getByText('medium confidence')).toBeInTheDocument();
      expect(container.querySelector('.bg-amber-100')).toBeInTheDocument();
    });

    it('displays low confidence badge with rose color', () => {
      const results = createMockSimulationResponse({
        fidelity: {
          ...createMockSimulationResponse().fidelity,
          confidence_level: ConfidenceLevel.LOW,
        },
      });
      const { container } = render(<ScenarioResults results={results} />);

      // Low confidence should use rose colors
      expect(screen.getByText('low confidence')).toBeInTheDocument();
      expect(container.querySelector('.bg-rose-100')).toBeInTheDocument();
    });
  });

  // ===========================================================================
  // FIDELITY SCORE COLORS
  // ===========================================================================

  describe('Fidelity Score Colors', () => {
    it('displays high fidelity score (>=0.8) with emerald color', () => {
      const results = createMockSimulationResponse({
        fidelity: {
          ...createMockSimulationResponse().fidelity,
          overall_score: 0.85,
        },
      });
      const { container } = render(<ScenarioResults results={results} />);

      // High score should use emerald color
      expect(screen.getByText('85%')).toBeInTheDocument();
      expect(container.querySelector('.text-emerald-600')).toBeInTheDocument();
    });

    it('displays medium fidelity score (0.6-0.8) with amber color', () => {
      const results = createMockSimulationResponse({
        fidelity: {
          ...createMockSimulationResponse().fidelity,
          overall_score: 0.7,
        },
      });
      const { container } = render(<ScenarioResults results={results} />);

      // Medium score should use amber color
      expect(screen.getByText('70%')).toBeInTheDocument();
      expect(container.querySelector('.text-amber-600')).toBeInTheDocument();
    });

    it('displays low fidelity score (<0.6) with rose color', () => {
      const results = createMockSimulationResponse({
        fidelity: {
          ...createMockSimulationResponse().fidelity,
          overall_score: 0.5,
        },
      });
      const { container } = render(<ScenarioResults results={results} />);

      // Low score should use rose color
      expect(screen.getByText('50%')).toBeInTheDocument();
      expect(container.querySelector('.text-rose-600')).toBeInTheDocument();
    });
  });

  // ===========================================================================
  // CHART RENDERING
  // ===========================================================================

  describe('Chart Rendering', () => {
    it('renders chart elements with projections data', () => {
      const results = createMockSimulationResponse();
      render(<ScenarioResults results={results} />);

      // Verify chart container is rendered
      expect(screen.getByTestId('responsive-container')).toBeInTheDocument();
      expect(screen.getByTestId('composed-chart')).toBeInTheDocument();
    });

    it('renders chart legend', () => {
      const results = createMockSimulationResponse();
      render(<ScenarioResults results={results} />);

      expect(screen.getByTestId('legend')).toBeInTheDocument();
    });

    it('renders chart axes', () => {
      const results = createMockSimulationResponse();
      render(<ScenarioResults results={results} />);

      expect(screen.getByTestId('x-axis')).toBeInTheDocument();
      expect(screen.getByTestId('y-axis')).toBeInTheDocument();
    });

    it('renders area and lines in chart', () => {
      const results = createMockSimulationResponse();
      render(<ScenarioResults results={results} />);

      // Chart may have multiple area/line elements
      const areas = screen.getAllByTestId('area');
      expect(areas.length).toBeGreaterThanOrEqual(1);
      // Lines for with/without intervention
      const lines = screen.getAllByTestId('line');
      expect(lines.length).toBeGreaterThanOrEqual(1);
    });
  });
});
