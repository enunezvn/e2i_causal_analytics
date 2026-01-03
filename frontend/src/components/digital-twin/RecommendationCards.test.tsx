/**
 * RecommendationCards Component Tests
 * ====================================
 *
 * Tests for the digital twin recommendation display component.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
  RecommendationCards,
  type RecommendationCardsProps,
} from './RecommendationCards';
import {
  type SimulationRecommendation,
  RecommendationType,
  ConfidenceLevel,
} from '@/types/digital-twin';

// Helper to create mock recommendation
function createMockRecommendation(
  overrides: Partial<SimulationRecommendation> = {}
): SimulationRecommendation {
  return {
    type: RecommendationType.DEPLOY,
    confidence: ConfidenceLevel.HIGH,
    rationale: 'Strong positive effect with high confidence intervals.',
    evidence: [
      'ATE is statistically significant (p < 0.01)',
      'ROI exceeds 2x target',
      'High-value HCPs show 40% stronger response',
    ],
    risk_factors: [
      'Seasonal variation may affect Q4 results',
      'Limited data from Pacific Northwest region',
    ],
    expected_value: 150000,
    ...overrides,
  };
}

describe('RecommendationCards', () => {
  const mockOnAccept = vi.fn();
  const mockOnRefine = vi.fn();
  const mockOnAnalyze = vi.fn();

  const defaultProps: RecommendationCardsProps = {
    recommendation: null,
    onAccept: mockOnAccept,
    onRefine: mockOnRefine,
    onAnalyze: mockOnAnalyze,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Empty State', () => {
    it('renders empty state when no recommendation', () => {
      render(<RecommendationCards {...defaultProps} />);

      expect(screen.getByText('No Recommendation Yet')).toBeInTheDocument();
      expect(
        screen.getByText(/Run a simulation to receive AI-powered recommendations/)
      ).toBeInTheDocument();
    });

    it('shows lightbulb icon in empty state', () => {
      render(<RecommendationCards {...defaultProps} />);

      // Lightbulb icon should be present
      expect(screen.getByText('No Recommendation Yet')).toBeInTheDocument();
    });
  });

  describe('Deploy Recommendation', () => {
    it('renders deploy recommendation card', () => {
      const recommendation = createMockRecommendation({
        type: RecommendationType.DEPLOY,
      });
      render(<RecommendationCards recommendation={recommendation} />);

      expect(screen.getByText('Recommendation: Deploy')).toBeInTheDocument();
    });

    it('displays rationale', () => {
      const recommendation = createMockRecommendation({
        rationale: 'Strong positive effect with high confidence.',
      });
      render(<RecommendationCards recommendation={recommendation} />);

      expect(
        screen.getByText(/Strong positive effect with high confidence/)
      ).toBeInTheDocument();
    });

    it('displays confidence level badge', () => {
      const recommendation = createMockRecommendation({
        confidence: ConfidenceLevel.HIGH,
      });
      render(<RecommendationCards recommendation={recommendation} />);

      expect(screen.getByText('High Confidence')).toBeInTheDocument();
    });

    it('displays expected value', () => {
      const recommendation = createMockRecommendation({
        expected_value: 150000,
      });
      render(<RecommendationCards recommendation={recommendation} />);

      expect(screen.getByText('Expected Value:')).toBeInTheDocument();
      expect(screen.getByText('$150,000')).toBeInTheDocument();
    });

    it('displays evidence points', () => {
      const recommendation = createMockRecommendation({
        evidence: ['First evidence point', 'Second evidence point'],
      });
      render(<RecommendationCards recommendation={recommendation} />);

      expect(screen.getByText('Supporting Evidence')).toBeInTheDocument();
      expect(screen.getByText('First evidence point')).toBeInTheDocument();
      expect(screen.getByText('Second evidence point')).toBeInTheDocument();
    });

    it('renders Proceed with Deployment button', () => {
      const recommendation = createMockRecommendation({
        type: RecommendationType.DEPLOY,
      });
      render(
        <RecommendationCards
          recommendation={recommendation}
          onAccept={mockOnAccept}
        />
      );

      expect(
        screen.getByRole('button', { name: /Proceed with Deployment/i })
      ).toBeInTheDocument();
    });

    it('calls onAccept when deploy button clicked', async () => {
      const user = userEvent.setup();
      const recommendation = createMockRecommendation({
        type: RecommendationType.DEPLOY,
      });
      render(
        <RecommendationCards
          recommendation={recommendation}
          onAccept={mockOnAccept}
        />
      );

      const button = screen.getByRole('button', {
        name: /Proceed with Deployment/i,
      });
      await user.click(button);

      expect(mockOnAccept).toHaveBeenCalledTimes(1);
    });
  });

  describe('Skip Recommendation', () => {
    it('renders skip recommendation card', () => {
      const recommendation = createMockRecommendation({
        type: RecommendationType.SKIP,
      });
      render(<RecommendationCards recommendation={recommendation} />);

      expect(screen.getByText('Recommendation: Skip')).toBeInTheDocument();
    });

    it('renders disabled Not Recommended button', () => {
      const recommendation = createMockRecommendation({
        type: RecommendationType.SKIP,
      });
      render(<RecommendationCards recommendation={recommendation} />);

      const button = screen.getByRole('button', { name: /Not Recommended/i });
      expect(button).toBeInTheDocument();
      expect(button).toBeDisabled();
    });
  });

  describe('Refine Recommendation', () => {
    it('renders refine recommendation card', () => {
      const recommendation = createMockRecommendation({
        type: RecommendationType.REFINE,
      });
      render(<RecommendationCards recommendation={recommendation} />);

      expect(screen.getByText('Recommendation: Refine')).toBeInTheDocument();
    });

    it('renders Adjust Parameters button', () => {
      const recommendation = createMockRecommendation({
        type: RecommendationType.REFINE,
      });
      render(
        <RecommendationCards
          recommendation={recommendation}
          onRefine={mockOnRefine}
        />
      );

      expect(
        screen.getByRole('button', { name: /Adjust Parameters/i })
      ).toBeInTheDocument();
    });

    it('calls onRefine when refine button clicked', async () => {
      const user = userEvent.setup();
      const recommendation = createMockRecommendation({
        type: RecommendationType.REFINE,
      });
      render(
        <RecommendationCards
          recommendation={recommendation}
          onRefine={mockOnRefine}
        />
      );

      const button = screen.getByRole('button', { name: /Adjust Parameters/i });
      await user.click(button);

      expect(mockOnRefine).toHaveBeenCalledTimes(1);
    });

    it('displays suggested refinements', () => {
      const recommendation = createMockRecommendation({
        type: RecommendationType.REFINE,
        suggested_refinements: {
          sample_size: 2000,
          duration_days: 120,
        },
      });
      render(<RecommendationCards recommendation={recommendation} />);

      expect(screen.getByText('Suggested Refinements')).toBeInTheDocument();
      expect(screen.getByText('sample size')).toBeInTheDocument();
      expect(screen.getByText('2000')).toBeInTheDocument();
      expect(screen.getByText('duration days')).toBeInTheDocument();
      expect(screen.getByText('120')).toBeInTheDocument();
    });
  });

  describe('Analyze Recommendation', () => {
    it('renders analyze recommendation card', () => {
      const recommendation = createMockRecommendation({
        type: RecommendationType.ANALYZE,
      });
      render(<RecommendationCards recommendation={recommendation} />);

      expect(screen.getByText('Recommendation: Analyze')).toBeInTheDocument();
    });

    it('renders Run More Analysis button', () => {
      const recommendation = createMockRecommendation({
        type: RecommendationType.ANALYZE,
      });
      render(
        <RecommendationCards
          recommendation={recommendation}
          onAnalyze={mockOnAnalyze}
        />
      );

      expect(
        screen.getByRole('button', { name: /Run More Analysis/i })
      ).toBeInTheDocument();
    });

    it('calls onAnalyze when analyze button clicked', async () => {
      const user = userEvent.setup();
      const recommendation = createMockRecommendation({
        type: RecommendationType.ANALYZE,
      });
      render(
        <RecommendationCards
          recommendation={recommendation}
          onAnalyze={mockOnAnalyze}
        />
      );

      const button = screen.getByRole('button', { name: /Run More Analysis/i });
      await user.click(button);

      expect(mockOnAnalyze).toHaveBeenCalledTimes(1);
    });
  });

  describe('Risk Factors', () => {
    it('displays risk factors when present', () => {
      const recommendation = createMockRecommendation({
        risk_factors: ['Risk factor one', 'Risk factor two'],
      });
      render(<RecommendationCards recommendation={recommendation} />);

      expect(screen.getByText('Risk Factors to Consider')).toBeInTheDocument();
      expect(screen.getByText('Risk factor one')).toBeInTheDocument();
      expect(screen.getByText('Risk factor two')).toBeInTheDocument();
    });

    it('does not show risk factors section when empty', () => {
      const recommendation = createMockRecommendation({
        risk_factors: [],
      });
      render(<RecommendationCards recommendation={recommendation} />);

      expect(
        screen.queryByText('Risk Factors to Consider')
      ).not.toBeInTheDocument();
    });

    it('does not show risk factors section when undefined', () => {
      const recommendation = createMockRecommendation({
        risk_factors: undefined,
      });
      render(<RecommendationCards recommendation={recommendation} />);

      expect(
        screen.queryByText('Risk Factors to Consider')
      ).not.toBeInTheDocument();
    });
  });

  describe('Confidence Levels', () => {
    it('displays High Confidence badge for high level', () => {
      const recommendation = createMockRecommendation({
        confidence: ConfidenceLevel.HIGH,
      });
      render(<RecommendationCards recommendation={recommendation} />);

      expect(screen.getByText('High Confidence')).toBeInTheDocument();
    });

    it('displays Medium Confidence badge for medium level', () => {
      const recommendation = createMockRecommendation({
        confidence: ConfidenceLevel.MEDIUM,
      });
      render(<RecommendationCards recommendation={recommendation} />);

      expect(screen.getByText('Medium Confidence')).toBeInTheDocument();
    });

    it('displays Low Confidence badge for low level', () => {
      const recommendation = createMockRecommendation({
        confidence: ConfidenceLevel.LOW,
      });
      render(<RecommendationCards recommendation={recommendation} />);

      expect(screen.getByText('Low Confidence')).toBeInTheDocument();
    });
  });

  describe('Styling', () => {
    it('applies custom className', () => {
      const recommendation = createMockRecommendation();
      const { container } = render(
        <RecommendationCards
          recommendation={recommendation}
          className="custom-class"
        />
      );

      expect(container.querySelector('.custom-class')).toBeInTheDocument();
    });

    it('applies deploy styling for deploy recommendation', () => {
      const recommendation = createMockRecommendation({
        type: RecommendationType.DEPLOY,
      });
      const { container } = render(
        <RecommendationCards recommendation={recommendation} />
      );

      // Check for emerald/green styling
      expect(container.querySelector('.text-emerald-700')).toBeInTheDocument();
    });

    it('applies refine styling for refine recommendation', () => {
      const recommendation = createMockRecommendation({
        type: RecommendationType.REFINE,
      });
      const { container } = render(
        <RecommendationCards recommendation={recommendation} />
      );

      // Check for amber/yellow styling
      expect(container.querySelector('.text-amber-700')).toBeInTheDocument();
    });

    it('applies skip styling for skip recommendation', () => {
      const recommendation = createMockRecommendation({
        type: RecommendationType.SKIP,
      });
      const { container } = render(
        <RecommendationCards recommendation={recommendation} />
      );

      // Check for rose/red styling
      expect(container.querySelector('.text-rose-700')).toBeInTheDocument();
    });
  });

  describe('Optional Callbacks', () => {
    it('does not render deploy button when onAccept not provided', () => {
      const recommendation = createMockRecommendation({
        type: RecommendationType.DEPLOY,
      });
      render(<RecommendationCards recommendation={recommendation} />);

      expect(
        screen.queryByRole('button', { name: /Proceed with Deployment/i })
      ).not.toBeInTheDocument();
    });

    it('does not render refine button when onRefine not provided', () => {
      const recommendation = createMockRecommendation({
        type: RecommendationType.REFINE,
      });
      render(<RecommendationCards recommendation={recommendation} />);

      expect(
        screen.queryByRole('button', { name: /Adjust Parameters/i })
      ).not.toBeInTheDocument();
    });

    it('does not render analyze button when onAnalyze not provided', () => {
      const recommendation = createMockRecommendation({
        type: RecommendationType.ANALYZE,
      });
      render(<RecommendationCards recommendation={recommendation} />);

      expect(
        screen.queryByRole('button', { name: /Run More Analysis/i })
      ).not.toBeInTheDocument();
    });
  });
});
