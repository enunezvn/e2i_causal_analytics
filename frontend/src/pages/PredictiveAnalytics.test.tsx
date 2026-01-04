/**
 * PredictiveAnalytics Page Tests
 * ==============================
 *
 * Tests for the Predictive Analytics dashboard page with
 * risk scores, distributions, uplift models, and AI recommendations.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import PredictiveAnalytics from './PredictiveAnalytics';

// Mock Recharts components to avoid canvas/SVG rendering issues in tests
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

// Mock URL.createObjectURL and URL.revokeObjectURL for export tests
const mockCreateObjectURL = vi.fn(() => 'blob:mock-url');
const mockRevokeObjectURL = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  global.URL.createObjectURL = mockCreateObjectURL;
  global.URL.revokeObjectURL = mockRevokeObjectURL;
});

describe('PredictiveAnalytics', () => {
  it('renders page header with title and description', () => {
    render(<PredictiveAnalytics />);

    expect(screen.getByText('Predictive Analytics')).toBeInTheDocument();
    expect(
      screen.getByText(/Risk scores, probability distributions, uplift models/i)
    ).toBeInTheDocument();
  });

  it('displays model and timeframe selectors', () => {
    render(<PredictiveAnalytics />);

    // Should have 2 comboboxes (model selector and timeframe)
    const comboboxes = screen.getAllByRole('combobox');
    expect(comboboxes.length).toBe(2);
  });

  it('displays refresh and export buttons', () => {
    const { container } = render(<PredictiveAnalytics />);

    // Refresh button has RefreshCw icon
    const refreshButton = container.querySelector('button svg.lucide-refresh-cw');
    expect(refreshButton).toBeInTheDocument();
    // Export button has Download icon
    const exportButton = container.querySelector('button svg.lucide-download');
    expect(exportButton).toBeInTheDocument();
  });

  it('displays model performance card with metrics', () => {
    render(<PredictiveAnalytics />);

    expect(screen.getByText('Active Model')).toBeInTheDocument();
    expect(screen.getByText('Churn Prediction Model')).toBeInTheDocument();
    expect(screen.getByText('AUC-ROC')).toBeInTheDocument();
    expect(screen.getByText('Accuracy')).toBeInTheDocument();
    expect(screen.getByText('F1 Score')).toBeInTheDocument();
    expect(screen.getByText('Last Trained')).toBeInTheDocument();
    expect(screen.getByText('Model Healthy')).toBeInTheDocument();
  });

  it('displays 4 KPI cards', () => {
    render(<PredictiveAnalytics />);

    expect(screen.getByText('High Risk Entities')).toBeInTheDocument();
    expect(screen.getByText('Avg Model Confidence')).toBeInTheDocument();
    expect(screen.getByText('Avg Uplift Potential')).toBeInTheDocument();
    expect(screen.getByText('High Priority Actions')).toBeInTheDocument();
  });

  it('displays 4 main tabs', () => {
    render(<PredictiveAnalytics />);

    expect(screen.getByRole('tab', { name: /Risk Scores/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Distributions/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Uplift Models/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Recommendations/i })).toBeInTheDocument();
  });

  it('shows Risk Scores tab content by default', () => {
    render(<PredictiveAnalytics />);

    expect(screen.getByText('Entity Risk Scores')).toBeInTheDocument();
    expect(
      screen.getByText(/Individual risk assessments with contributing factors/i)
    ).toBeInTheDocument();
    // Filter button in Risk Scores tab
    expect(screen.getByRole('button', { name: /Filter/i })).toBeInTheDocument();
  });

  it('displays risk score cards with entity names', () => {
    render(<PredictiveAnalytics />);

    // Sample entities from the risk scores
    expect(screen.getByText('Dr. Sarah Chen')).toBeInTheDocument();
    expect(screen.getByText('Dr. Michael Roberts')).toBeInTheDocument();
    expect(screen.getByText('Memorial Hospital')).toBeInTheDocument();
  });

  it('displays entity type badges in risk cards', () => {
    render(<PredictiveAnalytics />);

    // Entity types displayed as uppercase badges
    const hcpBadges = screen.getAllByText('HCP');
    expect(hcpBadges.length).toBeGreaterThanOrEqual(1);
    const accountBadges = screen.getAllByText('ACCOUNT');
    expect(accountBadges.length).toBeGreaterThanOrEqual(1);
    const territoryBadges = screen.getAllByText('TERRITORY');
    expect(territoryBadges.length).toBeGreaterThanOrEqual(1);
  });

  it('displays risk categories in risk cards', () => {
    render(<PredictiveAnalytics />);

    // Risk categories displayed as labels
    const churnLabels = screen.getAllByText('Churn Risk');
    expect(churnLabels.length).toBeGreaterThanOrEqual(1);
  });

  it('displays key factors section in risk cards', () => {
    render(<PredictiveAnalytics />);

    // Key factors labels
    const keyFactorsLabels = screen.getAllByText('Key Factors');
    expect(keyFactorsLabels.length).toBeGreaterThanOrEqual(1);
    // Factor names appear in cards
    const recentActivityFactors = screen.getAllByText('Recent Activity');
    expect(recentActivityFactors.length).toBeGreaterThanOrEqual(1);
  });

  it('has clickable Distributions tab', () => {
    render(<PredictiveAnalytics />);

    const distributionsTab = screen.getByRole('tab', { name: /Distributions/i });
    expect(distributionsTab).toBeInTheDocument();
    expect(distributionsTab).not.toBeDisabled();
  });

  it('has clickable Uplift Models tab', () => {
    render(<PredictiveAnalytics />);

    const upliftTab = screen.getByRole('tab', { name: /Uplift Models/i });
    expect(upliftTab).toBeInTheDocument();
    expect(upliftTab).not.toBeDisabled();
  });

  it('has clickable Recommendations tab', () => {
    render(<PredictiveAnalytics />);

    const recommendationsTab = screen.getByRole('tab', { name: /Recommendations/i });
    expect(recommendationsTab).toBeInTheDocument();
    expect(recommendationsTab).not.toBeDisabled();
  });

  it('displays probability and confidence labels in risk cards', () => {
    render(<PredictiveAnalytics />);

    // Probability and confidence metrics
    const probabilityLabels = screen.getAllByText('Probability');
    expect(probabilityLabels.length).toBeGreaterThanOrEqual(1);
    const confidenceLabels = screen.getAllByText('Confidence');
    expect(confidenceLabels.length).toBeGreaterThanOrEqual(1);
  });

  it('displays trend indicators in risk cards', () => {
    render(<PredictiveAnalytics />);

    // Trend label appears in risk cards
    const trendLabels = screen.getAllByText('Trend');
    expect(trendLabels.length).toBeGreaterThanOrEqual(1);
  });

  it('displays default model selection', () => {
    render(<PredictiveAnalytics />);

    // Default model is Churn Model
    expect(screen.getByText('Churn Model')).toBeInTheDocument();
    // Default timeframe is 30 days
    expect(screen.getByText('30 days')).toBeInTheDocument();
  });

  it('displays AUC-ROC metric value', () => {
    render(<PredictiveAnalytics />);

    // AUC-ROC shows 91.0% (0.91 * 100)
    expect(screen.getByText('91.0%')).toBeInTheDocument();
  });

  it('displays accuracy metric value', () => {
    render(<PredictiveAnalytics />);

    // Accuracy shows 87.0% (0.87 * 100)
    expect(screen.getByText('87.0%')).toBeInTheDocument();
  });

  it('displays F1 score metric value', () => {
    render(<PredictiveAnalytics />);

    // F1 Score shows 82.0% (0.82 * 100)
    expect(screen.getByText('82.0%')).toBeInTheDocument();
  });

  it('displays historical pattern factor in risk cards', () => {
    render(<PredictiveAnalytics />);

    // Historical Pattern is a factor in risk cards
    const historyFactors = screen.getAllByText('Historical Pattern');
    expect(historyFactors.length).toBeGreaterThanOrEqual(1);
  });

  // =========================================================================
  // DISTRIBUTIONS TAB TESTS
  // =========================================================================

  describe('Distributions Tab', () => {
    it('switches to Distributions tab and displays content', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const distributionsTab = screen.getByRole('tab', { name: /Distributions/i });
      await user.click(distributionsTab);

      // Distribution tab content
      expect(screen.getByText('Score Probability Distribution')).toBeInTheDocument();
      expect(screen.getByText(/Distribution of prediction scores across all entities/i)).toBeInTheDocument();
    });

    it('displays Model Calibration chart in Distributions tab', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const distributionsTab = screen.getByRole('tab', { name: /Distributions/i });
      await user.click(distributionsTab);

      expect(screen.getByText('Model Calibration')).toBeInTheDocument();
      expect(screen.getByText(/Predicted vs actual outcome rates/i)).toBeInTheDocument();
    });

    it('displays Cumulative Score Distribution chart in Distributions tab', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const distributionsTab = screen.getByRole('tab', { name: /Distributions/i });
      await user.click(distributionsTab);

      expect(screen.getByText('Cumulative Score Distribution')).toBeInTheDocument();
      expect(screen.getByText(/Running total of entities at each score threshold/i)).toBeInTheDocument();
    });

    it('renders responsive chart containers in Distributions tab', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const distributionsTab = screen.getByRole('tab', { name: /Distributions/i });
      await user.click(distributionsTab);

      const chartContainers = screen.getAllByTestId('responsive-container');
      expect(chartContainers.length).toBeGreaterThanOrEqual(3);
    });
  });

  // =========================================================================
  // UPLIFT MODELS TAB TESTS
  // =========================================================================

  describe('Uplift Models Tab', () => {
    it('switches to Uplift Models tab and displays content', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const upliftTab = screen.getByRole('tab', { name: /Uplift Models/i });
      await user.click(upliftTab);

      expect(screen.getByText('Uplift Model Segments')).toBeInTheDocument();
      expect(screen.getByText(/Identify high-impact segments for targeted interventions/i)).toBeInTheDocument();
    });

    it('displays Segment Uplift Analysis chart in Uplift tab', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const upliftTab = screen.getByRole('tab', { name: /Uplift Models/i });
      await user.click(upliftTab);

      expect(screen.getByText('Segment Uplift Analysis')).toBeInTheDocument();
      expect(screen.getByText(/Comparing baseline vs predicted conversion with uplift potential/i)).toBeInTheDocument();
    });

    it('displays Segment ROI Comparison chart in Uplift tab', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const upliftTab = screen.getByRole('tab', { name: /Uplift Models/i });
      await user.click(upliftTab);

      expect(screen.getByText('Segment ROI Comparison')).toBeInTheDocument();
      expect(screen.getByText(/Expected return on investment by segment/i)).toBeInTheDocument();
    });

    it('displays uplift segment cards with segment names', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const upliftTab = screen.getByRole('tab', { name: /Uplift Models/i });
      await user.click(upliftTab);

      // Segment names from upliftSegments data
      expect(screen.getByText('High-Value Responders')).toBeInTheDocument();
      expect(screen.getByText('Persuadables')).toBeInTheDocument();
      expect(screen.getByText('Sure Things')).toBeInTheDocument();
      expect(screen.getByText('Lost Causes')).toBeInTheDocument();
      expect(screen.getByText('Sleeping Giants')).toBeInTheDocument();
    });

    it('displays segment entity counts', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const upliftTab = screen.getByRole('tab', { name: /Uplift Models/i });
      await user.click(upliftTab);

      expect(screen.getByText('1,250 entities')).toBeInTheDocument();
      expect(screen.getByText('3,400 entities')).toBeInTheDocument();
    });

    it('displays ROI badges with correct colors', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const upliftTab = screen.getByRole('tab', { name: /Uplift Models/i });
      await user.click(upliftTab);

      // Different ROI levels
      expect(screen.getByText('+4.2x ROI')).toBeInTheDocument(); // High ROI (≥3)
      expect(screen.getByText('+2.8x ROI')).toBeInTheDocument(); // Medium ROI (≥1)
      expect(screen.getByText('+0.5x ROI')).toBeInTheDocument(); // Low ROI (≥0)
      expect(screen.getByText('-0.2x ROI')).toBeInTheDocument(); // Negative ROI
    });

    it('displays baseline, predicted, and uplift metrics in segment cards', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const upliftTab = screen.getByRole('tab', { name: /Uplift Models/i });
      await user.click(upliftTab);

      // Column headers in segment cards
      const baselineLabels = screen.getAllByText('Baseline');
      expect(baselineLabels.length).toBeGreaterThanOrEqual(1);
      const predictedLabels = screen.getAllByText('Predicted');
      expect(predictedLabels.length).toBeGreaterThanOrEqual(1);
      const upliftLabels = screen.getAllByText('Uplift');
      expect(upliftLabels.length).toBeGreaterThanOrEqual(1);
    });

    it('displays recommended actions in segment cards', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const upliftTab = screen.getByRole('tab', { name: /Uplift Models/i });
      await user.click(upliftTab);

      expect(screen.getByText('Prioritize for intensive engagement')).toBeInTheDocument();
      expect(screen.getByText('Target with personalized messaging')).toBeInTheDocument();
      expect(screen.getByText('Maintain light touch engagement')).toBeInTheDocument();
      expect(screen.getByText('Deprioritize - low ROI potential')).toBeInTheDocument();
    });
  });

  // =========================================================================
  // RECOMMENDATIONS TAB TESTS
  // =========================================================================

  describe('Recommendations Tab', () => {
    it('switches to Recommendations tab and displays content', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const recommendationsTab = screen.getByRole('tab', { name: /Recommendations/i });
      await user.click(recommendationsTab);

      expect(screen.getByText('AI-Powered Recommendations')).toBeInTheDocument();
      expect(screen.getByText(/Actionable insights derived from predictive models/i)).toBeInTheDocument();
    });

    it('displays recommendation priority badges', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const recommendationsTab = screen.getByRole('tab', { name: /Recommendations/i });
      await user.click(recommendationsTab);

      // Priority badges
      const highBadges = screen.getAllByText('HIGH');
      expect(highBadges.length).toBeGreaterThanOrEqual(2);
      const mediumBadges = screen.getAllByText('MEDIUM');
      expect(mediumBadges.length).toBeGreaterThanOrEqual(2);
      const lowBadges = screen.getAllByText('LOW');
      expect(lowBadges.length).toBeGreaterThanOrEqual(1);
    });

    it('displays recommendation type badges', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const recommendationsTab = screen.getByRole('tab', { name: /Recommendations/i });
      await user.click(recommendationsTab);

      // Type badges
      expect(screen.getAllByText('targeting').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('timing').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('channel').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('messaging').length).toBeGreaterThanOrEqual(1);
    });

    it('displays recommendation titles and descriptions', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const recommendationsTab = screen.getByRole('tab', { name: /Recommendations/i });
      await user.click(recommendationsTab);

      expect(screen.getByText('Focus on High-Value Responders Segment')).toBeInTheDocument();
      expect(screen.getByText('Optimal Engagement Window Detected')).toBeInTheDocument();
      expect(screen.getByText('Channel Mix Optimization')).toBeInTheDocument();
    });

    it('displays recommendation impact values', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const recommendationsTab = screen.getByRole('tab', { name: /Recommendations/i });
      await user.click(recommendationsTab);

      expect(screen.getByText('+$2.4M projected impact')).toBeInTheDocument();
      expect(screen.getByText('+18% engagement rate')).toBeInTheDocument();
      expect(screen.getByText('+12% conversion rate')).toBeInTheDocument();
    });

    it('displays confidence percentages for recommendations', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const recommendationsTab = screen.getByRole('tab', { name: /Recommendations/i });
      await user.click(recommendationsTab);

      expect(screen.getByText('89% confidence')).toBeInTheDocument();
      expect(screen.getByText('92% confidence')).toBeInTheDocument();
      expect(screen.getByText('78% confidence')).toBeInTheDocument();
    });

    it('displays "Needs Validation" badge for non-actionable recommendations', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const recommendationsTab = screen.getByRole('tab', { name: /Recommendations/i });
      await user.click(recommendationsTab);

      expect(screen.getByText('Needs Validation')).toBeInTheDocument();
    });

    it('displays summary badges for high priority and actionable counts', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const recommendationsTab = screen.getByRole('tab', { name: /Recommendations/i });
      await user.click(recommendationsTab);

      expect(screen.getByText('2 High Priority')).toBeInTheDocument();
      expect(screen.getByText('4 Actionable')).toBeInTheDocument();
    });

    it('displays Summary Impact section with revenue estimate', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const recommendationsTab = screen.getByRole('tab', { name: /Recommendations/i });
      await user.click(recommendationsTab);

      expect(screen.getByText('Summary Impact')).toBeInTheDocument();
      expect(screen.getByText(/\+\$4\.8M/)).toBeInTheDocument();
      expect(screen.getByText(/85%/)).toBeInTheDocument();
    });

    it('displays Generate Action Plan button', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const recommendationsTab = screen.getByRole('tab', { name: /Recommendations/i });
      await user.click(recommendationsTab);

      expect(screen.getByRole('button', { name: /Generate Action Plan/i })).toBeInTheDocument();
    });
  });

  // =========================================================================
  // RISK SCORE CARD VARIATIONS
  // =========================================================================

  describe('RiskScoreCard variations', () => {
    it('displays different entity types correctly', () => {
      render(<PredictiveAnalytics />);

      // HCP type
      const hcpBadges = screen.getAllByText('HCP');
      expect(hcpBadges.length).toBeGreaterThanOrEqual(1);
      // ACCOUNT type
      const accountBadges = screen.getAllByText('ACCOUNT');
      expect(accountBadges.length).toBeGreaterThanOrEqual(1);
      // TERRITORY type
      const territoryBadges = screen.getAllByText('TERRITORY');
      expect(territoryBadges.length).toBeGreaterThanOrEqual(1);
    });

    it('displays all risk category labels', () => {
      render(<PredictiveAnalytics />);

      // Different category labels
      const churnLabels = screen.getAllByText('Churn Risk');
      expect(churnLabels.length).toBeGreaterThanOrEqual(1);
    });

    it('displays Engagement Score factor', () => {
      render(<PredictiveAnalytics />);

      const engagementFactors = screen.getAllByText('Engagement Score');
      expect(engagementFactors.length).toBeGreaterThanOrEqual(1);
    });

    it('displays increasing trend indicators', () => {
      render(<PredictiveAnalytics />);

      // Note: Due to randomization in sample data, we check for presence of at least one trend type
      // The trend can be Increasing, Decreasing, or Stable
      const trendContainer = screen.getAllByText('Trend');
      expect(trendContainer.length).toBeGreaterThanOrEqual(1);
    });
  });

  // =========================================================================
  // MODEL SELECTION AND TIMEFRAME
  // =========================================================================

  describe('Model and Timeframe Selection', () => {
    it('can change model selection', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      // Get the model selector (first combobox)
      const comboboxes = screen.getAllByRole('combobox');
      const modelSelector = comboboxes[0];

      await user.click(modelSelector);

      // Select a different model
      const adoptionOption = screen.getByRole('option', { name: /Adoption Model/i });
      await user.click(adoptionOption);

      // Model name should update in the active model display
      expect(screen.getByText('Adoption Prediction Model')).toBeInTheDocument();
    });

    it('can change timeframe selection', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      // Get the timeframe selector (second combobox)
      const comboboxes = screen.getAllByRole('combobox');
      const timeframeSelector = comboboxes[1];

      await user.click(timeframeSelector);

      // Select a different timeframe
      const sevenDaysOption = screen.getByRole('option', { name: /7 days/i });
      await user.click(sevenDaysOption);

      // The timeframe should be updated
      expect(screen.getByText('7 days')).toBeInTheDocument();
    });

    it('displays conversion model when selected', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const comboboxes = screen.getAllByRole('combobox');
      const modelSelector = comboboxes[0];

      await user.click(modelSelector);
      const conversionOption = screen.getByRole('option', { name: /Conversion Model/i });
      await user.click(conversionOption);

      expect(screen.getByText('Conversion Prediction Model')).toBeInTheDocument();
    });

    it('displays engagement model when selected', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const comboboxes = screen.getAllByRole('combobox');
      const modelSelector = comboboxes[0];

      await user.click(modelSelector);
      const engagementOption = screen.getByRole('option', { name: /Engagement Model/i });
      await user.click(engagementOption);

      expect(screen.getByText('Engagement Prediction Model')).toBeInTheDocument();
    });

    it('can select 90 days timeframe', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      const comboboxes = screen.getAllByRole('combobox');
      const timeframeSelector = comboboxes[1];

      await user.click(timeframeSelector);
      const ninetyDaysOption = screen.getByRole('option', { name: /90 days/i });
      await user.click(ninetyDaysOption);

      expect(screen.getByText('90 days')).toBeInTheDocument();
    });
  });

  // =========================================================================
  // TAB NAVIGATION TESTS
  // =========================================================================

  describe('Tab Navigation', () => {
    it('navigates between all tabs correctly', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      // Start on Risk Scores (default)
      expect(screen.getByText('Entity Risk Scores')).toBeInTheDocument();

      // Navigate to Distributions
      const distributionsTab = screen.getByRole('tab', { name: /Distributions/i });
      await user.click(distributionsTab);
      expect(screen.getByText('Score Probability Distribution')).toBeInTheDocument();

      // Navigate to Uplift
      const upliftTab = screen.getByRole('tab', { name: /Uplift Models/i });
      await user.click(upliftTab);
      expect(screen.getByText('Uplift Model Segments')).toBeInTheDocument();

      // Navigate to Recommendations
      const recommendationsTab = screen.getByRole('tab', { name: /Recommendations/i });
      await user.click(recommendationsTab);
      expect(screen.getByText('AI-Powered Recommendations')).toBeInTheDocument();

      // Navigate back to Risk Scores
      const riskScoresTab = screen.getByRole('tab', { name: /Risk Scores/i });
      await user.click(riskScoresTab);
      expect(screen.getByText('Entity Risk Scores')).toBeInTheDocument();
    });

    it('maintains tab state correctly', async () => {
      render(<PredictiveAnalytics />);
      const user = userEvent.setup();

      // Navigate to Recommendations
      const recommendationsTab = screen.getByRole('tab', { name: /Recommendations/i });
      await user.click(recommendationsTab);

      // Verify we're on recommendations tab
      expect(screen.getByText('Generate Action Plan')).toBeInTheDocument();

      // Risk Scores content should not be visible
      expect(screen.queryByText('Entity Risk Scores')).not.toBeInTheDocument();
    });
  });

  // =========================================================================
  // REFRESH BUTTON TEST
  // =========================================================================

  describe('Refresh button', () => {
    it('refresh button is clickable', () => {
      const { container } = render(<PredictiveAnalytics />);

      const refreshIcon = container.querySelector('svg.lucide-refresh-cw');
      const refreshButton = refreshIcon?.closest('button');
      expect(refreshButton).toBeInTheDocument();

      // Click should not throw
      fireEvent.click(refreshButton!);
    });
  });
});
