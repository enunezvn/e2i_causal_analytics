/**
 * PredictiveAnalytics Page Tests
 * ==============================
 *
 * Tests for the Predictive Analytics dashboard page with
 * risk scores, distributions, uplift models, and AI recommendations.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import PredictiveAnalytics from './PredictiveAnalytics';

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
});
