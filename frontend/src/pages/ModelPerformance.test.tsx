/**
 * ModelPerformance Page Tests
 * ===========================
 *
 * Tests for the Model Performance analysis page.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ModelPerformance from './ModelPerformance';

// Mock URL.createObjectURL and URL.revokeObjectURL for export tests
const mockCreateObjectURL = vi.fn(() => 'blob:mock-url');
const mockRevokeObjectURL = vi.fn();
global.URL.createObjectURL = mockCreateObjectURL;
global.URL.revokeObjectURL = mockRevokeObjectURL;

describe('ModelPerformance', () => {
  it('renders page header with title', () => {
    render(<ModelPerformance />);

    expect(screen.getByText('Model Performance')).toBeInTheDocument();
    expect(
      screen.getByText(/View model metrics, confusion matrix, ROC curves/i)
    ).toBeInTheDocument();
  });

  it('displays model selector dropdown', () => {
    render(<ModelPerformance />);

    expect(screen.getByRole('combobox')).toBeInTheDocument();
  });

  it('shows default model info card', () => {
    render(<ModelPerformance />);

    // First model is Patient Churn Predictor (appears in dropdown and info card)
    const modelNames = screen.getAllByText('Patient Churn Predictor');
    expect(modelNames.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Production')).toBeInTheDocument();
    // Version appears in dropdown and info card
    const versions = screen.getAllByText('v3.2.1');
    expect(versions.length).toBeGreaterThanOrEqual(1);
  });

  it('displays 5 metric KPI cards', () => {
    render(<ModelPerformance />);

    // These labels appear in both KPI cards and performance trend chart
    expect(screen.getAllByText('Accuracy').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Precision').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Recall').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('F1 Score').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('AUC-ROC').length).toBeGreaterThanOrEqual(1);
  });

  it('renders metric values for first model', () => {
    render(<ModelPerformance />);

    // Churn model has accuracy 0.912 -> displayed as 91.2% (with unit)
    expect(screen.getAllByText(/91\.2/).length).toBeGreaterThanOrEqual(1);
    // AUC 0.945 displayed as 0.945
    expect(screen.getAllByText(/0\.945/).length).toBeGreaterThanOrEqual(1);
  });

  it('displays visualization tabs', () => {
    render(<ModelPerformance />);

    expect(screen.getByRole('tab', { name: /Confusion Matrix/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /ROC Curve/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Performance Trend/i })).toBeInTheDocument();
  });

  it('shows confusion matrix by default', () => {
    render(<ModelPerformance />);

    // Confusion matrix tab should be active by default (text appears in tab and content)
    expect(screen.getAllByText('Confusion Matrix').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/Classification results showing predicted vs actual/)).toBeInTheDocument();
  });

  it('has clickable ROC Curve tab', () => {
    render(<ModelPerformance />);

    const rocTab = screen.getByRole('tab', { name: /ROC Curve/i });
    expect(rocTab).toBeInTheDocument();
    expect(rocTab).not.toBeDisabled();
  });

  it('has clickable Performance Trend tab', () => {
    render(<ModelPerformance />);

    const trendTab = screen.getByRole('tab', { name: /Performance Trend/i });
    expect(trendTab).toBeInTheDocument();
    expect(trendTab).not.toBeDisabled();
  });

  it('displays refresh button', () => {
    const { container } = render(<ModelPerformance />);

    // Refresh button has RefreshCw icon
    const refreshButton = container.querySelector('button svg.lucide-refresh-cw');
    expect(refreshButton).toBeInTheDocument();
  });

  it('displays export button', () => {
    render(<ModelPerformance />);

    expect(screen.getByRole('button', { name: /Export/i })).toBeInTheDocument();
  });

  it('shows samples evaluated count', () => {
    render(<ModelPerformance />);

    expect(screen.getByText('Samples Evaluated')).toBeInTheDocument();
    // Churn model has 15420 samples
    expect(screen.getByText('15,420')).toBeInTheDocument();
  });

  it('displays model configuration section', () => {
    render(<ModelPerformance />);

    expect(screen.getByText('Model Configuration')).toBeInTheDocument();
    expect(screen.getByText('Model Type')).toBeInTheDocument();
    expect(screen.getByText('Algorithm')).toBeInTheDocument();
    expect(screen.getByText('XGBoost Classifier')).toBeInTheDocument();
    expect(screen.getByText('Features Used')).toBeInTheDocument();
    expect(screen.getByText('47')).toBeInTheDocument();
  });

  it('displays threshold settings section', () => {
    render(<ModelPerformance />);

    expect(screen.getByText('Threshold Settings')).toBeInTheDocument();
    expect(screen.getByText('Classification Threshold')).toBeInTheDocument();
    expect(screen.getByText('0.50')).toBeInTheDocument();
    expect(screen.getByText('Accuracy Target')).toBeInTheDocument();
    expect(screen.getByText('90%')).toBeInTheDocument();
  });

  it('shows model training and evaluation dates', () => {
    render(<ModelPerformance />);

    // Dates are inline with labels: "Trained: 2024-03-15" and "Evaluated: 2024-03-20"
    expect(screen.getByText(/Trained:.*2024-03-15/)).toBeInTheDocument();
    expect(screen.getByText(/Evaluated:.*2024-03-20/)).toBeInTheDocument();
  });

  it('handles export button click', () => {
    const mockClick = vi.fn();
    const originalCreateElement = document.createElement.bind(document);
    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      if (tag === 'a') {
        const link = originalCreateElement('a');
        link.click = mockClick;
        return link;
      }
      return originalCreateElement(tag);
    });

    render(<ModelPerformance />);

    const exportButton = screen.getByRole('button', { name: /Export/i });
    fireEvent.click(exportButton);

    expect(mockCreateObjectURL).toHaveBeenCalled();
    expect(mockClick).toHaveBeenCalled();

    vi.restoreAllMocks();
  });

  it('shows metric card values', () => {
    render(<ModelPerformance />);

    // Check that metric values are displayed correctly
    // Accuracy 91.2%, Precision 89.5% (0.895 * 100)
    expect(screen.getAllByText(/91\.2/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/89\.5/).length).toBeGreaterThanOrEqual(1);
  });

  it('renders confusion matrix labels for first model', () => {
    render(<ModelPerformance />);

    // Churn model has Retained/Churned labels (appear in both axes)
    expect(screen.getAllByText('Retained').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Churned').length).toBeGreaterThanOrEqual(1);
  });
});
