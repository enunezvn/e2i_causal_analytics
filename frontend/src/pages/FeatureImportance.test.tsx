/**
 * FeatureImportance Page Tests
 * ============================
 *
 * Tests for the Feature Importance analysis page with SHAP visualizations.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import FeatureImportance from './FeatureImportance';

// Mock URL.createObjectURL and URL.revokeObjectURL for export tests
const mockCreateObjectURL = vi.fn(() => 'blob:mock-url');
const mockRevokeObjectURL = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  global.URL.createObjectURL = mockCreateObjectURL;
  global.URL.revokeObjectURL = mockRevokeObjectURL;
});

describe('FeatureImportance', () => {
  it('renders page header with title and description', () => {
    render(<FeatureImportance />);

    expect(screen.getByText('Feature Importance')).toBeInTheDocument();
    expect(
      screen.getByText(/SHAP values, feature importance bar charts, beeswarm plots/i)
    ).toBeInTheDocument();
  });

  it('displays model selector dropdown', () => {
    render(<FeatureImportance />);

    expect(screen.getByRole('combobox')).toBeInTheDocument();
  });

  it('shows default model info card', () => {
    render(<FeatureImportance />);

    // First model is Patient Churn Predictor (appears in dropdown and info card)
    const modelNames = screen.getAllByText('Patient Churn Predictor');
    expect(modelNames.length).toBeGreaterThanOrEqual(1);
    // Version appears in dropdown and info card
    const versions = screen.getAllByText('v3.2.1');
    expect(versions.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('47 features')).toBeInTheDocument();
  });

  it('displays base value for selected model', () => {
    render(<FeatureImportance />);

    expect(screen.getByText('Base Value')).toBeInTheDocument();
    expect(screen.getByText('0.350')).toBeInTheDocument();
  });

  it('displays top feature label', () => {
    render(<FeatureImportance />);

    expect(screen.getByText('Top Feature')).toBeInTheDocument();
    // First feature is days_since_last_visit, displayed with spaces (may appear multiple times)
    const topFeatures = screen.getAllByText('days since last visit');
    expect(topFeatures.length).toBeGreaterThanOrEqual(1);
  });

  it('displays feature rankings section', () => {
    render(<FeatureImportance />);

    expect(screen.getByText('Feature Rankings')).toBeInTheDocument();
    // Badge showing feature count (10 may appear multiple times)
    const counts = screen.getAllByText('10');
    expect(counts.length).toBeGreaterThanOrEqual(1);
  });

  it('shows feature search input', () => {
    render(<FeatureImportance />);

    expect(screen.getByPlaceholderText('Search features...')).toBeInTheDocument();
  });

  it('displays visualization tabs', () => {
    render(<FeatureImportance />);

    expect(screen.getByRole('tab', { name: /Bar Chart/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Beeswarm/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Waterfall/i })).toBeInTheDocument();
  });

  it('shows Bar Chart tab content by default', () => {
    render(<FeatureImportance />);

    expect(screen.getByText('Global Feature Importance')).toBeInTheDocument();
    expect(
      screen.getByText(/Mean absolute SHAP values showing overall feature importance/i)
    ).toBeInTheDocument();
  });

  it('has clickable Beeswarm tab', () => {
    render(<FeatureImportance />);

    const beeswarmTab = screen.getByRole('tab', { name: /Beeswarm/i });
    expect(beeswarmTab).toBeInTheDocument();
    expect(beeswarmTab).not.toBeDisabled();
  });

  it('has clickable Waterfall tab', () => {
    render(<FeatureImportance />);

    const waterfallTab = screen.getByRole('tab', { name: /Waterfall/i });
    expect(waterfallTab).toBeInTheDocument();
    expect(waterfallTab).not.toBeDisabled();
  });

  it('displays refresh button', () => {
    const { container } = render(<FeatureImportance />);

    // Refresh button has RefreshCw icon
    const refreshButton = container.querySelector('button svg.lucide-refresh-cw');
    expect(refreshButton).toBeInTheDocument();
  });

  it('displays export button', () => {
    render(<FeatureImportance />);

    expect(screen.getByRole('button', { name: /Export/i })).toBeInTheDocument();
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

    render(<FeatureImportance />);

    const exportButton = screen.getByRole('button', { name: /Export/i });
    fireEvent.click(exportButton);

    expect(mockCreateObjectURL).toHaveBeenCalled();
    expect(mockClick).toHaveBeenCalled();

    vi.restoreAllMocks();
  });

  it('renders feature rows with SHAP values', () => {
    render(<FeatureImportance />);

    // Check for SHAP values displayed (days_since_last_visit has +0.3500)
    expect(screen.getByText('+0.3500')).toBeInTheDocument();
    // Negative value (total_prescriptions_ytd has -0.2800)
    expect(screen.getByText('-0.2800')).toBeInTheDocument();
  });

  it('shows feature value in feature rows', () => {
    render(<FeatureImportance />);

    // Feature values are displayed as "Value: X"
    expect(screen.getByText('Value: 45')).toBeInTheDocument();
    expect(screen.getByText('Value: 12')).toBeInTheDocument();
  });

  it('filters features when searching', () => {
    render(<FeatureImportance />);

    // Before filtering, multiple features visible in list
    const beforeFilter = screen.getAllByText(/Value:/);
    expect(beforeFilter.length).toBe(10);

    const searchInput = screen.getByPlaceholderText('Search features...');
    fireEvent.change(searchInput, { target: { value: 'territory' } });

    // After filtering, should show only territory_revenue feature in list
    expect(screen.getByText('territory revenue')).toBeInTheDocument();
    // Only one feature with "Value:" should remain in filtered list
    const afterFilter = screen.getAllByText(/Value:/);
    expect(afterFilter.length).toBe(1);
  });

  it('shows no results message when search has no matches', () => {
    render(<FeatureImportance />);

    const searchInput = screen.getByPlaceholderText('Search features...');
    fireEvent.change(searchInput, { target: { value: 'nonexistent' } });

    expect(screen.getByText('No features match your search')).toBeInTheDocument();
  });
});
