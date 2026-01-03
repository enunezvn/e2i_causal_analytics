/**
 * TimeSeries Page Tests
 * =====================
 *
 * Tests for the Time Series Analysis page with forecasting,
 * seasonality decomposition, and anomaly detection.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import TimeSeries from './TimeSeries';

// Mock URL.createObjectURL and URL.revokeObjectURL for export tests
const mockCreateObjectURL = vi.fn(() => 'blob:mock-url');
const mockRevokeObjectURL = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  global.URL.createObjectURL = mockCreateObjectURL;
  global.URL.revokeObjectURL = mockRevokeObjectURL;
});

describe('TimeSeries', () => {
  it('renders page header with title and description', () => {
    render(<TimeSeries />);

    expect(screen.getByText('Time Series Analysis')).toBeInTheDocument();
    expect(
      screen.getByText(/Time series trends, forecasting, seasonality decomposition/i)
    ).toBeInTheDocument();
  });

  it('displays KPI summary cards', () => {
    render(<TimeSeries />);

    expect(screen.getByText('Current Value')).toBeInTheDocument();
    expect(screen.getByText('Average')).toBeInTheDocument();
    expect(screen.getByText('Maximum')).toBeInTheDocument();
    expect(screen.getByText('Minimum')).toBeInTheDocument();
    // "Anomalies" appears in both KPI card and tab
    expect(screen.getAllByText('Anomalies').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Forecast MAPE')).toBeInTheDocument();
    expect(screen.getByText('Forecast R²')).toBeInTheDocument();
  });

  it('displays metric selector dropdown', () => {
    render(<TimeSeries />);

    // Should have at least 2 comboboxes (metric selector and time range)
    const comboboxes = screen.getAllByRole('combobox');
    expect(comboboxes.length).toBeGreaterThanOrEqual(2);
  });

  it('displays refresh and export buttons', () => {
    const { container } = render(<TimeSeries />);

    // Refresh and export are icon buttons
    const refreshButton = container.querySelector('button svg.lucide-refresh-cw');
    expect(refreshButton).toBeInTheDocument();
    const exportButton = container.querySelector('button svg.lucide-download');
    expect(exportButton).toBeInTheDocument();
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

    const { container } = render(<TimeSeries />);

    // Find export button by finding its svg icon and getting parent button
    const exportIcon = container.querySelector('svg.lucide-download');
    const exportButton = exportIcon?.closest('button');
    expect(exportButton).toBeInTheDocument();
    fireEvent.click(exportButton!);

    expect(mockCreateObjectURL).toHaveBeenCalled();
    expect(mockClick).toHaveBeenCalled();

    vi.restoreAllMocks();
  });

  it('displays 4 main tabs', () => {
    render(<TimeSeries />);

    expect(screen.getByRole('tab', { name: /Trends & Forecast/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Seasonality/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Anomalies/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Period Comparison/i })).toBeInTheDocument();
  });

  it('shows Trends & Forecast tab content by default', () => {
    render(<TimeSeries />);

    expect(screen.getByText('Time Series with Forecast')).toBeInTheDocument();
    expect(screen.getByText('95% CI')).toBeInTheDocument();
  });

  it('displays forecast metrics cards', () => {
    render(<TimeSeries />);

    // Forecast metrics section
    expect(screen.getByText('Mean Absolute Percentage Error')).toBeInTheDocument();
    expect(screen.getByText('Root Mean Square Error')).toBeInTheDocument();
    expect(screen.getByText('Mean Absolute Error')).toBeInTheDocument();
    expect(screen.getByText('Coefficient of Determination')).toBeInTheDocument();
  });

  it('displays default metric selection', () => {
    render(<TimeSeries />);

    // Default selected metric should be visible (TRx Volume is first metric)
    expect(screen.getByText('TRx Volume')).toBeInTheDocument();
    // Default time range should be visible
    expect(screen.getByText('90 Days')).toBeInTheDocument();
  });

  it('has clickable Seasonality tab', () => {
    render(<TimeSeries />);

    const seasonalityTab = screen.getByRole('tab', { name: /Seasonality/i });
    expect(seasonalityTab).toBeInTheDocument();
    expect(seasonalityTab).not.toBeDisabled();
  });

  it('has clickable Anomalies tab', () => {
    render(<TimeSeries />);

    const anomaliesTab = screen.getByRole('tab', { name: /Anomalies/i });
    expect(anomaliesTab).toBeInTheDocument();
    expect(anomaliesTab).not.toBeDisabled();
  });
});
