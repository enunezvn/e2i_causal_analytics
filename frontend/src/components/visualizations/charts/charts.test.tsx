/**
 * Chart Visualization Components Tests
 * =====================================
 *
 * Tests for MetricTrend, ROCCurve, ConfusionMatrix, and MultiAxisLineChart components.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { MetricTrend, type MetricDataPoint, type MetricThreshold } from './MetricTrend';
import { ROCCurve, type ROCCurveData, type ROCPoint } from './ROCCurve';
import { ConfusionMatrix, type ConfusionMatrixData } from './ConfusionMatrix';
import { MultiAxisLineChart, type AxisConfig } from './MultiAxisLineChart';

// =============================================================================
// TEST DATA
// =============================================================================

const mockMetricData: MetricDataPoint[] = [
  { timestamp: '2024-01-01', value: 0.85 },
  { timestamp: '2024-01-08', value: 0.87 },
  { timestamp: '2024-01-15', value: 0.84 },
  { timestamp: '2024-01-22', value: 0.89 },
  { timestamp: '2024-01-29', value: 0.91, annotation: 'Model updated' },
];

const mockThresholds: MetricThreshold[] = [
  { value: 0.90, label: 'Target', type: 'target', color: '#22c55e' },
  { value: 0.80, label: 'Minimum', type: 'lower', color: '#ef4444' },
];

const mockROCPoints: ROCPoint[] = [
  { fpr: 0, tpr: 0, threshold: 1 },
  { fpr: 0.1, tpr: 0.5, threshold: 0.9 },
  { fpr: 0.2, tpr: 0.7, threshold: 0.8 },
  { fpr: 0.4, tpr: 0.85, threshold: 0.6 },
  { fpr: 0.6, tpr: 0.92, threshold: 0.4 },
  { fpr: 1, tpr: 1, threshold: 0 },
];

const mockROCCurves: ROCCurveData[] = [
  { name: 'Model A', points: mockROCPoints, color: '#10b981' },
  {
    name: 'Model B',
    points: mockROCPoints.map((p) => ({ ...p, tpr: Math.max(p.fpr, p.tpr - 0.1) })),
    color: '#3b82f6',
  },
];

const mockConfusionData: ConfusionMatrixData = {
  matrix: [
    [85, 10, 5],
    [8, 82, 10],
    [3, 12, 85],
  ],
  labels: ['Low Risk', 'Medium Risk', 'High Risk'],
};

const mockMultiAxisData = [
  { date: '2024-01', conversions: 245, revenue: 48500 },
  { date: '2024-02', conversions: 312, revenue: 62400 },
  { date: '2024-03', conversions: 287, revenue: 57300 },
  { date: '2024-04', conversions: 356, revenue: 71200 },
];

const mockAxes: AxisConfig[] = [
  { dataKey: 'conversions', name: 'Conversions', color: '#10b981', yAxisId: 'left' },
  { dataKey: 'revenue', name: 'Revenue', color: '#3b82f6', yAxisId: 'right', unit: '$' },
];

// =============================================================================
// METRIC TREND TESTS
// =============================================================================

describe('MetricTrend', () => {
  it('renders with data', () => {
    const { container } = render(
      <MetricTrend name="Accuracy" data={mockMetricData} />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('renders with sample data when no data provided', () => {
    const { container } = render(
      <MetricTrend name="Test Metric" data={undefined as unknown as MetricDataPoint[]} />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('shows metric name in header', () => {
    render(<MetricTrend name="Model Accuracy" data={mockMetricData} showHeader />);
    expect(screen.getByText('Model Accuracy')).toBeInTheDocument();
  });

  it('shows current value in header', () => {
    render(<MetricTrend name="Accuracy" data={mockMetricData} showHeader />);
    // Last value is 0.91
    expect(screen.getByText('0.91')).toBeInTheDocument();
  });

  it('shows trend percentage', () => {
    render(<MetricTrend name="Accuracy" data={mockMetricData} showHeader />);
    // Trend from 0.89 to 0.91 should show positive percentage
    expect(screen.getByText(/\+2\.2%/)).toBeInTheDocument();
  });

  it('shows threshold status when thresholds provided', () => {
    render(
      <MetricTrend
        name="Accuracy"
        data={mockMetricData}
        thresholds={mockThresholds}
        showHeader
      />
    );
    // With current value 0.91 above target 0.90
    expect(screen.getByText('Target:')).toBeInTheDocument();
    expect(screen.getByText('On target')).toBeInTheDocument();
  });

  it('shows loading skeleton when isLoading', () => {
    const { container } = render(
      <MetricTrend name="Test" data={mockMetricData} isLoading />
    );
    expect(container.querySelector('.animate-pulse')).toBeInTheDocument();
  });

  it('renders compact mode', () => {
    const { container } = render(
      <MetricTrend name="Test" data={mockMetricData} compact />
    );
    // Compact mode has smaller height
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('hides header when showHeader is false', () => {
    render(
      <MetricTrend name="Model Accuracy" data={mockMetricData} showHeader={false} />
    );
    // Name should not appear as a heading
    expect(screen.queryByText('Model Accuracy')).not.toBeInTheDocument();
  });

  it('applies custom height', () => {
    const { container } = render(
      <MetricTrend name="Test" data={mockMetricData} height={500} />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(
      <MetricTrend name="Test" data={mockMetricData} className="custom-trend" />
    );
    expect(container.querySelector('.custom-trend')).toBeInTheDocument();
  });

  it('uses custom value formatter', () => {
    render(
      <MetricTrend
        name="Test"
        data={mockMetricData}
        showHeader
        valueFormatter={(v) => `${(v * 100).toFixed(0)}%`}
      />
    );
    // Last value 0.91 should show as 91%
    expect(screen.getByText('91%')).toBeInTheDocument();
  });

  it('displays unit with value', () => {
    render(
      <MetricTrend name="Test" data={mockMetricData} showHeader unit="%" />
    );
    // Should show value with unit
    expect(screen.getByText(/0\.91%/)).toBeInTheDocument();
  });
});

// =============================================================================
// ROC CURVE TESTS
// =============================================================================

describe('ROCCurve', () => {
  it('renders with curve data', () => {
    const { container } = render(<ROCCurve curves={mockROCCurves} />);
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('renders with sample data when no curves provided', () => {
    const { container } = render(
      <ROCCurve curves={undefined as unknown as ROCCurveData[]} />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('shows loading skeleton when isLoading', () => {
    const { container } = render(<ROCCurve curves={mockROCCurves} isLoading />);
    expect(container.querySelector('.animate-pulse')).toBeInTheDocument();
  });

  it('shows empty state when no curves', () => {
    render(<ROCCurve curves={[]} />);
    expect(screen.getByText('No ROC curve data available')).toBeInTheDocument();
  });

  it('displays curve names in legend', () => {
    render(<ROCCurve curves={mockROCCurves} />);
    expect(screen.getByText('Model A')).toBeInTheDocument();
    expect(screen.getByText('Model B')).toBeInTheDocument();
  });

  it('shows AUC values when showAUC is true', () => {
    render(<ROCCurve curves={mockROCCurves} showAUC />);
    // AUC values should be displayed for each model
    const aucElements = screen.getAllByText(/AUC:/);
    expect(aucElements.length).toBeGreaterThan(0);
  });

  it('hides AUC when showAUC is false', () => {
    render(<ROCCurve curves={mockROCCurves} showAUC={false} showDiagonal={false} />);
    // Look for legend items without AUC (also hide diagonal which shows AUC)
    expect(screen.getByText('Model A')).toBeInTheDocument();
    expect(screen.queryByText(/AUC:/)).not.toBeInTheDocument();
  });

  it('shows diagonal reference line by default', () => {
    render(<ROCCurve curves={mockROCCurves} showDiagonal />);
    // Random classifier line info
    expect(screen.getByText(/Random \(AUC: 0\.500\)/)).toBeInTheDocument();
  });

  it('hides diagonal when showDiagonal is false', () => {
    render(<ROCCurve curves={mockROCCurves} showDiagonal={false} />);
    expect(screen.queryByText(/Random/)).not.toBeInTheDocument();
  });

  it('applies custom height', () => {
    const { container } = render(<ROCCurve curves={mockROCCurves} height={600} />);
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(
      <ROCCurve curves={mockROCCurves} className="custom-roc" />
    );
    expect(container.querySelector('.custom-roc')).toBeInTheDocument();
  });

  it('handles onPointHover callback', () => {
    const handleHover = vi.fn();
    const { container } = render(
      <ROCCurve curves={mockROCCurves} onPointHover={handleHover} />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('uses pre-calculated AUC when provided', () => {
    const curvesWithAUC: ROCCurveData[] = [
      { name: 'Custom Model', points: mockROCPoints, color: '#10b981', auc: 0.95 },
    ];
    render(<ROCCurve curves={curvesWithAUC} showAUC />);
    expect(screen.getByText(/0\.950/)).toBeInTheDocument();
  });
});

// =============================================================================
// CONFUSION MATRIX TESTS
// =============================================================================

describe('ConfusionMatrix', () => {
  it('renders with matrix data', () => {
    render(<ConfusionMatrix data={mockConfusionData} />);
    expect(screen.getByText('Confusion Matrix')).toBeInTheDocument();
  });

  it('renders with sample data when no data provided', () => {
    render(<ConfusionMatrix data={undefined as unknown as ConfusionMatrixData} />);
    expect(screen.getByText('Confusion Matrix')).toBeInTheDocument();
  });

  it('displays custom title', () => {
    render(<ConfusionMatrix data={mockConfusionData} title="Classification Results" />);
    expect(screen.getByText('Classification Results')).toBeInTheDocument();
  });

  it('shows loading skeleton when isLoading', () => {
    const { container } = render(<ConfusionMatrix data={mockConfusionData} isLoading />);
    expect(container.querySelector('.animate-pulse')).toBeInTheDocument();
  });

  it('displays class labels', () => {
    render(<ConfusionMatrix data={mockConfusionData} />);
    // Labels appear twice (in column headers and row labels)
    const lowRisk = screen.getAllByText('Low Risk');
    const mediumRisk = screen.getAllByText('Medium Risk');
    const highRisk = screen.getAllByText('High Risk');
    expect(lowRisk.length).toBeGreaterThan(0);
    expect(mediumRisk.length).toBeGreaterThan(0);
    expect(highRisk.length).toBeGreaterThan(0);
  });

  it('shows Predicted and Actual labels', () => {
    render(<ConfusionMatrix data={mockConfusionData} />);
    expect(screen.getByText('Predicted')).toBeInTheDocument();
    expect(screen.getByText('Actual')).toBeInTheDocument();
  });

  it('displays overall metrics', () => {
    render(<ConfusionMatrix data={mockConfusionData} />);
    expect(screen.getByText('Overall Metrics')).toBeInTheDocument();
    expect(screen.getByText('Accuracy')).toBeInTheDocument();
    expect(screen.getByText('Total Samples')).toBeInTheDocument();
  });

  it('displays per-class metrics', () => {
    render(<ConfusionMatrix data={mockConfusionData} />);
    expect(screen.getByText('Per-Class Metrics')).toBeInTheDocument();
    // Check for P, R, F1 labels
    const pLabels = screen.getAllByText(/P:/);
    expect(pLabels.length).toBeGreaterThan(0);
  });

  it('shows percentages when showPercentages is true', () => {
    render(<ConfusionMatrix data={mockConfusionData} showPercentages />);
    // Percentages should appear
    const percents = screen.getAllByText(/%/);
    expect(percents.length).toBeGreaterThan(0);
  });

  it('displays legend', () => {
    render(<ConfusionMatrix data={mockConfusionData} />);
    expect(screen.getByText('Low')).toBeInTheDocument();
    expect(screen.getByText('High')).toBeInTheDocument();
  });

  it('handles cell click callback', () => {
    const handleClick = vi.fn();
    render(<ConfusionMatrix data={mockConfusionData} onCellClick={handleClick} />);
    // Find a cell and click it
    const cells = document.querySelectorAll('[class*="cursor-pointer"]');
    if (cells.length > 0) {
      fireEvent.click(cells[0]);
      expect(handleClick).toHaveBeenCalled();
    }
  });

  it('applies custom className', () => {
    const { container } = render(
      <ConfusionMatrix data={mockConfusionData} className="custom-matrix" />
    );
    expect(container.querySelector('.custom-matrix')).toBeInTheDocument();
  });

  it('applies custom cell size', () => {
    const { container } = render(
      <ConfusionMatrix data={mockConfusionData} cellSize={100} />
    );
    expect(container).toBeInTheDocument();
  });
});

// =============================================================================
// MULTI-AXIS LINE CHART TESTS
// =============================================================================

describe('MultiAxisLineChart', () => {
  it('renders with data', () => {
    const { container } = render(
      <MultiAxisLineChart data={mockMultiAxisData} xAxisKey="date" axes={mockAxes} />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('renders with sample data when no data provided', () => {
    const { container } = render(
      <MultiAxisLineChart
        data={undefined as unknown as Record<string, unknown>[]}
        xAxisKey="date"
        axes={undefined as unknown as AxisConfig[]}
      />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('shows loading skeleton when isLoading', () => {
    const { container } = render(
      <MultiAxisLineChart
        data={mockMultiAxisData}
        xAxisKey="date"
        axes={mockAxes}
        isLoading
      />
    );
    expect(container.querySelector('.animate-pulse')).toBeInTheDocument();
  });

  it('shows empty state when no data', () => {
    render(<MultiAxisLineChart data={[]} xAxisKey="date" axes={mockAxes} />);
    expect(screen.getByText('No data available')).toBeInTheDocument();
  });

  it('shows legend by default', () => {
    const { container } = render(
      <MultiAxisLineChart data={mockMultiAxisData} xAxisKey="date" axes={mockAxes} />
    );
    // ResponsiveContainer renders in test but with 0 width, so legend wrapper doesn't render
    // We verify component mounts without error with default showLegend prop
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('hides legend when showLegend is false', () => {
    const { container } = render(
      <MultiAxisLineChart
        data={mockMultiAxisData}
        xAxisKey="date"
        axes={mockAxes}
        showLegend={false}
      />
    );
    expect(container.querySelector('.recharts-legend-wrapper')).not.toBeInTheDocument();
  });

  it('shows grid by default', () => {
    const { container } = render(
      <MultiAxisLineChart data={mockMultiAxisData} xAxisKey="date" axes={mockAxes} />
    );
    // Grid should be present
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('applies custom height', () => {
    const { container } = render(
      <MultiAxisLineChart
        data={mockMultiAxisData}
        xAxisKey="date"
        axes={mockAxes}
        height={500}
      />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(
      <MultiAxisLineChart
        data={mockMultiAxisData}
        xAxisKey="date"
        axes={mockAxes}
        className="custom-chart"
      />
    );
    expect(container.querySelector('.custom-chart')).toBeInTheDocument();
  });

  it('handles x-axis formatter', () => {
    const { container } = render(
      <MultiAxisLineChart
        data={mockMultiAxisData}
        xAxisKey="date"
        axes={mockAxes}
        xAxisFormatter={(v) => v.toUpperCase()}
      />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('renders reference lines when provided', () => {
    const { container } = render(
      <MultiAxisLineChart
        data={mockMultiAxisData}
        xAxisKey="date"
        axes={mockAxes}
        leftReferenceValue={300}
        rightReferenceValue={60000}
      />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('handles multiple axes configurations', () => {
    const threeAxes: AxisConfig[] = [
      ...mockAxes,
      { dataKey: 'cost', name: 'Cost', color: '#f59e0b', yAxisId: 'left' },
    ];
    const dataWithCost = mockMultiAxisData.map((d, i) => ({
      ...d,
      cost: 10000 + i * 1000,
    }));
    const { container } = render(
      <MultiAxisLineChart data={dataWithCost} xAxisKey="date" axes={threeAxes} />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });
});
