/**
 * DriftVisualization Component Tests
 * ===================================
 *
 * Tests for the DriftVisualization component and its sub-components.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
  DriftVisualization,
  type DriftMetric,
  type DriftSummary,
  type DriftType,
  type DriftSeverity,
} from './DriftVisualization';

// Helper to create mock drift metrics
const createMockMetric = (overrides: Partial<DriftMetric> = {}): DriftMetric => ({
  name: 'Feature X',
  type: 'data',
  current_value: 0.15,
  threshold: 0.2,
  severity: 'low',
  trend: 'stable',
  history: [
    { timestamp: '2026-01-01T00:00:00Z', value: 0.10 },
    { timestamp: '2026-01-02T00:00:00Z', value: 0.12 },
    { timestamp: '2026-01-03T00:00:00Z', value: 0.15 },
  ],
  description: 'Distribution drift for Feature X',
  last_updated: '2026-01-04T12:00:00Z',
  ...overrides,
});

const createMockSummary = (overrides: Partial<DriftSummary> = {}): DriftSummary => ({
  total_features: 50,
  drifting_features: 5,
  model_drift_detected: false,
  data_drift_percentage: 10.0,
  last_check: '2026-01-04T12:00:00Z',
  alerts: [],
  ...overrides,
});

describe('DriftVisualization', () => {
  describe('Rendering', () => {
    it('renders without crashing', () => {
      render(<DriftVisualization metrics={[]} />);
      expect(screen.getByText(/No drift metrics/)).toBeInTheDocument();
    });

    it('renders metric cards for each metric', () => {
      const metrics = [
        createMockMetric({ name: 'Feature A' }),
        createMockMetric({ name: 'Feature B' }),
      ];
      render(<DriftVisualization metrics={metrics} />);

      expect(screen.getByText('Feature A')).toBeInTheDocument();
      expect(screen.getByText('Feature B')).toBeInTheDocument();
    });

    it('applies custom className', () => {
      const { container } = render(
        <DriftVisualization metrics={[]} className="custom-class" />
      );
      expect(container.firstChild).toHaveClass('custom-class');
    });
  });

  describe('Summary Panel', () => {
    it('renders summary panel when summary is provided', () => {
      const summary = createMockSummary();
      render(<DriftVisualization metrics={[]} summary={summary} />);

      expect(screen.getByText('Drift Summary')).toBeInTheDocument();
    });

    it('displays total features count', () => {
      const summary = createMockSummary({ total_features: 100 });
      render(<DriftVisualization metrics={[]} summary={summary} />);

      expect(screen.getByText('100')).toBeInTheDocument();
      expect(screen.getByText('Total Features')).toBeInTheDocument();
    });

    it('displays drifting features count', () => {
      const summary = createMockSummary({ drifting_features: 8 });
      render(<DriftVisualization metrics={[]} summary={summary} />);

      expect(screen.getByText('8')).toBeInTheDocument();
      expect(screen.getByText('Drifting')).toBeInTheDocument();
    });

    it('displays data drift percentage', () => {
      const summary = createMockSummary({ data_drift_percentage: 15.5 });
      render(<DriftVisualization metrics={[]} summary={summary} />);

      // The percentage may be rendered with toFixed(1) which produces "15.5%"
      expect(screen.getByText(/15\.5%/)).toBeInTheDocument();
      // "Data Drift" appears in both summary card label and filter dropdown
      const dataDriftMatches = screen.getAllByText(/Data Drift/i);
      expect(dataDriftMatches.length).toBeGreaterThan(0);
    });

    it('shows model drift status as Stable when not detected', () => {
      const summary = createMockSummary({ model_drift_detected: false });
      render(<DriftVisualization metrics={[]} summary={summary} />);

      expect(screen.getByText('Stable')).toBeInTheDocument();
    });

    it('shows model drift status as Detected when detected', () => {
      const summary = createMockSummary({ model_drift_detected: true });
      render(<DriftVisualization metrics={[]} summary={summary} />);

      expect(screen.getByText('Detected')).toBeInTheDocument();
    });

    it('displays alerts when present', () => {
      const summary = createMockSummary({
        alerts: [
          { feature: 'Feature Y', severity: 'high', message: 'Significant drift detected' },
        ],
      });
      render(<DriftVisualization metrics={[]} summary={summary} />);

      expect(screen.getByText('Active Alerts')).toBeInTheDocument();
      expect(screen.getByText('Feature Y:')).toBeInTheDocument();
      expect(screen.getByText('Significant drift detected')).toBeInTheDocument();
    });

    it('limits displayed alerts to 3', () => {
      const summary = createMockSummary({
        alerts: [
          { feature: 'Alert 1', severity: 'high', message: 'Message 1' },
          { feature: 'Alert 2', severity: 'medium', message: 'Message 2' },
          { feature: 'Alert 3', severity: 'low', message: 'Message 3' },
          { feature: 'Alert 4', severity: 'high', message: 'Message 4' },
        ],
      });
      render(<DriftVisualization metrics={[]} summary={summary} />);

      expect(screen.getByText('Alert 1:')).toBeInTheDocument();
      expect(screen.getByText('Alert 2:')).toBeInTheDocument();
      expect(screen.getByText('Alert 3:')).toBeInTheDocument();
      expect(screen.queryByText('Alert 4:')).not.toBeInTheDocument();
    });

    it('displays last check time', () => {
      const summary = createMockSummary({ last_check: '2026-01-04T12:00:00Z' });
      render(<DriftVisualization metrics={[]} summary={summary} />);

      expect(screen.getByText(/Last check:/)).toBeInTheDocument();
    });
  });

  describe('Drift Metric Cards', () => {
    it('displays metric name', () => {
      const metrics = [createMockMetric({ name: 'Important Feature' })];
      render(<DriftVisualization metrics={metrics} />);

      expect(screen.getByText('Important Feature')).toBeInTheDocument();
    });

    it('displays current value', () => {
      const metrics = [createMockMetric({ current_value: 0.123 })];
      render(<DriftVisualization metrics={metrics} />);

      expect(screen.getByText('0.123')).toBeInTheDocument();
    });

    it('displays threshold', () => {
      const metrics = [createMockMetric({ threshold: 0.200 })];
      render(<DriftVisualization metrics={metrics} />);

      expect(screen.getByText(/Threshold: 0.200/)).toBeInTheDocument();
    });

    it('displays description when provided', () => {
      const metrics = [createMockMetric({ description: 'This is a test description' })];
      render(<DriftVisualization metrics={metrics} />);

      expect(screen.getByText('This is a test description')).toBeInTheDocument();
    });

    it('displays last updated time', () => {
      const metrics = [createMockMetric({ last_updated: '2026-01-04T12:00:00Z' })];
      render(<DriftVisualization metrics={metrics} />);

      expect(screen.getByText(/Updated/)).toBeInTheDocument();
    });
  });

  describe('Severity Badges', () => {
    it.each([
      ['none', 'No Drift'],
      ['low', 'Low'],
      ['medium', 'Medium'],
      ['high', 'High'],
      ['critical', 'Critical'],
    ] as [DriftSeverity, string][])('displays %s severity as "%s"', (severity, label) => {
      const metrics = [createMockMetric({ severity })];
      render(<DriftVisualization metrics={metrics} />);

      // The label may appear multiple times (in badge and filter options)
      // so we check for at least one occurrence
      const matches = screen.getAllByText(label);
      expect(matches.length).toBeGreaterThan(0);
    });
  });

  describe('Trend Indicators', () => {
    it('shows increasing trend indicator', () => {
      const metrics = [createMockMetric({ trend: 'increasing' })];
      const { container } = render(<DriftVisualization metrics={metrics} />);

      // TrendingUp icon should be present in a red-colored container
      const trendContainer = container.querySelector('.text-red-500');
      expect(trendContainer).toBeInTheDocument();
      expect(trendContainer?.querySelector('svg')).toBeInTheDocument();
    });

    it('shows decreasing trend indicator', () => {
      const metrics = [createMockMetric({ trend: 'decreasing' })];
      const { container } = render(<DriftVisualization metrics={metrics} />);

      // TrendingDown icon should be present in a green-colored container
      const trendContainer = container.querySelector('.text-green-500');
      expect(trendContainer).toBeInTheDocument();
      expect(trendContainer?.querySelector('svg')).toBeInTheDocument();
    });

    it('shows stable trend indicator', () => {
      const metrics = [createMockMetric({ trend: 'stable' })];
      const { container } = render(<DriftVisualization metrics={metrics} />);

      // Activity icon should be present in a gray-colored container
      const trendContainer = container.querySelector('.text-gray-500');
      expect(trendContainer).toBeInTheDocument();
      expect(trendContainer?.querySelector('svg')).toBeInTheDocument();
    });
  });

  describe('Sparkline', () => {
    it('renders sparkline when showHistory is true and history has data', () => {
      const metrics = [
        createMockMetric({
          history: [
            { timestamp: '2026-01-01', value: 0.1 },
            { timestamp: '2026-01-02', value: 0.15 },
          ],
        }),
      ];
      const { container } = render(<DriftVisualization metrics={metrics} showHistory={true} />);

      const svg = container.querySelector('svg[width="120"]');
      expect(svg).toBeInTheDocument();
    });

    it('does not render sparkline when showHistory is false', () => {
      const metrics = [
        createMockMetric({
          history: [
            { timestamp: '2026-01-01', value: 0.1 },
            { timestamp: '2026-01-02', value: 0.15 },
          ],
        }),
      ];
      const { container } = render(<DriftVisualization metrics={metrics} showHistory={false} />);

      const svg = container.querySelector('svg[width="120"]');
      expect(svg).not.toBeInTheDocument();
    });

    it('does not render sparkline with insufficient history data', () => {
      const metrics = [
        createMockMetric({
          history: [{ timestamp: '2026-01-01', value: 0.1 }], // Only 1 point
        }),
      ];
      const { container } = render(<DriftVisualization metrics={metrics} showHistory={true} />);

      const svg = container.querySelector('svg[width="120"]');
      expect(svg).not.toBeInTheDocument();
    });
  });

  describe('Filtering', () => {
    const mixedMetrics = [
      createMockMetric({ name: 'Data Metric 1', type: 'data', severity: 'low' }),
      createMockMetric({ name: 'Data Metric 2', type: 'data', severity: 'high' }),
      createMockMetric({ name: 'Model Metric', type: 'model', severity: 'critical' }),
      createMockMetric({ name: 'Concept Metric', type: 'concept', severity: 'medium' }),
      createMockMetric({ name: 'Feature Metric', type: 'feature', severity: 'none' }),
    ];

    it('shows all metrics by default', () => {
      render(<DriftVisualization metrics={mixedMetrics} />);

      expect(screen.getByText('Data Metric 1')).toBeInTheDocument();
      expect(screen.getByText('Data Metric 2')).toBeInTheDocument();
      expect(screen.getByText('Model Metric')).toBeInTheDocument();
      expect(screen.getByText('Concept Metric')).toBeInTheDocument();
      expect(screen.getByText('Feature Metric')).toBeInTheDocument();
    });

    it('filters by drift type', async () => {
      const user = userEvent.setup();
      render(<DriftVisualization metrics={mixedMetrics} />);

      const typeSelect = screen.getAllByRole('combobox')[0];
      await user.selectOptions(typeSelect, 'data');

      expect(screen.getByText('Data Metric 1')).toBeInTheDocument();
      expect(screen.getByText('Data Metric 2')).toBeInTheDocument();
      expect(screen.queryByText('Model Metric')).not.toBeInTheDocument();
      expect(screen.queryByText('Concept Metric')).not.toBeInTheDocument();
      expect(screen.queryByText('Feature Metric')).not.toBeInTheDocument();
    });

    it('filters by severity', async () => {
      const user = userEvent.setup();
      render(<DriftVisualization metrics={mixedMetrics} />);

      const severitySelect = screen.getAllByRole('combobox')[1];
      await user.selectOptions(severitySelect, 'high');

      expect(screen.queryByText('Data Metric 1')).not.toBeInTheDocument();
      expect(screen.getByText('Data Metric 2')).toBeInTheDocument();
      expect(screen.queryByText('Model Metric')).not.toBeInTheDocument();
    });

    it('combines type and severity filters', async () => {
      const user = userEvent.setup();
      render(<DriftVisualization metrics={mixedMetrics} />);

      const typeSelect = screen.getAllByRole('combobox')[0];
      const severitySelect = screen.getAllByRole('combobox')[1];

      await user.selectOptions(typeSelect, 'data');
      await user.selectOptions(severitySelect, 'high');

      expect(screen.queryByText('Data Metric 1')).not.toBeInTheDocument();
      expect(screen.getByText('Data Metric 2')).toBeInTheDocument();
      expect(screen.queryByText('Model Metric')).not.toBeInTheDocument();
    });

    it('shows empty state when no metrics match filters', async () => {
      const user = userEvent.setup();
      const metrics = [createMockMetric({ type: 'data', severity: 'low' })];
      render(<DriftVisualization metrics={metrics} />);

      const severitySelect = screen.getAllByRole('combobox')[1];
      await user.selectOptions(severitySelect, 'critical');

      expect(screen.getByText(/No drift metrics match/)).toBeInTheDocument();
    });
  });

  describe('Grouping by Type', () => {
    it('groups metrics by drift type', () => {
      const metrics = [
        createMockMetric({ name: 'Data Feature', type: 'data' }),
        createMockMetric({ name: 'Model Feature', type: 'model' }),
      ];
      render(<DriftVisualization metrics={metrics} />);

      // Headers show "{type} Drift" pattern (CSS capitalize makes it appear capitalized)
      // Use getAllByText since there might be multiple matches (filters, headers, etc.)
      const dataMatches = screen.getAllByText(/data\s+Drift/i);
      const modelMatches = screen.getAllByText(/model\s+Drift/i);
      expect(dataMatches.length).toBeGreaterThan(0);
      expect(modelMatches.length).toBeGreaterThan(0);
    });

    it('shows count for each group', () => {
      const metrics = [
        createMockMetric({ name: 'Data 1', type: 'data' }),
        createMockMetric({ name: 'Data 2', type: 'data' }),
        createMockMetric({ name: 'Model 1', type: 'model' }),
      ];
      render(<DriftVisualization metrics={metrics} />);

      expect(screen.getByText('(2)')).toBeInTheDocument(); // data group
      expect(screen.getByText('(1)')).toBeInTheDocument(); // model group
    });
  });

  describe('Click Handler', () => {
    it('calls onMetricClick when metric card is clicked', async () => {
      const user = userEvent.setup();
      const handleClick = vi.fn();
      const metrics = [createMockMetric({ name: 'Clickable Metric' })];

      render(<DriftVisualization metrics={metrics} onMetricClick={handleClick} />);

      await user.click(screen.getByText('Clickable Metric'));
      expect(handleClick).toHaveBeenCalledTimes(1);
      expect(handleClick).toHaveBeenCalledWith(metrics[0]);
    });

    it('does not have click styling when onMetricClick is not provided', () => {
      const metrics = [createMockMetric({ name: 'Non-clickable' })];
      const { container } = render(<DriftVisualization metrics={metrics} />);

      const cards = container.querySelectorAll('[class*="cursor-pointer"]');
      expect(cards.length).toBe(0);
    });

    it('has click styling when onMetricClick is provided', () => {
      const metrics = [createMockMetric({ name: 'Clickable' })];
      const { container } = render(
        <DriftVisualization metrics={metrics} onMetricClick={() => {}} />
      );

      const cards = container.querySelectorAll('[class*="cursor-pointer"]');
      expect(cards.length).toBeGreaterThan(0);
    });
  });

  describe('Progress Bar', () => {
    it('shows green progress bar when under threshold', () => {
      const metrics = [createMockMetric({ current_value: 0.1, threshold: 0.2 })];
      const { container } = render(<DriftVisualization metrics={metrics} />);

      const progressBar = container.querySelector('.bg-green-500');
      expect(progressBar).toBeInTheDocument();
    });

    it('shows red progress bar when over threshold', () => {
      const metrics = [createMockMetric({ current_value: 0.25, threshold: 0.2 })];
      const { container } = render(<DriftVisualization metrics={metrics} />);

      const progressBar = container.querySelector('.bg-red-500');
      expect(progressBar).toBeInTheDocument();
    });

    it('caps progress bar at 100%', () => {
      const metrics = [createMockMetric({ current_value: 0.5, threshold: 0.2 })];
      const { container } = render(<DriftVisualization metrics={metrics} />);

      const progressBar = container.querySelector('.bg-red-500') as HTMLElement;
      expect(progressBar?.style.width).toBe('100%');
    });
  });

  describe('Drift Types', () => {
    it.each([
      ['data', 'data\\s+Drift'],
      ['model', 'model\\s+Drift'],
      ['concept', 'concept\\s+Drift'],
      ['feature', 'feature\\s+Drift'],
    ] as [DriftType, string][])('renders %s type with correct header', (type, expectedPattern) => {
      const metrics = [createMockMetric({ type })];
      render(<DriftVisualization metrics={metrics} />);

      // Headers show "{type} Drift" pattern (CSS capitalize makes it appear capitalized)
      // Use getAllByText since there might be multiple matches (filters, headers, etc.)
      const matches = screen.getAllByText(new RegExp(expectedPattern, 'i'));
      expect(matches.length).toBeGreaterThan(0);
    });
  });

  describe('Empty State', () => {
    it('shows empty state when no metrics', () => {
      render(<DriftVisualization metrics={[]} />);

      expect(screen.getByText(/No drift metrics match/)).toBeInTheDocument();
    });

    it('does not show summary when not provided', () => {
      render(<DriftVisualization metrics={[]} />);

      expect(screen.queryByText('Drift Summary')).not.toBeInTheDocument();
    });
  });
});
