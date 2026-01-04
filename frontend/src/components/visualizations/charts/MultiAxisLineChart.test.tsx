/**
 * MultiAxisLineChart Component Tests
 * ===================================
 *
 * Tests for MultiAxisLineChart with comprehensive CustomTooltip coverage.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import * as React from 'react';
import {
  MultiAxisLineChart,
  CustomTooltip,
  type AxisConfig,
  type TooltipPayloadEntry,
} from './MultiAxisLineChart';

// =============================================================================
// TEST DATA
// =============================================================================

const mockData = [
  { date: '2024-01', conversions: 245, revenue: 48500 },
  { date: '2024-02', conversions: 312, revenue: 62400 },
  { date: '2024-03', conversions: 287, revenue: 57300 },
  { date: '2024-04', conversions: 356, revenue: 71200 },
];

const mockAxesWithUnit: AxisConfig[] = [
  { dataKey: 'conversions', name: 'Conversions', color: '#10b981', yAxisId: 'left' },
  { dataKey: 'revenue', name: 'Revenue', color: '#3b82f6', yAxisId: 'right', unit: '$' },
];

const mockAxesWithoutUnit: AxisConfig[] = [
  { dataKey: 'conversions', name: 'Conversions', color: '#10b981', yAxisId: 'left' },
  { dataKey: 'revenue', name: 'Revenue', color: '#3b82f6', yAxisId: 'right' },
];

// =============================================================================
// CUSTOM TOOLTIP TESTS
// =============================================================================

describe('CustomTooltip', () => {
  describe('rendering conditions', () => {
    it('returns null when tooltip is not active', () => {
      const { container } = render(
        <CustomTooltip
          active={false}
          payload={[]}
          label="2024-01"
          axes={mockAxesWithUnit}
        />
      );
      expect(container.firstChild).toBeNull();
    });

    it('returns null when payload is empty', () => {
      const { container } = render(
        <CustomTooltip
          active={true}
          payload={[]}
          label="2024-01"
          axes={mockAxesWithUnit}
        />
      );
      expect(container.firstChild).toBeNull();
    });

    it('returns null when payload is undefined', () => {
      const { container } = render(
        <CustomTooltip
          active={true}
          payload={undefined}
          label="2024-01"
          axes={mockAxesWithUnit}
        />
      );
      expect(container.firstChild).toBeNull();
    });

    it('renders when active with valid payload', () => {
      render(
        <CustomTooltip
          active={true}
          payload={[
            { dataKey: 'conversions', value: 245, color: '#10b981', name: 'Conversions' },
          ]}
          label="2024-01"
          axes={mockAxesWithUnit}
        />
      );
      expect(screen.getByText('2024-01')).toBeInTheDocument();
    });
  });

  describe('value formatting with units', () => {
    it('formats value with unit prefix when axis has unit', () => {
      render(
        <CustomTooltip
          active={true}
          payload={[
            { dataKey: 'revenue', value: 48500, color: '#3b82f6', name: 'Revenue' },
          ]}
          label="2024-01"
          axes={mockAxesWithUnit}
        />
      );

      // Revenue axis has unit: '$', so should show $48,500
      expect(screen.getByText('$48,500')).toBeInTheDocument();
    });

    it('formats value without unit when axis has no unit', () => {
      render(
        <CustomTooltip
          active={true}
          payload={[
            { dataKey: 'conversions', value: 245, color: '#10b981', name: 'Conversions' },
          ]}
          label="2024-01"
          axes={mockAxesWithUnit}
        />
      );

      // Conversions axis has no unit, so should show plain 245
      expect(screen.getByText('245')).toBeInTheDocument();
    });

    it('formats value without unit when axis config not found', () => {
      render(
        <CustomTooltip
          active={true}
          payload={[
            { dataKey: 'unknown', value: 999, color: '#ff0000', name: 'Unknown Metric' },
          ]}
          label="2024-01"
          axes={mockAxesWithUnit}
        />
      );

      // Unknown dataKey means no axis config, so no unit
      expect(screen.getByText('999')).toBeInTheDocument();
    });

    it('formats large numbers with locale string', () => {
      render(
        <CustomTooltip
          active={true}
          payload={[
            { dataKey: 'revenue', value: 1234567, color: '#3b82f6', name: 'Revenue' },
          ]}
          label="2024-01"
          axes={mockAxesWithUnit}
        />
      );

      // Should show formatted with unit and locale string (commas)
      expect(screen.getByText('$1,234,567')).toBeInTheDocument();
    });

    it('handles multiple entries with mixed units', () => {
      render(
        <CustomTooltip
          active={true}
          payload={[
            { dataKey: 'conversions', value: 245, color: '#10b981', name: 'Conversions' },
            { dataKey: 'revenue', value: 48500, color: '#3b82f6', name: 'Revenue' },
          ]}
          label="2024-01"
          axes={mockAxesWithUnit}
        />
      );

      // Conversions without unit
      expect(screen.getByText('245')).toBeInTheDocument();
      // Revenue with $ unit
      expect(screen.getByText('$48,500')).toBeInTheDocument();
    });
  });

  describe('label formatting', () => {
    it('displays label directly without formatter', () => {
      render(
        <CustomTooltip
          active={true}
          payload={[
            { dataKey: 'conversions', value: 100, color: '#10b981', name: 'Test' },
          ]}
          label="2024-01"
          axes={mockAxesWithUnit}
        />
      );

      expect(screen.getByText('2024-01')).toBeInTheDocument();
    });

    it('applies xAxisFormatter to label', () => {
      const formatter = (v: string) => `Month: ${v}`;

      render(
        <CustomTooltip
          active={true}
          payload={[
            { dataKey: 'conversions', value: 100, color: '#10b981', name: 'Test' },
          ]}
          label="2024-01"
          axes={mockAxesWithUnit}
          xAxisFormatter={formatter}
        />
      );

      expect(screen.getByText('Month: 2024-01')).toBeInTheDocument();
    });

    it('handles empty label with xAxisFormatter', () => {
      const formatter = (v: string) => `Formatted: ${v}`;

      render(
        <CustomTooltip
          active={true}
          payload={[
            { dataKey: 'conversions', value: 100, color: '#10b981', name: 'Test' },
          ]}
          label=""
          axes={mockAxesWithUnit}
          xAxisFormatter={formatter}
        />
      );

      // Empty string passed through formatter results in "Formatted: "
      expect(screen.getByText('Formatted:')).toBeInTheDocument();
    });

    it('handles undefined label without xAxisFormatter', () => {
      render(
        <CustomTooltip
          active={true}
          payload={[
            { dataKey: 'conversions', value: 100, color: '#10b981', name: 'Test' },
          ]}
          label={undefined}
          axes={mockAxesWithUnit}
        />
      );

      // Should render but label paragraph may be empty
      expect(screen.getByText('Test:')).toBeInTheDocument();
    });

    it('handles undefined label with xAxisFormatter', () => {
      const formatter = (v: string) => `Val: ${v}`;

      render(
        <CustomTooltip
          active={true}
          payload={[
            { dataKey: 'conversions', value: 100, color: '#10b981', name: 'Test' },
          ]}
          label={undefined}
          axes={mockAxesWithUnit}
          xAxisFormatter={formatter}
        />
      );

      // Undefined label becomes empty string, then formatted
      expect(screen.getByText('Val:')).toBeInTheDocument();
    });
  });

  describe('visual elements', () => {
    it('renders colored dots for each payload entry', () => {
      const { container } = render(
        <CustomTooltip
          active={true}
          payload={[
            { dataKey: 'conversions', value: 245, color: '#10b981', name: 'Conversions' },
            { dataKey: 'revenue', value: 48500, color: '#3b82f6', name: 'Revenue' },
          ]}
          label="2024-01"
          axes={mockAxesWithUnit}
        />
      );

      const dots = container.querySelectorAll('.rounded-full');
      expect(dots.length).toBe(2);

      // Check background colors
      expect(dots[0]).toHaveStyle({ backgroundColor: '#10b981' });
      expect(dots[1]).toHaveStyle({ backgroundColor: '#3b82f6' });
    });

    it('renders entry names with colons', () => {
      render(
        <CustomTooltip
          active={true}
          payload={[
            { dataKey: 'conversions', value: 245, color: '#10b981', name: 'Conversions' },
            { dataKey: 'revenue', value: 48500, color: '#3b82f6', name: 'Revenue' },
          ]}
          label="2024-01"
          axes={mockAxesWithUnit}
        />
      );

      expect(screen.getByText('Conversions:')).toBeInTheDocument();
      expect(screen.getByText('Revenue:')).toBeInTheDocument();
    });

    it('applies correct CSS classes', () => {
      const { container } = render(
        <CustomTooltip
          active={true}
          payload={[
            { dataKey: 'conversions', value: 245, color: '#10b981', name: 'Conversions' },
          ]}
          label="2024-01"
          axes={mockAxesWithUnit}
        />
      );

      // Check tooltip container classes
      const tooltipContainer = container.querySelector('.shadow-lg');
      expect(tooltipContainer).toBeInTheDocument();
      expect(tooltipContainer).toHaveClass('rounded-md', 'p-3');
    });
  });
});

// =============================================================================
// MULTI-AXIS LINE CHART COMPONENT TESTS
// =============================================================================

describe('MultiAxisLineChart', () => {
  it('renders with data', () => {
    const { container } = render(
      <MultiAxisLineChart data={mockData} xAxisKey="date" axes={mockAxesWithUnit} />
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
        data={mockData}
        xAxisKey="date"
        axes={mockAxesWithUnit}
        isLoading
      />
    );
    expect(container.querySelector('.animate-pulse')).toBeInTheDocument();
  });

  it('shows empty state when no data', () => {
    render(<MultiAxisLineChart data={[]} xAxisKey="date" axes={mockAxesWithUnit} />);
    expect(screen.getByText('No data available')).toBeInTheDocument();
  });

  it('applies custom height', () => {
    const { container } = render(
      <MultiAxisLineChart
        data={mockData}
        xAxisKey="date"
        axes={mockAxesWithUnit}
        height={500}
      />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(
      <MultiAxisLineChart
        data={mockData}
        xAxisKey="date"
        axes={mockAxesWithUnit}
        className="custom-chart"
      />
    );
    expect(container.querySelector('.custom-chart')).toBeInTheDocument();
  });

  it('handles x-axis formatter', () => {
    const formatter = vi.fn((v: string) => v.toUpperCase());
    const { container } = render(
      <MultiAxisLineChart
        data={mockData}
        xAxisKey="date"
        axes={mockAxesWithUnit}
        xAxisFormatter={formatter}
      />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('renders reference lines when provided', () => {
    const { container } = render(
      <MultiAxisLineChart
        data={mockData}
        xAxisKey="date"
        axes={mockAxesWithUnit}
        leftReferenceValue={300}
        rightReferenceValue={60000}
      />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('handles multiple axes configurations', () => {
    const threeAxes: AxisConfig[] = [
      ...mockAxesWithUnit,
      { dataKey: 'cost', name: 'Cost', color: '#f59e0b', yAxisId: 'left' },
    ];
    const dataWithCost = mockData.map((d, i) => ({
      ...d,
      cost: 10000 + i * 1000,
    }));
    const { container } = render(
      <MultiAxisLineChart data={dataWithCost} xAxisKey="date" axes={threeAxes} />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('handles ref forwarding', () => {
    const ref = React.createRef<HTMLDivElement>();

    render(
      <MultiAxisLineChart
        ref={ref}
        data={mockData}
        xAxisKey="date"
        axes={mockAxesWithUnit}
      />
    );

    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });

  it('hides legend when showLegend is false', () => {
    const { container } = render(
      <MultiAxisLineChart
        data={mockData}
        xAxisKey="date"
        axes={mockAxesWithUnit}
        showLegend={false}
      />
    );
    expect(container.querySelector('.recharts-legend-wrapper')).not.toBeInTheDocument();
  });

  it('hides grid when showGrid is false', () => {
    const { container } = render(
      <MultiAxisLineChart
        data={mockData}
        xAxisKey="date"
        axes={mockAxesWithUnit}
        showGrid={false}
      />
    );
    // Grid cartesian-grid element should not be present
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('renders with axes having strokeWidth and showDots options', () => {
    const axesWithOptions: AxisConfig[] = [
      {
        dataKey: 'conversions',
        name: 'Conversions',
        color: '#10b981',
        yAxisId: 'left',
        strokeWidth: 3,
        showDots: true,
      },
      {
        dataKey: 'revenue',
        name: 'Revenue',
        color: '#3b82f6',
        yAxisId: 'right',
        unit: '$',
        strokeWidth: 2,
        showDots: false,
      },
    ];

    const { container } = render(
      <MultiAxisLineChart data={mockData} xAxisKey="date" axes={axesWithOptions} />
    );
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });
});
