/**
 * Multi-Axis Line Chart Component
 * ================================
 *
 * Line chart with multiple Y-axes for comparing metrics with different scales.
 * Supports time series data with synchronized tooltips.
 *
 * @module components/visualizations/charts/MultiAxisLineChart
 */

import * as React from 'react';
import { useMemo } from 'react';
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { cn } from '@/lib/utils';

// =============================================================================
// TYPES
// =============================================================================

export interface AxisConfig {
  /** Data key for this axis */
  dataKey: string;
  /** Display name for legend */
  name: string;
  /** Line color */
  color: string;
  /** Which Y-axis to use (left or right) */
  yAxisId: 'left' | 'right';
  /** Line stroke width */
  strokeWidth?: number;
  /** Whether to show dots on line */
  showDots?: boolean;
  /** Optional unit for formatting */
  unit?: string;
}

export interface MultiAxisLineChartProps {
  /** Chart data array */
  data: Record<string, unknown>[];
  /** X-axis data key (usually date/time) */
  xAxisKey: string;
  /** Configuration for each line/axis */
  axes: AxisConfig[];
  /** Chart height in pixels (default: 400) */
  height?: number;
  /** Whether data is loading */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Format X-axis labels */
  xAxisFormatter?: (value: string) => string;
  /** Reference line value for left Y-axis */
  leftReferenceValue?: number;
  /** Reference line value for right Y-axis */
  rightReferenceValue?: number;
  /** Whether to show the legend */
  showLegend?: boolean;
  /** Whether to show the grid */
  showGrid?: boolean;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_DATA = [
  { date: '2024-01', conversions: 245, revenue: 48500, cost: 12000 },
  { date: '2024-02', conversions: 312, revenue: 62400, cost: 14500 },
  { date: '2024-03', conversions: 287, revenue: 57300, cost: 13200 },
  { date: '2024-04', conversions: 356, revenue: 71200, cost: 15800 },
  { date: '2024-05', conversions: 398, revenue: 79600, cost: 17200 },
  { date: '2024-06', conversions: 425, revenue: 85000, cost: 18500 },
];

const SAMPLE_AXES: AxisConfig[] = [
  { dataKey: 'conversions', name: 'Conversions', color: 'hsl(var(--chart-1))', yAxisId: 'left' },
  { dataKey: 'revenue', name: 'Revenue ($)', color: 'hsl(var(--chart-2))', yAxisId: 'right', unit: '$' },
];

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * MultiAxisLineChart displays multiple metrics with different scales on a single chart.
 *
 * @example
 * ```tsx
 * <MultiAxisLineChart
 *   data={timeSeriesData}
 *   xAxisKey="date"
 *   axes={[
 *     { dataKey: 'trx', name: 'TRx', color: '#10b981', yAxisId: 'left' },
 *     { dataKey: 'marketShare', name: 'Market Share', color: '#3b82f6', yAxisId: 'right', unit: '%' },
 *   ]}
 * />
 * ```
 */
export const MultiAxisLineChart = React.forwardRef<HTMLDivElement, MultiAxisLineChartProps>(
  (
    {
      data: propData,
      xAxisKey = 'date',
      axes: propAxes,
      height = 400,
      isLoading = false,
      className,
      xAxisFormatter,
      leftReferenceValue,
      rightReferenceValue,
      showLegend = true,
      showGrid = true,
    },
    ref
  ) => {
    // Use provided data or sample
    const data = propData ?? SAMPLE_DATA;
    const axes = propAxes ?? SAMPLE_AXES;

    // Calculate domains for left and right axes
    const { leftDomain, rightDomain } = useMemo(() => {
      const leftKeys = axes.filter((a) => a.yAxisId === 'left').map((a) => a.dataKey);
      const rightKeys = axes.filter((a) => a.yAxisId === 'right').map((a) => a.dataKey);

      const getMinMax = (keys: string[]) => {
        if (keys.length === 0) return [0, 100];
        const values = data.flatMap((d) =>
          keys.map((k) => (typeof d[k] === 'number' ? d[k] as number : 0))
        );
        const min = Math.min(...values);
        const max = Math.max(...values);
        const padding = (max - min) * 0.1;
        return [Math.floor(min - padding), Math.ceil(max + padding)];
      };

      return {
        leftDomain: getMinMax(leftKeys) as [number, number],
        rightDomain: getMinMax(rightKeys) as [number, number],
      };
    }, [data, axes]);

    // Custom tooltip
    const CustomTooltip = ({
      active,
      payload,
      label,
    }: {
      active?: boolean;
      payload?: Array<{ dataKey: string; value: number; color: string; name: string }>;
      label?: string;
    }) => {
      if (!active || !payload || !payload.length) return null;

      return (
        <div className="bg-[var(--color-popover)] border border-[var(--color-border)] rounded-md shadow-lg p-3">
          <p className="font-medium text-[var(--color-foreground)] mb-2">
            {xAxisFormatter ? xAxisFormatter(label || '') : label}
          </p>
          {payload.map((entry, index) => {
            const axisConfig = axes.find((a) => a.dataKey === entry.dataKey);
            const value = axisConfig?.unit
              ? `${axisConfig.unit}${entry.value.toLocaleString()}`
              : entry.value.toLocaleString();
            return (
              <div key={index} className="flex items-center gap-2 text-sm">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: entry.color }}
                />
                <span className="text-[var(--color-muted-foreground)]">{entry.name}:</span>
                <span className="font-medium">{value}</span>
              </div>
            );
          })}
        </div>
      );
    };

    // Loading skeleton
    if (isLoading) {
      return (
        <div
          ref={ref}
          className={cn('animate-pulse', className)}
          style={{ height }}
        >
          <div className="h-full bg-[var(--color-muted)] rounded-md" />
        </div>
      );
    }

    // Empty state
    if (data.length === 0) {
      return (
        <div
          ref={ref}
          className={cn(
            'flex items-center justify-center text-[var(--color-muted-foreground)]',
            className
          )}
          style={{ height }}
        >
          No data available
        </div>
      );
    }

    return (
      <div ref={ref} className={cn('w-full', className)}>
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            {showGrid && (
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            )}
            <XAxis
              dataKey={xAxisKey}
              tickFormatter={xAxisFormatter}
              fontSize={12}
              tickLine={false}
            />
            <YAxis
              yAxisId="left"
              domain={leftDomain}
              orientation="left"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              yAxisId="right"
              domain={rightDomain}
              orientation="right"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip content={<CustomTooltip />} />
            {showLegend && (
              <Legend
                wrapperStyle={{ paddingTop: 10 }}
                iconType="line"
              />
            )}
            {leftReferenceValue !== undefined && (
              <ReferenceLine
                y={leftReferenceValue}
                yAxisId="left"
                stroke="var(--color-border)"
                strokeDasharray="3 3"
              />
            )}
            {rightReferenceValue !== undefined && (
              <ReferenceLine
                y={rightReferenceValue}
                yAxisId="right"
                stroke="var(--color-border)"
                strokeDasharray="3 3"
              />
            )}
            {axes.map((axis) => (
              <Line
                key={axis.dataKey}
                type="monotone"
                dataKey={axis.dataKey}
                name={axis.name}
                yAxisId={axis.yAxisId}
                stroke={axis.color}
                strokeWidth={axis.strokeWidth ?? 2}
                dot={axis.showDots ?? false}
                activeDot={{ r: 6, strokeWidth: 2 }}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    );
  }
);

MultiAxisLineChart.displayName = 'MultiAxisLineChart';

export default MultiAxisLineChart;
