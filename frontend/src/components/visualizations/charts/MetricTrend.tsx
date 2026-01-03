/**
 * Metric Trend Component
 * ======================
 *
 * Time series visualization for tracking metrics over time with trend indicators.
 * Supports thresholds, annotations, and change detection.
 *
 * @module components/visualizations/charts/MetricTrend
 */

import * as React from 'react';
import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
} from 'recharts';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { cn } from '@/lib/utils';

// =============================================================================
// TYPES
// =============================================================================

export interface MetricDataPoint {
  /** Timestamp or date string */
  timestamp: string;
  /** Metric value */
  value: number;
  /** Optional annotation */
  annotation?: string;
}

export interface MetricThreshold {
  /** Threshold value */
  value: number;
  /** Label for the threshold */
  label: string;
  /** Color for threshold line */
  color?: string;
  /** Whether this is an upper or lower bound */
  type: 'upper' | 'lower' | 'target';
}

export interface MetricTrendProps {
  /** Metric name */
  name: string;
  /** Metric data points */
  data: MetricDataPoint[];
  /** Optional unit for the metric */
  unit?: string;
  /** Optional thresholds */
  thresholds?: MetricThreshold[];
  /** Chart height in pixels (default: 250) */
  height?: number;
  /** Line color */
  color?: string;
  /** Whether to show the trend summary header */
  showHeader?: boolean;
  /** Whether to show the sparkline version (compact) */
  compact?: boolean;
  /** Whether data is loading */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Format for timestamp display */
  timestampFormatter?: (value: string) => string;
  /** Format for value display */
  valueFormatter?: (value: number) => string;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_DATA: MetricDataPoint[] = [
  { timestamp: '2024-01-01', value: 0.85 },
  { timestamp: '2024-01-08', value: 0.87 },
  { timestamp: '2024-01-15', value: 0.84 },
  { timestamp: '2024-01-22', value: 0.89 },
  { timestamp: '2024-01-29', value: 0.91 },
  { timestamp: '2024-02-05', value: 0.88 },
  { timestamp: '2024-02-12', value: 0.92 },
  { timestamp: '2024-02-19', value: 0.90, annotation: 'Model retrained' },
  { timestamp: '2024-02-26', value: 0.93 },
  { timestamp: '2024-03-04', value: 0.94 },
];

const SAMPLE_THRESHOLDS: MetricThreshold[] = [
  { value: 0.90, label: 'Target', type: 'target', color: '#22c55e' },
  { value: 0.80, label: 'Minimum', type: 'lower', color: '#ef4444' },
];

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

interface TrendAnalysis {
  direction: 'up' | 'down' | 'stable';
  changePercent: number;
  currentValue: number;
  previousValue: number;
  isImproving: boolean;
}

function analyzeTrend(data: MetricDataPoint[], thresholds?: MetricThreshold[]): TrendAnalysis {
  if (data.length < 2) {
    return {
      direction: 'stable',
      changePercent: 0,
      currentValue: data[0]?.value ?? 0,
      previousValue: data[0]?.value ?? 0,
      isImproving: true,
    };
  }

  const current = data[data.length - 1].value;
  const previous = data[data.length - 2].value;
  const change = current - previous;
  const changePercent = previous !== 0 ? (change / previous) * 100 : 0;

  const direction: 'up' | 'down' | 'stable' =
    Math.abs(changePercent) < 1 ? 'stable' : change > 0 ? 'up' : 'down';

  // Determine if direction is an improvement (higher is usually better)
  // Unless there's a target threshold that we're above
  const target = thresholds?.find((t) => t.type === 'target');
  let isImproving = change >= 0;
  if (target && current > target.value) {
    // Already above target, staying stable is fine
    isImproving = change >= 0;
  }

  return {
    direction,
    changePercent,
    currentValue: current,
    previousValue: previous,
    isImproving,
  };
}

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * MetricTrend displays a metric over time with trend indicators.
 *
 * @example
 * ```tsx
 * <MetricTrend
 *   name="Model Accuracy"
 *   data={accuracyHistory}
 *   unit="%"
 *   thresholds={[
 *     { value: 0.90, label: 'Target', type: 'target' },
 *   ]}
 * />
 * ```
 */
export const MetricTrend = React.forwardRef<HTMLDivElement, MetricTrendProps>(
  (
    {
      name: propName,
      data: propData,
      unit = '',
      thresholds: propThresholds,
      height = 250,
      color = 'hsl(var(--chart-1))',
      showHeader = true,
      compact = false,
      isLoading = false,
      className,
      timestampFormatter,
      valueFormatter = (v) => v.toFixed(2),
    },
    ref
  ) => {
    // Use provided data or sample
    const name = propName ?? 'Metric';
    const data = propData ?? SAMPLE_DATA;
    const thresholds = propThresholds ?? SAMPLE_THRESHOLDS;

    // Analyze trend
    const trend = useMemo(() => analyzeTrend(data, thresholds), [data, thresholds]);

    // Calculate Y-axis domain
    const domain = useMemo(() => {
      const values = data.map((d) => d.value);
      const thresholdValues = thresholds.map((t) => t.value);
      const allValues = [...values, ...thresholdValues];

      const min = Math.min(...allValues);
      const max = Math.max(...allValues);
      const padding = (max - min) * 0.1;

      return [Math.max(0, min - padding), max + padding];
    }, [data, thresholds]);

    // Find reference areas (between thresholds)
    const referenceAreas = useMemo(() => {
      const areas: { y1: number; y2: number; fill: string }[] = [];
      const lower = thresholds.find((t) => t.type === 'lower');
      const upper = thresholds.find((t) => t.type === 'upper');

      if (lower && upper) {
        // Add danger zone below lower
        areas.push({
          y1: domain[0],
          y2: lower.value,
          fill: 'rgba(239, 68, 68, 0.1)',
        });
        // Add success zone between target and upper (or above target)
        areas.push({
          y1: lower.value,
          y2: upper.value,
          fill: 'rgba(34, 197, 94, 0.1)',
        });
      }

      return areas;
    }, [thresholds, domain]);

    // Custom tooltip
    const CustomTooltip = ({
      active,
      payload,
      label,
    }: {
      active?: boolean;
      payload?: Array<{ value: number; payload: MetricDataPoint }>;
      label?: string;
    }) => {
      if (!active || !payload || !payload.length) return null;
      const point = payload[0].payload;

      return (
        <div className="bg-[var(--color-popover)] border border-[var(--color-border)] rounded-md shadow-lg p-3">
          <p className="text-sm text-[var(--color-muted-foreground)]">
            {timestampFormatter ? timestampFormatter(label || '') : label}
          </p>
          <p className="text-lg font-medium text-[var(--color-foreground)]">
            {valueFormatter(point.value)}{unit}
          </p>
          {point.annotation && (
            <p className="text-xs text-[var(--color-muted-foreground)] mt-1 italic">
              {point.annotation}
            </p>
          )}
        </div>
      );
    };

    // Get trend icon
    const TrendIcon = trend.direction === 'up' ? TrendingUp : trend.direction === 'down' ? TrendingDown : Minus;
    const trendColor = trend.isImproving ? 'text-emerald-500' : trend.direction === 'stable' ? 'text-gray-500' : 'text-rose-500';

    // Loading skeleton
    if (isLoading) {
      return (
        <div
          ref={ref}
          className={cn('animate-pulse', className)}
          style={{ height: compact ? 60 : height + (showHeader ? 50 : 0) }}
        >
          <div className="h-full bg-[var(--color-muted)] rounded-md" />
        </div>
      );
    }

    // Compact sparkline version
    if (compact) {
      return (
        <div ref={ref} className={cn('flex items-center gap-3', className)}>
          <div className="flex-1">
            <ResponsiveContainer width="100%" height={40}>
              <LineChart data={data} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke={color}
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-sm font-medium">{valueFormatter(trend.currentValue)}{unit}</span>
            <TrendIcon className={cn('h-4 w-4', trendColor)} />
          </div>
        </div>
      );
    }

    return (
      <div ref={ref} className={cn('w-full', className)}>
        {/* Header with current value and trend */}
        {showHeader && (
          <div className="flex items-center justify-between mb-4">
            <div>
              <h4 className="text-sm font-medium text-[var(--color-muted-foreground)]">
                {name}
              </h4>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold text-[var(--color-foreground)]">
                  {valueFormatter(trend.currentValue)}{unit}
                </span>
                <div className={cn('flex items-center gap-1 text-sm', trendColor)}>
                  <TrendIcon className="h-4 w-4" />
                  <span>
                    {trend.changePercent > 0 ? '+' : ''}
                    {trend.changePercent.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Threshold status */}
            {thresholds.length > 0 && (
              <div className="text-right">
                {thresholds.map((threshold) => {
                  const isAbove = trend.currentValue >= threshold.value;
                  const status =
                    threshold.type === 'lower'
                      ? isAbove
                        ? 'Above minimum'
                        : 'Below minimum'
                      : threshold.type === 'upper'
                        ? isAbove
                          ? 'Above maximum'
                          : 'Within range'
                        : isAbove
                          ? 'On target'
                          : 'Below target';

                  const statusColor =
                    threshold.type === 'lower'
                      ? isAbove
                        ? 'text-emerald-500'
                        : 'text-rose-500'
                      : threshold.type === 'upper'
                        ? isAbove
                          ? 'text-rose-500'
                          : 'text-emerald-500'
                        : isAbove
                          ? 'text-emerald-500'
                          : 'text-amber-500';

                  return (
                    <div key={threshold.label} className="text-xs">
                      <span className="text-[var(--color-muted-foreground)]">
                        {threshold.label}:
                      </span>{' '}
                      <span className={statusColor}>{status}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* Chart */}
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis
              dataKey="timestamp"
              tickFormatter={timestampFormatter}
              fontSize={12}
              tickLine={false}
            />
            <YAxis
              domain={domain}
              tickFormatter={valueFormatter}
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Reference areas */}
            {referenceAreas.map((area, i) => (
              <ReferenceArea
                key={i}
                y1={area.y1}
                y2={area.y2}
                fill={area.fill}
              />
            ))}

            {/* Threshold lines */}
            {thresholds.map((threshold) => (
              <ReferenceLine
                key={threshold.label}
                y={threshold.value}
                stroke={threshold.color || 'var(--color-border)'}
                strokeDasharray={threshold.type === 'target' ? '0' : '5 5'}
                label={{
                  value: threshold.label,
                  position: 'right',
                  fontSize: 10,
                  fill: 'var(--color-muted-foreground)',
                }}
              />
            ))}

            {/* Main line */}
            <Line
              type="monotone"
              dataKey="value"
              stroke={color}
              strokeWidth={2}
              dot={(props) => {
                const { payload } = props as { payload: MetricDataPoint };
                if (payload?.annotation) {
                  return (
                    <circle
                      cx={props.cx}
                      cy={props.cy}
                      r={5}
                      fill={color}
                      stroke="white"
                      strokeWidth={2}
                    />
                  );
                }
                return <circle cx={props.cx} cy={props.cy} r={0} />;
              }}
              activeDot={{ r: 6, strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  }
);

MetricTrend.displayName = 'MetricTrend';

export default MetricTrend;
