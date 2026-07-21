/**
 * Metric Trend Component
 * ======================
 *
 * Time series visualization for tracking metrics over time with trend indicators.
 * Supports thresholds, annotations, and change detection.
 *
 * Claims-lag support (backlog #45): points flagged `provisional` render as a
 * dashed tail with hollow markers (their claims are still arriving), and the
 * opt-in `showNowcast` prop overlays the grossed-up nowcast estimate with its
 * CI band. All of it is ADDITIVE — consumers passing only the legacy props
 * render exactly as before.
 *
 * @module components/visualizations/charts/MetricTrend
 */

import * as React from 'react';
import { useMemo } from 'react';
import {
  LineChart,
  ComposedChart,
  Line,
  Area,
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
  /**
   * Claims-lag: true while the period's claims are still maturing — the point
   * renders as part of the dashed provisional tail with a hollow marker.
   */
  provisional?: boolean;
  /** Estimated share of the period's claims arrived so far (null = unknown). */
  completionFactor?: number | null;
  /** Grossed-up nowcast estimate for a provisional period (null = none). */
  nowcastValue?: number | null;
  /** Nowcast CI lower bound (null = none). */
  nowcastCiLower?: number | null;
  /** Nowcast CI upper bound (null = none). */
  nowcastCiUpper?: number | null;
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
  /**
   * Opt-in claims-lag nowcast overlay: a faint line at each provisional
   * point's `nowcastValue` plus its CI band. Default OFF — the default view
   * stays the honest mature series. No-op when no point is provisional.
   */
  showNowcast?: boolean;
}

// NOTE: SAMPLE_DATA / SAMPLE_THRESHOLDS (a fabricated improving accuracy
// trend with a 'Model retrained' annotation) were DELETED. Omitted props
// render the empty state.

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

/**
 * Tooltip disclosure line for a provisional (claims-still-maturing) point.
 *
 * Returns null for mature points. Never fabricates: the completion share and
 * the nowcast estimate each appear ONLY when the estimator produced them —
 * a month younger than the observed lag support gets the bare disclosure.
 */
export function provisionalTooltipText(
  point: Pick<MetricDataPoint, 'provisional' | 'completionFactor' | 'nowcastValue'>,
  valueFormatter: (v: number) => string = (v) => v.toFixed(2),
  unit = ''
): string | null {
  if (!point.provisional) return null;
  const maturing =
    point.completionFactor != null
      ? `Provisional — claims still maturing (~${Math.round(point.completionFactor * 100)}% of this month's claims have arrived).`
      : 'Provisional — claims still maturing.';
  const nowcast =
    point.nowcastValue != null
      ? ` Nowcast estimate: ${valueFormatter(point.nowcastValue)}${unit}.`
      : '';
  return `${maturing}${nowcast}`;
}

/** Chart row for the provisional split: the mature segment and the dashed
 *  tail are separate recharts Lines fed by these derived keys. */
interface ProvisionalChartRow extends MetricDataPoint {
  matureValue: number | null;
  provisionalValue: number | null;
  nowcastBand: [number, number] | null;
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
      showNowcast = false,
    },
    ref
  ) => {
    // Data comes ONLY from props — never a sample fallback.
    const name = propName ?? 'Metric';
    const data = useMemo(() => propData ?? [], [propData]);
    const thresholds = useMemo(() => propThresholds ?? [], [propThresholds]);

    // Claims-lag: any provisional point switches the chart into the
    // mature-segment + dashed-tail split. Legacy data never sets the flag,
    // so legacy consumers keep the exact single-Line render below.
    const hasProvisional = useMemo(() => data.some((p) => p.provisional === true), [data]);
    const nowcastActive = showNowcast && hasProvisional;

    const chartData: MetricDataPoint[] | ProvisionalChartRow[] = useMemo(() => {
      if (!hasProvisional) return data;
      return data.map((p, i) => ({
        ...p,
        matureValue: p.provisional ? null : p.value,
        // The last mature point anchors the dashed tail so the line connects.
        provisionalValue:
          p.provisional || data[i + 1]?.provisional === true ? p.value : null,
        nowcastBand:
          p.provisional && p.nowcastCiLower != null && p.nowcastCiUpper != null
            ? ([p.nowcastCiLower, p.nowcastCiUpper] as [number, number])
            : null,
      }));
    }, [data, hasProvisional]);

    // Analyze trend
    const trend = useMemo(() => analyzeTrend(data, thresholds), [data, thresholds]);

    // Calculate Y-axis domain
    const domain = useMemo(() => {
      const values = data.map((d) => d.value);
      const thresholdValues = thresholds.map((t) => t.value);
      // With the overlay on, the CI band and nowcast line must not clip.
      const nowcastValues = nowcastActive
        ? data.flatMap((d) =>
            d.provisional
              ? [d.nowcastValue, d.nowcastCiLower, d.nowcastCiUpper].filter(
                  (v): v is number => v != null
                )
              : []
          )
        : [];
      const allValues = [...values, ...thresholdValues, ...nowcastValues];

      const min = Math.min(...allValues);
      const max = Math.max(...allValues);
      const padding = (max - min) * 0.1;

      return [Math.max(0, min - padding), max + padding];
    }, [data, thresholds, nowcastActive]);

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
      // Claims-lag disclosure for provisional points (null for mature ones).
      const provisionalNote = provisionalTooltipText(point, valueFormatter, unit);

      return (
        <div className="bg-[var(--color-popover)] border border-[var(--color-border)] rounded-md shadow-lg p-3">
          <p className="text-sm text-[var(--color-muted-foreground)]">
            {timestampFormatter ? timestampFormatter(label || '') : label}
          </p>
          <p className="text-lg font-medium text-[var(--color-foreground)]">
            {valueFormatter(point.value)}{unit}
          </p>
          {provisionalNote && (
            <p className="text-xs text-amber-600 dark:text-amber-400 mt-1 max-w-[240px]">
              {provisionalNote}
            </p>
          )}
          {point.annotation && (
            <p className="text-xs text-[var(--color-muted-foreground)] mt-1 italic">
              {point.annotation}
            </p>
          )}
        </div>
      );
    };

    // ComposedChart only when the provisional split needs its Area band —
    // the legacy path must keep the exact LineChart element tree.
    const ChartComponent = hasProvisional ? ComposedChart : LineChart;

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

    // Honest empty state — never a fabricated trend.
    if (data.length === 0) {
      return (
        <div
          ref={ref}
          className={cn(
            'flex items-center justify-center text-sm text-[var(--color-muted-foreground)]',
            className
          )}
          style={{ height: compact ? 60 : height }}
        >
          No metric data available
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

        {/* Chart. ComposedChart only for the provisional split (the CI band
            is an Area); the legacy path keeps the exact LineChart render. */}
        <ResponsiveContainer width="100%" height={height}>
          <ChartComponent data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
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

            {/* Nowcast CI band (under the lines) — opt-in, provisional tail only. */}
            {nowcastActive && (
              <Area
                dataKey="nowcastBand"
                stroke="none"
                fill={color}
                fillOpacity={0.12}
                activeDot={false}
                connectNulls={false}
              />
            )}

            {/* Main line (legacy path: no provisional points). */}
            {!hasProvisional && (
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
            )}

            {/* Mature segment (solid) — the values are the honest series. */}
            {hasProvisional && (
              <Line
                type="monotone"
                dataKey="matureValue"
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
            )}

            {/* Provisional tail (dashed, hollow markers): claims for these
                periods are still arriving — same values, honest styling. */}
            {hasProvisional && (
              <Line
                className="metric-trend-provisional"
                type="monotone"
                dataKey="provisionalValue"
                stroke={color}
                strokeWidth={2}
                strokeDasharray="6 4"
                dot={(props) => {
                  const { payload } = props as { payload: MetricDataPoint };
                  if (payload?.provisional) {
                    return (
                      <circle
                        cx={props.cx}
                        cy={props.cy}
                        r={3.5}
                        fill="var(--color-background)"
                        stroke={color}
                        strokeWidth={1.5}
                      />
                    );
                  }
                  // The mature anchor point of the tail carries no marker.
                  return <circle cx={props.cx} cy={props.cy} r={0} />;
                }}
                activeDot={{ r: 6, strokeWidth: 2 }}
              />
            )}

            {/* Nowcast estimate (faint) — opt-in, provisional tail only. */}
            {nowcastActive && (
              <Line
                className="metric-trend-nowcast"
                type="monotone"
                dataKey="nowcastValue"
                stroke={color}
                strokeOpacity={0.55}
                strokeWidth={1.5}
                strokeDasharray="2 3"
                dot={false}
                connectNulls={false}
                activeDot={{ r: 4, strokeWidth: 1 }}
              />
            )}
          </ChartComponent>
        </ResponsiveContainer>
      </div>
    );
  }
);

MetricTrend.displayName = 'MetricTrend';

export default MetricTrend;
