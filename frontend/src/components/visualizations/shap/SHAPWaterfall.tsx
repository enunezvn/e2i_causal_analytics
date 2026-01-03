/**
 * SHAP Waterfall Chart Component
 * ==============================
 *
 * Waterfall visualization showing step-by-step feature contributions
 * from base value to final prediction.
 *
 * @module components/visualizations/shap/SHAPWaterfall
 */

import * as React from 'react';
import { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { cn } from '@/lib/utils';
import type { FeatureContribution } from '@/types/explain';

// =============================================================================
// TYPES
// =============================================================================

export interface SHAPWaterfallProps {
  /** Base (expected) prediction value */
  baseValue: number;
  /** Feature contributions with SHAP values */
  features: FeatureContribution[];
  /** Maximum number of features to display (default: 10) */
  maxFeatures?: number;
  /** Chart height in pixels (default: 400) */
  height?: number;
  /** Color for positive contributions */
  positiveColor?: string;
  /** Color for negative contributions */
  negativeColor?: string;
  /** Color for base value bar */
  baseColor?: string;
  /** Color for output value bar */
  outputColor?: string;
  /** Whether data is loading */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Callback when a bar is clicked */
  onBarClick?: (feature: FeatureContribution | null) => void;
  /** Format for displaying values */
  valueFormatter?: (value: number) => string;
}

interface WaterfallDataPoint {
  name: string;
  start: number;
  end: number;
  value: number;
  isBase: boolean;
  isOutput: boolean;
  original?: FeatureContribution;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_FEATURES: FeatureContribution[] = [
  { feature_name: 'days_since_visit', feature_value: 45, shap_value: 0.15, contribution_direction: 'positive', contribution_rank: 1 },
  { feature_name: 'total_prescriptions', feature_value: 12, shap_value: -0.08, contribution_direction: 'negative', contribution_rank: 2 },
  { feature_name: 'territory_sales', feature_value: 150000, shap_value: 0.12, contribution_direction: 'positive', contribution_rank: 3 },
  { feature_name: 'specialty_type', feature_value: 'oncology', shap_value: 0.05, contribution_direction: 'positive', contribution_rank: 4 },
  { feature_name: 'engagement_score', feature_value: 3, shap_value: -0.04, contribution_direction: 'negative', contribution_rank: 5 },
  { feature_name: 'competitor_share', feature_value: 0.35, shap_value: -0.06, contribution_direction: 'negative', contribution_rank: 6 },
  { feature_name: 'formulary_tier', feature_value: 1, shap_value: 0.03, contribution_direction: 'positive', contribution_rank: 7 },
];

const SAMPLE_BASE = 0.45;

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * SHAPWaterfall displays feature contributions as a waterfall chart.
 *
 * @example
 * ```tsx
 * <SHAPWaterfall
 *   baseValue={0.45}
 *   features={explainResponse.top_features}
 *   onBarClick={(f) => f && console.log('Selected:', f.feature_name)}
 * />
 * ```
 */
export const SHAPWaterfall = React.forwardRef<HTMLDivElement, SHAPWaterfallProps>(
  (
    {
      baseValue: propBase,
      features: propFeatures,
      maxFeatures = 10,
      height = 400,
      positiveColor = 'hsl(var(--chart-1))',
      negativeColor = 'hsl(var(--chart-2))',
      baseColor = 'hsl(var(--chart-3))',
      outputColor = 'hsl(var(--chart-4))',
      isLoading = false,
      className,
      onBarClick,
      valueFormatter = (v) => v.toFixed(3),
    },
    ref
  ) => {
    // Use provided data or sample
    const baseValue = propBase ?? SAMPLE_BASE;
    const features = useMemo(
      () => propFeatures ?? SAMPLE_FEATURES,
      [propFeatures]
    );

    // Build waterfall data
    const chartData = useMemo<WaterfallDataPoint[]>(() => {
      const sorted = [...features]
        .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
        .slice(0, maxFeatures);

      const data: WaterfallDataPoint[] = [];
      let cumulative = baseValue;

      // Base value bar
      data.push({
        name: 'Base Value',
        start: 0,
        end: baseValue,
        value: baseValue,
        isBase: true,
        isOutput: false,
      });

      // Feature contribution bars
      sorted.forEach((f) => {
        const start = cumulative;
        const end = cumulative + f.shap_value;
        data.push({
          name: f.feature_name.replace(/_/g, ' '),
          start: Math.min(start, end),
          end: Math.max(start, end),
          value: f.shap_value,
          isBase: false,
          isOutput: false,
          original: f,
        });
        cumulative = end;
      });

      // Output value bar
      data.push({
        name: 'Output',
        start: 0,
        end: cumulative,
        value: cumulative,
        isBase: false,
        isOutput: true,
      });

      return data;
    }, [baseValue, features, maxFeatures]);

    // Calculate domain
    const domain = useMemo(() => {
      const values = chartData.flatMap((d) => [d.start, d.end]);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const padding = (max - min) * 0.1;
      return [min - padding, max + padding];
    }, [chartData]);

    // Get bar color
    const getBarColor = (entry: WaterfallDataPoint): string => {
      if (entry.isBase) return baseColor;
      if (entry.isOutput) return outputColor;
      return entry.value >= 0 ? positiveColor : negativeColor;
    };

    // Custom tooltip
    const CustomTooltip = ({
      active,
      payload,
    }: {
      active?: boolean;
      payload?: Array<{ payload: WaterfallDataPoint }>;
    }) => {
      if (!active || !payload || !payload.length) return null;
      const data = payload[0].payload;

      return (
        <div className="bg-[var(--color-popover)] border border-[var(--color-border)] rounded-md shadow-lg p-3">
          <p className="font-medium text-[var(--color-foreground)]">{data.name}</p>
          {data.isBase && (
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Expected model output
            </p>
          )}
          {data.isOutput && (
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Final prediction
            </p>
          )}
          {data.original && (
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Value: {String(data.original.feature_value)}
            </p>
          )}
          <p
            className={cn(
              'text-sm font-medium',
              data.isBase || data.isOutput
                ? 'text-[var(--color-foreground)]'
                : data.value >= 0
                  ? 'text-emerald-600'
                  : 'text-rose-600'
            )}
          >
            {data.isBase || data.isOutput
              ? valueFormatter(data.value)
              : `${data.value >= 0 ? '+' : ''}${valueFormatter(data.value)}`}
          </p>
        </div>
      );
    };

    // Custom bar shape for waterfall
    const WaterfallBar = (props: {
      x: number;
      y: number;
      width: number;
      height: number;
      payload: WaterfallDataPoint;
    }) => {
      const { x, y, width, height: h, payload } = props;
      const color = getBarColor(payload);

      // Calculate actual bar dimensions for floating bars
      const barHeight = Math.abs(h);
      const barY = payload.isBase || payload.isOutput ? y : y;

      return (
        <g>
          {/* Main bar */}
          <rect
            x={x}
            y={barY}
            width={width}
            height={barHeight}
            fill={color}
            rx={4}
            ry={4}
          />
          {/* Connector line for floating bars */}
          {!payload.isBase && !payload.isOutput && (
            <line
              x1={x - 5}
              y1={barY + (payload.value >= 0 ? barHeight : 0)}
              x2={x}
              y2={barY + (payload.value >= 0 ? barHeight : 0)}
              stroke="var(--color-border)"
              strokeWidth={1}
              strokeDasharray="2,2"
            />
          )}
        </g>
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
    if (features.length === 0) {
      return (
        <div
          ref={ref}
          className={cn(
            'flex items-center justify-center text-[var(--color-muted-foreground)]',
            className
          )}
          style={{ height }}
        >
          No feature data available
        </div>
      );
    }

    return (
      <div ref={ref} className={cn('w-full', className)}>
        <ResponsiveContainer width="100%" height={height}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 50, left: 120, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.3} />
            <XAxis
              type="number"
              domain={domain}
              tickFormatter={(v) => v.toFixed(2)}
              fontSize={12}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={110}
              fontSize={12}
              tickLine={false}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine x={0} stroke="var(--color-border)" strokeWidth={1} />
            <Bar
              dataKey="end"
              shape={(props: unknown) => <WaterfallBar {...(props as Parameters<typeof WaterfallBar>[0])} />}
              cursor={onBarClick ? 'pointer' : 'default'}
              onClick={(data: WaterfallDataPoint) =>
                onBarClick?.(data.original || null)
              }
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getBarColor(entry)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        {/* Legend */}
        <div className="flex justify-center gap-4 mt-2 flex-wrap">
          <div className="flex items-center gap-1 text-xs">
            <div
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: baseColor }}
            />
            <span className="text-[var(--color-muted-foreground)]">Base Value</span>
          </div>
          <div className="flex items-center gap-1 text-xs">
            <div
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: positiveColor }}
            />
            <span className="text-[var(--color-muted-foreground)]">Increases</span>
          </div>
          <div className="flex items-center gap-1 text-xs">
            <div
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: negativeColor }}
            />
            <span className="text-[var(--color-muted-foreground)]">Decreases</span>
          </div>
          <div className="flex items-center gap-1 text-xs">
            <div
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: outputColor }}
            />
            <span className="text-[var(--color-muted-foreground)]">Output</span>
          </div>
        </div>
      </div>
    );
  }
);

SHAPWaterfall.displayName = 'SHAPWaterfall';

export default SHAPWaterfall;
