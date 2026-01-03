/**
 * SHAP Bar Chart Component
 * ========================
 *
 * Horizontal bar chart visualization for SHAP feature importance values.
 * Shows positive contributions in one color and negative in another.
 *
 * @module components/visualizations/shap/SHAPBarChart
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

export interface SHAPBarChartProps {
  /** Feature contributions with SHAP values */
  features: FeatureContribution[];
  /** Maximum number of features to display (default: 10) */
  maxFeatures?: number;
  /** Chart height in pixels (default: 300) */
  height?: number;
  /** Color for positive contributions */
  positiveColor?: string;
  /** Color for negative contributions */
  negativeColor?: string;
  /** Whether to show the zero reference line */
  showReferenceLine?: boolean;
  /** Whether data is loading */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Callback when a bar is clicked */
  onBarClick?: (feature: FeatureContribution) => void;
  /** Whether to show values on bars */
  showValues?: boolean;
  /** Custom tooltip formatter */
  tooltipFormatter?: (value: number, name: string) => string;
}

interface ChartDataPoint {
  name: string;
  value: number;
  originalValue: unknown;
  isPositive: boolean;
  rank: number;
  original: FeatureContribution;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_FEATURES: FeatureContribution[] = [
  { feature_name: 'days_since_visit', feature_value: 45, shap_value: 0.35, contribution_direction: 'positive', contribution_rank: 1 },
  { feature_name: 'total_prescriptions', feature_value: 12, shap_value: -0.28, contribution_direction: 'negative', contribution_rank: 2 },
  { feature_name: 'territory_sales', feature_value: 150000, shap_value: 0.22, contribution_direction: 'positive', contribution_rank: 3 },
  { feature_name: 'specialty_oncology', feature_value: 1, shap_value: 0.18, contribution_direction: 'positive', contribution_rank: 4 },
  { feature_name: 'recent_engagement', feature_value: 3, shap_value: -0.15, contribution_direction: 'negative', contribution_rank: 5 },
  { feature_name: 'competitor_share', feature_value: 0.35, shap_value: -0.12, contribution_direction: 'negative', contribution_rank: 6 },
  { feature_name: 'formulary_status', feature_value: 'preferred', shap_value: 0.10, contribution_direction: 'positive', contribution_rank: 7 },
  { feature_name: 'experience_years', feature_value: 15, shap_value: 0.08, contribution_direction: 'positive', contribution_rank: 8 },
];

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * SHAPBarChart displays feature importance as a horizontal bar chart.
 *
 * @example
 * ```tsx
 * <SHAPBarChart
 *   features={explainResponse.top_features}
 *   maxFeatures={10}
 *   onBarClick={(f) => console.log('Selected:', f.feature_name)}
 * />
 * ```
 */
export const SHAPBarChart = React.forwardRef<HTMLDivElement, SHAPBarChartProps>(
  (
    {
      features: propFeatures,
      maxFeatures = 10,
      height = 300,
      positiveColor = 'hsl(var(--chart-1))',
      negativeColor = 'hsl(var(--chart-2))',
      showReferenceLine = true,
      isLoading = false,
      className,
      onBarClick,
      showValues = false,
      tooltipFormatter,
    },
    ref
  ) => {
    // Use provided features or sample data
    const features = useMemo(
      () => propFeatures ?? SAMPLE_FEATURES,
      [propFeatures]
    );

    // Transform and sort data
    const chartData = useMemo<ChartDataPoint[]>(() => {
      return features
        .slice(0, maxFeatures)
        .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
        .map((f) => ({
          name: f.feature_name.replace(/_/g, ' '),
          value: f.shap_value,
          originalValue: f.feature_value,
          isPositive: f.shap_value >= 0,
          rank: f.contribution_rank,
          original: f,
        }));
    }, [features, maxFeatures]);

    // Calculate domain for symmetric axis
    const maxAbsValue = useMemo(() => {
      const max = Math.max(...chartData.map((d) => Math.abs(d.value)));
      return Math.ceil(max * 10) / 10;
    }, [chartData]);

    // Custom tooltip
    const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: ChartDataPoint }> }) => {
      if (!active || !payload || !payload.length) return null;
      const data = payload[0].payload;

      return (
        <div className="bg-[var(--color-popover)] border border-[var(--color-border)] rounded-md shadow-lg p-3">
          <p className="font-medium text-[var(--color-foreground)]">{data.name}</p>
          <p className="text-sm text-[var(--color-muted-foreground)]">
            Value: {String(data.originalValue)}
          </p>
          <p className={cn(
            'text-sm font-medium',
            data.isPositive ? 'text-emerald-600' : 'text-rose-600'
          )}>
            SHAP: {tooltipFormatter ? tooltipFormatter(data.value, data.name) : data.value.toFixed(4)}
          </p>
          <p className="text-xs text-[var(--color-muted-foreground)]">
            Rank: #{data.rank}
          </p>
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
    if (chartData.length === 0) {
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
            margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.3} />
            <XAxis
              type="number"
              domain={[-maxAbsValue, maxAbsValue]}
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
            {showReferenceLine && (
              <ReferenceLine x={0} stroke="var(--color-border)" strokeWidth={1} />
            )}
            <Bar
              dataKey="value"
              radius={[0, 4, 4, 0]}
              cursor={onBarClick ? 'pointer' : 'default'}
              onClick={(data) => onBarClick?.(data.original)}
              label={showValues ? {
                position: 'right',
                formatter: (v: number) => v.toFixed(3),
                fontSize: 10,
              } : undefined}
            >
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.isPositive ? positiveColor : negativeColor}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  }
);

SHAPBarChart.displayName = 'SHAPBarChart';

export default SHAPBarChart;
