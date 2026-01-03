/**
 * SHAP Beeswarm Plot Component
 * ============================
 *
 * Beeswarm visualization showing SHAP value distributions across features.
 * Each point represents an instance, colored by feature value, positioned by SHAP value.
 *
 * @module components/visualizations/shap/SHAPBeeswarm
 */

import * as React from 'react';
import { useMemo, useCallback } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
  ZAxis,
} from 'recharts';
import { cn } from '@/lib/utils';

// =============================================================================
// TYPES
// =============================================================================

/**
 * Single data point for beeswarm plot
 */
export interface BeeswarmDataPoint {
  /** Feature name */
  feature: string;
  /** SHAP value for this instance */
  shapValue: number;
  /** Normalized feature value [0-1] for coloring */
  featureValue: number;
  /** Original feature value before normalization */
  originalValue: unknown;
  /** Optional instance ID */
  instanceId?: string;
}

export interface SHAPBeeswarmProps {
  /** Data points for the beeswarm plot */
  data: BeeswarmDataPoint[];
  /** Features to display (in order) */
  features?: string[];
  /** Maximum number of features to show */
  maxFeatures?: number;
  /** Chart height in pixels (default: 400) */
  height?: number;
  /** Low value color (start of gradient) */
  lowColor?: string;
  /** High value color (end of gradient) */
  highColor?: string;
  /** Point size (default: 5) */
  pointSize?: number;
  /** Whether data is loading */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Callback when a point is clicked */
  onPointClick?: (point: BeeswarmDataPoint) => void;
  /** Whether to show the zero reference line */
  showReferenceLine?: boolean;
  /** Whether to show the color legend */
  showLegend?: boolean;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_FEATURES = ['days_since_visit', 'total_prescriptions', 'territory_sales', 'specialty_type', 'engagement_score'];

function generateSampleData(): BeeswarmDataPoint[] {
  const data: BeeswarmDataPoint[] = [];
  const random = (min: number, max: number) => Math.random() * (max - min) + min;

  SAMPLE_FEATURES.forEach((feature) => {
    // Generate 30 sample points per feature
    for (let i = 0; i < 30; i++) {
      const featureValue = Math.random();
      // SHAP values tend to correlate with feature values
      const baseShap = (featureValue - 0.5) * 0.6;
      const noise = random(-0.1, 0.1);
      data.push({
        feature,
        shapValue: baseShap + noise,
        featureValue,
        originalValue: Math.round(featureValue * 100),
        instanceId: `instance_${i}`,
      });
    }
  });

  return data;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Interpolate between two colors based on value [0-1]
 */
function interpolateColor(lowColor: string, highColor: string, value: number): string {
  // Parse hex colors
  const parseHex = (hex: string) => {
    const clean = hex.replace('#', '');
    return {
      r: parseInt(clean.slice(0, 2), 16),
      g: parseInt(clean.slice(2, 4), 16),
      b: parseInt(clean.slice(4, 6), 16),
    };
  };

  const low = parseHex(lowColor);
  const high = parseHex(highColor);
  const t = Math.max(0, Math.min(1, value));

  const r = Math.round(low.r + (high.r - low.r) * t);
  const g = Math.round(low.g + (high.g - low.g) * t);
  const b = Math.round(low.b + (high.b - low.b) * t);

  return `rgb(${r}, ${g}, ${b})`;
}

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * SHAPBeeswarm displays SHAP value distributions as a beeswarm plot.
 *
 * @example
 * ```tsx
 * <SHAPBeeswarm
 *   data={beeswarmData}
 *   features={['feature_a', 'feature_b']}
 *   onPointClick={(point) => console.log('Selected:', point)}
 * />
 * ```
 */
export const SHAPBeeswarm = React.forwardRef<HTMLDivElement, SHAPBeeswarmProps>(
  (
    {
      data: propData,
      features: propFeatures,
      maxFeatures = 10,
      height = 400,
      lowColor = '#3b82f6', // Blue
      highColor = '#ef4444', // Red
      pointSize = 5,
      isLoading = false,
      className,
      onPointClick,
      showReferenceLine = true,
      showLegend = true,
    },
    ref
  ) => {
    // Use provided data or generate sample
    const rawData = useMemo(
      () => propData ?? generateSampleData(),
      [propData]
    );

    // Get unique features ordered by mean absolute SHAP
    const orderedFeatures = useMemo(() => {
      if (propFeatures) return propFeatures.slice(0, maxFeatures);

      const featureImportance = new Map<string, number>();
      rawData.forEach((d) => {
        const current = featureImportance.get(d.feature) || 0;
        featureImportance.set(d.feature, current + Math.abs(d.shapValue));
      });

      return Array.from(featureImportance.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, maxFeatures)
        .map(([feature]) => feature);
    }, [rawData, propFeatures, maxFeatures]);

    // Transform data for scatter chart - assign Y values based on feature index
    const chartData = useMemo(() => {
      const featureIndex = new Map(orderedFeatures.map((f, i) => [f, i]));

      return rawData
        .filter((d) => featureIndex.has(d.feature))
        .map((d) => ({
          ...d,
          x: d.shapValue,
          y: featureIndex.get(d.feature)! + (Math.random() - 0.5) * 0.4, // Add jitter
          color: interpolateColor(lowColor, highColor, d.featureValue),
        }));
    }, [rawData, orderedFeatures, lowColor, highColor]);

    // Calculate X axis domain
    const xDomain = useMemo(() => {
      const max = Math.max(...chartData.map((d) => Math.abs(d.x)));
      const padding = max * 0.1;
      return [-(max + padding), max + padding];
    }, [chartData]);

    // Custom tooltip
    const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: typeof chartData[0] }> }) => {
      if (!active || !payload || !payload.length) return null;
      const data = payload[0].payload;

      return (
        <div className="bg-[var(--color-popover)] border border-[var(--color-border)] rounded-md shadow-lg p-3">
          <p className="font-medium text-[var(--color-foreground)]">
            {data.feature.replace(/_/g, ' ')}
          </p>
          <p className="text-sm text-[var(--color-muted-foreground)]">
            Feature Value: {String(data.originalValue)}
          </p>
          <p className={cn(
            'text-sm font-medium',
            data.shapValue >= 0 ? 'text-emerald-600' : 'text-rose-600'
          )}>
            SHAP: {data.shapValue.toFixed(4)}
          </p>
          {data.instanceId && (
            <p className="text-xs text-[var(--color-muted-foreground)]">
              {data.instanceId}
            </p>
          )}
        </div>
      );
    };

    // Handle point click
    const handleClick = useCallback(
      (data: typeof chartData[0]) => {
        if (onPointClick) {
          onPointClick({
            feature: data.feature,
            shapValue: data.shapValue,
            featureValue: data.featureValue,
            originalValue: data.originalValue,
            instanceId: data.instanceId,
          });
        }
      },
      [onPointClick]
    );

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
          No data available for beeswarm plot
        </div>
      );
    }

    return (
      <div ref={ref} className={cn('w-full', className)}>
        <ResponsiveContainer width="100%" height={height}>
          <ScatterChart margin={{ top: 20, right: 30, left: 120, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis
              type="number"
              dataKey="x"
              domain={xDomain}
              tickFormatter={(v) => v.toFixed(2)}
              fontSize={12}
              label={{ value: 'SHAP Value', position: 'bottom', offset: 0 }}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[-0.5, orderedFeatures.length - 0.5]}
              ticks={orderedFeatures.map((_, i) => i)}
              tickFormatter={(v) => orderedFeatures[Math.round(v)]?.replace(/_/g, ' ') || ''}
              fontSize={12}
              width={110}
              tickLine={false}
            />
            <ZAxis range={[pointSize * 10, pointSize * 10]} />
            <Tooltip content={<CustomTooltip />} />
            {showReferenceLine && (
              <ReferenceLine x={0} stroke="var(--color-border)" strokeWidth={1} />
            )}
            <Scatter
              data={chartData}
              cursor={onPointClick ? 'pointer' : 'default'}
              onClick={(_, __, e) => {
                const point = e as unknown as { payload: typeof chartData[0] };
                if (point?.payload) handleClick(point.payload);
              }}
            >
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.color}
                  fillOpacity={0.7}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>

        {/* Color Legend */}
        {showLegend && (
          <div className="flex items-center justify-center gap-4 mt-4">
            <span className="text-xs text-[var(--color-muted-foreground)]">Low</span>
            <div
              className="w-32 h-3 rounded"
              style={{
                background: `linear-gradient(to right, ${lowColor}, ${highColor})`,
              }}
            />
            <span className="text-xs text-[var(--color-muted-foreground)]">High</span>
            <span className="text-xs text-[var(--color-muted-foreground)] ml-2">
              Feature Value
            </span>
          </div>
        )}
      </div>
    );
  }
);

SHAPBeeswarm.displayName = 'SHAPBeeswarm';

export default SHAPBeeswarm;
