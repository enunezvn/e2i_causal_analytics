/**
 * ROC Curve Component
 * ===================
 *
 * Receiver Operating Characteristic curve visualization for model evaluation.
 * Shows true positive rate vs false positive rate with AUC calculation.
 *
 * @module components/visualizations/charts/ROCCurve
 */

import * as React from 'react';
import { useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { cn } from '@/lib/utils';

// =============================================================================
// TYPES
// =============================================================================

export interface ROCPoint {
  /** False Positive Rate (1 - Specificity) */
  fpr: number;
  /** True Positive Rate (Sensitivity/Recall) */
  tpr: number;
  /** Optional threshold value */
  threshold?: number;
}

export interface ROCCurveData {
  /** Name/label for the curve */
  name: string;
  /** ROC curve points */
  points: ROCPoint[];
  /** Line color */
  color: string;
  /** Pre-calculated AUC (or will be computed) */
  auc?: number;
}

export interface ROCCurveProps {
  /** ROC curve data (single or multiple models) */
  curves: ROCCurveData[];
  /** Chart height in pixels (default: 400) */
  height?: number;
  /** Whether to show the random classifier line */
  showDiagonal?: boolean;
  /** Whether to show the AUC in legend */
  showAUC?: boolean;
  /** Whether to show area under curve */
  showArea?: boolean;
  /** Whether data is loading */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Callback when a point is hovered */
  onPointHover?: (curve: string, point: ROCPoint | null) => void;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

function generateSampleROC(): ROCPoint[] {
  const points: ROCPoint[] = [{ fpr: 0, tpr: 0, threshold: 1 }];

  // Generate a curve that's better than random
  for (let i = 1; i <= 20; i++) {
    const fpr = i / 20;
    // TPR should be above the diagonal for a good model
    const tpr = Math.min(1, fpr + 0.3 + (Math.random() * 0.2 - 0.1));
    points.push({ fpr, tpr, threshold: 1 - i / 20 });
  }

  points.push({ fpr: 1, tpr: 1, threshold: 0 });
  return points;
}

const SAMPLE_CURVES: ROCCurveData[] = [
  {
    name: 'Churn Model v2',
    points: generateSampleROC(),
    color: 'hsl(var(--chart-1))',
  },
  {
    name: 'Churn Model v1',
    points: generateSampleROC().map((p) => ({
      ...p,
      tpr: Math.max(p.fpr, p.tpr - 0.15),
    })),
    color: 'hsl(var(--chart-2))',
  },
];

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Calculate Area Under Curve using trapezoidal rule
 */
function calculateAUC(points: ROCPoint[]): number {
  const sorted = [...points].sort((a, b) => a.fpr - b.fpr);
  let auc = 0;

  for (let i = 1; i < sorted.length; i++) {
    const width = sorted[i].fpr - sorted[i - 1].fpr;
    const avgHeight = (sorted[i].tpr + sorted[i - 1].tpr) / 2;
    auc += width * avgHeight;
  }

  return auc;
}

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * ROCCurve displays model classification performance across thresholds.
 *
 * @example
 * ```tsx
 * <ROCCurve
 *   curves={[
 *     { name: 'Model A', points: rocPoints, color: '#10b981' },
 *   ]}
 *   showAUC
 *   showArea
 * />
 * ```
 */
export const ROCCurve = React.forwardRef<HTMLDivElement, ROCCurveProps>(
  (
    {
      curves: propCurves,
      height = 400,
      showDiagonal = true,
      showAUC = true,
      showArea = true,
      isLoading = false,
      className,
      onPointHover,
    },
    ref
  ) => {
    // Use provided data or sample
    const curves = propCurves ?? SAMPLE_CURVES;

    // Calculate AUC for each curve
    const curvesWithAUC = useMemo(() => {
      return curves.map((curve) => ({
        ...curve,
        auc: curve.auc ?? calculateAUC(curve.points),
      }));
    }, [curves]);

    // Prepare chart data - merge all curves into single data array
    const chartData = useMemo(() => {
      // Get all unique FPR values
      const allFPRs = new Set<number>();
      curves.forEach((curve) => {
        curve.points.forEach((p) => allFPRs.add(p.fpr));
      });

      // Sort FPR values
      const sortedFPRs = Array.from(allFPRs).sort((a, b) => a - b);

      // Create data points for each FPR
      return sortedFPRs.map((fpr) => {
        const point: Record<string, number> = { fpr };

        curves.forEach((curve) => {
          // Find closest point for this curve
          const exactPoint = curve.points.find((p) => p.fpr === fpr);
          if (exactPoint) {
            point[curve.name] = exactPoint.tpr;
          } else {
            // Interpolate
            const sorted = curve.points.sort((a, b) => a.fpr - b.fpr);
            const lower = sorted.filter((p) => p.fpr < fpr).pop();
            const upper = sorted.find((p) => p.fpr > fpr);

            if (lower && upper) {
              const ratio = (fpr - lower.fpr) / (upper.fpr - lower.fpr);
              point[curve.name] = lower.tpr + ratio * (upper.tpr - lower.tpr);
            } else if (lower) {
              point[curve.name] = lower.tpr;
            } else if (upper) {
              point[curve.name] = upper.tpr;
            }
          }
        });

        return point;
      });
    }, [curves]);

    // Custom tooltip
    const CustomTooltip = ({
      active,
      payload,
      label,
    }: {
      active?: boolean;
      payload?: Array<{ dataKey: string; value: number; color: string }>;
      label?: number;
    }) => {
      if (!active || !payload || !payload.length) return null;

      return (
        <div className="bg-[var(--color-popover)] border border-[var(--color-border)] rounded-md shadow-lg p-3">
          <p className="text-sm text-[var(--color-muted-foreground)] mb-2">
            FPR: {label?.toFixed(3)}
          </p>
          {payload.map((entry, index) => (
            <div key={index} className="flex items-center gap-2 text-sm">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-[var(--color-muted-foreground)]">{entry.dataKey}:</span>
              <span className="font-medium">TPR {entry.value?.toFixed(3)}</span>
            </div>
          ))}
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
    if (curves.length === 0) {
      return (
        <div
          ref={ref}
          className={cn(
            'flex items-center justify-center text-[var(--color-muted-foreground)]',
            className
          )}
          style={{ height }}
        >
          No ROC curve data available
        </div>
      );
    }

    return (
      <div ref={ref} className={cn('w-full', className)}>
        <ResponsiveContainer width="100%" height={height}>
          <AreaChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis
              dataKey="fpr"
              type="number"
              domain={[0, 1]}
              tickFormatter={(v) => v.toFixed(1)}
              fontSize={12}
              label={{ value: 'False Positive Rate', position: 'bottom', offset: 0 }}
            />
            <YAxis
              type="number"
              domain={[0, 1]}
              tickFormatter={(v) => v.toFixed(1)}
              fontSize={12}
              label={{ value: 'True Positive Rate', angle: -90, position: 'left', offset: 10 }}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Random classifier diagonal line */}
            {showDiagonal && (
              <ReferenceLine
                segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                stroke="var(--color-muted-foreground)"
                strokeDasharray="5 5"
                strokeWidth={1}
              />
            )}

            {/* ROC curves */}
            {curvesWithAUC.map((curve) => (
              <Area
                key={curve.name}
                type="monotone"
                dataKey={curve.name}
                stroke={curve.color}
                fill={showArea ? curve.color : 'transparent'}
                fillOpacity={showArea ? 0.1 : 0}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 2 }}
                onMouseEnter={() => onPointHover?.(curve.name, null)}
                onMouseLeave={() => onPointHover?.(curve.name, null)}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>

        {/* Legend with AUC */}
        <div className="flex flex-wrap justify-center gap-4 mt-4">
          {curvesWithAUC.map((curve) => (
            <div key={curve.name} className="flex items-center gap-2">
              <div
                className="w-4 h-0.5"
                style={{ backgroundColor: curve.color }}
              />
              <span className="text-sm text-[var(--color-foreground)]">
                {curve.name}
                {showAUC && (
                  <span className="text-[var(--color-muted-foreground)] ml-1">
                    (AUC: {curve.auc.toFixed(3)})
                  </span>
                )}
              </span>
            </div>
          ))}
          {showDiagonal && (
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 border-t border-dashed border-[var(--color-muted-foreground)]" />
              <span className="text-sm text-[var(--color-muted-foreground)]">
                Random (AUC: 0.500)
              </span>
            </div>
          )}
        </div>
      </div>
    );
  }
);

ROCCurve.displayName = 'ROCCurve';

export default ROCCurve;
