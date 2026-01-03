/**
 * SHAP Force Plot Component
 * =========================
 *
 * Force plot visualization showing how features push prediction from base to output.
 * Uses horizontal stacked bars with arrows indicating push direction.
 *
 * @module components/visualizations/shap/SHAPForcePlot
 */

import * as React from 'react';
import { useMemo } from 'react';
import { cn } from '@/lib/utils';
import type { FeatureContribution } from '@/types/explain';

// =============================================================================
// TYPES
// =============================================================================

export interface SHAPForcePlotProps {
  /** Base (expected) prediction value */
  baseValue: number;
  /** Final output prediction value */
  outputValue: number;
  /** Feature contributions with SHAP values */
  features: FeatureContribution[];
  /** Maximum number of features to display (default: 8) */
  maxFeatures?: number;
  /** Height in pixels (default: 120) */
  height?: number;
  /** Color for positive contributions */
  positiveColor?: string;
  /** Color for negative contributions */
  negativeColor?: string;
  /** Whether data is loading */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Format for displaying values */
  valueFormatter?: (value: number) => string;
  /** Show feature labels inside bars */
  showLabels?: boolean;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_FEATURES: FeatureContribution[] = [
  { feature_name: 'days_since_visit', feature_value: 45, shap_value: 0.18, contribution_direction: 'positive', contribution_rank: 1 },
  { feature_name: 'total_prescriptions', feature_value: 12, shap_value: -0.12, contribution_direction: 'negative', contribution_rank: 2 },
  { feature_name: 'territory_sales', feature_value: 150000, shap_value: 0.08, contribution_direction: 'positive', contribution_rank: 3 },
  { feature_name: 'specialty_oncology', feature_value: 1, shap_value: 0.06, contribution_direction: 'positive', contribution_rank: 4 },
  { feature_name: 'engagement_score', feature_value: 3, shap_value: -0.05, contribution_direction: 'negative', contribution_rank: 5 },
];

const SAMPLE_BASE = 0.35;
const SAMPLE_OUTPUT = 0.5;

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * SHAPForcePlot displays feature contributions as a force diagram.
 *
 * @example
 * ```tsx
 * <SHAPForcePlot
 *   baseValue={0.35}
 *   outputValue={0.72}
 *   features={explainResponse.top_features}
 * />
 * ```
 */
export const SHAPForcePlot = React.forwardRef<HTMLDivElement, SHAPForcePlotProps>(
  (
    {
      baseValue: propBase,
      outputValue: propOutput,
      features: propFeatures,
      maxFeatures = 8,
      height = 120,
      positiveColor = '#10b981', // Emerald
      negativeColor = '#f43f5e', // Rose
      isLoading = false,
      className,
      valueFormatter = (v) => v.toFixed(2),
      showLabels = true,
    },
    ref
  ) => {
    // Use provided data or sample
    const baseValue = propBase ?? SAMPLE_BASE;
    const outputValue = propOutput ?? SAMPLE_OUTPUT;
    const features = useMemo(
      () => propFeatures ?? SAMPLE_FEATURES,
      [propFeatures]
    );

    // Process features and calculate layout
    const { positiveFeatures, negativeFeatures, totalWidth, scale } = useMemo(() => {
      const sorted = [...features]
        .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
        .slice(0, maxFeatures);

      const positive = sorted.filter((f) => f.shap_value > 0);
      const negative = sorted.filter((f) => f.shap_value < 0);

      const posSum = positive.reduce((sum, f) => sum + f.shap_value, 0);
      const negSum = Math.abs(negative.reduce((sum, f) => sum + f.shap_value, 0));
      const total = Math.max(posSum, negSum, 0.1);

      return {
        positiveFeatures: positive,
        negativeFeatures: negative,
        totalWidth: total,
        scale: (v: number) => (Math.abs(v) / total) * 100,
      };
    }, [features, maxFeatures]);

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

    return (
      <div ref={ref} className={cn('w-full', className)}>
        {/* Header with base and output values */}
        <div className="flex justify-between items-center mb-2 px-2">
          <div className="text-sm">
            <span className="text-[var(--color-muted-foreground)]">Base: </span>
            <span className="font-medium">{valueFormatter(baseValue)}</span>
          </div>
          <div className="text-sm">
            <span className="text-[var(--color-muted-foreground)]">Output: </span>
            <span className="font-bold text-lg">{valueFormatter(outputValue)}</span>
          </div>
        </div>

        {/* Force plot visualization */}
        <div
          className="relative bg-[var(--color-muted)] rounded-lg overflow-hidden"
          style={{ height: height - 50 }}
        >
          {/* Center line */}
          <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-[var(--color-border)] z-10" />

          {/* Negative features (left side) */}
          <div
            className="absolute top-0 bottom-0 right-1/2 flex flex-row-reverse items-center"
            style={{ width: '50%' }}
          >
            {negativeFeatures.map((feature, index) => {
              const width = scale(feature.shap_value);
              const accumulated = negativeFeatures
                .slice(0, index)
                .reduce((sum, f) => sum + scale(f.shap_value), 0);

              return (
                <div
                  key={feature.feature_name}
                  className="h-8 flex items-center justify-end px-1 relative group"
                  style={{
                    width: `${width}%`,
                    backgroundColor: negativeColor,
                    marginRight: index === 0 ? 0 : undefined,
                    left: `-${accumulated}%`,
                  }}
                  title={`${feature.feature_name}: ${feature.shap_value.toFixed(4)}`}
                >
                  {showLabels && width > 10 && (
                    <span className="text-white text-xs truncate">
                      {feature.feature_name.replace(/_/g, ' ')}
                    </span>
                  )}
                  {/* Arrow */}
                  <div
                    className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1"
                    style={{
                      width: 0,
                      height: 0,
                      borderTop: '10px solid transparent',
                      borderBottom: '10px solid transparent',
                      borderRight: `8px solid ${negativeColor}`,
                    }}
                  />
                  {/* Tooltip on hover */}
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-[var(--color-popover)] border border-[var(--color-border)] rounded shadow-lg text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity z-20 pointer-events-none">
                    <div className="font-medium">{feature.feature_name}</div>
                    <div>Value: {String(feature.feature_value)}</div>
                    <div className="text-rose-500">SHAP: {feature.shap_value.toFixed(4)}</div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Positive features (right side) */}
          <div
            className="absolute top-0 bottom-0 left-1/2 flex items-center"
            style={{ width: '50%' }}
          >
            {positiveFeatures.map((feature, index) => {
              const width = scale(feature.shap_value);
              const accumulated = positiveFeatures
                .slice(0, index)
                .reduce((sum, f) => sum + scale(f.shap_value), 0);

              return (
                <div
                  key={feature.feature_name}
                  className="h-8 flex items-center justify-start px-1 relative group"
                  style={{
                    width: `${width}%`,
                    backgroundColor: positiveColor,
                    marginLeft: index === 0 ? 0 : undefined,
                    left: `${accumulated}%`,
                  }}
                  title={`${feature.feature_name}: ${feature.shap_value.toFixed(4)}`}
                >
                  {showLabels && width > 10 && (
                    <span className="text-white text-xs truncate">
                      {feature.feature_name.replace(/_/g, ' ')}
                    </span>
                  )}
                  {/* Arrow */}
                  <div
                    className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1"
                    style={{
                      width: 0,
                      height: 0,
                      borderTop: '10px solid transparent',
                      borderBottom: '10px solid transparent',
                      borderLeft: `8px solid ${positiveColor}`,
                    }}
                  />
                  {/* Tooltip on hover */}
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-[var(--color-popover)] border border-[var(--color-border)] rounded shadow-lg text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity z-20 pointer-events-none">
                    <div className="font-medium">{feature.feature_name}</div>
                    <div>Value: {String(feature.feature_value)}</div>
                    <div className="text-emerald-500">SHAP: +{feature.shap_value.toFixed(4)}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Legend */}
        <div className="flex justify-center gap-6 mt-3 text-xs text-[var(--color-muted-foreground)]">
          <div className="flex items-center gap-1">
            <div
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: negativeColor }}
            />
            <span>Decreases prediction</span>
          </div>
          <div className="flex items-center gap-1">
            <div
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: positiveColor }}
            />
            <span>Increases prediction</span>
          </div>
        </div>
      </div>
    );
  }
);

SHAPForcePlot.displayName = 'SHAPForcePlot';

export default SHAPForcePlot;
