/**
 * Confusion Matrix Component
 * ==========================
 *
 * Heatmap visualization for model classification performance.
 * Shows true vs predicted labels with color-coded cell counts.
 *
 * @module components/visualizations/charts/ConfusionMatrix
 */

import * as React from 'react';
import { useMemo } from 'react';
import { cn } from '@/lib/utils';

// =============================================================================
// TYPES
// =============================================================================

export interface ConfusionMatrixData {
  /** Matrix values as 2D array [actual][predicted] */
  matrix: number[][];
  /** Labels for each class */
  labels: string[];
}

export interface ConfusionMatrixProps {
  /** Confusion matrix data */
  data: ConfusionMatrixData;
  /** Title for the matrix */
  title?: string;
  /** Whether to show percentages instead of counts */
  showPercentages?: boolean;
  /** Whether to normalize by row (actual class) */
  normalizeByRow?: boolean;
  /** Color for low values */
  lowColor?: string;
  /** Color for high values */
  highColor?: string;
  /** Cell size in pixels */
  cellSize?: number;
  /** Whether data is loading */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Callback when a cell is clicked */
  onCellClick?: (actual: string, predicted: string, value: number) => void;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_DATA: ConfusionMatrixData = {
  matrix: [
    [85, 10, 5],
    [8, 82, 10],
    [3, 12, 85],
  ],
  labels: ['Low Risk', 'Medium Risk', 'High Risk'],
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function interpolateColor(lowColor: string, highColor: string, value: number): string {
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

function getContrastColor(bgColor: string): string {
  // Parse rgb color
  const match = bgColor.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
  if (!match) return '#000000';

  const r = parseInt(match[1]);
  const g = parseInt(match[2]);
  const b = parseInt(match[3]);

  // Calculate relative luminance
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance > 0.5 ? '#000000' : '#ffffff';
}

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * ConfusionMatrix displays classification model performance as a heatmap.
 *
 * @example
 * ```tsx
 * <ConfusionMatrix
 *   data={{
 *     matrix: [[90, 10], [15, 85]],
 *     labels: ['Negative', 'Positive'],
 *   }}
 *   showPercentages
 *   onCellClick={(actual, predicted, value) => console.log({ actual, predicted, value })}
 * />
 * ```
 */
export const ConfusionMatrix = React.forwardRef<HTMLDivElement, ConfusionMatrixProps>(
  (
    {
      data: propData,
      title = 'Confusion Matrix',
      showPercentages = false,
      normalizeByRow = false,
      lowColor = '#f0f9ff',
      highColor = '#1e40af',
      cellSize = 80,
      isLoading = false,
      className,
      onCellClick,
    },
    ref
  ) => {
    // Use provided data or sample
    const data = propData ?? SAMPLE_DATA;

    // Calculate normalized values and metrics
    const { normalizedMatrix, metrics } = useMemo(() => {
      const { matrix, labels } = data;
      const numClasses = labels.length;

      // Calculate row sums for normalization
      const rowSums = matrix.map((row) => row.reduce((sum, val) => sum + val, 0));
      const total = rowSums.reduce((sum, val) => sum + val, 0);

      // Normalize matrix
      const normalized = matrix.map((row, i) =>
        row.map((val) => (normalizeByRow && rowSums[i] > 0 ? val / rowSums[i] : val / total))
      );

      // Calculate overall metrics
      let correctCount = 0;
      for (let i = 0; i < numClasses; i++) {
        correctCount += matrix[i][i];
      }

      const accuracy = total > 0 ? correctCount / total : 0;

      // Calculate per-class precision and recall
      const classMetrics = labels.map((_, i) => {
        const tp = matrix[i][i];
        const colSum = matrix.reduce((sum, row) => sum + row[i], 0);
        const precision = colSum > 0 ? tp / colSum : 0;
        const recall = rowSums[i] > 0 ? tp / rowSums[i] : 0;
        const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
        return { precision, recall, f1 };
      });

      return {
        normalizedMatrix: normalized,
        metrics: {
          accuracy,
          classMetrics,
          total,
        },
      };
    }, [data, normalizeByRow]);

    // Get cell value for display
    const getCellDisplay = (row: number, col: number): string => {
      const rawValue = data.matrix[row][col];
      if (showPercentages) {
        const pct = normalizedMatrix[row][col] * 100;
        return `${pct.toFixed(1)}%`;
      }
      return rawValue.toLocaleString();
    };

    // Get cell background color
    const getCellColor = (row: number, col: number): string => {
      const value = normalizedMatrix[row][col];
      const maxValue = Math.max(...normalizedMatrix.flat());
      const normalizedValue = maxValue > 0 ? value / maxValue : 0;
      return interpolateColor(lowColor, highColor, normalizedValue);
    };

    // Loading skeleton
    if (isLoading) {
      return (
        <div
          ref={ref}
          className={cn('animate-pulse', className)}
        >
          <div className="h-64 bg-[var(--color-muted)] rounded-md" />
        </div>
      );
    }

    const { labels } = data;

    return (
      <div ref={ref} className={cn('w-full', className)}>
        {/* Title */}
        {title && (
          <h3 className="text-lg font-semibold text-[var(--color-foreground)] mb-4">
            {title}
          </h3>
        )}

        <div className="flex gap-6">
          {/* Matrix */}
          <div className="flex-shrink-0">
            {/* Column labels (Predicted) */}
            <div className="flex ml-24">
              {labels.map((label) => (
                <div
                  key={`col-${label}`}
                  className="text-xs text-[var(--color-muted-foreground)] font-medium text-center truncate"
                  style={{ width: cellSize }}
                  title={label}
                >
                  {label}
                </div>
              ))}
            </div>
            <div className="text-xs text-[var(--color-muted-foreground)] text-center mb-1 ml-24">
              Predicted
            </div>

            <div className="flex">
              {/* Row labels (Actual) */}
              <div className="flex flex-col items-end justify-center pr-2">
                <div className="text-xs text-[var(--color-muted-foreground)] font-medium -rotate-90 whitespace-nowrap mb-2">
                  Actual
                </div>
                {labels.map((label) => (
                  <div
                    key={`row-${label}`}
                    className="text-xs text-[var(--color-muted-foreground)] font-medium text-right truncate w-20"
                    style={{ height: cellSize }}
                    title={label}
                  >
                    <span className="inline-flex items-center h-full">{label}</span>
                  </div>
                ))}
              </div>

              {/* Matrix cells */}
              <div className="grid" style={{ gridTemplateColumns: `repeat(${labels.length}, ${cellSize}px)` }}>
                {data.matrix.map((row, i) =>
                  row.map((_, j) => {
                    const bgColor = getCellColor(i, j);
                    const textColor = getContrastColor(bgColor);
                    const isDiagonal = i === j;

                    return (
                      <div
                        key={`cell-${i}-${j}`}
                        className={cn(
                          'flex items-center justify-center font-medium text-sm transition-transform',
                          isDiagonal && 'ring-2 ring-inset ring-white/30',
                          onCellClick && 'cursor-pointer hover:scale-105'
                        )}
                        style={{
                          width: cellSize,
                          height: cellSize,
                          backgroundColor: bgColor,
                          color: textColor,
                        }}
                        onClick={() => onCellClick?.(labels[i], labels[j], data.matrix[i][j])}
                        title={`Actual: ${labels[i]}, Predicted: ${labels[j]}\nCount: ${data.matrix[i][j]}`}
                      >
                        {getCellDisplay(i, j)}
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>

          {/* Metrics Summary */}
          <div className="flex-1 space-y-4">
            <div className="bg-[var(--color-muted)] rounded-lg p-4">
              <h4 className="text-sm font-medium text-[var(--color-foreground)] mb-2">
                Overall Metrics
              </h4>
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-[var(--color-muted-foreground)]">Accuracy</span>
                  <span className="font-medium">{(metrics.accuracy * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-[var(--color-muted-foreground)]">Total Samples</span>
                  <span className="font-medium">{metrics.total.toLocaleString()}</span>
                </div>
              </div>
            </div>

            <div className="bg-[var(--color-muted)] rounded-lg p-4">
              <h4 className="text-sm font-medium text-[var(--color-foreground)] mb-2">
                Per-Class Metrics
              </h4>
              <div className="space-y-2">
                {labels.map((label, i) => (
                  <div key={label} className="text-xs">
                    <div className="font-medium text-[var(--color-foreground)]">{label}</div>
                    <div className="flex gap-4 text-[var(--color-muted-foreground)]">
                      <span>P: {(metrics.classMetrics[i].precision * 100).toFixed(0)}%</span>
                      <span>R: {(metrics.classMetrics[i].recall * 100).toFixed(0)}%</span>
                      <span>F1: {(metrics.classMetrics[i].f1 * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center justify-center gap-2 mt-4">
          <span className="text-xs text-[var(--color-muted-foreground)]">Low</span>
          <div
            className="w-32 h-3 rounded"
            style={{
              background: `linear-gradient(to right, ${lowColor}, ${highColor})`,
            }}
          />
          <span className="text-xs text-[var(--color-muted-foreground)]">High</span>
        </div>
      </div>
    );
  }
);

ConfusionMatrix.displayName = 'ConfusionMatrix';

export default ConfusionMatrix;
