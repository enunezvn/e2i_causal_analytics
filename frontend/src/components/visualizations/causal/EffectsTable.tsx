/**
 * EffectsTable Component
 * ======================
 *
 * A table component for displaying causal effect estimates with confidence intervals.
 * Supports sorting by different columns and visual indicators for effect significance.
 *
 * @module components/visualizations/causal/EffectsTable
 */

import * as React from 'react';
import { useState, useMemo, useCallback } from 'react';
import { cn } from '@/lib/utils';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { ArrowUpDown, ArrowUp, ArrowDown, TrendingUp, TrendingDown, Minus } from 'lucide-react';

// =============================================================================
// TYPES
// =============================================================================

export interface CausalEffect {
  /** Unique identifier for the effect */
  id: string;
  /** Name of the treatment/exposure variable */
  treatment: string;
  /** Name of the outcome variable */
  outcome: string;
  /** Point estimate of the causal effect */
  estimate: number;
  /** Standard error of the estimate */
  standardError?: number;
  /** Lower bound of confidence interval */
  ciLower: number;
  /** Upper bound of confidence interval */
  ciUpper: number;
  /** Confidence level (e.g., 0.95 for 95% CI) */
  confidenceLevel?: number;
  /** P-value for significance testing */
  pValue?: number;
  /** Whether the effect is statistically significant */
  isSignificant?: boolean;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

export type SortColumn = 'treatment' | 'outcome' | 'estimate' | 'ciLower' | 'ciUpper' | 'pValue';
export type SortDirection = 'asc' | 'desc';

export interface EffectsTableProps {
  /** Array of causal effects to display */
  effects: CausalEffect[];
  /** Whether the table is in a loading state */
  isLoading?: boolean;
  /** Callback when a row is selected */
  onRowSelect?: (effect: CausalEffect) => void;
  /** Currently selected effect ID */
  selectedEffectId?: string;
  /** Show visual confidence interval bars */
  showCIBars?: boolean;
  /** Number of decimal places for effect estimates */
  decimalPlaces?: number;
  /** Significance threshold for p-values */
  significanceThreshold?: number;
  /** Additional CSS classes */
  className?: string;
  /** Table caption for accessibility */
  caption?: string;
  /** Enable sorting functionality */
  sortable?: boolean;
  /** Initial sort column */
  initialSortColumn?: SortColumn;
  /** Initial sort direction */
  initialSortDirection?: SortDirection;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Formats a number to a specified number of decimal places
 */
function formatNumber(value: number, decimals: number): string {
  return value.toFixed(decimals);
}

/**
 * Formats a p-value with appropriate precision
 */
function formatPValue(pValue: number): string {
  if (pValue < 0.001) return '< 0.001';
  if (pValue < 0.01) return pValue.toFixed(3);
  return pValue.toFixed(2);
}

/**
 * Determines if an effect is positive, negative, or neutral
 */
function getEffectDirection(estimate: number, threshold = 0.001): 'positive' | 'negative' | 'neutral' {
  if (estimate > threshold) return 'positive';
  if (estimate < -threshold) return 'negative';
  return 'neutral';
}

/**
 * Calculates the width percentage for CI bar visualization
 */
function calculateCIBarWidth(value: number, min: number, max: number): number {
  const range = max - min;
  if (range === 0) return 50;
  return Math.max(0, Math.min(100, ((value - min) / range) * 100));
}

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * EffectsTable displays causal effect estimates with confidence intervals.
 *
 * @example
 * ```tsx
 * <EffectsTable
 *   effects={causalEffects}
 *   onRowSelect={(effect) => setSelectedEffect(effect)}
 *   showCIBars
 * />
 * ```
 */
const EffectsTable = React.forwardRef<HTMLDivElement, EffectsTableProps>(
  (
    {
      effects,
      isLoading = false,
      onRowSelect,
      selectedEffectId,
      showCIBars = true,
      decimalPlaces = 3,
      significanceThreshold = 0.05,
      className,
      caption = 'Causal effect estimates with confidence intervals',
      sortable = true,
      initialSortColumn = 'estimate',
      initialSortDirection = 'desc',
    },
    ref
  ) => {
    const [sortColumn, setSortColumn] = useState<SortColumn>(initialSortColumn);
    const [sortDirection, setSortDirection] = useState<SortDirection>(initialSortDirection);

    // Calculate min/max for CI bar visualization
    const { minValue, maxValue } = useMemo(() => {
      if (effects.length === 0) return { minValue: -1, maxValue: 1 };
      const allValues = effects.flatMap((e) => [e.ciLower, e.ciUpper, e.estimate]);
      return {
        minValue: Math.min(...allValues),
        maxValue: Math.max(...allValues),
      };
    }, [effects]);

    // Sort effects
    const sortedEffects = useMemo(() => {
      if (!sortable) return effects;

      return [...effects].sort((a, b) => {
        let aValue: string | number;
        let bValue: string | number;

        switch (sortColumn) {
          case 'treatment':
            aValue = a.treatment.toLowerCase();
            bValue = b.treatment.toLowerCase();
            break;
          case 'outcome':
            aValue = a.outcome.toLowerCase();
            bValue = b.outcome.toLowerCase();
            break;
          case 'estimate':
            aValue = a.estimate;
            bValue = b.estimate;
            break;
          case 'ciLower':
            aValue = a.ciLower;
            bValue = b.ciLower;
            break;
          case 'ciUpper':
            aValue = a.ciUpper;
            bValue = b.ciUpper;
            break;
          case 'pValue':
            aValue = a.pValue ?? 1;
            bValue = b.pValue ?? 1;
            break;
          default:
            return 0;
        }

        if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
        if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
        return 0;
      });
    }, [effects, sortColumn, sortDirection, sortable]);

    // Handle sort column click
    const handleSort = useCallback(
      (column: SortColumn) => {
        if (!sortable) return;
        if (sortColumn === column) {
          setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'));
        } else {
          setSortColumn(column);
          setSortDirection('desc');
        }
      },
      [sortColumn, sortable]
    );

    // Render sort icon
    const renderSortIcon = useCallback(
      (column: SortColumn) => {
        if (!sortable) return null;
        if (sortColumn !== column) {
          return <ArrowUpDown className="ml-1 h-3 w-3 text-[var(--color-muted-foreground)]" />;
        }
        return sortDirection === 'asc' ? (
          <ArrowUp className="ml-1 h-3 w-3" />
        ) : (
          <ArrowDown className="ml-1 h-3 w-3" />
        );
      },
      [sortColumn, sortDirection, sortable]
    );

    // Render effect direction icon
    const renderEffectIcon = useCallback((estimate: number) => {
      const direction = getEffectDirection(estimate);
      switch (direction) {
        case 'positive':
          return <TrendingUp className="h-4 w-4 text-emerald-500" aria-label="Positive effect" />;
        case 'negative':
          return <TrendingDown className="h-4 w-4 text-red-500" aria-label="Negative effect" />;
        default:
          return <Minus className="h-4 w-4 text-[var(--color-muted-foreground)]" aria-label="No effect" />;
      }
    }, []);

    // Render CI bar visualization
    const renderCIBar = useCallback(
      (effect: CausalEffect) => {
        const estimatePos = calculateCIBarWidth(effect.estimate, minValue, maxValue);
        const lowerPos = calculateCIBarWidth(effect.ciLower, minValue, maxValue);
        const upperPos = calculateCIBarWidth(effect.ciUpper, minValue, maxValue);
        const zeroPos = calculateCIBarWidth(0, minValue, maxValue);

        return (
          <div className="relative h-4 w-full min-w-[100px] bg-[var(--color-muted)]/30 rounded">
            {/* Zero line indicator */}
            <div
              className="absolute top-0 h-full w-px bg-[var(--color-muted-foreground)]/40"
              style={{ left: `${zeroPos}%` }}
            />
            {/* Confidence interval bar */}
            <div
              className="absolute top-1/2 -translate-y-1/2 h-1.5 bg-[var(--color-primary)]/30 rounded"
              style={{
                left: `${lowerPos}%`,
                width: `${upperPos - lowerPos}%`,
              }}
            />
            {/* Point estimate marker */}
            <div
              className={cn(
                'absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-3 w-3 rounded-full border-2',
                effect.isSignificant ?? (effect.pValue !== undefined && effect.pValue < significanceThreshold)
                  ? 'bg-[var(--color-primary)] border-[var(--color-primary)]'
                  : 'bg-[var(--color-muted-foreground)] border-[var(--color-muted-foreground)]'
              )}
              style={{ left: `${estimatePos}%` }}
            />
          </div>
        );
      },
      [minValue, maxValue, significanceThreshold]
    );

    // Handle row click
    const handleRowClick = useCallback(
      (effect: CausalEffect) => {
        onRowSelect?.(effect);
      },
      [onRowSelect]
    );

    // Empty state
    if (effects.length === 0 && !isLoading) {
      return (
        <div
          ref={ref}
          className={cn(
            'flex items-center justify-center rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-8',
            className
          )}
        >
          <div className="text-center">
            <div className="text-[var(--color-muted-foreground)] mb-2">
              No causal effects available
            </div>
            <p className="text-sm text-[var(--color-muted-foreground)]/60">
              Run causal analysis to estimate treatment effects
            </p>
          </div>
        </div>
      );
    }

    return (
      <div
        ref={ref}
        className={cn(
          'relative rounded-lg border border-[var(--color-border)] bg-[var(--color-card)]',
          className
        )}
      >
        {/* Loading overlay */}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-[var(--color-background)]/80 z-10 rounded-lg">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-[var(--color-muted)] border-t-[var(--color-primary)]" />
          </div>
        )}

        <Table>
          <caption className="sr-only">{caption}</caption>
          <TableHeader>
            <TableRow>
              <TableHead
                className={cn(sortable && 'cursor-pointer select-none hover:bg-[var(--color-muted)]/50')}
                onClick={() => handleSort('treatment')}
              >
                <div className="flex items-center">
                  Treatment
                  {renderSortIcon('treatment')}
                </div>
              </TableHead>
              <TableHead
                className={cn(sortable && 'cursor-pointer select-none hover:bg-[var(--color-muted)]/50')}
                onClick={() => handleSort('outcome')}
              >
                <div className="flex items-center">
                  Outcome
                  {renderSortIcon('outcome')}
                </div>
              </TableHead>
              <TableHead
                className={cn(
                  'text-right',
                  sortable && 'cursor-pointer select-none hover:bg-[var(--color-muted)]/50'
                )}
                onClick={() => handleSort('estimate')}
              >
                <div className="flex items-center justify-end">
                  Estimate
                  {renderSortIcon('estimate')}
                </div>
              </TableHead>
              <TableHead className="text-center">95% CI</TableHead>
              {showCIBars && <TableHead className="min-w-[120px]">CI Visualization</TableHead>}
              <TableHead
                className={cn(
                  'text-right',
                  sortable && 'cursor-pointer select-none hover:bg-[var(--color-muted)]/50'
                )}
                onClick={() => handleSort('pValue')}
              >
                <div className="flex items-center justify-end">
                  P-value
                  {renderSortIcon('pValue')}
                </div>
              </TableHead>
              <TableHead className="text-center">Sig.</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedEffects.map((effect) => {
              const isSignificant =
                effect.isSignificant ?? (effect.pValue !== undefined && effect.pValue < significanceThreshold);
              const isSelected = selectedEffectId === effect.id;

              return (
                <TableRow
                  key={effect.id}
                  className={cn(
                    onRowSelect && 'cursor-pointer',
                    isSelected && 'bg-[var(--color-muted)]'
                  )}
                  onClick={() => handleRowClick(effect)}
                  data-state={isSelected ? 'selected' : undefined}
                >
                  <TableCell className="font-medium">
                    <div className="flex items-center gap-2">
                      {renderEffectIcon(effect.estimate)}
                      <span>{effect.treatment}</span>
                    </div>
                  </TableCell>
                  <TableCell>{effect.outcome}</TableCell>
                  <TableCell className="text-right font-mono">
                    {formatNumber(effect.estimate, decimalPlaces)}
                  </TableCell>
                  <TableCell className="text-center font-mono text-xs text-[var(--color-muted-foreground)]">
                    [{formatNumber(effect.ciLower, decimalPlaces)}, {formatNumber(effect.ciUpper, decimalPlaces)}]
                  </TableCell>
                  {showCIBars && <TableCell>{renderCIBar(effect)}</TableCell>}
                  <TableCell className="text-right font-mono text-xs">
                    {effect.pValue !== undefined ? formatPValue(effect.pValue) : '—'}
                  </TableCell>
                  <TableCell className="text-center">
                    {effect.pValue !== undefined && (
                      <Badge
                        variant={isSignificant ? 'success' : 'secondary'}
                        className="text-xs"
                      >
                        {isSignificant ? 'Yes' : 'No'}
                      </Badge>
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    );
  }
);

EffectsTable.displayName = 'EffectsTable';

export { EffectsTable };
export default EffectsTable;
