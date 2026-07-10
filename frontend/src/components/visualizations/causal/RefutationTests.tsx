/**
 * RefutationTests Component
 * =========================
 *
 * A component for displaying refutation test results from causal inference analysis.
 * Refutation tests help validate causal estimates by testing key assumptions such as
 * random common cause, placebo treatment, and data subset validity.
 *
 * @module components/visualizations/causal/RefutationTests
 */

import * as React from 'react';
import { useMemo, useCallback } from 'react';
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
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { CheckCircle, XCircle, AlertCircle, Info } from 'lucide-react';

// =============================================================================
// TYPES
// =============================================================================

export type RefutationMethod =
  | 'random_common_cause'
  | 'placebo_treatment'
  | 'data_subset'
  | 'bootstrap'
  | 'add_unobserved_common_cause'
  | 'dummy_outcome';

export interface RefutationResult {
  /** Unique identifier for this test result */
  id: string;
  /** Name of the refutation method */
  method: RefutationMethod;
  /** Human-readable name for the method */
  methodName?: string;
  /** Original causal effect estimate */
  originalEstimate: number;
  /** Effect estimate after refutation */
  refutedEstimate: number;
  /** P-value for the refutation test */
  pValue: number;
  /** Whether the refutation test passed */
  passed: boolean;
  /** Description of what this test validates */
  description?: string;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

export interface RefutationTestsProps {
  /** Array of refutation test results */
  results: RefutationResult[];
  /** Whether data is loading */
  isLoading?: boolean;
  /** Significance threshold for pass/fail determination */
  significanceThreshold?: number;
  /** Number of decimal places for estimates */
  decimalPlaces?: number;
  /** Show summary statistics card */
  showSummary?: boolean;
  /** Show the comparison chart */
  showChart?: boolean;
  /** Callback when a row is clicked */
  onRowSelect?: (result: RefutationResult) => void;
  /** Currently selected result ID */
  selectedResultId?: string;
  /** Additional CSS classes */
  className?: string;
  /** Table caption for accessibility */
  caption?: string;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const METHOD_LABELS: Record<RefutationMethod, string> = {
  random_common_cause: 'Random Common Cause',
  placebo_treatment: 'Placebo Treatment',
  data_subset: 'Data Subset',
  bootstrap: 'Bootstrap Validation',
  add_unobserved_common_cause: 'Unobserved Common Cause',
  dummy_outcome: 'Dummy Outcome',
};

const METHOD_DESCRIPTIONS: Record<RefutationMethod, string> = {
  random_common_cause:
    'Tests if adding a random common cause variable changes the estimate significantly',
  placebo_treatment:
    'Tests if replacing treatment with a random placebo still shows an effect',
  data_subset:
    'Tests if the effect holds on random subsets of the data',
  bootstrap:
    'Tests stability of the estimate across bootstrap samples',
  add_unobserved_common_cause:
    'Tests sensitivity to unmeasured confounding',
  dummy_outcome:
    'Tests if treatment affects a dummy/random outcome variable',
};

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
 * Gets the method label, using custom name if provided
 */
function getMethodLabel(result: RefutationResult): string {
  return result.methodName || METHOD_LABELS[result.method] || result.method;
}

/**
 * Gets the method description
 */
function getMethodDescription(result: RefutationResult): string {
  return result.description || METHOD_DESCRIPTIONS[result.method] || '';
}

/**
 * Calculates the percentage change between original and refuted estimates
 */
function calculateChangePercent(original: number, refuted: number): number {
  if (original === 0) return refuted === 0 ? 0 : 100;
  return ((refuted - original) / Math.abs(original)) * 100;
}

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

interface ComparisonChartProps {
  results: RefutationResult[];
  decimalPlaces: number;
}

/**
 * Grouped bar chart comparing the ORIGINAL causal estimate against each test's
 * REFUTED estimate. Layout notes:
 *  - Left gutter carries y-axis tick labels so the scale is readable (previously
 *    only the 0-line was labelled, so the magnitude was invisible / cut off).
 *  - Each group draws two bars (original + refuted) whose value labels are
 *    COLOR-MATCHED to their bar, so a near-zero refuted value (e.g. a passing
 *    placebo test at -0.001) can never be mistaken for the tall original bar
 *    beside it.
 *  - The legend sits in the lower-left BELOW the plot, not floating over the bars.
 */
function ComparisonChart({ results, decimalPlaces }: ComparisonChartProps) {
  // Plot geometry
  const marginLeft = 56; // room for y-axis tick labels
  const marginRight = 20;
  const marginTop = 16;
  const chartHeight = 200;
  const barGroupWidth = 76;
  const barWidth = 22;
  const intraGap = 8; // gap between the original and refuted bar within a group

  const plotWidth = results.length * barGroupWidth;
  // Floor wide enough that the 3-item legend below the plot never clips: its last
  // item ("Refuted (Fail)") runs to ~marginLeft + 300 ≈ 350px. 380 also matches the
  // natural width of the standard 4-test suite, so this only widens the degraded
  // (<= 3 completed tests) case — the 4-test case is unchanged.
  const legendMinWidth = 380;
  const chartWidth = Math.max(plotWidth + marginLeft + marginRight, legendMinWidth);
  const plotBottom = marginTop + chartHeight;
  const xLabelY = plotBottom + 18;
  const legendY = plotBottom + 40;
  const svgHeight = plotBottom + 66;

  // Symmetric-ish scale that always includes 0.
  const allValues = results.flatMap((r) => [r.originalEstimate, r.refutedEstimate]);
  const minValue = Math.min(...allValues, 0);
  const maxValue = Math.max(...allValues, 0);
  const range = maxValue - minValue || 1;
  const padding = range * 0.12;
  const yMin = minValue - padding;
  const yMax = maxValue + padding;
  const yRange = yMax - yMin;

  const yToPixel = (v: number) => plotBottom - ((v - yMin) / yRange) * chartHeight;
  const zeroY = yToPixel(0);

  // Evenly spaced y-axis ticks across the range (5 divisions).
  const tickCount = 5;
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) => yMin + (yRange * i) / tickCount);

  return (
    <div className="w-full overflow-x-auto">
      <svg
        width={chartWidth}
        height={svgHeight}
        viewBox={`0 0 ${chartWidth} ${svgHeight}`}
        className="mx-auto"
        role="img"
        aria-label="Comparison chart of original vs refuted estimates"
      >
        {/* Y-axis gridlines + scale labels */}
        {ticks.map((t, i) => {
          const y = yToPixel(t);
          const isZero = Math.abs(t) < yRange * 1e-6;
          return (
            <g key={`tick-${i}`}>
              <line
                x1={marginLeft}
                y1={y}
                x2={chartWidth - marginRight}
                y2={y}
                stroke={isZero ? 'var(--color-muted-foreground)' : 'var(--color-border)'}
                strokeWidth={1}
                strokeDasharray={isZero ? '4,4' : undefined}
                opacity={isZero ? 0.9 : 0.4}
              />
              <text
                x={marginLeft - 8}
                y={y + 3}
                textAnchor="end"
                fontSize={9}
                fill="var(--color-muted-foreground)"
              >
                {t.toFixed(decimalPlaces)}
              </text>
            </g>
          );
        })}

        {/* Y-axis line */}
        <line
          x1={marginLeft}
          y1={marginTop}
          x2={marginLeft}
          y2={plotBottom}
          stroke="var(--color-border)"
          strokeWidth={1}
        />

        {/* Bars */}
        {results.map((result, index) => {
          const groupCenter = marginLeft + index * barGroupWidth + barGroupWidth / 2;
          const origX = groupCenter - barWidth - intraGap / 2;
          const refX = groupCenter + intraGap / 2;
          const refColor = result.passed
            ? 'var(--color-success)'
            : 'var(--color-destructive)';

          // Bar rectangles anchored at the zero line.
          const origHeight = Math.max((Math.abs(result.originalEstimate) / yRange) * chartHeight, 1);
          const origY = result.originalEstimate >= 0 ? zeroY - origHeight : zeroY;
          const refHeight = Math.max((Math.abs(result.refutedEstimate) / yRange) * chartHeight, 1);
          const refY = result.refutedEstimate >= 0 ? zeroY - refHeight : zeroY;

          // Value labels sit just OUTSIDE their own bar (above a positive bar,
          // below a negative one) and are color-matched to the bar so the pairing
          // is unambiguous even when one bar is near zero.
          const origLabelY = result.originalEstimate >= 0 ? origY - 4 : origY + origHeight + 11;
          const refLabelY = result.refutedEstimate >= 0 ? refY - 4 : refY + refHeight + 11;

          return (
            <g key={result.id}>
              <rect
                x={origX}
                y={origY}
                width={barWidth}
                height={origHeight}
                fill="var(--color-primary)"
                opacity={0.85}
              />
              <rect
                x={refX}
                y={refY}
                width={barWidth}
                height={refHeight}
                fill={refColor}
                opacity={0.85}
              />
              {/* Color-matched value labels */}
              <text
                x={origX + barWidth / 2}
                y={origLabelY}
                textAnchor="middle"
                fontSize={9}
                fontWeight={600}
                fill="var(--color-primary)"
              >
                {formatNumber(result.originalEstimate, decimalPlaces)}
              </text>
              <text
                x={refX + barWidth / 2}
                y={refLabelY}
                textAnchor="middle"
                fontSize={9}
                fontWeight={600}
                fill={refColor}
              >
                {formatNumber(result.refutedEstimate, decimalPlaces)}
              </text>
              {/* Method label under the group */}
              <text
                x={groupCenter}
                y={xLabelY}
                textAnchor="middle"
                fontSize={9}
                fill="var(--color-foreground)"
              >
                {getMethodLabel(result).split(' ')[0]}
              </text>
            </g>
          );
        })}

        {/* Legend — lower-left, below the plot (never over the bars) */}
        <g transform={`translate(${marginLeft}, ${legendY})`}>
          <rect x={0} y={0} width={11} height={11} rx={2} fill="var(--color-primary)" opacity={0.85} />
          <text x={15} y={9} fontSize={10} fill="var(--color-foreground)">
            Original
          </text>
          <rect x={78} y={0} width={11} height={11} rx={2} fill="var(--color-success)" opacity={0.85} />
          <text x={93} y={9} fontSize={10} fill="var(--color-foreground)">
            Refuted (Pass)
          </text>
          <rect
            x={188}
            y={0}
            width={11}
            height={11}
            rx={2}
            fill="var(--color-destructive)"
            opacity={0.85}
          />
          <text x={203} y={9} fontSize={10} fill="var(--color-foreground)">
            Refuted (Fail)
          </text>
        </g>
      </svg>
    </div>
  );
}

interface SummaryCardProps {
  results: RefutationResult[];
}

/**
 * Summary statistics card showing overall test results
 */
function SummaryCard({ results }: SummaryCardProps) {
  const { passed, failed, total } = useMemo(() => {
    const passed = results.filter((r) => r.passed).length;
    return {
      passed,
      failed: results.length - passed,
      total: results.length,
    };
  }, [results]);

  const passRate = total > 0 ? (passed / total) * 100 : 0;

  return (
    <div className="grid grid-cols-3 gap-4">
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-[var(--color-muted-foreground)]">
                Tests Passed
              </p>
              <p className="text-2xl font-bold text-[var(--color-success)]">
                {passed}
              </p>
            </div>
            <CheckCircle className="h-8 w-8 text-[var(--color-success)]/20" />
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-[var(--color-muted-foreground)]">
                Tests Failed
              </p>
              <p className="text-2xl font-bold text-[var(--color-destructive)]">
                {failed}
              </p>
            </div>
            <XCircle className="h-8 w-8 text-[var(--color-destructive)]/20" />
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-[var(--color-muted-foreground)]">
                Pass Rate
              </p>
              <p
                className={cn(
                  'text-2xl font-bold',
                  passRate >= 80
                    ? 'text-[var(--color-success)]'
                    : passRate >= 50
                      ? 'text-[var(--color-warning)]'
                      : 'text-[var(--color-destructive)]'
                )}
              >
                {passRate.toFixed(0)}%
              </p>
            </div>
            <AlertCircle
              className={cn(
                'h-8 w-8',
                passRate >= 80
                  ? 'text-[var(--color-success)]/20'
                  : passRate >= 50
                    ? 'text-[var(--color-warning)]/20'
                    : 'text-[var(--color-destructive)]/20'
              )}
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

/**
 * RefutationTests displays results from causal refutation tests.
 *
 * @example
 * ```tsx
 * <RefutationTests
 *   results={refutationResults}
 *   showSummary
 *   showChart
 *   onRowSelect={(result) => setSelectedResult(result)}
 * />
 * ```
 */
const RefutationTests = React.forwardRef<HTMLDivElement, RefutationTestsProps>(
  (
    {
      results,
      isLoading = false,
      significanceThreshold = 0.05,
      decimalPlaces = 3,
      showSummary = true,
      showChart = true,
      onRowSelect,
      selectedResultId,
      className,
      caption = 'Refutation test results for causal effect validation',
    },
    ref
  ) => {
    // Handle row click
    const handleRowClick = useCallback(
      (result: RefutationResult) => {
        onRowSelect?.(result);
      },
      [onRowSelect]
    );

    // Empty state
    if (results.length === 0 && !isLoading) {
      return (
        <div
          ref={ref}
          className={cn(
            'flex items-center justify-center rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-8',
            className
          )}
        >
          <div className="text-center">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-[var(--color-muted)]">
              <Info className="h-6 w-6 text-[var(--color-muted-foreground)]" />
            </div>
            <div className="text-[var(--color-muted-foreground)] mb-2">
              No refutation tests for this run
            </div>
            <p className="text-sm text-[var(--color-muted-foreground)]/60">
              This analysis produced no refutation test results to display.
            </p>
          </div>
        </div>
      );
    }

    return (
      <div ref={ref} className={cn('space-y-4', className)}>
        {/* Summary Cards */}
        {showSummary && results.length > 0 && <SummaryCard results={results} />}

        {/* Comparison Chart */}
        {showChart && results.length > 0 && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Estimate Comparison</CardTitle>
              <CardDescription>
                Original vs refuted effect estimates across all tests
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ComparisonChart results={results} decimalPlaces={decimalPlaces} />
            </CardContent>
          </Card>
        )}

        {/* Results Table */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Refutation Test Results</CardTitle>
            <CardDescription>
              Individual test results with p-values (threshold: {significanceThreshold})
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <div className="relative rounded-lg">
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
                    <TableHead className="w-[40px]">Status</TableHead>
                    <TableHead>Method</TableHead>
                    <TableHead className="text-right">Original</TableHead>
                    <TableHead className="text-right">Refuted</TableHead>
                    <TableHead className="text-right">Change</TableHead>
                    <TableHead className="text-right">P-value</TableHead>
                    <TableHead className="text-center">Result</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {results.map((result) => {
                    const isSelected = selectedResultId === result.id;
                    const changePercent = calculateChangePercent(
                      result.originalEstimate,
                      result.refutedEstimate
                    );

                    return (
                      <TableRow
                        key={result.id}
                        className={cn(
                          onRowSelect && 'cursor-pointer',
                          isSelected && 'bg-[var(--color-muted)]'
                        )}
                        onClick={() => handleRowClick(result)}
                        data-state={isSelected ? 'selected' : undefined}
                      >
                        <TableCell>
                          {result.passed ? (
                            <CheckCircle
                              className="h-5 w-5 text-[var(--color-success)]"
                              aria-label="Passed"
                            />
                          ) : (
                            <XCircle
                              className="h-5 w-5 text-[var(--color-destructive)]"
                              aria-label="Failed"
                            />
                          )}
                        </TableCell>
                        <TableCell>
                          <div>
                            <div className="font-medium">{getMethodLabel(result)}</div>
                            <div className="text-xs text-[var(--color-muted-foreground)] max-w-[200px] truncate">
                              {getMethodDescription(result)}
                            </div>
                          </div>
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          {formatNumber(result.originalEstimate, decimalPlaces)}
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          {formatNumber(result.refutedEstimate, decimalPlaces)}
                        </TableCell>
                        <TableCell
                          className={cn(
                            'text-right font-mono text-xs',
                            Math.abs(changePercent) > 20
                              ? 'text-[var(--color-destructive)]'
                              : 'text-[var(--color-muted-foreground)]'
                          )}
                        >
                          {changePercent >= 0 ? '+' : ''}
                          {changePercent.toFixed(1)}%
                        </TableCell>
                        <TableCell className="text-right font-mono text-xs">
                          {formatPValue(result.pValue)}
                        </TableCell>
                        <TableCell className="text-center">
                          <Badge
                            variant={result.passed ? 'success' : 'destructive'}
                            className="text-xs"
                          >
                            {result.passed ? 'Pass' : 'Fail'}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }
);

RefutationTests.displayName = 'RefutationTests';

export { RefutationTests };
export default RefutationTests;
