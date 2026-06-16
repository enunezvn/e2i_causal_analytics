/**
 * Heterogeneous Treatment Effects Component
 * ==========================================
 *
 * Displays segment-level Conditional Average Treatment Effects (CATE)
 * sourced from the real hierarchical analysis endpoint
 * (`POST /api/causal/hierarchical/analyze`, EconML-within-CausalML
 * segmentation served by the live heterogeneous-optimizer substrate).
 *
 * Honest-state contract (no fabricated data):
 * - Before any run: explicit empty state with a "Run CATE analysis" action.
 *   The analysis trains a real CausalForestDML server-side (can take
 *   minutes), so it is user-triggered rather than auto-fired on mount.
 * - Pending: labeled loading state.
 * - Completed: real `segment_results` (name, n, CATE, CI). Significance is
 *   derived from the CI excluding zero — no invented p-values or "drivers".
 * - Failed / demo-mode placeholder: labeled error / demo notice.
 *
 * @module components/insights/HeterogeneousTreatmentEffects
 */

import { useCallback } from 'react';
import {
  Users,
  TrendingUp,
  TrendingDown,
  BarChart3,
  RefreshCw,
  Info,
  AlertTriangle,
  Play,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { EmptyState } from '@/components/ui/EmptyState';
import { useRunHierarchicalAnalysisAndWait } from '@/hooks/api/use-causal';
import type {
  HierarchicalAnalysisRequest,
  SegmentCATEResult,
} from '@/types/causal';

// =============================================================================
// TYPES
// =============================================================================

interface HeterogeneousTreatmentEffectsProps {
  className?: string;
  /** Treatment variable for the CATE analysis (documented default). */
  treatmentVar?: string;
  /** Outcome variable for the CATE analysis (documented default). */
  outcomeVar?: string;
}

// =============================================================================
// HELPERS
// =============================================================================

/**
 * A segment effect is reported "significant" only when its confidence
 * interval excludes zero — derived from real bounds, never invented.
 */
function ciExcludesZero(seg: SegmentCATEResult): boolean | null {
  if (seg.cate_ci_lower === undefined || seg.cate_ci_upper === undefined) {
    return null;
  }
  return seg.cate_ci_lower > 0 || seg.cate_ci_upper < 0;
}

function fmtPct(value: number | undefined | null, digits = 1): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  const pct = value * 100;
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(digits)}%`;
}

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

function SegmentCard({ segment }: { segment: SegmentCATEResult }) {
  const significant = ciExcludesZero(segment);
  const isPositive = (segment.cate_mean ?? 0) >= 0;

  return (
    <div
      className={cn(
        'p-4 rounded-lg border',
        significant
          ? 'border-emerald-500/30 bg-emerald-500/5'
          : 'border-[var(--color-border)] bg-[var(--color-card)]'
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-2 mb-2">
        <div>
          <h4 className="text-sm font-medium text-[var(--color-foreground)]">
            {segment.segment_name}
          </h4>
          <p className="text-xs text-[var(--color-muted-foreground)]">
            Uplift range [{segment.uplift_range[0].toFixed(2)},{' '}
            {segment.uplift_range[1].toFixed(2)}]
          </p>
        </div>
        {significant !== null && (
          <Badge
            variant="outline"
            className={cn(
              'text-xs',
              significant
                ? 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20'
                : 'bg-gray-500/10 text-gray-600 border-gray-500/20'
            )}
          >
            {significant ? 'CI excludes 0' : 'CI includes 0'}
          </Badge>
        )}
      </div>

      {/* Treatment Effect */}
      <div className="grid grid-cols-3 gap-3">
        <div className="p-2 rounded bg-[var(--color-muted)]/30">
          <div className="text-xs text-[var(--color-muted-foreground)]">CATE</div>
          <div
            className={cn(
              'text-lg font-bold flex items-center gap-1',
              isPositive ? 'text-emerald-600' : 'text-rose-600'
            )}
          >
            {isPositive ? (
              <TrendingUp className="h-4 w-4" />
            ) : (
              <TrendingDown className="h-4 w-4" />
            )}
            {fmtPct(segment.cate_mean)}
          </div>
        </div>
        <div className="p-2 rounded bg-[var(--color-muted)]/30">
          {/* The CATE schema reports raw bounds with no confidence-level field */}
          <div className="text-xs text-[var(--color-muted-foreground)]">CI</div>
          <div className="text-sm font-medium text-[var(--color-foreground)] pt-1">
            {segment.cate_ci_lower !== undefined && segment.cate_ci_upper !== undefined
              ? `[${fmtPct(segment.cate_ci_lower)}, ${fmtPct(segment.cate_ci_upper)}]`
              : '—'}
          </div>
        </div>
        <div className="p-2 rounded bg-[var(--color-muted)]/30">
          <div className="text-xs text-[var(--color-muted-foreground)]">Sample</div>
          <div className="text-lg font-bold text-[var(--color-foreground)]">
            {segment.n_samples.toLocaleString()}
          </div>
        </div>
      </div>

      {!segment.success && (
        <div className="mt-3 pt-2 border-t border-[var(--color-border)] text-xs text-rose-600">
          Estimation failed{segment.error_message ? `: ${segment.error_message}` : ''}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function HeterogeneousTreatmentEffects({
  className,
  treatmentVar = 'rep_visits',
  outcomeVar = 'trx_count',
}: HeterogeneousTreatmentEffectsProps) {
  const {
    mutate: runAnalysis,
    data: analysis,
    error,
    isPending,
  } = useRunHierarchicalAnalysisAndWait();

  const handleRun = useCallback(() => {
    const request: HierarchicalAnalysisRequest = {
      treatment_var: treatmentVar,
      outcome_var: outcomeVar,
      n_segments: 3,
    };
    runAnalysis({ request, pollIntervalMs: 3000, maxWaitMs: 300000 });
  }, [runAnalysis, treatmentVar, outcomeVar]);

  // A demo-mode response is a pinned-zero placeholder, NOT a real analysis
  // (backend flags it explicitly). Never present it as real numbers.
  const isDemo =
    (analysis as { is_demo?: boolean | null } | undefined)?.is_demo === true;
  const failed = analysis?.status === 'failed';
  const segments: SegmentCATEResult[] =
    !isDemo && !failed && analysis?.segment_results ? analysis.segment_results : [];
  const hasResults = segments.length > 0;
  const significantCount = segments.filter((s) => ciExcludesZero(s) === true).length;

  // The hierarchical CATE endpoint fails closed with a 503 + explanatory message
  // when no real estimation-data backend is wired (the default in this
  // deployment). That is an honest "not available yet" condition, not a service
  // outage, so present it as a calm informational state rather than a red alarm.
  const noDataBackend =
    !!error &&
    (error as { status?: number }).status === 503 &&
    /no real data backend|no production data source/i.test(
      (error as { message?: string }).message ?? '',
    );

  return (
    <Card className={cn('bg-[var(--color-card)] border-[var(--color-border)]', className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-indigo-500/10">
              <BarChart3 className="h-5 w-5 text-indigo-500" />
            </div>
            <div>
              <CardTitle className="text-base font-semibold">
                Heterogeneous Treatment Effects
              </CardTitle>
              <p className="text-xs text-[var(--color-muted-foreground)]">
                Segment-level CATE — {treatmentVar} → {outcomeVar}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {hasResults && (
              <Badge variant="outline" className="text-xs bg-indigo-500/10 text-indigo-600">
                {significantCount}/{segments.length} CI excl. 0
              </Badge>
            )}
            <Button
              variant="ghost"
              size="icon"
              onClick={handleRun}
              disabled={isPending}
              className="h-8 w-8"
              aria-label="Refresh segment analysis"
            >
              <RefreshCw className={cn('h-4 w-4', isPending && 'animate-spin')} />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Loading State */}
        {isPending && (
          <div className="flex items-center justify-center py-8">
            <div className="flex items-center gap-3 text-[var(--color-muted-foreground)]">
              <RefreshCw className="h-5 w-5 animate-spin" />
              <span className="text-sm">
                Analyzing segment effects... (trains a causal forest server-side;
                this can take a few minutes)
              </span>
            </div>
          </div>
        )}

        {/* Honest "data not wired yet" state — an expected condition, not a failure */}
        {!isPending && noDataBackend && (
          <EmptyState
            title="Live CATE data isn’t wired yet"
            description="Segment-level heterogeneous treatment effects need a real estimation dataset (treatment, outcome and effect-modifier columns). That data source isn’t connected on this surface yet, so no analysis can run — this is an honest empty state, not a service failure."
          />
        )}

        {/* Error State (genuine failures only) */}
        {!isPending && !noDataBackend && (error || failed) && (
          <div className="flex items-start gap-2 p-3 rounded-lg bg-rose-500/5 border border-rose-500/20">
            <AlertTriangle className="h-4 w-4 text-rose-500 mt-0.5" />
            <div className="text-xs text-[var(--color-muted-foreground)]">
              <span className="font-medium text-rose-600">CATE analysis failed:</span>{' '}
              {error?.message ?? analysis?.errors?.join('; ') ?? 'Unknown error'}
            </div>
          </div>
        )}

        {/* Demo-placeholder notice — never presented as real analysis */}
        {!isPending && !error && isDemo && (
          <EmptyState
            title="Demo-mode placeholder response"
            description="The causal engine returned a demo placeholder, not a real analysis. Results are suppressed to avoid presenting pinned values as measured effects."
          />
        )}

        {/* Empty state with explicit run action */}
        {!isPending && !error && !failed && !isDemo && !hasResults && (
          <EmptyState
            title="No CATE analysis has been run"
            description="Run a segment-level CATE analysis against live data. The server trains a real causal forest, so this can take a few minutes."
            action={
              <Button onClick={handleRun} size="sm">
                <Play className="h-4 w-4 mr-2" />
                Run CATE analysis
              </Button>
            }
          />
        )}

        {/* Results */}
        {!isPending && hasResults && (
          <>
            {/* Summary Stats — real overall ATE and heterogeneity from the API */}
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="p-3 rounded-lg bg-[var(--color-muted)]/30 border border-[var(--color-border)]">
                <div className="flex items-center gap-2 mb-1">
                  <Users className="h-4 w-4 text-[var(--color-muted-foreground)]" />
                  <span className="text-xs text-[var(--color-muted-foreground)]">
                    Overall ATE
                  </span>
                </div>
                <div className="text-xl font-bold text-[var(--color-foreground)]">
                  {fmtPct(analysis?.overall_ate)}
                </div>
              </div>
              <div className="p-3 rounded-lg bg-[var(--color-muted)]/30 border border-[var(--color-border)]">
                <div className="flex items-center gap-2 mb-1">
                  <TrendingUp className="h-4 w-4 text-[var(--color-muted-foreground)]" />
                  <span className="text-xs text-[var(--color-muted-foreground)]">
                    Heterogeneity (I²)
                  </span>
                </div>
                <div className="text-xl font-bold text-[var(--color-foreground)]">
                  {typeof analysis?.segment_heterogeneity === 'number'
                    ? analysis.segment_heterogeneity.toFixed(2)
                    : '—'}
                </div>
              </div>
            </div>

            <div className="grid gap-3">
              {segments.map((segment) => (
                <SegmentCard key={segment.segment_id} segment={segment} />
              ))}
            </div>

            {analysis?.warnings && analysis.warnings.length > 0 && (
              <div className="flex items-start gap-2 p-3 rounded-lg bg-amber-500/5 border border-amber-500/20">
                <AlertTriangle className="h-4 w-4 text-amber-500 mt-0.5" />
                <div className="text-xs text-[var(--color-muted-foreground)]">
                  {analysis.warnings.join(' ')}
                </div>
              </div>
            )}
          </>
        )}

        {/* Info Banner */}
        <div className="flex items-start gap-2 p-3 rounded-lg bg-indigo-500/5 border border-indigo-500/20">
          <Info className="h-4 w-4 text-indigo-500 mt-0.5" />
          <div className="text-xs text-[var(--color-muted-foreground)]">
            <span className="font-medium text-indigo-600">CATE Analysis:</span>{' '}
            Conditional Average Treatment Effects show how intervention impact
            varies across uplift segments. Segments whose confidence interval
            excludes zero indicate reliable targeting opportunities.
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default HeterogeneousTreatmentEffects;
