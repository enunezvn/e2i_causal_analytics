/**
 * Heterogeneous Treatment Effects Component
 * ==========================================
 *
 * Displays segment-level Conditional Average Treatment Effects (CATE) sourced
 * from the real segment-analysis endpoint (`POST /api/segments/analyze`), which
 * runs the Heterogeneous Optimizer agent (EconML CausalForestDML) over the
 * patient_journeys clinical substrate — the same contract as /segment-analysis.
 * The backend fixes the clinical segment set, effect modifiers, and confounders
 * server-side; only the treatment/outcome pair (curated allowlist) and an
 * optional brand row-filter are caller-selectable.
 *
 * Honest-state contract (no fabricated data):
 * - Before any run: explicit empty state with a "Run CATE analysis" action. The
 *   server trains a real causal forest (can take a minute), so it is
 *   user-triggered rather than auto-fired on mount.
 * - Pending: labeled loading state.
 * - Completed: real per-segment CATE (value, CATE, CI, n, significance) from the
 *   API's `cate_by_segment`. Significance is the API's `statistical_significance`,
 *   never invented.
 * - Failed: labeled error (no silent fallback).
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
import { useRunSegmentAnalysisAndWait } from '@/hooks/api/use-segments';
import type { CATEResult, RunSegmentAnalysisRequest } from '@/types/segments';

// =============================================================================
// TYPES
// =============================================================================

interface HeterogeneousTreatmentEffectsProps {
  className?: string;
  /** Treatment variable (patient_journeys allowlist; backend default). */
  treatmentVar?: string;
  /** Outcome variable (patient_journeys allowlist; backend default). */
  outcomeVar?: string;
  /** Optional brand row-filter (NOT a causal variable); undefined => all brands. */
  brand?: string;
}

/** A single CATE row flattened out of the API's `cate_by_segment` map. */
interface FlatSegment extends CATEResult {
  dimension: string;
}

// =============================================================================
// HELPERS
// =============================================================================

/**
 * Binary outcomes on the patient_journeys clinical contract — the backend's
 * curated outcome list (src/api/routes/causal.py `_CAUSAL_DATASET_SPECS`).
 * A CATE on these is a probability delta, displayed in percentage points (pp),
 * the same headline unit /segment-analysis uses: 0.017 on persistent_180d is
 * +1.7 pp of persistence probability — NOT "+1.7%".
 */
const BINARY_OUTCOMES = new Set([
  'persistent_180d',
  'discontinued_180d',
  'treatment_initiated',
  'adherent_180d',
  'low_gap_180d',
]);

/**
 * Format a treatment effect honestly for the outcome's scale: percentage
 * points for binary outcomes, the raw effect size otherwise. The backend
 * validator also accepts covariates (e.g. engagement_score) as outcome_var,
 * so an effect on a non-binary outcome must never be labeled "pp".
 */
function fmtEffect(value: number | undefined | null, binaryOutcome: boolean, digits = 1): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  if (!binaryOutcome) return `${value >= 0 ? '+' : ''}${value.toFixed(3)}`;
  const pp = value * 100;
  return `${pp >= 0 ? '+' : ''}${pp.toFixed(digits)} pp`;
}

/**
 * Flatten `cate_by_segment` (Record<dimension, CATEResult[]>) into a flat list,
 * tagging each row with its segment dimension for display.
 */
function flattenCATE(bySegment: Record<string, CATEResult[]> | undefined): FlatSegment[] {
  if (!bySegment) return [];
  const out: FlatSegment[] = [];
  for (const [dimension, results] of Object.entries(bySegment)) {
    for (const r of results ?? []) out.push({ ...r, dimension });
  }
  return out;
}

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

function SegmentCard({
  segment,
  binaryOutcome,
}: {
  segment: FlatSegment;
  binaryOutcome: boolean;
}) {
  const significant = segment.statistical_significance;
  const isPositive = (segment.cate_estimate ?? 0) >= 0;

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
            {segment.segment_value}
          </h4>
          <p className="text-xs text-[var(--color-muted-foreground)]">by {segment.dimension}</p>
        </div>
        <Badge
          variant="outline"
          className={cn(
            'text-xs',
            significant
              ? 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20'
              : 'bg-gray-500/10 text-gray-600 border-gray-500/20'
          )}
        >
          {significant ? 'Significant' : 'Not significant'}
        </Badge>
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
            {fmtEffect(segment.cate_estimate, binaryOutcome)}
          </div>
        </div>
        <div className="p-2 rounded bg-[var(--color-muted)]/30">
          <div className="text-xs text-[var(--color-muted-foreground)]">CI</div>
          <div className="text-sm font-medium text-[var(--color-foreground)] pt-1">
            [{fmtEffect(segment.cate_ci_lower, binaryOutcome)},{' '}
            {fmtEffect(segment.cate_ci_upper, binaryOutcome)}]
          </div>
        </div>
        <div className="p-2 rounded bg-[var(--color-muted)]/30">
          <div className="text-xs text-[var(--color-muted-foreground)]">Sample</div>
          <div className="text-lg font-bold text-[var(--color-foreground)]">
            {segment.sample_size.toLocaleString()}
          </div>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function HeterogeneousTreatmentEffects({
  className,
  treatmentVar = 'treatment_arm',
  outcomeVar = 'persistent_180d',
  brand,
}: HeterogeneousTreatmentEffectsProps) {
  const {
    mutate: runAnalysis,
    data: analysis,
    error,
    isPending,
  } = useRunSegmentAnalysisAndWait();

  const handleRun = useCallback(() => {
    // Segment set, effect modifiers, confounders, and data source are FIXED
    // server-side for the clinical patient_journeys path — send only what the
    // caller may choose (same contract as /segment-analysis).
    const request: RunSegmentAnalysisRequest = {
      query: `Treatment effect heterogeneity of ${treatmentVar} on ${outcomeVar} across clinical segments`,
      treatment_var: treatmentVar,
      outcome_var: outcomeVar,
      brand,
    };
    runAnalysis({ request, pollIntervalMs: 3000, maxWaitMs: 300000 });
  }, [runAnalysis, treatmentVar, outcomeVar, brand]);

  const failed = analysis?.status === 'failed';
  const segments: FlatSegment[] = failed ? [] : flattenCATE(analysis?.cate_by_segment);
  const hasResults = segments.length > 0;
  const significantCount = segments.filter((s) => s.statistical_significance).length;
  // pp labeling is only honest for probability outcomes; a continuous outcome
  // (allowed by the backend's union allowlist) renders the raw effect size.
  const binaryOutcome = BINARY_OUTCOMES.has(outcomeVar);

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
                Segment-level CATE — {treatmentVar} → {outcomeVar} across clinical segments
                {brand ? ` (${brand})` : ''}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {hasResults && (
              <Badge variant="outline" className="text-xs bg-indigo-500/10 text-indigo-600">
                {significantCount}/{segments.length} significant
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
                this can take a minute)
              </span>
            </div>
          </div>
        )}

        {/* Error State (genuine failures only — no silent fallback) */}
        {!isPending && (error || failed) && (
          <div className="flex items-start gap-2 p-3 rounded-lg bg-rose-500/5 border border-rose-500/20">
            <AlertTriangle className="h-4 w-4 text-rose-500 mt-0.5" />
            <div className="text-xs text-[var(--color-muted-foreground)]">
              <span className="font-medium text-rose-600">CATE analysis failed:</span>{' '}
              {error?.message ?? analysis?.warnings?.join('; ') ?? 'Unknown error'}
            </div>
          </div>
        )}

        {/* Empty state with explicit run action */}
        {!isPending && !error && !failed && !hasResults && (
          <EmptyState
            title="No CATE analysis has been run"
            description="Run a segment-level CATE analysis over the live cohort. The server trains a real causal forest, so this can take a minute."
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
                  <span className="text-xs text-[var(--color-muted-foreground)]">Overall ATE</span>
                </div>
                <div className="text-xl font-bold text-[var(--color-foreground)]">
                  {fmtEffect(analysis?.overall_ate, binaryOutcome)}
                </div>
              </div>
              <div className="p-3 rounded-lg bg-[var(--color-muted)]/30 border border-[var(--color-border)]">
                <div className="flex items-center gap-2 mb-1">
                  <TrendingUp className="h-4 w-4 text-[var(--color-muted-foreground)]" />
                  <span className="text-xs text-[var(--color-muted-foreground)]">Heterogeneity</span>
                </div>
                <div className="text-xl font-bold text-[var(--color-foreground)]">
                  {typeof analysis?.heterogeneity_score === 'number'
                    ? analysis.heterogeneity_score.toFixed(2)
                    : '—'}
                </div>
              </div>
            </div>

            <div className="grid gap-3">
              {segments.map((segment) => (
                <SegmentCard
                  key={`${segment.dimension}:${segment.segment_value}`}
                  segment={segment}
                  binaryOutcome={binaryOutcome}
                />
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
            <span className="font-medium text-indigo-600">CATE Analysis:</span> Conditional Average
            Treatment Effects show how intervention impact varies across segments. A segment flagged
            significant has a confidence interval that excludes zero — a reliable targeting
            opportunity.
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default HeterogeneousTreatmentEffects;
