// frontend/src/components/causal/CausalAnalysisDetail.tsx
/**
 * CausalAnalysisDetail — the shared deep view for one validated causal effect.
 * ===========================================================================
 *
 * Lifted verbatim from the post-#1030 CausalDiscovery drill-down so BOTH the
 * unified page's leaderboard drill-down and its "Pose your own question" manual
 * result render the identical deep view: discovered DAG + per-test refutation,
 * the data-driven estimator-comparison panel (#1030), and the interpretation
 * (executive summary / narrative / key insights / recommendations). All values
 * are REAL agent outputs surfaced from the response — never fabricated.
 *
 * @module components/causal/CausalAnalysisDetail
 */

import { useEffect, useMemo, useRef, useState } from 'react';

import { CausalDiscovery as CausalDiscoveryViz } from '@/components/visualizations/CausalDiscovery';
import type { CausalNode, CausalEdge } from '@/components/visualizations/causal/CausalDAG';
import type {
  RefutationResult,
  RefutationMethod,
} from '@/components/visualizations/causal/RefutationTests';
import { Badge } from '@/components/ui/badge';
import { EmptyState } from '@/components/ui/EmptyState';
import type {
  AgentCausalAnalysisResponse,
  EstimatorComparison,
  RefutationTestDetail,
} from '@/types/causal';
import { useClinicalContext, useClinicalNarrativeInsight } from '@/hooks/api';
import { ClinicalContextPanel } from './ClinicalContextPanel';

// The dataset each grain estimates over (mirrors the page's GRAINS list); the
// narrative endpoint wants the grain word, the result carries the dataset.
const DATASET_GRAIN: Record<string, string> = {
  patient_journeys: 'patient',
  hcp_adoption: 'hcp',
  nba_triggers: 'trigger',
};

function formatEffect(ate: number | null | undefined): string {
  if (ate === null || ate === undefined || Number.isNaN(ate)) return 'N/A';
  return ate.toFixed(4);
}

function formatCI(lower?: number | null, upper?: number | null): string {
  if (lower === null || lower === undefined || upper === null || upper === undefined) return '—';
  return `[${lower.toFixed(3)}, ${upper.toFixed(3)}]`;
}

function gateBadge(decision?: string | null) {
  if (decision === 'proceed') return <Badge variant="default">Proceed</Badge>;
  if (decision === 'review') return <Badge variant="secondary">Review</Badge>;
  if (decision === 'block') return <Badge variant="destructive">Blocked</Badge>;
  return <Badge variant="outline">—</Badge>;
}

// Backend refutation test_name -> the viz's RefutationMethod union. The backend
// surfaces the contract key, so the sensitivity test arrives as
// `unobserved_common_cause`; `sensitivity_e_value` (the raw enum) is mapped too
// as defense-in-depth so it can never fall through to "Random Common Cause".
const REFUTATION_METHOD_MAP: Record<string, RefutationMethod> = {
  placebo_treatment: 'placebo_treatment',
  random_common_cause: 'random_common_cause',
  data_subset: 'data_subset',
  bootstrap: 'bootstrap',
  unobserved_common_cause: 'add_unobserved_common_cause',
  add_unobserved_common_cause: 'add_unobserved_common_cause',
  sensitivity_e_value: 'add_unobserved_common_cause',
};

function toRefutationResults(tests: RefutationTestDetail[] | undefined | null): RefutationResult[] {
  if (!tests) return [];
  return tests.map((t, i) => ({
    id: `${t.test_name}-${i}`,
    method: REFUTATION_METHOD_MAP[t.test_name] ?? 'random_common_cause',
    originalEstimate: t.original_effect ?? 0,
    refutedEstimate: t.new_effect ?? 0,
    pValue: t.p_value ?? 0,
    passed: t.passed,
    // #1867: three-state verdict — a 'warning' must not render as a failure.
    // Unknown/absent values are dropped so the viz falls back to `passed`.
    status:
      t.status === 'passed' || t.status === 'warning' || t.status === 'failed'
        ? t.status
        : undefined,
    description: t.details ?? undefined,
  }));
}

// The agent fits and energy-scores several estimators and picks the lowest score
// with a robust-over-fast tie-break. Surface that evaluation so the analyst sees
// WHAT was compared and WHY the winner won — not just the winner's name.
function EstimatorComparisonPanel({
  comparison,
  efficiency = false,
}: {
  comparison: EstimatorComparison;
  /** #1188: RCT variance-reduction run — OLS is the unbiased anchor. */
  efficiency?: boolean;
}) {
  // Rank fit estimators by energy score; sink skipped/failed ones to the bottom
  // (a skipped estimator has no score and is not-applicable, not a loser).
  const ranked = [...comparison.candidates].sort((a, b) => {
    if (a.skipped && !b.skipped) return 1;
    if (b.skipped && !a.skipped) return -1;
    if (a.energy_score == null) return 1;
    if (b.energy_score == null) return -1;
    return a.energy_score - b.energy_score;
  });
  const nSkipped = comparison.candidates.filter((c) => c.skipped).length;
  const nApplicable = comparison.candidates.length - nSkipped;
  const nFit = comparison.candidates.filter((c) => c.success).length;
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-sm font-medium">Estimator selection (data-driven)</p>
        <p className="text-xs text-muted-foreground">
          {nFit}/{nApplicable} applicable estimator{nApplicable === 1 ? '' : 's'} fit
          {nSkipped > 0 ? ` · ${nSkipped} not applicable (no covariates)` : ''} · lower energy
          score is better
        </p>
      </div>
      <div className="overflow-x-auto rounded-md border">
        <table className="w-full text-sm">
          <thead className="border-b bg-muted/40 text-left text-xs uppercase text-muted-foreground">
            <tr>
              <th className="p-2 font-medium">Estimator</th>
              <th className="p-2 font-medium">Energy score</th>
              <th className="p-2 font-medium">ATE</th>
              <th className="p-2 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {ranked.map((c) => (
              <tr
                key={c.estimator}
                className={`border-b last:border-0 ${c.is_selected ? 'bg-muted/50 font-medium' : ''} ${
                  c.skipped ? 'opacity-70' : ''
                }`}
              >
                <td className="p-2 capitalize">
                  {c.estimator.replace(/_/g, ' ')}
                  {c.is_selected && (
                    <Badge variant="default" className="ml-2 align-middle">
                      Selected
                    </Badge>
                  )}
                  {c.skipped && (
                    <Badge variant="outline" className="ml-2 align-middle font-normal">
                      Not applicable
                    </Badge>
                  )}
                  {efficiency && c.estimator === 'ols' && (
                    <Badge variant="outline" className="ml-2 align-middle font-normal">
                      Unbiased anchor
                    </Badge>
                  )}
                </td>
                <td className="p-2">{c.energy_score != null ? c.energy_score.toFixed(4) : '—'}</td>
                <td className="p-2">{c.ate != null ? c.ate.toFixed(4) : '—'}</td>
                <td className="p-2 text-xs text-muted-foreground">
                  {c.success
                    ? 'fit'
                    : c.skipped
                      ? (c.error ?? 'not applicable to this design')
                      : `failed${c.error ? `: ${c.error}` : ''}`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {comparison.selection_reason && (
        <p className="text-xs text-muted-foreground">
          <span className="font-medium">Why this estimator:</span> {comparison.selection_reason}
          {comparison.quality_tier ? ` (quality: ${comparison.quality_tier})` : ''}
        </p>
      )}
    </div>
  );
}

// Show the NAIVE (unadjusted) diff-in-means next to the ADJUSTED estimate and the
// confounding bias adjustment removed (Option D). The gold-standard treatment_arm
// is assigned by a covariate propensity, so the naive estimate is biased — making
// the gap between naive and adjusted the headline demonstration of WHY the causal
// adjustment matters. Binary-treatment only; honest not-applicable otherwise.
function ConfoundingAdjustmentPanel({ result }: { result: AgentCausalAnalysisResponse }) {
  const naive = result.naive_ate;
  const hasNaive = naive !== null && naive !== undefined && !Number.isNaN(naive);
  const delta = result.confounding_bias_removed;
  const hasDelta = delta !== null && delta !== undefined && !Number.isNaN(delta);

  // #1188: on a randomized (RCT) run the baselines are EFFICIENCY controls —
  // adjustment tightens the interval, it does not remove confounding bias.
  // Presenting it with the observational "bias removed" prose would misstate
  // what happened, so the panel switches to an honest precision framing with
  // BOTH intervals visible (the tightening IS the deliverable).
  if (result.adjustment_type === 'efficiency') {
    const baselines = result.baseline_covariates ?? [];
    return (
      <div className="rounded-md border bg-muted/30 p-3 space-y-1">
        <p className="text-sm font-medium">Precision adjustment (randomized design)</p>
        <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1 text-sm">
          <span className="text-muted-foreground">
            Unadjusted (anchor):{' '}
            <span className="font-semibold text-foreground">{formatEffect(naive)}</span>{' '}
            <span className="text-xs">
              95% CI {formatCI(result.naive_ate_ci_lower, result.naive_ate_ci_upper)}
            </span>
          </span>
          <span className="text-muted-foreground">
            Adjusted:{' '}
            <span className="font-semibold text-primary">{formatEffect(result.ate)}</span>{' '}
            <span className="text-xs">95% CI {formatCI(result.ate_ci_lower, result.ate_ci_upper)}</span>
          </span>
        </div>
        <p className="text-xs text-muted-foreground">
          Treatment is randomized here, so both point estimates are unbiased — the baseline
          covariates enter only for variance reduction (ANCOVA-style precision), tightening the
          confidence interval around the same effect. The unadjusted difference-in-means stays the
          reference anchor.
        </p>
        {baselines.length > 0 && (
          <p className="text-xs text-muted-foreground">
            <span className="font-medium">Baseline covariates (pre-treatment):</span>{' '}
            {baselines.join(', ')}
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="rounded-md border bg-muted/30 p-3 space-y-1">
      <p className="text-sm font-medium">Confounding adjustment</p>
      {hasNaive ? (
        <>
          <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1 text-sm">
            <span className="text-muted-foreground">
              Naive (unadjusted):{' '}
              <span className="font-semibold text-foreground">{formatEffect(naive)}</span>{' '}
              <span className="text-xs">
                95% CI {formatCI(result.naive_ate_ci_lower, result.naive_ate_ci_upper)}
              </span>
            </span>
            <span className="text-muted-foreground">
              Adjusted:{' '}
              <span className="font-semibold text-primary">{formatEffect(result.ate)}</span>
            </span>
          </div>
          {hasDelta && (
            <p className="text-xs text-muted-foreground">
              {Math.abs(delta) < 0.005
                ? 'Adjustment left the estimate essentially unchanged — little confounding on the adjusted covariates.'
                : delta > 0
                  ? `Adjustment removed ${Math.abs(delta).toFixed(4)} of upward confounding bias — the naive difference-in-means overstated the effect because treated and untreated units differ on the adjusted covariates.`
                  : `Adjustment corrected ${Math.abs(delta).toFixed(4)} of downward confounding bias — the naive difference-in-means understated the effect.`}
            </p>
          )}
        </>
      ) : (
        <p className="text-xs text-muted-foreground">
          Naive (unadjusted) contrast: not applicable (non-binary treatment).
        </p>
      )}
    </div>
  );
}

/** The shared deep view for one validated causal effect (agent-analyze result). */
export function CausalAnalysisDetail({
  result,
  brand,
  labelFor,
}: {
  result: AgentCausalAnalysisResponse;
  /** Page-owned column display labels (see ClinicalContextPanel.labelFor). */
  labelFor?: (col: string) => string;
  brand?: string | null;
}) {
  // Map the analysis's DAG onto the shared causal-graph visualization. The
  // treatment->outcome edge carries the estimated effect.
  const { vizNodes, vizEdges } = useMemo((): {
    vizNodes: CausalNode[];
    vizEdges: CausalEdge[];
  } => {
    const dag = result.dag;
    const treatmentSet = new Set(dag.treatment_nodes);
    const outcomeSet = new Set(dag.outcome_nodes);
    const confounderSet = new Set(dag.adjustment_sets.flat());
    const nodes: CausalNode[] = dag.nodes.map((name) => ({
      id: name,
      label: name,
      type: treatmentSet.has(name)
        ? 'treatment'
        : outcomeSet.has(name)
          ? 'outcome'
          : confounderSet.has(name)
            ? 'confounder'
            : 'variable',
    }));
    const edges: CausalEdge[] = dag.edges.map(([source, target]) => {
      const isEffectEdge = treatmentSet.has(source) && outcomeSet.has(target);
      return {
        id: `${source}->${target}`,
        source,
        target,
        type: 'causal' as const,
        ...(isEffectEdge && result.ate !== null && result.ate !== undefined
          ? { effect: result.ate }
          : {}),
      };
    });
    return { vizNodes: nodes, vizEdges: edges };
  }, [result]);

  const refutationResults = useMemo(
    () => toRefutationResults(result.refutation?.tests),
    [result]
  );

  // Additive clinical narrative for THIS analysis (brand + treatment -> outcome).
  // Disabled until brand and outcome are present; never touches the estimate above.
  const clinicalContext = useClinicalContext(brand, result.outcome_var, result.treatment_var);

  // LLM narrative for THIS analysis: auto-fire once the clinical context AND
  // the result are both in (keyed so one distinct analysis fires exactly once,
  // not on every re-render). The scope tag suppresses a late response from a
  // previous analysis (mirrors the page's manualScope stale-scope guard).
  const narrativeInsight = useClinicalNarrativeInsight();
  const { mutate: generateNarrative, reset: resetNarrative } = narrativeInsight;
  const narrativeKeyRef = useRef<string | null>(null);
  const [narrativeScope, setNarrativeScope] = useState<string | null>(null);
  const narrativeKey = `${brand ?? ''}|${result.dataset}|${result.treatment_var}|${result.outcome_var}|${result.ate ?? 'null'}`;
  useEffect(() => {
    if (!brand || !clinicalContext.data) return;
    if (narrativeKeyRef.current === narrativeKey) return;
    narrativeKeyRef.current = narrativeKey;
    setNarrativeScope(narrativeKey);
    resetNarrative();
    generateNarrative({
      brand,
      grain: DATASET_GRAIN[result.dataset] ?? result.dataset,
      treatment: result.treatment_var,
      outcome: result.outcome_var,
      ate: result.ate ?? null,
      ate_ci_lower: result.ate_ci_lower ?? null,
      ate_ci_upper: result.ate_ci_upper ?? null,
      gate_decision: result.refutation.gate_decision ?? null,
      // #1868: per-test verdicts so the narrative names warnings honestly.
      refutation_tests: (result.refutation.tests ?? []).map((t) => ({
        test_name: t.test_name,
        passed: t.passed,
        status: t.status ?? null,
        details: t.details ?? null,
      })),
    });
  }, [brand, clinicalContext.data, narrativeKey, result, generateNarrative, resetNarrative]);
  const narrativeInScope = narrativeScope === narrativeKey;
  const narrative = narrativeInScope ? narrativeInsight.data ?? null : null;
  const narrativeLoading = narrativeInScope && narrativeInsight.isPending;

  return (
    <div className="space-y-6">
      <div className="grid md:grid-cols-3 gap-6">
        <div className="text-center">
          <div className="text-3xl font-bold text-primary">{formatEffect(result.ate)}</div>
          <div className="text-sm text-muted-foreground mt-1">
            ATE · 95% CI {formatCI(result.ate_ci_lower, result.ate_ci_upper)}
          </div>
          {result.p_value !== null && result.p_value !== undefined && (
            <div className="text-xs text-muted-foreground mt-1">p = {result.p_value.toFixed(4)}</div>
          )}
        </div>
        <div className="text-center">
          <div className="text-lg font-semibold capitalize">
            {result.selected_estimator ? result.selected_estimator.replace(/_/g, ' ') : 'N/A'}
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            Estimator (data-driven) · {result.n_rows.toLocaleString()} rows
          </div>
        </div>
        <div className="text-center">
          <div className="flex items-center justify-center gap-2">
            {gateBadge(result.refutation.gate_decision)}
            {result.statistical_significance ? (
              <Badge variant="default">Significant</Badge>
            ) : (
              <Badge variant="secondary">Not significant</Badge>
            )}
          </div>
          <div className="text-xs text-muted-foreground mt-2">
            Refutation: {result.refutation.tests_passed ?? '—'}
            {result.refutation.tests_total !== null && result.refutation.tests_total !== undefined
              ? ` / ${result.refutation.tests_total}`
              : ''}{' '}
            passed
          </div>
        </div>
      </div>

      {result.discovered_confounders && result.discovered_confounders.length > 0 && (
        <p className="text-xs text-muted-foreground">
          Confounders the data identified (adjusted for in the estimate):{' '}
          <span className="font-medium">{result.discovered_confounders.join(', ')}</span>
        </p>
      )}

      <ConfoundingAdjustmentPanel result={result} />

      {vizNodes.length > 0 ? (
        <div className="space-y-2">
          <CausalDiscoveryViz
            nodes={vizNodes}
            edges={vizEdges}
            refutationResults={refutationResults}
            showEffectsTable={false}
          />
          <p className="text-xs text-muted-foreground">
            Confounders point into both treatment and outcome (the backdoor paths the estimate
            adjusts for). A node drawn without edges has no detected causal link to this question.
          </p>
        </div>
      ) : (
        <EmptyState
          title="No DAG produced"
          description="The agent did not return a causal graph for this run."
        />
      )}

      {result.estimator_comparison && (
        <EstimatorComparisonPanel
          comparison={result.estimator_comparison}
          efficiency={result.adjustment_type === 'efficiency'}
        />
      )}

      {clinicalContext.data && (
        <ClinicalContextPanel
          context={clinicalContext.data}
          narrative={narrative}
          narrativeLoading={narrativeLoading}
          labelFor={labelFor}
        />
      )}

      {(result.executive_summary ||
        result.narrative ||
        result.key_insights.length > 0 ||
        result.recommendations.length > 0) && (
        <div className="space-y-4 text-sm">
          {result.executive_summary && <p className="font-medium">{result.executive_summary}</p>}
          {result.narrative && (
            <p className="text-muted-foreground whitespace-pre-line">{result.narrative}</p>
          )}
          {result.key_insights.length > 0 && (
            <div>
              <p className="mb-1 font-medium">Key insights</p>
              <ul className="list-disc space-y-1 pl-5 text-muted-foreground">
                {result.key_insights.map((k, i) => (
                  <li key={i}>{k}</li>
                ))}
              </ul>
            </div>
          )}
          {result.recommendations.length > 0 && (
            <div>
              <p className="mb-1 font-medium">Recommended actions</p>
              <ul className="list-disc space-y-1 pl-5 text-muted-foreground">
                {result.recommendations.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default CausalAnalysisDetail;
