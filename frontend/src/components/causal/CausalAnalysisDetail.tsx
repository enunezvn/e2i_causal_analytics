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

import { useMemo } from 'react';

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
    description: t.details ?? undefined,
  }));
}

// The agent fits and energy-scores several estimators and picks the lowest score
// with a robust-over-fast tie-break. Surface that evaluation so the analyst sees
// WHAT was compared and WHY the winner won — not just the winner's name.
function EstimatorComparisonPanel({ comparison }: { comparison: EstimatorComparison }) {
  const ranked = [...comparison.candidates].sort((a, b) => {
    if (a.energy_score == null) return 1;
    if (b.energy_score == null) return -1;
    return a.energy_score - b.energy_score;
  });
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-sm font-medium">Estimator selection (data-driven)</p>
        <p className="text-xs text-muted-foreground">
          {comparison.n_succeeded}/{comparison.n_evaluated} estimators fit · lower energy score is
          better
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
                className={`border-b last:border-0 ${c.is_selected ? 'bg-muted/50 font-medium' : ''}`}
              >
                <td className="p-2 capitalize">
                  {c.estimator.replace(/_/g, ' ')}
                  {c.is_selected && (
                    <Badge variant="default" className="ml-2 align-middle">
                      Selected
                    </Badge>
                  )}
                </td>
                <td className="p-2">{c.energy_score != null ? c.energy_score.toFixed(4) : '—'}</td>
                <td className="p-2">{c.ate != null ? c.ate.toFixed(4) : '—'}</td>
                <td className="p-2 text-xs text-muted-foreground">
                  {c.success ? 'fit' : `failed${c.error ? `: ${c.error}` : ''}`}
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

/** The shared deep view for one validated causal effect (agent-analyze result). */
export function CausalAnalysisDetail({ result }: { result: AgentCausalAnalysisResponse }) {
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
        <EstimatorComparisonPanel comparison={result.estimator_comparison} />
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
