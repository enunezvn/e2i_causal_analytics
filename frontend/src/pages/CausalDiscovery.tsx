/**
 * CausalDiscovery Page — validated-effects leaderboard
 * ====================================================
 *
 * The page SURFACES the causal effects the causal_impact agent can validate from
 * the data and RANKS them by confidence (robustness gate + significance) and
 * impact (effect size). The analyst clicks "Discover causal effects"; the agent
 * runs its full pipeline (guided DAG discovery + data-driven estimator +
 * refutation gate) for each candidate question, and the ranked leaderboard fills
 * in progressively. Click any validated row to drill into its DAG + refutation.
 *
 * No question-picking, no method knobs — the agent decides; the analyst reads the
 * ranked, validated results.
 *
 * @module pages/CausalDiscovery
 */

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  GitBranch,
  Network,
  Play,
  Loader2,
  AlertTriangle,
  Sparkles,
  ChevronRight,
} from 'lucide-react';

import { CausalDiscovery as CausalDiscoveryViz } from '@/components/visualizations/CausalDiscovery';
import type { CausalNode, CausalEdge } from '@/components/visualizations/causal/CausalDAG';
import type {
  RefutationResult,
  RefutationMethod,
} from '@/components/visualizations/causal/RefutationTests';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { EmptyState } from '@/components/ui/EmptyState';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

import { useDiscoverEffects, useCausalBrands } from '@/hooks/api';
import { getCausalAgentAnalysis } from '@/api/causal';
import type { DiscoveredEffect, RefutationTestDetail } from '@/types/causal';

const DATASET = 'patient_journeys';

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

// One column conveys both the run state and the robustness verdict. A computed
// effect is shown by its gate (Proceed/Review/Blocked); a run that produced no
// estimate is Failed; in-flight rows are Queued/Running.
function verdictBadge(e: DiscoveredEffect) {
  switch (e.status) {
    case 'completed':
      return <Badge variant="default">Proceed</Badge>;
    case 'needs_review':
      return <Badge variant="secondary">Review</Badge>;
    case 'blocked':
      return <Badge variant="destructive">Blocked</Badge>;
    case 'running':
      return (
        <Badge variant="outline" className="gap-1">
          <Loader2 className="h-3 w-3 animate-spin" /> Running
        </Badge>
      );
    case 'pending':
      return <Badge variant="outline">Queued</Badge>;
    default:
      return <Badge variant="destructive">Failed</Badge>;
  }
}

// "All brands" sentinel — the Select needs a non-empty value; null is sent to the API.
const ALL_BRANDS = '__all__';

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

// Map the agent's per-test refutation results onto the table's row shape. These
// are REAL refuter outputs surfaced from the response — never fabricated.
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

export default function CausalDiscovery() {
  const [selectedBrand, setSelectedBrand] = useState<string>(ALL_BRANDS);
  const brandArg = selectedBrand === ALL_BRANDS ? null : selectedBrand;
  const brandsQuery = useCausalBrands(DATASET);
  const { start, isStarting, startError, job } = useDiscoverEffects(DATASET, brandArg);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Drill-down: the full validated analysis for the selected leaderboard row.
  const detail = useQuery({
    queryKey: ['causal', 'agent-analyze', selectedId],
    queryFn: () => getCausalAgentAnalysis(selectedId as string),
    enabled: !!selectedId,
  });
  const result = detail.data;

  const effects: DiscoveredEffect[] = useMemo(() => job?.effects ?? [], [job]);
  const running = !!job && job.status !== 'completed';

  // Map the selected analysis's DAG onto the shared causal-graph visualization.
  const { vizNodes, vizEdges } = useMemo((): {
    vizNodes: CausalNode[];
    vizEdges: CausalEdge[];
  } => {
    if (!result) return { vizNodes: [], vizEdges: [] };
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

  // Per-test refutation results for the drill-down table (placebo / random common
  // cause / data subset / bootstrap). Empty when refutation did not run.
  const refutationResults = useMemo(
    () => toRefutationResults(result?.refutation?.tests),
    [result]
  );

  return (
    <div className="container mx-auto px-4 py-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
            <GitBranch className="h-8 w-8" />
            Causal Discovery
          </h1>
          <p className="text-muted-foreground">
            The agent surfaces the causal effects it can validate from the data and ranks them by
            confidence and impact. No questions to pick — the agent decides.
          </p>
        </div>
        <Badge variant="outline" className="flex items-center gap-1 self-start">
          <Sparkles className="h-3 w-3" />
          Agent-driven
        </Badge>
      </div>

      {/* Run control */}
      <Card>
        <CardHeader>
          <CardTitle>Discovered causal effects</CardTitle>
          <CardDescription>
            For each candidate question the agent builds the DAG, selects the estimator
            data-drivenly, and runs the refutation gate — then ranks the validated effects. This
            takes a few minutes; results fill in as each completes.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <label
                htmlFor="brand-select"
                className="text-sm font-medium text-muted-foreground"
              >
                Brand
              </label>
              <Select
                value={selectedBrand}
                onValueChange={setSelectedBrand}
                disabled={isStarting || running}
              >
                <SelectTrigger id="brand-select" className="w-48" aria-label="Brand">
                  <SelectValue placeholder="All brands" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={ALL_BRANDS}>All brands</SelectItem>
                  {(brandsQuery.data?.brands ?? []).map((b) => (
                    <SelectItem key={b} value={b}>
                      {b}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button onClick={() => start()} disabled={isStarting || running}>
              {isStarting || running ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  {running && job
                    ? `Validating… (${job.completed}/${job.total})`
                    : 'Starting…'}
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  {job ? 'Re-run discovery' : 'Discover causal effects'}
                </>
              )}
            </Button>
            {job && (
              <span className="text-sm text-muted-foreground">
                {job.completed}/{job.total} questions validated
              </span>
            )}
          </div>
          {startError && (
            <Alert variant="destructive" className="mt-4">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Discovery could not start</AlertTitle>
              <AlertDescription>Please try again.</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Leaderboard */}
      {!job ? (
        <EmptyState
          title="No discovery run yet"
          description="Click Discover causal effects. The agent validates each candidate question and ranks the effects by confidence (robustness gate + significance) and impact (effect size)."
        />
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>Ranked causal effects</CardTitle>
            <CardDescription>
              Validated by the agent (discovered DAG + estimator + refutation gate), ranked by
              confidence then impact. Click a validated row for its DAG and robustness detail.
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b bg-muted/40 text-left text-xs uppercase text-muted-foreground">
                  <tr>
                    <th className="p-3 font-medium">#</th>
                    <th className="p-3 font-medium">Causal question</th>
                    <th className="p-3 font-medium">Confidence</th>
                    <th className="p-3 font-medium">Impact (ATE)</th>
                    <th className="p-3 font-medium">95% CI</th>
                    <th className="p-3 font-medium">Estimator</th>
                    <th className="p-3" />
                  </tr>
                </thead>
                <tbody>
                  {effects.map((e, i) => {
                    const clickable =
                      e.status === 'completed' ||
                      e.status === 'needs_review' ||
                      e.status === 'blocked';
                    const isSelected = !!e.analysis_id && e.analysis_id === selectedId;
                    return (
                      <tr
                        key={`${e.treatment}->${e.outcome}`}
                        className={`border-b last:border-0 ${
                          clickable ? 'cursor-pointer hover:bg-muted/40' : 'opacity-80'
                        } ${isSelected ? 'bg-muted/60' : ''}`}
                        onClick={() => {
                          if (clickable && e.analysis_id) setSelectedId(e.analysis_id);
                        }}
                      >
                        <td className="p-3 text-muted-foreground">{i + 1}</td>
                        <td className="p-3 font-medium">
                          <span>{e.treatment}</span>{' '}
                          <span className="text-muted-foreground">&rarr;</span>{' '}
                          <span>{e.outcome}</span>
                        </td>
                        <td className="p-3">{verdictBadge(e)}</td>
                        <td className="p-3 font-medium">{formatEffect(e.ate)}</td>
                        <td className="p-3 text-muted-foreground">
                          {formatCI(e.ate_ci_lower, e.ate_ci_upper)}
                        </td>
                        <td className="p-3 capitalize">
                          {e.selected_estimator ? e.selected_estimator.replace(/_/g, ' ') : '—'}
                        </td>
                        <td className="p-3 text-right">
                          {clickable && (
                            <ChevronRight className="inline h-4 w-4 text-muted-foreground" />
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Drill-down detail for the selected effect */}
      {selectedId && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Network className="h-5 w-5" />
              {result ? (
                <>
                  {result.treatment_var} &rarr; {result.outcome_var}
                </>
              ) : (
                'Loading effect…'
              )}
            </CardTitle>
            <CardDescription>
              The agent&apos;s validated causal model: discovered DAG, estimated effect, and
              robustness gate.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {detail.isLoading || !result ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" /> Loading the validated analysis…
              </div>
            ) : (
              <>
                <div className="grid md:grid-cols-3 gap-6">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-primary">{formatEffect(result.ate)}</div>
                    <div className="text-sm text-muted-foreground mt-1">
                      ATE · 95% CI {formatCI(result.ate_ci_lower, result.ate_ci_upper)}
                    </div>
                    {result.p_value !== null && result.p_value !== undefined && (
                      <div className="text-xs text-muted-foreground mt-1">
                        p = {result.p_value.toFixed(4)}
                      </div>
                    )}
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-semibold capitalize">
                      {result.selected_estimator
                        ? result.selected_estimator.replace(/_/g, ' ')
                        : 'N/A'}
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
                      {result.refutation.tests_total !== null &&
                      result.refutation.tests_total !== undefined
                        ? ` / ${result.refutation.tests_total}`
                        : ''}{' '}
                      passed
                    </div>
                  </div>
                </div>

                {result.discovered_confounders && result.discovered_confounders.length > 0 && (
                  <p className="text-xs text-muted-foreground">
                    Confounders the data identified:{' '}
                    <span className="font-medium">{result.discovered_confounders.join(', ')}</span>
                  </p>
                )}

                {vizNodes.length > 0 ? (
                  <CausalDiscoveryViz
                    nodes={vizNodes}
                    edges={vizEdges}
                    refutationResults={refutationResults}
                    showEffectsTable={false}
                  />
                ) : (
                  <EmptyState
                    title="No DAG produced"
                    description="The agent did not return a causal graph for this run."
                  />
                )}

                {(result.executive_summary || result.narrative) && (
                  <div className="space-y-2 text-sm">
                    {result.executive_summary && (
                      <p className="font-medium">{result.executive_summary}</p>
                    )}
                    {result.narrative && (
                      <p className="text-muted-foreground whitespace-pre-line">
                        {result.narrative}
                      </p>
                    )}
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
