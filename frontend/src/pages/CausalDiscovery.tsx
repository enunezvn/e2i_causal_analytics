/**
 * CausalDiscovery Page
 * ====================
 *
 * Agent-driven causal discovery. The analyst picks ONLY the causal question
 * (treatment -> outcome); the `causal_impact` agent does everything else:
 *
 * 1. LEARNS the causal structure FROM THE DATA via guided structure discovery
 *    (PC with background-knowledge tiers anchoring treatment as cause / outcome
 *    as effect) — the data selects which covariates are confounders.
 * 2. Estimates the treatment -> outcome effect with a data-driven estimator
 *    (energy-score routing across the registry).
 * 3. Runs refutation + sensitivity (the robustness gate).
 *
 * This replaced the previous manual workbench (library routing + parallel
 * pipeline + KG-chain buttons) — those exposed agent-internal method choices as
 * user decisions. The dedicated agent now makes them.
 *
 * @module pages/CausalDiscovery
 */

import { useMemo, useState } from 'react';
import { GitBranch, Network, Play, Loader2, AlertTriangle, Sparkles } from 'lucide-react';

import { CausalDiscovery as CausalDiscoveryViz } from '@/components/visualizations/CausalDiscovery';
import type { CausalNode, CausalEdge } from '@/components/visualizations/causal/CausalDAG';
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

import { useCausalVariables, useRunCausalAgentAnalysis } from '@/hooks/api';

const DATASET = 'patient_journeys';
const DEFAULT_TREATMENT = 'treatment_arm';
const DEFAULT_OUTCOME = 'persistent_180d';

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

/** Honest label for how the DAG was built. */
function dagSourceLabel(source?: string): { label: string; learned: boolean } {
  switch (source) {
    case 'discovered':
      return { label: 'Learned from data', learned: true };
    case 'augmented':
      return { label: 'Domain model + data-discovered edges', learned: true };
    default:
      return { label: 'Domain-knowledge model', learned: false };
  }
}

export default function CausalDiscovery() {
  const [treatmentVar, setTreatmentVar] = useState(DEFAULT_TREATMENT);
  const [outcomeVar, setOutcomeVar] = useState(DEFAULT_OUTCOME);

  const { data: variables } = useCausalVariables(DATASET);
  const runAgent = useRunCausalAgentAnalysis();
  const result = runAgent.data;

  const treatmentCandidates = variables?.treatment_candidates ?? [DEFAULT_TREATMENT];
  const outcomeCandidates = variables?.outcome_candidates ?? [DEFAULT_OUTCOME];

  // Map the agent's learned DAG onto the shared causal-graph visualization.
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

  const handleAnalyze = async () => {
    try {
      await runAgent.mutateAsync({
        treatment_var: treatmentVar,
        outcome_var: outcomeVar,
        dataset: DATASET,
        // Learn the DAG from data (default), and let the agent pick the
        // estimator (Auto). The analyst supplies only the question.
      });
    } catch (error) {
      // Surfaced via runAgent.isError below; nothing fabricated.
      console.error('Causal discovery failed:', error);
    }
  };

  const source = dagSourceLabel(result?.dag_source);

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
            The agent learns the causal structure from the data, then estimates and validates the
            treatment&rarr;outcome effect — you choose only the question.
          </p>
        </div>
        <Badge variant="outline" className="flex items-center gap-1 self-start">
          <Sparkles className="h-3 w-3" />
          Agent-driven
        </Badge>
      </div>

      {/* Question form */}
      <Card>
        <CardHeader>
          <CardTitle>Causal question</CardTitle>
          <CardDescription>
            Pick a treatment and an outcome. The agent discovers the confounders, builds the DAG,
            selects the estimator, and runs robustness checks — no method knobs to set.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
            <div className="space-y-2">
              <label className="text-sm font-medium block" htmlFor="treatment-var">
                Treatment (cause)
              </label>
              <Select value={treatmentVar} onValueChange={setTreatmentVar}>
                <SelectTrigger id="treatment-var" aria-label="Treatment variable">
                  <SelectValue placeholder="Select treatment" />
                </SelectTrigger>
                <SelectContent>
                  {treatmentCandidates.map((candidate) => (
                    <SelectItem key={candidate} value={candidate}>
                      {candidate}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium block" htmlFor="outcome-var">
                Outcome (effect)
              </label>
              <Select value={outcomeVar} onValueChange={setOutcomeVar}>
                <SelectTrigger id="outcome-var" aria-label="Outcome variable">
                  <SelectValue placeholder="Select outcome" />
                </SelectTrigger>
                <SelectContent>
                  {outcomeCandidates.map((candidate) => (
                    <SelectItem key={candidate} value={candidate}>
                      {candidate}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button onClick={handleAnalyze} disabled={runAgent.isPending}>
              {runAgent.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Discovering &amp; analyzing…
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Discover &amp; Analyze
                </>
              )}
            </Button>
          </div>
          {runAgent.isPending && (
            <p className="text-sm text-muted-foreground mt-4">
              The agent is learning the DAG from the data, estimating the effect, and running
              refutation. This can take a minute or two.
            </p>
          )}
          {runAgent.isError && (
            <Alert variant="destructive" className="mt-4">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Discovery could not run</AlertTitle>
              <AlertDescription>
                The causal agent did not return a result. Please try again.
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Results / empty state */}
      {!result ? (
        <EmptyState
          title="No discovery run yet"
          description="Choose a treatment and outcome, then click Discover & Analyze. The agent learns the causal graph from the data, estimates the effect, and runs robustness checks."
        />
      ) : (
        <>
          {result.status !== 'completed' && (
            <Alert variant={result.status === 'failed' ? 'destructive' : 'default'}>
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>
                {result.status === 'failed'
                  ? 'No validated effect was produced'
                  : 'Estimate needs expert review'}
              </AlertTitle>
              <AlertDescription>
                {result.warnings.length > 0
                  ? result.warnings.join(' ')
                  : 'The estimate did not fully pass the robustness gate.'}
              </AlertDescription>
            </Alert>
          )}

          {/* Learned causal graph — the heart of the discovery page. */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Network className="h-5 w-5" />
                Causal structure
              </CardTitle>
              <CardDescription className="flex flex-wrap items-center gap-2">
                <Badge variant={source.learned ? 'default' : 'outline'}>{source.label}</Badge>
                {result.discovered_confounders && result.discovered_confounders.length > 0 && (
                  <span className="text-xs">
                    Confounders the data identified:{' '}
                    <span className="font-medium">
                      {result.discovered_confounders.join(', ')}
                    </span>
                  </span>
                )}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {vizNodes.length > 0 ? (
                <CausalDiscoveryViz nodes={vizNodes} edges={vizEdges} showEffectsTable={false} />
              ) : (
                <EmptyState
                  title="No graph produced"
                  description="The agent did not return a causal graph for this run."
                />
              )}
            </CardContent>
          </Card>

          {/* Effect + estimator + robustness */}
          <div className="grid md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>
                  {result.treatment_var} &rarr; {result.outcome_var}
                </CardTitle>
                <CardDescription>Average Treatment Effect</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center">
                  <div className="text-4xl font-bold text-primary">{formatEffect(result.ate)}</div>
                  <div className="text-sm text-muted-foreground mt-2">
                    95% CI: {formatCI(result.ate_ci_lower, result.ate_ci_upper)}
                  </div>
                  <div className="mt-4 flex items-center justify-center gap-2">
                    {result.statistical_significance ? (
                      <Badge variant="default">Significant</Badge>
                    ) : (
                      <Badge variant="secondary">Not significant</Badge>
                    )}
                  </div>
                  {result.p_value !== null && result.p_value !== undefined && (
                    <div className="text-xs text-muted-foreground mt-2">
                      p = {result.p_value.toFixed(4)}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Estimator</CardTitle>
                <CardDescription>Selected data-drivenly (energy-score)</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center">
                  <div className="text-2xl font-bold capitalize">
                    {result.selected_estimator
                      ? result.selected_estimator.replace(/_/g, ' ')
                      : 'N/A'}
                  </div>
                  <div className="text-xs text-muted-foreground mt-3">
                    Ran on {result.n_rows.toLocaleString()} rows ({result.data_source}) in{' '}
                    {(result.latency_ms / 1000).toFixed(1)}s
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Robustness</CardTitle>
                <CardDescription>Refutation &amp; sensitivity gate</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Gate:</span>
                    {gateBadge(result.refutation.gate_decision)}
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Tests passed:</span>
                    <span className="font-medium">
                      {result.refutation.tests_passed ?? '—'}
                      {result.refutation.tests_total !== null &&
                      result.refutation.tests_total !== undefined
                        ? ` / ${result.refutation.tests_total}`
                        : ''}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Sensitivity E-value:</span>
                    <span className="font-medium">
                      {result.refutation.sensitivity_e_value !== null &&
                      result.refutation.sensitivity_e_value !== undefined
                        ? result.refutation.sensitivity_e_value.toFixed(2)
                        : '—'}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Interpretation */}
          {(result.narrative ||
            result.executive_summary ||
            result.recommendations.length > 0) && (
            <Card>
              <CardHeader>
                <CardTitle>Interpretation</CardTitle>
                <CardDescription>Natural-language reading of the result</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 text-sm">
                {result.executive_summary && (
                  <p className="font-medium">{result.executive_summary}</p>
                )}
                {result.narrative && (
                  <p className="text-muted-foreground whitespace-pre-line">{result.narrative}</p>
                )}
                {result.recommendations.length > 0 && (
                  <div>
                    <p className="font-medium mb-1">Recommendations</p>
                    <ul className="list-disc pl-5 text-muted-foreground space-y-1">
                      {result.recommendations.map((rec, i) => (
                        <li key={i}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
