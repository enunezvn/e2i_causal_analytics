/**
 * CausalDiscovery Page
 * ====================
 *
 * Page component for causal discovery analysis. Combines:
 *
 * - Library routing form (`useRouteQuery`) — given treatment / outcome /
 *   covariates, recommend the best causal-inference library + alternatives.
 * - Optional KG chain discovery (`useCausalChains`) — find causal chains
 *   in the knowledge graph for the chosen outcome KPI.
 * - Existing DAG visualization (`CausalDiscoveryViz`) — interactive view
 *   of effect estimates and refutation tests.
 *
 * Issue #303 wires this page from a placeholder UI to the live
 * `/api/causal/route` and `/api/graph/causal-chains` endpoints.
 *
 * @module pages/CausalDiscovery
 */

import { useMemo, useState, type FormEvent } from 'react';
import { Brain, FlaskConical, GitBranch, Shield, Loader2 } from 'lucide-react';

import { CausalDiscovery as CausalDiscoveryViz } from '@/components/visualizations/CausalDiscovery';
import type { CausalNode, CausalEdge } from '@/components/visualizations/causal/CausalDAG';
import type { CausalEffect } from '@/components/visualizations/causal/EffectsTable';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { QueryErrorState } from '@/components/ui/query-error-state';

import { useRouteQuery, useRunParallelPipeline } from '@/hooks/api/use-causal';
import { useCausalChains } from '@/hooks/api/use-graph';
import type {
  CausalLibrary,
  ParallelPipelineRequest,
  RouteQueryRequest,
} from '@/types/causal';
import type { CausalChainRequest } from '@/types/graph';

// =============================================================================
// HELPERS
// =============================================================================

/**
 * Parse a comma- or whitespace-separated string of variable names into a
 * trimmed, non-empty list. Returns an empty array when no tokens are present.
 */
function parseCovariates(raw: string): string[] {
  return raw
    .split(/[,\n]/g)
    .map((token) => token.trim())
    .filter((token) => token.length > 0);
}

/** Format a number safely, returning a dash if undefined / null / NaN. */
function fmtNumber(value: unknown, digits: number = 3): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return '-';
  return value.toFixed(digits);
}

/** Format a CI tuple from a per-library result. */
function fmtCI(lower: unknown, upper: unknown): string {
  if (typeof lower !== 'number' || typeof upper !== 'number') return '-';
  return `[${lower.toFixed(3)}, ${upper.toFixed(3)}]`;
}

// =============================================================================
// COMPONENT
// =============================================================================

function CausalDiscovery() {
  // Form state ---------------------------------------------------------------
  const [queryText, setQueryText] = useState('Does X cause Y?');
  const [treatmentVar, setTreatmentVar] = useState('rep_visits');
  const [outcomeVar, setOutcomeVar] = useState('trx_count');
  const [covariatesText, setCovariatesText] = useState('age, region');

  // Hooks --------------------------------------------------------------------
  const {
    mutate: routeMutate,
    data: routeData,
    isPending: isRouting,
    error: routingError,
  } = useRouteQuery();

  const {
    mutate: chainsMutate,
    data: chainsData,
    isPending: isDiscoveringChains,
    error: chainsError,
  } = useCausalChains();

  const {
    mutate: runPipelineMutate,
    data: pipelineData,
    isPending: isRunningPipeline,
    error: pipelineError,
  } = useRunParallelPipeline();

  // Handlers -----------------------------------------------------------------
  const handleRouteSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const covariates = parseCovariates(covariatesText);
    // The backend router (`src/causal_engine/pipeline/router.py`) classifies
    // by question keywords. Use the user-typed question verbatim so the
    // router can branch to EconML / CausalML / NetworkX when the user is
    // asking about heterogeneity / targeting / impact flow rather than ATE.
    const trimmed = queryText.trim();
    const query =
      trimmed.length > 0
        ? trimmed
        : `Effect of ${treatmentVar || '<treatment>'} on ${outcomeVar || '<outcome>'}` +
          (covariates.length > 0
            ? ` controlling for ${covariates.join(', ')}`
            : '');
    const request: RouteQueryRequest = {
      query,
      treatment_var: treatmentVar || undefined,
      outcome_var: outcomeVar || undefined,
      // RouteQueryRequest does not expose `covariates` directly. Pass via
      // `context` so the backend routing layer can use it for downstream
      // pipeline selection without breaking schema.
      context: covariates.length > 0 ? { covariates } : undefined,
    };
    routeMutate(request);
  };

  const handleDiscoverChains = () => {
    const request: CausalChainRequest = {
      kpi_name: outcomeVar || undefined,
      min_confidence: 0.5,
      max_chain_length: 4,
    };
    chainsMutate(request);
  };

  /**
   * Fire a parallel multi-library pipeline. Libraries default to the routing
   * recommendation when present, otherwise the canonical pharma-comm trio
   * (DoWhy / EconML / CausalML). The result populates real effect estimates
   * and CIs in the results table.
   */
  const handleRunPipeline = () => {
    const libraries: CausalLibrary[] = [];
    if (routeData?.primary_library) {
      libraries.push(routeData.primary_library);
    }
    for (const lib of routeData?.secondary_libraries ?? []) {
      if (!libraries.includes(lib)) libraries.push(lib);
    }
    // Parallel pipeline requires 2-4 libraries
    if (libraries.length < 2) {
      const fallback: CausalLibrary[] = [
        'dowhy' as CausalLibrary,
        'econml' as CausalLibrary,
        'causalml' as CausalLibrary,
      ];
      for (const lib of fallback) {
        if (!libraries.includes(lib) && libraries.length < 3) {
          libraries.push(lib);
        }
      }
    }

    const request: ParallelPipelineRequest = {
      treatment_var: treatmentVar,
      outcome_var: outcomeVar,
      covariates: parseCovariates(covariatesText),
      libraries: libraries.slice(0, 4),
    };
    runPipelineMutate({ request, asyncMode: false });
  };

  // Derived ------------------------------------------------------------------
  const primaryLibrary = routeData?.primary_library;
  const secondaryLibraries = routeData?.secondary_libraries ?? [];
  const recommendedEstimators = routeData?.recommended_estimators ?? [];
  const routingConfidencePct =
    routeData?.routing_confidence !== undefined
      ? Math.round(routeData.routing_confidence * 100)
      : null;

  const chains = useMemo(() => chainsData?.chains ?? [], [chainsData]);

  // Backend reports a human-readable caveat when the DoWhy refutation /
  // sensitivity suite did NOT run for this estimate. (Field exists on the
  // wire schema; the local ParallelPipelineResponse type predates it.)
  const robustnessWarning = (
    pipelineData as { robustness_warning?: string | null } | undefined
  )?.robustness_warning;

  // ---------------------------------------------------------------------
  // Real-data threading into the DAG visualization (fix: the viz formerly
  // received NO data props and fell back to a fabricated SAMPLE_ analysis).
  // ---------------------------------------------------------------------

  /** DAG nodes/edges derived from the real KG chain-discovery results. */
  const { vizNodes, vizEdges } = useMemo((): {
    vizNodes: CausalNode[];
    vizEdges: CausalEdge[];
  } => {
    const nodes: CausalNode[] = [];
    const edges: CausalEdge[] = [];
    const seenNodes = new Set<string>();
    const seenEdges = new Set<string>();

    for (const chain of chains) {
      for (const node of chain.nodes) {
        const id = node.id ?? node.name;
        if (!id || seenNodes.has(id)) continue;
        seenNodes.add(id);
        nodes.push({ id, label: node.name ?? id, type: 'variable' });
      }
      for (const rel of chain.relationships ?? []) {
        const edgeId = `${rel.source_id}->${rel.target_id}`;
        if (seenEdges.has(edgeId)) continue;
        seenEdges.add(edgeId);
        edges.push({
          id: edgeId,
          source: rel.source_id,
          target: rel.target_id,
          type: 'causal',
          confidence: rel.confidence,
        });
      }
    }
    return { vizNodes: nodes, vizEdges: edges };
  }, [chains]);

  /**
   * Effect estimates derived from the real parallel-pipeline run:
   * one row per library result plus the consensus row. Fields the API does
   * not return (e.g. p-values) are simply omitted — never invented.
   */
  const vizEffects = useMemo((): CausalEffect[] => {
    if (!pipelineData) return [];
    const effects: CausalEffect[] = [];

    // CI bounds are included ONLY when the backend reports both of them —
    // a missing interval stays missing (rendered as an em dash), never
    // synthesized from the point estimate. No confidence level is labeled
    // either: the pipeline schema carries no confidence_level field, so
    // asserting "95%" would be fabrication.
    const realCI = (
      lower: unknown,
      upper: unknown
    ): { ciLower: number; ciUpper: number } | undefined =>
      typeof lower === 'number' && typeof upper === 'number'
        ? { ciLower: lower, ciUpper: upper }
        : undefined;

    for (const [library, raw] of Object.entries(
      pipelineData.library_results ?? {}
    )) {
      const result = raw as {
        effect_estimate?: number;
        ci_lower?: number;
        ci_upper?: number;
      };
      if (typeof result?.effect_estimate !== 'number') continue;
      effects.push({
        id: `lib-${library}`,
        treatment: treatmentVar,
        outcome: outcomeVar,
        estimate: result.effect_estimate,
        ...realCI(result.ci_lower, result.ci_upper),
        metadata: { library },
      });
    }

    if (typeof pipelineData.consensus_effect === 'number') {
      effects.push({
        id: 'consensus',
        treatment: treatmentVar,
        outcome: outcomeVar,
        estimate: pipelineData.consensus_effect,
        ...realCI(
          pipelineData.consensus_ci_lower,
          pipelineData.consensus_ci_upper
        ),
        metadata: { library: 'consensus' },
      });
    }
    return effects;
  }, [pipelineData, treatmentVar, outcomeVar]);

  return (
    <div className="container mx-auto px-4 py-8 space-y-6">
      {/* ===================================================================
          HEADER
          =================================================================== */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold mb-2">Causal Discovery</h1>
          <p className="text-muted-foreground">
            Causal analysis with DAG visualization, effect estimates, and refutation tests.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="flex items-center gap-1">
            <Brain className="h-3 w-3" />
            DoWhy
          </Badge>
          <Badge variant="outline" className="flex items-center gap-1">
            <FlaskConical className="h-3 w-3" />
            EconML
          </Badge>
          <Badge variant="outline" className="flex items-center gap-1">
            <GitBranch className="h-3 w-3" />
            DAG
          </Badge>
          <Badge variant="outline" className="flex items-center gap-1">
            <Shield className="h-3 w-3" />
            Refutation
          </Badge>
        </div>
      </div>

      {/* ===================================================================
          ROUTING FORM
          =================================================================== */}
      <Card>
        <CardHeader>
          <CardTitle>Library routing</CardTitle>
          <CardDescription>
            Describe a causal question. We will recommend the best library and
            estimator for it.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form
            onSubmit={handleRouteSubmit}
            aria-label="Library routing form"
            className="grid grid-cols-1 md:grid-cols-3 gap-4"
          >
            <div className="space-y-2 md:col-span-3">
              <Label htmlFor="causal-query">Causal question</Label>
              <Input
                id="causal-query"
                value={queryText}
                placeholder='e.g. "Does X cause Y?", "Who should we target?", "How does effect vary?"'
                onChange={(event) => setQueryText(event.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                The router classifies the wording to recommend DoWhy
                (causation), EconML (heterogeneity), CausalML (targeting), or
                NetworkX (impact flow).
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="treatment-var">Treatment variable</Label>
              <Input
                id="treatment-var"
                value={treatmentVar}
                placeholder="e.g. rep_visits"
                onChange={(event) => setTreatmentVar(event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="outcome-var">Outcome variable</Label>
              <Input
                id="outcome-var"
                value={outcomeVar}
                placeholder="e.g. trx_count"
                onChange={(event) => setOutcomeVar(event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="covariates">Covariates (comma-separated)</Label>
              <Input
                id="covariates"
                value={covariatesText}
                placeholder="e.g. age, region, specialty"
                onChange={(event) => setCovariatesText(event.target.value)}
              />
            </div>
            <div className="md:col-span-3 flex flex-wrap items-center gap-3">
              <Button type="submit" disabled={isRouting}>
                {isRouting ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Routing...
                  </>
                ) : (
                  'Run routing'
                )}
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={handleDiscoverChains}
                disabled={isDiscoveringChains || !outcomeVar}
              >
                {isDiscoveringChains ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Discovering...
                  </>
                ) : (
                  'Discover chains in KG'
                )}
              </Button>
              <Button
                type="button"
                variant="secondary"
                onClick={handleRunPipeline}
                disabled={isRunningPipeline || !treatmentVar || !outcomeVar}
              >
                {isRunningPipeline ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Running pipeline...
                  </>
                ) : (
                  'Run parallel pipeline'
                )}
              </Button>
              {isRouting && (
                <span
                  data-testid="routing-loading"
                  className="text-sm text-muted-foreground inline-flex items-center gap-2"
                >
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Routing query...
                </span>
              )}
            </div>
          </form>

          {/* Routing error */}
          <QueryErrorState
            error={routingError}
            onRetry={() =>
              handleRouteSubmit({
                preventDefault: () => undefined,
              } as unknown as FormEvent<HTMLFormElement>)
            }
            isRetrying={isRouting}
            title="Routing failed"
            size="sm"
            className="mt-4"
          />

          {/* KG-chains error */}
          <QueryErrorState
            error={chainsError}
            onRetry={handleDiscoverChains}
            isRetrying={isDiscoveringChains}
            title="KG chain discovery failed"
            size="sm"
            className="mt-2"
          />

          {/* Pipeline error */}
          <QueryErrorState
            error={pipelineError}
            onRetry={handleRunPipeline}
            isRetrying={isRunningPipeline}
            title="Parallel pipeline failed"
            size="sm"
            className="mt-2"
          />
        </CardContent>
      </Card>

      {/* ===================================================================
          ROUTING RESULTS
          =================================================================== */}
      {routeData && (
        <Card>
          <CardHeader>
            <CardTitle>Recommended approach</CardTitle>
            <CardDescription>{routeData.routing_rationale}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium">Primary library:</span>
              <Badge data-testid="routing-primary-library" variant="default">
                {primaryLibrary ?? 'unknown'}
              </Badge>
              {routingConfidencePct !== null && (
                <Badge variant="secondary">
                  {routingConfidencePct}% confidence
                </Badge>
              )}
            </div>
            <div
              data-testid="routing-alternatives"
              className="flex flex-wrap items-center gap-2"
            >
              <span className="text-sm font-medium">Alternatives:</span>
              {secondaryLibraries.length > 0 ? (
                secondaryLibraries.map((library) => (
                  <Badge key={library} variant="outline">
                    {library}
                  </Badge>
                ))
              ) : (
                <span className="text-sm text-muted-foreground">none</span>
              )}
            </div>

            {/* Results table: combines routing recommendations with parallel
                pipeline outputs once available. Effect / CI columns are
                "pending pipeline run" until the pipeline mutation resolves. */}
            <div className="rounded-md border overflow-hidden">
              <table
                data-testid="routing-results-table"
                className="w-full text-sm"
              >
                <thead className="bg-[var(--color-muted)]/50">
                  <tr>
                    <th className="text-left p-2 font-medium">Library</th>
                    <th className="text-left p-2 font-medium">
                      Recommended estimator
                    </th>
                    <th className="text-left p-2 font-medium">
                      Effect estimate
                    </th>
                    {/* The pipeline schema reports raw ci_lower/ci_upper with no
                        confidence-level field — claiming 95% would be fabrication. */}
                    <th className="text-left p-2 font-medium">CI</th>
                    <th className="text-left p-2 font-medium">Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {[primaryLibrary, ...secondaryLibraries]
                    .filter((library): library is CausalLibrary => Boolean(library))
                    .map((library, idx) => {
                      const libraryResult = pipelineData?.library_results?.[
                        library
                      ] as
                        | {
                            effect_estimate?: number;
                            ci_lower?: number;
                            ci_upper?: number;
                            estimator?: string;
                            estimator_type?: string;
                          }
                        | undefined;
                      const isPrimary = idx === 0;
                      // The routing endpoint returns `recommended_estimators`
                      // only for the primary library. Showing them by row
                      // index would mis-label secondary rows (e.g. DoWhy
                      // estimators on the EconML row). Use them for the
                      // primary row only; fall back to whatever estimator the
                      // parallel pipeline actually used for each library.
                      const primaryEstimator =
                        isPrimary && recommendedEstimators.length > 0
                          ? recommendedEstimators.join(', ')
                          : null;
                      const pipelineEstimator =
                        libraryResult?.estimator ??
                        libraryResult?.estimator_type ??
                        null;
                      const estimatorCell =
                        primaryEstimator ?? pipelineEstimator ?? '-';
                      return (
                        <tr key={library} className="border-t">
                          <td className="p-2 font-medium">{library}</td>
                          <td className="p-2">{estimatorCell}</td>
                          <td className="p-2">
                            {libraryResult
                              ? fmtNumber(libraryResult.effect_estimate)
                              : (
                                <span className="text-muted-foreground italic">
                                  pending pipeline run
                                </span>
                              )}
                          </td>
                          <td className="p-2">
                            {libraryResult
                              ? fmtCI(libraryResult.ci_lower, libraryResult.ci_upper)
                              : (
                                <span className="text-muted-foreground italic">
                                  pending pipeline run
                                </span>
                              )}
                          </td>
                          <td className="p-2">
                            {isPrimary && routingConfidencePct !== null
                              ? `${routingConfidencePct}%`
                              : (
                                <span className="text-muted-foreground">
                                  alternative
                                </span>
                              )}
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>

            {/* Consensus row from parallel pipeline */}
            {pipelineData && (
              <div
                data-testid="pipeline-consensus"
                className="rounded-md border p-3 text-sm flex flex-wrap items-center gap-4"
              >
                <span className="font-medium">Consensus effect:</span>
                <Badge variant="default">
                  {fmtNumber(pipelineData.consensus_effect)}
                </Badge>
                <span className="font-medium">CI:</span>
                <span>
                  {fmtCI(
                    pipelineData.consensus_ci_lower,
                    pipelineData.consensus_ci_upper,
                  )}
                </span>
                {pipelineData.library_agreement_score !== undefined && (
                  <>
                    <span className="font-medium">Library agreement:</span>
                    <Badge variant="secondary">
                      {Math.round(pipelineData.library_agreement_score * 100)}%
                    </Badge>
                  </>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* ===================================================================
          KG CHAINS PANEL
          =================================================================== */}
      {chainsData && (
        <Card>
          <CardHeader>
            <CardTitle>Knowledge graph chains</CardTitle>
            <CardDescription>
              Discovered causal chains from the knowledge graph for{' '}
              <span className="font-medium">{outcomeVar || 'the outcome'}</span>
              .
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div data-testid="kg-chains-panel" className="space-y-3">
              {chains.length === 0 && (
                <p className="text-sm text-muted-foreground italic">
                  No chains discovered for this outcome.
                </p>
              )}
              {chains.map((chain, idx) => {
                const totalConfidencePct =
                  chain.total_confidence !== undefined
                    ? Math.round(chain.total_confidence * 100)
                    : null;
                return (
                  <div
                    key={idx}
                    className="rounded-md border p-3 flex flex-col gap-2"
                  >
                    <div className="flex items-center gap-2 flex-wrap">
                      {chain.nodes.map((node, nodeIdx) => (
                        <span
                          key={node.id ?? `${idx}-${nodeIdx}`}
                          className="inline-flex items-center gap-1"
                        >
                          <Badge variant="secondary">{node.name}</Badge>
                          {nodeIdx < chain.nodes.length - 1 && (
                            <span className="text-muted-foreground"></span>
                          )}
                        </span>
                      ))}
                    </div>
                    <div className="text-xs text-muted-foreground flex items-center gap-3">
                      <span>Length: {chain.path_length}</span>
                      {totalConfidencePct !== null && (
                        <span>Confidence: {totalConfidencePct}%</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* ===================================================================
          DAG VISUALIZATION — fed exclusively by the real runs above.
          Nodes/edges come from KG chain discovery; effect rows from the
          parallel pipeline. Refutation details are not returned by the
          pipeline endpoint, so that table stays honestly empty (the
          robustness caveat, when reported, is surfaced as a warning).
          =================================================================== */}
      {robustnessWarning && (
        <div className="rounded-md border border-amber-500/30 bg-amber-500/5 p-3 text-sm text-muted-foreground">
          <span className="font-medium text-amber-600">Robustness caveat:</span>{' '}
          {robustnessWarning}
        </div>
      )}
      <CausalDiscoveryViz
        showControls
        showDetails
        showEffectsTable
        showRefutationTests
        nodes={vizNodes}
        edges={vizEdges}
        effects={vizEffects}
        refutationResults={[]}
        isLoading={isRunningPipeline || isDiscoveringChains}
      />
    </div>
  );
}

export default CausalDiscovery;
