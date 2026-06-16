/**
 * Intervention Impact Page
 * ========================
 *
 * Intervention analysis dashboard. Real, live-wired substrate:
 * - Causal Impact tab: recent recorded causal effect-estimates
 *   (`GET /api/causal/history`) — a real list of completed analyses, NOT a
 *   fabricated per-intervention counterfactual series.
 * - Treatment Effects tab: per cohort × brand ATE / CI / p-value
 *   (`GET /api/causal/treatment-effects`) over the synthetic-gold causal
 *   substrate (patient_journeys + hcp_brand_adoption), run through the real
 *   DoWhy+EconML sequential pipeline.
 * - Segment Analysis tab: real CATE-by-region (`POST /api/segments/analyze`,
 *   EconML/CausalML) for HCP engagement → conversion over business_metrics.
 * - Digital Twin tab: `POST /api/digital-twin/simulate` — run a simulation and
 *   see the real ATE/CI/recommendation; failures surface honestly (onError).
 *
 * Honestly gated (no backend substrate yet — verified against the live
 * OpenAPI spec):
 * - Interventions catalog: no endpoint serves a list of real intervention
 *   programs, so there is no selector. The previous fabricated INTERVENTIONS
 *   catalog (four invented pharma programs) was DELETED.
 * - Before/After tab: no within-subject pre/post endpoint exists, so it
 *   renders an explicit empty state rather than a fabricated comparison.
 *
 * @module pages/InterventionImpact
 */

import { useState, useMemo } from 'react';
import {
  Activity,
  Beaker,
  GitBranch,
  ArrowRight,
  Info,
  FlaskConical,
  Download,
  Loader2,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { EmptyState } from '@/components/ui/EmptyState';
import { SimulationPanel, ScenarioResults, RecommendationCards } from '@/components/digital-twin';
import { useRunSimulation } from '@/hooks/api/use-digital-twin';
import { useCausalAnalysisHistory, useTreatmentEffects } from '@/hooks/api/use-causal';
import { useRunSegmentAnalysisAndWait } from '@/hooks/api/use-segments';
import type { ApiError } from '@/lib/api-client';
import type { CohortName } from '@/types/causal';
import type { CATEResult } from '@/types/segments';
import type { SimulationRequest, SimulationResponse, SimulationRecommendation } from '@/types/digital-twin';
import { RecommendationType, ConfidenceLevel, Recommendation } from '@/types/digital-twin';

// Treatment Effects selector options (the 12 cells = 4 cohorts x 3 brands).
const TE_COHORT_OPTIONS: { value: CohortName; label: string }[] = [
  { value: 'initiation', label: 'Initiation (patient)' },
  { value: 'persistence', label: 'Persistence 180d (patient)' },
  { value: 'discontinuation', label: 'Discontinuation 180d (patient)' },
  { value: 'hcp_adoption', label: 'HCP adoption' },
];
const TE_BRAND_OPTIONS = ['Remibrutinib', 'Fabhalta', 'Kisqali'] as const;

function fmt(value: number | null | undefined, digits = 4): string {
  return value === null || value === undefined ? '—' : value.toFixed(digits);
}

// Local copy of CausalAnalysis.tsx's private formatEffect (not exported there).
// Renders 'N/A' for null/undefined and preserves the sign for negative ATEs.
function formatEffect(effect: number | null | undefined, decimals = 3): string {
  if (effect === null || effect === undefined) return 'N/A';
  return effect.toFixed(decimals);
}

// =============================================================================
// COMPONENT
// =============================================================================

function InterventionImpact() {
  const [simulationResults, setSimulationResults] = useState<SimulationResponse | null>(null);
  // A failed simulation must surface honestly (not look like "never ran").
  const [simulationError, setSimulationError] = useState<ApiError | null>(null);

  // Causal Impact tab: real recorded causal effect-estimates from the causal
  // engine (GET /api/causal/history). NOT a fabricated per-intervention series —
  // no per-intervention counterfactual-series endpoint exists.
  const {
    data: historyData,
    isLoading: historyLoading,
    isError: historyError,
  } = useCausalAnalysisHistory();

  // Treatment Effects selectors. The query is gated behind an explicit Run
  // (teRun) because the endpoint runs a real DoWhy+EconML fit (~5-30s) — we do
  // not want to fire it on every dropdown change.
  const [teCohort, setTeCohort] = useState<CohortName>('persistence');
  const [teBrand, setTeBrand] = useState<string>('Remibrutinib');
  const [teRun, setTeRun] = useState(false);
  const {
    data: teData,
    isFetching: teFetching,
    isError: teIsError,
    error: teError,
  } = useTreatmentEffects(teCohort, teBrand, { enabled: teRun });

  // Segment Analysis tab: real DE-CONFOUNDED CATE-by-region over the live
  // business_metrics per_hcp_rollup substrate (engagement_score -> conversion_rate
  // by region). Gated behind an explicit Run — the agent fits EconML/CausalML
  // (~10-60s). Mirrors the standalone SegmentAnalysis page request EXACTLY.
  //   - effect_modifiers (X) = ['region']: region is the heterogeneity dimension,
  //     so CausalForestDML estimates the treatment effect PER region directly
  //     (rather than post-hoc averaging a market/volume-indexed effect, which is
  //     what swapped the low-effect south/midwest pair in the prior version).
  //   - confounders (W) = market_share + total_rx_count: routed into the DML
  //     nuisance model and residualized out, adjusting for the continuous
  //     confounding. Together these recover the planted ordering NE>W>S>MW with
  //     all four regions significant and tight CIs. (Keeping the continuous
  //     covariates in X too was rejected after adversarial review: it inflates
  //     the per-region CIs ~5.7x and flips the true-positive midwest to "not
  //     significant" — a precision regression on this primary view.)
  const segmentRun = useRunSegmentAnalysisAndWait();
  const runSegmentAnalysis = () => {
    segmentRun.mutate({
      request: {
        query: 'Treatment effect heterogeneity of engagement on conversion by region',
        treatment_var: 'engagement_score',
        outcome_var: 'conversion_rate',
        segment_vars: ['region'],
        effect_modifiers: ['region'],
        confounders: ['market_share', 'total_rx_count'],
        data_source: 'business_metrics',
        filters: { metric_type: 'per_hcp_rollup' },
        n_estimators: 100,
        min_samples_leaf: 10,
        significance_level: 0.05,
      },
    });
  };

  // Digital Twin simulation mutation (real API)
  const { mutate: runSimulation, isPending: isSimulating } = useRunSimulation({
    onSuccess: (data) => {
      setSimulationResults(data);
      setSimulationError(null);
    },
    onError: (err) => {
      // Surface the real failure (503 no-model, 408 timeout, 5xx, network)
      // instead of silently falling back to the empty state.
      setSimulationError(err);
      setSimulationResults(null);
    },
  });

  // Handle simulation request - converts legacy SimulationRequest to SimulateRequest
  const handleSimulate = (request: SimulationRequest) => {
    // Reset stale result/error before dispatching a retry.
    setSimulationError(null);
    setSimulationResults(null);
    runSimulation({
      intervention: {
        intervention_type: request.intervention_type,
        duration_weeks: Math.ceil(request.duration_days / 7),
      },
      brand: request.brand,
      twin_count: request.sample_size,
    });
  };

  // Convert SimulationResponse recommendation to SimulationRecommendation interface
  const simulationRecommendation = useMemo((): SimulationRecommendation | null => {
    if (!simulationResults) return null;

    // Map Recommendation enum to RecommendationType
    const typeMap: Record<Recommendation, RecommendationType> = {
      [Recommendation.DEPLOY]: RecommendationType.DEPLOY,
      [Recommendation.SKIP]: RecommendationType.SKIP,
      [Recommendation.REFINE]: RecommendationType.REFINE,
    };

    // Derive confidence level from simulation_confidence score
    let confidence: ConfidenceLevel;
    if (simulationResults.simulation_confidence >= 0.7) {
      confidence = ConfidenceLevel.HIGH;
    } else if (simulationResults.simulation_confidence >= 0.4) {
      confidence = ConfidenceLevel.MEDIUM;
    } else {
      confidence = ConfidenceLevel.LOW;
    }

    // Build evidence array from real simulation results
    const evidence: string[] = [];
    if (simulationResults.is_significant) {
      evidence.push(`Effect is statistically significant (ATE: ${simulationResults.simulated_ate.toFixed(3)})`);
    }
    if (simulationResults.effect_size_cohens_d) {
      evidence.push(`Effect size (Cohen's d): ${simulationResults.effect_size_cohens_d.toFixed(2)}`);
    }
    if (simulationResults.statistical_power) {
      evidence.push(`Statistical power: ${(simulationResults.statistical_power * 100).toFixed(0)}%`);
    }
    evidence.push(`CI: [${simulationResults.simulated_ci_lower.toFixed(3)}, ${simulationResults.simulated_ci_upper.toFixed(3)}]`);

    return {
      type: typeMap[simulationResults.recommendation],
      confidence,
      rationale: simulationResults.recommendation_rationale,
      evidence,
      risk_factors: simulationResults.fidelity_warning ? [simulationResults.fidelity_warning_reason ?? 'Fidelity warning present'] : undefined,
    };
  }, [simulationResults]);

  // Export: only real simulation output, never fabricated analysis blobs.
  const handleExport = () => {
    if (!simulationResults) return;
    const blob = new Blob([JSON.stringify(simulationResults, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `digital-twin-simulation-${simulationResults.simulation_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">Intervention Impact</h1>
          <p className="text-muted-foreground">
            Before/after comparisons, treatment effects, and counterfactual analysis.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="icon"
            onClick={handleExport}
            disabled={!simulationResults}
            aria-label="Export simulation results"
            title={
              simulationResults
                ? 'Export the latest simulation results as JSON'
                : 'Run a digital-twin simulation to enable export'
            }
          >
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Interventions catalog — honestly gated. The backend exposes no
          interventions-registry endpoint, so there is nothing real to put
          in a selector. The former fabricated catalog was removed. */}
      <div className="mb-8">
        <EmptyState
          title="No intervention catalog available"
          description="The backend does not yet expose an interventions registry, so historical intervention analyses cannot be selected here. Use the Digital Twin tab to pre-screen intervention scenarios against the live twin simulator."
        />
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="causal" className="space-y-6">
        <TabsList>
          <TabsTrigger value="causal" className="gap-2">
            <Activity className="h-4 w-4" />
            Causal Impact
          </TabsTrigger>
          <TabsTrigger value="beforeafter" className="gap-2">
            <ArrowRight className="h-4 w-4" />
            Before/After
          </TabsTrigger>
          <TabsTrigger value="effects" className="gap-2">
            <Beaker className="h-4 w-4" />
            Treatment Effects
          </TabsTrigger>
          <TabsTrigger value="segments" className="gap-2">
            <GitBranch className="h-4 w-4" />
            Segment Analysis
          </TabsTrigger>
          <TabsTrigger value="digital-twin" className="gap-2">
            <FlaskConical className="h-4 w-4" />
            Digital Twin
          </TabsTrigger>
        </TabsList>

        {/* Causal Impact Tab — REAL recorded causal effect-estimates.
            No per-intervention counterfactual-series endpoint exists, so this
            is honestly framed as a table of recent recorded analyses (mirrors
            CausalAnalysis.tsx's History tab) rather than a fictional series. */}
        <TabsContent value="causal" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Recent Causal Analyses
              </CardTitle>
              <CardDescription>
                Recorded causal effect-estimates from the causal engine, newest first
                {historyData ? ` (${historyData.total})` : ''}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {historyLoading ? (
                <div className="py-8 text-center text-sm text-muted-foreground">
                  Loading causal analyses…
                </div>
              ) : historyError ? (
                <EmptyState
                  title="Could not load causal analyses"
                  description="A server error occurred while reading recent analyses. Try refreshing the page."
                />
              ) : historyData && historyData.items.length > 0 ? (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Completed</TableHead>
                      <TableHead>Summary</TableHead>
                      <TableHead>Agent</TableHead>
                      <TableHead className="text-right">ATE</TableHead>
                      <TableHead className="text-right">Confidence</TableHead>
                      <TableHead>Model</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {historyData.items.map((item) => (
                      <TableRow key={item.memory_id}>
                        <TableCell className="whitespace-nowrap text-sm">
                          {new Date(item.occurred_at).toLocaleString()}
                        </TableCell>
                        <TableCell className="max-w-md truncate text-sm">
                          {item.description ?? '—'}
                        </TableCell>
                        <TableCell className="text-sm">{item.agent_name ?? '—'}</TableCell>
                        <TableCell className="text-right text-sm">
                          {formatEffect(item.ate_estimate)}
                        </TableCell>
                        <TableCell className="text-right text-sm">
                          {item.confidence === null || item.confidence === undefined
                            ? 'N/A'
                            : `${(item.confidence * 100).toFixed(0)}%`}
                        </TableCell>
                        <TableCell className="text-sm">{item.model_used ?? '—'}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <EmptyState
                  title="No causal analyses recorded yet"
                  description="Completed causal analyses will appear here as they are run."
                />
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Before/After Tab — no pre/post snapshot endpoint yet */}
        <TabsContent value="beforeafter" className="space-y-6">
          <EmptyState
            title="No before/after data available"
            description="Pre- and post-intervention metric snapshots will appear once a per-intervention analysis endpoint exists."
          />
        </TabsContent>

        {/* Treatment Effects Tab — REAL DoWhy+EconML ATE per (cohort, brand) */}
        <TabsContent value="effects" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Beaker className="h-5 w-5" />
                Treatment Effect by Cohort &amp; Brand
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Selectors + Run. The query is gated on Run so the heavy
                  DoWhy+EconML fit is not fired on every dropdown change. */}
              <div className="flex flex-wrap items-end gap-4">
                <div className="flex flex-col gap-1">
                  <label htmlFor="te-cohort" className="text-sm font-medium">
                    Cohort
                  </label>
                  <select
                    id="te-cohort"
                    value={teCohort}
                    onChange={(e) => {
                      setTeCohort(e.target.value as CohortName);
                      setTeRun(false);
                    }}
                    className="p-2 border rounded-md text-sm bg-background"
                  >
                    {TE_COHORT_OPTIONS.map((c) => (
                      <option key={c.value} value={c.value}>
                        {c.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex flex-col gap-1">
                  <label htmlFor="te-brand" className="text-sm font-medium">
                    Brand
                  </label>
                  <select
                    id="te-brand"
                    value={teBrand}
                    onChange={(e) => {
                      setTeBrand(e.target.value);
                      setTeRun(false);
                    }}
                    className="p-2 border rounded-md text-sm bg-background"
                  >
                    {TE_BRAND_OPTIONS.map((b) => (
                      <option key={b} value={b}>
                        {b}
                      </option>
                    ))}
                  </select>
                </div>
                <Button onClick={() => setTeRun(true)} disabled={teFetching}>
                  {teFetching ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Estimating…
                    </>
                  ) : (
                    'Run estimate'
                  )}
                </Button>
              </div>

              <p className="text-xs text-muted-foreground">
                Runs the live DoWhy + EconML pipeline over the confounded cohort
                (de-confounded backdoor adjustment). A single fit takes ~5-30s.
              </p>

              {/* States: loading / error (503/408/etc.) / result / prompt */}
              {teFetching && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Computing the treatment effect — this runs a real causal fit…
                </div>
              )}

              {!teFetching && teIsError && (
                <EmptyState
                  title="Estimate unavailable"
                  description={
                    teError?.message ??
                    'The estimate could not be computed (the cohort data was unavailable, the compute slot was saturated, or the request timed out). Try again shortly.'
                  }
                />
              )}

              {!teFetching && !teIsError && teData && (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="rounded-lg border p-4">
                      <div className="text-xs text-muted-foreground">ATE</div>
                      <div className="text-2xl font-bold">{fmt(teData.ate)}</div>
                    </div>
                    <div className="rounded-lg border p-4">
                      <div className="text-xs text-muted-foreground">95% CI</div>
                      <div className="text-lg font-semibold">
                        {teData.ci_lower === null || teData.ci_lower === undefined
                          ? '—'
                          : `[${fmt(teData.ci_lower)}, ${fmt(teData.ci_upper)}]`}
                      </div>
                    </div>
                    <div className="rounded-lg border p-4">
                      <div className="text-xs text-muted-foreground">p-value</div>
                      <div className="text-lg font-semibold">
                        {teData.p_value === null || teData.p_value === undefined
                          ? '—'
                          : teData.p_value < 0.001
                            ? '< 0.001'
                            : fmt(teData.p_value, 3)}
                      </div>
                    </div>
                    <div className="rounded-lg border p-4">
                      <div className="text-xs text-muted-foreground">n</div>
                      <div className="text-2xl font-bold">{teData.n.toLocaleString()}</div>
                    </div>
                  </div>

                  <div className="text-sm text-muted-foreground space-y-1">
                    <div>
                      <span className="font-medium text-foreground">Estimator:</span>{' '}
                      {teData.estimator ?? '—'} · <span className="font-medium text-foreground">Method:</span>{' '}
                      {teData.method}
                    </div>
                    <div>
                      <span className="font-medium text-foreground">Treatment:</span>{' '}
                      {teData.treatment_var} → <span className="font-medium text-foreground">Outcome:</span>{' '}
                      {teData.outcome_var}
                    </div>
                    <div>
                      <span className="font-medium text-foreground">Adjusted for:</span>{' '}
                      {teData.confounders.join(', ') || '—'}
                    </div>
                    <div>
                      <span className="font-medium text-foreground">Std. error:</span>{' '}
                      {fmt(teData.std_error)} · <span className="font-medium text-foreground">Latency:</span>{' '}
                      {teData.latency_ms.toLocaleString()} ms
                    </div>
                  </div>

                  {teData.is_synthetic && (
                    <div className="rounded-md border border-amber-300 bg-amber-50 dark:bg-amber-950/30 p-3 text-xs text-amber-800 dark:text-amber-300">
                      Synthetic-gold showcase substrate — values are real estimates over synthetic data.
                    </div>
                  )}

                  {teData.warnings.length > 0 && (
                    <ul className="list-disc pl-5 text-xs text-muted-foreground space-y-1">
                      {teData.warnings.map((w, i) => (
                        <li key={i}>{w}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}

              {!teFetching && !teIsError && !teData && (
                <EmptyState
                  title="Select a cohort and brand, then Run"
                  description="Pick one of the 4 cohorts and 3 brands and click Run estimate to compute a real, de-confounded average treatment effect."
                />
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Segment Analysis Tab — real CATE-by-region from the causal engine over
            the live business_metrics per_hcp_rollup substrate. Not per-intervention:
            it is the engagement->conversion effect, heterogeneous by region. */}
        <TabsContent value="segments" className="space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between gap-4">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <GitBranch className="h-5 w-5" />
                    Segment heterogeneity (CATE by region)
                  </CardTitle>
                  <CardDescription>
                    Conditional treatment effect of HCP engagement on conversion, by
                    region — estimated live with EconML/CausalML over the synthetic-gold
                    per-HCP substrate (de-confounded by market share &amp; total Rx).
                  </CardDescription>
                </div>
                <Button onClick={runSegmentAnalysis} disabled={segmentRun.isPending}>
                  {segmentRun.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Estimating…
                    </>
                  ) : (
                    'Run segment analysis'
                  )}
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {segmentRun.isPending ? (
                <div className="py-8 text-center text-sm text-muted-foreground">
                  Estimating CATE by region (a real EconML/CausalML fit; this can take
                  ~10–60s)…
                </div>
              ) : segmentRun.isError ? (
                <EmptyState
                  title="Segment analysis failed"
                  description={
                    segmentRun.error?.message ??
                    'The CATE engine returned an error. Try running the analysis again.'
                  }
                />
              ) : segmentRun.data ? (
                (() => {
                  const res = segmentRun.data;
                  const regionRows = res.cate_by_segment?.region ?? [];
                  const ciPct =
                    res.confidence_level != null
                      ? `${(res.confidence_level * 100).toFixed(0)}%`
                      : '95%';
                  return (
                    <div className="space-y-6">
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div className="rounded-lg border p-4">
                          <div className="text-sm text-muted-foreground">Overall ATE</div>
                          <div className="text-2xl font-bold">{fmt(res.overall_ate, 3)}</div>
                        </div>
                        <div className="rounded-lg border p-4">
                          <div className="text-sm text-muted-foreground">Heterogeneity</div>
                          <div className="text-2xl font-bold">
                            {res.heterogeneity_score != null
                              ? `${(res.heterogeneity_score * 100).toFixed(0)}%`
                              : '—'}
                          </div>
                        </div>
                      </div>
                      {regionRows.length > 0 ? (
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Region</TableHead>
                              <TableHead className="text-right">CATE</TableHead>
                              <TableHead className="text-right">{ciPct} CI</TableHead>
                              <TableHead className="text-right">n</TableHead>
                              <TableHead className="text-right">Significant</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {regionRows.map((c: CATEResult) => (
                              <TableRow key={c.segment_value}>
                                <TableCell className="font-medium">{c.segment_value}</TableCell>
                                <TableCell className="text-right">
                                  {fmt(c.cate_estimate, 3)}
                                </TableCell>
                                <TableCell className="text-right whitespace-nowrap">
                                  [{fmt(c.cate_ci_lower, 3)}, {fmt(c.cate_ci_upper, 3)}]
                                </TableCell>
                                <TableCell className="text-right">{c.sample_size}</TableCell>
                                <TableCell className="text-right">
                                  {c.statistical_significance ? 'Yes' : 'No'}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      ) : (
                        <EmptyState
                          title="No segment effects returned"
                          description="The analysis completed but produced no per-region CATE rows."
                        />
                      )}
                      {res.warnings && res.warnings.length > 0 && (
                        <ul className="list-disc pl-5 text-xs text-muted-foreground space-y-1">
                          {res.warnings.map((w, i) => (
                            <li key={i}>{w}</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  );
                })()
              ) : (
                <EmptyState
                  title="No segment analysis run yet"
                  description="Click “Run segment analysis” to estimate CATE by region from the live causal engine. The full responder / policy / uplift breakdown is on the Segment Analysis page."
                />
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Digital Twin Tab — REAL substrate, fully wired */}
        <TabsContent value="digital-twin" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Simulation Panel - Left Side */}
            <div className="lg:col-span-1">
              <SimulationPanel
                onSimulate={handleSimulate}
                isSimulating={isSimulating}
                initialBrand="Remibrutinib"
                brands={['Remibrutinib', 'Fabhalta', 'Kisqali']}
              />
            </div>

            {/* Results and Recommendations - Right Side */}
            <div className="lg:col-span-2 space-y-6">
              {/* The real simulation response is threaded through (the
                  former results={null} TODO hid every completed run). */}
              <ScenarioResults
                results={simulationResults}
                isLoading={isSimulating}
                error={simulationError}
              />

              {/* No deployment / refinement / deep-analysis flows exist in
                  the backend yet, so no action callbacks are wired —
                  RecommendationCards hides its action buttons rather than
                  showing dead controls. */}
              <RecommendationCards recommendation={simulationRecommendation} />
            </div>
          </div>

          {/* Digital Twin Context Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Info className="h-5 w-5" />
                About Digital Twin Simulation
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="space-y-2">
                  <h4 className="font-medium text-blue-600">Pre-Screen Interventions</h4>
                  <p className="text-sm text-muted-foreground">
                    Test intervention scenarios virtually before committing real resources.
                    The digital twin models HCP behavior and market dynamics to predict outcomes.
                  </p>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium text-emerald-600">Causal Inference Engine</h4>
                  <p className="text-sm text-muted-foreground">
                    Powered by DoWhy and EconML, the simulation uses causal models trained
                    on historical data to estimate treatment effects and confidence intervals.
                  </p>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium text-amber-600">Fidelity Metrics</h4>
                  <p className="text-sm text-muted-foreground">
                    Each simulation includes fidelity scores indicating how well the model
                    represents your specific market conditions and data coverage.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default InterventionImpact;
