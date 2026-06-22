/**
 * Causal Analysis Page — unified, agent-led
 * =========================================
 *
 * ONE page (the former /causal-discovery + /causal-analysis collapsed). The
 * LANDING is the validated-effects leaderboard: the causal_impact agent
 * proposes the causal questions from the gold-standard SSOT (no empty form),
 * validates each (guided DAG discovery + data-driven estimator + refutation
 * gate), and ranks them by confidence then impact. Facets: grain (Patient /
 * HCP / Trigger) + brand. Each row surfaces its brand + plain-language summary
 * and drills into the deep view (DAG + per-test refutation + estimator
 * comparison + interpretation).
 *
 * A secondary "Pose your own question" panel keeps the manual treatment /
 * outcome / brand / estimator path (sourced from GET /causal/variables) for
 * power users; its result renders through the SAME deep view.
 *
 * @module pages/CausalAnalysis
 */

import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  ChevronRight,
  GitBranch,
  Layers,
  Loader2,
  Network,
  Play,
  Settings,
  Sparkles,
  TrendingUp,
} from 'lucide-react';

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
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { KPICard } from '@/components/visualizations';
import { CausalAnalysisDetail } from '@/components/causal/CausalAnalysisDetail';
import {
  useCausalHealth,
  useCausalAnalysisHistory,
  useCausalVariables,
  useCausalBrands,
  useDiscoverEffects,
  useRunCausalAgentAnalysis,
  useEstimators,
  useClinicalContext,
} from '@/hooks/api';
import { getCausalAgentAnalysis } from '@/api/causal';
import type { DiscoveredEffect } from '@/types/causal';

// =============================================================================
// CONSTANTS
// =============================================================================

// Each grain is a `dataset` the agent estimates over. All three are live now —
// the backend specs/loaders and the SSOT `causal_paths` rows (6 HCP + 6 Trigger,
// all 3 brands) shipped, so their discover-effects leaderboards are non-empty.
// `ready` is kept as the extensibility gate: a future grain added with
// `ready: false` is shown disabled + "(coming soon)" (honest, not faked) until
// its data lands.
interface GrainOption {
  value: string;
  dataset: string;
  label: string;
  ready: boolean;
}
const GRAINS: GrainOption[] = [
  { value: 'patient', dataset: 'patient_journeys', label: 'Patient', ready: true },
  { value: 'hcp', dataset: 'hcp_adoption', label: 'HCP', ready: true },
  { value: 'trigger', dataset: 'nba_triggers', label: 'Trigger', ready: true },
];

// "All brands" sentinel — the Select needs a non-empty value; null is sent to the API.
const ALL_BRANDS = '__all__';

// Estimator selection for the manual panel. "auto" = the agent's data-driven
// energy-score routing — the DEFAULT. The override is an expert escape hatch;
// values MUST be members of the backend's AGENT_FORCEABLE_ESTIMATORS allowlist.
const AUTO_ESTIMATOR = 'auto';
const ESTIMATOR_OPTIONS: Array<{ value: string; label: string }> = [
  { value: AUTO_ESTIMATOR, label: 'Auto — agent decides (recommended)' },
  { value: 'CausalForestDML', label: 'Causal Forest — EconML' },
  { value: 'LinearDML', label: 'Linear DML — EconML' },
  { value: 'drlearner', label: 'DR-Learner — EconML' },
  { value: 'ols', label: 'Linear Regression (OLS)' },
  { value: 'propensity_score_weighting', label: 'Propensity Score Weighting — DoWhy' },
];

const LIBRARY_COLORS: Record<string, string> = {
  dowhy: '#3b82f6',
  econml: '#8b5cf6',
  causalml: '#06b6d4',
  networkx: '#f59e0b',
};

const DEFAULT_HEALTH = {
  status: 'unknown',
  libraries_available: { dowhy: false, econml: false, causalml: false, networkx: false },
  estimators_loaded: 0,
  pipeline_orchestrator_ready: false,
  hierarchical_analyzer_ready: false,
  analysis_count_24h: 0,
  average_latency_ms: null as number | null,
};

// =============================================================================
// HELPERS
// =============================================================================

function formatEffect(ate: number | null | undefined): string {
  if (ate === null || ate === undefined || Number.isNaN(ate)) return 'N/A';
  return ate.toFixed(4);
}

function formatCI(lower?: number | null, upper?: number | null): string {
  if (lower === null || lower === undefined || upper === null || upper === undefined) return '—';
  return `[${lower.toFixed(3)}, ${upper.toFixed(3)}]`;
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

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function CausalAnalysis() {
  // ── Facets ────────────────────────────────────────────────────────────────
  const [grain, setGrain] = useState<string>('patient');
  const activeGrain = GRAINS.find((g) => g.value === grain) ?? GRAINS[0];
  const dataset = activeGrain.dataset;
  const [selectedBrand, setSelectedBrand] = useState<string>(ALL_BRANDS);
  const brandArg = selectedBrand === ALL_BRANDS ? null : selectedBrand;

  // ── Leaderboard (landing): the agent's validated, ranked effects ───────────
  const brandsQuery = useCausalBrands(dataset);
  const { start, isStarting, startError, job } = useDiscoverEffects(dataset, brandArg);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const detail = useQuery({
    queryKey: ['causal', 'agent-analyze', selectedId],
    queryFn: () => getCausalAgentAnalysis(selectedId as string),
    enabled: !!selectedId,
  });
  const detailResult = detail.data;
  const effects: DiscoveredEffect[] = useMemo(() => job?.effects ?? [], [job]);
  const running = !!job && job.status !== 'completed';

  // A drilled-into effect belongs to the leaderboard that produced it; that
  // leaderboard is scoped to (dataset, brand). When the grain or brand changes,
  // the open deep view no longer matches the new scope — close it so we never
  // show e.g. a Patient analysis under the HCP grain (the leaderboard itself is
  // reset in useDiscoverEffects).
  useEffect(() => {
    setSelectedId(null);
  }, [dataset, brandArg]);

  // ── Manual "Pose your own question" panel ──────────────────────────────────
  const [manualOpen, setManualOpen] = useState(false);
  const { data: variables } = useCausalVariables(dataset);
  const treatmentCandidates = useMemo(
    () => variables?.treatment_candidates ?? ['treatment_arm'],
    [variables]
  );
  const outcomeCandidates = useMemo(
    () => variables?.outcome_candidates ?? ['persistent_180d'],
    [variables]
  );
  const [treatmentVar, setTreatmentVar] = useState('treatment_arm');
  const [outcomeVar, setOutcomeVar] = useState('persistent_180d');
  const [estimator, setEstimator] = useState(AUTO_ESTIMATOR);
  const confounders = useMemo(
    () =>
      (variables?.covariate_candidates ?? []).filter(
        (c) => c !== treatmentVar && c !== outcomeVar
      ),
    [variables, treatmentVar, outcomeVar]
  );
  // Keep the manual panel's treatment/outcome valid for the active dataset. The
  // candidate sets are dataset-specific (e.g. HCP's only outcome is `adopted`,
  // Trigger's treatments are control_group_flag/acceptance_status), so the
  // Patient-grain defaults become invalid on a grain switch. Clamp any stale
  // selection to the first valid candidate once the new dataset's variables load
  // — otherwise a manual run submits a column the backend allowlist rejects (400).
  useEffect(() => {
    if (!variables) return;
    if (treatmentCandidates.length && !treatmentCandidates.includes(treatmentVar)) {
      setTreatmentVar(treatmentCandidates[0]);
    }
    if (outcomeCandidates.length && !outcomeCandidates.includes(outcomeVar)) {
      setOutcomeVar(outcomeCandidates[0]);
    }
  }, [variables, treatmentCandidates, outcomeCandidates, treatmentVar, outcomeVar]);

  const runAgent = useRunCausalAgentAnalysis();
  const manualResult = runAgent.data;

  // The brand of the effect currently drilled into (each effect is brand-scoped,
  // even when the run filter is "All brands"); falls back to the filter.
  const selectedEffect = effects.find((e) => e.analysis_id === selectedId);
  // Brand-level clinical context for the leaderboard MoA chip (single brand only).
  const leaderboardContext = useClinicalContext(brandArg, outcomeVar);

  const handleRunManual = async () => {
    try {
      await runAgent.mutateAsync({
        treatment_var: treatmentVar,
        outcome_var: outcomeVar,
        dataset,
        estimator: estimator === AUTO_ESTIMATOR ? undefined : estimator,
        brand: brandArg ?? undefined,
      });
    } catch (error) {
      console.error('Causal agent analysis failed:', error);
    }
  };

  // ── Estimators + History tabs ──────────────────────────────────────────────
  const { data: healthData } = useCausalHealth();
  const {
    data: historyData,
    isLoading: historyLoading,
    isError: historyError,
  } = useCausalAnalysisHistory();
  const {
    data: estimatorsData,
    isLoading: estimatorsLoading,
    isError: estimatorsError,
  } = useEstimators();
  const [selectedLibrary, setSelectedLibrary] = useState<string>('all');
  const health = healthData ?? DEFAULT_HEALTH;
  const estimators = estimatorsData?.estimators ?? [];
  const visibleEstimators = estimators.filter(
    (e) => selectedLibrary === 'all' || e.library === selectedLibrary
  );
  const overviewMetrics = useMemo(() => {
    const availableLibraries = Object.values(health.libraries_available).filter(Boolean).length;
    const totalLibraries = Object.keys(health.libraries_available).length;
    return {
      librariesAvailable: `${availableLibraries}/${totalLibraries}`,
      estimatorsLoaded: estimatorsData?.total ?? health.estimators_loaded,
      analysisCount: health.analysis_count_24h,
      avgLatency: health.average_latency_ms
        ? `${(health.average_latency_ms / 1000).toFixed(1)}s`
        : 'N/A',
    };
  }, [health, estimatorsData]);

  return (
    <div className="container mx-auto px-4 py-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
            <GitBranch className="h-8 w-8" />
            Causal Analysis
          </h1>
          <p className="text-muted-foreground">
            The agent proposes the causal questions from the gold-standard data, validates each
            (DAG + estimator + refutation gate), and ranks them by confidence and impact. No empty
            form — the agent decides; you read the ranked, validated results.
          </p>
        </div>
        <Badge variant="outline" className="flex items-center gap-1 self-start">
          <Sparkles className="h-3 w-3" />
          Agent-driven
        </Badge>
      </div>

      {/* Service Health Banner */}
      {health.status === 'healthy' ? (
        <Alert className="border-green-200 bg-green-50">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertTitle className="text-green-800">Causal Engine Healthy</AlertTitle>
          <AlertDescription className="text-green-700">
            All {Object.values(health.libraries_available).filter(Boolean).length} causal libraries
            available. {health.analysis_count_24h} analyses completed in the last 24 hours.
          </AlertDescription>
        </Alert>
      ) : (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Service Issue</AlertTitle>
          <AlertDescription>
            Some causal libraries may be unavailable. Check service health for details.
          </AlertDescription>
        </Alert>
      )}

      {/* Overview Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KPICard
          title="Libraries"
          value={overviewMetrics.librariesAvailable}
          icon={<Layers className="h-5 w-5" />}
        />
        <KPICard
          title="Estimators"
          value={overviewMetrics.estimatorsLoaded}
          icon={<Settings className="h-5 w-5" />}
        />
        <KPICard
          title="Analyses (24h)"
          value={overviewMetrics.analysisCount}
          icon={<Activity className="h-5 w-5" />}
        />
        <KPICard
          title="Avg Latency"
          value={overviewMetrics.avgLatency}
          icon={<TrendingUp className="h-5 w-5" />}
        />
      </div>

      <Tabs defaultValue="leaderboard" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="leaderboard">Validated effects</TabsTrigger>
          <TabsTrigger value="estimators">Estimators</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        {/* Leaderboard Tab (landing) */}
        <TabsContent value="leaderboard" className="space-y-6">
          {/* Facets + run control */}
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
                  <label htmlFor="grain-select" className="text-sm font-medium text-muted-foreground">
                    Grain
                  </label>
                  <Select value={grain} onValueChange={setGrain} disabled={isStarting || running}>
                    <SelectTrigger id="grain-select" className="w-44" aria-label="Grain">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {GRAINS.map((g) => (
                        <SelectItem key={g.value} value={g.value} disabled={!g.ready}>
                          {g.label}
                          {g.ready ? '' : ' (coming soon)'}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-center gap-2">
                  <label htmlFor="brand-select" className="text-sm font-medium text-muted-foreground">
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
                        <th className="p-3 font-medium">Brand</th>
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
                            key={`${e.treatment}->${e.outcome}->${e.brand ?? 'all'}`}
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
                              {e.summary && (
                                <div className="mt-1 max-w-md text-xs font-normal text-muted-foreground">
                                  {e.summary}
                                </div>
                              )}
                            </td>
                            <td className="p-3 text-muted-foreground">{e.brand ?? 'All'}</td>
                            <td className="p-3">{verdictBadge(e)}</td>
                            <td className="p-3 font-medium">{formatEffect(e.ate)}</td>
                            <td className="p-3 text-muted-foreground">
                              {formatCI(e.ate_ci_lower, e.ate_ci_upper)}
                            </td>
                            <td className="p-3 capitalize">
                              {e.selected_estimator ? e.selected_estimator.replace(/_/g, ' ') : '—'}
                            </td>
                            <td className="p-3 text-right">
                              <div className="flex items-center justify-end gap-2">
                                {e.clinical_context?.competitor_landscape &&
                                  e.clinical_context.competitor_landscape.count > 0 && (
                                    <Badge
                                      variant="outline"
                                      className="text-xs font-normal text-muted-foreground"
                                      title={e.clinical_context.competitor_landscape.competitors.join(', ')}
                                    >
                                      {e.clinical_context.competitor_landscape.count}{' '}
                                      {e.clinical_context.competitor_landscape.count === 1
                                        ? 'rival'
                                        : 'rivals'}
                                    </Badge>
                                  )}
                                {clickable && (
                                  <ChevronRight className="inline h-4 w-4 text-muted-foreground" />
                                )}
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="border-t px-3 py-3 text-xs text-muted-foreground">
                  Why these {effects.length} question{effects.length === 1 ? '' : 's'}? The agent
                  only proposes the treatment&rarr;outcome relationships this grain&rsquo;s
                  gold-standard causal spec defines, scoped per brand, and collapses complementary
                  outcomes (e.g. &ldquo;discontinued&rdquo; is the inverse of
                  &ldquo;persistent&rdquo;) and self-pairs. Clinical markers (eGFR, LDH, …) are
                  designated adjustment covariates, not treatments or outcomes, so they enter the
                  model as confounders rather than as questions.
                </p>
                {brandArg && leaderboardContext.data && (
                  <p className="border-t px-3 py-2 text-xs text-muted-foreground">
                    <span className="font-medium">{leaderboardContext.data.drug_name}</span> (
                    {brandArg}) —{' '}
                    <span className="font-medium">
                      {leaderboardContext.data.mechanism.mechanism_of_action}
                    </span>{' '}
                    {leaderboardContext.data.mechanism.source === 'chembl'
                      ? '(mechanism from ChEMBL)'
                      : '(mechanism from a curated clinical reference)'}
                    . Estimates run on a synthetic cohort; open a row for the full sourced clinical
                    context.
                  </p>
                )}
              </CardContent>
            </Card>
          )}

          {/* Drill-down detail for the selected effect */}
          {selectedId && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Network className="h-5 w-5" />
                  {detailResult ? (
                    <>
                      {detailResult.treatment_var} &rarr; {detailResult.outcome_var}
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
              <CardContent>
                {detail.isError ? (
                  <Alert variant="destructive">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertTitle>Could not load this analysis</AlertTitle>
                    <AlertDescription>
                      Its result may have expired (analyses are kept for about an hour). Re-run
                      discovery to regenerate it.
                    </AlertDescription>
                  </Alert>
                ) : detail.isLoading || !detailResult ? (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" /> Loading the validated analysis…
                  </div>
                ) : (
                  <CausalAnalysisDetail result={detailResult} brand={selectedEffect?.brand ?? brandArg} />
                )}
              </CardContent>
            </Card>
          )}

          {/* Secondary: Pose your own question (manual path) */}
          <Card>
            <CardHeader>
              <button
                type="button"
                onClick={() => setManualOpen((o) => !o)}
                className="flex w-full items-center justify-between text-left"
                aria-expanded={manualOpen}
              >
                <div>
                  <CardTitle>Pose your own question</CardTitle>
                  <CardDescription>
                    Power users: pick a treatment + outcome from the gold-standard frame and run the
                    agent on that single hypothesis. Segmentation and method are decided by the
                    engine, not set by hand.
                  </CardDescription>
                </div>
                <ChevronRight
                  className={`h-5 w-5 shrink-0 text-muted-foreground transition-transform ${manualOpen ? 'rotate-90' : ''}`}
                />
              </button>
            </CardHeader>
            {manualOpen && (
              <CardContent className="space-y-4">
                <div className="grid md:grid-cols-3 gap-4">
                  <div>
                    <label className="text-sm font-medium mb-2 block">Treatment variable</label>
                    <Select value={treatmentVar} onValueChange={setTreatmentVar}>
                      <SelectTrigger aria-label="Treatment variable">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {treatmentCandidates.map((c) => (
                          <SelectItem key={c} value={c}>
                            {c}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-medium mb-2 block">Outcome variable</label>
                    <Select value={outcomeVar} onValueChange={setOutcomeVar}>
                      <SelectTrigger aria-label="Outcome variable">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {outcomeCandidates.map((c) => (
                          <SelectItem key={c} value={c}>
                            {c}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-medium mb-2 block">
                      Estimator <span className="text-muted-foreground">(optional override)</span>
                    </label>
                    <Select value={estimator} onValueChange={setEstimator}>
                      <SelectTrigger aria-label="Estimator">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {ESTIMATOR_OPTIONS.map((opt) => (
                          <SelectItem key={opt.value} value={opt.value}>
                            {opt.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                {confounders.length > 0 && (
                  <p className="text-xs text-muted-foreground">
                    Controlling for (confounders, data-driven):{' '}
                    <span className="font-medium">{confounders.join(', ')}</span>
                  </p>
                )}
                <div className="flex items-center gap-3">
                  <Button onClick={handleRunManual} disabled={runAgent.isPending}>
                    <Play className="mr-2 h-4 w-4" />
                    {runAgent.isPending ? 'Running…' : 'Run analysis'}
                  </Button>
                  <span className="text-xs text-muted-foreground">
                    Scoped to: {brandArg ?? 'All brands'} · {activeGrain.label} grain
                  </span>
                </div>

                {runAgent.isError && (
                  <Alert variant="destructive">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertTitle>Analysis could not run</AlertTitle>
                    <AlertDescription>
                      {runAgent.error?.message ? `${runAgent.error.message} ` : ''}
                      The causal agent is fail-closed: it estimates on real gold-standard data and
                      will not fabricate an effect. Try a different treatment / outcome pairing.
                    </AlertDescription>
                  </Alert>
                )}

                {runAgent.isPending && (
                  <Alert className="border-blue-200 bg-blue-50">
                    <Activity className="h-4 w-4 text-blue-600 animate-pulse" />
                    <AlertTitle className="text-blue-800">Analyzing…</AlertTitle>
                    <AlertDescription className="text-blue-700">
                      The agent is building the causal DAG, selecting an estimator across the
                      registry, and running robustness checks. This can take a minute or two.
                    </AlertDescription>
                  </Alert>
                )}

                {manualResult && <CausalAnalysisDetail result={manualResult} brand={brandArg} />}
              </CardContent>
            )}
          </Card>
        </TabsContent>

        {/* Estimators Tab */}
        <TabsContent value="estimators" className="space-y-6">
          <div className="flex gap-4 mb-4">
            <Select value={selectedLibrary} onValueChange={setSelectedLibrary}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Filter by library" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Libraries</SelectItem>
                <SelectItem value="econml">EconML</SelectItem>
                <SelectItem value="causalml">CausalML</SelectItem>
                <SelectItem value="dowhy">DoWhy</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {estimatorsError ? (
            <EmptyState
              icon={<AlertTriangle className="h-8 w-8" aria-hidden="true" />}
              title="Estimator registry unavailable"
              description="Could not load the estimator registry from /causal/estimators. Try refreshing."
            />
          ) : estimatorsLoading ? (
            <EmptyState
              icon={<Activity className="h-8 w-8" aria-hidden="true" />}
              title="Loading estimators…"
              description="Fetching the supported estimator registry."
            />
          ) : visibleEstimators.length === 0 ? (
            <EmptyState
              icon={<GitBranch className="h-8 w-8" aria-hidden="true" />}
              title="No estimators for this library"
              description="No registered estimators match the selected library filter."
            />
          ) : (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {visibleEstimators.map((est) => (
                <Card key={est.name} className="hover:shadow-md transition-shadow">
                  <CardHeader className="pb-2">
                    <div className="flex justify-between items-start">
                      <CardTitle className="text-base capitalize">
                        {est.name.replace(/_/g, ' ')}
                      </CardTitle>
                      <Badge
                        style={{ backgroundColor: LIBRARY_COLORS[est.library] }}
                        className="text-white"
                      >
                        {est.library}
                      </Badge>
                    </div>
                    <CardDescription>{est.estimator_type}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <p className="text-sm text-muted-foreground">{est.description}</p>
                    <div className="flex gap-4 text-sm">
                      <div className="flex items-center gap-1">
                        {est.supports_confidence_intervals ? (
                          <CheckCircle className="h-4 w-4 text-green-500" />
                        ) : (
                          <AlertTriangle className="h-4 w-4 text-gray-400" />
                        )}
                        <span>CI</span>
                      </div>
                      <div className="flex items-center gap-1">
                        {est.supports_heterogeneous_effects ? (
                          <CheckCircle className="h-4 w-4 text-green-500" />
                        ) : (
                          <AlertTriangle className="h-4 w-4 text-gray-400" />
                        )}
                        <span>HTE</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Analysis History</CardTitle>
              <CardDescription>
                Recent completed causal analyses, newest first
                {historyData ? ` (${historyData.total})` : ''}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {historyLoading ? (
                <div className="py-8 text-center text-sm text-muted-foreground">
                  Loading analysis history…
                </div>
              ) : historyError ? (
                <EmptyState
                  title="Could not load analysis history"
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
                  title="No analyses recorded yet"
                  description="Completed causal analyses will appear here as they are run."
                />
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
