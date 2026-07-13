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

import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  AlertTriangle,
  Beaker,
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
import { StrategicInsightCard } from '@/components/insights';
import { usePageChatContext } from '@/providers/E2ICopilotProvider';
import {
  useCausalHealth,
  useCausalAnalysisHistory,
  useCausalVariables,
  useCausalBrands,
  useDiscoverEffects,
  useCausalDiscoveryInsight,
  useRunCausalAgentAnalysis,
  useEstimators,
  useClinicalContext,
  useTreatmentEffects,
  useTreatmentEffectInsight,
} from '@/hooks/api';
import { getCausalAgentAnalysis } from '@/api/causal';
import type { DiscoveredEffect, CohortName, TreatmentEffectResponse } from '@/types/causal';
import type { TreatmentEffectInsightRequest } from '@/types/insights';

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

// Treatment Effects selector options (the 12 cells = 4 cohorts x 3 brands).
const TE_COHORT_OPTIONS: { value: CohortName; label: string }[] = [
  { value: 'initiation', label: 'Initiation (patient)' },
  { value: 'persistence', label: 'Persistence 180d (patient)' },
  { value: 'discontinuation', label: 'Discontinuation 180d (patient)' },
  { value: 'hcp_adoption', label: 'HCP adoption' },
];
const TE_BRAND_OPTIONS = ['Remibrutinib', 'Fabhalta', 'Kisqali'] as const;

// 4-digit numeric formatter for the treatment-effect readouts ('—' for null).
function fmt(value: number | null | undefined, digits = 4): string {
  return value === null || value === undefined ? '—' : value.toFixed(digits);
}

// The grounded strategic-insight request body for a treatment-effect estimate.
// Built in one place so both the auto-generate effect and the card's manual
// re-generate stay in sync (module-scope → stable identity, no dep-array churn).
function buildTeInsightPayload(d: TreatmentEffectResponse): TreatmentEffectInsightRequest {
  return {
    cohort: d.cohort,
    brand: d.brand,
    treatment_var: d.treatment_var,
    outcome_var: d.outcome_var,
    confounders: d.confounders,
    ate: d.ate,
    ci_lower: d.ci_lower ?? undefined,
    ci_upper: d.ci_upper ?? undefined,
    p_value: d.p_value ?? undefined,
    n: d.n,
    estimator: d.estimator ?? undefined,
  };
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

  // Agentic strategic read of the discovered-effects leaderboard (on-demand LLM
  // interpretation grounded in the real ranked effects).
  const causalInsight = useCausalDiscoveryInsight();

  // ── Manual "Pose your own question" panel ──────────────────────────────────
  const [manualOpen, setManualOpen] = useState(false);
  // Brand-scoped candidates: the offered covariates must match what estimation
  // will actually adjust for (a Fabhalta question is never offered UAS7 — the
  // off-brand gated column is NULL for that cohort and estimation drops it).
  const { data: variables } = useCausalVariables(dataset, brandArg);
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
<<<<<<< HEAD
  // #1188: curated PRE-TREATMENT baselines (RCT grains only). When present the
  // panel offers an OPT-IN variance-reduction adjustment — distinct from the
  // confounders above, which de-bias observational questions.
  const baselineCandidates = useMemo(() => variables?.baseline_candidates ?? [], [variables]);
  const [adjustBaselines, setAdjustBaselines] = useState(false);
  // Opt-in is dataset-scoped: reset on grain switch so a choice made for the
  // Trigger RCT never silently rides into another grain.
  useEffect(() => {
    setAdjustBaselines(false);
  }, [dataset]);
=======
  // Split the adjustment set for display: generic cross-brand confounders vs
  // the selected brand's own indication biomarkers (server-classified via the
  // brand-independent clinical_biomarkers union — no hardcoded column list).
  const biomarkerSet = useMemo(
    () => new Set(variables?.clinical_biomarkers ?? []),
    [variables]
  );
  const genericConfounders = confounders.filter((c) => !biomarkerSet.has(c));
  const brandBiomarkers = confounders.filter((c) => biomarkerSet.has(c));
>>>>>>> origin/main
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
  const resetManual = runAgent.reset;
  // The (dataset, brand) a manual run was SUBMITTED for. The response echoes its
  // dataset but NOT its brand, so we tag the submitted scope ourselves and only
  // surface a result while BOTH still match the active facets. This closes the
  // same-dataset brand-switch race: a run started for brand A that resolves after
  // the user moves to brand B (resetManual cleared the old data, but the late
  // mutation repopulates runAgent.data) is suppressed instead of rendered — and
  // mislabeled — under brand B. Mirrors useDiscoverEffects' scope-tagging.
  const [manualScope, setManualScope] = useState<{ dataset: string; brand: string | null } | null>(
    null
  );
  const manualResult =
    runAgent.data && runAgent.data.dataset === dataset && manualScope?.brand === brandArg
      ? runAgent.data
      : undefined;

  // Both the drilled-into deep view and a completed manual analysis are scoped
  // to (dataset, brand); on a grain/brand switch, drop them so nothing from the
  // previous scope lingers under the new one. (The leaderboard itself resets in
  // useDiscoverEffects.)
  useEffect(() => {
    setSelectedId(null);
    resetManual();
  }, [dataset, brandArg, resetManual]);

  // The brand of the effect currently drilled into (each effect is brand-scoped,
  // even when the run filter is "All brands"); falls back to the filter.
  const selectedEffect = effects.find((e) => e.analysis_id === selectedId);
  // Brand-level clinical context for the leaderboard MoA chip (single brand only).
  const leaderboardContext = useClinicalContext(brandArg, outcomeVar);

  const handleRunManual = async () => {
    // Tag the scope this run is submitted for, so its result can only surface
    // while (dataset, brand) still match (see manualResult above).
    setManualScope({ dataset, brand: brandArg });
    try {
      await runAgent.mutateAsync({
        treatment_var: treatmentVar,
        outcome_var: outcomeVar,
        dataset,
        estimator: estimator === AUTO_ESTIMATOR ? undefined : estimator,
        brand: brandArg ?? undefined,
        // #1188: opt-in RCT baseline adjustment (only meaningful when the
        // dataset offers curated baselines; false otherwise).
        adjust_baselines: adjustBaselines && baselineCandidates.length > 0,
      });
    } catch (error) {
      console.error('Causal agent analysis failed:', error);
    }
  };

  // Treatment Effects tab: per cohort × brand ATE / CI / p-value via the real
  // DoWhy+EconML pipeline. Gated behind an explicit Run (teRun) because each fit
  // takes ~5-30s — do not fire on every dropdown change.
  const [teCohort, setTeCohort] = useState<CohortName>('persistence');
  const [teBrand, setTeBrand] = useState<string>('Remibrutinib');
  const [teRun, setTeRun] = useState(false);
  const {
    data: teData,
    isFetching: teFetching,
    isError: teIsError,
    error: teError,
  } = useTreatmentEffects(teCohort, teBrand, { enabled: teRun });

  // Agentic strategic read of THIS estimate. Auto-generate once a fresh result
  // lands (keyed on the estimate identity so it fires once per distinct result,
  // not on every re-render). Manual re-generate stays available on the card.
  const teInsight = useTreatmentEffectInsight();
  const { mutate: generateTeInsight } = teInsight;
  const teInsightKeyRef = useRef<string | null>(null);
  useEffect(() => {
    if (!teData) return;
    const key = `${teData.cohort}-${teData.brand}-${teData.ate}`;
    if (teInsightKeyRef.current === key) return;
    teInsightKeyRef.current = key;
    generateTeInsight(buildTeInsightPayload(teData));
  }, [teData, generateTeInsight]);

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

  // Publish a compact on-screen data summary so the chat pane can generate
  // opener pills grounded in what this page is showing (usePageChatContext →
  // POST /chat/suggestions page_context).
  const pageChatSummary = useMemo(() => {
    const brandLabel = selectedBrand === ALL_BRANDS ? 'All brands' : selectedBrand;
    const lines: string[] = [
      `Causal Analysis page. Brand filter: ${brandLabel}; dataset: ${dataset}.`,
    ];
    if (effects.length > 0) {
      const top = effects
        .slice(0, 3)
        .map(
          (e) =>
            `${e.treatment} → ${e.outcome} (ATE ${e.ate?.toFixed(3)}, confidence ${e.confidence_score.toFixed(2)})`
        )
        .join('; ');
      lines.push(`Discovered effects: ${effects.length} total. Top: ${top}.`);
    }
    if (teData) {
      lines.push(
        `Treatment-effect estimate on screen: ${teData.treatment_var} → ${teData.outcome_var} for ${teData.brand}, ATE ${teData.ate.toFixed(3)} (n=${teData.n}).`
      );
    }
    return lines.join('\n');
  }, [selectedBrand, dataset, effects, teData]);
  usePageChatContext(pageChatSummary);

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
            (DAG + estimator + refutation gate), and ranks them by confidence and impact. Power
            users can also pose a custom question or estimate a specific cohort&rsquo;s treatment
            effect on demand.
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
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="leaderboard">Validated effects</TabsTrigger>
          <TabsTrigger value="estimators">Estimators</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
          <TabsTrigger value="treatment-effects">Treatment effects</TabsTrigger>
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

          {/* Agentic strategic interpretation of the ranked effects (always
              available; grounds an on-demand LLM read in the real discovered
              effects). */}
          <StrategicInsightCard
            isLoading={causalInsight.isPending}
            error={causalInsight.error?.message ?? null}
            insight={causalInsight.data?.insight}
            keyTakeaways={causalInsight.data?.key_takeaways}
            grounding={causalInsight.data?.grounding}
            isFallback={causalInsight.data?.is_fallback}
            provenance={causalInsight.data?.provenance}
            generatedAt={causalInsight.data?.generated_at}
            onGenerate={() =>
              causalInsight.mutate({
                brand: brandArg ?? 'All brands',
                grain,
                effects: (effects ?? [])
                  .filter((e): e is DiscoveredEffect & { ate: number } => e.ate != null)
                  .map((e) => ({
                    treatment: e.treatment,
                    outcome: e.outcome,
                    ate: e.ate,
                    ate_ci_lower: e.ate_ci_lower ?? undefined,
                    ate_ci_upper: e.ate_ci_upper ?? undefined,
                    status: e.status,
                    selected_estimator: e.selected_estimator ?? undefined,
                  })),
              })
            }
          />

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
                className="flex w-full items-center justify-between text-left bg-transparent hover:bg-transparent text-foreground p-0"
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
                    {'Controlling for — generic confounders (all brands): '}
                    <span className="font-medium">{genericConfounders.join(', ')}</span>
                    {brandBiomarkers.length > 0 && (
                      <>
                        {` · indication-specific biomarkers (${brandArg}): `}
                        <span className="font-medium">{brandBiomarkers.join(', ')}</span>
                      </>
                    )}
                  </p>
                )}
                {baselineCandidates.length > 0 && (
                  <div className="space-y-1">
                    <label className="flex items-start gap-2 text-xs text-muted-foreground">
                      <input
                        type="checkbox"
                        className="mt-0.5"
                        checked={adjustBaselines}
                        onChange={(e) => setAdjustBaselines(e.target.checked)}
                      />
                      <span>
                        Adjust for baseline covariates (variance reduction) —{' '}
                        <span className="font-medium">{baselineCandidates.join(', ')}</span>.
                        Treatment is randomized on this grain, so the point estimate is unbiased
                        either way; pre-treatment baselines only tighten the confidence interval
                        (ANCOVA-style precision, not de-confounding).
                      </span>
                    </label>
                  </div>
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

        {/* Treatment Effects Tab — REAL DoWhy+EconML ATE per (cohort, brand) */}
        <TabsContent value="treatment-effects" className="space-y-6">
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
                (de-confounded backdoor adjustment). A single fit takes ~10-90s.
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

                  {/* Agentic strategic read of THIS estimate (auto-generated
                      when the result lands; grounded in the returned ATE/CI/p/n). */}
                  <StrategicInsightCard
                    title="Strategic insight"
                    description="Agentic interpretation of this treatment-effect estimate, grounded in the returned ATE, CI, p-value, and n."
                    isLoading={teInsight.isPending}
                    error={teInsight.error?.message ?? null}
                    insight={teInsight.data?.insight}
                    keyTakeaways={teInsight.data?.key_takeaways}
                    grounding={teInsight.data?.grounding}
                    isFallback={teInsight.data?.is_fallback}
                    provenance={teInsight.data?.provenance}
                    generatedAt={teInsight.data?.generated_at}
                    onGenerate={() => generateTeInsight(buildTeInsightPayload(teData))}
                  />
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
      </Tabs>
    </div>
  );
}
