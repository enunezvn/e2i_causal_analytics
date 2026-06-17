/**
 * Causal Analysis Page
 * ====================
 *
 * Agent-driven causal inference. The page leverages the causal_impact agent:
 * the analyst picks a treatment + outcome (data-driven dropdowns from the
 * gold-standard frame) and optionally forces an estimator; the agent then
 * BUILDS the causal DAG, selects an estimator data-drivenly (energy-score
 * routing across the registry) unless one is forced, estimates the
 * treatment->outcome effect, and runs refutation + sensitivity. There are no
 * manual segment / estimator knobs — the engine decides.
 *
 * @module pages/CausalAnalysis
 */

import { useMemo, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { EmptyState } from '@/components/ui/EmptyState';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  useCausalHealth,
  useCausalAnalysisHistory,
  useCausalVariables,
  useRunCausalAgentAnalysis,
  useEstimators,
} from '@/hooks/api';
import { CausalDiscovery as CausalDiscoveryViz } from '@/components/visualizations/CausalDiscovery';
import type { CausalNode, CausalEdge } from '@/components/visualizations/causal/CausalDAG';
import { KPICard } from '@/components/visualizations';
import {
  Play,
  CheckCircle,
  AlertTriangle,
  Activity,
  GitBranch,
  Layers,
  Network,
  TrendingUp,
  Settings,
} from 'lucide-react';

// =============================================================================
// CONSTANTS
// =============================================================================

// The gold-standard causal frame the agent estimates over. Treatment / outcome
// / covariate dropdowns are populated from GET /causal/variables for THIS
// dataset (real columns intersected with the live table). Defaults are real
// columns (treatment_arm -> persistent_180d), NOT the old fictional
// rep_visits / trx_count.
const DATASET = 'patient_journeys';
const DEFAULT_TREATMENT = 'treatment_arm';
const DEFAULT_OUTCOME = 'persistent_180d';

// Estimator selection. "auto" = the agent's data-driven energy-score routing
// across the registry; override values MUST be members of the backend's
// AGENT_FORCEABLE_ESTIMATORS allowlist (schemas/causal.py). Causal Forest is the
// DEFAULT because the gold-standard frame has boosted confounders that OLS
// (Auto's energy-score pick) under-adjusts → it doesn't survive refutation,
// whereas Causal Forest recovers the planted effect ROBUSTLY (gate=proceed).
const AUTO_ESTIMATOR = 'auto';
const DEFAULT_ESTIMATOR = 'CausalForestDML';
const ESTIMATOR_OPTIONS: Array<{ value: string; label: string }> = [
  { value: 'CausalForestDML', label: 'Causal Forest (recommended)' },
  { value: AUTO_ESTIMATOR, label: 'Auto (data-driven)' },
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
  libraries_available: {
    dowhy: false,
    econml: false,
    causalml: false,
    networkx: false,
  },
  estimators_loaded: 0,
  pipeline_orchestrator_ready: false,
  hierarchical_analyzer_ready: false,
  analysis_count_24h: 0,
  average_latency_ms: null as number | null,
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function formatEffect(effect: number | null | undefined, decimals = 3): string {
  if (effect === null || effect === undefined) return 'N/A';
  return effect.toFixed(decimals);
}

function formatCI(lower: number | null | undefined, upper: number | null | undefined): string {
  if (lower === null || lower === undefined || upper === null || upper === undefined) return 'N/A';
  return `[${lower.toFixed(3)}, ${upper.toFixed(3)}]`;
}

/** Badge for the agent run status (completed / needs_review / failed). */
function statusBadge(status: string | undefined) {
  if (status === 'completed') return <Badge variant="default">Completed</Badge>;
  if (status === 'needs_review') return <Badge variant="secondary">Needs review</Badge>;
  return <Badge variant="destructive">Failed</Badge>;
}

/** Badge for the refutation robustness gate. */
function gateBadge(gate: string | null | undefined) {
  if (gate === 'proceed') return <Badge variant="default">Proceed</Badge>;
  if (gate === 'review') return <Badge variant="secondary">Review</Badge>;
  if (gate === 'block') return <Badge variant="destructive">Blocked</Badge>;
  return <Badge variant="secondary">Not run</Badge>;
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function CausalAnalysis() {
  const [treatmentVar, setTreatmentVar] = useState(DEFAULT_TREATMENT);
  const [outcomeVar, setOutcomeVar] = useState(DEFAULT_OUTCOME);
  const [estimator, setEstimator] = useState(DEFAULT_ESTIMATOR);
  const [selectedLibrary, setSelectedLibrary] = useState<string>('all');

  // API hooks
  const { data: healthData } = useCausalHealth();
  const {
    data: historyData,
    isLoading: historyLoading,
    isError: historyError,
  } = useCausalAnalysisHistory();
  // Real treatment / outcome / covariate candidates for the dropdowns (curated
  // causally-meaningful columns intersected with the live schema).
  const { data: variables } = useCausalVariables(DATASET);
  // The 12-estimator registry powers the Estimators tab AND tells the analyst
  // what Auto routes across.
  const {
    data: estimatorsData,
    isLoading: estimatorsLoading,
    isError: estimatorsError,
  } = useEstimators();
  const runAgent = useRunCausalAgentAnalysis();
  const result = runAgent.data;

  const health = healthData ?? DEFAULT_HEALTH;
  const estimators = estimatorsData?.estimators ?? [];
  const visibleEstimators = estimators.filter(
    (e) => selectedLibrary === 'all' || e.library === selectedLibrary
  );

  const treatmentCandidates = variables?.treatment_candidates ?? [DEFAULT_TREATMENT];
  const outcomeCandidates = variables?.outcome_candidates ?? [DEFAULT_OUTCOME];
  // The agent controls for these confounders (data-driven from the dataset's
  // curated covariates); a variable can never control for itself.
  const confounders = useMemo(
    () =>
      (variables?.covariate_candidates ?? []).filter(
        (c) => c !== treatmentVar && c !== outcomeVar
      ),
    [variables, treatmentVar, outcomeVar]
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

  // Map the agent's DAG onto the shared causal-graph visualization. The
  // treatment->outcome edge carries the estimated effect.
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
        type: 'causal',
        ...(isEffectEdge && result.ate !== null && result.ate !== undefined
          ? { effect: result.ate }
          : {}),
      };
    });
    return { vizNodes: nodes, vizEdges: edges };
  }, [result]);

  const handleRunAnalysis = async () => {
    try {
      await runAgent.mutateAsync({
        treatment_var: treatmentVar,
        outcome_var: outcomeVar,
        dataset: DATASET,
        // Omit covariates -> the backend uses the dataset's curated confounders.
        estimator: estimator === AUTO_ESTIMATOR ? undefined : estimator,
      });
    } catch (error) {
      // Error surfaced via runAgent.isError below; nothing to fabricate.
      console.error('Causal agent analysis failed:', error);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold text-foreground flex items-center gap-2">
            <GitBranch className="h-8 w-8" />
            Causal Analysis
          </h1>
          <p className="text-muted-foreground mt-1">
            Agent-driven causal inference — builds the DAG and estimates the treatment&rarr;outcome
            effect
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={handleRunAnalysis} disabled={runAgent.isPending}>
            <Play className="mr-2 h-4 w-4" />
            {runAgent.isPending ? 'Running…' : 'Run Analysis'}
          </Button>
        </div>
      </div>

      {/* Service Health Banner */}
      {health.status === 'healthy' ? (
        <Alert className="mb-6 border-green-200 bg-green-50">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertTitle className="text-green-800">Causal Engine Healthy</AlertTitle>
          <AlertDescription className="text-green-700">
            All {Object.values(health.libraries_available).filter(Boolean).length} causal libraries
            available. {health.analysis_count_24h} analyses completed in the last 24 hours.
          </AlertDescription>
        </Alert>
      ) : (
        <Alert variant="destructive" className="mb-6">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Service Issue</AlertTitle>
          <AlertDescription>
            Some causal libraries may be unavailable. Check service health for details.
          </AlertDescription>
        </Alert>
      )}

      {/* Run failure — surfaced honestly. The agent is fail-closed: it estimates
          on real data and never fabricates an effect. */}
      {runAgent.isError && (
        <Alert variant="destructive" className="mb-6">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Analysis could not run</AlertTitle>
          <AlertDescription>
            {runAgent.error?.message ? `${runAgent.error.message} ` : ''}
            The causal agent is fail-closed: it estimates on real gold-standard data and will not
            fabricate an effect. Try a different treatment / outcome pairing, or check the engine
            health above.
          </AlertDescription>
        </Alert>
      )}

      {/* Running — the agent's energy-score selection + refutation runs server-side
          (submit -> poll), which takes a minute or two. */}
      {runAgent.isPending && (
        <Alert className="mb-6 border-blue-200 bg-blue-50">
          <Activity className="h-4 w-4 text-blue-600 animate-pulse" />
          <AlertTitle className="text-blue-800">Analyzing…</AlertTitle>
          <AlertDescription className="text-blue-700">
            The agent is building the causal DAG, selecting an estimator across the registry, and
            running robustness checks. This can take a minute or two.
          </AlertDescription>
        </Alert>
      )}

      {/* Overview Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
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

      {/* Main Content Tabs */}
      <Tabs defaultValue="analysis" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
          <TabsTrigger value="estimators">Estimators</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        {/* Analysis Tab */}
        <TabsContent value="analysis" className="space-y-6">
          {/* Configuration */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Analysis Configuration
              </CardTitle>
              <CardDescription>
                Pick a treatment and outcome; the agent builds the DAG and selects the estimator.
                Segmentation and method are decided by the engine, not set by hand.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Treatment Variable</label>
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
                  <label className="text-sm font-medium mb-2 block">Outcome Variable</label>
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
                  <label className="text-sm font-medium mb-2 block">Estimator</label>
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
                <p className="text-xs text-muted-foreground mt-4">
                  Controlling for (confounders, data-driven):{' '}
                  <span className="font-medium">{confounders.join(', ')}</span>
                </p>
              )}
            </CardContent>
          </Card>

          {/* Results / empty state */}
          {!result ? (
            <EmptyState
              title="No analysis run yet"
              description="Choose a treatment and outcome, then click Run Analysis. The agent will build the causal DAG, estimate the effect, and run robustness checks."
            />
          ) : (
            <>
              {/* Status / warnings */}
              {result.status !== 'completed' && (
                <Alert variant={result.status === 'failed' ? 'destructive' : 'default'}>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>
                    {result.status === 'failed'
                      ? 'Analysis did not produce a validated effect'
                      : 'Estimate needs expert review'}
                  </AlertTitle>
                  <AlertDescription>
                    {result.warnings.length > 0
                      ? result.warnings.join(' ')
                      : 'The estimate did not fully pass the robustness gate.'}
                  </AlertDescription>
                </Alert>
              )}

              {/* Effect summary */}
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
                      <div className="text-4xl font-bold text-primary">
                        {formatEffect(result.ate)}
                      </div>
                      <div className="text-sm text-muted-foreground mt-2">
                        95% CI: {formatCI(result.ate_ci_lower, result.ate_ci_upper)}
                      </div>
                      <div className="mt-4 flex items-center justify-center gap-2">
                        {statusBadge(result.status)}
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
                    <CardDescription>
                      {estimator === AUTO_ESTIMATOR
                        ? 'Selected data-drivenly (energy-score)'
                        : 'Estimator used for this run'}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="text-center">
                      <div className="text-2xl font-bold capitalize">
                        {result.selected_estimator
                          ? result.selected_estimator.replace(/_/g, ' ')
                          : 'N/A'}
                      </div>
                      {result.confidence !== null && result.confidence !== undefined && (
                        <div className="text-sm text-muted-foreground mt-2">
                          Confidence: {(result.confidence * 100).toFixed(0)}%
                        </div>
                      )}
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
                    <CardDescription>Refutation &amp; sensitivity</CardDescription>
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

              {/* Causal DAG */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Network className="h-5 w-5" />
                    Causal DAG
                  </CardTitle>
                  <CardDescription>
                    The structure the agent built and adjusted for. The treatment&rarr;outcome edge
                    carries the estimated effect.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {vizNodes.length > 0 ? (
                    <CausalDiscoveryViz nodes={vizNodes} edges={vizEdges} showEffectsTable={false} />
                  ) : (
                    <EmptyState
                      title="No DAG produced"
                      description="The agent did not return a causal graph for this run."
                    />
                  )}
                </CardContent>
              </Card>

              {/* Interpretation */}
              {(result.narrative || result.executive_summary || result.recommendations.length > 0) && (
                <Card>
                  <CardHeader>
                    <CardTitle>Interpretation</CardTitle>
                    <CardDescription>Natural-language reading of the estimate</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4 text-sm">
                    {result.executive_summary && (
                      <p className="font-medium">{result.executive_summary}</p>
                    )}
                    {result.narrative && (
                      <p className="text-muted-foreground whitespace-pre-line">
                        {result.narrative}
                      </p>
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
