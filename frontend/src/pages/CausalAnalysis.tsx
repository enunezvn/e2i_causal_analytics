/**
 * Causal Analysis Page
 * ====================
 *
 * Causal inference dashboard for multi-library analysis.
 * Supports hierarchical CATE, pipeline execution, cross-validation,
 * and library routing.
 *
 * @module pages/CausalAnalysis
 */

import { useState, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ErrorBar,
  Cell,
} from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { EmptyState } from '@/components/ui/EmptyState';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  useCausalHealth,
  useRunHierarchicalAnalysis,
} from '@/hooks/api';
import {
  CausalAnalysisStatus,
  EstimatorType,
  SegmentationMethod,
} from '@/types/causal';
import { KPICard } from '@/components/visualizations';
import {
  RefreshCw,
  Play,
  CheckCircle,
  AlertTriangle,
  Activity,
  GitBranch,
  Layers,
  BarChart3,
  Network,
  TrendingUp,
  Settings,
} from 'lucide-react';

// =============================================================================
// DEFAULTS
// =============================================================================
// F-002 fix: this page no longer ships fabricated SAMPLE_* analysis results.
// API hook results are surfaced when available; otherwise the page renders
// explicit empty states. The list of supported estimators below is a
// configuration constant (display-only metadata), NOT analysis output.

const SUPPORTED_ESTIMATORS = [
  { name: 'Causal Forest', library: 'econml', type: 'CATE', supports_ci: true, supports_hte: true },
  { name: 'Linear DML', library: 'econml', type: 'CATE', supports_ci: true, supports_hte: true },
  { name: 'X-Learner', library: 'econml', type: 'Meta-learner', supports_ci: true, supports_hte: true },
  { name: 'Uplift Random Forest', library: 'causalml', type: 'Uplift', supports_ci: true, supports_hte: true },
  { name: 'Propensity Score Matching', library: 'dowhy', type: 'Identification', supports_ci: true, supports_hte: false },
  { name: 'Instrumental Variable', library: 'dowhy', type: 'Identification', supports_ci: true, supports_hte: false },
];

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
// CONSTANTS
// =============================================================================

const COLORS = {
  primary: '#3b82f6',
  secondary: '#8b5cf6',
  tertiary: '#06b6d4',
  success: '#10b981',
  warning: '#f59e0b',
  error: '#ef4444',
};

const LIBRARY_COLORS: Record<string, string> = {
  dowhy: '#3b82f6',
  econml: '#8b5cf6',
  causalml: '#06b6d4',
  networkx: '#f59e0b',
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function getStatusBadge(status: CausalAnalysisStatus) {
  const variants: Record<CausalAnalysisStatus, 'default' | 'secondary' | 'destructive'> = {
    [CausalAnalysisStatus.COMPLETED]: 'default',
    [CausalAnalysisStatus.RUNNING]: 'secondary',
    [CausalAnalysisStatus.PENDING]: 'secondary',
    [CausalAnalysisStatus.FAILED]: 'destructive',
  };
  return (
    <Badge variant={variants[status]} className="capitalize">
      {status}
    </Badge>
  );
}

function formatEffect(effect: number | null | undefined, decimals: number = 3): string {
  if (effect === null || effect === undefined) return 'N/A';
  return effect.toFixed(decimals);
}

function formatCI(lower: number | null | undefined, upper: number | null | undefined): string {
  if (lower === null || lower === undefined || upper === null || upper === undefined) return 'N/A';
  return `[${lower.toFixed(3)}, ${upper.toFixed(3)}]`;
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function CausalAnalysis() {
  const [selectedLibrary, setSelectedLibrary] = useState<string>('all');
  const [treatmentVar, setTreatmentVar] = useState('rep_visits');
  const [outcomeVar, setOutcomeVar] = useState('trx_count');
  const [nSegments, setNSegments] = useState(4);

  // API hooks
  const { data: healthData } = useCausalHealth();
  const runAnalysisMutation = useRunHierarchicalAnalysis();

  // Use API data or fall back to neutral defaults (no fabricated SAMPLE_).
  const health = healthData ?? DEFAULT_HEALTH;
  // Hierarchical result + library comparison come ONLY from API. When the
  // mutation has not run (or has not yet returned), these are `undefined`
  // and the render path below surfaces explicit empty states.
  const hierarchicalResult = runAnalysisMutation.data;
  const libraryComparison: Array<{
    library: string;
    effect: number | null;
    ci_lower: number | null;
    ci_upper: number | null;
    latency: number;
    note?: string;
  }> = []; // F-002: no longer fabricated; surfaced via API when available

  // Calculate overview metrics
  const overviewMetrics = useMemo(() => {
    const availableLibraries = Object.values(health.libraries_available).filter(Boolean).length;
    const totalLibraries = Object.keys(health.libraries_available).length;

    return {
      librariesAvailable: `${availableLibraries}/${totalLibraries}`,
      estimatorsLoaded: health.estimators_loaded,
      analysisCount: health.analysis_count_24h,
      avgLatency: health.average_latency_ms ? `${(health.average_latency_ms / 1000).toFixed(1)}s` : 'N/A',
      pipelineReady: health.pipeline_orchestrator_ready,
      hierarchicalReady: health.hierarchical_analyzer_ready,
    };
  }, [health]);

  // Segment chart data — empty when no analysis has been run yet
  const segmentChartData = useMemo(() => {
    if (!hierarchicalResult) return [];
    return hierarchicalResult.segment_results.map((seg) => ({
      name: seg.segment_name,
      cate: seg.cate_mean ?? 0,
      ci_lower: seg.cate_ci_lower ?? 0,
      ci_upper: seg.cate_ci_upper ?? 0,
      samples: seg.n_samples,
      errorY: [
        (seg.cate_mean ?? 0) - (seg.cate_ci_lower ?? 0),
        (seg.cate_ci_upper ?? 0) - (seg.cate_mean ?? 0),
      ],
    }));
  }, [hierarchicalResult]);

  const handleRunAnalysis = async () => {
    try {
      await runAnalysisMutation.mutateAsync({
        request: {
          treatment_var: treatmentVar,
          outcome_var: outcomeVar,
          n_segments: nSegments,
          segmentation_method: SegmentationMethod.QUANTILE,
          estimator_type: EstimatorType.CAUSAL_FOREST,
        },
        asyncMode: true,
      });
    } catch (error) {
      console.error('Analysis failed:', error);
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
            Multi-library causal inference with hierarchical CATE estimation
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={handleRunAnalysis}
            disabled={runAnalysisMutation.isPending}
          >
            <Play className="mr-2 h-4 w-4" />
            Run Analysis
          </Button>
          <Button variant="outline">
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Service Health Banner */}
      {health.status === 'healthy' ? (
        <Alert className="mb-6 border-green-200 bg-green-50">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertTitle className="text-green-800">Causal Engine Healthy</AlertTitle>
          <AlertDescription className="text-green-700">
            All {Object.values(health.libraries_available).filter(Boolean).length} causal libraries available.
            {' '}{health.analysis_count_24h} analyses completed in the last 24 hours.
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

      {/* Overview Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
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
        <KPICard
          title="Pipeline"
          value={overviewMetrics.pipelineReady ? 'Ready' : 'Down'}
          icon={<Network className="h-5 w-5" />}
          valueColor={overviewMetrics.pipelineReady ? 'text-green-600' : 'text-red-600'}
        />
        <KPICard
          title="Hierarchical"
          value={overviewMetrics.hierarchicalReady ? 'Ready' : 'Down'}
          icon={<BarChart3 className="h-5 w-5" />}
          valueColor={overviewMetrics.hierarchicalReady ? 'text-green-600' : 'text-red-600'}
        />
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="hierarchical" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="hierarchical">Hierarchical CATE</TabsTrigger>
          <TabsTrigger value="libraries">Library Comparison</TabsTrigger>
          <TabsTrigger value="estimators">Estimators</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        {/* Hierarchical CATE Tab */}
        <TabsContent value="hierarchical" className="space-y-6">
          {/* Analysis Configuration */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Analysis Configuration
              </CardTitle>
              <CardDescription>Configure hierarchical CATE analysis parameters</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-4 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Treatment Variable</label>
                  <Input
                    value={treatmentVar}
                    onChange={(e) => setTreatmentVar(e.target.value)}
                    placeholder="e.g., rep_visits"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Outcome Variable</label>
                  <Input
                    value={outcomeVar}
                    onChange={(e) => setOutcomeVar(e.target.value)}
                    placeholder="e.g., trx_count"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Number of Segments</label>
                  <Select value={String(nSegments)} onValueChange={(v) => setNSegments(Number(v))}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {[2, 3, 4, 5, 6].map((n) => (
                        <SelectItem key={n} value={String(n)}>
                          {n} segments
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Estimator</label>
                  <Select defaultValue="causal_forest">
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="causal_forest">Causal Forest</SelectItem>
                      <SelectItem value="linear_dml">Linear DML</SelectItem>
                      <SelectItem value="dr_learner">DR Learner</SelectItem>
                      <SelectItem value="x_learner">X-Learner</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Analysis Results — empty state when no analysis run yet (F-002) */}
          {!hierarchicalResult ? (
            <EmptyState
              title="No hierarchical CATE analysis available"
              description="Configure parameters above and click Run Analysis to estimate segment-level treatment effects."
            />
          ) : (
            <>
              {/* Analysis Results Summary */}
              <div className="grid md:grid-cols-3 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Overall ATE</CardTitle>
                    <CardDescription>Average Treatment Effect</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="text-center">
                      <div className="text-4xl font-bold text-primary">
                        {formatEffect(hierarchicalResult.overall_ate)}
                      </div>
                      <div className="text-sm text-muted-foreground mt-2">
                        CI: {formatCI(hierarchicalResult.overall_ci_lower, hierarchicalResult.overall_ci_upper)}
                      </div>
                      <div className="mt-4">
                        {getStatusBadge(hierarchicalResult.status)}
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Heterogeneity (I²)</CardTitle>
                    <CardDescription>Between-segment variance</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="text-center">
                      <div className={`text-4xl font-bold ${
                        (hierarchicalResult.segment_heterogeneity ?? 0) > 50
                          ? 'text-yellow-600'
                          : 'text-green-600'
                      }`}>
                        {hierarchicalResult.segment_heterogeneity?.toFixed(1)}%
                      </div>
                      <div className="text-sm text-muted-foreground mt-2">
                        {(hierarchicalResult.segment_heterogeneity ?? 0) > 50
                          ? 'Substantial heterogeneity'
                          : 'Moderate heterogeneity'}
                      </div>
                      <div className="mt-4 text-xs text-muted-foreground">
                        τ² = {hierarchicalResult.nested_ci?.tau_squared?.toFixed(4)}
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Analysis Details</CardTitle>
                    <CardDescription>Configuration used</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Segments:</span>
                        <span className="font-medium">{hierarchicalResult.n_segments_analyzed}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Method:</span>
                        <span className="font-medium capitalize">{hierarchicalResult.segmentation_method}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Estimator:</span>
                        <span className="font-medium capitalize">{hierarchicalResult.estimator_type.replace('_', ' ')}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Total Samples:</span>
                        <span className="font-medium">{hierarchicalResult.nested_ci?.total_sample_size.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Latency:</span>
                        <span className="font-medium">{(hierarchicalResult.latency_ms / 1000).toFixed(1)}s</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Segment-Level Results */}
              <Card>
                <CardHeader>
                  <CardTitle>Segment-Level CATE Estimates</CardTitle>
                  <CardDescription>
                    Conditional Average Treatment Effects by uplift segment with 95% confidence intervals
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={350}>
                    <BarChart data={segmentChartData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" domain={[-0.1, 0.6]} />
                      <YAxis dataKey="name" type="category" width={120} />
                      <Tooltip
                        formatter={(value) => typeof value === 'number' ? value.toFixed(3) : value}
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                              <div className="bg-background border rounded-lg p-3 shadow-lg">
                                <p className="font-semibold">{data.name}</p>
                                <p>CATE: {data.cate.toFixed(3)}</p>
                                <p>CI: [{data.ci_lower.toFixed(3)}, {data.ci_upper.toFixed(3)}]</p>
                                <p>Samples: {data.samples.toLocaleString()}</p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Legend />
                      <Bar dataKey="cate" fill={COLORS.primary} name="CATE">
                        {segmentChartData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={entry.cate > 0.2 ? COLORS.success : entry.cate > 0.1 ? COLORS.primary : COLORS.warning}
                          />
                        ))}
                        <ErrorBar dataKey="errorY" width={4} strokeWidth={2} stroke="#333" />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Segment Details Table */}
              <Card>
                <CardHeader>
                  <CardTitle>Segment Details</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-2 px-4">Segment</th>
                          <th className="text-left py-2 px-4">Samples</th>
                          <th className="text-left py-2 px-4">Uplift Range</th>
                          <th className="text-left py-2 px-4">CATE</th>
                          <th className="text-left py-2 px-4">CI</th>
                          <th className="text-left py-2 px-4">Contribution</th>
                          <th className="text-left py-2 px-4">Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {hierarchicalResult.segment_results.map((seg) => (
                          <tr key={seg.segment_id} className="border-b hover:bg-muted/50">
                            <td className="py-2 px-4 font-medium">{seg.segment_name}</td>
                            <td className="py-2 px-4">{seg.n_samples.toLocaleString()}</td>
                            <td className="py-2 px-4 font-mono text-xs">
                              [{seg.uplift_range[0].toFixed(2)}, {seg.uplift_range[1].toFixed(2)}]
                            </td>
                            <td className="py-2 px-4 font-semibold">{formatEffect(seg.cate_mean)}</td>
                            <td className="py-2 px-4 font-mono text-xs">
                              {formatCI(seg.cate_ci_lower ?? null, seg.cate_ci_upper ?? null)}
                            </td>
                            <td className="py-2 px-4">
                              {((hierarchicalResult.nested_ci?.segment_contributions[String(seg.segment_id)] ?? 0) * 100).toFixed(0)}%
                            </td>
                            <td className="py-2 px-4">
                              {seg.success ? (
                                <CheckCircle className="h-4 w-4 text-green-500" />
                              ) : (
                                <AlertTriangle className="h-4 w-4 text-red-500" />
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </TabsContent>

        {/* Library Comparison Tab */}
        <TabsContent value="libraries" className="space-y-6">
          {libraryComparison.length === 0 ? (
            <EmptyState
              title="No library comparison available"
              description="Run a causal analysis to compare effect estimates across DoWhy, EconML, and CausalML."
            />
          ) : (
          <>
          <div className="grid md:grid-cols-2 gap-6">
            {/* Effect Comparison Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Effect Estimates by Library</CardTitle>
                <CardDescription>Comparing causal effect estimates across libraries</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={libraryComparison.filter((d) => d.effect !== null)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="library" />
                    <YAxis domain={[0, 0.4]} />
                    <Tooltip formatter={(value) => typeof value === 'number' ? value.toFixed(3) : value} />
                    <Legend />
                    <Bar dataKey="effect" name="Effect Estimate">
                      {libraryComparison.filter((d) => d.effect !== null).map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={LIBRARY_COLORS[entry.library.toLowerCase()] || COLORS.primary} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Latency Comparison */}
            <Card>
              <CardHeader>
                <CardTitle>Execution Latency</CardTitle>
                <CardDescription>Analysis time by library (ms)</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={libraryComparison} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="library" type="category" width={80} />
                    <Tooltip />
                    <Bar dataKey="latency" fill={COLORS.secondary} name="Latency (ms)" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Library Details */}
          <Card>
            <CardHeader>
              <CardTitle>Library Comparison Details</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-4">Library</th>
                      <th className="text-left py-2 px-4">Effect Estimate</th>
                      <th className="text-left py-2 px-4">CI</th>
                      <th className="text-left py-2 px-4">Latency</th>
                      <th className="text-left py-2 px-4">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {libraryComparison.map((lib) => (
                      <tr key={lib.library} className="border-b hover:bg-muted/50">
                        <td className="py-2 px-4">
                          <div className="flex items-center gap-2">
                            <div
                              className="w-3 h-3 rounded-full"
                              style={{ backgroundColor: LIBRARY_COLORS[lib.library.toLowerCase()] }}
                            />
                            <span className="font-medium">{lib.library}</span>
                          </div>
                        </td>
                        <td className="py-2 px-4 font-mono">
                          {lib.effect !== null ? lib.effect.toFixed(3) : 'N/A'}
                        </td>
                        <td className="py-2 px-4 font-mono text-xs">
                          {formatCI(lib.ci_lower, lib.ci_upper)}
                        </td>
                        <td className="py-2 px-4">{lib.latency}ms</td>
                        <td className="py-2 px-4">
                          {lib.effect !== null ? (
                            <Badge variant="default">Success</Badge>
                          ) : (
                            <Badge variant="secondary">{lib.note || 'No estimate'}</Badge>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
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

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {SUPPORTED_ESTIMATORS.filter(
              (e) => selectedLibrary === 'all' || e.library === selectedLibrary
            ).map((estimator) => (
              <Card key={estimator.name} className="hover:shadow-md transition-shadow">
                <CardHeader className="pb-2">
                  <div className="flex justify-between items-start">
                    <CardTitle className="text-base">{estimator.name}</CardTitle>
                    <Badge
                      style={{ backgroundColor: LIBRARY_COLORS[estimator.library] }}
                      className="text-white"
                    >
                      {estimator.library}
                    </Badge>
                  </div>
                  <CardDescription>{estimator.type}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex gap-4 text-sm">
                    <div className="flex items-center gap-1">
                      {estimator.supports_ci ? (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 text-gray-400" />
                      )}
                      <span>CI</span>
                    </div>
                    <div className="flex items-center gap-1">
                      {estimator.supports_hte ? (
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
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Analysis History (7 Days)</CardTitle>
              <CardDescription>Daily analysis count and average treatment effect</CardDescription>
            </CardHeader>
            <CardContent>
              {/* F-002: history requires an API endpoint not yet wired; show empty state. */}
              <EmptyState
                title="Analysis history not yet available"
                description="Run analyses and the daily count + average ATE will appear here once the history endpoint is wired."
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
