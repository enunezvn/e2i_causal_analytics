/**
 * Experiments Page
 * ================
 *
 * A/B Testing & Experiments management dashboard.
 * Displays experiment health, enrollment stats, SRM checks,
 * interim analyses, and Digital Twin fidelity tracking.
 *
 * @module pages/Experiments
 */

import { useState, useMemo, type ReactNode } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
} from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
// Aliased: recharts already exports a `Tooltip` (imported above for charts).
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  useTriggerMonitoring,
  useInterimAnalyses,
  useFidelityComparisons,
  useExperimentsInsight,
} from '@/hooks/api';
import { StrategicInsightCard } from '@/components/insights';
import { EmptyState } from '@/components/ui/EmptyState';
import {
  AlertSeverity,
  ExperimentHealthStatus,
  StoppingDecision,
  MonitorAlert,
} from '@/types/experiments';
import { KPICard } from '@/components/visualizations';
import {
  RefreshCw,
  Play,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Activity,
  Target,
  Beaker,
  Shield,
} from 'lucide-react';

// =============================================================================
// LOCAL TYPES
// =============================================================================

interface LocalExperiment {
  experiment_id: string;
  experiment_name: string;
  health_status: ExperimentHealthStatus;
  /** WHY the flag is what it is — hover explanation (null = not reported). */
  health_reason?: string | null;
  total_enrolled: number;
  enrollment_rate_per_day: number;
  /** Fraction of the recorded enrollment plan (capped at 1). Null = the row
   *  carries no plan — progress unknowable, rendered as an honest dash. */
  current_information_fraction: number | null;
  /** Planned enrollment (migration 101). Null = no plan recorded. */
  target_enrollment?: number | null;
  has_srm: boolean;
  active_alerts: number;
  last_checked: string;
  variant_breakdown: Record<string, number>;
  start_date: string;
  primary_metric: string;
  /** Explainability (2026-07-11): what the experiment tests and why. Null =
   *  not recorded on the row; the card renders honest absence. */
  brand?: string | null;
  description?: string | null;
  intervention_channel?: string | null;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const COLORS = {
  healthy: '#10b981',
  warning: '#f59e0b',
  critical: '#ef4444',
  primary: '#3b82f6',
  secondary: '#8b5cf6',
  tertiary: '#06b6d4',
};

const PIE_COLORS = [COLORS.primary, COLORS.secondary, COLORS.tertiary, '#f97316'];

/** Brand scope options — same convention as the other pages' brand selectors. */
const BRANDS = [
  { value: 'All', label: 'All Brands' },
  { value: 'Remibrutinib', label: 'Remibrutinib' },
  { value: 'Kisqali', label: 'Kisqali' },
  { value: 'Fabhalta', label: 'Fabhalta' },
];

/**
 * Staleness threshold sent with every monitoring run: the synthetic A/B
 * substrate refreshes WEEKLY (Mon 3AM cron), so "no new data for 24h" is the
 * expected steady state 6 days out of 7, not an incident. 8 days = one refresh
 * cycle + a day of slack; the backend scales alert severity to this value.
 */
const STALE_THRESHOLD_HOURS = 192;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Hover explanation for the per-experiment health flag (icon + badge). The
 * reason string is computed server-side from the experiment's real enrollment
 * pace and recorded plan — when absent, say so honestly instead of inventing
 * an explanation.
 */
function HealthTooltip({ reason, children }: { reason?: string | null; children: ReactNode }) {
  return (
    <TooltipProvider delayDuration={150}>
      <UITooltip>
        <TooltipTrigger asChild>
          <span className="inline-flex cursor-help">{children}</span>
        </TooltipTrigger>
        <TooltipContent className="max-w-xs whitespace-normal">
          {reason ?? 'No health explanation reported for this experiment.'}
        </TooltipContent>
      </UITooltip>
    </TooltipProvider>
  );
}

function getHealthIcon(status: ExperimentHealthStatus) {
  switch (status) {
    case ExperimentHealthStatus.HEALTHY:
      return <CheckCircle className="h-4 w-4 text-green-500" />;
    case ExperimentHealthStatus.WARNING:
      return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
    case ExperimentHealthStatus.CRITICAL:
      return <XCircle className="h-4 w-4 text-red-500" />;
  }
}

function getHealthBadge(status: ExperimentHealthStatus) {
  const variants: Record<ExperimentHealthStatus, 'default' | 'secondary' | 'destructive'> = {
    [ExperimentHealthStatus.HEALTHY]: 'default',
    [ExperimentHealthStatus.WARNING]: 'secondary',
    [ExperimentHealthStatus.CRITICAL]: 'destructive',
  };
  return (
    <Badge variant={variants[status]} className="capitalize">
      {status}
    </Badge>
  );
}

function getSeverityIcon(severity: AlertSeverity) {
  switch (severity) {
    case AlertSeverity.CRITICAL:
      return <XCircle className="h-4 w-4 text-red-500" />;
    case AlertSeverity.WARNING:
      return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
    case AlertSeverity.INFO:
      return <Activity className="h-4 w-4 text-blue-500" />;
  }
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleString();
}

/** 'speaker_program_invitation' -> 'Speaker Program Invitation'. */
function formatChannel(value: string): string {
  return value
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

function formatTimeAgo(timestamp: string): string {
  const seconds = Math.floor((Date.now() - new Date(timestamp).getTime()) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function Experiments() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);
  // Provenance opt-in (#894). Defaults OFF (real-mode default-exclude, matching
  // the gap_analyzer/het resolver convention); the reviewer explicitly opts in
  // to surface the synthetic-gold A/B substrate this deployment runs on.
  const [includeSynthetic, setIncludeSynthetic] = useState(false);
  // Brand scope for the monitoring sweep ('All' = the 3-brand portfolio,
  // interleaved server-side so one generation batch can't monopolize the roster).
  const [selectedBrand, setSelectedBrand] = useState('All');

  // API hooks. No sample-data fallback — honest empty states only.
  const { data: monitorData, isPending: isLoadingMonitor, mutate: triggerMonitor } = useTriggerMonitoring();
  // Strategic portfolio read (grounded server-side in per-channel A/B effects).
  const expInsight = useExperimentsInsight();

  // Live interim analyses + fidelity for the currently-selected experiment.
  // Both endpoints are keyed by a single experiment id.
  const interimQuery = useInterimAnalyses(selectedExperiment ?? '', {
    enabled: !!selectedExperiment,
  });
  const fidelityQuery = useFidelityComparisons(selectedExperiment ?? '', {
    enabled: !!selectedExperiment,
  });
  const interimAnalyses = interimQuery.data?.analyses ?? [];
  const fidelityPoints = (fidelityQuery.data?.comparisons ?? []).map((c) => ({
    date: c.timestamp.split('T')[0],
    score: c.fidelity_score,
  }));

  // Derive experiments from live monitor data only — no sample fallback.
  // LocalExperiment extends ExperimentHealthSummary with additional UI fields
  const experiments = useMemo((): LocalExperiment[] => {
    if (monitorData?.experiments?.length) {
      // API data may not have all LocalExperiment fields, provide defaults
      return monitorData.experiments.map((exp) => ({
        ...exp,
        variant_breakdown: {},
        start_date: exp.last_checked.split('T')[0],
        primary_metric: 'conversion_rate',
        brand: exp.brand ?? null,
        description: exp.description ?? null,
        intervention_channel: exp.intervention_channel ?? null,
      }));
    }
    return [];
  }, [monitorData]);

  const alerts = useMemo(() => {
    return monitorData?.alerts ?? [];
  }, [monitorData]);

  // Filter experiments based on search
  const filteredExperiments = useMemo(() => {
    if (!searchQuery) return experiments;
    const query = searchQuery.toLowerCase();
    return experiments.filter(
      (exp) =>
        exp.experiment_name.toLowerCase().includes(query) ||
        exp.experiment_id.toLowerCase().includes(query)
    );
  }, [experiments, searchQuery]);

  // Calculate overview metrics
  const overviewMetrics = useMemo(() => {
    const total = experiments.length;
    const healthy = experiments.filter((e) => e.health_status === ExperimentHealthStatus.HEALTHY).length;
    const warning = experiments.filter((e) => e.health_status === ExperimentHealthStatus.WARNING).length;
    const critical = experiments.filter((e) => e.health_status === ExperimentHealthStatus.CRITICAL).length;
    const totalEnrolled = experiments.reduce((sum, e) => sum + e.total_enrolled, 0);
    // Guard 0/0 -> NaN: the page loads no data until the first monitoring run
    const avgEnrollmentRate =
      total > 0 ? experiments.reduce((sum, e) => sum + e.enrollment_rate_per_day, 0) / total : 0;
    const srmCount = experiments.filter((e) => e.has_srm).length;
    const totalAlerts = alerts.length;
    const criticalAlerts = alerts.filter((a) => a.severity === AlertSeverity.CRITICAL).length;

    return {
      total,
      healthy,
      warning,
      critical,
      totalEnrolled,
      avgEnrollmentRate: avgEnrollmentRate.toFixed(1),
      srmCount,
      totalAlerts,
      criticalAlerts,
    };
  }, [experiments, alerts]);

  // Enrollment trend chart REMOVED (honesty fix): the prior weekly series was
  // hardcoded scaffolding (W1..W6 demo values), never wired to a real source.
  // No per-week enrollment time-series exists in the API — the monitor returns
  // only per-experiment totals (total_enrolled / enrollment_rate_per_day), and
  // the /enrollments EnrollmentStats dataclass carries no enrollment_trend.
  // Rather than fabricate chrome on a page that otherwise shows honest empty
  // states, the chart is replaced by an honest empty state below.

  // Health distribution data
  const healthDistributionData = useMemo(() => {
    return [
      { name: 'Healthy', value: overviewMetrics.healthy, color: COLORS.healthy },
      { name: 'Warning', value: overviewMetrics.warning, color: COLORS.warning },
      { name: 'Critical', value: overviewMetrics.critical, color: COLORS.critical },
    ].filter((d) => d.value > 0);
  }, [overviewMetrics]);

  // Provenance honesty (#894). `synthetic_data_forced` means the deployment is a
  // synthetic-gold SHOWCASE instance (E2I_INCLUDE_SYNTHETIC) — its entire A/B
  // substrate is synthetic-gold, so the checkbox is inert. We drive the context
  // banner off forced-OR-included rather than the per-row is_synthetic flag,
  // because the goldstd experiments are deliberately tagged is_synthetic=False
  // (kept servable in real-mode) — so is_synthetic alone under-reports the
  // synthetic substrate. The deployment flag is the reliable signal.
  const syntheticForced = monitorData?.synthetic_data_forced ?? false;
  const syntheticIncluded = monitorData?.synthetic_data_included ?? false;
  const syntheticSubstrate = syntheticForced || syntheticIncluded;

  // One request shape for Run/Refresh: brand scope + provenance opt-in + the
  // weekly-cadence staleness threshold (see STALE_THRESHOLD_HOURS).
  const monitorRequest = useMemo(
    () => ({
      include_synthetic: includeSynthetic,
      brand: selectedBrand === 'All' ? undefined : selectedBrand,
      stale_data_threshold_hours: STALE_THRESHOLD_HOURS,
    }),
    [includeSynthetic, selectedBrand]
  );

  const handleRunMonitoring = () => {
    triggerMonitor(monitorRequest);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold text-foreground flex items-center gap-2">
            <Beaker className="h-8 w-8" />
            A/B Testing & Experiments
          </h1>
          <p className="text-muted-foreground mt-1">
            Monitor experiment health, enrollment, and statistical analysis
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {/* Brand scope (2026-07-11): 'All Brands' = the 3-brand portfolio,
              interleaved server-side; a single brand scopes the sweep. */}
          <Select value={selectedBrand} onValueChange={setSelectedBrand}>
            <SelectTrigger className="w-40" aria-label="Brand">
              <SelectValue placeholder="Brand" />
            </SelectTrigger>
            <SelectContent>
              {BRANDS.map((brand) => (
                <SelectItem key={brand.value} value={brand.value}>
                  {brand.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {/* Provenance opt-in (#894): all A/B substrate in this deployment is
              synthetic-gold, so a reviewer must opt in to see it. Default OFF
              matches the backend real-mode default-exclude. When the deployment
              FORCES synthetic inclusion (E2I_INCLUDE_SYNTHETIC) the flag is inert,
              so we disable the checkbox and say so rather than implying a control
              the page does not actually have here. */}
          <div className="flex items-center gap-2 mr-2">
            <Checkbox
              id="include-synthetic"
              checked={syntheticForced ? true : includeSynthetic}
              disabled={syntheticForced}
              onCheckedChange={(checked) => setIncludeSynthetic(checked === true)}
            />
            <Label htmlFor="include-synthetic" className="text-sm cursor-pointer">
              {syntheticForced
                ? 'Synthetic data (always included in this deployment)'
                : 'Include synthetic data'}
            </Label>
          </div>
          <Button onClick={handleRunMonitoring} disabled={isLoadingMonitor}>
            <Play className="mr-2 h-4 w-4" />
            Run Monitoring
          </Button>
          <Button variant="outline" onClick={() => triggerMonitor(monitorRequest)}>
            <RefreshCw className={`mr-2 h-4 w-4 ${isLoadingMonitor ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Synthetic-substrate context banner (honesty). The alerts below are
          REAL computations, but this deployment runs on the weekly-refreshed
          synthetic-gold A/B substrate, so "data staleness" and zero-enrollment
          alerts reflect the refresh cadence, not a broken pipeline. Surfacing
          this prevents the alarms from being mistaken for a live incident. */}
      {syntheticSubstrate && (
        <Alert className="mb-6">
          <Beaker className="h-4 w-4" />
          <AlertTitle>In-silico A/B testing on the synthetic-gold substrate</AlertTitle>
          <AlertDescription>
            Each experiment randomizes one engagement intervention (from the
            digital-twin taxonomy) against standard practice on the synthetic HCP
            panel, with known ground-truth effects planted so estimator recovery
            is verifiable. The substrate refreshes weekly (Mon 3AM), so data
            staleness is judged against an 8-day threshold — freshness alerts
            reflect the refresh cadence, not a broken pipeline.
          </AlertDescription>
        </Alert>
      )}

      {/* Critical Alerts Banner */}
      {overviewMetrics.criticalAlerts > 0 && (
        <Alert variant="destructive" className="mb-6">
          <XCircle className="h-4 w-4" />
          <AlertTitle>Critical Alerts</AlertTitle>
          <AlertDescription>
            {overviewMetrics.criticalAlerts} experiment(s) require immediate attention.
            {alerts
              .filter((a) => a.severity === AlertSeverity.CRITICAL)
              .slice(0, 2)
              .map((a) => (
                <div key={a.alert_id} className="mt-1 text-sm">
                  <strong>{a.experiment_name}:</strong> {a.message}
                </div>
              ))}
          </AlertDescription>
        </Alert>
      )}

      {/* Overview Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
        <KPICard
          title="Active Experiments"
          value={monitorData?.total_running || overviewMetrics.total}
          description={
            monitorData?.total_running
              ? `${overviewMetrics.total} newest monitored this sweep`
              : 'Currently running'
          }
        />
        <KPICard
          title="Healthy"
          value={overviewMetrics.healthy}
          status="healthy"
          description="Passing all checks"
        />
        <KPICard
          title="Total Enrolled"
          value={overviewMetrics.totalEnrolled.toLocaleString()}
          description="Participants enrolled"
        />
        <KPICard
          title="Avg Enrollment/Day"
          value={overviewMetrics.avgEnrollmentRate}
          description="Daily enrollment rate"
        />
        <KPICard
          title="SRM Detected"
          value={overviewMetrics.srmCount}
          status={overviewMetrics.srmCount > 0 ? 'critical' : 'healthy'}
          description="Sample ratio mismatch"
        />
        <KPICard
          title="Active Alerts"
          value={overviewMetrics.totalAlerts}
          status={overviewMetrics.criticalAlerts > 0 ? 'critical' : 'warning'}
          description="Requiring attention"
        />
      </div>

      {/* Strategic Interpretation — agentic read of the A/B portfolio, grounded
          SERVER-SIDE in per-channel effects from ab_experiment_results (which
          interventions show real lift, what adopting them would be worth, and
          which channels honestly show nothing). Rendered always so the header
          is present on mount; generation is on demand. */}
      <div className="mb-8">
        <StrategicInsightCard
          title="Portfolio Strategic Read"
          description="Which tested interventions show causal lift on the brand outcome — and which don't"
          isLoading={expInsight.isPending}
          error={expInsight.error?.message ?? null}
          insight={expInsight.data?.insight}
          keyTakeaways={expInsight.data?.key_takeaways}
          grounding={expInsight.data?.grounding}
          isFallback={expInsight.data?.is_fallback}
          provenance={expInsight.data?.provenance}
          generatedAt={expInsight.data?.generated_at}
          onGenerate={() =>
            expInsight.mutate({
              brand: selectedBrand,
              include_synthetic: syntheticForced ? true : includeSynthetic,
            })
          }
        />
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="experiments" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="experiments">Experiments</TabsTrigger>
          <TabsTrigger value="alerts">Alerts ({alerts.length})</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="fidelity">Digital Twin</TabsTrigger>
        </TabsList>

        {/* Experiments Tab */}
        <TabsContent value="experiments" className="space-y-6">
          {/* Search */}
          <div className="flex gap-4">
            <Input
              placeholder="Search experiments..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="max-w-md"
            />
          </div>

          {filteredExperiments.length === 0 && (
            <EmptyState
              icon={<Beaker className="h-8 w-8" aria-hidden="true" />}
              title="No experiments loaded yet"
              description={
                searchQuery
                  ? 'No experiments match your search. Clear the search or run monitoring to load the live roster.'
                  : 'This page loads on demand. Click below to run a live monitoring sweep and load experiment health, enrollment, and alerts.'
              }
              action={
                !searchQuery ? (
                  <Button onClick={handleRunMonitoring} disabled={isLoadingMonitor}>
                    <Play className="mr-2 h-4 w-4" />
                    {isLoadingMonitor ? 'Running monitoring…' : 'Run Monitoring'}
                  </Button>
                ) : undefined
              }
            />
          )}

          {/* Experiment Cards */}
          <div className="grid gap-4">
            {filteredExperiments.map((experiment) => (
              <Card
                key={experiment.experiment_id}
                className={`cursor-pointer transition-all hover:shadow-md ${
                  selectedExperiment === experiment.experiment_id ? 'ring-2 ring-primary' : ''
                }`}
                onClick={() =>
                  setSelectedExperiment(
                    selectedExperiment === experiment.experiment_id ? null : experiment.experiment_id
                  )
                }
              >
                <CardHeader className="pb-2">
                  <div className="flex justify-between items-start">
                    <div className="flex items-center gap-2">
                      <HealthTooltip reason={experiment.health_reason}>
                        {getHealthIcon(experiment.health_status)}
                      </HealthTooltip>
                      <CardTitle className="text-lg">{experiment.experiment_name}</CardTitle>
                    </div>
                    <HealthTooltip reason={experiment.health_reason}>
                      {getHealthBadge(experiment.health_status)}
                    </HealthTooltip>
                  </div>
                  {/* Explainability badges: brand + intervention under test.
                      Rendered only when recorded — no fabricated metadata. */}
                  {(experiment.brand || experiment.intervention_channel) && (
                    <div className="flex flex-wrap items-center gap-2 mt-1">
                      {experiment.brand && (
                        <Badge variant="outline">{experiment.brand}</Badge>
                      )}
                      {experiment.intervention_channel && (
                        <Badge variant="secondary">
                          {formatChannel(experiment.intervention_channel)}
                        </Badge>
                      )}
                    </div>
                  )}
                  {/* WHAT this experiment tests and WHY (hypothesis + in-silico
                      design). Falls back to the id when the row predates the
                      explainability metadata. */}
                  {experiment.description ? (
                    <CardDescription className="mt-1 whitespace-normal">
                      {experiment.description}
                    </CardDescription>
                  ) : (
                    <CardDescription>{experiment.experiment_id}</CardDescription>
                  )}
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Enrolled</span>
                      <p className="font-semibold">{experiment.total_enrolled.toLocaleString()}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Rate/Day</span>
                      <p className="font-semibold">{experiment.enrollment_rate_per_day.toFixed(1)}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Plan Progress</span>
                      {/* Null = no enrollment plan recorded on the row — show an
                          honest dash, never a fabricated 0% or 100%. */}
                      <p className="font-semibold">
                        {experiment.current_information_fraction != null
                          ? `${(experiment.current_information_fraction * 100).toFixed(0)}%${
                              experiment.target_enrollment
                                ? ` of ${experiment.target_enrollment.toLocaleString()}`
                                : ''
                            }`
                          : '—'}
                      </p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">SRM</span>
                      <p className={`font-semibold ${experiment.has_srm ? 'text-red-600' : 'text-green-600'}`}>
                        {experiment.has_srm ? 'Detected' : 'None'}
                      </p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Alerts</span>
                      <p className={`font-semibold ${experiment.active_alerts > 0 ? 'text-yellow-600' : ''}`}>
                        {experiment.active_alerts}
                      </p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Last Check</span>
                      <p className="font-semibold">{formatTimeAgo(experiment.last_checked)}</p>
                    </div>
                  </div>

                  {/* Variant Breakdown Progress Bar */}
                  {Object.keys(experiment.variant_breakdown).length > 0 && (
                  <div className="mt-4">
                    <div className="flex justify-between text-xs text-muted-foreground mb-1">
                      <span>Variant Distribution</span>
                      <span>
                        {Object.entries(experiment.variant_breakdown)
                          .map(([k, v]) => `${k}: ${v}`)
                          .join(' | ')}
                      </span>
                    </div>
                    <div className="flex h-2 rounded-full overflow-hidden">
                      {Object.entries(experiment.variant_breakdown).map(([variant, countValue], idx) => {
                        const count = countValue as number;
                        const total = (Object.values(experiment.variant_breakdown) as number[]).reduce((a, b) => a + b, 0);
                        const percent = total > 0 ? (count / total) * 100 : 0;
                        return (
                          <div
                            key={variant}
                            style={{
                              width: `${percent}%`,
                              backgroundColor: PIE_COLORS[idx % PIE_COLORS.length],
                            }}
                          />
                        );
                      })}
                    </div>
                  </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Alerts Tab */}
        <TabsContent value="alerts" className="space-y-4">
          {syntheticSubstrate && alerts.length > 0 && (
            <p className="text-xs text-muted-foreground">
              These alerts are computed on the weekly-refreshed synthetic-gold
              substrate (Mon 3AM); freshness and enrollment alerts are judged
              against that refresh cadence, not a live data pipeline.
            </p>
          )}
          {alerts.length === 0 ? (
            <Card>
              <CardContent className="py-12 text-center">
                <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
                <h3 className="text-lg font-semibold">No Active Alerts</h3>
                <p className="text-muted-foreground">All experiments are running smoothly.</p>
              </CardContent>
            </Card>
          ) : (
            alerts.map((alert: MonitorAlert) => (
              <Card
                key={alert.alert_id}
                className={`border-l-4 ${
                  alert.severity === AlertSeverity.CRITICAL
                    ? 'border-l-red-500'
                    : alert.severity === AlertSeverity.WARNING
                    ? 'border-l-yellow-500'
                    : 'border-l-blue-500'
                }`}
              >
                <CardHeader className="pb-2">
                  <div className="flex justify-between items-start">
                    <div className="flex items-center gap-2">
                      {getSeverityIcon(alert.severity)}
                      <CardTitle className="text-base">{alert.message}</CardTitle>
                    </div>
                    <Badge
                      variant={
                        alert.severity === AlertSeverity.CRITICAL
                          ? 'destructive'
                          : alert.severity === AlertSeverity.WARNING
                          ? 'secondary'
                          : 'default'
                      }
                    >
                      {alert.severity}
                    </Badge>
                  </div>
                  <CardDescription>
                    {alert.experiment_name} &middot; {formatTimeAgo(alert.timestamp)}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm">
                      <Target className="h-4 w-4 text-muted-foreground" />
                      <span className="font-medium">Recommended Action:</span>
                      <span className="text-muted-foreground">{alert.recommended_action}</span>
                    </div>
                    {Object.keys(alert.details).length > 0 && (
                      <div className="bg-muted/50 rounded p-2 text-xs font-mono">
                        {JSON.stringify(alert.details, null, 2)}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="analytics" className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            {/* Health Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Experiment Health Distribution</CardTitle>
                <CardDescription>Current status of all active experiments</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={healthDistributionData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={90}
                      paddingAngle={5}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}`}
                    >
                      {healthDistributionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Total Enrolled by experiment (real monitor data; honest empty
                state when no live roster is loaded). Replaces the prior
                hardcoded weekly demo series — no per-week enrollment time-series
                source exists. */}
            <Card>
              <CardHeader>
                <CardTitle>Enrollment by Experiment</CardTitle>
                <CardDescription>Total enrolled per active experiment (live)</CardDescription>
              </CardHeader>
              <CardContent>
                {experiments.length === 0 ? (
                  <EmptyState
                    title="No enrollment data"
                    description='Run "Run Monitoring" to load total enrollment per experiment from the monitoring service.'
                  />
                ) : (
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart
                      data={experiments.map((e) => ({
                        name: e.experiment_name,
                        enrolled: e.total_enrolled,
                      }))}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="enrolled" fill={COLORS.primary} name="Enrolled" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Interim Analyses */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Interim Analyses</CardTitle>
              <CardDescription>Statistical stopping decisions with alpha spending</CardDescription>
            </CardHeader>
            <CardContent>
              {!selectedExperiment ? (
                <EmptyState
                  title="Select an experiment to view interim analyses"
                  description="Click an experiment in the Experiments tab to load its interim-analysis history."
                />
              ) : interimQuery.isLoading ? (
                <EmptyState title="Loading interim analyses…" />
              ) : interimQuery.isError ? (
                <EmptyState
                  title="Could not load interim analyses"
                  description={interimQuery.error?.message ?? 'The interim-analyses endpoint returned an error.'}
                />
              ) : interimAnalyses.length === 0 ? (
                <EmptyState
                  title="No interim analyses yet"
                  description="This experiment has no recorded interim analyses."
                />
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-2 px-4">Experiment</th>
                        <th className="text-left py-2 px-4">Analysis #</th>
                        <th className="text-left py-2 px-4">Info Fraction</th>
                        <th className="text-left py-2 px-4">P-Value</th>
                        <th className="text-left py-2 px-4">Decision</th>
                        <th className="text-left py-2 px-4">Date</th>
                      </tr>
                    </thead>
                    <tbody>
                      {interimAnalyses.map((analysis) => (
                        <tr key={analysis.analysis_id} className="border-b hover:bg-muted/50">
                          <td className="py-2 px-4">{selectedExperiment}</td>
                          <td className="py-2 px-4">{analysis.analysis_number}</td>
                          <td className="py-2 px-4">{(analysis.information_fraction * 100).toFixed(0)}%</td>
                          <td className="py-2 px-4 font-mono">{analysis.p_value.toFixed(4)}</td>
                          <td className="py-2 px-4">
                            <Badge
                              variant={
                                analysis.decision === StoppingDecision.CONTINUE
                                  ? 'secondary'
                                  : analysis.decision === StoppingDecision.STOP_EFFICACY
                                  ? 'default'
                                  : 'destructive'
                              }
                            >
                              {analysis.decision}
                            </Badge>
                          </td>
                          <td className="py-2 px-4 text-muted-foreground">
                            {formatTimestamp(analysis.performed_at)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Digital Twin Fidelity Tab */}
        <TabsContent value="fidelity" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Digital Twin Fidelity Tracking
              </CardTitle>
              <CardDescription>
                Comparing Digital Twin predictions with actual experiment outcomes
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!selectedExperiment ? (
                <EmptyState
                  title="Select an experiment to view fidelity tracking"
                  description="Click an experiment in the Experiments tab to compare Digital Twin predictions with observed outcomes."
                />
              ) : fidelityQuery.isLoading ? (
                <EmptyState title="Loading fidelity comparisons…" />
              ) : fidelityQuery.isError ? (
                <EmptyState
                  title="Could not load fidelity comparisons"
                  description={fidelityQuery.error?.message ?? 'The fidelity endpoint returned an error.'}
                />
              ) : fidelityPoints.length === 0 ? (
                <EmptyState
                  title="No fidelity comparisons yet"
                  description="This experiment has no recorded Digital Twin fidelity comparisons."
                />
              ) : (
                <>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={fidelityPoints}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis domain={[0.5, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                      <Tooltip formatter={(value) => `${((value as number) * 100).toFixed(1)}%`} />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="score"
                        stroke={COLORS.primary}
                        strokeWidth={2}
                        name="Fidelity Score"
                        dot={{ r: 4 }}
                      />
                      <Line
                        type="monotone"
                        dataKey={() => 0.85}
                        stroke={COLORS.warning}
                        strokeDasharray="5 5"
                        name="Threshold"
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>

                  <div className="mt-6 grid md:grid-cols-3 gap-4">
                    <Card className="bg-muted/50">
                      <CardContent className="pt-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold">
                            {(fidelityPoints[0].score * 100).toFixed(0)}%
                          </div>
                          <div className="text-sm text-muted-foreground">Initial Fidelity</div>
                        </div>
                      </CardContent>
                    </Card>
                    <Card className="bg-muted/50">
                      <CardContent className="pt-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-yellow-600">
                            {(fidelityPoints[fidelityPoints.length - 1].score * 100).toFixed(0)}%
                          </div>
                          <div className="text-sm text-muted-foreground">Current Fidelity</div>
                        </div>
                      </CardContent>
                    </Card>
                    <Card className="bg-muted/50">
                      <CardContent className="pt-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-red-600">
                            {(
                              (fidelityPoints[fidelityPoints.length - 1].score -
                                fidelityPoints[0].score) *
                              100
                            ).toFixed(0)}
                            %
                          </div>
                          <div className="text-sm text-muted-foreground">Change</div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  {fidelityPoints[fidelityPoints.length - 1].score < 0.85 && (
                    <Alert className="mt-6">
                      <AlertTriangle className="h-4 w-4" />
                      <AlertTitle>Calibration Recommended</AlertTitle>
                      <AlertDescription>
                        Digital Twin fidelity has dropped below the 85% threshold. Consider updating the
                        model parameters based on the latest experiment observations to improve prediction
                        accuracy.
                      </AlertDescription>
                    </Alert>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
