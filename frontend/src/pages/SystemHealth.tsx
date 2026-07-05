/**
 * System Health Page
 * ==================
 *
 * Dashboard for monitoring E2I system health including:
 * - Service status grid (API, Database, Redis, FalkorDB, BentoML)
 * - Model health cards with health scores
 * - Active alerts list
 * - Auto-refresh every 30s
 *
 * @module pages/SystemHealth
 */

import { useState, useMemo, useCallback, useEffect } from 'react';
import {
  Server,
  Activity,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  Brain,
  TrendingUp,
  TrendingDown,
  Minus,
  Workflow,
  Bot,
  Shield,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useAlerts, useMonitoringRuns } from '@/hooks/api/use-monitoring';
import {
  useFullHealthCheck,
  usePipelineHealth,
  useAgentHealth,
  useHealthHistory,
  useComponentHealth,
  useModelHealth,
} from '@/hooks/api';
import { AlertStatus } from '@/types/monitoring';
import type { AlertItem } from '@/types/monitoring';
import { HealthGrade } from '@/types/health-score';
import type {
  AgentHealth,
  ComponentHealth as ApiComponentHealth,
  ModelHealth as ApiModelHealth,
} from '@/types/health-score';
import { AlertList } from '@/components/visualizations/dashboard/AlertCard';
import { StatusBadge, StatusDot } from '@/components/visualizations/dashboard/StatusBadge';
import { ProgressRing } from '@/components/visualizations/dashboard/ProgressRing';
import { EmptyState } from '@/components/ui/EmptyState';
import type { AlertSeverity } from '@/components/visualizations/dashboard/AlertCard';
import type { StatusType } from '@/components/visualizations/dashboard/StatusBadge';

// =============================================================================
// TYPES
// =============================================================================

// View-model for the Service Status card. Derived 1:1 from the backend
// /health-score/components ComponentHealth shape — no fabricated fields.
interface ServiceStatus {
  name: string;
  status: 'healthy' | 'warning' | 'error' | 'unknown';
  latencyMs?: number;
  icon: React.ElementType;
}

// View-model for the Model Health card. Mirrors the backend
// /health-score/models ModelHealth shape EXACTLY. The real endpoint reports
// status + (optionally) accuracy / error_rate / predictions_last_24h; it does
// NOT report a drift score or a performance trend, so those are intentionally
// absent here rather than fabricated. null = unmeasured -> rendered as "—".
interface ModelHealthView {
  modelId: string;
  name: string;
  status: 'healthy' | 'warning' | 'critical';
  accuracy: number | null;
  errorRate: number | null;
  predictions24h: number | null;
}

// =============================================================================
// CONSTANTS
// =============================================================================
// F-002 fix: SAMPLE_* fixtures formerly inlined here have been DELETED.
// The page surfaces API data only; when the API hasn't returned, the
// section renders empty states. No fabricated values reachable from
// production rendering paths.

// Grade color mapping
const GRADE_COLORS: Record<HealthGrade | string, string> = {
  [HealthGrade.A]: 'text-emerald-600 bg-emerald-100 border-emerald-300',
  [HealthGrade.B]: 'text-green-600 bg-green-100 border-green-300',
  [HealthGrade.C]: 'text-amber-600 bg-amber-100 border-amber-300',
  [HealthGrade.D]: 'text-orange-600 bg-orange-100 border-orange-300',
  [HealthGrade.F]: 'text-rose-600 bg-rose-100 border-rose-300',
};

const TIER_NAMES: Record<number, string> = {
  0: 'Foundation',
  1: 'Orchestration',
  2: 'Causal',
  3: 'Monitoring',
  4: 'ML Predictions',
  5: 'Learning',
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function mapHealthToStatus(health: string): StatusType {
  switch (health) {
    case 'healthy':
      return 'healthy';
    case 'warning':
      return 'warning';
    case 'critical':
    case 'error':
      return 'error';
    default:
      return 'unknown';
  }
}

// A response's items are surfaced as real data only when the backend tagged the
// wrapper with a trusted provenance. "measured" = live probe; "partial" = some
// sub-fields unmeasured but the measured signals are real. "placeholder" /
// "unknown" / absent are NOT trustworthy and degrade to the honest empty state.
function isTrustedProvenance(provenance: string | undefined): boolean {
  return provenance === 'measured' || provenance === 'partial';
}

// Map the backend ComponentStatus enum (healthy | degraded | unhealthy |
// unknown) to the Service Status card's status vocabulary.
function mapComponentStatus(status: string): ServiceStatus['status'] {
  switch (status) {
    case 'healthy':
      return 'healthy';
    case 'degraded':
      return 'warning';
    case 'unhealthy':
      return 'error';
    default:
      return 'unknown';
  }
}

// Map the backend ModelStatus enum (healthy | degraded | unhealthy) to the
// ProgressRing/badge status vocabulary used by the Model Health card.
function mapModelStatus(status: string): ModelHealthView['status'] {
  switch (status) {
    case 'healthy':
      return 'healthy';
    case 'degraded':
      return 'warning';
    case 'unhealthy':
      return 'critical';
    default:
      return 'warning';
  }
}

function mapAlertSeverity(severity: string): AlertSeverity {
  switch (severity.toLowerCase()) {
    case 'critical':
      return 'critical';
    case 'high':
      return 'error';
    case 'medium':
      return 'warning';
    default:
      return 'info';
  }
}

function formatDate(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function formatRelativeTime(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  const days = Math.floor(diff / (24 * 60 * 60 * 1000));

  if (days === 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days} days ago`;
  if (days < 30) return `${Math.floor(days / 7)} weeks ago`;
  return formatDate(d);
}

// =============================================================================
// COMPONENT
// =============================================================================

function SystemHealth() {
  const [lastRefresh, setLastRefresh] = useState(new Date());
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  // Fetch alerts from API
  const {
    data: alertsData,
    isLoading: isLoadingAlerts,
    refetch: refetchAlerts,
  } = useAlerts({ status: AlertStatus.ACTIVE, limit: 10 }, {
    refetchInterval: 30000, // Auto-refresh every 30s
  });

  // Fetch monitoring runs
  const {
    data: _runsData,
    isLoading: isLoadingRuns,
    refetch: refetchRuns,
  } = useMonitoringRuns({ days: 7, limit: 5 });

  // Health Score API hooks. Use the FULL check (all four dimensions), NOT the
  // quick (component-only) check: quick leaves model/pipeline/agent UNMEASURED,
  // which made the composer emit bogus "wire a real X backend" recommendations
  // and a component-only overall score (a misleading 100/A) even though this page
  // separately measures those dimensions. Full measures all four (e.g. 87.5/B)
  // and yields real, actionable recommendations.
  const {
    data: fullHealthData,
    refetch: refetchHealth,
  } = useFullHealthCheck({ refetchInterval: 60000 });

  const { data: agentHealthData } = useAgentHealth({ refetchInterval: 60000 });
  const { data: pipelineHealthData } = usePipelineHealth({ refetchInterval: 60000 });
  const { data: healthHistoryData } = useHealthHistory(20, { refetchInterval: 120000 });

  // Service Status + Model Health are now wired to the real backend endpoints
  // (GET /health-score/components, GET /health-score/models). #927 built these
  // endpoints + hooks + Zod wire-schemas; this surfaces them in the two cards.
  const {
    data: componentHealthData,
    refetch: refetchComponents,
  } = useComponentHealth({ refetchInterval: 30000 });
  const {
    data: modelHealthData,
    refetch: refetchModels,
  } = useModelHealth({ refetchInterval: 60000 });

  // Honesty model (mirrors #927): only data the backend explicitly tagged as
  // trustworthy (data_provenance "measured" or "partial") is surfaced as real.
  // Dev-offline "placeholder", "unknown", or an absent provenance are all
  // treated as NO data, so the cards render their honest empty states rather
  // than presenting unmeasured/untrusted values as real. A null/undefined
  // payload (e.g. a crashed/errored query) likewise degrades to empty — never
  // to a fabricated default.
  const services = useMemo<ServiceStatus[]>(() => {
    if (!componentHealthData || !isTrustedProvenance(componentHealthData.data_provenance)) {
      return [];
    }
    return componentHealthData.components.map((c: ApiComponentHealth) => ({
      name: c.component_name,
      status: mapComponentStatus(c.status),
      latencyMs: c.latency_ms ?? undefined,
      icon: Server,
    }));
  }, [componentHealthData]);

  const models = useMemo<ModelHealthView[]>(() => {
    if (!modelHealthData || !isTrustedProvenance(modelHealthData.data_provenance)) {
      return [];
    }
    return modelHealthData.models.map((m: ApiModelHealth) => ({
      modelId: m.model_id,
      name: m.model_name,
      status: mapModelStatus(m.status),
      // null = unmeasured (no ml_performance_metrics source). Preserve null so
      // the card renders "—" instead of a fabricated 0 / 0% / 0.00.
      accuracy: m.accuracy ?? null,
      errorRate: m.error_rate ?? null,
      predictions24h: m.predictions_last_24h ?? null,
    }));
  }, [modelHealthData]);

  // Use API data when present; otherwise render empty/neutral values. Dev-offline
  // placeholder data (data_provenance="placeholder") is treated as no score, so
  // the headline shows "Awaiting health check…" rather than a fabricated number.
  const isHealthPlaceholder = fullHealthData?.data_provenance === 'placeholder';
  const healthScore =
    fullHealthData && !isHealthPlaceholder ? (fullHealthData.overall_health_score ?? null) : null;
  const healthGrade =
    fullHealthData && !isHealthPlaceholder ? (fullHealthData.health_grade ?? null) : null;
  // Ported from the retired /ai-insights System Health Score card: surface the
  // composer's human-readable summary and flag a non-fully-measured check
  // ("partial") instead of presenting it as indistinguishable from measured.
  const healthSummary =
    fullHealthData && !isHealthPlaceholder ? (fullHealthData.health_summary || null) : null;
  const healthProvenance =
    fullHealthData && !isHealthPlaceholder ? (fullHealthData.data_provenance ?? null) : null;
  // When the backend composer actually ran its check — distinct from the page
  // header's "Last updated", which is only the local UI refresh clock. The
  // retired /ai-insights card surfaced this as "Last Check"; without it a
  // stale health score is indistinguishable from a fresh one.
  const healthCheckedAt =
    fullHealthData && !isHealthPlaceholder ? (fullHealthData.timestamp || null) : null;
  const agents = agentHealthData?.agents ?? [];
  const pipelines = pipelineHealthData?.pipelines ?? [];
  const healthHistory = healthHistoryData?.checks ?? [];
  // null = no history -> rendered as "Unknown", not a fabricated "stable".
  const healthTrend = healthHistoryData?.trend ?? null;

  // Group agents by tier
  const agentsByTier = useMemo(() => {
    const grouped: Record<number, AgentHealth[]> = {};
    agents.forEach(agent => {
      if (!grouped[agent.tier]) grouped[agent.tier] = [];
      grouped[agent.tier].push(agent);
    });
    return grouped;
  }, [agents]);

  // Prepare chart data
  const historyChartData = useMemo(() => {
    return healthHistory.map(item => ({
      date: formatDate(item.timestamp),
      score: item.overall_health_score,
      grade: item.health_grade,
    }));
  }, [healthHistory]);

  const componentScoreData = useMemo(() => {
    // No fabricated scores: show no bars until a real health check loads, and
    // never chart dev-offline placeholder data (data_provenance="placeholder")
    // as if it were measured.
    if (!fullHealthData || fullHealthData.data_provenance === 'placeholder') {
      return [];
    }
    // Only chart dimensions a real backend MEASURED. A null dimension score is
    // unmeasured and is omitted (not rendered as a fabricated 0% bar).
    const dims: Array<{ name: string; score: number | null | undefined; fill: string }> = [
      { name: 'Components', score: fullHealthData.component_health_score, fill: '#10b981' },
      { name: 'Models', score: fullHealthData.model_health_score, fill: '#3b82f6' },
      { name: 'Pipelines', score: fullHealthData.pipeline_health_score, fill: '#8b5cf6' },
      { name: 'Agents', score: fullHealthData.agent_health_score, fill: '#f59e0b' },
    ];
    return dims
      .filter((d) => d.score != null)
      .map((d) => ({ name: d.name, score: Math.round((d.score as number) * 100), fill: d.fill }));
  }, [fullHealthData]);

  // Convert API alerts to AlertCard format
  const alerts = useMemo(() => {
    if (!alertsData?.alerts) return [];
    return alertsData.alerts.map((alert: AlertItem) => ({
      severity: mapAlertSeverity(alert.severity),
      title: alert.title,
      message: alert.description,
      timestamp: alert.triggered_at,
      source: alert.model_version,
      isNew: alert.status === AlertStatus.ACTIVE,
    }));
  }, [alertsData?.alerts]);

  // Calculate overall health stats
  const healthStats = useMemo(() => {
    const healthyServices = services.filter(s => s.status === 'healthy').length;
    const totalServices = services.length;
    // Average only over services that actually report a latency. A missing/null
    // latency is unmeasured, NOT zero — counting it as 0 would fabricate a
    // misleadingly-low average. null when nothing measured -> rendered as "—".
    const measuredLatencies = services
      .map(s => s.latencyMs)
      .filter((ms): ms is number => ms != null);
    const avgLatency = measuredLatencies.length > 0
      ? Math.round(measuredLatencies.reduce((sum, ms) => sum + ms, 0) / measuredLatencies.length)
      : null;

    const healthyModels = models.filter(m => m.status === 'healthy').length;
    const warningModels = models.filter(m => m.status === 'warning').length;
    const criticalModels = models.filter(m => m.status === 'critical').length;

    return {
      healthyServices,
      totalServices,
      avgLatency,
      healthyModels,
      warningModels,
      criticalModels,
      totalAlerts: alertsData?.active_count || 0,
    };
  }, [services, models, alertsData?.active_count]);

  // Refresh handler
  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true);
    await Promise.all([
      refetchAlerts(),
      refetchRuns(),
      refetchHealth(),
      refetchComponents(),
      refetchModels(),
    ]);
    setLastRefresh(new Date());
    setIsRefreshing(false);
  }, [refetchAlerts, refetchRuns, refetchHealth, refetchComponents, refetchModels]);

  // Auto-refresh timestamp update
  useEffect(() => {
    const interval = setInterval(() => {
      setLastRefresh(new Date());
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  const isLoading = isLoadingAlerts || isLoadingRuns;

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Page Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">System Health</h1>
          <p className="text-[var(--color-muted-foreground)]">
            Comprehensive system monitoring with health scores
          </p>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-sm text-[var(--color-muted-foreground)]">
            Last updated: {lastRefresh.toLocaleTimeString()}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={isRefreshing}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Overall Health Score Card */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
        <Card className="md:col-span-1">
          <CardHeader className="pb-2">
            <CardDescription>Overall Health</CardDescription>
          </CardHeader>
          <CardContent>
            {healthScore === null ? (
              <p className="text-sm text-[var(--color-muted-foreground)]">
                Awaiting health check…
              </p>
            ) : (
              <>
                <div className="flex items-center gap-4">
                  <div className="text-4xl font-bold">{healthScore}</div>
                  {healthGrade !== null && (
                    <div className={`px-3 py-1 rounded-lg border text-xl font-bold ${GRADE_COLORS[healthGrade] || GRADE_COLORS[HealthGrade.C]}`}>
                      {healthGrade}
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-1 mt-2 text-sm text-[var(--color-muted-foreground)]">
                  {healthTrend === 'improving' && <TrendingUp className="h-4 w-4 text-emerald-500" />}
                  {healthTrend === 'declining' && <TrendingDown className="h-4 w-4 text-rose-500" />}
                  {healthTrend === 'stable' && <Minus className="h-4 w-4 text-slate-500" />}
                  {healthTrend ? healthTrend.charAt(0).toUpperCase() + healthTrend.slice(1) : 'Unknown'}
                </div>
                {healthCheckedAt && (
                  <div className="mt-1 text-xs text-[var(--color-muted-foreground)]">
                    {`Health check: ${new Date(healthCheckedAt).toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}`}
                  </div>
                )}
                {healthProvenance && healthProvenance !== 'measured' && (
                  <Badge
                    variant="outline"
                    className="mt-2 text-xs text-amber-600 border-amber-300"
                  >
                    provenance: {healthProvenance}
                  </Badge>
                )}
                {healthSummary && (
                  <p className="mt-2 text-xs text-[var(--color-muted-foreground)] whitespace-pre-line">
                    {healthSummary}
                  </p>
                )}
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Services</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {healthStats.healthyServices}/{healthStats.totalServices}
              <StatusDot status="healthy" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Avg latency: {healthStats.avgLatency != null ? `${healthStats.avgLatency}ms` : '—'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Models</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {healthStats.healthyModels} / {models.length}
              {healthStats.warningModels > 0 && (
                <Badge variant="outline" className="text-amber-600 border-amber-300">
                  {healthStats.warningModels} warn
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              {healthStats.criticalModels} critical
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Agents</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {agents.filter(a => a.available).length} / {agents.length}
              <Bot className="h-5 w-5 text-[var(--color-muted-foreground)]" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              {agents.filter(a => !a.available).length} unavailable
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Active Alerts</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {healthStats.totalAlerts}
              {healthStats.totalAlerts > 0 ? (
                <AlertCircle className="h-5 w-5 text-amber-500" />
              ) : (
                <CheckCircle2 className="h-5 w-5 text-emerald-500" />
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              {healthStats.totalAlerts === 0 ? 'All clear' : 'Requires attention'}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Tabs for different health views */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5 lg:w-auto lg:inline-flex">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="agents">Agents</TabsTrigger>
          <TabsTrigger value="pipelines">Pipelines</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          {/* Component Score Breakdown + Health History */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="h-5 w-5" />
                  Component Scores
                </CardTitle>
                <CardDescription>Health breakdown by system component</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={componentScoreData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" domain={[0, 100]} />
                    <YAxis type="category" dataKey="name" width={80} />
                    <Tooltip formatter={(value) => [`${value ?? 0}%`, 'Score']} />
                    <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                      {componentScoreData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5" />
                  Health Trend
                </CardTitle>
                <CardDescription>7-day health score history</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={historyChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis domain={[60, 100]} />
                    <Tooltip formatter={(value) => [`${value ?? 0}`, 'Health Score']} />
                    <Line
                      type="monotone"
                      dataKey="score"
                      stroke="#10b981"
                      strokeWidth={2}
                      dot={{ fill: '#10b981', strokeWidth: 2 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Services and Models Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Services Status */}
            <Card className="lg:col-span-1">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Server className="h-5 w-5" />
                  Service Status
                </CardTitle>
                <CardDescription>Infrastructure components</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {services.length === 0 ? (
                  <EmptyState
                    title="No service status available"
                    description="No component health was reported. Service status appears here once measured."
                  />
                ) : (
                services.map((service) => {
                  const Icon = service.icon;
                  return (
                    <div
                      key={service.name}
                      className="flex items-center justify-between p-3 rounded-lg bg-[var(--color-muted)]/50"
                    >
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-[var(--color-background)]">
                          <Icon className="h-4 w-4 text-[var(--color-muted-foreground)]" />
                        </div>
                        <div>
                          <p className="font-medium text-sm">{service.name}</p>
                          {service.latencyMs !== undefined && (
                            <p className="text-xs text-[var(--color-muted-foreground)]">
                              {service.latencyMs}ms
                            </p>
                          )}
                        </div>
                      </div>
                      <StatusBadge status={mapHealthToStatus(service.status)} size="sm" />
                    </div>
                  );
                }))}
              </CardContent>
            </Card>

            {/* Model Health Cards */}
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  Model Health
                </CardTitle>
                <CardDescription>ML model status and live performance</CardDescription>
              </CardHeader>
              <CardContent>
                {models.length === 0 ? (
                  <EmptyState
                    title="No model health data"
                    description="No production model health was reported by the registry. Status and performance appear here once measured."
                  />
                ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                  {models.map((model) => (
                    <div
                      key={model.modelId}
                      className="p-4 rounded-lg border border-[var(--color-border)] bg-[var(--color-card)]"
                    >
                      <div className="flex items-start justify-between mb-3 gap-2">
                        <div className="min-w-0">
                          <h4 className="font-semibold text-sm truncate">{model.name}</h4>
                          <p className="text-xs text-[var(--color-muted-foreground)] truncate">
                            {model.modelId}
                          </p>
                        </div>
                        {/* Accuracy is the only measured 0-1 metric the /models
                            endpoint exposes; drive the ring off it when present,
                            otherwise show a status badge (no fabricated ring). */}
                        {model.accuracy !== null ? (
                          <ProgressRing
                            value={Math.round(model.accuracy * 100)}
                            size={48}
                            strokeWidth={4}
                            status={model.status}
                          />
                        ) : (
                          <StatusBadge status={mapHealthToStatus(model.status)} size="sm" />
                        )}
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-[var(--color-muted-foreground)]">Accuracy</span>
                          <span>
                            {model.accuracy !== null ? `${(model.accuracy * 100).toFixed(0)}%` : '—'}
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-[var(--color-muted-foreground)]">Error rate</span>
                          <span className={model.errorRate !== null && model.errorRate > 0.1 ? 'text-amber-600' : ''}>
                            {model.errorRate !== null ? `${(model.errorRate * 100).toFixed(1)}%` : '—'}
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-[var(--color-muted-foreground)]">Predictions (24h)</span>
                          <span>
                            {model.predictions24h !== null ? model.predictions24h.toLocaleString() : '—'}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Agents Tab */}
        <TabsContent value="agents" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bot className="h-5 w-5" />
                Agent Health by Tier
              </CardTitle>
              <CardDescription>21-agent tiered orchestration system status</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {Object.entries(agentsByTier).sort((a, b) => parseInt(a[0]) - parseInt(b[0])).map(([tier, tierAgents]) => (
                <div key={tier}>
                  <div className="flex items-center gap-2 mb-3">
                    <Badge variant="outline">Tier {tier}</Badge>
                    <span className="text-sm font-medium">{TIER_NAMES[parseInt(tier)] || 'Unknown'}</span>
                    <span className="text-xs text-[var(--color-muted-foreground)]">
                      ({tierAgents.filter(a => a.available).length}/{tierAgents.length} available)
                    </span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {tierAgents.map(agent => (
                      <div
                        key={agent.agent_name}
                        className={`p-4 rounded-lg border ${
                          agent.available
                            ? 'border-emerald-200 bg-emerald-50/50 dark:bg-emerald-950/20'
                            : 'border-rose-200 bg-rose-50/50 dark:bg-rose-950/20'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-sm">{agent.agent_name}</span>
                          <StatusDot status={agent.available ? 'healthy' : 'error'} />
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs text-[var(--color-muted-foreground)]">
                          <div>
                            <p>Latency</p>
                            <p className="font-medium text-[var(--color-foreground)]">{agent.avg_latency_ms != null ? `${agent.avg_latency_ms}ms` : '—'}</p>
                          </div>
                          <div>
                            <p>Success</p>
                            <p className="font-medium text-[var(--color-foreground)]">{agent.success_rate != null ? `${(agent.success_rate * 100).toFixed(0)}%` : '—'}</p>
                          </div>
                          <div className="col-span-2">
                            <p>24h Invocations: <span className="font-medium text-[var(--color-foreground)]">{agent.invocations_24h}</span></p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Pipelines Tab */}
        <TabsContent value="pipelines" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Workflow className="h-5 w-5" />
                Data Pipeline Health
              </CardTitle>
              <CardDescription>ETL and data processing pipeline status</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {pipelines.map(pipeline => (
                  <div
                    key={pipeline.pipeline_name}
                    className="flex items-center justify-between p-4 rounded-lg border border-[var(--color-border)]"
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium">{pipeline.pipeline_name}</span>
                        <StatusBadge
                          status={
                            pipeline.status === 'healthy' ? 'healthy' :
                            pipeline.status === 'stale' ? 'warning' : 'error'
                          }
                          size="sm"
                        />
                      </div>
                      <p className="text-sm text-[var(--color-muted-foreground)]">
                        Last run: {formatRelativeTime(pipeline.last_run)} | Rows: {pipeline.rows_processed.toLocaleString()}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium">
                        {pipeline.freshness_hours < 1
                          ? `${Math.round(pipeline.freshness_hours * 60)}m`
                          : `${pipeline.freshness_hours.toFixed(1)}h`} fresh
                      </p>
                      <p className="text-xs text-[var(--color-muted-foreground)]">Data freshness</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Health Score History
              </CardTitle>
              <CardDescription>Historical health check records</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={historyChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis domain={[60, 100]} />
                  <Tooltip
                    formatter={(value, name) => [value ?? 0, name === 'score' ? 'Health Score' : name]}
                    labelFormatter={(label) => `Date: ${label}`}
                  />
                  <Line
                    type="monotone"
                    dataKey="score"
                    stroke="#10b981"
                    strokeWidth={2}
                    dot={{ fill: '#10b981', strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
              <div className="mt-4 grid grid-cols-3 gap-4 text-center">
                <div className="p-3 rounded-lg bg-[var(--color-muted)]/50">
                  <p className="text-sm text-[var(--color-muted-foreground)]">Average</p>
                  <p className="text-2xl font-bold">{healthHistoryData?.avg_health_score?.toFixed(1) ?? '—'}</p>
                </div>
                <div className="p-3 rounded-lg bg-[var(--color-muted)]/50">
                  <p className="text-sm text-[var(--color-muted-foreground)]">Trend</p>
                  <p className="text-2xl font-bold flex items-center justify-center gap-1">
                    {healthTrend === 'improving' && <TrendingUp className="h-5 w-5 text-emerald-500" />}
                    {healthTrend === 'declining' && <TrendingDown className="h-5 w-5 text-rose-500" />}
                    {healthTrend === 'stable' && <Minus className="h-5 w-5 text-slate-500" />}
                    {healthTrend ? healthTrend.charAt(0).toUpperCase() + healthTrend.slice(1) : 'Unknown'}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-[var(--color-muted)]/50">
                  <p className="text-sm text-[var(--color-muted-foreground)]">Total Checks</p>
                  <p className="text-2xl font-bold">{healthHistoryData?.total_checks ?? healthHistory.length}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Alerts Tab */}
        <TabsContent value="alerts" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertCircle className="h-5 w-5" />
                Active Alerts
              </CardTitle>
              <CardDescription>Recent alerts requiring attention</CardDescription>
            </CardHeader>
            <CardContent>
              <AlertList
                alerts={alerts}
                compact={false}
                maxItems={10}
                isLoading={isLoading}
                emptyMessage="No active alerts - all systems operational"
              />
            </CardContent>
          </Card>

          {/* Issues and Recommendations from Health Check */}
          {fullHealthData && (fullHealthData.critical_issues?.length > 0 || fullHealthData.warnings?.length > 0) && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {fullHealthData.critical_issues?.length > 0 && (
                <Card className="border-rose-200">
                  <CardHeader>
                    <CardTitle className="text-rose-600">Critical Issues</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2">
                      {fullHealthData.critical_issues.map((issue, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm">
                          <AlertCircle className="h-4 w-4 text-rose-500 mt-0.5 flex-shrink-0" />
                          {issue}
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}
              {fullHealthData.warnings?.length > 0 && (
                <Card className="border-amber-200">
                  <CardHeader>
                    <CardTitle className="text-amber-600">Warnings</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2">
                      {fullHealthData.warnings.map((warning, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm">
                          <AlertCircle className="h-4 w-4 text-amber-500 mt-0.5 flex-shrink-0" />
                          {warning}
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}
            </div>
          )}

          {(fullHealthData?.recommendations?.length ?? 0) > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                  Recommendations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {fullHealthData?.recommendations?.map((rec, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm">
                      <CheckCircle2 className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                      {rec}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default SystemHealth;
