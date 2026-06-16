/**
 * Monitoring Page
 * ===============
 *
 * Dashboard for the E2I Model Monitoring backend.
 *
 * Live wiring (issue #297):
 *   - `useAlerts`           → recent monitoring alerts (error log feed).
 *   - `useMonitoringRuns`   → recent monitoring runs (runs feed).
 *   - `useModelHealth`      → currently-selected model's health summary.
 *
 * Endpoints (FastAPI): `src/api/routes/monitoring.py`.
 *
 * Loading + error states use `QueryErrorState`, matching `SystemHealth.tsx`.
 *
 * @module pages/Monitoring
 */

import { useState, useMemo, useCallback, useEffect } from 'react';
import {
  Activity,
  RefreshCw,
  Download,
  Clock,
  AlertCircle,
  Search,
  Filter,
  Server,
  BarChart3,
  List,
  Brain,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  AreaChart,
  Area,
} from 'recharts';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { KPICard, StatusBadge } from '@/components/visualizations';
import { QueryErrorState } from '@/components/ui/query-error-state';
import { useAlerts, useMonitoringRuns, useModelHealth } from '@/hooks/api/use-monitoring';
import { useModelsStatus } from '@/hooks/api/use-predictions';
import { AlertStatus } from '@/types/monitoring';
import type { AlertItem, MonitoringRunItem } from '@/types/monitoring';

// =============================================================================
// CONSTANTS
// =============================================================================

/**
 * Model option shown in the monitoring selector.
 *
 * The list is driven live from the backend `/api/models/status` endpoint
 * (registry-backed: `ml_model_registry` production rows) via
 * {@link useModelsStatus}. It is NO LONGER a hardcoded set of fictional
 * handles (`propensity_v2.1.0`, `churn_v1.5.2`, ...), none of which were
 * registered models. See dispatch contract for issue #297.
 */
interface MonitoringModelOption {
  id: string;
  label: string;
}

/**
 * Map a UI time-range string → API `days` parameter.
 *
 * The backend `/api/monitoring/runs` endpoint takes day-resolution windows
 * only, so for sub-day windows (`1h`, `6h`) we fetch the minimum supported
 * day window (1) and then filter client-side via {@link timeRangeToMs}.
 */
function timeRangeToDays(tr: string): number {
  switch (tr) {
    case '1h':
    case '6h':
    case '24h':
      return 1;
    case '7d':
      return 7;
    case '30d':
      return 30;
    default:
      return 1;
  }
}

/**
 * Map a UI time-range string → the corresponding window size in milliseconds.
 *
 * Used for client-side filtering of runs by `started_at`. We always apply this
 * filter defensively for every range, not just `1h`/`6h`. Rationale:
 *
 *   - The backend `/api/monitoring/runs` route currently passes `days` only
 *     loosely (the parameter is accepted but the underlying repository call
 *     `get_recent_runs(model_version, limit)` does not use it as of writing).
 *     Filtering client-side keeps the page truthful regardless.
 *   - This also keeps `1h` and `6h` correct, which the API can't express at
 *     all.
 */
function timeRangeToMs(tr: string): number {
  switch (tr) {
    case '1h':
      return 60 * 60 * 1000;
    case '6h':
      return 6 * 60 * 60 * 1000;
    case '24h':
      return 24 * 60 * 60 * 1000;
    case '7d':
      return 7 * 24 * 60 * 60 * 1000;
    case '30d':
      return 30 * 24 * 60 * 60 * 1000;
    default:
      return 24 * 60 * 60 * 1000;
  }
}

/**
 * Map an alert's severity string to a UI level used for styling.
 * Centralised so error-log severity styles stay consistent with API values.
 */
function severityToErrorLevel(severity: string): 'critical' | 'error' | 'warning' {
  const s = severity.toLowerCase();
  if (s === 'critical') return 'critical';
  if (s === 'high') return 'error';
  return 'warning';
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function formatNumber(num: number): string {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num.toString();
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleString();
}

function getErrorLevelStyle(level: string): { bg: string; text: string } {
  switch (level) {
    case 'critical':
      return { bg: 'bg-red-100 dark:bg-red-900/30', text: 'text-red-700 dark:text-red-400' };
    case 'error':
      return { bg: 'bg-rose-100 dark:bg-rose-900/30', text: 'text-rose-700 dark:text-rose-400' };
    case 'warning':
      return { bg: 'bg-amber-100 dark:bg-amber-900/30', text: 'text-amber-700 dark:text-amber-400' };
    default:
      return { bg: 'bg-gray-100', text: 'text-gray-700' };
  }
}

// =============================================================================
// COMPONENT
// =============================================================================

function Monitoring() {
  const [timeRange, setTimeRange] = useState<string>('24h');
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState('');
  const [errorLevelFilter, setErrorLevelFilter] = useState<string>('all');

  const days = useMemo(() => timeRangeToDays(timeRange), [timeRange]);

  // --- Registry-driven model selector ----------------------------------------
  // The list of selectable models comes from the backend `/api/models/status`
  // endpoint, which is backed by `ml_model_registry` production rows. No more
  // hardcoded fictional handles.
  const { data: modelsStatus, isLoading: isLoadingModels } = useModelsStatus();
  const monitoringModels: MonitoringModelOption[] = useMemo(
    () =>
      (modelsStatus?.models ?? []).map((m) => ({
        id: m.model_name,
        label: m.model_name,
      })),
    [modelsStatus?.models]
  );

  // Default the selection to the first registered model once it resolves.
  useEffect(() => {
    if (!selectedModelId && monitoringModels.length > 0) {
      setSelectedModelId(monitoringModels[0].id);
    }
  }, [selectedModelId, monitoringModels]);

  // --- Live data hooks --------------------------------------------------------
  const {
    data: alertsData,
    isLoading: isLoadingAlerts,
    isError: isErrorAlerts,
    error: alertsError,
    refetch: refetchAlerts,
    isFetching: isFetchingAlerts,
  } = useAlerts(
    { model_id: selectedModelId, status: AlertStatus.ACTIVE, limit: 50 },
    { refetchInterval: 30_000 }
  );

  const {
    data: runsData,
    isLoading: isLoadingRuns,
    isError: isErrorRuns,
    error: runsError,
    refetch: refetchRuns,
    isFetching: isFetchingRuns,
  } = useMonitoringRuns({ model_id: selectedModelId, days, limit: 100 });

  const {
    data: healthData,
    isLoading: isLoadingHealth,
    isError: isErrorHealth,
    error: healthError,
    refetch: refetchHealth,
    isFetching: isFetchingHealth,
  } = useModelHealth(selectedModelId);

  // --- Derived state ----------------------------------------------------------
  const alerts: AlertItem[] = useMemo(() => alertsData?.alerts ?? [], [alertsData?.alerts]);

  /**
   * Runs from the backend further filtered client-side by `started_at` so
   * the rendered set always reflects the user's chosen time window (see
   * {@link timeRangeToMs} for the rationale).
   */
  const runs: MonitoringRunItem[] = useMemo(() => {
    const fetched = runsData?.runs ?? [];
    const windowMs = timeRangeToMs(timeRange);
    const cutoff = Date.now() - windowMs;
    return fetched.filter((r) => {
      const t = new Date(r.started_at).getTime();
      return Number.isFinite(t) && t >= cutoff;
    });
  }, [runsData?.runs, timeRange]);

  /**
   * Build the per-period request/drift/alert telemetry from the live runs feed.
   * One bucket per run; runs from the API arrive newest-first, we reverse so
   * the chart reads left-to-right by time.
   */
  const runTelemetry = useMemo(() => {
    return runs
      .slice()
      .reverse()
      .map((run) => ({
        timestamp: new Date(run.started_at).toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
        }),
        features_checked: run.features_checked,
        drift_detected_count: run.drift_detected_count,
        alerts_generated: run.alerts_generated,
        duration_ms: run.duration_ms,
      }));
  }, [runs]);

  /**
   * Aggregate overview metrics from the live runs + alerts feeds.
   * Mirrors the prior overview cards but with live values.
   */
  const overviewMetrics = useMemo(() => {
    const totalFeatures = runs.reduce((sum, r) => sum + (r.features_checked ?? 0), 0);
    const totalDrift = runs.reduce((sum, r) => sum + (r.drift_detected_count ?? 0), 0);
    const avgDuration =
      runs.length > 0
        ? Math.round(
            runs.reduce((sum, r) => sum + (r.duration_ms ?? 0), 0) / runs.length,
          )
        : 0;
    const driftRate = totalFeatures > 0 ? (totalDrift / totalFeatures) * 100 : 0;
    const driftScore = Math.round((healthData?.drift_score ?? 0) * 100);

    // totalRuns reflects the *displayed* runs, i.e. after the client-side
    // sub-day filter has been applied. Otherwise the KPI would show the
    // server's day-bucket total while charts/tables show the narrower
    // sub-day window — they must agree (see codex iter-2 MED finding).
    return {
      totalFeatures,
      totalDrift,
      avgDuration,
      driftRate: driftRate.toFixed(2),
      driftScore,
      activeAlerts: alertsData?.active_count ?? alerts.length,
      totalRuns: runs.length,
    };
  }, [runs, alerts.length, alertsData?.active_count, healthData?.drift_score]);

  // Filter alerts for the error-log tab.
  const filteredAlerts = useMemo(() => {
    return alerts.filter((alert) => {
      const level = severityToErrorLevel(alert.severity);
      const matchesLevel = errorLevelFilter === 'all' || level === errorLevelFilter;
      const matchesSearch =
        searchQuery === '' ||
        alert.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        alert.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        alert.alert_type.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesLevel && matchesSearch;
    });
  }, [alerts, errorLevelFilter, searchQuery]);

  const isAnyLoading = isLoadingAlerts || isLoadingRuns || isLoadingHealth;
  const isAnyFetching = isFetchingAlerts || isFetchingRuns || isFetchingHealth;

  const handleRefresh = useCallback(() => {
    void Promise.all([refetchAlerts(), refetchRuns(), refetchHealth()]);
  }, [refetchAlerts, refetchRuns, refetchHealth]);

  const handleExport = () => {
    const report = {
      generatedAt: new Date().toISOString(),
      timeRange,
      modelId: selectedModelId,
      overview: overviewMetrics,
      runs,
      alerts,
      health: healthData,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `monitoring-report-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Aggregate error from the three queries; the first non-null wins.
  const firstError = alertsError ?? runsError ?? healthError ?? null;
  const isAnyError = isErrorAlerts || isErrorRuns || isErrorHealth;

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Activity className="h-8 w-8" />
            Monitoring
          </h1>
          <p className="text-muted-foreground">
            User activity logs, API usage statistics, error tracking, and performance metrics.
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {/* Model selector — driven from ml_model_registry via /api/models/status */}
          <Select
            value={selectedModelId}
            onValueChange={setSelectedModelId}
            disabled={isLoadingModels || monitoringModels.length === 0}
          >
            <SelectTrigger className="w-56" aria-label="Model">
              <Brain className="h-4 w-4 mr-2" />
              <SelectValue
                placeholder={
                  isLoadingModels
                    ? 'Loading models…'
                    : monitoringModels.length === 0
                      ? 'No registered models'
                      : 'Model'
                }
              />
            </SelectTrigger>
            <SelectContent>
              {monitoringModels.map((model) => (
                <SelectItem key={model.id} value={model.id}>
                  {model.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-32" aria-label="Time Range">
              <Clock className="h-4 w-4 mr-2" />
              <SelectValue placeholder="Time Range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1h">Last Hour</SelectItem>
              <SelectItem value="6h">Last 6 Hours</SelectItem>
              <SelectItem value="24h">Last 24 Hours</SelectItem>
              <SelectItem value="7d">Last 7 Days</SelectItem>
              <SelectItem value="30d">Last 30 Days</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" onClick={handleRefresh} disabled={isAnyFetching}>
            <RefreshCw
              className={`h-4 w-4 mr-2 ${isAnyFetching ? 'animate-spin' : ''}`}
            />
            Refresh
          </Button>
          <Button variant="outline" onClick={handleExport}>
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Loading state */}
      {isAnyLoading && (
        <div
          role="status"
          aria-busy="true"
          className="flex items-center gap-2 mb-6 text-muted-foreground"
        >
          <RefreshCw className="h-4 w-4 animate-spin" />
          <span>Loading monitoring data…</span>
        </div>
      )}

      {/* Error state */}
      {isAnyError && firstError && (
        <div className="mb-6">
          <QueryErrorState
            error={firstError as Error}
            onRetry={handleRefresh}
            isRetrying={isAnyFetching}
          />
        </div>
      )}

      {/* Overview Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
        <KPICard
          title="Total Runs"
          value={overviewMetrics.totalRuns}
          status="healthy"
          description="Monitoring runs in selected period"
          sparklineData={runTelemetry.map((m) => m.features_checked)}
          size="sm"
        />
        <KPICard
          title="Drift Rate"
          value={parseFloat(overviewMetrics.driftRate)}
          unit="%"
          status={parseFloat(overviewMetrics.driftRate) < 5 ? 'healthy' : 'warning'}
          description="Features with drift / features checked"
          sparklineData={runTelemetry.map((m) => m.drift_detected_count)}
          higherIsBetter={false}
          size="sm"
        />
        <KPICard
          title="Avg Run Duration"
          value={Math.round(overviewMetrics.avgDuration / 1000)}
          unit="s"
          status={overviewMetrics.avgDuration < 60_000 ? 'healthy' : 'warning'}
          description="Mean run duration"
          sparklineData={runTelemetry.map((m) => m.duration_ms)}
          higherIsBetter={false}
          size="sm"
        />
        <KPICard
          title="Active Alerts"
          value={overviewMetrics.activeAlerts}
          status={overviewMetrics.activeAlerts === 0 ? 'healthy' : 'warning'}
          description="Currently active alerts"
          higherIsBetter={false}
          size="sm"
        />
        <KPICard
          title="Drift Events"
          value={overviewMetrics.totalDrift}
          status={overviewMetrics.totalDrift < 5 ? 'healthy' : 'warning'}
          description="Drift events detected"
          sparklineData={runTelemetry.map((m) => m.drift_detected_count)}
          higherIsBetter={false}
          size="sm"
        />
        <KPICard
          title="Health Score"
          value={overviewMetrics.driftScore}
          unit="%"
          status={
            healthData?.overall_health === 'healthy'
              ? 'healthy'
              : healthData?.overall_health === 'critical'
                ? 'critical'
                : 'warning'
          }
          description="Drift score (×100)"
          higherIsBetter={false}
          size="sm"
        />
      </div>

      {/* Tabs for different views */}
      <Tabs defaultValue="api" className="space-y-4">
        <TabsList>
          <TabsTrigger value="api" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Drift Trend
          </TabsTrigger>
          <TabsTrigger value="activity" className="flex items-center gap-2">
            <List className="h-4 w-4" />
            Runs
          </TabsTrigger>
          <TabsTrigger value="errors" className="flex items-center gap-2">
            <AlertCircle className="h-4 w-4" />
            Errors
            <Badge variant="destructive" className="ml-1">
              {alerts.filter((a) => {
                const lvl = severityToErrorLevel(a.severity);
                return lvl === 'critical' || lvl === 'error';
              }).length}
            </Badge>
          </TabsTrigger>
          <TabsTrigger value="system" className="flex items-center gap-2">
            <Server className="h-4 w-4" />
            System
          </TabsTrigger>
        </TabsList>

        {/* Drift Trend Tab — per-run features-checked vs drift-detected (NOT API
            usage; this was mislabeled "API Usage" while charting drift telemetry). */}
        <TabsContent value="api" className="space-y-4">
          {/* Run Volume / Drift Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Features Checked &amp; Drift Detected
              </CardTitle>
              <CardDescription>Per-run features checked vs drift events</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={runTelemetry}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="timestamp" stroke="var(--muted-foreground)" fontSize={12} />
                    <YAxis stroke="var(--muted-foreground)" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'var(--card)',
                        border: '1px solid var(--border)',
                        borderRadius: '8px',
                      }}
                    />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="features_checked"
                      stroke="#10b981"
                      fill="#10b981"
                      fillOpacity={0.2}
                      name="Features Checked"
                    />
                    <Area
                      type="monotone"
                      dataKey="drift_detected_count"
                      stroke="#ef4444"
                      fill="#ef4444"
                      fillOpacity={0.3}
                      name="Drift Detected"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Duration Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Run Duration</CardTitle>
              <CardDescription>Per-run duration over time</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={runTelemetry}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="timestamp" stroke="var(--muted-foreground)" fontSize={12} />
                    <YAxis stroke="var(--muted-foreground)" fontSize={12} unit="ms" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'var(--card)',
                        border: '1px solid var(--border)',
                        borderRadius: '8px',
                      }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="duration_ms"
                      stroke="#3b82f6"
                      name="Duration (ms)"
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Recent Runs Table */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Runs</CardTitle>
              <CardDescription>Latest monitoring runs for the selected model</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">
                        Run Type
                      </th>
                      <th className="text-left py-3 px-4 font-medium text-muted-foreground">
                        Started
                      </th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">
                        Features
                      </th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">
                        Drift
                      </th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">
                        Alerts
                      </th>
                      <th className="text-right py-3 px-4 font-medium text-muted-foreground">
                        Duration
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.length === 0 ? (
                      <tr>
                        <td
                          colSpan={6}
                          className="py-6 text-center text-sm text-muted-foreground"
                        >
                          No monitoring runs found for the selected window.
                        </td>
                      </tr>
                    ) : (
                      runs.map((run) => (
                        <tr key={run.id} className="border-b border-border hover:bg-muted/50">
                          <td className="py-3 px-4">
                            <Badge variant="outline">{run.run_type}</Badge>
                          </td>
                          <td className="py-3 px-4 text-sm text-muted-foreground">
                            {formatTimestamp(run.started_at)}
                          </td>
                          <td className="py-3 px-4 text-right font-medium">
                            {formatNumber(run.features_checked)}
                          </td>
                          <td className="py-3 px-4 text-right">
                            {run.drift_detected_count > 0 ? (
                              <span className="text-rose-500">
                                {formatNumber(run.drift_detected_count)}
                              </span>
                            ) : (
                              <span className="text-muted-foreground">0</span>
                            )}
                          </td>
                          <td className="py-3 px-4 text-right">
                            {run.alerts_generated > 0 ? (
                              <span className="text-amber-500">{run.alerts_generated}</span>
                            ) : (
                              <span className="text-muted-foreground">0</span>
                            )}
                          </td>
                          <td className="py-3 px-4 text-right">
                            <span
                              className={run.duration_ms > 60_000 ? 'text-amber-500' : ''}
                            >
                              {Math.round(run.duration_ms / 1000)}s
                            </span>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Runs Tab (full feed) */}
        <TabsContent value="activity">
          <Card>
            <CardHeader>
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <List className="h-5 w-5" />
                    Monitoring Runs
                  </CardTitle>
                  <CardDescription>
                    Live feed of monitoring runs from {' '}
                    <code className="text-xs">/api/monitoring/runs</code>
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {runs.length === 0 ? (
                  <p className="py-6 text-center text-sm text-muted-foreground">
                    No monitoring runs found.
                  </p>
                ) : (
                  runs.map((run) => (
                    <div
                      key={run.id}
                      className="p-4 rounded-lg border border-border hover:bg-muted/50"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <Badge variant="outline">{run.run_type}</Badge>
                            <span className="text-xs text-muted-foreground">
                              {run.features_checked} features checked
                            </span>
                          </div>
                          <p className="font-medium">
                            {run.drift_detected_count} drift event
                            {run.drift_detected_count === 1 ? '' : 's'} · {run.alerts_generated}
                            {' '}alert{run.alerts_generated === 1 ? '' : 's'} generated
                          </p>
                          <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              {formatTimestamp(run.started_at)}
                            </span>
                            <span>Duration: {Math.round(run.duration_ms / 1000)}s</span>
                            <span>Model: {run.model_version}</span>
                          </div>
                          {run.error_message && (
                            <p className="mt-2 text-xs text-rose-500">{run.error_message}</p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Errors Tab — live alerts feed */}
        <TabsContent value="errors">
          <Card>
            <CardHeader>
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <AlertCircle className="h-5 w-5" />
                    Alert Feed
                  </CardTitle>
                  <CardDescription>
                    Live alerts from {' '}
                    <code className="text-xs">/api/monitoring/alerts</code>
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Search alerts..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-9 w-64"
                    />
                  </div>
                  <Select value={errorLevelFilter} onValueChange={setErrorLevelFilter}>
                    <SelectTrigger className="w-36" aria-label="Severity Filter">
                      <Filter className="h-4 w-4 mr-2" />
                      <SelectValue placeholder="Level" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Levels</SelectItem>
                      <SelectItem value="critical">Critical</SelectItem>
                      <SelectItem value="error">Error</SelectItem>
                      <SelectItem value="warning">Warning</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {filteredAlerts.length === 0 ? (
                  <p className="py-6 text-center text-sm text-muted-foreground">
                    No alerts found for the selected filters.
                  </p>
                ) : (
                  filteredAlerts.map((alert) => {
                    const level = severityToErrorLevel(alert.severity);
                    const levelStyle = getErrorLevelStyle(level);
                    return (
                      <div
                        key={alert.id}
                        className={`p-4 rounded-lg border border-border ${levelStyle.bg}`}
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <Badge
                                className={`${levelStyle.bg} ${levelStyle.text} uppercase text-xs`}
                              >
                                {level}
                              </Badge>
                              <code className="text-xs bg-muted px-1.5 py-0.5 rounded">
                                {alert.alert_type}
                              </code>
                              <span className="text-xs text-muted-foreground">
                                {alert.model_version}
                              </span>
                            </div>
                            <p className={`font-medium ${levelStyle.text}`}>{alert.title}</p>
                            {alert.description && (
                              <p className="text-sm text-muted-foreground mt-1">
                                {alert.description}
                              </p>
                            )}
                            <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                              <span className="flex items-center gap-1">
                                <Clock className="h-3 w-3" />
                                {formatTimestamp(alert.triggered_at)}
                              </span>
                              <span>Status: {alert.status}</span>
                              {alert.acknowledged_by && (
                                <span>Ack&apos;d by: {alert.acknowledged_by}</span>
                              )}
                            </div>
                          </div>
                          <Button variant="outline" size="sm">
                            View Details
                          </Button>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* System Tab — model health summary */}
        <TabsContent value="system">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Server className="h-5 w-5" />
                  Model Health
                </CardTitle>
                <CardDescription>
                  Health summary from {' '}
                  <code className="text-xs">/api/monitoring/health/{selectedModelId}</code>
                </CardDescription>
              </CardHeader>
              <CardContent>
                {healthData ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Overall Health</span>
                      <StatusBadge
                        status={
                          healthData.overall_health === 'critical'
                            ? 'error'
                            : healthData.overall_health
                        }
                        size="sm"
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Drift Score</span>
                      <span className="font-medium">
                        {(healthData.drift_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Active Alerts</span>
                      <span className="font-medium">{healthData.active_alerts}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Performance Trend</span>
                      <Badge variant="outline">{healthData.performance_trend}</Badge>
                    </div>
                    {healthData.last_retrained && (
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Last Retrained</span>
                        <span className="text-sm">
                          {formatTimestamp(healthData.last_retrained)}
                        </span>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    No model health data available.
                  </p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Recommendations</CardTitle>
                <CardDescription>Actions suggested by the monitoring service</CardDescription>
              </CardHeader>
              <CardContent>
                {healthData?.recommendations && healthData.recommendations.length > 0 ? (
                  <ul className="space-y-2 list-disc list-inside text-sm">
                    {healthData.recommendations.map((rec, i) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-muted-foreground">No recommendations.</p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default Monitoring;
