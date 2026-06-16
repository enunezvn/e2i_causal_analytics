/**
 * Model Performance Page
 * ======================
 *
 * Dashboard for analyzing ML model performance with metrics,
 * confusion matrix, ROC curves, and performance trends.
 *
 * Features:
 * - Live model selector populated from useModelsStatus (/api/predictions/models/status)
 * - Live performance trend via usePerformanceTrend (/api/monitoring/performance/{id}/trend)
 * - Live performance alerts via usePerformanceAlerts (/api/monitoring/performance/{id}/alerts)
 * - Live A/B comparison via useModelComparison (/api/monitoring/performance/{id}/compare/{other})
 * - Loading skeletons and QueryErrorState for failures
 *
 * Wires Issue #298 (PR #293 connectivity audit fix).
 *
 * @module pages/ModelPerformance
 */

import { useState, useMemo } from 'react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { QueryErrorState } from '@/components/ui/query-error-state';
import {
  Target,
  Activity,
  RefreshCw,
  Download,
  Clock,
  AlertTriangle,
} from 'lucide-react';
import {
  MetricTrend,
  type MetricDataPoint,
} from '@/components/visualizations';
import { KPICard } from '@/components/visualizations/dashboard';
import {
  usePerformanceTrend,
  usePerformanceAlerts,
  useModelComparison,
} from '@/hooks/api/use-monitoring';
import { useModelsStatus } from '@/hooks/api/use-predictions';
import type { ModelEndpointHealth } from '@/types/predictions';
import type { PerformanceAlertItem, PerformanceMetricItem } from '@/types/monitoring';

// =============================================================================
// HELPERS
// =============================================================================

/**
 * Inline loading skeleton — a small div w/ `animate-pulse` rather than a
 * shared component. Marked with `data-loading="true"` so tests can assert.
 */
function LoadingPulse({ className = 'h-10 w-full' }: { className?: string }) {
  return (
    <div
      className={`bg-muted rounded animate-pulse ${className}`}
      data-loading="true"
    />
  );
}

/**
 * Status badge for a single model endpoint.
 *
 * Maps backend health status -> visual variant.
 */
function ModelHealthStatusBadge({ status }: { status: string }) {
  const normalized = status.toLowerCase();
  if (normalized === 'healthy') {
    return (
      <Badge variant="default" className="bg-emerald-500">
        Healthy
      </Badge>
    );
  }
  if (normalized === 'unhealthy') {
    return <Badge variant="destructive">Unhealthy</Badge>;
  }
  return <Badge variant="outline">{status}</Badge>;
}

/**
 * Convert backend PerformanceMetricItem[] into MetricTrend's MetricDataPoint[].
 *
 * Backend gives us `metric_value` and `recorded_at`; the chart expects
 * `value` + `timestamp`.
 */
function toMetricDataPoints(history: PerformanceMetricItem[] | undefined): MetricDataPoint[] {
  if (!history || history.length === 0) return [];
  return history.map((item) => ({
    timestamp: item.recorded_at,
    value: item.metric_value,
  }));
}

/**
 * Inline page-local model selector — NOT exported.
 *
 * Per Issue #298 implementation guidance, this is purposely page-local
 * (named ModelPerformanceModelSelector) to avoid premature abstraction.
 */
function ModelPerformanceModelSelector({
  models,
  selectedModelId,
  onSelectModel,
  disabled,
}: {
  models: ModelEndpointHealth[];
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
  disabled?: boolean;
}) {
  return (
    <Select value={selectedModelId} onValueChange={onSelectModel} disabled={disabled}>
      <SelectTrigger className="w-[280px]">
        <SelectValue placeholder="Select a model" />
      </SelectTrigger>
      <SelectContent>
        {models.map((model) => (
          <SelectItem key={model.model_name} value={model.model_name}>
            <div className="flex items-center gap-2">
              <span>{model.model_name}</span>
              <span className="text-xs text-muted-foreground capitalize">{model.status}</span>
            </div>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

function ModelPerformance() {
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [compareModelId, setCompareModelId] = useState<string>('');

  // -- Live data ----------------------------------------------------------
  const modelsQuery = useModelsStatus();
  // Stabilise `models` reference so memo deps don't change on every render.
  const models = useMemo<ModelEndpointHealth[]>(
    () => modelsQuery.data?.models ?? [],
    [modelsQuery.data?.models]
  );

  // Auto-select first model once data lands.
  // If `selectedModelId` is non-empty but no longer present in `models`
  // (e.g. backend removed it), fall back to `models[0]` instead of keeping
  // the trend/alerts/comparison hooks pointing at a stale id.
  const effectiveModelId = useMemo(() => {
    if (selectedModelId && models.some((m) => m.model_name === selectedModelId)) {
      return selectedModelId;
    }
    return models[0]?.model_name ?? '';
  }, [selectedModelId, models]);

  const selectedModel = useMemo(
    () => models.find((m) => m.model_name === effectiveModelId),
    [models, effectiveModelId]
  );

  // Performance metrics are recorded ~monthly (backtest sweep), so a 30-day
  // window catches only ~1-2 points and the trend chart renders degenerate /
  // empty. Use a 1-year window to surface the full accuracy-over-time history
  // (the backend's own default is also 365). Cards (current/baseline/trend)
  // come from the tracker independently of this window.
  const trendQuery = usePerformanceTrend(
    { model_id: effectiveModelId, metric_name: 'accuracy', days: 365 },
    { enabled: !!effectiveModelId }
  );

  const alertsQuery = usePerformanceAlerts(effectiveModelId, {
    enabled: !!effectiveModelId,
  });

  // Validate the comparison id the same way as the primary: must exist in
  // the live model list AND differ from the primary. Anything else means
  // the comparison query stays disabled (avoids self-comparison + stale-id
  // requests after the primary changes).
  const effectiveCompareModelId = useMemo(() => {
    if (!compareModelId) return '';
    if (compareModelId === effectiveModelId) return '';
    if (!models.some((m) => m.model_name === compareModelId)) return '';
    return compareModelId;
  }, [compareModelId, effectiveModelId, models]);

  const comparisonQuery = useModelComparison(
    effectiveModelId,
    effectiveCompareModelId,
    'accuracy',
    { enabled: !!effectiveModelId && !!effectiveCompareModelId }
  );

  const accuracyHistory = useMemo(
    () => toMetricDataPoints(trendQuery.data?.history),
    [trendQuery.data?.history]
  );

  // -- Handlers -----------------------------------------------------------
  const handleRefresh = async () => {
    await Promise.all([
      modelsQuery.refetch(),
      trendQuery.refetch?.(),
      alertsQuery.refetch?.(),
      // Comparison may not be enabled — refetch only if effective second model exists
      effectiveCompareModelId ? comparisonQuery.refetch?.() : Promise.resolve(),
    ]);
  };

  const handleExport = () => {
    const exportData = {
      model: selectedModel,
      trend: trendQuery.data,
      alerts: alertsQuery.data,
      comparison: comparisonQuery.data,
      exportedAt: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = `${effectiveModelId || 'model'}-performance.json`;
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);
  };

  // -- Loading / error short-circuits ------------------------------------
  const isModelsLoading = modelsQuery.isLoading;
  const isModelsError = modelsQuery.isError;
  const isTrendLoading = trendQuery.isLoading && !!effectiveModelId;
  const isTrendError = trendQuery.isError && !!effectiveModelId;

  const isRefetching =
    modelsQuery.isRefetching ||
    trendQuery.isRefetching ||
    alertsQuery.isRefetching ||
    comparisonQuery.isRefetching;

  // =============================================================================
  // RENDER
  // =============================================================================
  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">Model Performance</h1>
          <p className="text-muted-foreground">
            View model metrics, confusion matrix, ROC curves, and performance trends.
          </p>
        </div>

        <div className="flex items-center gap-3">
          {isModelsLoading ? (
            <LoadingPulse className="h-10 w-[280px]" />
          ) : (
            <ModelPerformanceModelSelector
              models={models}
              selectedModelId={effectiveModelId}
              onSelectModel={setSelectedModelId}
              disabled={models.length === 0}
            />
          )}

          <Button
            variant="outline"
            size="icon"
            onClick={handleRefresh}
            disabled={isRefetching || !effectiveModelId}
          >
            <RefreshCw className={`h-4 w-4 ${isRefetching ? 'animate-spin' : ''}`} />
          </Button>

          <Button variant="outline" onClick={handleExport} disabled={!effectiveModelId}>
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Models hook error -> top-level error state */}
      {isModelsError && (
        <div className="mb-6">
          <QueryErrorState
            error={modelsQuery.error ?? new Error('Failed to load models')}
            onRetry={() => {
              void modelsQuery.refetch();
            }}
            isRetrying={modelsQuery.isRefetching}
          />
        </div>
      )}

      {/* Empty / no-selection state */}
      {!isModelsLoading && !isModelsError && !effectiveModelId && (
        <Card className="mb-6">
          <CardContent className="pt-6 text-sm text-muted-foreground">
            No registered models available. Please register a model to view performance.
          </CardContent>
        </Card>
      )}

      {/* Model Info Card */}
      {effectiveModelId && selectedModel && (
        <Card className="mb-6">
          <CardContent className="pt-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-primary/10">
                  <Target className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-xl font-semibold">{selectedModel.model_name}</h2>
                    <ModelHealthStatusBadge status={String(selectedModel.status)} />
                  </div>
                  <div className="flex items-center gap-4 text-sm text-muted-foreground mt-1">
                    <span className="flex items-center gap-1">
                      <Activity className="h-4 w-4" />
                      {selectedModel.endpoint}
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock className="h-4 w-4" />
                      Last check: {selectedModel.last_check}
                    </span>
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm text-muted-foreground">Performance trend</div>
                <div className="text-2xl font-bold capitalize">
                  {trendQuery.data?.trend ?? '—'}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Trend error -> contextual error state */}
      {isTrendError && (
        <div className="mb-6">
          <QueryErrorState
            error={trendQuery.error ?? new Error('Failed to load performance trend')}
            onRetry={() => {
              void trendQuery.refetch();
            }}
            isRetrying={trendQuery.isRefetching}
          />
        </div>
      )}

      {/* Trend loading skeleton (KPI block) */}
      {isTrendLoading && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          {[0, 1, 2, 3].map((idx) => (
            <LoadingPulse key={idx} className="h-24 w-full" />
          ))}
        </div>
      )}

      {/* Metrics KPI Cards — wired to live trend */}
      {!isTrendLoading && trendQuery.data && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <KPICard
            title={`Current ${trendQuery.data.metric_name}`}
            value={(trendQuery.data.current_value * 100).toFixed(1)}
            unit="%"
            status={trendQuery.data.alert_threshold_breached ? 'critical' : 'healthy'}
          />
          <KPICard
            title={`Baseline ${trendQuery.data.metric_name}`}
            value={(trendQuery.data.baseline_value * 100).toFixed(1)}
            unit="%"
            status="healthy"
          />
          <KPICard
            title="Change"
            value={trendQuery.data.change_percent.toFixed(2)}
            unit="%"
            status={trendQuery.data.trend === 'degrading' ? 'critical' : 'healthy'}
          />
          <KPICard
            title="Trend"
            value={trendQuery.data.trend}
            status={trendQuery.data.trend === 'degrading' ? 'critical' : 'healthy'}
          />
        </div>
      )}

      {/* Visualization Tabs */}
      <Tabs defaultValue="trend" className="space-y-6">
        <TabsList>
          <TabsTrigger value="trend">Performance Trend</TabsTrigger>
          <TabsTrigger value="alerts">Performance Alerts</TabsTrigger>
          <TabsTrigger value="compare">Comparison</TabsTrigger>
          <TabsTrigger value="confusion">Confusion Matrix</TabsTrigger>
          <TabsTrigger value="roc">ROC Curve</TabsTrigger>
        </TabsList>

        <TabsContent value="trend">
          <Card>
            <CardHeader>
              <CardTitle>Performance Trend</CardTitle>
              <CardDescription>
                {trendQuery.data?.metric_name ?? 'Accuracy'} over time, with baseline + alert thresholds.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isTrendLoading ? (
                <LoadingPulse className="h-[350px] w-full" />
              ) : accuracyHistory.length > 0 ? (
                <MetricTrend
                  name={trendQuery.data?.metric_name ?? 'accuracy'}
                  data={accuracyHistory}
                  unit=""
                  height={350}
                  showHeader={false}
                  thresholds={
                    trendQuery.data
                      ? [
                          {
                            value: trendQuery.data.baseline_value,
                            label: 'Baseline',
                            type: 'target' as const,
                            color: '#22c55e',
                          },
                          // Alert-threshold line: the level below which a
                          // performance alert fires. Only plotted when the API
                          // reports a real (>0) threshold (it is 0 when there
                          // is no baseline history to derive it from).
                          ...(trendQuery.data.alert_threshold > 0
                            ? [
                                {
                                  value: trendQuery.data.alert_threshold,
                                  label: 'Alert threshold',
                                  type: 'lower' as const,
                                  color: '#ef4444',
                                },
                              ]
                            : []),
                        ]
                      : []
                  }
                  valueFormatter={(v) => (v * 100).toFixed(1) + '%'}
                  timestampFormatter={(ts) => {
                    const date = new Date(ts);
                    return date.toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                    });
                  }}
                />
              ) : (
                <div className="py-12 text-center text-sm text-muted-foreground">
                  No performance history available for this model yet.
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="alerts">
          <Card>
            <CardHeader>
              <CardTitle>Performance Alerts</CardTitle>
              <CardDescription>
                Alerts emitted when metric drift exceeds configured thresholds.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {alertsQuery.isLoading ? (
                <LoadingPulse className="h-32 w-full" />
              ) : alertsQuery.isError ? (
                <QueryErrorState
                  error={alertsQuery.error ?? new Error('Failed to load alerts')}
                  onRetry={() => {
                    void alertsQuery.refetch();
                  }}
                  isRetrying={alertsQuery.isRefetching}
                />
              ) : alertsQuery.data && alertsQuery.data.alert_count > 0 ? (
                <div className="space-y-3">
                  {alertsQuery.data.alerts.map((alert: PerformanceAlertItem, idx: number) => (
                    <div
                      key={`${alert.metric_name}-${idx}`}
                      className="flex items-start gap-3 p-3 rounded-md border"
                    >
                      <AlertTriangle className="h-4 w-4 mt-1 text-amber-500" />
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium capitalize">{alert.metric_name}</span>
                          <Badge variant="outline" className="capitalize">
                            {alert.severity}
                          </Badge>
                        </div>
                        <div className="text-sm text-muted-foreground mt-1">{alert.message}</div>
                        <div className="text-xs text-muted-foreground mt-1">
                          {alert.current_value.toFixed(3)} vs baseline{' '}
                          {alert.baseline_value.toFixed(3)} ({alert.change_percent.toFixed(2)}%)
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="py-8 text-center text-sm text-muted-foreground">
                  No active performance alerts.
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="compare">
          <Card>
            <CardHeader>
              <CardTitle>Model Comparison</CardTitle>
              <CardDescription>
                Compare the selected model against another by a chosen metric.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="mb-4 flex items-center gap-3">
                <span className="text-sm text-muted-foreground">Compare with:</span>
                <Select
                  value={compareModelId}
                  onValueChange={setCompareModelId}
                  disabled={models.length < 2}
                >
                  <SelectTrigger className="w-[260px]">
                    <SelectValue placeholder="Select model to compare" />
                  </SelectTrigger>
                  <SelectContent>
                    {models
                      .filter((m) => m.model_name !== effectiveModelId)
                      .map((model) => (
                        <SelectItem key={model.model_name} value={model.model_name}>
                          {model.model_name}
                        </SelectItem>
                      ))}
                  </SelectContent>
                </Select>
              </div>

              {!effectiveCompareModelId ? (
                <div className="py-8 text-center text-sm text-muted-foreground">
                  Pick a second model above to run a comparison.
                </div>
              ) : comparisonQuery.isLoading ? (
                <LoadingPulse className="h-32 w-full" />
              ) : comparisonQuery.isError ? (
                <QueryErrorState
                  error={comparisonQuery.error ?? new Error('Failed to load comparison')}
                  onRetry={() => {
                    void comparisonQuery.refetch();
                  }}
                  isRetrying={comparisonQuery.isRefetching}
                />
              ) : comparisonQuery.data ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <KPICard
                    title={`${comparisonQuery.data.model_id} ${comparisonQuery.data.metric_name}`}
                    value={(comparisonQuery.data.model_value * 100).toFixed(1)}
                    unit="%"
                    status="healthy"
                  />
                  <KPICard
                    title={`${comparisonQuery.data.other_model_id} ${comparisonQuery.data.metric_name}`}
                    value={(comparisonQuery.data.other_model_value * 100).toFixed(1)}
                    unit="%"
                    status="healthy"
                  />
                  <KPICard
                    title="Difference"
                    value={(comparisonQuery.data.difference * 100).toFixed(2)}
                    unit="%"
                    status="healthy"
                  />
                  <KPICard
                    title="Better"
                    value={comparisonQuery.data.better_model}
                    status="healthy"
                  />
                </div>
              ) : null}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="confusion">
          <Card>
            <CardHeader>
              <CardTitle>Confusion Matrix</CardTitle>
              <CardDescription>
                Confusion matrix is not yet exposed by the monitoring API.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="py-8 text-center text-sm text-muted-foreground">
                Confusion matrix data will appear here once the monitoring API exposes
                per-class breakdowns.
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="roc">
          <Card>
            <CardHeader>
              <CardTitle>ROC Curve Comparison</CardTitle>
              <CardDescription>
                ROC curve data is not yet exposed by the monitoring API.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="py-8 text-center text-sm text-muted-foreground">
                ROC curve points will appear here once the monitoring API exposes them.
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default ModelPerformance;
