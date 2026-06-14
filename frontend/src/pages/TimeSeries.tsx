/**
 * Time Series Analysis Page
 * =========================
 *
 * Two-mode time-series page wired to live monitoring + KPI APIs.
 *
 * Modes:
 *  - "Model performance" — `usePerformanceTrend({ model_id, metric_name, days })`
 *  - "KPI history"       — `useKPIValue(kpiId)` (history embedded in `metadata.history`)
 *
 * Issue #302 retired the page's 38 `sample*` constants; all series are
 * sourced from real hooks. Loading / error are surfaced via QueryErrorState.
 *
 * @module pages/TimeSeries
 */

import { useState, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type {
  Formatter as TooltipValueFormatter,
  NameType,
  ValueType,
} from 'recharts/types/component/DefaultTooltipContent';
import { RefreshCw, Download, Activity } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { KPICard } from '@/components/visualizations';
import { QueryErrorState } from '@/components/ui/query-error-state';
import { usePerformanceTrend } from '@/hooks/api/use-monitoring';
import { useKPIValue, useKPIMetadata, useKPIList } from '@/hooks/api/use-kpi';

// =============================================================================
// CONSTANTS
// =============================================================================

// Per-brand model default: cohort=persistence, brand=Remibrutinib.
// Handle convention: `{cohort}_{brand_lower}_goldstd_lr_v1`
const DEFAULT_COHORT = 'persistence';
const DEFAULT_BRAND = 'Remibrutinib';
const DEFAULT_MODEL_ID = `${DEFAULT_COHORT}_${DEFAULT_BRAND.toLowerCase()}_goldstd_lr_v1`;
const EXAMPLE_MODEL_ID = 'csu_treatment_initiation_lr_balanced_v1';

const COHORT_OPTIONS: { value: string; label: string }[] = [
  { value: 'initiation', label: 'Initiation' },
  { value: 'persistence', label: 'Persistence' },
  { value: 'discontinuation', label: 'Discontinuation' },
];

const BRAND_OPTIONS: { value: string; label: string }[] = [
  { value: 'Remibrutinib', label: 'Remibrutinib' },
  { value: 'Fabhalta', label: 'Fabhalta' },
  { value: 'Kisqali', label: 'Kisqali' },
];
const DEFAULT_METRIC = 'accuracy';
const DEFAULT_KPI_ID = 'WS1-DQ-001';

const TIME_RANGES: { value: string; label: string; days: number }[] = [
  { value: '30d', label: '30 Days', days: 30 },
  { value: '60d', label: '60 Days', days: 60 },
  { value: '90d', label: '90 Days', days: 90 },
  { value: '180d', label: '6 Months', days: 180 },
  { value: '365d', label: '1 Year', days: 365 },
  { value: '1825d', label: '5 Years', days: 1825 },
];

const METRIC_OPTIONS: { value: string; label: string }[] = [
  { value: 'accuracy', label: 'Accuracy' },
  { value: 'precision', label: 'Precision' },
  { value: 'recall', label: 'Recall' },
  { value: 'f1', label: 'F1 Score' },
  { value: 'auc_roc', label: 'AUC-ROC' },
];

// =============================================================================
// HELPERS
// =============================================================================

function formatDate(dateStr: string | undefined): string {
  if (!dateStr) return '';
  const date = new Date(dateStr);
  if (Number.isNaN(date.getTime())) return dateStr;
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function rangeToDays(range: string): number {
  return TIME_RANGES.find((r) => r.value === range)?.days ?? 90;
}

/**
 * Recharts Tooltip value formatter matching `Formatter<ValueType, NameType>`.
 *
 * `ValueType` is `number | string | ReadonlyArray<number | string>`, so we
 * narrow before calling `.toFixed`. Arrays + undefined fall through to the
 * default string coercion (recharts handles `undefined`/`null` cells itself).
 */
const formatTooltipValue: TooltipValueFormatter<ValueType, NameType> = (value) => {
  if (typeof value === 'number') return value.toFixed(4);
  if (typeof value === 'string') return value;
  return '';
};

interface ChartPoint {
  date: string;
  value: number;
}

/**
 * Extract a `[{date, value}]` series from a `KPIResult.metadata.history` shape.
 *
 * The backend embeds history as a `metadata` blob; tolerate a few common
 * shapes (`{recorded_at, value}` or `{date, value}`) without throwing.
 */
function kpiHistoryToSeries(metadata: Record<string, unknown> | undefined): ChartPoint[] {
  if (!metadata) return [];
  const raw = (metadata as { history?: unknown }).history;
  if (!Array.isArray(raw)) return [];
  return raw
    .map((entry): ChartPoint | null => {
      if (typeof entry !== 'object' || entry === null) return null;
      const e = entry as Record<string, unknown>;
      const date =
        typeof e.recorded_at === 'string'
          ? e.recorded_at
          : typeof e.date === 'string'
            ? e.date
            : null;
      const valueRaw = e.value ?? e.metric_value;
      const value = typeof valueRaw === 'number' ? valueRaw : Number(valueRaw);
      if (date === null || Number.isNaN(value)) return null;
      return { date, value };
    })
    .filter((p): p is ChartPoint => p !== null);
}

// =============================================================================
// COMPONENT
// =============================================================================

function TimeSeries() {
  const [mode, setMode] = useState<'performance' | 'kpi'>('performance');

  // Performance mode state — cohort/brand dropdowns drive modelId; free-text overrides
  const [cohort, setCohort] = useState<string>(DEFAULT_COHORT);
  const [brand, setBrand] = useState<string>(DEFAULT_BRAND);
  const [modelId, setModelId] = useState<string>(DEFAULT_MODEL_ID);
  const [metricName, setMetricName] = useState<string>(DEFAULT_METRIC);
  const [timeRange, setTimeRange] = useState<string>('1825d');

  // Derive model handle from cohort/brand and sync into modelId whenever they change.
  const handleCohortChange = (newCohort: string) => {
    setCohort(newCohort);
    setModelId(`${newCohort}_${brand.toLowerCase()}_goldstd_lr_v1`);
  };

  const handleBrandChange = (newBrand: string) => {
    setBrand(newBrand);
    setModelId(`${cohort}_${newBrand.toLowerCase()}_goldstd_lr_v1`);
  };

  // KPI mode state
  const [kpiId, setKpiId] = useState<string>(DEFAULT_KPI_ID);

  const days = rangeToDays(timeRange);

  // ---- Performance mode hook ----
  const performanceTrend = usePerformanceTrend({
    model_id: modelId,
    metric_name: metricName,
    days,
  });

  // ---- KPI mode hooks ----
  const kpiList = useKPIList();
  const kpiMetadata = useKPIMetadata(kpiId);
  const kpiValue = useKPIValue(kpiId);

  // ---- Chart series ----
  const performanceSeries: ChartPoint[] = useMemo(() => {
    const history = performanceTrend.data?.history ?? [];
    return history.map((h) => ({ date: h.recorded_at, value: h.metric_value }));
  }, [performanceTrend.data]);

  const kpiSeries: ChartPoint[] = useMemo(() => {
    const full = kpiHistoryToSeries(kpiValue.data?.metadata);
    if (full.length === 0) return full;
    // Apply the same time-range filter to KPI history (AC #2 — both modes).
    const cutoffMs = Date.now() - days * 24 * 60 * 60 * 1000;
    return full.filter((p) => {
      const t = Date.parse(p.date);
      return Number.isNaN(t) ? true : t >= cutoffMs;
    });
  }, [kpiValue.data, days]);

  const currentSeries = mode === 'performance' ? performanceSeries : kpiSeries;
  const currentSeriesLabel =
    mode === 'performance'
      ? METRIC_OPTIONS.find((m) => m.value === metricName)?.label ?? metricName
      : kpiMetadata.data?.name ?? kpiId;

  // ---- Summary metrics ----
  const summary = useMemo(() => {
    const values = currentSeries.map((p) => p.value);
    if (values.length === 0) {
      return { current: 0, average: 0, max: 0, min: 0, count: 0 };
    }
    const current = values[values.length - 1];
    const sum = values.reduce((a, b) => a + b, 0);
    return {
      current,
      average: sum / values.length,
      max: Math.max(...values),
      min: Math.min(...values),
      count: values.length,
    };
  }, [currentSeries]);

  // Sparkline series for the "Current Value" KPI card — real hook data, not
  // KPICard's `SAMPLE_SPARKLINE` fallback.
  const sparklineSeries = useMemo(
    () => currentSeries.map((p) => p.value),
    [currentSeries],
  );

  // ---- Loading / error per mode ----
  const isLoading =
    mode === 'performance' ? performanceTrend.isLoading : kpiValue.isLoading;
  const error = mode === 'performance' ? performanceTrend.error : kpiValue.error;
  const refetch =
    mode === 'performance' ? performanceTrend.refetch : kpiValue.refetch;
  const isRefetching =
    mode === 'performance'
      ? performanceTrend.isRefetching
      : kpiValue.isRefetching;

  const handleRefresh = () => {
    refetch();
  };

  const handleExport = () => {
    const exportData = {
      mode,
      modelId: mode === 'performance' ? modelId : undefined,
      metric: mode === 'performance' ? metricName : undefined,
      kpiId: mode === 'kpi' ? kpiId : undefined,
      days,
      series: currentSeries,
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `time-series-${mode}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ---- Available KPI options ----
  const availableKpis = useMemo(() => {
    const list = kpiList.data?.kpis ?? [];
    if (list.length === 0) {
      return [{ id: DEFAULT_KPI_ID, name: DEFAULT_KPI_ID }];
    }
    return list.map((k) => ({ id: k.id, name: k.name }));
  }, [kpiList.data]);

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Synthetic demo-data disclosure: the KPI-history series is computed over
          synthetic-gold rows in E2I_KPI_INCLUDE_SYNTHETIC mode — label it so a
          reviewer never reads the trend as real-world data. */}
      {kpiValue.data?.data_source === 'synthetic' && (
        <div
          role="status"
          className="mb-6 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-200"
        >
          Showing <strong>synthetic demo data</strong> — this KPI series is computed on a
          synthetic dataset for review, not real-world data.
        </div>
      )}
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">Time Series Analysis</h1>
          <p className="text-muted-foreground">
            Time series trends, forecasting, seasonality decomposition, and anomaly detection.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {mode === 'performance' ? (
            <Select value={metricName} onValueChange={setMetricName}>
              <SelectTrigger className="w-[160px]" aria-label="metric">
                <SelectValue placeholder="Select metric" />
              </SelectTrigger>
              <SelectContent>
                {METRIC_OPTIONS.map((m) => (
                  <SelectItem key={m.value} value={m.value}>
                    {m.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Select value={kpiId} onValueChange={setKpiId}>
              <SelectTrigger className="w-[220px]" aria-label="kpi">
                <SelectValue placeholder="Select KPI" />
              </SelectTrigger>
              <SelectContent>
                {availableKpis.map((k) => (
                  <SelectItem key={k.id} value={k.id}>
                    {k.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          {/* Time-range filter applies to BOTH modes (AC #2). */}
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-[120px]" aria-label="time range">
              <SelectValue placeholder="Time range" />
            </SelectTrigger>
            <SelectContent>
              {TIME_RANGES.map((r) => (
                <SelectItem key={r.value} value={r.value}>
                  {r.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            variant="outline"
            size="icon"
            onClick={handleRefresh}
            disabled={isRefetching}
            aria-label="Refresh"
          >
            <RefreshCw className={cn('h-4 w-4', isRefetching && 'animate-spin')} />
          </Button>
          <Button variant="outline" size="icon" onClick={handleExport} aria-label="Export">
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Mode toggle */}
      <Tabs
        value={mode}
        onValueChange={(v) => setMode(v as 'performance' | 'kpi')}
        className="space-y-6"
      >
        <TabsList>
          <TabsTrigger value="performance" className="gap-2">
            <Activity className="h-4 w-4" />
            Model performance
          </TabsTrigger>
          <TabsTrigger value="kpi" className="gap-2">
            <Activity className="h-4 w-4" />
            KPI history
          </TabsTrigger>
        </TabsList>

        {/* Error state — both modes */}
        {error && (
          <QueryErrorState
            error={error}
            onRetry={refetch}
            isRetrying={isRefetching}
            title={
              mode === 'performance'
                ? 'Failed to load performance trend'
                : 'Failed to load KPI history'
            }
            size="sm"
          />
        )}

        {/* Loading state — both modes */}
        {isLoading && (
          <div
            data-testid="timeseries-loading"
            className="flex items-center gap-2 text-sm text-muted-foreground"
          >
            <RefreshCw className="h-4 w-4 animate-spin" />
            <span>Loading time series data...</span>
          </div>
        )}

        {/* KPI Summary cards.
            Pass real `sparklineSeries` everywhere — never the KPICard
            default fallback (SAMPLE_SPARKLINE), which would render
            fake trend visuals from a fixture array. Empty `[]` opts
            out of sparkline rendering for non-trend metrics. */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <KPICard
            title="Current Value"
            value={summary.current.toLocaleString()}
            description="Latest value"
            sparklineData={sparklineSeries}
          />
          <KPICard
            title="Average"
            value={Math.round(summary.average * 1000) / 1000 + ''}
            description="Over period"
            sparklineData={[]}
          />
          <KPICard
            title="Maximum"
            value={summary.max.toLocaleString()}
            description="Peak value"
            status="healthy"
            sparklineData={[]}
          />
          <KPICard
            title="Minimum"
            value={summary.min.toLocaleString()}
            description="Lowest value"
            sparklineData={[]}
          />
          <KPICard
            title="Data Points"
            value={summary.count.toLocaleString()}
            description="Observations"
            sparklineData={[]}
          />
        </div>

        {/* Performance mode tab */}
        <TabsContent value="performance" className="space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Performance Trend</CardTitle>
                  <CardDescription>
                    {currentSeriesLabel} over the last {days} days for model{' '}
                    <Badge variant="outline">{modelId}</Badge>
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={performanceSeries} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="date" tickFormatter={formatDate} fontSize={12} tickLine={false} />
                  <YAxis fontSize={12} tickLine={false} axisLine={false} />
                  <Tooltip
                    formatter={formatTooltipValue}
                    labelFormatter={formatDate}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="hsl(var(--chart-1))"
                    strokeWidth={2}
                    dot={false}
                    name={currentSeriesLabel}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {performanceTrend.data && (
            <Card>
              <CardHeader>
                <CardTitle>Trend Summary</CardTitle>
                <CardDescription>Statistical snapshot from the backend</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Current</p>
                    <p className="font-medium">{performanceTrend.data.current_value.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Baseline</p>
                    <p className="font-medium">{performanceTrend.data.baseline_value.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Change</p>
                    <p className="font-medium">
                      {performanceTrend.data.change_percent > 0 ? '+' : ''}
                      {performanceTrend.data.change_percent.toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Trend</p>
                    <p className="font-medium capitalize">{performanceTrend.data.trend}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* KPI history mode tab */}
        <TabsContent value="kpi" className="space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>KPI History</CardTitle>
                  <CardDescription>
                    {currentSeriesLabel} ({kpiId}) historical values — last {days} days
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={kpiSeries} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="date" tickFormatter={formatDate} fontSize={12} tickLine={false} />
                  <YAxis fontSize={12} tickLine={false} axisLine={false} />
                  <Tooltip
                    formatter={formatTooltipValue}
                    labelFormatter={formatDate}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="hsl(var(--chart-2))"
                    strokeWidth={2}
                    dot={false}
                    name={currentSeriesLabel}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {kpiValue.data && (
            <Card>
              <CardHeader>
                <CardTitle>Current KPI Status</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Value</p>
                    <p className="font-medium">
                      {kpiValue.data.value !== undefined
                        ? kpiValue.data.value.toFixed(4)
                        : '—'}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Status</p>
                    <p className="font-medium capitalize">{kpiValue.data.status}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Cached</p>
                    <p className="font-medium">{kpiValue.data.cached ? 'Yes' : 'No'}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Calculated</p>
                    <p className="font-medium">{formatDate(kpiValue.data.calculated_at)}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      {/* Model selector (performance mode only) — placed below for visibility */}
      {mode === 'performance' && (
        <Card className="mt-6">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Model Selection
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Cohort + Brand dropdowns — resolve to the per-brand model handle */}
            <div className="flex flex-wrap gap-4">
              <div className="flex flex-col gap-1">
                <label htmlFor="ts-cohort" className="text-sm font-medium">
                  Cohort
                </label>
                <select
                  id="ts-cohort"
                  value={cohort}
                  onChange={(e) => handleCohortChange(e.target.value)}
                  className="p-2 border rounded-md text-sm bg-background"
                >
                  {COHORT_OPTIONS.map((c) => (
                    <option key={c.value} value={c.value}>
                      {c.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex flex-col gap-1">
                <label htmlFor="ts-brand" className="text-sm font-medium">
                  Brand
                </label>
                <select
                  id="ts-brand"
                  value={brand}
                  onChange={(e) => handleBrandChange(e.target.value)}
                  className="p-2 border rounded-md text-sm bg-background"
                >
                  {BRAND_OPTIONS.map((b) => (
                    <option key={b.value} value={b.value}>
                      {b.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            {/* Free-text model ID — advanced override; typing here wins directly */}
            <div>
              <label htmlFor="ts-model-id" className="text-sm font-medium mb-2 block">
                Model ID (advanced override)
              </label>
              <input
                id="ts-model-id"
                type="text"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                className="w-full p-2 border rounded-md text-sm"
                placeholder={EXAMPLE_MODEL_ID}
              />
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default TimeSeries;
