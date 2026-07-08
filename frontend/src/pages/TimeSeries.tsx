/**
 * Time Series — KPI History
 * =========================
 *
 * Single-mode page charting KPI metric history over time, wired to the live
 * KPI APIs:
 *
 *  - `useKPIList()`     — populates the KPI select (sorted alphabetically)
 *  - `useKPIHistory()`  — real monthly series from the backend (`kpi_history`)
 *  - `useKPIValue()`    — current point-in-time value (status card)
 *  - `useKPIMetadata()` — display name for the selected KPI
 *
 * The page's former "Model performance" mode (walk-forward backtest trends per
 * cohort × brand) moved to the Model Performance page — see the sibling PR.
 * Issue #302 retired the page's 38 `sample*` constants; all series are
 * sourced from real hooks. Loading / error are surfaced via QueryErrorState.
 *
 * @module pages/TimeSeries
 */

import { useState, useMemo, useEffect } from 'react';
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
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { cn } from '@/lib/utils';
import { KPICard } from '@/components/visualizations';
import { QueryErrorState } from '@/components/ui/query-error-state';
import {
  useKPIValue,
  useKPIHistory,
  useKPIHistoryCoverage,
  useKPIMetadata,
  useKPIList,
} from '@/hooks/api/use-kpi';

// =============================================================================
// CONSTANTS
// =============================================================================

// Default to WS3-BI-010 (Return on Investment): the deepest real series in
// `kpi_history`, so the page lands on a populated chart instead of an
// empty-state.
const DEFAULT_KPI_ID = 'WS3-BI-010';

// Families deliberately NOT offered here:
// - ws1_model_performance: model-metric trends are served from the walk-forward
//   ml_performance_metrics table on /model-performance (with per-brand compare);
//   duplicating them into kpi_history would blur provenance.
// - causal_metrics: CM-* are per-analysis estimates carrying CIs/p-values — a
//   platform "monthly history" is not defined for them.
const HIDDEN_WORKSTREAMS = new Set(['ws1_model_performance', 'causal_metrics']);

// Dropdown group order + display labels (only workstreams offered here).
const WORKSTREAM_GROUPS: { key: string; label: string }[] = [
  { key: 'ws1_data_quality', label: 'Data Quality (WS1)' },
  { key: 'ws2_triggers', label: 'Trigger Performance (WS2)' },
  { key: 'ws3_business', label: 'Business Impact (WS3)' },
  { key: 'brand_specific', label: 'Brand-Specific' },
];

// Radix SelectItem forbids an empty-string value, so the global scope rides a
// sentinel in the brand <Select> and is mapped back to '' for the API.
const GLOBAL_BRAND = '__global__';

const TIME_RANGES: { value: string; label: string; days: number }[] = [
  { value: '30d', label: '30 Days', days: 30 },
  { value: '60d', label: '60 Days', days: 60 },
  { value: '90d', label: '90 Days', days: 90 },
  { value: '180d', label: '6 Months', days: 180 },
  { value: '365d', label: '1 Year', days: 365 },
  { value: '1825d', label: '5 Years', days: 1825 },
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

// =============================================================================
// COMPONENT
// =============================================================================

function TimeSeries() {
  const [kpiId, setKpiId] = useState<string>(DEFAULT_KPI_ID);
  // '' = global / all brands (the kpi_history scope convention).
  const [brand, setBrand] = useState<string>('');
  const [timeRange, setTimeRange] = useState<string>('1825d');

  const days = rangeToDays(timeRange);

  // ---- KPI hooks ----
  const kpiList = useKPIList();
  const kpiMetadata = useKPIMetadata(kpiId);
  // Coverage map: which KPIs have a real series, in which brand scopes.
  const kpiCoverage = useKPIHistoryCoverage();
  const coverageMap = useMemo(
    () => new Map((kpiCoverage.data?.coverage ?? []).map((e) => [e.kpi_id, e])),
    [kpiCoverage.data]
  );
  const coverageEntry = coverageMap.get(kpiId);

  // Keep the brand scope valid for the selected KPI: per-brand-only KPIs
  // (e.g. WS3-BI-007 NBRx — a global series is undefined by design) snap to
  // their first brand instead of showing a false empty-state.
  useEffect(() => {
    const entry = coverageMap.get(kpiId);
    if (!entry) {
      setBrand('');
      return;
    }
    if (!entry.brands.includes(brand)) {
      setBrand(entry.brands.includes('') ? '' : (entry.brands[0] ?? ''));
    }
  }, [kpiId, coverageMap, brand]);

  const kpiValue = useKPIValue(kpiId, brand || undefined); // point-in-time value (status card)
  // Real monthly history from the backend (kpi_history). Empty for point-in-time
  // KPIs — the chart then shows an honest empty-state, never a fabricated series.
  const kpiHistory = useKPIHistory(kpiId, brand);

  // ---- Chart series ----
  const kpiSeries: ChartPoint[] = useMemo(() => {
    const full = (kpiHistory.data?.points ?? []).map((p) => ({
      date: p.metric_date,
      value: p.value,
    }));
    if (full.length === 0) return full;
    // Client-side time-range cutoff over the monthly history.
    const cutoffMs = Date.now() - days * 24 * 60 * 60 * 1000;
    return full.filter((p) => {
      const t = Date.parse(p.date);
      return Number.isNaN(t) ? true : t >= cutoffMs;
    });
  }, [kpiHistory.data, days]);

  const seriesLabel = kpiMetadata.data?.name ?? kpiId;

  // ---- Summary metrics ----
  const summary = useMemo(() => {
    const values = kpiSeries.map((p) => p.value);
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
  }, [kpiSeries]);

  // Sparkline series for the summary KPI cards — real hook data, not
  // KPICard's `SAMPLE_SPARKLINE` fallback.
  const sparklineSeries = useMemo(() => kpiSeries.map((p) => p.value), [kpiSeries]);

  // ---- Loading / error ----
  // The chart is driven by kpiHistory (the time series); kpiValue backs only
  // the current-status card.
  const isLoading = kpiHistory.isLoading;
  const error = kpiHistory.error;
  const refetch = kpiHistory.refetch;
  const isRefetching = kpiHistory.isRefetching;

  const handleRefresh = () => {
    refetch();
  };

  const handleExport = () => {
    const exportData = {
      kpiId,
      brand,
      days,
      series: kpiSeries,
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `time-series-kpi-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ---- Available KPI options ----
  // Grouped by workstream with the KPI ID visible (a name-only alphabetical
  // list made whole families unidentifiable — "ROC-AUC" gives no hint it's
  // WS1-MP). Model-performance and causal-metrics families are excluded here
  // (see HIDDEN_WORKSTREAMS); entries without a series yet are labeled.
  const kpiGroups = useMemo(() => {
    const list = (kpiList.data?.kpis ?? []).filter(
      (k) => !HIDDEN_WORKSTREAMS.has(String(k.workstream))
    );
    if (list.length === 0) {
      return [
        {
          key: 'fallback',
          label: 'KPIs',
          options: [{ id: DEFAULT_KPI_ID, name: DEFAULT_KPI_ID, hasHistory: true }],
        },
      ];
    }
    return WORKSTREAM_GROUPS.map((group) => ({
      ...group,
      options: list
        .filter((k) => String(k.workstream) === group.key)
        .map((k) => ({ id: k.id, name: k.name, hasHistory: coverageMap.has(k.id) }))
        .sort((a, b) => a.id.localeCompare(b.id)),
    })).filter((group) => group.options.length > 0);
  }, [kpiList.data, coverageMap]);

  // Brand scopes available for the selected KPI (from real coverage, not
  // guesses). No coverage entry -> only the global scope is offered.
  const brandScopes = coverageEntry?.brands ?? [''];
  const namedBrands = brandScopes.filter((b) => b !== '');
  const hasGlobalScope = !coverageEntry || brandScopes.includes('');

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
          <p className="text-muted-foreground">KPI metric history over time.</p>
        </div>
        <div className="flex items-center gap-3">
          <Select value={kpiId} onValueChange={setKpiId}>
            <SelectTrigger className="w-[320px]" aria-label="kpi">
              <SelectValue placeholder="Select KPI" />
            </SelectTrigger>
            <SelectContent>
              {kpiGroups.map((group) => (
                <SelectGroup key={group.key}>
                  <SelectLabel>{group.label}</SelectLabel>
                  {group.options.map((k) => (
                    <SelectItem key={k.id} value={k.id}>
                      <span className="font-mono text-xs text-muted-foreground">{k.id}</span>{' '}
                      {k.name}
                      {!k.hasHistory && (
                        <span className="text-xs text-muted-foreground"> · no history yet</span>
                      )}
                    </SelectItem>
                  ))}
                </SelectGroup>
              ))}
            </SelectContent>
          </Select>
          {(namedBrands.length > 0 || !hasGlobalScope) && (
            <Select
              value={brand === '' ? GLOBAL_BRAND : brand}
              onValueChange={(v) => setBrand(v === GLOBAL_BRAND ? '' : v)}
            >
              <SelectTrigger className="w-[150px]" aria-label="brand">
                <SelectValue placeholder="Brand" />
              </SelectTrigger>
              <SelectContent>
                {hasGlobalScope && <SelectItem value={GLOBAL_BRAND}>All Brands</SelectItem>}
                {namedBrands.map((b) => (
                  <SelectItem key={b} value={b}>
                    {b}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
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

      <div className="space-y-6">
        {/* Error state */}
        {error && (
          <QueryErrorState
            error={error}
            onRetry={refetch}
            isRetrying={isRefetching}
            title="Failed to load KPI history"
            size="sm"
          />
        )}

        {/* Loading state */}
        {isLoading && (
          <div
            data-testid="timeseries-loading"
            className="flex items-center gap-2 text-sm text-muted-foreground"
          >
            <RefreshCw className="h-4 w-4 animate-spin" />
            <span>Loading time series data...</span>
          </div>
        )}

        {/* Summary cards. Every card shows the real `sparklineSeries` trend
            sparkline (never the KPICard SAMPLE_SPARKLINE fallback, which would
            render a fixture array). */}
        {/* No-series honesty: with 0 points every stat is undefined — show an
            em-dash, never a fabricated-looking 0 (or a green badge on it). */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <KPICard
            title="Current Value"
            value={summary.count > 0 ? summary.current.toLocaleString() : '—'}
            description="Latest value"
            sparklineData={sparklineSeries}
          />
          <KPICard
            title="Average"
            value={summary.count > 0 ? Math.round(summary.average * 1000) / 1000 + '' : '—'}
            description="Over period"
            sparklineData={sparklineSeries}
          />
          <KPICard
            title="Maximum"
            value={summary.count > 0 ? summary.max.toLocaleString() : '—'}
            description="Peak value"
            status={summary.count > 0 ? 'healthy' : undefined}
            sparklineData={sparklineSeries}
          />
          <KPICard
            title="Minimum"
            value={summary.count > 0 ? summary.min.toLocaleString() : '—'}
            description="Lowest value"
            sparklineData={sparklineSeries}
          />
          <KPICard
            title="Data Points"
            value={summary.count.toLocaleString()}
            description="Observations"
            sparklineData={sparklineSeries}
          />
        </div>

        {/* KPI history chart */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>KPI History</CardTitle>
                <CardDescription>
                  {seriesLabel} ({kpiId}) historical values — last {days} days
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {/* Honest empty-state: monthly history is materialized only for KPIs
                with an honest temporal source. When this KPI has none, DON'T
                render a blank chart that reads as real flat data — say so. */}
            {!isLoading && kpiSeries.length === 0 ? (
              <div
                data-testid="kpi-history-empty"
                className="flex h-[300px] flex-col items-center justify-center gap-2 text-center text-sm text-muted-foreground"
              >
                <Activity className="h-8 w-8 opacity-40" />
                <p className="font-medium">No historical data for this KPI in this scope</p>
                <p className="max-w-md text-xs">
                  {coverageEntry && !brandScopes.includes(brand)
                    ? 'This KPI has no series for the selected brand scope — pick another brand above.'
                    : coverageEntry
                      ? 'The series exists but has no points inside the selected time range — widen the range above.'
                      : 'Point-in-time KPIs accrue real history from weekly live captures — the series grows one honest point per week; no past values are fabricated. See the current status below.'}
                </p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={kpiSeries} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="date" tickFormatter={formatDate} fontSize={12} tickLine={false} />
                  <YAxis fontSize={12} tickLine={false} axisLine={false} />
                  <Tooltip formatter={formatTooltipValue} labelFormatter={formatDate} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="var(--color-chart-2)"
                    strokeWidth={2}
                    // A young captured series (1-2 weekly points) is invisible as a
                    // dot-less line — render dots until the line carries the shape.
                    dot={kpiSeries.length <= 24 ? { r: 3 } : false}
                    name={seriesLabel}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Current point-in-time status */}
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
                    {kpiValue.data.value !== undefined ? kpiValue.data.value.toFixed(4) : '—'}
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
      </div>
    </div>
  );
}

export default TimeSeries;
