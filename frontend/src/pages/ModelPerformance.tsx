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
 *   with metric + time-range selectors and an optional all-brands overlay for
 *   gold-standard models (ported from the Time Series page)
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
import { Checkbox } from '@/components/ui/checkbox';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { QueryErrorState } from '@/components/ui/query-error-state';
import {
  Target,
  Activity,
  RefreshCw,
  Download,
  Clock,
  AlertTriangle,
  Trophy,
} from 'lucide-react';
import {
  MetricTrend,
  type MetricDataPoint,
} from '@/components/visualizations';
import { KPICard } from '@/components/visualizations/dashboard';
import { StrategicInsightCard } from '@/components/insights';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  ReferenceLine,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
} from 'recharts';
import {
  usePerformanceTrend,
  usePerformanceAlerts,
  useModelComparison,
  useConfusionMatrix,
  useRocCurve,
} from '@/hooks/api/use-monitoring';
import { useModelsStatus } from '@/hooks/api/use-predictions';
import { useKPIList } from '@/hooks/api/use-kpi';
import { useModelPerformanceInsight } from '@/hooks/api';
import type { ModelEndpointHealth } from '@/types/predictions';
import { Workstream, type KPIThreshold } from '@/types/kpi';
import type {
  ConfusionMatrixResponse,
  PerformanceAlertItem,
  PerformanceMetricItem,
  RocCurveResponse,
} from '@/types/monitoring';
import {
  describeModel,
  interpretConfusion,
  interpretRoc,
} from '@/lib/model-performance/interpret';
import { mergeBrandSeries } from '@/lib/timeseries-brands';

// =============================================================================
// CONSTANTS (trend controls — ported from the Time Series page)
// =============================================================================

const DEFAULT_TREND_METRIC = 'accuracy';

const METRIC_OPTIONS: { value: string; label: string }[] = [
  { value: 'accuracy', label: 'Accuracy' },
  { value: 'precision', label: 'Precision' },
  { value: 'recall', label: 'Recall' },
  { value: 'f1', label: 'F1 Score' },
  { value: 'auc_roc', label: 'AUC-ROC' },
];

// Trend metrics that ARE canonical WS1 model-performance KPIs. The Home KPI
// grid statuses these same metrics (brand-aggregated over the gold-standard
// holdouts) against kpi_definitions.yaml targets; the Current card here must
// speak the same threshold language or the two surfaces contradict each other
// (green here read as "meets target" when it only meant "not degrading").
// accuracy/precision/recall have no canonical KPI threshold and keep the
// alert/degradation semantics.
const TREND_METRIC_KPI_ID: Record<string, string> = {
  auc_roc: 'WS1-MP-001',
  f1: 'WS1-MP-003',
};

/**
 * Mirror of the backend's higher-is-better Threshold.evaluate
 * (src/kpi/models.py): >= target GOOD; < critical CRITICAL; else WARNING.
 * The threshold values come from the KPI list API (kpi_definitions.yaml is
 * the single source of truth) — only the evaluation is mirrored here.
 * Returns null when the KPI has no usable threshold (caller falls back to
 * the existing alert semantics).
 */
function kpiThresholdStatus(
  value: number,
  threshold: KPIThreshold | null | undefined
): 'healthy' | 'warning' | 'critical' | null {
  if (!threshold || threshold.target == null) return null;
  if (value >= threshold.target) return 'healthy';
  if (threshold.critical != null && value < threshold.critical) return 'critical';
  return 'warning';
}

// Performance metrics are recorded ~monthly (backtest sweep), so a 30-day
// window catches only ~1-2 points and the trend chart renders degenerate /
// empty. Default to a 1-year window to surface the full metric-over-time
// history (the backend's own default is also 365). Cards (current/baseline/
// trend) come from the tracker independently of this window.
const DEFAULT_TREND_RANGE = '365d';

const TIME_RANGES: { value: string; label: string; days: number }[] = [
  { value: '30d', label: '30 Days', days: 30 },
  { value: '60d', label: '60 Days', days: 60 },
  { value: '90d', label: '90 Days', days: 90 },
  { value: '180d', label: '6 Months', days: 180 },
  { value: '365d', label: '1 Year', days: 365 },
  { value: '1825d', label: '5 Years', days: 1825 },
];

// Gold-standard per-brand model handles follow the convention
// `{cohort}_{brand_lower}_goldstd_lr_v1`. When the selected model matches,
// the "Compare all brands" toggle overlays its two sibling-brand models.
const GOLDSTD_MODEL_RE =
  /^(initiation|persistence|discontinuation|hcp_adoption)_(remibrutinib|fabhalta|kisqali)_goldstd_lr_v1$/;

const GOLDSTD_BRANDS = ['Remibrutinib', 'Fabhalta', 'Kisqali'] as const;
type GoldstdBrand = (typeof GOLDSTD_BRANDS)[number];

// Per-brand line colors for the overlay chart (matches TimeSeries).
const BRAND_COLORS: Record<GoldstdBrand, string> = {
  Remibrutinib: 'var(--color-chart-1)',
  Fabhalta: 'var(--color-chart-2)',
  Kisqali: 'var(--color-chart-3)',
};

// =============================================================================
// HELPERS
// =============================================================================

function rangeToDays(range: string): number {
  return TIME_RANGES.find((r) => r.value === range)?.days ?? 365;
}

function formatTrendDate(dateStr: string | undefined): string {
  if (!dateStr) return '';
  const date = new Date(dateStr);
  if (Number.isNaN(date.getTime())) return dateStr;
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

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
 * Confusion matrix (2x2) for the latest holdout evaluation. Counts are EXACT —
 * computed in the gold-standard eval, not derived from rounded scalar metrics.
 */
function ConfusionMatrixView({ data, modelName }: { data: ConfusionMatrixResponse; modelName: string }) {
  const cells = [
    { label: 'True Negative', value: data.tn, good: true },
    { label: 'False Positive', value: data.fp, good: false },
    { label: 'False Negative', value: data.fn, good: false },
    { label: 'True Positive', value: data.tp, good: true },
  ];
  const meaning = describeModel(modelName);
  const interp = interpretConfusion(data, meaning);
  const metrics = [
    { label: 'Precision', m: interp.precision },
    { label: 'Recall', m: interp.recall },
    { label: 'Specificity', m: interp.specificity },
    { label: 'Accuracy', m: interp.accuracy },
    { label: 'F1', m: interp.f1 },
  ];
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2 max-w-md">
        {cells.map((c) => (
          <div
            key={c.label}
            className={`rounded-md border p-4 text-center ${
              c.good ? 'bg-emerald-50 dark:bg-emerald-900/20' : 'bg-rose-50 dark:bg-rose-900/20'
            }`}
          >
            <div className="text-2xl font-bold">{c.value.toLocaleString()}</div>
            <div className="text-xs text-muted-foreground mt-1">{c.label}</div>
          </div>
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        Holdout @ threshold {data.threshold.toFixed(2)}
        {data.sample_size ? ` · n=${data.sample_size.toLocaleString()}` : ''} · rows = actual,
        columns = predicted
      </p>
      <div className="flex flex-wrap gap-4 text-sm">
        {metrics.map(({ label, m }) => (
          <div key={label} className="flex flex-col">
            <span className="text-muted-foreground text-xs">{label}</span>
            <span className="font-medium">{m.pct}</span>
          </div>
        ))}
      </div>
      <div className="rounded-md bg-muted p-3 text-sm">{interp.verdict}</div>
    </div>
  );
}

/** ROC curve (TPR vs FPR) for the latest holdout evaluation, with the chance diagonal. */
function RocCurveView({ data, modelName }: { data: RocCurveResponse; modelName: string }) {
  const meaning = describeModel(modelName);
  const roc = interpretRoc(data.auc, meaning);
  return (
    <div className="space-y-2">
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data.points} margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="fpr"
            domain={[0, 1]}
            tickFormatter={(v: number) => v.toFixed(1)}
            label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -12 }}
          />
          <YAxis
            type="number"
            domain={[0, 1]}
            tickFormatter={(v: number) => v.toFixed(1)}
            label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft' }}
          />
          <RechartsTooltip
            formatter={(value) => (typeof value === 'number' ? value.toFixed(3) : value)}
          />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke="#94a3b8"
            strokeDasharray="4 4"
          />
          <Line
            dataKey="tpr"
            stroke="#2563eb"
            strokeWidth={2}
            dot={false}
            type="monotone"
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="text-xs text-muted-foreground">
        AUC = {data.auc.toFixed(3)}
        {data.sample_size ? ` · n=${data.sample_size.toLocaleString()}` : ''} · holdout
      </p>
      <div className="rounded-md bg-muted p-3 text-sm">{roc.text}</div>
    </div>
  );
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

  // Trend controls (ported from TimeSeries): metric + time-range selectors
  // and the gold-standard all-brands overlay toggle.
  const [trendMetric, setTrendMetric] = useState<string>(DEFAULT_TREND_METRIC);
  const [trendRange, setTrendRange] = useState<string>(DEFAULT_TREND_RANGE);
  const [compareAllBrands, setCompareAllBrands] = useState<boolean>(false);

  // -- Live data ----------------------------------------------------------
  const modelsQuery = useModelsStatus();

  // Canonical WS1 KPI thresholds for the Current-metric card (Home KPI-grid
  // parity). Metadata only — cheap, cached 5 min by react-query. Failure or
  // loading degrades to the previous alert-only semantics (never blocks the
  // page).
  const kpiListQuery = useKPIList({ workstream: Workstream.WS1_MODEL_PERFORMANCE });
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

  // Gold-standard model detection for the all-brands overlay. When the
  // selected model matches `{cohort}_{brand}_goldstd_lr_v1`, the "Compare all
  // brands" toggle overlays the two sibling-brand models of the same cohort.
  const goldstdMatch = useMemo(
    () => GOLDSTD_MODEL_RE.exec(effectiveModelId),
    [effectiveModelId]
  );
  const goldstdCohort = goldstdMatch?.[1] ?? '';
  const goldstdBrand = useMemo<GoldstdBrand | null>(
    () =>
      goldstdMatch
        ? GOLDSTD_BRANDS.find((b) => b.toLowerCase() === goldstdMatch[2]) ?? null
        : null,
    [goldstdMatch]
  );
  const siblingBrands = useMemo<GoldstdBrand[]>(
    () => (goldstdBrand ? GOLDSTD_BRANDS.filter((b) => b !== goldstdBrand) : []),
    [goldstdBrand]
  );
  const overlayActive = !!goldstdBrand && compareAllBrands;
  const siblingHandle = (brand: GoldstdBrand | undefined) =>
    brand && goldstdCohort ? `${goldstdCohort}_${brand.toLowerCase()}_goldstd_lr_v1` : '';

  const trendDays = rangeToDays(trendRange);

  // Rules-of-hooks pattern (PR #1045): all three usePerformanceTrend hooks are
  // called UNCONDITIONALLY in a stable order — never in a loop/condition —
  // varying only `enabled`. The selected model's query is the one enabled by
  // default; the two sibling-brand queries only run while the overlay is on.
  const trendQuery = usePerformanceTrend(
    { model_id: effectiveModelId, metric_name: trendMetric, days: trendDays },
    { enabled: !!effectiveModelId }
  );
  const siblingTrendA = usePerformanceTrend(
    { model_id: siblingHandle(siblingBrands[0]), metric_name: trendMetric, days: trendDays },
    { enabled: overlayActive && !!siblingHandle(siblingBrands[0]) }
  );
  const siblingTrendB = usePerformanceTrend(
    { model_id: siblingHandle(siblingBrands[1]), metric_name: trendMetric, days: trendDays },
    { enabled: overlayActive && !!siblingHandle(siblingBrands[1]) }
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

  // Holdout confusion matrix + ROC curve (eval-persisted; honest empty until populated).
  const confusionQuery = useConfusionMatrix(effectiveModelId, {
    enabled: !!effectiveModelId,
  });
  const rocQuery = useRocCurve(effectiveModelId, { enabled: !!effectiveModelId });

  // Agentic strategic interpretation of the current model's performance
  // (on-demand mutation; grounded in the selected model version).
  const perfInsight = useModelPerformanceInsight();

  const trendHistory = useMemo(
    () => toMetricDataPoints(trendQuery.data?.history),
    [trendQuery.data?.history]
  );

  // Overlay rows ({ date, <brand>: value }) — one line per gold-standard brand.
  // mergeBrandSeries is shared with TimeSeries (imported, not copied).
  const overlayRows = useMemo(() => {
    if (!overlayActive || !goldstdBrand) return [];
    const toSeries = (history: PerformanceMetricItem[] | undefined) =>
      (history ?? []).map((item) => ({ date: item.recorded_at, value: item.metric_value }));
    const perBrand: Record<string, { date: string; value: number }[]> = {
      [goldstdBrand]: toSeries(trendQuery.data?.history),
    };
    if (siblingBrands[0]) perBrand[siblingBrands[0]] = toSeries(siblingTrendA.data?.history);
    if (siblingBrands[1]) perBrand[siblingBrands[1]] = toSeries(siblingTrendB.data?.history);
    return mergeBrandSeries(perBrand, GOLDSTD_BRANDS);
  }, [
    overlayActive,
    goldstdBrand,
    siblingBrands,
    trendQuery.data?.history,
    siblingTrendA.data?.history,
    siblingTrendB.data?.history,
  ]);

  // #970: recorded_at is the DATA boundary (latest holdout journey_start_date),
  // NOT wall-clock now. Surface the latest covered date so the x-axis is read
  // as data coverage, not "today".
  const latestDataDate = useMemo<string | null>(() => {
    const history = trendQuery.data?.history ?? [];
    if (history.length === 0) return null;
    return history.reduce(
      (max, h) => (Date.parse(h.recorded_at) > Date.parse(max) ? h.recorded_at : max),
      history[0].recorded_at
    );
  }, [trendQuery.data?.history]);

  const trendMetricLabel =
    METRIC_OPTIONS.find((m) => m.value === trendMetric)?.label ?? trendMetric;

  // -- Handlers -----------------------------------------------------------
  const handleRefresh = async () => {
    await Promise.all([
      modelsQuery.refetch(),
      trendQuery.refetch?.(),
      alertsQuery.refetch?.(),
      // Comparison may not be enabled — refetch only if effective second model exists
      effectiveCompareModelId ? comparisonQuery.refetch?.() : Promise.resolve(),
      // Sibling-brand trends only participate while the overlay is on.
      overlayActive ? siblingTrendA.refetch?.() : Promise.resolve(),
      overlayActive ? siblingTrendB.refetch?.() : Promise.resolve(),
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
  // Sibling queries are disabled unless the overlay is on, so this only ever
  // gates the chart while the overlay is actually fetching.
  const isOverlayLoading =
    overlayActive && (siblingTrendA.isLoading || siblingTrendB.isLoading);

  const isRefetching =
    modelsQuery.isRefetching ||
    trendQuery.isRefetching ||
    alertsQuery.isRefetching ||
    comparisonQuery.isRefetching ||
    siblingTrendA.isRefetching ||
    siblingTrendB.isRefetching;

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

      {/* Strategic Interpretation — always rendered above the metric KPI cards */}
      <div className="mb-6">
        <StrategicInsightCard
          onGenerate={() => {
            if (effectiveModelId) perfInsight.mutate({ model_version: effectiveModelId });
          }}
          isLoading={perfInsight.isPending}
          error={perfInsight.error?.message ?? null}
          insight={perfInsight.data?.insight}
          keyTakeaways={perfInsight.data?.key_takeaways}
          grounding={perfInsight.data?.grounding}
          isFallback={perfInsight.data?.is_fallback}
          provenance={perfInsight.data?.provenance}
          generatedAt={perfInsight.data?.generated_at}
        />
      </div>

      {/* Trend loading skeleton (KPI block) */}
      {isTrendLoading && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          {[0, 1, 2, 3].map((idx) => (
            <LoadingPulse key={idx} className="h-24 w-full" />
          ))}
        </div>
      )}

      {/* Metrics KPI Cards — wired to live trend */}
      {!isTrendLoading && trendQuery.data && (() => {
        // Status the Current card against the canonical WS1 KPI threshold
        // (same thresholds as the Home Model Performance tiles) when the
        // metric IS one of those KPIs; an alert breach still escalates to
        // critical. Keyed on the RESPONSE's metric_name (not selector state)
        // so the color never races ahead of the value it describes.
        const kpiId = TREND_METRIC_KPI_ID[trendQuery.data.metric_name];
        const kpiThreshold = kpiId
          ? kpiListQuery.data?.kpis?.find((k) => k.id === kpiId)?.threshold
          : null;
        const targetStatus = kpiThresholdStatus(trendQuery.data.current_value, kpiThreshold);
        const currentStatus = trendQuery.data.alert_threshold_breached
          ? 'critical'
          : targetStatus ?? 'healthy';
        return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <KPICard
            className="perf-current-card"
            title={`Current ${trendQuery.data.metric_name}`}
            value={(trendQuery.data.current_value * 100).toFixed(1)}
            unit="%"
            status={currentStatus}
            description={
              kpiThreshold?.target != null
                ? `Statused against the ${kpiId} target (≥ ${(kpiThreshold.target * 100).toFixed(0)}%) — the same holdout KPI thresholds as the Home Model Performance tiles.`
                : undefined
            }
          />
          <KPICard
            className="perf-baseline-card"
            title={`Baseline ${trendQuery.data.metric_name}`}
            value={(trendQuery.data.baseline_value * 100).toFixed(1)}
            unit="%"
            status="neutral"
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
        );
      })()}

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
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <CardTitle>Performance Trend</CardTitle>
                  <CardDescription>
                    {trendMetricLabel} over the last {trendDays} days, with baseline + alert
                    thresholds.
                  </CardDescription>
                  {/* #969 + #970 (ported from TimeSeries): be honest about what
                      this trend is. It is a per-month walk-forward backtest
                      (source='backtest_wf'), an UNCALIBRATED LogisticRegression
                      refit each month — not the served CalibratedClassifierCV
                      champion. AUC-ROC is calibration-invariant (exact), but
                      threshold metrics differ. And recorded_at is the data
                      boundary, not wall-clock. */}
                  <p
                    data-testid="perf-trend-provenance-note"
                    className="mt-1 max-w-prose text-xs text-muted-foreground"
                  >
                    Per-month walk-forward backtest (uncalibrated): AUC-ROC matches the
                    served champion, but threshold metrics (accuracy / precision / recall /
                    F1) are a diagnostic, not the calibrated champion.
                    {latestDataDate
                      ? ` Dates reflect data coverage through ${formatTrendDate(latestDataDate)}, not wall-clock.`
                      : ''}
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-3">
                  <Select value={trendMetric} onValueChange={setTrendMetric}>
                    <SelectTrigger className="w-[150px]" aria-label="metric">
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
                  <Select value={trendRange} onValueChange={setTrendRange}>
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
                  {/* Overlay toggle only offered for gold-standard per-brand models
                      (the sibling handles are derivable from the selected one). */}
                  {goldstdBrand && (
                    <div className="flex items-center gap-2">
                      <Checkbox
                        id="mp-compare-brands"
                        checked={compareAllBrands}
                        onCheckedChange={(checked) => setCompareAllBrands(checked === true)}
                      />
                      <label
                        htmlFor="mp-compare-brands"
                        className="text-sm font-medium cursor-pointer"
                      >
                        Compare all brands
                      </label>
                    </div>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {isTrendLoading || isOverlayLoading ? (
                <LoadingPulse className="h-[350px] w-full" />
              ) : overlayActive ? (
                overlayRows.length === 0 ? (
                  <div className="py-12 text-center text-sm text-muted-foreground">
                    No performance history available for these models yet.
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height={350}>
                    <LineChart
                      data={overlayRows}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                      <XAxis
                        dataKey="date"
                        tickFormatter={formatTrendDate}
                        fontSize={12}
                        tickLine={false}
                      />
                      <YAxis fontSize={12} tickLine={false} axisLine={false} />
                      <RechartsTooltip
                        formatter={(value) =>
                          typeof value === 'number' ? value.toFixed(4) : value
                        }
                        labelFormatter={formatTrendDate}
                      />
                      <Legend />
                      {/* One line per gold-standard brand, colored like TimeSeries. */}
                      {GOLDSTD_BRANDS.map((b) => (
                        <Line
                          key={b}
                          type="monotone"
                          dataKey={b}
                          stroke={BRAND_COLORS[b]}
                          strokeWidth={2}
                          dot={false}
                          connectNulls
                          name={b}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                )
              ) : trendHistory.length > 0 ? (
                <MetricTrend
                  name={trendQuery.data?.metric_name ?? trendMetric}
                  data={trendHistory}
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
                  {/* Comparison values are informational — a green status here
                      previously claimed "healthy" for BOTH sides of the
                      comparison regardless of the numbers. Neutral is honest;
                      the verdict card below carries the conclusion. */}
                  <KPICard
                    className="compare-value-card"
                    title={`${comparisonQuery.data.model_id} ${comparisonQuery.data.metric_name}`}
                    value={(comparisonQuery.data.model_value * 100).toFixed(1)}
                    unit="%"
                    status="neutral"
                  />
                  <KPICard
                    className="compare-value-card"
                    title={`${comparisonQuery.data.other_model_id} ${comparisonQuery.data.metric_name}`}
                    value={(comparisonQuery.data.other_model_value * 100).toFixed(1)}
                    unit="%"
                    status="neutral"
                  />
                  <KPICard
                    className="compare-value-card"
                    title="Difference"
                    value={(comparisonQuery.data.difference * 100).toFixed(2)}
                    unit="%"
                    status="neutral"
                  />
                  {/*
                    "Better model" is a model HANDLE (a long string like
                    `initiation_remibrutinib_goldstd_lr_v1`), not a numeric KPI.
                    Rendering it as a KPICard value (text-2xl, no wrap) overflowed
                    the card, so it gets its own card whose name wraps + truncates
                    with a hover title.
                  */}
                  <div className="rounded-lg border border-l-4 border-l-emerald-500 bg-[var(--color-card)] p-4">
                    <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground mb-1">
                      <Trophy className="h-4 w-4" />
                      Better model
                    </div>
                    <div
                      className="font-semibold break-words"
                      title={comparisonQuery.data.better_model}
                    >
                      {comparisonQuery.data.better_model}
                    </div>
                    {comparisonQuery.data.is_significant === false && (
                      <div className="text-xs text-muted-foreground mt-1">
                        difference not significant
                      </div>
                    )}
                  </div>
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
                Holdout confusion matrix at the 0.5 decision threshold.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {confusionQuery.isLoading ? (
                <LoadingPulse className="h-40 w-full" />
              ) : confusionQuery.data?.available ? (
                <ConfusionMatrixView data={confusionQuery.data} modelName={effectiveModelId} />
              ) : (
                <div className="py-8 text-center text-sm text-muted-foreground">
                  No confusion matrix recorded for this model yet. It is computed and
                  stored by the gold-standard evaluation run.
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="roc">
          <Card>
            <CardHeader>
              <CardTitle>ROC Curve</CardTitle>
              <CardDescription>
                Holdout ROC curve (true-positive vs false-positive rate) with the chance
                diagonal.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {rocQuery.isLoading ? (
                <LoadingPulse className="h-[320px] w-full" />
              ) : rocQuery.data?.available && rocQuery.data.points.length > 0 ? (
                <RocCurveView data={rocQuery.data} modelName={effectiveModelId} />
              ) : (
                <div className="py-8 text-center text-sm text-muted-foreground">
                  No ROC curve recorded for this model yet. It is computed and stored by
                  the gold-standard evaluation run.
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default ModelPerformance;
