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
 *  - `useKPIHistoryMultiBrand()` — per-brand series for the "Compare Brands"
 *    overlay (one line per brand scope, offered when coverage has ≥2 brands)
 *  - `useKPIHistoryNowcast()` — claims-lag provisional/nowcast series for the
 *    Rx-volume family only (backlog #45): recent months whose claims are
 *    still arriving render as a dashed provisional tail (via MetricTrend),
 *    with an opt-in nowcast overlay (default OFF, persisted per user)
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
import { RefreshCw, Download, Activity, GitCompare } from 'lucide-react';
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
import { mergeBrandSeries, type DatedValue } from '@/lib/timeseries-brands';
import { KPICard, MetricTrend, type MetricDataPoint } from '@/components/visualizations';
import { usePageChatContext } from '@/providers/E2ICopilotProvider';
import { QueryErrorState } from '@/components/ui/query-error-state';
import {
  useKPIValue,
  useKPIHistory,
  useKPIHistoryCoverage,
  useKPIHistoryMultiBrand,
  useKPIHistoryNowcast,
  useKPIMetadata,
  useKPIList,
  RX_VOLUME_KPI_IDS,
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

// Same Radix constraint for the region scope ('' = all regions, #1536).
const ALL_REGIONS = '__all__';

// Per-brand line colors for the compare overlay (matches ModelPerformance's
// BRAND_COLORS so the same brand reads the same color across pages). Brands
// outside the fixed map fall back to the remaining chart palette by position.
const BRAND_COLORS: Record<string, string> = {
  Remibrutinib: 'var(--color-chart-1)',
  Fabhalta: 'var(--color-chart-2)',
  Kisqali: 'var(--color-chart-3)',
};
const FALLBACK_BRAND_COLORS = [
  'var(--color-chart-4)',
  'var(--color-chart-5)',
  'var(--color-chart-6)',
];

function brandColor(brand: string, index: number): string {
  return BRAND_COLORS[brand] ?? FALLBACK_BRAND_COLORS[index % FALLBACK_BRAND_COLORS.length];
}

// "Show Nowcast" is a per-user VIEW preference — default OFF (the honest
// mature series stays the default view) — persisted like the Home
// sibling-brand toggle (PR #1303 `e2i-<page>-<what>` key convention).
const SHOW_NOWCAST_STORAGE_KEY = 'e2i-timeseries-show-nowcast';

function loadShowNowcast(): boolean {
  if (typeof window === 'undefined') return false;
  try {
    return localStorage.getItem(SHOW_NOWCAST_STORAGE_KEY) === 'true';
  } catch {
    return false;
  }
}

function saveShowNowcast(show: boolean): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(SHOW_NOWCAST_STORAGE_KEY, String(show));
  } catch {
    // Ignore storage errors (private mode etc.) — the in-session state applies.
  }
}

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

/**
 * Honest, user-facing cause for a nowcast-less response. Mirrors the
 * endpoint's `insufficient_maturity` reason contract — never a fabricated
 * fallback completion factor, never a silent hide.
 */
function nowcastUnavailableText(reason: string | null | undefined): string {
  if (!reason) return 'not estimable for this scope.';
  if (reason === 'arrival_plane_not_populated')
    return 'claims-arrival dates are not populated on this dataset yet.';
  if (reason.startsWith('arrival_plane_partial'))
    return 'claims-arrival dates cover only part of this dataset.';
  if (reason === 'insufficient_mature_months')
    return 'too few mature months to estimate claims completion honestly.';
  if (reason === 'no_arrived_claims')
    return 'no claims have arrived inside the observation window.';
  if (reason === 'no_data') return 'no claims data exists for this scope.';
  return `${reason}.`;
}

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
  // '' = all regions; named regions exist only where the backfill mirrors a
  // vetted live region variant (#1536 — coverage `scopes` is the authority).
  const [region, setRegion] = useState<string>('');
  const [timeRange, setTimeRange] = useState<string>('1825d');
  const [compareBrands, setCompareBrands] = useState<boolean>(false);
  // Claims-lag nowcast overlay preference (Rx-volume KPIs only). Default OFF —
  // the honest mature series is the default view; persisted per user.
  const [showNowcast, setShowNowcast] = useState<boolean>(loadShowNowcast);

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

  // Brand scopes available for the selected KPI (from real coverage, not
  // guesses). No coverage entry -> only the global scope is offered.
  const brandScopes = coverageEntry?.brands ?? [''];
  const namedBrands = useMemo(
    () => (coverageEntry?.brands ?? []).filter((b) => b !== ''),
    [coverageEntry]
  );
  const hasGlobalScope = !coverageEntry || brandScopes.includes('');

  // Region scopes for the selected KPI + brand, from the coverage scope
  // lattice (real coverage, not guesses): a region is offered only when a
  // series for THIS (brand, region) actually exists.
  const namedRegions = useMemo(
    () =>
      (coverageEntry?.scopes ?? [])
        .filter((s) => s.brand === brand && s.region !== '')
        .map((s) => s.region),
    [coverageEntry, brand]
  );

  // Brands comparable in the CURRENT region scope: with a region selected,
  // only brands with a real brand×region series can chart a line.
  const compareScopeBrands = useMemo(
    () =>
      region === ''
        ? namedBrands
        : (coverageEntry?.scopes ?? [])
            .filter((s) => s.region === region && s.brand !== '')
            .map((s) => s.brand),
    [region, namedBrands, coverageEntry]
  );

  // Compare mode needs ≥2 named brand scopes (in the selected region) to
  // mean anything. The toggle state survives KPI switches but only takes
  // effect where eligible.
  const canCompare = compareScopeBrands.length >= 2;
  const comparing = compareBrands && canCompare;

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

  // Keep the region scope valid for the selected KPI + brand: a scope with
  // no real series would chart a false empty-state, so snap back to ''.
  useEffect(() => {
    if (region !== '' && !namedRegions.includes(region)) {
      setRegion('');
    }
  }, [region, namedRegions]);

  // Point-in-time value for the status card — same (brand, region) scope as
  // the charted series (the live calculators honor region via the vetted
  // region variants), so the card never mislabels a portfolio number.
  const kpiValue = useKPIValue(kpiId, brand || undefined, region || undefined);
  // Real monthly history from the backend (kpi_history). Empty for point-in-time
  // KPIs — the chart then shows an honest empty-state, never a fabricated series.
  const kpiHistory = useKPIHistory(kpiId, brand, region);
  // Per-brand series for the compare overlay. Zero queries while off; each
  // query shares the single-brand cache key, so toggling never refetches.
  const compareQueries = useKPIHistoryMultiBrand(
    kpiId,
    comparing ? compareScopeBrands : [],
    region
  );
  // Claims-lag provisional/nowcast series — Rx-volume family ONLY (the
  // endpoint 422s other KPIs; the hook carries the same hard gate). The
  // nowcast has no region dimension: gate it to the all-regions scope.
  const isRxVolume = RX_VOLUME_KPI_IDS.has(kpiId);
  const kpiNowcast = useKPIHistoryNowcast(kpiId, brand, {
    enabled: isRxVolume && region === '',
  });

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

  // Compare-mode chart rows: per-brand series aligned by date (a brand's cell
  // is absent — NOT zero — on months it has no point), same time-range cutoff.
  const compareRows = useMemo(() => {
    if (!comparing) return [];
    const perBrand: Record<string, DatedValue[]> = {};
    compareScopeBrands.forEach((b, i) => {
      perBrand[b] = (compareQueries[i]?.data?.points ?? []).map((p) => ({
        date: p.metric_date,
        value: p.value,
      }));
    });
    const cutoffMs = Date.now() - days * 24 * 60 * 60 * 1000;
    return mergeBrandSeries(perBrand, compareScopeBrands).filter((r) => {
      const t = Date.parse(r.date);
      return Number.isNaN(t) ? true : t >= cutoffMs;
    });
  }, [comparing, compareQueries, compareScopeBrands, days]);

  // Per-brand latest-value cards for compare mode. The single-series stat
  // cards (average/max/min) are undefined across multiple series — replace
  // them instead of averaging brands into a number nobody asked for.
  const compareBrandStats = useMemo(
    () =>
      compareScopeBrands.map((b) => {
        const values = compareRows
          .map((r) => r[b])
          .filter((v): v is number => typeof v === 'number');
        return {
          brand: b,
          latest: values.length > 0 ? values[values.length - 1] : null,
          count: values.length,
          sparkline: values,
        };
      }),
    [compareScopeBrands, compareRows]
  );

  const seriesLabel = kpiMetadata.data?.name ?? kpiId;

  // ---- Claims-lag provisional/nowcast view (backlog #45) ----
  const nowcastData = isRxVolume ? kpiNowcast.data : undefined;
  // Chart rows for the provisional view, sourced from the nowcast endpoint:
  // its mature_value matches /history by contract, and the anchor-cap
  // frontier month is excluded server-side (its arrivals pile up at the
  // anchor date) — so it stays excluded here. Same time-range cutoff.
  const nowcastSeries: MetricDataPoint[] = useMemo(() => {
    if (!nowcastData || nowcastData.insufficient_maturity) return [];
    const cutoffMs = Date.now() - days * 24 * 60 * 60 * 1000;
    return (nowcastData.points ?? [])
      .filter((p) => {
        const t = Date.parse(p.metric_date);
        return Number.isNaN(t) ? true : t >= cutoffMs;
      })
      .map((p) => ({
        timestamp: p.metric_date,
        value: p.mature_value,
        provisional: p.provisional,
        completionFactor: p.completion_factor ?? null,
        nowcastValue: p.nowcast_value ?? null,
        nowcastCiLower: p.nowcast_ci_lower ?? null,
        nowcastCiUpper: p.nowcast_ci_upper ?? null,
      }));
  }, [nowcastData, days]);
  const provisionalCount = useMemo(
    () => nowcastSeries.filter((p) => p.provisional).length,
    [nowcastSeries]
  );
  // The provisional view replaces the plain chart ONLY when it adds honest
  // information (a provisional tail exists in range). Every other state —
  // off-family KPI, compare mode, fetch error, insufficient maturity, or an
  // all-mature window — keeps the existing chart untouched. A selected
  // region also blocks it: the nowcast describes the all-regions series, and
  // react-query retains the last fetch while the query is disabled — without
  // this gate a cached GLOBAL overlay would drape over a REGION series.
  const showProvisionalChart =
    !comparing && region === '' && !kpiNowcast.error && provisionalCount > 0;
  // Rx KPI whose nowcast is honestly unavailable: disabled toggle + reason.
  const nowcastUnavailable =
    isRxVolume && !comparing && nowcastData?.insufficient_maturity === true;
  // Rx KPI with a region selected: the nowcast has no region dimension, so
  // the toggle stays visible but disabled with the reason spelled out —
  // never silently absent, never a mislabeled portfolio overlay.
  const nowcastRegionBlocked = isRxVolume && !comparing && region !== '';

  const toggleShowNowcast = () => {
    setShowNowcast((v) => {
      const next = !v;
      saveShowNowcast(next);
      return next;
    });
  };

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
  // The chart is driven by kpiHistory (single mode) or the per-brand compare
  // queries; kpiValue backs only the current-status card.
  const isLoading = comparing ? compareQueries.some((q) => q.isLoading) : kpiHistory.isLoading;
  const error = comparing
    ? (compareQueries.find((q) => q.error)?.error ?? null)
    : kpiHistory.error;
  const isRefetching = comparing
    ? compareQueries.some((q) => q.isRefetching)
    : kpiHistory.isRefetching;

  const refetch = () => {
    if (comparing) {
      compareQueries.forEach((q) => q.refetch());
    } else {
      kpiHistory.refetch();
    }
  };

  const handleRefresh = () => {
    refetch();
  };

  const handleExport = () => {
    const exportData = comparing
      ? { kpiId, days, brands: compareScopeBrands, region, series: compareRows }
      : {
          kpiId,
          brand,
          region,
          days,
          series: kpiSeries,
          // Provenance for the on-screen provisional view: the claims
          // frontier and the per-month provisional/nowcast fields ride along
          // so an exported chart cannot silently shed its maturity caveats.
          ...(showProvisionalChart
            ? {
                nowcast: {
                  data_through: nowcastData?.data_through ?? null,
                  anchor_cap_month: nowcastData?.anchor_cap_month ?? null,
                  ci_level: nowcastData?.ci_level ?? null,
                  points: nowcastSeries,
                },
              }
            : {}),
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
    const knownKeys = new Set(WORKSTREAM_GROUPS.map((g) => g.key));
    const toOptions = (matches: typeof list) =>
      matches
        .map((k) => ({ id: k.id, name: k.name, hasHistory: coverageMap.has(k.id) }))
        .sort((a, b) => a.id.localeCompare(b.id));
    return [
      ...WORKSTREAM_GROUPS.map((group) => ({
        ...group,
        options: toOptions(list.filter((k) => String(k.workstream) === group.key)),
      })),
      // Registry-drift safety: a workstream this page doesn't know yet surfaces
      // under "Other" instead of silently vanishing from the dropdown.
      {
        key: 'other',
        label: 'Other',
        options: toOptions(list.filter((k) => !knownKeys.has(String(k.workstream)))),
      },
    ].filter((group) => group.options.length > 0);
  }, [kpiList.data, coverageMap]);

  // Publish a compact on-screen data summary so the chat pane can generate
  // opener pills grounded in what this page is showing (usePageChatContext →
  // POST /chat/suggestions page_context).
  const pageChatSummary = useMemo(() => {
    const lines: string[] = [
      `Time Series page. KPI: ${seriesLabel} (${kpiId}); brand scope: ${brand || 'All brands'}; region scope: ${region || 'all regions'}; time range: ${timeRange}.`,
    ];
    if (comparing) {
      const stats = compareBrandStats
        .filter((s) => s.latest != null)
        .map((s) => `${s.brand}: latest ${s.latest} (${s.count} months)`)
        .join('; ');
      if (stats) lines.push(`Comparing brands — ${stats}.`);
    } else if (summary.count > 0) {
      lines.push(
        `Series on screen: ${summary.count} monthly points — current ${summary.current}, average ${summary.average.toFixed(1)}, min ${summary.min}, max ${summary.max}.`
      );
    }
    if (showProvisionalChart) {
      lines.push(
        `Claims lag: the ${provisionalCount} most recent charted month(s) are provisional (claims still maturing); nowcast overlay ${showNowcast ? 'shown' : 'available but off'}.`
      );
    }
    return lines.join('\n');
  }, [
    seriesLabel,
    kpiId,
    brand,
    region,
    timeRange,
    comparing,
    compareBrandStats,
    summary,
    showProvisionalChart,
    provisionalCount,
    showNowcast,
  ]);
  usePageChatContext(pageChatSummary);

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
              {/* Compare mode charts every brand at once — the single-brand
                  pick is inert until compare is toggled off. */}
              <SelectTrigger className="w-[150px]" aria-label="brand" disabled={comparing}>
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
          {/* Region scope — offered only where a real region series exists for
              the selected KPI + brand (coverage scope lattice, #1536). Stays
              ACTIVE in compare mode: it scopes the whole comparison. */}
          {namedRegions.length > 0 && (
            <Select
              value={region === '' ? ALL_REGIONS : region}
              onValueChange={(v) => setRegion(v === ALL_REGIONS ? '' : v)}
            >
              <SelectTrigger className="w-[150px]" aria-label="region">
                <SelectValue placeholder="Region" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={ALL_REGIONS}>All Regions</SelectItem>
                {namedRegions.map((r) => (
                  <SelectItem key={r} value={r}>
                    {r}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          {/* Offered only when the selected KPI has ≥2 named brand scopes in
              real coverage — a single-brand or global-only KPI has nothing to
              compare. */}
          {canCompare && (
            <Button
              variant={comparing ? 'default' : 'outline'}
              onClick={() => setCompareBrands((v) => !v)}
              aria-pressed={comparing}
            >
              <GitCompare className="mr-2 h-4 w-4" />
              Compare Brands
            </Button>
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

        {/* Compare mode: one latest-value card per brand (the single-series
            stats below are undefined across multiple series). */}
        {comparing && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {compareBrandStats.map((s) => (
              <KPICard
                key={s.brand}
                title={s.brand}
                value={s.latest !== null ? s.latest.toLocaleString() : '—'}
                description={`Latest value · ${s.count} points in range`}
                sparklineData={s.sparkline}
              />
            ))}
          </div>
        )}

        {/* Summary cards. Every card shows the real `sparklineSeries` trend
            sparkline (never the KPICard SAMPLE_SPARKLINE fallback, which would
            render a fixture array). */}
        {/* No-series honesty: with 0 points every stat is undefined — show an
            em-dash, never a fabricated-looking 0 (or a green badge on it). */}
        {!comparing && (
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
        )}

        {/* KPI history chart */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>KPI History</CardTitle>
                <CardDescription>
                  {seriesLabel} ({kpiId}){' '}
                  {comparing ? 'per-brand comparison' : 'historical values'}
                  {region ? ` · region: ${region}` : ''} — last {days} days
                </CardDescription>
              </div>
              {/* Claims-lag nowcast toggle — Rx-volume KPIs only. Offered
                  when the provisional view is on screen; while the arrival
                  plane cannot support an honest nowcast it stays DISABLED
                  with the reason spelled out (never silently absent, never
                  a fabricated overlay). */}
              {(showProvisionalChart || nowcastUnavailable || nowcastRegionBlocked) && (
                <div className="flex flex-col items-end gap-1">
                  <Button
                    variant={showNowcast && showProvisionalChart ? 'default' : 'outline'}
                    size="sm"
                    onClick={toggleShowNowcast}
                    aria-pressed={showNowcast && showProvisionalChart}
                    disabled={!showProvisionalChart}
                  >
                    Show Nowcast
                  </Button>
                  {(nowcastUnavailable || nowcastRegionBlocked) && (
                    <p className="max-w-[280px] text-right text-xs text-muted-foreground">
                      Nowcast unavailable —{' '}
                      {nowcastRegionBlocked
                        ? 'the claims-lag nowcast has no region dimension; switch to All Regions to see it.'
                        : nowcastUnavailableText(nowcastData?.reason)}
                    </p>
                  )}
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {/* Honest empty-state: monthly history is materialized only for KPIs
                with an honest temporal source. When this KPI has none, DON'T
                render a blank chart that reads as real flat data — say so. */}
            {!isLoading && (comparing ? compareRows.length === 0 : kpiSeries.length === 0) ? (
              <div
                data-testid="kpi-history-empty"
                className="flex h-[300px] flex-col items-center justify-center gap-2 text-center text-sm text-muted-foreground"
              >
                <Activity className="h-8 w-8 opacity-40" />
                <p className="font-medium">No historical data for this KPI in this scope</p>
                <p className="max-w-md text-xs">
                  {comparing
                    ? 'No brand series has points inside the selected time range — widen the range above.'
                    : coverageEntry && !brandScopes.includes(brand)
                      ? 'This KPI has no series for the selected brand scope — pick another brand above.'
                      : coverageEntry
                        ? 'The series exists but has no points inside the selected time range — widen the range above.'
                        : 'Point-in-time KPIs accrue real history from weekly live captures — the series grows one honest point per week; no past values are fabricated. See the current status below.'}
                </p>
              </div>
            ) : comparing ? (
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={compareRows} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="date" tickFormatter={formatDate} fontSize={12} tickLine={false} />
                  <YAxis fontSize={12} tickLine={false} axisLine={false} />
                  <Tooltip formatter={formatTooltipValue} labelFormatter={formatDate} />
                  <Legend />
                  {/* One line per brand, colored like ModelPerformance's
                      overlay so a brand reads the same across pages. */}
                  {compareScopeBrands.map((b, i) => (
                    <Line
                      key={b}
                      type="monotone"
                      dataKey={b}
                      stroke={brandColor(b, i)}
                      strokeWidth={2}
                      dot={compareRows.length <= 24 ? { r: 3 } : false}
                      connectNulls
                      name={b}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            ) : showProvisionalChart ? (
              <>
                {/* Provisional view (Rx-volume KPIs with a populated claims
                    arrival plane): MetricTrend splits the series into a solid
                    mature segment + dashed provisional tail, with the opt-in
                    nowcast overlay. Values are the same mature series the
                    plain chart shows — only the recency styling is added. */}
                <MetricTrend
                  name={seriesLabel}
                  data={nowcastSeries}
                  height={400}
                  color="var(--color-chart-2)"
                  showHeader={false}
                  showNowcast={showNowcast}
                  timestampFormatter={formatDate}
                  valueFormatter={(v) => v.toLocaleString()}
                />
                {/* Legend/footnote: define "provisional", disclose the claims
                    frontier + the excluded anchor-cap month, and echo the
                    synthetic-substrate disclosure shown at the top. */}
                <div
                  data-testid="nowcast-footnote"
                  className="mt-3 space-y-1 text-xs text-muted-foreground"
                >
                  <p>
                    <span className="font-medium text-foreground">Provisional</span> months
                    (dashed tail, hollow markers): claims for these service months are still
                    arriving, so a real-world feed would still revise them upward. Claims
                    observed through {formatDate(nowcastData?.data_through ?? undefined)}.
                  </p>
                  {nowcastData?.anchor_cap_month && (
                    <p>
                      The frontier month ({formatDate(nowcastData.anchor_cap_month)}) is
                      excluded from this view — its claim arrivals pile up at the anchor
                      date and would read as a false spike.
                    </p>
                  )}
                  {showNowcast && (
                    <p>
                      Nowcast (faint line): the provisional count grossed up by the
                      empirically estimated completion factor; shaded band ={' '}
                      {Math.round((nowcastData?.ci_level ?? 0.95) * 100)}% bootstrap CI.
                    </p>
                  )}
                  {kpiValue.data?.data_source === 'synthetic' && (
                    <p>
                      Series computed on the synthetic demo substrate (see the demo-data
                      notice above).
                    </p>
                  )}
                </div>
              </>
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

        {/* Current point-in-time status — a single-scope value; hidden while
            comparing (which brand would it be?). */}
        {!comparing && kpiValue.data && (
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
