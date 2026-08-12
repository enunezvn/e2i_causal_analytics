/**
 * KPI chart routing
 * =================
 *
 * Decides HOW any registry KPI becomes a chart, and fetches the real figures to
 * back it. The chat model names a KPI; this module picks the endpoint that can
 * actually serve it and the chart shape that suits what comes back.
 *
 * Why routing is needed at all: only a minority of the 44 registry KPIs have a
 * materialized monthly series in `kpi_history`, and only the three Rx-volume
 * KPIs accept a patient axis. The rest are point-in-time — real values, no
 * trend. The original chat action charted a line or nothing, so asking to plot
 * ROC-AUC or Cross-source Match Rate produced an empty frame even though the
 * value was one API call away. Routing turns "no history" into a current-value
 * chart instead of a dead end.
 *
 *   compareBy + axis-capable KPI  -> segmented history   -> multi-series lines
 *   materialized history          -> monthly history     -> line / area / bar
 *   several KPIs named            -> batch calculate     -> comparison bars
 *   otherwise                     -> single calculate    -> KPI card / bar
 *
 * Every branch reads real API responses. Nothing here synthesises a value, and
 * a KPI that genuinely has no data returns an empty result the UI states
 * plainly rather than a chart of zeros.
 *
 * @module lib/kpi-chart-router
 */

import {
  batchCalculateKPIs,
  getKPIHistory,
  getKPIHistorySegmented,
  getKPIValue,
} from '@/api/kpi';
import { KPI_CATALOG } from './kpi-catalog.generated';
import type { KpiCatalogEntry } from './kpi-catalog.generated';
import {
  regionClarifyMessage,
  resolveBrand,
  resolveCompareAxis,
  resolveKpiId,
  resolveRegion,
  resolveSegment,
  resolveTherapyLine,
} from './kpi-alias';
// The Rx-volume family is the ONLY family the segmented endpoint serves (it
// 422s the rest). That set already exists as the gate on useKPIHistoryNowcast
// — reused here rather than restated, so the two cannot drift apart.
import { RX_VOLUME_KPI_IDS } from '@/hooks/api/use-kpi';
// Type-only import: erased at build time, so this module does NOT pull
// flint-chart into the eager bundle. Compilation happens inside FlintChart,
// which loads lib/flint-chart lazily — see the bundling note there.
import type {
  ChartRow,
  ErrorBarSpec,
  LogicalEncoding,
  SupportedChartType,
} from './flint-chart-types';

/**
 * Semantic type for a KPI's month axis. Registry history is monthly, so Flint
 * picks month-grained tick labels. Inlined rather than imported to keep this
 * module free of any flint-chart reference.
 */
const MONTH_SEMANTIC = 'Date';

const CATALOG_BY_ID: ReadonlyMap<string, KpiCatalogEntry> = new Map(
  KPI_CATALOG.map((entry) => [entry.id, entry])
);

/** Look up catalog metadata for a resolved registry code. */
export function lookupKpi(kpiId: string): KpiCatalogEntry | undefined {
  return CATALOG_BY_ID.get(kpiId);
}

/** What the chart layer needs to draw, plus the provenance to caption it. */
export interface KpiChartData {
  /** Real rows, ready for Flint. Empty when the KPI genuinely has no data. */
  rows: ChartRow[];
  semanticTypes: Record<string, string>;
  /**
   * The chart shape in DATA terms. FlintChart maps it onto whichever channels
   * the chosen template declares (they differ: KPI Card takes metric/value,
   * Histogram takes only x). Keeping it logical here is what lets this module
   * stay free of any flint-chart import.
   */
  encoding: LogicalEncoding;
  /** Chart type chosen for this data shape; a caller override wins over it. */
  chartType: SupportedChartType;
  /**
   * Confidence-interval bounds to draw as whiskers. Set only when the KPI
   * actually returned an interval — never synthesised.
   */
  errorBars?: ErrorBarSpec;
  title: string;
  /** Scope line: brand / axis / span. Rendered under the title. */
  subtitle: string;
  /** Set when there is nothing to draw — shown verbatim instead of a chart. */
  emptyReason?: string;
}

export interface KpiChartQuery {
  /** One or more KPI references as the model typed them. */
  kpis: string[];
  brand?: string;
  /**
   * Geographic region scope (#1536/#1538). The materialized-history path
   * fetches the region series directly; the current-value and comparison
   * paths pass the region to the calculate endpoints and obey the response's
   * REGION PROVENANCE — a figure is captioned with the region only when
   * `region_status === 'applied'`. The segmented endpoint has no region
   * dimension, so that branch still refuses explicitly.
   */
  region?: string;
  compareBy?: string;
  segment?: string;
  therapyLine?: string;
  /** Explicit chart type from the model; overrides the routed default. */
  chartType?: SupportedChartType;
  title?: string;
}

/**
 * Default chart type per data shape, used when the model does not choose.
 *
 * These are NOT arbitrary. Flint ships `recommendChartTypes`, which derives a
 * type from a data profile, and it was measured against every shape this
 * router produces before these defaults were kept:
 *
 *   monthly series     -> Line Chart   (88)  agrees
 *   segmented series   -> Line Chart   (88)  agrees
 *   multi-KPI compare  -> Bar Chart    (60)  agrees
 *   single value       -> Bar Chart    (60)  WRONG: never suggests KPI Card, so
 *                                            a lone figure becomes a one-bar chart
 *   value + interval   -> Scatter Plot (84)  WRONG: counts ci_low/ci_high as
 *                                            "two or more measures (relationship)"
 *                                            and proposes plotting the bounds
 *                                            against each other
 *
 * The recommender reads a data PROFILE; it cannot see that CI bounds are not
 * independent measures, or that n=1 wants an indicator rather than one bar.
 * Those are domain facts, so the defaults encode them. Re-measure against a
 * newer Flint before changing this — the disagreements are the reason, not
 * inertia.
 */
const DEFAULT_TIME_SERIES_CHART: SupportedChartType = 'Line Chart';
const DEFAULT_COMPARISON_CHART: SupportedChartType = 'Bar Chart';

function displayName(kpiId: string): string {
  return lookupKpi(kpiId)?.name ?? kpiId;
}

function valueSemantic(kpiId: string): string {
  // The catalog's semanticType union is already Flint's vocabulary;
  // validateChartRequest re-checks it against the runtime list at compile time.
  return lookupKpi(kpiId)?.semanticType ?? 'Number';
}

function scopeLabel(brand: string | undefined, region?: string): string {
  const brandPart = brand && brand.length > 0 ? brand : 'All brands';
  return region && region.length > 0 ? `${brandPart} · ${region}` : brandPart;
}

/**
 * Fetch and shape whatever the named KPI(s) can actually provide.
 *
 * Never throws for "no data" — that is an `emptyReason`. Transport failures do
 * propagate, so the caller can distinguish a broken request from an empty one.
 */
export async function routeKpiChart(query: KpiChartQuery): Promise<KpiChartData> {
  const resolvedIds = query.kpis.map(resolveKpiId).filter((id) => id.length > 0);
  const brand = resolveBrand(query.brand);
  const region = resolveRegion(query.region);

  // Unmappable region: refuse BEFORE any fetch (#1538). region is an enum in
  // the substrate — a junk value can never match a row, so passing it through
  // would produce a 0-value figure under a junk caption. Same fail-fast the
  // backend chat tool applies — and since #1565 the refusal is a CLARIFY
  // question naming the census regions (the backend hint's user-facing
  // mirror), so ambiguity ("East Coast") produces a question, not a dead end.
  if (region === null) {
    return emptyResult(regionClarifyMessage(String(query.region)), query.title ?? 'Chart');
  }

  if (resolvedIds.length === 0) {
    return emptyResult('No KPI was named.', query.title ?? 'Chart');
  }

  // --- Several KPIs named: compare their current values side by side. -------
  if (resolvedIds.length > 1) {
    return await routeComparison(resolvedIds, brand, region, query);
  }

  const kpiId = resolvedIds[0];
  const compareAxis = resolveCompareAxis(query.compareBy);
  const segmentValue = resolveSegment(query.segment);
  const lineValue = resolveTherapyLine(query.therapyLine);
  const axis =
    compareAxis ?? (segmentValue ? 'segment' : lineValue ? 'therapy_line' : undefined);

  // --- Patient-axis split (TRx/NRx/NBRx only). ------------------------------
  if (axis) {
    if (region) {
      return emptyResult(
        `${displayName(kpiId)} by severity tier / line of therapy is global-only — the ` +
          `segmented series has no region dimension. Drop the region scope, or chart the ` +
          `plain ${region} trend.`,
        query.title ?? `${displayName(kpiId)} trend`
      );
    }
    if (!RX_VOLUME_KPI_IDS.has(kpiId)) {
      return emptyResult(
        `${displayName(kpiId)} is not tracked by severity tier or line of therapy — ` +
          'only TRx, NRx and NBRx carry a patient axis.',
        query.title ?? `${displayName(kpiId)} trend`
      );
    }
    return await routeSegmented(kpiId, axis, brand, compareAxis, segmentValue, lineValue, query);
  }

  // --- Materialized monthly history, when this KPI has one. ----------------
  const history = await getKPIHistory(kpiId, brand, region);
  const points = history.points ?? [];
  if (points.length > 0) {
    const rows: ChartRow[] = points.map((point) => ({
      month: point.metric_date,
      value: point.value,
    }));
    const chartType = query.chartType ?? DEFAULT_TIME_SERIES_CHART;
    return {
      rows,
      semanticTypes: { month: MONTH_SEMANTIC, value: valueSemantic(kpiId) },
      encoding: { axis: 'month', value: 'value' },
      chartType,
      title: query.title ?? `${displayName(kpiId)} trend`,
      subtitle: `${scopeLabel(history.brand || brand, history.region || region)} · ${
        points.length
      } month${points.length === 1 ? '' : 's'}`,
    };
  }

  // --- No series: chart the current value instead of drawing nothing. -------
  // Under a region scope this is now safe (#1538): the calculate endpoint
  // reports REGION PROVENANCE, and routeCurrentValue obeys it — a figure is
  // captioned with the region only when the backend attests it was applied.
  return await routeCurrentValue(kpiId, brand, region, query);
}

async function routeSegmented(
  kpiId: string,
  axis: 'segment' | 'therapy_line',
  brand: string | undefined,
  compareAxis: 'segment' | 'therapy_line' | undefined,
  segmentValue: string | undefined,
  lineValue: string | undefined,
  query: KpiChartQuery
): Promise<KpiChartData> {
  const value = compareAxis ? undefined : axis === 'segment' ? segmentValue : lineValue;
  const response = await getKPIHistorySegmented(kpiId, axis, brand, value);
  const series = response.series ?? [];
  const title = query.title ?? `${displayName(kpiId)} trend`;

  if (series.length === 0 || (series[0]?.points ?? []).length === 0) {
    return emptyResult(
      `No ${axis === 'segment' ? 'severity-tier' : 'line-of-therapy'} series available for ` +
        `${displayName(kpiId)}${brand ? ` (${brand})` : ''}.`,
      title
    );
  }

  // Long format — one row per (month, bucket) — so Flint can colour by bucket.
  // Wide format would need a trace per bucket declared up front.
  const rows: ChartRow[] = [];
  for (const bucket of series) {
    for (const point of bucket.points ?? []) {
      rows.push({ month: point.metric_date, bucket: bucket.label, value: point.value });
    }
  }

  const months = series[0]?.points?.length ?? 0;
  const axisLabel = response.axis === 'segment' ? 'by severity tier' : 'by line of therapy';
  const chartType = query.chartType ?? DEFAULT_TIME_SERIES_CHART;
  return {
    rows,
    semanticTypes: {
      month: MONTH_SEMANTIC,
      // Severity tiers and lines of therapy are ORDINAL, not categorical.
      // Measured: 'Category' -> tableau10 (#636efa/#EF553B/#00cc96, unordered);
      // 'Rank' -> a single-hue sequential ramp (#440154/#46327e/#365c8d), which
      // is how a low->high scale should read.
      bucket: 'Rank',
      value: valueSemantic(kpiId),
    },
    encoding: { axis: 'month', value: 'value', series: 'bucket' },
    chartType,
    title,
    subtitle:
      `${scopeLabel(response.brand || brand)} · ${axisLabel} · ${months} month` +
      `${months === 1 ? '' : 's'}` +
      (response.data_through ? ` · data through ${response.data_through}` : ''),
  };
}

async function routeCurrentValue(
  kpiId: string,
  brand: string | undefined,
  region: string | undefined,
  query: KpiChartQuery
): Promise<KpiChartData> {
  const result = await getKPIValue(kpiId, brand, region);
  const title = query.title ?? displayName(kpiId);

  if (result.value === undefined || result.value === null) {
    return emptyResult(
      result.error
        ? `${displayName(kpiId)} could not be calculated: ${result.error}`
        : `${displayName(kpiId)} has no historical series and no current value.`,
      title
    );
  }

  // Region provenance gate (#1538): chart a region-captioned figure ONLY when
  // the backend attests the region-scoped variant computed it. Anything else
  // — 'not_applicable' (no variant for this calculator) or a backend that
  // reports no provenance at all (pre-#1538) — is a global value, and drawing
  // it under the region caption is the exact mislabel this gate removes.
  if (region && result.region_status !== 'applied') {
    return emptyResult(
      result.region_status === 'not_applicable'
        ? `${displayName(kpiId)} has no ${region} scope — its calculator is global-only, ` +
          'so the current value covers all regions.'
        : `${displayName(kpiId)} could not be verified as ${region}-scoped — the backend ` +
          'reported no region provenance for this value.',
      title
    );
  }

  // A confidence interval turns a single number into a range worth DRAWING —
  // the causal metrics (CM-*) carry one, and whether it crosses zero is the
  // point of reading it. Flint has no error-bar template, but its output is a
  // plain Plotly figure and Plotly draws error bars natively, so the interval
  // is attached after assembly (see attachErrorBars).
  const interval = result.confidence_interval;
  const entry = lookupKpi(kpiId);
  const target = entry?.target;

  const rows: ChartRow[] = [
    {
      kpi: displayName(kpiId),
      value: result.value,
      // Only present when the registry declares a target; never invented.
      ...(target !== undefined ? { target } : {}),
      ...(interval ? { ci_low: interval[0], ci_high: interval[1] } : {}),
    },
  ];

  const semantic = valueSemantic(kpiId);
  // The caption carries the region only when it survived the provenance gate
  // above — result.region_applied IS the applied enum label at this point.
  const scope = scopeLabel(brand, region ? (result.region_applied ?? region) : undefined);
  const ciNote = interval
    ? ` · 95% CI ${formatBound(interval[0])} to ${formatBound(interval[1])}`
    : '';

  // A KPI Card compiles to a Plotly `indicator`, which has no error bars — so
  // a KPI that reports an interval is drawn as a bar with whiskers instead.
  // Without an interval the card is the better read of a lone figure, with the
  // registry's threshold target as its goal marker.
  const chartType = query.chartType ?? (interval ? 'Bar Chart' : 'KPI Card');

  return {
    rows,
    semanticTypes: {
      kpi: 'Category',
      value: semantic,
      ...(target !== undefined ? { target: semantic } : {}),
      ...(interval ? { ci_low: semantic, ci_high: semantic } : {}),
    },
    encoding: {
      axis: 'kpi',
      value: 'value',
      ...(target !== undefined ? { goal: 'target' } : {}),
    },
    chartType,
    ...(interval ? { errorBars: { low: 'ci_low', high: 'ci_high' } } : {}),
    title,
    subtitle:
      `${scope} · point-in-time value${ciNote}` +
      (target !== undefined ? ` · target ${formatBound(target)}` : '') +
      (result.calculated_at ? ` · as of ${result.calculated_at.slice(0, 10)}` : ''),
  };
}

async function routeComparison(
  kpiIds: string[],
  brand: string | undefined,
  region: string | undefined,
  query: KpiChartQuery
): Promise<KpiChartData> {
  const response = await batchCalculateKPIs({
    kpi_ids: kpiIds,
    context: brand || region ? { ...(brand ? { brand } : {}), ...(region ? { region } : {}) } : undefined,
  });
  const withValue = (response.results ?? []).filter(
    (r) => r.value !== undefined && r.value !== null
  );
  const title = query.title ?? 'KPI comparison';

  // Region provenance gate (#1538): a comparison axis captioned with the
  // region may draw ONLY values the backend attests are region-scoped —
  // mixing an attested northeast figure with a global one on the same labeled
  // axis would mislabel the global one. KPIs whose calculators are
  // global-only are omitted and counted in the caption.
  const results = region
    ? withValue.filter((r) => r.region_status === 'applied')
    : withValue;
  const regionOmitted = region ? withValue.length - results.length : 0;

  if (results.length === 0) {
    return emptyResult(
      region && withValue.length > 0
        ? `None of ${kpiIds.map(displayName).join(', ')} has a ${region} scope — their ` +
          'calculators are global-only, so a region-labeled comparison would mislabel ' +
          'global values.'
        : `None of ${kpiIds.map(displayName).join(', ')} returned a value.`,
      title
    );
  }

  // Mixed semantic types on one axis would mislabel the ticks (a percentage and
  // a raw count cannot share a formatted axis). Fall back to an unformatted
  // numeric axis and say so, rather than formatting everything as the first.
  const semantics = new Set(results.map((r) => valueSemantic(r.kpi_id)));
  const shared = semantics.size === 1 ? [...semantics][0] : 'Number';
  const mixedNote = semantics.size > 1 ? ' · mixed units, axis unformatted' : '';

  // Several causal metrics side by side, each with its interval, IS the forest
  // plot. Draw the whiskers only when EVERY plotted KPI reports an interval —
  // whiskers on some bars and not others reads as "these three are certain",
  // which is the opposite of what a missing interval means.
  const intervals = results.map((r) => r.confidence_interval);
  const allHaveIntervals = intervals.every(
    (ci): ci is [number, number] => Array.isArray(ci) && ci.length === 2
  );

  const rows: ChartRow[] = results.map((result, i) => ({
    kpi: displayName(result.kpi_id),
    value: result.value as number,
    ...(allHaveIntervals
      ? { ci_low: intervals[i]![0], ci_high: intervals[i]![1] }
      : {}),
  }));

  const someMissing = !allHaveIntervals && intervals.some((ci) => Array.isArray(ci));
  const skipped = kpiIds.length - results.length - regionOmitted;
  const chartType = query.chartType ?? DEFAULT_COMPARISON_CHART;
  return {
    rows,
    semanticTypes: {
      kpi: 'Category',
      value: shared,
      ...(allHaveIntervals ? { ci_low: shared, ci_high: shared } : {}),
    },
    encoding: { axis: 'kpi', value: 'value' },
    chartType,
    ...(allHaveIntervals ? { errorBars: { low: 'ci_low', high: 'ci_high' } } : {}),
    title,
    subtitle:
      `${scopeLabel(brand, region)} · ${results.length} KPI${results.length === 1 ? '' : 's'}` +
      (regionOmitted > 0
        ? ` · ${regionOmitted} global-only (no ${region} scope), omitted`
        : '') +
      (skipped > 0 ? ` · ${skipped} returned no value` : '') +
      (allHaveIntervals ? ' · 95% CI' : '') +
      (someMissing ? ' · intervals omitted (not reported for every KPI)' : '') +
      mixedNote,
  };
}

function formatBound(value: number): string {
  return Number.isFinite(value) ? value.toFixed(3).replace(/\.?0+$/, '') : String(value);
}

function emptyResult(reason: string, title: string): KpiChartData {
  return {
    rows: [],
    semanticTypes: {},
    encoding: { axis: '', value: '' },
    chartType: DEFAULT_COMPARISON_CHART,
    title,
    subtitle: '',
    emptyReason: reason,
  };
}
