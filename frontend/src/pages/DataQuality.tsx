/**
 * Data Quality Page
 * =================
 *
 * Dashboard for data profiling, completeness metrics, accuracy checks,
 * and validation rule monitoring.
 *
 * Wires to backend surfaces:
 *   - `useKPIList({ workstream: 'ws1_data_quality' })` (KPI list)
 *   - `useKPIDetail(kpi_id)`                           (per-KPI drill-down: metadata + current value)
 *   - `useKPIDetail` on the five dimension-source KPIs (dimension cards, portfolio scope)
 *
 * The quality dimension cards derive from MEASURED WS1 data-quality KPI values
 * (see `DIMENSION_KPI_IDS`), not from drift monitoring: the page previously read
 * drift for a `data_quality_pipeline` model id that is not a registered model —
 * no sweep enumerates it and no `ml_predictions` rows exist for it, so the
 * cards could structurally never leave "No data". Model & data drift are
 * monitored on the Monitoring page (`/monitoring`); this page links there
 * instead of duplicating a permanently-empty drift section.
 *
 * Issues addressed:
 *   - #301 (replace mock with live wiring)
 *   - #306 (preserve Playwright DOM contract: DataProfilingTab, QualityIssuesTab,
 *           ValidationRulesTab, RefreshButton; dimension cards by name)
 *
 * @module pages/DataQuality
 */

import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import {
  Database,
  RefreshCw,
  Download,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  MinusCircle,
  Search,
  Filter,
  Table as TableIcon,
  BarChart3,
  FileText,
  Loader2,
} from 'lucide-react';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { KPICard } from '@/components/visualizations';
import { QueryErrorState } from '@/components/ui/query-error-state';
import { useKPIList, useKPIDetail } from '@/hooks/api/use-kpi';
import { formatKpiValue } from '@/lib/kpi-format';
import { queryKeys } from '@/lib/query-client';
import { Workstream } from '@/types/kpi';
import type { KPIMetadata } from '@/types/kpi';

// =============================================================================
// CONSTANTS
// =============================================================================

/**
 * WS1 KPI ids each quality dimension card derives from (portfolio scope).
 *
 * Completeness = DQ-005 completeness pass rate (a real measured fraction — NOT
 * the old `70 + 2×KPI-count` registry-size proxy). Accuracy = DQ-003
 * cross-source match rate. Consistency = 1 − DQ-006 max regional share gap
 * (DQ-006 is lower-is-better, so the card shows the agreement complement).
 * Timeliness = mean target-attainment of DQ-007 data lag and DQ-009
 * time-to-release (both lower-is-better; attainment = target/actual, capped
 * at 100%). Card STATUS comes from each KPI's backend-authoritative,
 * direction-aware status — never re-scored client-side (an 80% match rate
 * beating its 75% target is healthy, not "below 85%").
 */
const DIMENSION_KPI_IDS = {
  completeness: 'WS1-DQ-005',
  accuracy: 'WS1-DQ-003',
  consistency: 'WS1-DQ-006',
  dataLag: 'WS1-DQ-007',
  timeToRelease: 'WS1-DQ-009',
} as const;

/**
 * Brand / region cuts the DQ rule values can be sliced by.
 *
 * The KPI calculators are brand- and region-aware (mig 078) and `GET
 * /api/kpis/{id}` accepts both, but the page never forwarded either — so every
 * rule value read the portfolio aggregate ("why only aggregated metrics?").
 * `All` / `All US` mean "no filter" (portfolio). Region is sent lowercased to
 * match the backend (KPI SQL matches case-insensitively; region data is stored
 * lowercase) — mirrors Home.tsx's `regionToParam`.
 *
 * SCOPE (honest): only the per-rule values in the Validation Rules table are
 * brand/region-aware. The dimension cards read the portfolio aggregate
 * regardless of these selectors (their source KPIs are portfolio-scoped).
 */
const DQ_BRAND_OPTIONS = [
  { value: 'All', label: 'All Brands' },
  { value: 'Remibrutinib', label: 'Remibrutinib' },
  { value: 'Fabhalta', label: 'Fabhalta' },
  { value: 'Kisqali', label: 'Kisqali' },
] as const;

const DQ_REGION_OPTIONS = [
  { value: 'All US', label: 'All US Regions' },
  { value: 'Northeast', label: 'Northeast' },
  { value: 'South', label: 'South' },
  { value: 'Midwest', label: 'Midwest' },
  { value: 'West', label: 'West' },
] as const;

// =============================================================================
// HELPERS
// =============================================================================

function formatTimestamp(timestamp: string | undefined): string {
  if (!timestamp) return '—';
  try {
    return new Date(timestamp).toLocaleString();
  } catch {
    return timestamp;
  }
}

/** Row status — backend-authoritative; `unknown` = no data, NOT a failure. */
type RuleStatus = 'pass' | 'warning' | 'fail' | 'unknown';

// Map the backend KPIStatus (good/warning/critical/unknown) to the row's
// display status. The backend status is AUTHORITATIVE and direction-aware:
// lower-is-better DQ KPIs (geographic gap, data lag, time-to-release) are scored
// correctly server-side, and a null/no-rows value comes back as `unknown` with
// an error reason. The page previously RE-derived status from (value, threshold)
// with a naive higher-is-better rule that (a) turned an UNKNOWN/null value into a
// fail-X — showing "no data" as a quality failure — and (b) mis-scored
// lower-is-better KPIs. Trust the backend status instead.
function ruleStatusFromKPI(value?: { status?: string | null }): RuleStatus {
  switch ((value?.status ?? '').toString().toLowerCase()) {
    case 'good':
      return 'pass';
    case 'warning':
      return 'warning';
    case 'critical':
      return 'fail';
    default:
      return 'unknown';
  }
}

/** Dimension-card status, mapped from the backend KPI status (same authority
 * as `ruleStatusFromKPI`); `neutral` = no data. */
type CardStatus = 'healthy' | 'warning' | 'critical' | 'neutral';

function cardStatusFromKPI(value?: { status?: string | null }): CardStatus {
  switch ((value?.status ?? '').toString().toLowerCase()) {
    case 'good':
      return 'healthy';
    case 'warning':
      return 'warning';
    case 'critical':
      return 'critical';
    default:
      return 'neutral';
  }
}

/** Worst-of composition for multi-KPI dimensions (critical > warning > healthy). */
function worstCardStatus(...statuses: CardStatus[]): CardStatus {
  if (statuses.includes('critical')) return 'critical';
  if (statuses.includes('warning')) return 'warning';
  if (statuses.includes('healthy')) return 'healthy';
  return 'neutral';
}

function statusIcon(status: RuleStatus) {
  switch (status) {
    case 'pass':
      return <CheckCircle2 className="h-4 w-4 text-emerald-500" />;
    case 'warning':
      return <AlertTriangle className="h-4 w-4 text-amber-500" />;
    case 'fail':
      return <XCircle className="h-4 w-4 text-rose-500" />;
    case 'unknown':
      return <MinusCircle className="h-4 w-4 text-muted-foreground" />;
  }
}

// =============================================================================
// SUB-COMPONENT — KPI DRILL-DOWN ROW
// =============================================================================

/**
 * Drill-down row for a single KPI: fetches metadata + current value via
 * `useKPIDetail` and renders threshold-aware status.
 *
 * Page-local (NOT exported) to avoid collisions with sibling wiring PRs.
 */
function KPIDrilldownRow({
  kpi,
  statusFilter = 'all',
  onStatusComputed,
  brand,
  region,
}: {
  kpi: KPIMetadata;
  statusFilter?: string;
  onStatusComputed?: (kpiId: string, status: RuleStatus) => void;
  brand?: string;
  region?: string;
}) {
  const { metadata, value, isLoading, error } = useKPIDetail(kpi.id, brand, region);

  // Prefer freshly fetched metadata; fall back to the list item to avoid a flash
  const effectiveMeta = metadata ?? kpi;
  const numericValue = typeof value?.value === 'number' ? value.value : undefined;
  const threshold = effectiveMeta.threshold;
  // Backend-authoritative, direction-aware status (unknown = no data, not fail).
  const ruleStatus = ruleStatusFromKPI(value);

  // #322 — wire the status filter to the per-rule status. The per-KPI value
  // endpoint returns the backend `status` (good/warning/critical/unknown);
  // selecting Pass/Warning/Fail hides non-matching rows and the parent uses the
  // reported status to drive the empty-state when ALL rows are filtered.
  // `unknown` rows never match Pass/Warning/Fail — "no data" is not a failure.
  useEffect(() => {
    onStatusComputed?.(kpi.id, ruleStatus);
  }, [kpi.id, ruleStatus, onStatusComputed]);

  if (statusFilter !== 'all' && statusFilter !== ruleStatus) {
    return null;
  }

  return (
    <tr className="border-b border-border hover:bg-muted/50">
      <td className="py-3 px-4">{statusIcon(ruleStatus)}</td>
      <td className="py-3 px-4">
        <div>
          <p className="font-medium">{effectiveMeta.name}</p>
          <p className="text-xs text-muted-foreground">{effectiveMeta.definition}</p>
        </div>
      </td>
      <td className="py-3 px-4 text-sm capitalize">
        {effectiveMeta.workstream.replace(/_/g, ' ')}
      </td>
      <td className="py-3 px-4">
        <code className="text-sm bg-muted px-1.5 py-0.5 rounded">
          {effectiveMeta.tables?.[0] ?? '—'}
        </code>
      </td>
      <td className="py-3 px-4">
        <Badge variant="outline" className="capitalize">
          {effectiveMeta.calculation_type.replace(/_/g, ' ')}
        </Badge>
      </td>
      <td className="py-3 px-4 text-right">
        {isLoading ? (
          <span className="text-muted-foreground inline-flex items-center gap-1">
            <Loader2 className="h-3 w-3 animate-spin" /> Loading
          </span>
        ) : error ? (
          <span className="text-rose-500 text-sm">error</span>
        ) : numericValue !== undefined ? (
          <span className="font-medium">
            {formatKpiValue(numericValue, {
              unit: effectiveMeta.unit,
              valueFormat: effectiveMeta.value_format,
            })}
          </span>
        ) : (
          <span
            className="text-muted-foreground"
            title={ruleStatus === 'unknown' ? (value?.error ?? undefined) : undefined}
          >
            {ruleStatus === 'unknown' ? 'No data' : '—'}
          </span>
        )}
        {threshold?.target !== undefined && (
          <span className="text-muted-foreground text-xs ml-1">
            / target{' '}
            {formatKpiValue(threshold.target, {
              unit: effectiveMeta.unit,
              valueFormat: effectiveMeta.value_format,
            })}
          </span>
        )}
        {/* Synthetic-mode provenance: label the figure so a reviewer never reads
            a synthetic-gold value as real data (E2I_KPI_INCLUDE_SYNTHETIC). */}
        {value?.data_source === 'synthetic' && numericValue !== undefined && (
          <Badge variant="outline" className="ml-2 text-[10px] px-1 py-0">
            synthetic data
          </Badge>
        )}
      </td>
      <td className="py-3 px-4 text-sm text-muted-foreground">
        {formatTimestamp(value?.calculated_at)}
      </td>
    </tr>
  );
}

// =============================================================================
// SUB-COMPONENT — QUALITY ISSUE ROW
// =============================================================================

/**
 * Quality Issues entry for a single KPI: renders ONLY when the KPI's
 * backend-authoritative status is warning/critical (a real threshold breach).
 * Portfolio scope — an issue is an issue regardless of the rules-table cut.
 * Reports its status up so the parent can render the no-issues empty-state.
 */
function KPIIssueRow({
  kpi,
  onStatusComputed,
}: {
  kpi: KPIMetadata;
  onStatusComputed?: (
    kpiId: string,
    status: RuleStatus | 'pending' | 'error',
    fetching: boolean
  ) => void;
}) {
  const { metadata, value, isLoading, isFetching, error } = useKPIDetail(kpi.id);
  const effectiveMeta = metadata ?? kpi;
  const ruleStatus = ruleStatusFromKPI(value);
  const numericValue = typeof value?.value === 'number' ? value.value : undefined;
  // A failed fetch with NO cached value means the check itself never ran —
  // report 'error' and render a visible failed-check row, or an errored KPI
  // is indistinguishable from a healthy one (ruleStatusFromKPI maps both
  // no-data and errored to 'unknown'). With a cached value present the stale
  // settled status wins — same stale-beats-blank policy as the dimension
  // cards; a transient refetch blip must not flip real data into alarm.
  const fetchFailed = Boolean(error) && value === undefined;

  // Report 'pending' until the detail fetch settles: the parent's no-issues
  // empty-state must wait for every row, or a slow /api/kpis/{id} response
  // reads as a clean bill of health moments before a breach pops in.
  // During a BACKGROUND refetch of cached data (Refresh click, prod
  // window-focus refetch) isLoading stays false — the last settled status
  // keeps being reported so listed issues and the count never flicker — but
  // the fetching flag tells the parent to hold the "no issues" claim until
  // the recheck lands.
  useEffect(() => {
    onStatusComputed?.(
      kpi.id,
      isLoading ? 'pending' : fetchFailed ? 'error' : ruleStatus,
      isLoading || isFetching
    );
  }, [kpi.id, ruleStatus, isLoading, isFetching, fetchFailed, onStatusComputed]);

  if (fetchFailed) {
    return (
      <li className="p-3 rounded-lg border border-border bg-card flex items-start justify-between gap-3">
        <div>
          <p className="font-medium text-sm">
            {effectiveMeta.name} <span className="text-muted-foreground">({kpi.id})</span>
          </p>
          <p className="text-xs text-rose-500 mt-0.5">
            Quality check failed to load — thresholds could not be verified.
          </p>
        </div>
        <Badge variant="outline" className="capitalize text-rose-500 border-rose-500">
          check failed
        </Badge>
      </li>
    );
  }

  if (ruleStatus !== 'warning' && ruleStatus !== 'fail') {
    return null;
  }

  return (
    <li className="p-3 rounded-lg border border-border bg-card flex items-start justify-between gap-3">
      <div>
        <p className="font-medium text-sm">
          {effectiveMeta.name} <span className="text-muted-foreground">({kpi.id})</span>
        </p>
        <p className="text-xs text-muted-foreground mt-0.5">
          Current{' '}
          {numericValue !== undefined
            ? formatKpiValue(numericValue, {
                unit: effectiveMeta.unit,
                valueFormat: effectiveMeta.value_format,
              })
            : '—'}
          {effectiveMeta.threshold?.target !== undefined && (
            <>
              {' '}
              vs target{' '}
              {formatKpiValue(effectiveMeta.threshold.target, {
                unit: effectiveMeta.unit,
                valueFormat: effectiveMeta.value_format,
              })}
            </>
          )}
        </p>
      </div>
      <Badge
        variant={ruleStatus === 'fail' ? 'destructive' : 'secondary'}
        className="capitalize"
      >
        {ruleStatus === 'fail' ? 'critical' : 'warning'}
      </Badge>
    </li>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

function DataQuality() {
  const [searchQuery, setSearchQuery] = useState('');
  const [ruleStatusFilter, setRuleStatusFilter] = useState<string>('all');
  // F3 — brand/region cut for the per-rule values. Default = portfolio aggregate.
  const [selectedBrand, setSelectedBrand] = useState<string>('All');
  const [selectedRegion, setSelectedRegion] = useState<string>('All US');
  // #322 — per-row computed status reported up by each KPIDrilldownRow. Lets us
  // render the "No data quality KPIs match your filters" empty-state when the
  // status filter hides every row. Rules rows report the brand/region-cut
  // status; the Quality Issues tab keeps its own portfolio-scoped map below so
  // a cut status can never leak into the issues empty-state.
  const [kpiStatuses, setKpiStatuses] = useState<Record<string, RuleStatus>>({});
  // Quality Issues rows report 'pending' until their detail fetch settles — the
  // no-issues empty-state must not render from not-yet-loaded rows. `fetching`
  // is tracked separately from status: a background refetch keeps the last
  // settled status (issue rows/count stay stable) while still suppressing the
  // "no issues" claim until the recheck lands.
  const [issueStatuses, setIssueStatuses] = useState<
    Record<string, { status: RuleStatus | 'pending' | 'error'; fetching: boolean }>
  >({});

  const queryClient = useQueryClient();

  // ---------------------------------------------------------------------------
  // LIVE DATA — KPI workstream ws1_data_quality
  // ---------------------------------------------------------------------------
  const {
    data: kpiList,
    isLoading: kpiLoading,
    error: kpiError,
    refetch: refetchKpis,
    isRefetching: kpiRefetching,
  } = useKPIList({ workstream: Workstream.WS1_DATA_QUALITY });

  // ---------------------------------------------------------------------------
  // LIVE DATA — dimension-source KPIs (portfolio scope; see DIMENSION_KPI_IDS)
  // ---------------------------------------------------------------------------
  const completenessKpi = useKPIDetail(DIMENSION_KPI_IDS.completeness);
  const accuracyKpi = useKPIDetail(DIMENSION_KPI_IDS.accuracy);
  const consistencyKpi = useKPIDetail(DIMENSION_KPI_IDS.consistency);
  const dataLagKpi = useKPIDetail(DIMENSION_KPI_IDS.dataLag);
  const ttrKpi = useKPIDetail(DIMENSION_KPI_IDS.timeToRelease);

  // ---------------------------------------------------------------------------
  // DERIVED VALUES
  // ---------------------------------------------------------------------------
  const allKpis = useMemo<KPIMetadata[]>(() => kpiList?.kpis ?? [], [kpiList]);

  // F3 — map the selected cut to backend query params. 'All' / 'All US' = no
  // filter (portfolio); region lowercased to match the backend's stored case.
  const brandParam = selectedBrand === 'All' ? undefined : selectedBrand;
  const regionParam =
    selectedRegion === 'All US' ? undefined : selectedRegion.toLowerCase();

  const filteredKpis = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return allKpis.filter((kpi) => {
      const matchesSearch =
        q === '' ||
        kpi.name.toLowerCase().includes(q) ||
        kpi.id.toLowerCase().includes(q) ||
        (kpi.definition ?? '').toLowerCase().includes(q);
      return matchesSearch;
    });
    // Status filter (#322) is applied per-row inside KPIDrilldownRow since the
    // computed status depends on useKPIDetail's per-row value fetch.
  }, [allKpis, searchQuery]);

  // #322 — visible-row count after status filter is applied. Drives the
  // empty-state message when no row matches the chosen status.
  const visibleKpiCount = useMemo(() => {
    if (ruleStatusFilter === 'all') return filteredKpis.length;
    return filteredKpis.reduce((count, kpi) => {
      const status = kpiStatuses[kpi.id];
      // Conservatively treat unknown (not yet reported) as visible; the row
      // will hide itself once it reports its status on the next render tick.
      if (status === undefined || status === ruleStatusFilter) return count + 1;
      return count;
    }, 0);
  }, [filteredKpis, ruleStatusFilter, kpiStatuses]);

  // Quality Issues tab — count of KPIs whose backend status breaches thresholds,
  // plus whether every issue row's detail fetch has settled (rows report
  // 'pending' while in flight; a row that hasn't mounted yet reports nothing).
  // issueCount deliberately ignores the fetching flag — it reads the last
  // settled statuses so real issues never blink out during a refetch.
  const issueCount = useMemo(
    () =>
      allKpis.filter((kpi) => {
        const s = issueStatuses[kpi.id]?.status;
        return s === 'warning' || s === 'fail';
      }).length,
    [allKpis, issueStatuses]
  );
  // Failed checks count separately from breaches: an errored fetch (no cached
  // value) is NOT a breach, but an unverifiable check is not a passing check
  // either — it must block the unqualified "no issues" claim.
  const issueErrorCount = useMemo(
    () => allKpis.filter((kpi) => issueStatuses[kpi.id]?.status === 'error').length,
    [allKpis, issueStatuses]
  );
  // Settled = every row has a non-pending status AND no fetch is in flight.
  // A background refetch un-settles this, downgrading the "no issues" claim
  // to the checking indicator until the recheck lands. 'error' IS settled —
  // a failed check reports immediately and renders its own row.
  const issuesSettled = useMemo(
    () =>
      allKpis.length > 0 &&
      allKpis.every((kpi) => {
        const entry = issueStatuses[kpi.id];
        return entry !== undefined && entry.status !== 'pending' && !entry.fetching;
      }),
    [allKpis, issueStatuses]
  );

  // Dimension scores, derived from MEASURED KPI values (see DIMENSION_KPI_IDS).
  // A dimension whose source KPI has no value stays undefined → the card reads
  // an honest "No data" — never a fabricated healthy default.
  const qualityScores = useMemo<{
    completeness: number | undefined;
    accuracy: number | undefined;
    consistency: number | undefined;
    timeliness: number | undefined;
    overall: number | undefined;
    statuses: {
      completeness: CardStatus;
      accuracy: CardStatus;
      consistency: CardStatus;
      timeliness: CardStatus;
      overall: CardStatus;
    };
  }>(() => {
    const clampPct = (n: number) => Math.max(0, Math.min(100, n));
    const fraction = (v: unknown): number | undefined =>
      typeof v === 'number' ? clampPct(v * 100) : undefined;

    // DQ-005 completeness pass rate and DQ-003 cross-source match rate are 0-1
    // fractions of records; the card shows them directly as percentages.
    const completeness = fraction(completenessKpi.value?.value);
    const accuracy = fraction(accuracyKpi.value?.value);

    // DQ-006 is the MAX regional share gap vs the reference universe (lower is
    // better); the card shows the agreement complement (1 − gap).
    const gap = consistencyKpi.value?.value;
    const consistency =
      typeof gap === 'number' ? clampPct((1 - gap) * 100) : undefined;

    // DQ-007 (median lag, days) and DQ-009 (time-to-release, hours) are
    // lower-is-better vs their configured targets; attainment = target/actual,
    // capped at 100% (beating the target is full attainment, not >100%).
    const attainment = (v: unknown, target: unknown): number | undefined => {
      if (typeof v !== 'number' || typeof target !== 'number' || target <= 0) {
        return undefined;
      }
      if (v <= 0) return 100; // instantaneous is full attainment
      return clampPct((target / v) * 100);
    };
    const lagScore = attainment(
      dataLagKpi.value?.value,
      dataLagKpi.metadata?.threshold?.target
    );
    const ttrScore = attainment(ttrKpi.value?.value, ttrKpi.metadata?.threshold?.target);
    const timelinessParts = [lagScore, ttrScore].filter(
      (x): x is number => x !== undefined
    );
    const timeliness = timelinessParts.length
      ? timelinessParts.reduce((a, b) => a + b, 0) / timelinessParts.length
      : undefined;

    // Card statuses come from the backend's direction-aware KPI statuses.
    const statuses = {
      completeness: cardStatusFromKPI(completenessKpi.value),
      accuracy: cardStatusFromKPI(accuracyKpi.value),
      consistency: cardStatusFromKPI(consistencyKpi.value),
      timeliness: worstCardStatus(
        cardStatusFromKPI(dataLagKpi.value),
        cardStatusFromKPI(ttrKpi.value)
      ),
      overall: 'neutral' as CardStatus,
    };

    const present = (
      [
        { v: completeness, s: statuses.completeness },
        { v: accuracy, s: statuses.accuracy },
        { v: consistency, s: statuses.consistency },
        { v: timeliness, s: statuses.timeliness },
      ] as const
    ).filter((p): p is { v: number; s: CardStatus } => p.v !== undefined);
    const overall = present.length
      ? present.reduce((a, p) => a + p.v, 0) / present.length
      : undefined;
    // Overall status = worst of the measured dimensions (a quality composite
    // must not read healthy while a component dimension is critical).
    statuses.overall = present.length
      ? worstCardStatus(...present.map((p) => p.s))
      : 'neutral';

    return { completeness, accuracy, consistency, timeliness, overall, statuses };
  }, [
    completenessKpi.value,
    accuracyKpi.value,
    consistencyKpi.value,
    dataLagKpi.value,
    dataLagKpi.metadata,
    ttrKpi.value,
    ttrKpi.metadata,
  ]);

  const dimensionsLoading =
    kpiLoading ||
    completenessKpi.isLoading ||
    accuracyKpi.isLoading ||
    consistencyKpi.isLoading ||
    dataLagKpi.isLoading ||
    ttrKpi.isLoading;

  // A failed dimension-source fetch must be distinguishable from a genuine
  // no-data gap: a 500 on a /api/kpis/{id} read is an API outage, not missing
  // data. Errored cards read a rose "Error" instead of the neutral "No data".
  // (A dimension holding cached data keeps showing its value through a failed
  // refetch — stale beats blank.)
  const dimensionErrors = {
    completeness: Boolean(completenessKpi.error),
    accuracy: Boolean(accuracyKpi.error),
    consistency: Boolean(consistencyKpi.error),
    timeliness: Boolean(dataLagKpi.error || ttrKpi.error),
  };

  // ---------------------------------------------------------------------------
  // HANDLERS
  // ---------------------------------------------------------------------------
  const handleRefresh = () => {
    // Invalidate every KPI query (list + per-KPI metadata/value) so the
    // dimension cards, rules table, and issues tab all refetch live values.
    void queryClient.invalidateQueries({ queryKey: queryKeys.kpi.all() });
    void refetchKpis();
  };

  const handleExport = () => {
    const report = {
      generatedAt: new Date().toISOString(),
      qualityScores,
      kpis: allKpis,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `data-quality-report-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const isRefreshing = kpiRefetching;
  // #325 — Export button disabled while any underlying dataset is still loading.
  // Prevents partial JSON export (undefined dimension scores / [] kpis).
  const isAnyLoading = dimensionsLoading;

  // ---------------------------------------------------------------------------
  // RENDER
  // ---------------------------------------------------------------------------
  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Database className="h-8 w-8" />
            Data Quality
          </h1>
          <p className="text-muted-foreground">
            Data profiling, completeness metrics, accuracy checks, and validation rules.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={handleRefresh} disabled={isRefreshing}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button variant="outline" onClick={handleExport} disabled={isAnyLoading}>
            <Download className="h-4 w-4 mr-2" />
            Export Report
          </Button>
        </div>
      </div>

      {/* Error states for KPI list */}
      {kpiError && (
        <div className="mb-6">
          <QueryErrorState
            error={kpiError}
            onRetry={refetchKpis}
            isRetrying={kpiRefetching}
            title="Could not load KPI list"
          />
        </div>
      )}

      {/* Quality Score Overview (dimension cards — names preserved for spec) */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-2">
        {dimensionsLoading ? (
          <div className="col-span-full flex items-center gap-2 text-muted-foreground py-6">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>Loading quality dimensions...</span>
          </div>
        ) : (
          <>
            {(
              [
                {
                  title: 'Overall Quality',
                  v: qualityScores.overall,
                  status: qualityScores.statuses.overall,
                  // Overall reads "Error" only when NOTHING measured and at
                  // least one source errored; with partial data it stays the
                  // honest mean of the measured dimensions.
                  error:
                    qualityScores.overall === undefined &&
                    Object.values(dimensionErrors).some(Boolean),
                  description: 'Mean of the measured dimension scores (status = worst dimension)',
                },
                {
                  title: 'Completeness',
                  v: qualityScores.completeness,
                  status: qualityScores.statuses.completeness,
                  error: dimensionErrors.completeness,
                  description: 'Completeness pass rate across brand-critical fields (WS1-DQ-005)',
                },
                {
                  title: 'Accuracy',
                  v: qualityScores.accuracy,
                  status: qualityScores.statuses.accuracy,
                  error: dimensionErrors.accuracy,
                  description: 'Cross-source match rate (WS1-DQ-003)',
                },
                {
                  title: 'Consistency',
                  v: qualityScores.consistency,
                  status: qualityScores.statuses.consistency,
                  error: dimensionErrors.consistency,
                  description:
                    '100% minus the max regional share gap vs the reference universe (WS1-DQ-006)',
                },
                {
                  title: 'Timeliness',
                  v: qualityScores.timeliness,
                  status: qualityScores.statuses.timeliness,
                  error: dimensionErrors.timeliness,
                  description:
                    'Target attainment of data lag (WS1-DQ-007) and time-to-release (WS1-DQ-009)',
                },
              ] as const
            ).map((d) => (
              <KPICard
                key={d.title}
                title={d.title}
                // Honest empties: a measured value renders (stale data beats
                // blank through a failed refetch); otherwise a fetch error
                // reads "Error" (rose) and a genuine gap reads "No data" —
                // never a fabricated healthy default (see qualityScores).
                value={
                  d.v !== undefined
                    ? Math.round(d.v * 10) / 10
                    : d.error
                      ? 'Error'
                      : 'No data'
                }
                unit={d.v === undefined ? '' : '%'}
                status={d.v === undefined ? 'neutral' : d.status}
                valueColor={d.v === undefined && d.error ? 'text-rose-500' : undefined}
                description={d.description}
                sparklineData={[]}
                higherIsBetter
              />
            ))}
          </>
        )}
      </div>

      {/* Drift consolidation: model & data drift live on /monitoring, not here.
          The old per-page drift section read a `data_quality_pipeline` model id
          that no sweep monitors — it could never show data. */}
      <p className="text-sm text-muted-foreground mb-8">
        Dimension scores are derived from the measured WS1 data-quality KPIs in the
        table below. Model and data drift are monitored on the{' '}
        <Link to="/monitoring" className="underline underline-offset-2 hover:text-foreground">
          Monitoring
        </Link>{' '}
        page.
      </p>

      {/* Tabs: Validation Rules, Data Profiling, Quality Issues — names preserved for spec */}
      <Tabs defaultValue="rules" className="space-y-4">
        <TabsList>
          <TabsTrigger value="rules" className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4" />
            Validation Rules
          </TabsTrigger>
          <TabsTrigger value="profiling" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Data Profiling
          </TabsTrigger>
          <TabsTrigger value="issues" className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            Quality Issues
          </TabsTrigger>
        </TabsList>

        {/* Validation Rules: list of DQ KPIs with drill-down via useKPIDetail */}
        <TabsContent value="rules">
          <Card>
            <CardHeader>
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <CardTitle>Validation Rules</CardTitle>
                  <CardDescription>
                    Data quality KPIs from workstream <code>ws1_data_quality</code>. Rule
                    values reflect the selected brand and region.
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <div className="relative">
                    <Search
                      className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground"
                      aria-hidden="true"
                    />
                    {/* #328 — sr-only label associates accessible name with the input */}
                    <Label htmlFor="dq-search" className="sr-only">
                      Search validation rules
                    </Label>
                    <Input
                      id="dq-search"
                      placeholder="Search rules..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-9 w-64"
                    />
                  </div>
                  {/* F3 — brand/region cut selectors. Scoped here (Validation
                      Rules header) because only these per-rule values are
                      brand/region-aware; the dimension cards are not. */}
                  <Select value={selectedBrand} onValueChange={setSelectedBrand}>
                    <SelectTrigger className="w-40" aria-label="Filter rules by brand">
                      <SelectValue placeholder="Brand" />
                    </SelectTrigger>
                    <SelectContent>
                      {DQ_BRAND_OPTIONS.map((b) => (
                        <SelectItem key={b.value} value={b.value}>
                          {b.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Select value={selectedRegion} onValueChange={setSelectedRegion}>
                    <SelectTrigger className="w-44" aria-label="Filter rules by region">
                      <SelectValue placeholder="Region" />
                    </SelectTrigger>
                    <SelectContent>
                      {DQ_REGION_OPTIONS.map((r) => (
                        <SelectItem key={r.value} value={r.value}>
                          {r.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Select value={ruleStatusFilter} onValueChange={setRuleStatusFilter}>
                    {/* #328 — aria-label gives the SelectTrigger an accessible name */}
                    <SelectTrigger className="w-32" aria-label="Filter rules by status">
                      <Filter className="h-4 w-4 mr-2" aria-hidden="true" />
                      <SelectValue placeholder="Status" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All</SelectItem>
                      <SelectItem value="pass">Pass</SelectItem>
                      <SelectItem value="warning">Warning</SelectItem>
                      <SelectItem value="fail">Fail</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {kpiLoading ? (
                <div className="flex items-center gap-2 text-muted-foreground py-4">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Loading KPIs...</span>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left py-3 px-4 font-medium text-muted-foreground">
                          Status
                        </th>
                        <th className="text-left py-3 px-4 font-medium text-muted-foreground">
                          Rule Name
                        </th>
                        <th className="text-left py-3 px-4 font-medium text-muted-foreground">
                          Data Source
                        </th>
                        <th className="text-left py-3 px-4 font-medium text-muted-foreground">
                          Target Field
                        </th>
                        <th className="text-left py-3 px-4 font-medium text-muted-foreground">
                          Type
                        </th>
                        <th className="text-right py-3 px-4 font-medium text-muted-foreground">
                          Current Value
                        </th>
                        <th className="text-left py-3 px-4 font-medium text-muted-foreground">
                          Last Checked
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredKpis.map((kpi) => (
                        <KPIDrilldownRow
                          key={kpi.id}
                          kpi={kpi}
                          statusFilter={ruleStatusFilter}
                          brand={brandParam}
                          region={regionParam}
                          onStatusComputed={(id, status) =>
                            setKpiStatuses((prev) =>
                              prev[id] === status ? prev : { ...prev, [id]: status }
                            )
                          }
                        />
                      ))}
                    </tbody>
                  </table>
                  {visibleKpiCount === 0 && (
                    <div className="text-center py-8 text-muted-foreground">
                      <FileText className="h-8 w-8 mx-auto mb-2 opacity-50" />
                      <p>No data quality KPIs match your filters</p>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Data Profiling: column / table metadata derived from KPI registry */}
        <TabsContent value="profiling">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TableIcon className="h-5 w-5" />
                Data Profiling
              </CardTitle>
              <CardDescription>
                Source tables and columns covered by data quality monitoring
              </CardDescription>
            </CardHeader>
            <CardContent>
              {kpiLoading ? (
                <div className="flex items-center gap-2 text-muted-foreground py-4">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Loading profiling data...</span>
                </div>
              ) : allKpis.length === 0 ? (
                <p className="text-muted-foreground text-sm">
                  No KPIs registered for workstream <code>ws1_data_quality</code>.
                </p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left py-3 px-4 font-medium text-muted-foreground">
                          KPI
                        </th>
                        <th className="text-left py-3 px-4 font-medium text-muted-foreground">
                          Tables
                        </th>
                        <th className="text-left py-3 px-4 font-medium text-muted-foreground">
                          Columns
                        </th>
                        <th className="text-left py-3 px-4 font-medium text-muted-foreground">
                          Frequency
                        </th>
                        <th className="text-left py-3 px-4 font-medium text-muted-foreground">
                          Causal Library
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {allKpis.map((kpi) => (
                        <tr key={kpi.id} className="border-b border-border hover:bg-muted/50">
                          <td className="py-3 px-4">
                            <div>
                              <p className="font-medium">{kpi.name}</p>
                              <p className="text-xs text-muted-foreground">{kpi.id}</p>
                            </div>
                          </td>
                          <td className="py-3 px-4">
                            <div className="flex gap-1 flex-wrap">
                              {(kpi.tables ?? []).map((t) => (
                                <code key={t} className="text-xs bg-muted px-1.5 py-0.5 rounded">
                                  {t}
                                </code>
                              ))}
                            </div>
                          </td>
                          <td className="py-3 px-4">
                            <div className="flex gap-1 flex-wrap">
                              {(kpi.columns ?? []).slice(0, 4).map((c) => (
                                <code key={c} className="text-xs bg-muted px-1.5 py-0.5 rounded">
                                  {c}
                                </code>
                              ))}
                              {(kpi.columns ?? []).length > 4 && (
                                <span className="text-xs text-muted-foreground">
                                  +{(kpi.columns ?? []).length - 4} more
                                </span>
                              )}
                            </div>
                          </td>
                          <td className="py-3 px-4 text-sm">{kpi.frequency}</td>
                          <td className="py-3 px-4 text-sm capitalize">
                            {kpi.primary_causal_library}
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

        {/* Quality Issues: KPIs breaching their thresholds (backend status) */}
        <TabsContent value="issues">
          <Card>
            <CardHeader>
              <CardTitle>Quality Issues</CardTitle>
              <CardDescription>
                Data-quality KPIs currently breaching their warning or critical
                thresholds (portfolio scope)
              </CardDescription>
            </CardHeader>
            <CardContent>
              {kpiLoading ? (
                <div className="flex items-center gap-2 text-muted-foreground py-4">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Loading quality issues...</span>
                </div>
              ) : allKpis.length === 0 ? (
                <p className="text-muted-foreground text-sm">
                  No KPIs registered for workstream <code>ws1_data_quality</code>.
                </p>
              ) : (
                <>
                  {/* The no-issues empty-state renders only after EVERY row's
                      detail fetch has settled — a slow /api/kpis/{id} response
                      must not read as a clean bill of health. */}
                  {!issuesSettled ? (
                    <div className="flex items-center gap-2 text-muted-foreground text-sm mb-3">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span>Checking KPI thresholds...</span>
                    </div>
                  ) : issueCount === 0 && issueErrorCount > 0 ? (
                    <p className="text-muted-foreground text-sm mb-3">
                      No breaches detected among the KPIs that could be checked —
                      but {issueErrorCount}{' '}
                      {issueErrorCount === 1 ? 'quality check' : 'quality checks'}{' '}
                      failed to load and could not be verified.
                    </p>
                  ) : issueCount === 0 ? (
                    <p className="text-muted-foreground text-sm mb-3">
                      No data-quality KPIs are breaching their thresholds. Model and
                      data drift are monitored on the{' '}
                      <Link
                        to="/monitoring"
                        className="underline underline-offset-2 hover:text-foreground"
                      >
                        Monitoring
                      </Link>{' '}
                      page.
                    </p>
                  ) : null}
                  <ul className="space-y-3">
                    {allKpis.map((kpi) => (
                      <KPIIssueRow
                        key={kpi.id}
                        kpi={kpi}
                        onStatusComputed={(id, status, fetching) =>
                          setIssueStatuses((prev) => {
                            const cur = prev[id];
                            return cur && cur.status === status && cur.fetching === fetching
                              ? prev
                              : { ...prev, [id]: { status, fetching } };
                          })
                        }
                      />
                    ))}
                  </ul>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default DataQuality;
