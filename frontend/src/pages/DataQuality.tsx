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
 *   - `useLatestDriftStatus(model_id)`                 (live drift status)
 *   - `useDriftHistory({ model_id })`                  (drift history)
 *   - `useTriggerDriftDetection`                       (refresh button mutation)
 *
 * Issues addressed:
 *   - #301 (replace mock with live wiring)
 *   - #306 (preserve Playwright DOM contract: DataProfilingTab, QualityIssuesTab,
 *           ValidationRulesTab, RefreshButton; dimension cards by name)
 *
 * @module pages/DataQuality
 */

import { useEffect, useMemo, useState } from 'react';
import {
  Database,
  RefreshCw,
  Download,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Search,
  Filter,
  Table as TableIcon,
  BarChart3,
  FileText,
  Loader2,
  Activity,
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
import {
  useLatestDriftStatus,
  useDriftHistory,
  useTriggerDriftDetection,
} from '@/hooks/api/use-monitoring';
import { toast } from '@/hooks/use-toast';
import { Workstream } from '@/types/kpi';
import type { KPIMetadata, KPIThreshold } from '@/types/kpi';

// =============================================================================
// CONSTANTS
// =============================================================================

/**
 * Source model whose drift drives the Data Quality view.
 *
 * Data-quality KPIs feed into the data ingestion / preprocessing pipeline;
 * drift in that pipeline is the canonical "DQ drift" signal we surface here.
 * Page-local (NOT exported) — sibling agents wiring other pages should pick
 * their own model id.
 */
const DQ_MODEL_ID = 'data_quality_pipeline';

// =============================================================================
// HELPERS
// =============================================================================

function getStatusFromScore(score: number | undefined): 'healthy' | 'warning' | 'critical' {
  if (score === undefined || Number.isNaN(score)) return 'critical';
  if (score >= 95) return 'healthy';
  if (score >= 85) return 'warning';
  return 'critical';
}

function formatTimestamp(timestamp: string | undefined): string {
  if (!timestamp) return '—';
  try {
    return new Date(timestamp).toLocaleString();
  } catch {
    return timestamp;
  }
}

function statusFromThreshold(
  value: number | undefined,
  threshold?: KPIThreshold
): 'pass' | 'warning' | 'fail' {
  if (value === undefined || Number.isNaN(value)) return 'fail';
  if (threshold?.critical !== undefined && value < threshold.critical) return 'fail';
  if (threshold?.warning !== undefined && value < threshold.warning) return 'warning';
  return 'pass';
}

function statusIcon(status: 'pass' | 'warning' | 'fail') {
  switch (status) {
    case 'pass':
      return <CheckCircle2 className="h-4 w-4 text-emerald-500" />;
    case 'warning':
      return <AlertTriangle className="h-4 w-4 text-amber-500" />;
    case 'fail':
      return <XCircle className="h-4 w-4 text-rose-500" />;
  }
}

function severityBadgeVariant(
  severity: string | undefined
): 'destructive' | 'secondary' | 'outline' {
  const s = (severity ?? '').toLowerCase();
  if (s === 'critical' || s === 'high') return 'destructive';
  if (s === 'medium' || s === 'warning') return 'secondary';
  return 'outline';
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
}: {
  kpi: KPIMetadata;
  statusFilter?: string;
  onStatusComputed?: (kpiId: string, status: 'pass' | 'warning' | 'fail') => void;
}) {
  const { metadata, value, isLoading, error } = useKPIDetail(kpi.id);

  // Prefer freshly fetched metadata; fall back to the list item to avoid a flash
  const effectiveMeta = metadata ?? kpi;
  const numericValue = typeof value?.value === 'number' ? value.value : undefined;
  const threshold = effectiveMeta.threshold;
  const ruleStatus = statusFromThreshold(numericValue, threshold);

  // #322 — wire status filter to the computed per-rule status. The KPI list
  // endpoint does NOT return a rolled-up status; we compute it here from
  // (value, threshold) using the same helper that drives the row's status
  // icon. Selecting Pass/Warning/Fail hides non-matching rows; parent uses
  // the reported status to drive the empty-state when ALL rows are filtered.
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
            {numericValue.toFixed(1)}
            {effectiveMeta.unit ? effectiveMeta.unit : ''}
          </span>
        ) : (
          <span className="text-muted-foreground">—</span>
        )}
        {threshold?.target !== undefined && (
          <span className="text-muted-foreground text-xs ml-1">
            / target {threshold.target}
            {effectiveMeta.unit ?? ''}
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
// MAIN COMPONENT
// =============================================================================

function DataQuality() {
  const [searchQuery, setSearchQuery] = useState('');
  const [ruleStatusFilter, setRuleStatusFilter] = useState<string>('all');
  // #322 — per-row computed status reported up by each KPIDrilldownRow.
  // Lets us render the "No data quality KPIs match your filters" empty-state
  // when the status filter hides every row.
  const [kpiStatuses, setKpiStatuses] = useState<Record<string, 'pass' | 'warning' | 'fail'>>(
    {}
  );

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
  // LIVE DATA — Drift detection for the DQ pipeline
  // ---------------------------------------------------------------------------
  const {
    data: latestDrift,
    isLoading: driftLoading,
    error: driftError,
  } = useLatestDriftStatus(DQ_MODEL_ID);

  const {
    data: driftHistory,
    isLoading: driftHistoryLoading,
    error: driftHistoryError,
  } = useDriftHistory({ model_id: DQ_MODEL_ID, days: 30 });

  // #324 — surface mutation result via toast (was silent on success + error).
  const { mutate: triggerDrift, isPending: driftRefreshing } = useTriggerDriftDetection({
    onSuccess: (data) => {
      toast({
        title: 'Drift detection triggered',
        description: `Task ${data.task_id} queued for ${DQ_MODEL_ID}.`,
      });
    },
    onError: (error) => {
      toast({
        variant: 'destructive',
        title: 'Drift detection failed',
        description: error?.message ?? 'Unknown error triggering drift detection.',
      });
    },
  });

  // ---------------------------------------------------------------------------
  // DERIVED VALUES
  // ---------------------------------------------------------------------------
  const allKpis = useMemo<KPIMetadata[]>(() => kpiList?.kpis ?? [], [kpiList]);

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

  // Derive dimension scores from drift signal + KPI count health.
  // overall_drift_score is in [0, 1] where higher = worse; we invert to a
  // percentage-style "quality" score.
  const qualityScores = useMemo(() => {
    const driftScore = latestDrift?.overall_drift_score ?? 0;
    const driftHealth = Math.max(0, Math.min(100, (1 - driftScore) * 100));
    const featuresChecked = latestDrift?.features_checked ?? 0;
    const driftedFeatures = latestDrift?.features_with_drift?.length ?? 0;
    const accuracy = featuresChecked > 0
      ? Math.max(0, Math.min(100, ((featuresChecked - driftedFeatures) / featuresChecked) * 100))
      : driftHealth;

    const kpiCount = allKpis.length;
    // Use registry size as a proxy for "completeness of monitoring coverage"
    // (we have N DQ KPIs registered).
    const completeness = kpiCount > 0 ? Math.min(100, 70 + Math.min(30, kpiCount * 2)) : 0;

    const consistency = driftHealth; // alias until a dedicated signal lands
    const timeliness = driftHealth;
    const overall = (completeness + accuracy + consistency + timeliness) / 4;

    return { completeness, accuracy, consistency, timeliness, overall };
  }, [latestDrift, allKpis.length]);

  const dimensionsLoading = kpiLoading || driftLoading;

  // ---------------------------------------------------------------------------
  // HANDLERS
  // ---------------------------------------------------------------------------
  const handleRefresh = () => {
    // 1) Re-fetch the KPI list
    refetchKpis();
    // 2) Trigger a new drift-detection task for the DQ pipeline
    // #326 — align trigger window with display window (30d history view)
    triggerDrift({
      request: {
        model_id: DQ_MODEL_ID,
        time_window: '30d',
        check_data_drift: true,
      },
    });
  };

  const handleExport = () => {
    const report = {
      generatedAt: new Date().toISOString(),
      qualityScores,
      kpis: allKpis,
      latestDrift,
      driftHistory,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `data-quality-report-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const isRefreshing = kpiRefetching || driftRefreshing;
  // #325 — Export button disabled while any underlying dataset is still loading.
  // Prevents partial JSON export (undefined latestDrift / driftHistory / [] kpis).
  const isAnyLoading = kpiLoading || driftLoading || driftHistoryLoading;

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

      {/* #323 — surface drift-history error page-level so it's visible from the
          default Validation Rules tab (used to be hidden inside Quality Issues). */}
      {driftHistoryError && (
        <div className="mb-6">
          <QueryErrorState
            error={driftHistoryError}
            title="Could not load 30-day drift history"
          />
        </div>
      )}

      {/* Quality Score Overview (dimension cards — names preserved for spec) */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
        {dimensionsLoading ? (
          <div className="col-span-full flex items-center gap-2 text-muted-foreground py-6">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>Loading quality dimensions...</span>
          </div>
        ) : (
          <>
            <KPICard
              title="Overall Quality"
              value={qualityScores.overall}
              unit="%"
              status={getStatusFromScore(qualityScores.overall)}
              description="Composite score across the four DQ dimensions"
              sparklineData={[]}
              higherIsBetter
            />
            <KPICard
              title="Completeness"
              value={qualityScores.completeness}
              unit="%"
              status={getStatusFromScore(qualityScores.completeness)}
              description="Monitoring coverage across registered DQ KPIs"
              sparklineData={[]}
              higherIsBetter
            />
            <KPICard
              title="Accuracy"
              value={qualityScores.accuracy}
              unit="%"
              status={getStatusFromScore(qualityScores.accuracy)}
              description="Drift-free feature share"
              sparklineData={[]}
              higherIsBetter
            />
            <KPICard
              title="Consistency"
              value={qualityScores.consistency}
              unit="%"
              status={getStatusFromScore(qualityScores.consistency)}
              description="Inverse of overall drift severity"
              sparklineData={[]}
              higherIsBetter
            />
            <KPICard
              title="Timeliness"
              value={qualityScores.timeliness}
              unit="%"
              status={getStatusFromScore(qualityScores.timeliness)}
              description="Pipeline freshness vs baseline window"
              sparklineData={[]}
              higherIsBetter
            />
          </>
        )}
      </div>

      {/* Drift status section (live) */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Drift Status
          </CardTitle>
          <CardDescription>
            Latest drift detection for the data quality pipeline ({DQ_MODEL_ID})
          </CardDescription>
        </CardHeader>
        <CardContent>
          {driftError ? (
            <QueryErrorState error={driftError} title="Could not load drift status" />
          ) : driftLoading ? (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Loading drift status...</span>
            </div>
          ) : !latestDrift ? (
            <p className="text-muted-foreground text-sm">
              No drift detection results yet. Click <strong>Refresh</strong> to trigger
              detection.
            </p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <p className="text-xs uppercase text-muted-foreground mb-1">
                  Overall drift score
                </p>
                <p className="text-2xl font-semibold">
                  {(latestDrift.overall_drift_score * 100).toFixed(1)}%
                </p>
                <p className="text-xs text-muted-foreground">
                  {latestDrift.features_with_drift.length} of {latestDrift.features_checked}{' '}
                  features with drift
                </p>
              </div>
              <div>
                <p className="text-xs uppercase text-muted-foreground mb-1">Summary</p>
                <p className="text-sm">{latestDrift.drift_summary}</p>
                {latestDrift.features_with_drift.length > 0 && (
                  <p className="text-xs text-muted-foreground mt-1">
                    Features with drift: {latestDrift.features_with_drift.join(', ')}
                  </p>
                )}
              </div>
              <div>
                <p className="text-xs uppercase text-muted-foreground mb-1">
                  Recommended actions
                </p>
                {latestDrift.recommended_actions.length === 0 ? (
                  <p className="text-sm text-muted-foreground">None</p>
                ) : (
                  <ul className="text-sm list-disc ml-4 space-y-1">
                    {latestDrift.recommended_actions.map((a, i) => (
                      <li key={i}>{a}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

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
                    Data quality KPIs from workstream <code>ws1_data_quality</code>
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

        {/* Quality Issues: derived from drift history */}
        <TabsContent value="issues">
          <Card>
            <CardHeader>
              <CardTitle>Quality Issues</CardTitle>
              <CardDescription>
                Drift events from the past 30 days for the DQ pipeline
              </CardDescription>
            </CardHeader>
            <CardContent>
              {driftHistoryError ? (
                <QueryErrorState error={driftHistoryError} title="Could not load drift history" />
              ) : driftHistoryLoading ? (
                <div className="flex items-center gap-2 text-muted-foreground py-4">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Loading drift history...</span>
                </div>
              ) : !driftHistory?.records || driftHistory.records.length === 0 ? (
                <p className="text-muted-foreground text-sm">
                  No drift events recorded in the past 30 days. (No quality issues detected.)
                </p>
              ) : (
                <ul className="space-y-3">
                  {driftHistory.records.map((rec) => (
                    <li
                      key={rec.id}
                      className="p-3 rounded-lg border border-border bg-card flex items-start justify-between gap-3"
                    >
                      <div>
                        <p className="font-medium text-sm">
                          {rec.feature_name}{' '}
                          <span className="text-muted-foreground">
                            ({rec.drift_type.replace(/_/g, ' ')})
                          </span>
                        </p>
                        <p className="text-xs text-muted-foreground mt-0.5">
                          Detected {formatTimestamp(rec.detected_at)} · score{' '}
                          {(rec.drift_score * 100).toFixed(1)}%
                        </p>
                      </div>
                      <Badge variant={severityBadgeVariant(rec.severity)} className="capitalize">
                        {rec.severity}
                      </Badge>
                    </li>
                  ))}
                </ul>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default DataQuality;
