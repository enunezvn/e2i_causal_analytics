/**
 * Resource Optimization Page
 * ==========================
 *
 * Mathematical optimization dashboard for resource allocation across
 * entities (territories, HCPs, regions). Uses Tier 4 Resource Optimizer
 * agent with scipy optimization backend.
 *
 * Features:
 * - Budget/rep time/samples/calls optimization
 * - Multi-objective optimization (maximize outcome, ROI, minimize cost)
 * - Scenario comparison and sensitivity analysis
 * - Constraint management (budget, capacity, coverage)
 * - Allocation visualization with before/after comparison
 *
 * @module pages/ResourceOptimization
 */

import { useMemo, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LabelList,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { KPICard } from '@/components/visualizations';
import { EmptyState } from '@/components/ui/EmptyState';
import { WarningBanner } from '@/components/ui/WarningBanner';
import { StrategicInsightCard } from '@/components/insights';
import { usePageChatContext } from '@/providers/E2ICopilotProvider';
import {
  useResourceHealth,
  useRunOptimizationAndWait,
  useResourceOptimizationInsight,
  useScenarios,
} from '@/hooks/api';
import type {
  AllocationResult,
  ScenarioResult,
} from '@/types/resources';

// =============================================================================
// CHART COLORS
// =============================================================================

const COLORS = {
  primary: '#3b82f6',
  secondary: '#8b5cf6',
  success: '#10b981',
  warning: '#f59e0b',
  danger: '#ef4444',
  muted: '#6b7280',
};

const PIE_COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444'];

// The platform's three brands (synthetic DGP). "All Brands" seeds from
// territory_metrics; a specific brand seeds from brand-filtered treatment
// events — both clearly-labelled synthetic sources.
const BRAND_OPTIONS = ['All Brands', 'Remibrutinib', 'Fabhalta', 'Kisqali'];

// With every territory in the optimization universe (~40), the comparison
// bar chart caps at the biggest movers to stay readable; the table below
// lists everything.
const CHART_TOP_MOVERS = 12;

// =============================================================================
// ALLOCATION COMPARISON CHART
// =============================================================================

interface AllocationChartProps {
  allocations: AllocationResult[];
}

function AllocationComparisonChart({ allocations }: AllocationChartProps) {
  // Backend sorts by |change| descending — take the biggest movers.
  const shown = allocations.slice(0, CHART_TOP_MOVERS);
  const chartData = shown.map((a) => ({
    name: a.entity_id.replace('territory_', '').replace(/_/g, ' '),
    current: a.current_allocation / 1000,
    optimized: a.optimized_allocation / 1000,
    // null = new allocation from zero current (no percentage exists); the
    // current/optimized bars still carry the move.
    change: a.change_percentage ?? 0,
  }));
  const hiddenCount = allocations.length - shown.length;

  return (
    <>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis label={{ value: 'Allocation ($K)', angle: -90, position: 'insideLeft' }} />
          <Tooltip
            formatter={(value, name) => [
              `$${(value as number)?.toFixed(0) ?? 0}K`,
              name === 'current' ? 'Current' : 'Optimized',
            ]}
          />
          <Legend />
          <Bar dataKey="current" fill={COLORS.muted} name="Current" />
          <Bar dataKey="optimized" fill={COLORS.primary} name="Optimized" />
        </BarChart>
      </ResponsiveContainer>
      {hiddenCount > 0 && (
        <p className="mt-1 text-xs text-muted-foreground">
          Top {shown.length} territories by allocation change; {hiddenCount} more in the
          table below.
        </p>
      )}
    </>
  );
}

// =============================================================================
// SCENARIO COMPARISON CHART
// =============================================================================

interface ScenarioChartProps {
  scenarios: ScenarioResult[];
}

const fmtWhole = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 });

function ScenarioComparisonChart({ scenarios }: ScenarioChartProps) {
  // Every generated scenario reallocates the SAME fixed budget (the backend
  // seeds budget = sum of current allocations), so total allocation is
  // near-identical across scenarios. The previous allocation-vs-outcome
  // scatter therefore collapsed onto one vertical line whose ticks rendered
  // float noise (3358.5000000000005). With spend held constant the only
  // differentiating dimension is OUTCOME — bars, with the shared budget
  // stated once below and per-scenario detail in the tooltip.
  const chartData = scenarios.map((s) => ({
    // Compact tick label — "Current Allocation (Baseline)" would clip in the
    // half-width card; the tooltip keeps the full name.
    label: (s.scenario_name.replace(/\s*allocation\s*/i, ' ').trim() || s.scenario_name)
      + (s.constraint_violations.length > 0 ? ' ⚠' : ''),
    name: s.scenario_name,
    allocation: s.total_allocation / 1000,
    outcome: s.projected_outcome,
    returnPer1k: s.roi * 1000,
    hasViolations: s.constraint_violations.length > 0,
  }));

  if (chartData.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No scenarios yet — run an optimization with scenarios enabled.
      </p>
    );
  }

  const allocations = chartData.map((d) => d.allocation);
  const maxAllocation = Math.max(...allocations);
  const sharedBudget =
    maxAllocation > 0 && (maxAllocation - Math.min(...allocations)) / maxAllocation < 0.005;

  // Emphasis: the optimized scenario is the point; the rest are context. When
  // no scenario is named "optimized" (custom feeds), a single hue for all.
  const isOptimized = (name: string) => /optimi[sz]ed/i.test(name);
  const hasEmphasis = chartData.some((d) => isOptimized(d.name));

  return (
    <>
      {/* Height tracks content (bars + x-axis band + axis label) so the axis
          legend is never clipped by a fixed container. */}
      <ResponsiveContainer width="100%" height={chartData.length * 56 + 64}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 8, right: 56, left: 8, bottom: 24 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            tickFormatter={fmtWhole}
            label={{ value: 'Projected outcome (TRx-equivalents)', position: 'bottom' }}
          />
          <YAxis type="category" dataKey="label" width={130} interval={0} tick={{ fontSize: 12 }} />
          <Tooltip
            cursor={{ fill: 'rgba(107, 114, 128, 0.08)' }}
            content={({ payload }) => {
              if (!payload || payload.length === 0) return null;
              const data = payload[0].payload;
              return (
                <div className="bg-background border rounded-lg p-3 shadow-lg">
                  <p className="font-medium">{data.name}</p>
                  <p className="text-sm text-muted-foreground">
                    Allocation: ${fmtWhole(data.allocation)}K
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Outcome: {fmtWhole(data.outcome)} TRx-equivalents
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Return: {data.returnPer1k.toFixed(2)} TRx-eq per $1K
                  </p>
                  {data.hasViolations && (
                    <p className="text-sm text-destructive">Has constraint violations</p>
                  )}
                </div>
              );
            }}
          />
          <Bar dataKey="outcome" barSize={22} radius={[0, 4, 4, 0]}>
            <LabelList
              dataKey="outcome"
              position="right"
              formatter={(value) => fmtWhole(Number(value))}
              fontSize={12}
              fill={COLORS.muted}
            />
            {chartData.map((d) => (
              <Cell
                key={d.name}
                fill={!hasEmphasis || isOptimized(d.name) ? COLORS.primary : COLORS.muted}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      {sharedBudget && (
        <p className="mt-1 text-xs text-muted-foreground">
          All scenarios deploy the same ${fmtWhole(maxAllocation)}K total budget, so bars
          compare outcome at equal spend; hover for return per $1K.
        </p>
      )}
    </>
  );
}

// =============================================================================
// SENSITIVITY ANALYSIS CHART
// =============================================================================

interface SensitivityChartProps {
  sensitivity: Record<string, number>;
  sensitivityCurrent?: Record<string, number>;
}

function SensitivityAnalysisChart({ sensitivity, sensitivityCurrent }: SensitivityChartProps) {
  // Backend values are marginal returns per +$1 (shown per +$1K for readable
  // magnitudes). At an ROI optimum the OPTIMIZED marginals equalize to the hurdle
  // across territories, so on their own they render as a wall of identical bars
  // that reads as broken. Pairing them with the CURRENT marginals (dispersed by
  // productivity) shows the before->after equalization: over-productive
  // territories start high and are funded until their marginal falls to the
  // hurdle; under-productive ones start low and are cut until theirs rises to it.
  const hasCurrent = !!sensitivityCurrent && Object.keys(sensitivityCurrent).length > 0;

  const chartData = Object.entries(sensitivity)
    .map(([key, value]) => ({
      name: key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
      optimized: value * 1000,
      current: hasCurrent ? (sensitivityCurrent?.[key] ?? 0) * 1000 : undefined,
    }))
    // Sort by CURRENT marginal descending so the chart reads as a funnel that
    // converges on the equalized hurdle. Fall back to optimized when the current
    // series is unavailable (e.g. an older cached response).
    .sort((a, b) =>
      hasCurrent ? (b.current ?? 0) - (a.current ?? 0) : b.optimized - a.optimized
    );

  // The equalized hurdle ~ the (near-constant) optimized marginals; use their
  // median as the reference line the current marginals converge onto.
  const optimizedVals = chartData.map((d) => d.optimized).sort((a, b) => a - b);
  const hurdle = optimizedVals[Math.floor(optimizedVals.length / 2)] ?? 0;

  return (
    <ResponsiveContainer
      width="100%"
      height={Math.max(220, chartData.length * (hasCurrent ? 30 : 24))}
    >
      <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          type="number"
          label={{ value: 'Marginal return (outcome units per +$1K)', position: 'bottom' }}
        />
        <YAxis type="category" dataKey="name" width={110} interval={0} tick={{ fontSize: 11 }} />
        <Tooltip
          formatter={(value, name) => [
            `${(value as number)?.toFixed(2) ?? 0} units per +$1K`,
            name,
          ]}
        />
        {hasCurrent && <Legend verticalAlign="top" />}
        {hasCurrent && (
          <ReferenceLine
            x={hurdle}
            stroke={COLORS.warning}
            strokeDasharray="4 4"
            label={{ value: `hurdle ${hurdle.toFixed(2)}`, position: 'insideTopRight', fontSize: 11 }}
          />
        )}
        {hasCurrent && <Bar dataKey="current" name="At current allocation" fill={COLORS.primary} />}
        <Bar dataKey="optimized" name="At optimized allocation" fill={COLORS.secondary} />
      </BarChart>
    </ResponsiveContainer>
  );
}

// =============================================================================
// IMPACT BY SEGMENT CHART
// =============================================================================

interface ImpactChartProps {
  impactBySegment: Record<string, number>;
}

function ImpactBySegmentChart({ impactBySegment }: ImpactChartProps) {
  // Backend values are % shares of the projected outcome by region (sum ~100).
  const chartData = Object.entries(impactBySegment).map(([key, value]) => ({
    name: key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
    value,
  }));

  return (
    <ResponsiveContainer width="100%" height={200}>
      <PieChart>
        <Pie
          data={chartData}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={({ name, value }) => `${name}: ${(value as number).toFixed(1)}%`}
          outerRadius={80}
          dataKey="value"
        >
          {chartData.map((_, index) => (
            <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
          ))}
        </Pie>
        <Tooltip
          formatter={(value) => [`${((value as number) ?? 0).toFixed(1)}%`, 'Share of outcome']}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}

// NOTE: an "Allocation Trend" card used to live on this page. The backend has
// never produced a time-series (an earlier version FABRICATED quarters from
// hardcoded multipliers, then it became a permanently-empty placeholder). A
// card whose only state is "no data" is UI debt — removed until a real
// allocation history exists server-side.

// =============================================================================
// MAIN PAGE COMPONENT
// =============================================================================

export default function ResourceOptimization() {
  const [activeTab, setActiveTab] = useState('allocations');
  const [selectedResourceType, setSelectedResourceType] = useState<string>('budget');
  const [selectedObjective, setSelectedObjective] = useState<string>('maximize_roi');
  const [selectedBrand, setSelectedBrand] = useState<string>('All Brands');

  // API hooks
  const { data: healthData, isLoading: healthLoading } = useResourceHealth();
  const { data: scenariosData } = useScenarios({ limit: 10 });
  // Run-and-WAIT: the optimize endpoint is async (returns a PENDING id and
  // computes in a background task); this hook polls GET /resources/{id} until
  // the result is COMPLETED. The fire-and-forget useRunOptimization never
  // retrieved the result, so "Run Optimization" appeared to do nothing.
  const runOptimization = useRunOptimizationAndWait();
  const runError = runOptimization.error;

  // Strategic Interpretation: DSPy/LLM business read of the solver result
  // (allocation moves, lift, provenance) with a deterministic factual
  // fallback when the LM is unavailable — same pattern as the other pages.
  const resInsight = useResourceOptimizationInsight();

  // Live optimization output (undefined until the user runs one). The
  // Scenarios tab additionally consumes the standalone scenarios feed.
  const optimizationResult = runOptimization.data;
  const scenarios = optimizationResult?.scenarios ?? scenariosData?.scenarios ?? [];

  // Provenance: the backend seeds a clearly-labelled SYNTHETIC allocation
  // problem when no targets are supplied (no real budget substrate exists), and
  // tags the response with a "SYNTHETIC DATA:" warning. Surface it honestly.
  const resultWarnings = optimizationResult?.warnings ?? [];

  // Health status
  const isHealthy = healthData?.agent_available && healthData?.scipy_available;
  // Storage degraded -> the optimizations store fell back to a process-local
  // in-memory dict, so cross-worker reads can 404 (the page polls GET by id).
  // Surface this honestly rather than showing an unqualified "Solver Ready".
  const storageDegraded = healthData?.storage_mode === 'degraded';

  // Handle optimization run
  const handleRunOptimization = () => {
    runOptimization.mutate({
      request: {
        query: `Optimize ${selectedResourceType} allocation to ${selectedObjective.replace('_', ' ')}`,
        resource_type: selectedResourceType as never,
        // Send NO targets: the backend seeds a clearly-labelled synthetic
        // allocation problem from real-shaped territory data. (Re-sending the
        // previous result's allocations here was a chicken-and-egg bug — empty
        // on the first run, and chain-optimizing an already-optimized state
        // thereafter.)
        allocation_targets: [],
        objective: selectedObjective as never,
        brand: selectedBrand === 'All Brands' ? null : selectedBrand,
        run_scenarios: true,
        scenario_count: 3,
      },
      pollIntervalMs: 3000,
      maxWaitMs: 120000,
    });
  };

  // Grounding for the strategic interpretation: top movers from the live result.
  // Sorted/filtered by dollar change (not percentage) so a new allocation from
  // zero current — whose change_percentage is null — still counts as a mover.
  const sortedByChange = [...(optimizationResult?.optimal_allocations ?? [])].sort(
    (a, b) => b.change - a.change
  );
  const topIncreases = sortedByChange
    .filter((a) => a.change > 0)
    .slice(0, 5)
    .map((a) => ({
      entity_id: a.entity_id,
      change_percentage: a.change_percentage,
      change: a.change,
    }));
  const topDecreases = sortedByChange
    .filter((a) => a.change < 0)
    .slice(-5)
    .reverse()
    .map((a) => ({
      entity_id: a.entity_id,
      change_percentage: a.change_percentage,
      change: a.change,
    }));
  const totalBudget = (optimizationResult?.optimal_allocations ?? []).reduce(
    (sum, a) => sum + a.current_allocation,
    0
  );
  // Actual recommended spend — maximize_roi can intentionally deploy less than
  // the budget (marginal return below hurdle); the insight must know both.
  const deployedSpend = (optimizationResult?.optimal_allocations ?? []).reduce(
    (sum, a) => sum + a.optimized_allocation,
    0
  );

  // Publish a compact on-screen data summary so the chat pane can generate
  // opener pills grounded in what this page is showing (usePageChatContext →
  // POST /chat/suggestions page_context).
  const pageChatSummary = useMemo(() => {
    const lines: string[] = [
      `Resource Optimization page. Brand: ${selectedBrand}; resource type: ${selectedResourceType}; objective: ${selectedObjective}.`,
    ];
    if (optimizationResult) {
      lines.push(
        `Optimization result on screen: ${optimizationResult.optimal_allocations.length} allocations, projected ROI ${optimizationResult.projected_roi}, projected total outcome ${optimizationResult.projected_total_outcome}.`
      );
      if (topIncreases[0]) {
        lines.push(`Largest increase: ${topIncreases[0].entity_id} (+${topIncreases[0].change}).`);
      }
      if (topDecreases[0]) {
        lines.push(`Largest decrease: ${topDecreases[0].entity_id} (${topDecreases[0].change}).`);
      }
    } else {
      lines.push('No optimization run yet on this visit.');
    }
    return lines.join('\n');
    // topIncreases/topDecreases are derived from optimizationResult inline
    // (new arrays each render) — depend on the source result instead.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedBrand, selectedResourceType, selectedObjective, optimizationResult]);
  usePageChatContext(pageChatSummary);

  return (
    <div className="container mx-auto py-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Resource Optimization</h1>
          <p className="text-muted-foreground">
            Mathematical optimization for budget, rep time, and resource allocation
          </p>
        </div>
        <div className="flex items-center gap-4">
          <Badge variant={isHealthy ? 'default' : 'destructive'}>
            {healthLoading ? 'Checking...' : isHealthy ? 'Solver Ready' : 'Solver Unavailable'}
          </Badge>
          {!healthLoading && storageDegraded && (
            <Badge variant="destructive" title="Optimization results store fell back to in-memory; cross-worker reads can fail until Redis is restored.">
              Storage Degraded
            </Badge>
          )}
          {healthData?.optimizations_24h !== undefined && (
            <Badge variant="outline">{healthData.optimizations_24h} optimizations today</Badge>
          )}
          <Badge
            variant="outline"
            className="border-amber-400 text-amber-700 dark:text-amber-300"
            title="No real per-entity budget source is wired yet. Runs use a clearly-labelled synthetic allocation problem seeded from territory data — the optimization math is real, the dollar values are illustrative."
          >
            Illustrative · synthetic data
          </Badge>
        </div>
      </div>

      {/* Configuration Panel */}
      <Card>
        <CardHeader>
          <CardTitle>Optimization Configuration</CardTitle>
          <CardDescription>
            Optimizes across every active territory (~40, all regions). Activity source:
            territory metrics for All Brands, brand-filtered treatment events for a
            specific brand.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Brand</label>
              <select
                className="w-full p-2 border rounded-md"
                value={selectedBrand}
                onChange={(e) => setSelectedBrand(e.target.value)}
              >
                {BRAND_OPTIONS.map((b) => (
                  <option key={b} value={b}>
                    {b}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Resource Type</label>
              <select
                className="w-full p-2 border rounded-md"
                value={selectedResourceType}
                onChange={(e) => setSelectedResourceType(e.target.value)}
              >
                <option value="budget">Budget</option>
                <option value="rep_time">Rep Time</option>
                <option value="samples">Samples</option>
                <option value="calls">Calls</option>
              </select>
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Objective</label>
              <select
                className="w-full p-2 border rounded-md"
                value={selectedObjective}
                onChange={(e) => setSelectedObjective(e.target.value)}
              >
                <option value="maximize_roi">Maximize ROI</option>
                <option value="maximize_outcome">Maximize Outcome</option>
                <option value="minimize_cost">Minimize Cost</option>
                <option value="balance">Balanced</option>
              </select>
            </div>
            <div className="flex items-end">
              <Button
                onClick={handleRunOptimization}
                disabled={runOptimization.isPending}
                className="w-full"
              >
                {runOptimization.isPending ? 'Optimizing...' : 'Run Optimization'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Strategic Interpretation — agentic read of the optimization result.
          Rendered always (not gated on a completed run) so the header is
          present on mount; onGenerate feeds the REAL optimization output
          (existing optimization_summary + recommendations) into the shared card. */}
      <StrategicInsightCard
        isLoading={resInsight.isPending}
        error={resInsight.error?.message ?? null}
        insight={resInsight.data?.insight}
        keyTakeaways={resInsight.data?.key_takeaways}
        grounding={resInsight.data?.grounding}
        isFallback={resInsight.data?.is_fallback}
        provenance={resInsight.data?.provenance}
        generatedAt={resInsight.data?.generated_at}
        onGenerate={() =>
          resInsight.mutate({
            optimization_summary: optimizationResult?.optimization_summary ?? '',
            recommendations: optimizationResult?.recommendations ?? [],
            projected_lift_pct:
              optimizationResult?.projected_roi != null
                ? optimizationResult.projected_roi * 100
                : null,
            solver_status: optimizationResult?.solver_status ?? null,
            objective: optimizationResult?.objective ?? selectedObjective,
            brand: selectedBrand === 'All Brands' ? null : selectedBrand,
            resource_type: optimizationResult?.resource_type ?? selectedResourceType,
            entity_count: optimizationResult?.optimal_allocations?.length ?? null,
            total_budget: totalBudget > 0 ? totalBudget : null,
            // null only when there is no result to read spend from — a genuine
            // $0 spend over a real result must reach the insight as 0, not null
            total_spend: optimizationResult?.optimal_allocations?.length
              ? deployedSpend
              : null,
            top_increases: topIncreases,
            top_decreases: topDecreases,
            synthetic: resultWarnings.some((w) => w.startsWith('SYNTHETIC DATA:')),
          })
        }
      />

      {/* Running indicator while the async optimization computes + is polled */}
      {runOptimization.isPending && (
        <Card>
          <CardContent className="flex items-center gap-3 py-4">
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            <p className="text-sm text-muted-foreground">
              Running optimization for {selectedResourceType} (
              {selectedObjective.replace('_', ' ')})… solving and analyzing scenarios.
            </p>
          </CardContent>
        </Card>
      )}

      {/* Honest failure surface — the mutation previously swallowed errors,
          so a failed/empty run looked like "nothing happened". */}
      {runError && (
        <WarningBanner
          title="Optimization failed"
          messages={[runError instanceof Error ? runError.message : String(runError)]}
        />
      )}

      {!optimizationResult ? (
        <EmptyState
          title="Run an optimization to see results"
          description={
            'Choose a resource type and objective above, then click “Run Optimization”. ' +
            'Results — allocations, scenarios, sensitivity, and recommendations — will appear here.'
          }
        />
      ) : (
        <>
          {/* Provenance: synthetic-data + any solver caveats, surfaced up front */}
          {resultWarnings.length > 0 && (
            <WarningBanner
              title="Illustrative result — synthetic data"
              messages={resultWarnings}
            />
          )}

          {/* KPI Summary */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <KPICard
              title="Projected Outcome Lift"
              value={
                optimizationResult.projected_roi != null
                  ? `${optimizationResult.projected_roi >= 0 ? '+' : ''}${(
                      optimizationResult.projected_roi * 100
                    ).toFixed(1)}%`
                  : '—'
              }
              description="outcome gain vs current allocation"
            />
            <KPICard
              title="Projected Outcome"
              value={(optimizationResult.projected_total_outcome || 0).toLocaleString(
                undefined,
                { maximumFractionDigits: 0 }
              )}
              description="outcome units (TRx-equivalents, illustrative)"
            />
            <KPICard
              title="Solve Time"
              value={`${optimizationResult.solve_time_ms}ms`}
              description={optimizationResult.solver_status || 'optimal'}
            />
            <KPICard
              title="Allocations"
              value={optimizationResult.optimal_allocations.length.toString()}
              description="entities optimized"
            />
          </div>

          {/* Main Content Tabs */}
          <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="allocations">Allocations</TabsTrigger>
          <TabsTrigger value="scenarios">Scenarios</TabsTrigger>
          <TabsTrigger value="sensitivity">Sensitivity</TabsTrigger>
          <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
        </TabsList>

        {/* Allocations Tab */}
        <TabsContent value="allocations" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Allocation Comparison</CardTitle>
                <CardDescription>Current vs optimized allocation by territory</CardDescription>
              </CardHeader>
              <CardContent>
                <AllocationComparisonChart allocations={optimizationResult.optimal_allocations} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Impact by Region</CardTitle>
                <CardDescription>Share of projected outcome by region</CardDescription>
              </CardHeader>
              <CardContent>
                {optimizationResult.impact_by_segment && (
                  <ImpactBySegmentChart impactBySegment={optimizationResult.impact_by_segment} />
                )}
              </CardContent>
            </Card>
          </div>

          {/* Allocation Details Table */}
          <Card>
            <CardHeader>
              <CardTitle>Allocation Details</CardTitle>
              <CardDescription>Detailed breakdown of optimized allocations</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Entity</th>
                      <th className="text-right p-2">Current</th>
                      <th className="text-right p-2">Optimized</th>
                      <th className="text-right p-2">Change</th>
                      <th className="text-right p-2">Projected Outcome (TRx-eq)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {optimizationResult.optimal_allocations.map((alloc) => (
                      <tr key={alloc.entity_id} className="border-b">
                        <td className="p-2 font-medium">
                          {alloc.entity_id.replace('territory_', '').replace(/_/g, ' ')}
                        </td>
                        <td className="p-2 text-right">
                          ${(alloc.current_allocation / 1000).toFixed(0)}K
                        </td>
                        <td className="p-2 text-right">
                          ${(alloc.optimized_allocation / 1000).toFixed(0)}K
                        </td>
                        <td className="p-2 text-right">
                          {alloc.change_percentage == null ? (
                            /* New allocation from zero current — a percentage
                               of zero is undefined; 0% would hide the move. */
                            <span className="text-green-600">New</span>
                          ) : (
                            <span
                              className={
                                alloc.change_percentage >= 0
                                  ? 'text-green-600'
                                  : 'text-red-600'
                              }
                            >
                              {alloc.change_percentage >= 0 ? '+' : ''}
                              {alloc.change_percentage.toFixed(1)}%
                            </span>
                          )}
                        </td>
                        <td className="p-2 text-right">
                          {alloc.expected_impact.toLocaleString(undefined, {
                            maximumFractionDigits: 0,
                          })}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Scenarios Tab */}
        <TabsContent value="scenarios" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Scenario Comparison</CardTitle>
                <CardDescription>
                  Projected outcome (TRx-equivalents) by allocation strategy
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ScenarioComparisonChart scenarios={scenarios} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Scenario Details</CardTitle>
                <CardDescription>Individual scenario analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {scenarios.map((scenario, idx) => (
                    <div
                      key={idx}
                      className="p-4 border rounded-lg space-y-2"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{scenario.scenario_name}</span>
                        <Badge
                          variant={
                            scenario.constraint_violations.length > 0
                              ? 'destructive'
                              : 'default'
                          }
                        >
                          {(scenario.roi * 1000).toFixed(2)} TRx-eq/$1K
                        </Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground">
                        <div>
                          Allocation: ${fmtWhole(scenario.total_allocation / 1000)}K
                        </div>
                        <div>
                          Outcome:{' '}
                          {scenario.projected_outcome.toLocaleString(undefined, {
                            maximumFractionDigits: 0,
                          })}{' '}
                          TRx-equivalents
                        </div>
                      </div>
                      {scenario.constraint_violations.length > 0 && (
                        <div className="text-sm text-destructive">
                          Violations: {scenario.constraint_violations.join(', ')}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Sensitivity Tab */}
        <TabsContent value="sensitivity" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Marginal Returns — Current vs Optimized</CardTitle>
              <CardDescription>
                What one more $1K buys in each territory, before and after optimization.
                Current marginals are dispersed by productivity; the optimizer moves money
                until they equalize at the hurdle rate — over-productive territories are
                funded until their marginal falls to the line, under-productive ones cut
                until theirs rises to it. The equalized line is the signature of an optimal
                allocation, not a rendering bug.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {optimizationResult.sensitivity_analysis && (
                <SensitivityAnalysisChart
                  sensitivity={optimizationResult.sensitivity_analysis}
                  sensitivityCurrent={optimizationResult.sensitivity_analysis_current}
                />
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Per-Territory Detail</CardTitle>
              <CardDescription>
                Marginal return per additional $1K at the current then optimized allocation,
                with the direction the optimizer moved each territory's budget.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {optimizationResult.sensitivity_analysis &&
                  (() => {
                    const current = optimizationResult.sensitivity_analysis_current;
                    const entries = Object.entries(
                      optimizationResult.sensitivity_analysis
                    );
                    return entries.map(([key, optValue]) => {
                      const curValue = current?.[key];
                      // Concave response: funding a territory MORE lowers its
                      // marginal, funding it LESS raises it. So current > optimized
                      // means the optimizer grew this territory, and vice-versa.
                      const rel =
                        curValue !== undefined && optValue
                          ? (curValue - optValue) / optValue
                          : 0;
                      const direction =
                        curValue === undefined
                          ? null
                          : rel > 0.005
                            ? 'up'
                            : rel < -0.005
                              ? 'down'
                              : 'held';
                      return (
                        <div
                          key={key}
                          className="flex items-center justify-between p-3 border rounded-lg"
                        >
                          <div>
                            <p className="font-medium">
                              {key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                            </p>
                            <p className="text-sm text-muted-foreground">
                              {curValue !== undefined
                                ? `+${(curValue * 1000).toFixed(2)} → +${(optValue * 1000).toFixed(2)} outcome units per additional $1K`
                                : `+${(optValue * 1000).toFixed(2)} outcome units per additional $1K`}
                            </p>
                          </div>
                          {direction === 'up' && <Badge variant="default">Funded up</Badge>}
                          {direction === 'down' && <Badge variant="outline">Funded down</Badge>}
                          {direction === 'held' && <Badge variant="outline">Held</Badge>}
                          {direction === null && <Badge variant="outline">At optimum</Badge>}
                        </div>
                      );
                    });
                  })()}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Recommendations Tab */}
        <TabsContent value="recommendations" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Optimization Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-lg">{optimizationResult.optimization_summary}</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Actionable Recommendations</CardTitle>
              <CardDescription>
                AI-generated recommendations based on optimization results
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {optimizationResult.recommendations.map((rec, idx) => (
                  <div
                    key={idx}
                    className="flex items-start gap-3 p-3 bg-muted/50 rounded-lg"
                  >
                    <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-medium">
                      {idx + 1}
                    </div>
                    <p>{rec}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Warnings are surfaced once, up front, in the always-visible
              "Illustrative result — synthetic data" banner above the tabs.
              A second per-tab "Warnings" card here only re-printed the same
              (now backend-deduplicated) provenance line. */}

          {/* Metadata */}
          <Card>
            <CardHeader>
              <CardTitle>Optimization Metadata</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Optimization ID</p>
                  <p className="font-mono">{optimizationResult.optimization_id}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Formulation Time</p>
                  <p>{optimizationResult.formulation_latency_ms}ms</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Solve Time</p>
                  <p>{optimizationResult.optimization_latency_ms}ms</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Total Time</p>
                  <p>{optimizationResult.total_latency_ms}ms</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  );
}
