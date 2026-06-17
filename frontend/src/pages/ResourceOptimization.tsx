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

import { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { KPICard } from '@/components/visualizations';
import { EmptyState } from '@/components/ui/EmptyState';
import { WarningBanner } from '@/components/ui/WarningBanner';
import {
  useResourceHealth,
  useRunOptimizationAndWait,
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

// =============================================================================
// ALLOCATION COMPARISON CHART
// =============================================================================

interface AllocationChartProps {
  allocations: AllocationResult[];
}

function AllocationComparisonChart({ allocations }: AllocationChartProps) {
  const chartData = allocations.map((a) => ({
    name: a.entity_id.replace('territory_', '').replace(/_/g, ' '),
    current: a.current_allocation / 1000,
    optimized: a.optimized_allocation / 1000,
    change: a.change_percentage,
  }));

  return (
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
  );
}

// =============================================================================
// SCENARIO COMPARISON CHART
// =============================================================================

interface ScenarioChartProps {
  scenarios: ScenarioResult[];
}

function ScenarioComparisonChart({ scenarios }: ScenarioChartProps) {
  const chartData = scenarios.map((s) => ({
    name: s.scenario_name,
    allocation: s.total_allocation / 1000,
    outcome: s.projected_outcome / 1000,
    roi: s.roi,
    hasViolations: s.constraint_violations.length > 0,
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="allocation"
          name="Allocation"
          label={{ value: 'Total Allocation ($K)', position: 'bottom' }}
        />
        <YAxis
          dataKey="outcome"
          name="Outcome"
          label={{ value: 'Projected Outcome ($K)', angle: -90, position: 'insideLeft' }}
        />
        <Tooltip
          formatter={(value, name) => [
            name === 'ROI' ? `${(value as number)?.toFixed(2) ?? 0}x` : `$${(value as number)?.toFixed(0) ?? 0}K`,
            name as string,
          ]}
          content={({ payload }) => {
            if (!payload || payload.length === 0) return null;
            const data = payload[0].payload;
            return (
              <div className="bg-background border rounded-lg p-3 shadow-lg">
                <p className="font-medium">{data.name}</p>
                <p className="text-sm text-muted-foreground">
                  Allocation: ${data.allocation}K
                </p>
                <p className="text-sm text-muted-foreground">
                  Outcome: ${data.outcome}K
                </p>
                <p className="text-sm text-muted-foreground">ROI: {data.roi.toFixed(2)}x</p>
                {data.hasViolations && (
                  <p className="text-sm text-destructive">Has constraint violations</p>
                )}
              </div>
            );
          }}
        />
        <Scatter
          data={chartData}
          fill={COLORS.primary}
          shape={(props: unknown) => {
            const { cx, cy, payload } = props as { cx: number; cy: number; payload: { hasViolations: boolean } };
            return (
              <circle
                cx={cx}
                cy={cy}
                r={8}
                fill={payload.hasViolations ? COLORS.warning : COLORS.success}
                stroke={COLORS.primary}
                strokeWidth={2}
              />
            );
          }}
        />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

// =============================================================================
// SENSITIVITY ANALYSIS CHART
// =============================================================================

interface SensitivityChartProps {
  sensitivity: Record<string, number>;
}

function SensitivityAnalysisChart({ sensitivity }: SensitivityChartProps) {
  const chartData = Object.entries(sensitivity).map(([key, value]) => ({
    name: key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
    sensitivity: value * 100,
  }));

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" label={{ value: 'Sensitivity (%)', position: 'bottom' }} />
        <YAxis type="category" dataKey="name" />
        <Tooltip formatter={(value) => [`${(value as number)?.toFixed(1) ?? 0}%`, 'Sensitivity']} />
        <Bar dataKey="sensitivity" fill={COLORS.secondary} />
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
          label={({ name, value }) => `${name}: ${value}%`}
          outerRadius={80}
          dataKey="value"
        >
          {chartData.map((_, index) => (
            <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
          ))}
        </Pie>
        <Tooltip formatter={(value) => [`${value ?? 0}%`, 'Impact Share']} />
      </PieChart>
    </ResponsiveContainer>
  );
}

// =============================================================================
// ALLOCATION TREND CHART
// =============================================================================
//
// The backend optimization result currently has no time-series / historical
// allocation-trend field (OptimizationResponse exposes only the before/after
// snapshot in `optimal_allocations`). The previous implementation FABRICATED a
// "Q1/Q2/Q3/Q4" trend client-side from hardcoded 0.9/0.95 multipliers of the
// current allocation — presenting invented quarters as "Historical and
// projected allocation by territory". That is fabricated data and has been
// removed (see PR: anti-fabrication discipline).
//
// Until the backend emits a real allocation trend, render an honest empty
// state rather than inventing one. The `allocations` prop is retained so the
// real series can be wired in here the moment the backend provides it.

interface AllocationTrendProps {
  allocations: AllocationResult[];
}

function AllocationTrendChart(_props: AllocationTrendProps) {
  return (
    <div className="flex h-[300px] flex-col items-center justify-center text-center">
      <p className="text-sm font-medium text-muted-foreground">No allocation trend data</p>
      <p className="mt-1 max-w-md text-xs text-muted-foreground">
        Historical allocation trend is not available from the optimizer. Only the current
        vs optimized snapshot above is real; a time-series trend will appear here once the
        backend provides one.
      </p>
    </div>
  );
}

// =============================================================================
// MAIN PAGE COMPONENT
// =============================================================================

export default function ResourceOptimization() {
  const [activeTab, setActiveTab] = useState('allocations');
  const [selectedResourceType, setSelectedResourceType] = useState<string>('budget');
  const [selectedObjective, setSelectedObjective] = useState<string>('maximize_roi');

  // API hooks
  const { data: healthData, isLoading: healthLoading } = useResourceHealth();
  const { data: scenariosData } = useScenarios({ limit: 10 });
  // Run-and-WAIT: the optimize endpoint is async (returns a PENDING id and
  // computes in a background task); this hook polls GET /resources/{id} until
  // the result is COMPLETED. The fire-and-forget useRunOptimization never
  // retrieved the result, so "Run Optimization" appeared to do nothing.
  const runOptimization = useRunOptimizationAndWait();
  const runError = runOptimization.error;

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
        run_scenarios: true,
        scenario_count: 3,
      },
      pollIntervalMs: 3000,
      maxWaitMs: 120000,
    });
  };

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
            Select resource type and optimization objective
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
              description="optimal reallocation gain vs current"
            />
            <KPICard
              title="Projected Outcome"
              value={`$${((optimizationResult.projected_total_outcome || 0) / 1000).toFixed(0)}K`}
              description="total projected value"
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
                <CardTitle>Impact by Segment</CardTitle>
                <CardDescription>Expected impact distribution</CardDescription>
              </CardHeader>
              <CardContent>
                {optimizationResult.impact_by_segment && (
                  <ImpactBySegmentChart impactBySegment={optimizationResult.impact_by_segment} />
                )}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Allocation Trend</CardTitle>
              <CardDescription>Allocation trend over time (when available from the optimizer)</CardDescription>
            </CardHeader>
            <CardContent>
              <AllocationTrendChart allocations={optimizationResult.optimal_allocations} />
            </CardContent>
          </Card>

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
                      <th className="text-right p-2">Expected Impact</th>
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
                        </td>
                        <td className="p-2 text-right">
                          <Badge
                            variant={
                              alloc.expected_impact >= 1.2
                                ? 'default'
                                : alloc.expected_impact >= 1.0
                                ? 'secondary'
                                : 'outline'
                            }
                          >
                            {alloc.expected_impact.toFixed(2)}x
                          </Badge>
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
                  Compare allocation vs outcome across scenarios
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
                          {scenario.roi.toFixed(2)}x ROI
                        </Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground">
                        <div>
                          Allocation: ${(scenario.total_allocation / 1000).toFixed(0)}K
                        </div>
                        <div>
                          Outcome: ${(scenario.projected_outcome / 1000).toFixed(0)}K
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
              <CardTitle>Sensitivity Analysis</CardTitle>
              <CardDescription>
                How sensitive is the objective to constraint changes
              </CardDescription>
            </CardHeader>
            <CardContent>
              {optimizationResult.sensitivity_analysis && (
                <SensitivityAnalysisChart
                  sensitivity={optimizationResult.sensitivity_analysis}
                />
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Constraint Impact</CardTitle>
              <CardDescription>
                Understanding the effect of relaxing constraints
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {optimizationResult.sensitivity_analysis &&
                  Object.entries(optimizationResult.sensitivity_analysis).map(
                    ([key, value]) => (
                      <div key={key} className="flex items-center justify-between p-3 border rounded-lg">
                        <div>
                          <p className="font-medium">
                            {key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                          </p>
                          <p className="text-sm text-muted-foreground">
                            A 10% relaxation would improve objective by{' '}
                            {(value * 10).toFixed(1)}%
                          </p>
                        </div>
                        <Badge variant={value > 0.1 ? 'default' : 'outline'}>
                          {value > 0.1 ? 'High Impact' : 'Low Impact'}
                        </Badge>
                      </div>
                    )
                  )}
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

          {optimizationResult.warnings.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-warning">Warnings</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="list-disc list-inside space-y-1">
                  {optimizationResult.warnings.map((warning, idx) => (
                    <li key={idx} className="text-warning">
                      {warning}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}

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
