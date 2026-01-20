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
  LineChart,
  Line,
  ScatterChart,
  Scatter,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { KPICard } from '@/components/visualizations';
import {
  useResourceHealth,
  useRunOptimization,
  useScenarios,
} from '@/hooks/api';
import type {
  AllocationResult,
  ScenarioResult,
  OptimizationResponse,
  ResourceType,
  OptimizationObjective,
} from '@/types/resources';

// =============================================================================
// SAMPLE DATA FOR DEVELOPMENT
// =============================================================================

const sampleOptimizationResult: OptimizationResponse = {
  optimization_id: 'opt_abc123',
  status: 'completed' as never,
  resource_type: 'budget' as ResourceType,
  objective: 'maximize_roi' as OptimizationObjective,
  optimal_allocations: [
    {
      entity_id: 'territory_northeast',
      entity_type: 'territory',
      current_allocation: 50000,
      optimized_allocation: 72000,
      change: 22000,
      change_percentage: 44.0,
      expected_impact: 1.35,
    },
    {
      entity_id: 'territory_southeast',
      entity_type: 'territory',
      current_allocation: 40000,
      optimized_allocation: 35000,
      change: -5000,
      change_percentage: -12.5,
      expected_impact: 0.85,
    },
    {
      entity_id: 'territory_midwest',
      entity_type: 'territory',
      current_allocation: 60000,
      optimized_allocation: 68000,
      change: 8000,
      change_percentage: 13.3,
      expected_impact: 1.2,
    },
    {
      entity_id: 'territory_west',
      entity_type: 'territory',
      current_allocation: 50000,
      optimized_allocation: 45000,
      change: -5000,
      change_percentage: -10.0,
      expected_impact: 0.95,
    },
    {
      entity_id: 'territory_southwest',
      entity_type: 'territory',
      current_allocation: 30000,
      optimized_allocation: 40000,
      change: 10000,
      change_percentage: 33.3,
      expected_impact: 1.45,
    },
  ],
  objective_value: 245000,
  solver_status: 'optimal',
  solve_time_ms: 150,
  scenarios: [
    {
      scenario_name: 'Conservative (+10%)',
      total_allocation: 220000,
      projected_outcome: 280000,
      roi: 1.27,
      constraint_violations: [],
    },
    {
      scenario_name: 'Aggressive (+25%)',
      total_allocation: 260000,
      projected_outcome: 340000,
      roi: 1.31,
      constraint_violations: ['Budget cap exceeded'],
    },
    {
      scenario_name: 'Balanced',
      total_allocation: 240000,
      projected_outcome: 310000,
      roi: 1.29,
      constraint_violations: [],
    },
  ],
  sensitivity_analysis: {
    budget_constraint: 0.15,
    min_coverage: 0.08,
    max_frequency: 0.05,
  },
  projected_total_outcome: 320000,
  projected_roi: 1.28,
  impact_by_segment: {
    high_value: 45,
    medium_value: 35,
    low_value: 20,
  },
  optimization_summary:
    'Optimized budget allocation across 5 territories with projected ROI of 1.28x. Recommend shifting resources to Northeast and Southwest territories for maximum impact.',
  recommendations: [
    'Increase Northeast territory budget by 44% to capture high-response HCPs',
    'Reduce Southeast allocation where response rates have declined',
    'Monitor Southwest territory closely - high growth potential identified',
    'Consider scenario analysis for Q2 budget planning',
  ],
  formulation_latency_ms: 45,
  optimization_latency_ms: 150,
  total_latency_ms: 280,
  timestamp: new Date().toISOString(),
  warnings: [],
};

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
          formatter={(value: number, name: string) => [
            `$${value.toFixed(0)}K`,
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
          formatter={(value: number, name: string) => [
            name === 'ROI' ? `${value.toFixed(2)}x` : `$${value.toFixed(0)}K`,
            name,
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
          shape={(props) => {
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

interface AllocationTrendProps {
  allocations: AllocationResult[];
}

function AllocationTrendChart({ allocations }: AllocationTrendProps) {
  // Simulate trend data
  const trendData = allocations.map((a) => ({
    territory: a.entity_id.replace('territory_', ''),
    q1: a.current_allocation * 0.9 / 1000,
    q2: a.current_allocation * 0.95 / 1000,
    q3: a.current_allocation / 1000,
    q4_projected: a.optimized_allocation / 1000,
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={trendData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="territory" />
        <YAxis label={{ value: 'Allocation ($K)', angle: -90, position: 'insideLeft' }} />
        <Tooltip formatter={(value) => [`$${(value as number)?.toFixed(0) ?? 0}K`, '']} />
        <Legend />
        <Line type="monotone" dataKey="q1" stroke={COLORS.muted} name="Q1" />
        <Line type="monotone" dataKey="q2" stroke={COLORS.muted} name="Q2" strokeDasharray="5 5" />
        <Line type="monotone" dataKey="q3" stroke={COLORS.secondary} name="Q3 (Current)" />
        <Line type="monotone" dataKey="q4_projected" stroke={COLORS.primary} name="Q4 (Optimized)" strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
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
  const { data: _scenariosData } = useScenarios({ limit: 10 });
  const runOptimization = useRunOptimization();

  // Use sample data for now (API may not be available)
  const optimizationResult = sampleOptimizationResult;
  const scenarios = optimizationResult.scenarios;

  // Health status
  const isHealthy = healthData?.agent_available && healthData?.scipy_available;

  // Handle optimization run
  const handleRunOptimization = () => {
    runOptimization.mutate({
      request: {
        query: `Optimize ${selectedResourceType} allocation to ${selectedObjective.replace('_', ' ')}`,
        resource_type: selectedResourceType as never,
        allocation_targets: optimizationResult.optimal_allocations.map((a) => ({
          entity_id: a.entity_id,
          entity_type: a.entity_type,
          current_allocation: a.current_allocation,
          min_allocation: a.current_allocation * 0.5,
          max_allocation: a.current_allocation * 1.5,
          expected_response: a.expected_impact,
        })),
        objective: selectedObjective as never,
        run_scenarios: true,
        scenario_count: 3,
      },
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
          {healthData?.optimizations_24h !== undefined && (
            <Badge variant="outline">{healthData.optimizations_24h} optimizations today</Badge>
          )}
        </div>
      </div>

      {/* KPI Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard
          title="Projected ROI"
          value={`${optimizationResult.projected_roi?.toFixed(2)}x`}
          description="vs current allocation"
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
              <CardDescription>Historical and projected allocation by territory</CardDescription>
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
    </div>
  );
}
