/**
 * Analytics Page
 * ==============
 *
 * Agent performance analytics and metrics dashboard.
 * Shows query execution metrics, agent latency breakdowns, and trends.
 *
 * Features:
 * - Query execution metrics over time
 * - Latency breakdown charts
 * - Per-agent success/failure rates
 * - p50/p95/p99 latency percentiles
 *
 * @module pages/Analytics
 */

import { useState, useMemo } from 'react';
import {
  BarChart3,
  Activity,
  Clock,
  CheckCircle,
  XCircle,
  TrendingUp,
  Zap,
  RefreshCw,
  ChevronDown,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useAnalyticsDashboard } from '@/hooks/api/use-analytics';
import { useDataFreshness } from '@/hooks/use-data-freshness';
import { DataFreshnessIndicator } from '@/components/ui/data-freshness-indicator';
import {
  type AnalyticsPeriod,
  type AgentMetrics,
  PERIOD_LABELS,
  formatLatency,
  formatPercent,
  formatNumber,
  getTierLabel,
} from '@/types/analytics';

// =============================================================================
// COMPONENTS
// =============================================================================

interface StatCardProps {
  title: string;
  value: string | number;
  subtext?: string;
  icon: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

function StatCard({ title, value, subtext, icon, trend, className }: StatCardProps) {
  return (
    <Card className={cn('', className)}>
      <CardContent className="p-4">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm text-muted-foreground">{title}</p>
            <p className="text-2xl font-bold mt-1">{value}</p>
            {subtext && (
              <p className="text-xs text-muted-foreground mt-1 flex items-center gap-1">
                {trend === 'up' && <TrendingUp className="h-3 w-3 text-emerald-500" />}
                {trend === 'down' && <TrendingUp className="h-3 w-3 text-rose-500 rotate-180" />}
                {subtext}
              </p>
            )}
          </div>
          <div className="p-2 rounded-lg bg-primary/10 text-primary">
            {icon}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

interface LatencyBarProps {
  label: string;
  value: number;
  maxValue: number;
  color?: string;
}

function LatencyBar({ label, value, maxValue, color = 'bg-primary' }: LatencyBarProps) {
  const percentage = maxValue > 0 ? (value / maxValue) * 100 : 0;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium">{formatLatency(value)}</span>
      </div>
      <div className="h-2 bg-muted rounded-full overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all', color)}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
    </div>
  );
}

interface AgentTableRowProps {
  agent: AgentMetrics;
  isExpanded: boolean;
  onToggle: () => void;
}

function AgentTableRow({ agent, isExpanded, onToggle }: AgentTableRowProps) {
  const successColor =
    agent.success_rate >= 95
      ? 'text-emerald-600'
      : agent.success_rate >= 80
        ? 'text-amber-600'
        : 'text-rose-600';

  return (
    <>
      <TableRow
        className="cursor-pointer hover:bg-muted/50"
        onClick={onToggle}
      >
        <TableCell>
          <div className="flex items-center gap-2">
            <ChevronDown
              className={cn(
                'h-4 w-4 text-muted-foreground transition-transform',
                isExpanded && 'rotate-180'
              )}
            />
            <span className="font-medium">{agent.agent_name}</span>
          </div>
        </TableCell>
        <TableCell>
          <Badge variant="outline">{getTierLabel(agent.agent_tier)}</Badge>
        </TableCell>
        <TableCell className="text-right">
          {formatNumber(agent.total_invocations)}
        </TableCell>
        <TableCell className={cn('text-right font-medium', successColor)}>
          {formatPercent(agent.success_rate)}
        </TableCell>
        <TableCell className="text-right">{formatLatency(agent.avg_latency_ms)}</TableCell>
        <TableCell className="text-right text-muted-foreground">
          {formatLatency(agent.p50_latency_ms)}
        </TableCell>
        <TableCell className="text-right text-muted-foreground">
          {formatLatency(agent.p95_latency_ms)}
        </TableCell>
        <TableCell className="text-right text-muted-foreground">
          {formatLatency(agent.p99_latency_ms)}
        </TableCell>
      </TableRow>
      {isExpanded && (
        <TableRow>
          <TableCell colSpan={8} className="bg-muted/30 p-4">
            <div className="grid grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Successful</p>
                <p className="font-medium text-emerald-600">
                  {formatNumber(agent.successful_invocations)}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Failed</p>
                <p className="font-medium text-rose-600">
                  {formatNumber(agent.failed_invocations)}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Min Latency</p>
                <p className="font-medium">{formatLatency(agent.min_latency_ms)}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Max Latency</p>
                <p className="font-medium">{formatLatency(agent.max_latency_ms)}</p>
              </div>
              {agent.avg_confidence !== null && (
                <div>
                  <p className="text-muted-foreground">Avg Confidence</p>
                  <p className="font-medium">{(agent.avg_confidence * 100).toFixed(1)}%</p>
                </div>
              )}
            </div>
          </TableCell>
        </TableRow>
      )}
    </>
  );
}

// =============================================================================
// MAIN PAGE
// =============================================================================

export default function Analytics() {
  const [period, setPeriod] = useState<AnalyticsPeriod>('7d');
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null);

  const {
    data: dashboard,
    isLoading,
    error,
    refetch,
    isFetching,
    dataUpdatedAt,
  } = useAnalyticsDashboard(period);

  const freshness = useDataFreshness(dataUpdatedAt);

  // Calculate max latency for breakdown visualization
  const maxBreakdownLatency = useMemo(() => {
    if (!dashboard?.latency_breakdown) return 1;
    const breakdown = dashboard.latency_breakdown;
    return Math.max(
      breakdown.classification_ms,
      breakdown.routing_ms,
      breakdown.agent_dispatch_ms,
      breakdown.synthesis_ms,
      1
    );
  }, [dashboard?.latency_breakdown]);

  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 space-y-6">
        <Card>
          <CardContent className="p-6 text-center">
            <XCircle className="h-12 w-12 text-rose-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold">Failed to load analytics</h3>
            <p className="text-muted-foreground mt-2">{error.message}</p>
            <Button onClick={() => refetch()} className="mt-4">
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const summary = dashboard?.summary;
  const breakdown = dashboard?.latency_breakdown;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-3">
            <BarChart3 className="h-7 w-7 text-primary" />
            Agent Analytics
          </h1>
          <p className="text-muted-foreground mt-1">
            Performance metrics and latency analysis for all agents
          </p>
        </div>
        <div className="flex items-center gap-3">
          <DataFreshnessIndicator
            {...freshness}
            showRefreshButton
            onRefresh={() => refetch()}
            isRefreshing={isFetching}
          />
          <Select value={period} onValueChange={(v) => setPeriod(v as AnalyticsPeriod)}>
            <SelectTrigger className="w-[180px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(PERIOD_LABELS).map(([value, label]) => (
                <SelectItem key={value} value={value}>
                  {label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Queries"
          value={formatNumber(summary?.total_queries ?? 0)}
          subtext={`${formatPercent(summary?.success_rate ?? 0)} success rate`}
          icon={<Activity className="h-5 w-5" />}
          trend={summary?.success_rate && summary.success_rate >= 95 ? 'up' : 'neutral'}
        />
        <StatCard
          title="Avg Latency"
          value={formatLatency(summary?.avg_latency_ms ?? 0)}
          subtext={`p95: ${formatLatency(summary?.p95_latency_ms ?? 0)}`}
          icon={<Clock className="h-5 w-5" />}
        />
        <StatCard
          title="Successful"
          value={formatNumber(summary?.successful_queries ?? 0)}
          icon={<CheckCircle className="h-5 w-5" />}
          className="border-emerald-200 dark:border-emerald-800"
        />
        <StatCard
          title="Failed"
          value={formatNumber(summary?.failed_queries ?? 0)}
          icon={<XCircle className="h-5 w-5" />}
          className="border-rose-200 dark:border-rose-800"
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Latency Breakdown */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-primary" />
              Latency Breakdown
            </CardTitle>
            <CardDescription>Average time spent in each processing stage</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <LatencyBar
              label="Classification"
              value={breakdown?.classification_ms ?? 0}
              maxValue={maxBreakdownLatency}
              color="bg-blue-500"
            />
            <LatencyBar
              label="Routing"
              value={breakdown?.routing_ms ?? 0}
              maxValue={maxBreakdownLatency}
              color="bg-purple-500"
            />
            <LatencyBar
              label="Agent Dispatch"
              value={breakdown?.agent_dispatch_ms ?? 0}
              maxValue={maxBreakdownLatency}
              color="bg-amber-500"
            />
            <LatencyBar
              label="Synthesis"
              value={breakdown?.synthesis_ms ?? 0}
              maxValue={maxBreakdownLatency}
              color="bg-emerald-500"
            />
            <div className="pt-4 border-t">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">Total Average</span>
                <span className="font-bold text-lg">
                  {formatLatency(breakdown?.total_ms ?? 0)}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Percentile Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              Latency Percentiles
            </CardTitle>
            <CardDescription>Query response time distribution</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              <div className="text-center">
                <p className="text-4xl font-bold">{formatLatency(summary?.p50_latency_ms ?? 0)}</p>
                <p className="text-sm text-muted-foreground">p50 (Median)</p>
              </div>
              <div className="grid grid-cols-2 gap-4 pt-4 border-t">
                <div className="text-center">
                  <p className="text-2xl font-semibold text-amber-600">
                    {formatLatency(summary?.p95_latency_ms ?? 0)}
                  </p>
                  <p className="text-xs text-muted-foreground">p95</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-semibold text-rose-600">
                    {formatLatency(summary?.p99_latency_ms ?? 0)}
                  </p>
                  <p className="text-xs text-muted-foreground">p99</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Top Agents */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              Top Agents
            </CardTitle>
            <CardDescription>Most active agents by invocation count</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {summary?.top_agents.slice(0, 5).map((agentName, index) => {
                const agent = dashboard?.agent_metrics.find((a) => a.agent_name === agentName);
                return (
                  <div
                    key={agentName}
                    className="flex items-center justify-between p-2 rounded-lg bg-muted/50"
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-lg font-bold text-muted-foreground">
                        #{index + 1}
                      </span>
                      <div>
                        <p className="font-medium">{agentName}</p>
                        {agent && (
                          <p className="text-xs text-muted-foreground">
                            {getTierLabel(agent.agent_tier)}
                          </p>
                        )}
                      </div>
                    </div>
                    {agent && (
                      <span className="text-sm font-medium">
                        {formatNumber(agent.total_invocations)}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Agent Metrics Table */}
      <Card>
        <CardHeader>
          <CardTitle>Agent Performance</CardTitle>
          <CardDescription>
            Detailed performance metrics for all {dashboard?.agent_metrics.length ?? 0} agents
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Agent</TableHead>
                <TableHead>Tier</TableHead>
                <TableHead className="text-right">Invocations</TableHead>
                <TableHead className="text-right">Success Rate</TableHead>
                <TableHead className="text-right">Avg Latency</TableHead>
                <TableHead className="text-right">p50</TableHead>
                <TableHead className="text-right">p95</TableHead>
                <TableHead className="text-right">p99</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {dashboard?.agent_metrics.map((agent) => (
                <AgentTableRow
                  key={agent.agent_name}
                  agent={agent}
                  isExpanded={expandedAgent === agent.agent_name}
                  onToggle={() =>
                    setExpandedAgent(
                      expandedAgent === agent.agent_name ? null : agent.agent_name
                    )
                  }
                />
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
