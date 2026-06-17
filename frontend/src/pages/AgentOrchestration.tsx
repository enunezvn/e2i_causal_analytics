/**
 * Agent Orchestration Page
 * ========================
 *
 * Comprehensive dashboard for the 21-agent tiered orchestration system.
 * Displays agent status, activity feeds, tier overview, and recent insights.
 *
 * @module pages/AgentOrchestration
 */

import * as React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { EmptyState } from '@/components/ui/EmptyState';
import { useE2ICopilot } from '@/providers/E2ICopilotProvider';
import { TierOverview, type AgentTier } from '@/components/visualizations/agents/AgentTierBadge';
import { AgentStatusPanel } from '@/components/chat/AgentStatusPanel';
import { getValidated } from '@/lib/api-client';
import {
  AgentStatusResponseSchema,
  TierMetricsResponseSchema,
  AgentActivityResponseSchema,
} from '@/lib/api-schemas';
import { useMetricsSummary } from '@/hooks/api/use-analytics';
import {
  Activity,
  Bot,
  CheckCircle2,
  Clock,
  AlertTriangle,
  RefreshCw,
  Zap,
  Brain,
  Target,
  LineChart,
  Sparkles,
  Layers,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// =============================================================================
// TYPES
// =============================================================================

interface AgentActivity {
  id: string;
  agentId: string;
  agentName: string;
  tier: 0 | 1 | 2 | 3 | 4 | 5;
  action: string;
  timestamp: string;
  duration?: number;
  status: 'completed' | 'in_progress' | 'failed';
  details?: string;
}

interface TierMetrics {
  tier: 0 | 1 | 2 | 3 | 4 | 5;
  name: string;
  activeAgents: number;
  totalAgents: number;
  /** Per-tier performance metrics are not yet served by /agents/status. */
  avgResponseTime: number | null;
  successRate: number | null;
  tasksCompleted: number | null;
}

interface OrchestrationStats {
  totalAgents: number;
  activeAgents: number;
  processingAgents: number;
  errorAgents: number;
  /** From /analytics/summary; null when telemetry is unavailable. */
  avgResponseTime: number | null;
  /** Cognitive queries processed in the period; null when unavailable. */
  queries24h: number | null;
  /** Percent (0-100); null when unavailable. */
  successRate: number | null;
}

// =============================================================================
// DEFAULTS
// =============================================================================
// F-002 fix: SAMPLE_ACTIVITIES formerly inlined here was DELETED. The Activity
// Feed is now wired to GET /agents/activity (real rows from
// audit_chain_entries; the automated health poller is excluded server-side) —
// see the `activities` query in the component below. No fabricated values are
// reachable from production rendering paths; an empty feed is an honest
// "no recent activity".

// Tier display names. Counts are derived from the live /agents/status
// payload; per-tier perf metrics (success rate, response time, tasks) are a
// backend gap (the endpoint returns only agent list + status), rendered "—".
const TIER_NAMES: Record<number, string> = {
  0: 'ML Foundation',
  1: 'Orchestration',
  2: 'Causal Analytics',
  3: 'Monitoring',
  4: 'ML Predictions',
  5: 'Self-Improvement',
};

// NOTE: ORCHESTRATION_STATS (fabricated tasksToday 3,010 / 680ms /
// 97.5% / 21 agents) and the hardcoded trend arrows were DELETED.
// Telemetry now comes from GET /analytics/summary (real query counts,
// latency, and success rate); absence renders an em dash, never a fake.

// =============================================================================
// TIER ICONS
// =============================================================================

const TIER_ICONS: Record<number, React.ReactNode> = {
  0: <Layers className="h-4 w-4" />,
  1: <Bot className="h-4 w-4" />,
  2: <Target className="h-4 w-4" />,
  3: <Activity className="h-4 w-4" />,
  4: <LineChart className="h-4 w-4" />,
  5: <Sparkles className="h-4 w-4" />,
};

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

function StatCard({
  title,
  value,
  subtitle,
  icon,
  className,
}: {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  className?: string;
}) {
  return (
    <Card className={cn('', className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <div className="text-muted-foreground">{icon}</div>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
      </CardContent>
    </Card>
  );
}

function ActivityItem({ activity }: { activity: AgentActivity }) {
  const statusIcon = {
    completed: <CheckCircle2 className="h-4 w-4 text-green-500" />,
    in_progress: <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />,
    failed: <AlertTriangle className="h-4 w-4 text-red-500" />,
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="flex items-start gap-3 p-3 rounded-lg hover:bg-muted/50 transition-colors">
      <div className="mt-0.5">{statusIcon[activity.status]}</div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-medium text-sm">{activity.agentName}</span>
          <Badge variant="outline" className="text-xs">
            Tier {activity.tier}
          </Badge>
        </div>
        <p className="text-sm text-muted-foreground mt-0.5">{activity.action}</p>
        {activity.details && (
          <p className="text-xs text-muted-foreground/80 mt-1 truncate">{activity.details}</p>
        )}
      </div>
      <div className="text-xs text-muted-foreground whitespace-nowrap">
        <div>{formatTime(activity.timestamp)}</div>
        {activity.duration && (
          <div className="text-right">{activity.duration}ms</div>
        )}
      </div>
    </div>
  );
}

function TierMetricsCard({ metrics }: { metrics: TierMetrics }) {
  const utilizationPercent =
    metrics.totalAgents > 0 ? (metrics.activeAgents / metrics.totalAgents) * 100 : 0;
  const fmt = (v: number | null, suffix = '') => (v === null ? '—' : `${v}${suffix}`);

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {TIER_ICONS[metrics.tier]}
            <CardTitle className="text-base">Tier {metrics.tier}: {metrics.name}</CardTitle>
          </div>
          <Badge variant={metrics.totalAgents > 0 && metrics.activeAgents === metrics.totalAgents ? 'default' : 'secondary'}>
            {metrics.activeAgents}/{metrics.totalAgents} active
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-muted-foreground">Utilization</span>
            <span>{utilizationPercent.toFixed(0)}%</span>
          </div>
          <Progress value={utilizationPercent} className="h-2" />
        </div>
        <div className="grid grid-cols-3 gap-2 text-sm">
          <div>
            <p className="text-muted-foreground text-xs">Avg Response</p>
            <p className="font-medium">{fmt(metrics.avgResponseTime, 'ms')}</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Success Rate</p>
            <p className="font-medium">{fmt(metrics.successRate, '%')}</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Tasks Today</p>
            <p className="font-medium">{fmt(metrics.tasksCompleted)}</p>
          </div>
        </div>
        {metrics.tasksCompleted === null ? (
          <p className="text-xs text-muted-foreground">
            Per-tier performance is temporarily unavailable.
          </p>
        ) : metrics.tasksCompleted === 0 ? (
          <p className="text-xs text-muted-foreground">
            No agent activity recorded for this tier in the last 24h.
          </p>
        ) : (
          <p className="text-xs text-muted-foreground">
            Success rate is not recorded per tier (validation is sparse).
          </p>
        )}
      </CardContent>
    </Card>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function AgentOrchestration() {
  const { agents } = useE2ICopilot();
  const [selectedTier, setSelectedTier] = React.useState<AgentTier | null>(null);

  // Fetch agent status from API (with fallback to context data)
  // Uses apiClient for auth headers, correlation IDs, and response validation
  const { data: agentStatus, isLoading: _isLoading, refetch: refetchAgents } = useQuery({
    queryKey: ['agent-status'],
    queryFn: () => getValidated(
      AgentStatusResponseSchema,
      '/agents/status'
    ),
    refetchInterval: 30000, // Refresh every 30 seconds
    retry: false,
  });

  // Real 24h telemetry from /analytics/summary (query counts, latency,
  // success rate). When unavailable the stat cards render an em dash.
  const { data: summary, refetch: refetchSummary } = useMetricsSummary('24h');

  // Real per-tier performance from GET /analytics/tier-metrics (audit_chain_entries,
  // automated health poller excluded). Avg Response / Tasks are real; per-tier
  // success rate is honestly null ("—") — validation is too sparse to compute.
  const { data: tierData, refetch: refetchTiers } = useQuery({
    queryKey: ['tier-metrics'],
    queryFn: () => getValidated(TierMetricsResponseSchema, '/analytics/tier-metrics?hours=24'),
    refetchInterval: 30000,
    retry: false,
  });

  // Real agent activity from GET /agents/activity (audit_chain_entries, newest
  // first; the automated health poller is excluded server-side). An empty list
  // is an honest "no recent activity", never fabricated.
  const { data: activityData, refetch: refetchActivity } = useQuery({
    queryKey: ['agent-activity'],
    queryFn: () =>
      getValidated(AgentActivityResponseSchema, '/agents/activity?hours=24&limit=50'),
    refetchInterval: 30000,
    retry: false,
  });

  // Map tier -> served perf (avg response, tasks). Absent => null ("—").
  const tierPerfByTier = React.useMemo(() => {
    const m = new Map<number, { avg: number | null; tasks: number | null }>();
    (tierData?.tiers ?? []).forEach((t) =>
      m.set(t.tier, {
        avg: t.avg_response_time_ms ?? null,
        tasks: t.tasks_completed ?? null,
      }),
    );
    return m;
  }, [tierData]);

  const activities: AgentActivity[] = React.useMemo(
    () =>
      (activityData?.activities ?? []).map((a) => ({
        id: a.entry_id,
        agentId: a.agent_id,
        agentName: a.agent_name,
        tier: a.tier as 0 | 1 | 2 | 3 | 4 | 5,
        action: a.action,
        timestamp: a.timestamp,
        duration: a.duration_ms ?? undefined,
        status: a.status,
        details: a.details ?? undefined,
      })),
    [activityData],
  );

  // Use context agents if API not available
  const displayAgents = agentStatus?.agents ?? agents;

  // Derive per-tier active/total counts from the live agent roster; merge in
  // real Avg-Response / Tasks from /analytics/tier-metrics. Success rate stays
  // null ("—") — not reliably recorded per tier.
  const tierMetrics = React.useMemo((): TierMetrics[] => {
    return ([0, 1, 2, 3, 4, 5] as const).map((tier) => {
      const inTier = displayAgents.filter((a: { tier: number }) => a.tier === tier);
      const active = inTier.filter((a: { status: string }) => a.status === 'active').length;
      const perf = tierPerfByTier.get(tier);
      return {
        tier,
        name: TIER_NAMES[tier],
        activeAgents: active,
        totalAgents: inTier.length,
        avgResponseTime: perf?.avg ?? null,
        successRate: null,
        tasksCompleted: perf?.tasks ?? null,
      };
    });
  }, [displayAgents, tierPerfByTier]);

  // Filter agents by selected tier
  const filteredAgents = selectedTier !== null
    ? displayAgents.filter((a: { tier: number }) => a.tier === selectedTier)
    : displayAgents;

  // Stats: roster counts from the live agent list; telemetry from
  // /analytics/summary. Nothing is fabricated — missing data is null.
  const stats: OrchestrationStats = React.useMemo(() => {
    const active = displayAgents.filter((a: { status: string }) => a.status === 'active').length;
    const processing = displayAgents.filter((a: { status: string }) => a.status === 'processing').length;
    const error = displayAgents.filter((a: { status: string }) => a.status === 'error').length;
    return {
      totalAgents: agentStatus?.total ?? displayAgents.length,
      activeAgents: active,
      processingAgents: processing,
      errorAgents: error,
      queries24h: summary?.total_queries ?? null,
      // avg_latency_ms is null == UNMEASURED (no audit entry carried a real
      // duration_ms in the window). A value of exactly 0 is the same artifact:
      // before the agent graphs were instrumented, a genesis-only window left
      // the latency list empty and the backend reported 0.0, which rendered a
      // misleading "0ms / instant". Treat both null AND 0-with-no-real-latency
      // as unmeasured -> the card shows "—". A real per-node duration is always
      // >= ~1ms, so a measured 0 is not a plausible real average.
      avgResponseTime:
        typeof summary?.avg_latency_ms === 'number' && summary.avg_latency_ms > 0
          ? Math.round(summary.avg_latency_ms)
          : null,
      successRate: typeof summary?.success_rate === 'number' ? summary.success_rate : null,
    };
  }, [displayAgents, agentStatus, summary]);

  const handleRefresh = React.useCallback(() => {
    refetchAgents();
    refetchSummary();
    refetchTiers();
    refetchActivity();
  }, [refetchAgents, refetchSummary, refetchTiers, refetchActivity]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Agent Orchestration</h1>
          <p className="text-muted-foreground">
            Monitor and manage the 21-agent tiered orchestration system
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* "Pause All" was removed: no pause/resume endpoint exists, so a
              dead control would fake orchestration management capability. */}
          <Button variant="outline" size="sm" onClick={handleRefresh}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Stats Overview — roster counts are live; telemetry comes from
          /analytics/summary and renders an em dash when unavailable. */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Total Agents"
          value={stats.totalAgents}
          subtitle={`${stats.activeAgents} active, ${stats.processingAgents} processing`}
          icon={<Bot className="h-4 w-4" />}
        />
        <StatCard
          title="Queries (24h)"
          value={stats.queries24h === null ? '—' : stats.queries24h.toLocaleString()}
          subtitle="Cognitive queries processed"
          icon={<Zap className="h-4 w-4" />}
        />
        <StatCard
          title="Avg Response Time"
          value={stats.avgResponseTime === null ? '—' : `${stats.avgResponseTime}ms`}
          subtitle="24h average latency"
          icon={<Clock className="h-4 w-4" />}
        />
        <StatCard
          title="Success Rate"
          value={stats.successRate === null ? '—' : `${stats.successRate}%`}
          subtitle="24h query success rate"
          icon={<CheckCircle2 className="h-4 w-4" />}
        />
      </div>

      {/* Main Content */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="activity">Activity Feed</TabsTrigger>
          <TabsTrigger value="tiers">Tier Metrics</TabsTrigger>
          <TabsTrigger value="agents">All Agents</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-3">
            {/* Tier Overview */}
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>Tier Architecture</CardTitle>
                <CardDescription>
                  6-tier hierarchy with 21 specialized agents
                </CardDescription>
              </CardHeader>
              <CardContent>
                <TierOverview
                  activeTier={selectedTier ?? undefined}
                  onTierSelect={(tier: AgentTier) => setSelectedTier(tier === selectedTier ? null : tier)}
                />
              </CardContent>
            </Card>

            {/* Quick Status */}
            <Card>
              <CardHeader>
                <CardTitle>Agent Status</CardTitle>
                <CardDescription>Current agent states</CardDescription>
              </CardHeader>
              <CardContent>
                <AgentStatusPanel
                  agents={filteredAgents}
                  compact={true}
                />
              </CardContent>
            </Card>
          </div>

          {/* Recent Activity Preview — no action buttons until the activity
              endpoint exists (dead controls would fake capability). */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Activity</CardTitle>
              <CardDescription>Latest agent actions and events</CardDescription>
            </CardHeader>
            <CardContent>
              {activities.length === 0 ? (
                <EmptyState
                  title="No recent activity"
                  description="No agent activity recorded in the last 24 hours."
                />
              ) : (
                <div className="space-y-1">
                  {activities.slice(0, 3).map((activity) => (
                    <ActivityItem key={activity.id} activity={activity} />
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Activity Feed Tab */}
        <TabsContent value="activity" className="space-y-4">
          <Card>
            <CardHeader>
              {/* Filter/Export controls intentionally absent: the activity
                  endpoint is unwired, so there is nothing to filter or
                  export — dead buttons would fake capability. */}
              <CardTitle>Activity Feed</CardTitle>
              <CardDescription>Complete log of agent actions</CardDescription>
            </CardHeader>
            <CardContent>
              {activities.length === 0 ? (
                <EmptyState
                  title="No activity to display"
                  description="No agent activity in the last 24 hours. Actions appear here as agents run."
                />
              ) : (
                <div className="space-y-1">
                  {activities.map((activity) => (
                    <ActivityItem key={activity.id} activity={activity} />
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tier Metrics Tab */}
        <TabsContent value="tiers" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {tierMetrics.map((metrics) => (
              <TierMetricsCard key={metrics.tier} metrics={metrics} />
            ))}
          </div>
        </TabsContent>

        {/* All Agents Tab */}
        <TabsContent value="agents" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>All Agents ({filteredAgents.length})</CardTitle>
                  <CardDescription>
                    {selectedTier !== null
                      ? `Showing Tier ${selectedTier} agents`
                      : `All ${displayAgents.length} agents across 6 tiers`}
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  {selectedTier !== null && (
                    <Button variant="ghost" size="sm" onClick={() => setSelectedTier(null)}>
                      Clear Filter
                    </Button>
                  )}
                  <select
                    className="px-3 py-1.5 text-sm border rounded-md"
                    value={selectedTier ?? ''}
                    onChange={(e) => setSelectedTier(e.target.value ? (Number(e.target.value) as AgentTier) : null)}
                  >
                    <option value="">All Tiers</option>
                    <option value="0">Tier 0 - ML Foundation</option>
                    <option value="1">Tier 1 - Orchestration</option>
                    <option value="2">Tier 2 - Causal Analytics</option>
                    <option value="3">Tier 3 - Monitoring</option>
                    <option value="4">Tier 4 - ML Predictions</option>
                    <option value="5">Tier 5 - Self-Improvement</option>
                  </select>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                {filteredAgents.map((agent: { id: string; name: string; tier: number; status: string; capabilities: string[] }) => (
                  <Card key={agent.id} className="hover:shadow-md transition-shadow">
                    <CardContent className="p-4">
                      {/* No per-agent Play control: no run-agent endpoint
                          exists, so a launch button would fake capability. */}
                      <div className="flex items-start justify-between">
                        <div>
                          <div className="flex items-center gap-2">
                            <Brain className="h-4 w-4 text-primary" />
                            <span className="font-medium">{agent.name}</span>
                          </div>
                          <div className="flex items-center gap-2 mt-1">
                            <Badge variant="outline" className="text-xs">
                              Tier {agent.tier}
                            </Badge>
                            <Badge
                              variant={
                                agent.status === 'active' ? 'default' :
                                agent.status === 'processing' ? 'secondary' :
                                agent.status === 'error' ? 'destructive' : 'outline'
                              }
                              className="text-xs"
                            >
                              {agent.status}
                            </Badge>
                          </div>
                        </div>
                      </div>
                      <div className="mt-3">
                        <p className="text-xs text-muted-foreground">Capabilities:</p>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {agent.capabilities.slice(0, 3).map((cap: string) => (
                            <Badge key={cap} variant="secondary" className="text-xs">
                              {cap.replace('_', ' ')}
                            </Badge>
                          ))}
                          {agent.capabilities.length > 3 && (
                            <Badge variant="secondary" className="text-xs">
                              +{agent.capabilities.length - 3}
                            </Badge>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
