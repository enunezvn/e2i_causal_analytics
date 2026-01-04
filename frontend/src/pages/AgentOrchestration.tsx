/**
 * Agent Orchestration Page
 * ========================
 *
 * Comprehensive dashboard for the 18-agent tiered orchestration system.
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
import { useE2ICopilot } from '@/providers/E2ICopilotProvider';
import { TierOverview } from '@/components/visualizations/agents/AgentTierBadge';
import { AgentStatusPanel } from '@/components/chat/AgentStatusPanel';
import {
  Activity,
  Bot,
  CheckCircle2,
  Clock,
  AlertTriangle,
  Play,
  Pause,
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
  avgResponseTime: number;
  successRate: number;
  tasksCompleted: number;
}

interface OrchestrationStats {
  totalAgents: number;
  activeAgents: number;
  processingAgents: number;
  errorAgents: number;
  avgResponseTime: number;
  tasksToday: number;
  successRate: number;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_ACTIVITIES: AgentActivity[] = [
  {
    id: 'act-1',
    agentId: 'orchestrator',
    agentName: 'Orchestrator',
    tier: 1,
    action: 'Routed query to Causal Impact agent',
    timestamp: new Date(Date.now() - 2 * 60000).toISOString(),
    duration: 145,
    status: 'completed',
    details: 'Query: "What drove Remibrutinib growth in Q4?"',
  },
  {
    id: 'act-2',
    agentId: 'causal-impact',
    agentName: 'Causal Impact',
    tier: 2,
    action: 'Traced causal chain for HCP engagement → TRx',
    timestamp: new Date(Date.now() - 5 * 60000).toISOString(),
    duration: 3200,
    status: 'completed',
    details: 'Found 3 significant causal paths with ATE = 0.23',
  },
  {
    id: 'act-3',
    agentId: 'drift-monitor',
    agentName: 'Drift Monitor',
    tier: 3,
    action: 'Data drift check scheduled',
    timestamp: new Date(Date.now() - 8 * 60000).toISOString(),
    status: 'in_progress',
    details: 'Checking conversion_model features',
  },
  {
    id: 'act-4',
    agentId: 'explainer',
    agentName: 'Explainer',
    tier: 5,
    action: 'Generated SHAP narrative for prediction',
    timestamp: new Date(Date.now() - 12 * 60000).toISOString(),
    duration: 890,
    status: 'completed',
    details: 'Patient journey explanation with 5 key factors',
  },
  {
    id: 'act-5',
    agentId: 'gap-analyzer',
    agentName: 'Gap Analyzer',
    tier: 2,
    action: 'ROI opportunity detection',
    timestamp: new Date(Date.now() - 15 * 60000).toISOString(),
    status: 'failed',
    details: 'Timeout waiting for feature store response',
  },
];

const TIER_METRICS: TierMetrics[] = [
  { tier: 0, name: 'ML Foundation', activeAgents: 5, totalAgents: 7, avgResponseTime: 450, successRate: 98.5, tasksCompleted: 234 },
  { tier: 1, name: 'Orchestration', activeAgents: 2, totalAgents: 2, avgResponseTime: 120, successRate: 99.8, tasksCompleted: 1205 },
  { tier: 2, name: 'Causal Analytics', activeAgents: 2, totalAgents: 3, avgResponseTime: 2800, successRate: 94.2, tasksCompleted: 156 },
  { tier: 3, name: 'Monitoring', activeAgents: 3, totalAgents: 3, avgResponseTime: 350, successRate: 99.1, tasksCompleted: 892 },
  { tier: 4, name: 'ML Predictions', activeAgents: 1, totalAgents: 2, avgResponseTime: 180, successRate: 97.6, tasksCompleted: 445 },
  { tier: 5, name: 'Self-Improvement', activeAgents: 1, totalAgents: 2, avgResponseTime: 560, successRate: 96.3, tasksCompleted: 78 },
];

const ORCHESTRATION_STATS: OrchestrationStats = {
  totalAgents: 19,
  activeAgents: 14,
  processingAgents: 3,
  errorAgents: 0,
  avgResponseTime: 680,
  tasksToday: 3010,
  successRate: 97.5,
};

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
  trend,
  className,
}: {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  trend?: { value: number; positive: boolean };
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
        {trend && (
          <p className={cn('text-xs mt-1', trend.positive ? 'text-green-600' : 'text-red-600')}>
            {trend.positive ? '+' : ''}{trend.value}% from last hour
          </p>
        )}
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
  const utilizationPercent = (metrics.activeAgents / metrics.totalAgents) * 100;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {TIER_ICONS[metrics.tier]}
            <CardTitle className="text-base">Tier {metrics.tier}: {metrics.name}</CardTitle>
          </div>
          <Badge variant={metrics.activeAgents === metrics.totalAgents ? 'default' : 'secondary'}>
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
            <p className="font-medium">{metrics.avgResponseTime}ms</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Success Rate</p>
            <p className="font-medium">{metrics.successRate}%</p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Tasks Today</p>
            <p className="font-medium">{metrics.tasksCompleted}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function AgentOrchestration() {
  const { agents } = useE2ICopilot();
  const [selectedTier, setSelectedTier] = React.useState<number | null>(null);

  // Fetch agent status from API (with fallback to context data)
  const { data: agentStatus, isLoading } = useQuery({
    queryKey: ['agent-status'],
    queryFn: async () => {
      const response = await fetch('/api/agents/status');
      if (!response.ok) throw new Error('Failed to fetch agent status');
      return response.json();
    },
    refetchInterval: 30000, // Refresh every 30 seconds
    retry: false,
  });

  // Use context agents if API not available
  const displayAgents = agentStatus?.agents ?? agents;

  // Filter agents by selected tier
  const filteredAgents = selectedTier !== null
    ? displayAgents.filter((a: { tier: number }) => a.tier === selectedTier)
    : displayAgents;

  // Calculate stats from agents
  const stats = React.useMemo(() => {
    const active = displayAgents.filter((a: { status: string }) => a.status === 'active').length;
    const processing = displayAgents.filter((a: { status: string }) => a.status === 'processing').length;
    const error = displayAgents.filter((a: { status: string }) => a.status === 'error').length;
    return {
      ...ORCHESTRATION_STATS,
      activeAgents: active,
      processingAgents: processing,
      errorAgents: error,
    };
  }, [displayAgents]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Agent Orchestration</h1>
          <p className="text-muted-foreground">
            Monitor and manage the 18-agent tiered orchestration system
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm">
            <Pause className="h-4 w-4 mr-2" />
            Pause All
          </Button>
          <Button variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Total Agents"
          value={stats.totalAgents}
          subtitle={`${stats.activeAgents} active, ${stats.processingAgents} processing`}
          icon={<Bot className="h-4 w-4" />}
        />
        <StatCard
          title="Tasks Today"
          value={stats.tasksToday.toLocaleString()}
          subtitle="Across all tiers"
          icon={<Zap className="h-4 w-4" />}
          trend={{ value: 12, positive: true }}
        />
        <StatCard
          title="Avg Response Time"
          value={`${stats.avgResponseTime}ms`}
          subtitle="Last hour average"
          icon={<Clock className="h-4 w-4" />}
          trend={{ value: 5, positive: false }}
        />
        <StatCard
          title="Success Rate"
          value={`${stats.successRate}%`}
          subtitle="Task completion rate"
          icon={<CheckCircle2 className="h-4 w-4" />}
          trend={{ value: 0.3, positive: true }}
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
                  6-tier hierarchy with 18 specialized agents
                </CardDescription>
              </CardHeader>
              <CardContent>
                <TierOverview
                  activeTier={selectedTier ?? undefined}
                  onTierClick={(tier) => setSelectedTier(tier === selectedTier ? null : tier)}
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
                  compact={true}
                  onAgentClick={(agentId) => console.log('Agent clicked:', agentId)}
                />
              </CardContent>
            </Card>
          </div>

          {/* Recent Activity Preview */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Recent Activity</CardTitle>
                  <CardDescription>Latest agent actions and events</CardDescription>
                </div>
                <Button variant="ghost" size="sm">
                  View All
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-1">
                {SAMPLE_ACTIVITIES.slice(0, 3).map((activity) => (
                  <ActivityItem key={activity.id} activity={activity} />
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Activity Feed Tab */}
        <TabsContent value="activity" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Activity Feed</CardTitle>
                  <CardDescription>Complete log of agent actions</CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm">
                    Filter
                  </Button>
                  <Button variant="outline" size="sm">
                    Export
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-1">
                {SAMPLE_ACTIVITIES.map((activity) => (
                  <ActivityItem key={activity.id} activity={activity} />
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tier Metrics Tab */}
        <TabsContent value="tiers" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {TIER_METRICS.map((metrics) => (
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
                      : 'All 18 agents across 6 tiers'}
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
                    onChange={(e) => setSelectedTier(e.target.value ? Number(e.target.value) : null)}
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
                        <Button variant="ghost" size="sm">
                          <Play className="h-4 w-4" />
                        </Button>
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
