/**
 * System Health Page
 * ==================
 *
 * Dashboard for monitoring E2I system health including:
 * - Service status grid (API, Database, Redis, FalkorDB, BentoML)
 * - Model health cards with health scores
 * - Active alerts list
 * - Auto-refresh every 30s
 *
 * @module pages/SystemHealth
 */

import { useState, useMemo, useCallback, useEffect } from 'react';
import {
  Server,
  Database,
  HardDrive,
  Cpu,
  Activity,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  Brain,
  TrendingUp,
  TrendingDown,
  Minus,
  Workflow,
  Bot,
  Shield,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useAlerts, useMonitoringRuns } from '@/hooks/api/use-monitoring';
import {
  useQuickHealthCheck,
  usePipelineHealth,
  useAgentHealth,
  useHealthHistory,
} from '@/hooks/api';
import { AlertStatus } from '@/types/monitoring';
import type { AlertItem } from '@/types/monitoring';
import { HealthGrade } from '@/types/health-score';
import type { AgentHealth, PipelineHealth as PipelineHealthType } from '@/types/health-score';
import { PipelineStatus } from '@/types/health-score';
import { AlertList } from '@/components/visualizations/dashboard/AlertCard';
import { StatusBadge, StatusDot } from '@/components/visualizations/dashboard/StatusBadge';
import { ProgressRing } from '@/components/visualizations/dashboard/ProgressRing';
import type { AlertSeverity } from '@/components/visualizations/dashboard/AlertCard';
import type { StatusType } from '@/components/visualizations/dashboard/StatusBadge';

// =============================================================================
// TYPES
// =============================================================================

interface ServiceStatus {
  name: string;
  status: 'healthy' | 'warning' | 'error' | 'unknown';
  latencyMs?: number;
  lastCheck?: Date;
  icon: React.ElementType;
}

interface ModelHealth {
  modelId: string;
  name: string;
  healthScore: number;
  status: 'healthy' | 'warning' | 'critical';
  driftScore: number;
  activeAlerts: number;
  lastRetrained?: Date;
  performanceTrend: 'improving' | 'stable' | 'degrading';
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_SERVICES: ServiceStatus[] = [
  { name: 'API Gateway', status: 'healthy', latencyMs: 45, lastCheck: new Date(), icon: Server },
  { name: 'PostgreSQL', status: 'healthy', latencyMs: 12, lastCheck: new Date(), icon: Database },
  { name: 'Redis Cache', status: 'healthy', latencyMs: 3, lastCheck: new Date(), icon: HardDrive },
  { name: 'FalkorDB', status: 'healthy', latencyMs: 28, lastCheck: new Date(), icon: Activity },
  { name: 'BentoML', status: 'healthy', latencyMs: 156, lastCheck: new Date(), icon: Cpu },
];

const SAMPLE_MODELS: ModelHealth[] = [
  {
    modelId: 'propensity_v2.1.0',
    name: 'Propensity Model',
    healthScore: 92,
    status: 'healthy',
    driftScore: 0.15,
    activeAlerts: 0,
    lastRetrained: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    performanceTrend: 'stable',
  },
  {
    modelId: 'churn_v1.5.2',
    name: 'Churn Prediction',
    healthScore: 78,
    status: 'warning',
    driftScore: 0.42,
    activeAlerts: 2,
    lastRetrained: new Date(Date.now() - 21 * 24 * 60 * 60 * 1000),
    performanceTrend: 'degrading',
  },
  {
    modelId: 'conversion_v3.0.1',
    name: 'Conversion Model',
    healthScore: 88,
    status: 'healthy',
    driftScore: 0.22,
    activeAlerts: 1,
    lastRetrained: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000),
    performanceTrend: 'improving',
  },
];

// Grade color mapping
const GRADE_COLORS: Record<HealthGrade | string, string> = {
  [HealthGrade.A]: 'text-emerald-600 bg-emerald-100 border-emerald-300',
  [HealthGrade.B]: 'text-green-600 bg-green-100 border-green-300',
  [HealthGrade.C]: 'text-amber-600 bg-amber-100 border-amber-300',
  [HealthGrade.D]: 'text-orange-600 bg-orange-100 border-orange-300',
  [HealthGrade.F]: 'text-rose-600 bg-rose-100 border-rose-300',
};

const TIER_NAMES: Record<number, string> = {
  0: 'Foundation',
  1: 'Orchestration',
  2: 'Causal',
  3: 'Monitoring',
  4: 'ML Predictions',
  5: 'Learning',
};

// Sample history data for fallback
const SAMPLE_HISTORY = [
  { timestamp: new Date(Date.now() - 6 * 24 * 60 * 60 * 1000).toISOString(), overall_health_score: 85, health_grade: 'B' },
  { timestamp: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(), overall_health_score: 82, health_grade: 'B' },
  { timestamp: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(), overall_health_score: 88, health_grade: 'B' },
  { timestamp: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(), overall_health_score: 91, health_grade: 'A' },
  { timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(), overall_health_score: 89, health_grade: 'B' },
  { timestamp: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(), overall_health_score: 92, health_grade: 'A' },
  { timestamp: new Date().toISOString(), overall_health_score: 94, health_grade: 'A' },
];

// Sample agent health data
const SAMPLE_AGENT_HEALTH: AgentHealth[] = [
  { agent_name: 'Orchestrator', tier: 1, available: true, avg_latency_ms: 120, success_rate: 0.98, invocations_24h: 245 },
  { agent_name: 'ToolComposer', tier: 1, available: true, avg_latency_ms: 85, success_rate: 0.99, invocations_24h: 180 },
  { agent_name: 'CausalImpact', tier: 2, available: true, avg_latency_ms: 450, success_rate: 0.95, invocations_24h: 67 },
  { agent_name: 'GapAnalyzer', tier: 2, available: true, avg_latency_ms: 320, success_rate: 0.97, invocations_24h: 89 },
  { agent_name: 'HeterogeneousOptimizer', tier: 2, available: true, avg_latency_ms: 380, success_rate: 0.94, invocations_24h: 45 },
  { agent_name: 'DriftMonitor', tier: 3, available: true, avg_latency_ms: 200, success_rate: 0.99, invocations_24h: 156 },
  { agent_name: 'ExperimentDesigner', tier: 3, available: true, avg_latency_ms: 280, success_rate: 0.96, invocations_24h: 34 },
  { agent_name: 'HealthScore', tier: 3, available: true, avg_latency_ms: 150, success_rate: 0.99, invocations_24h: 312 },
  { agent_name: 'PredictionSynthesizer', tier: 4, available: true, avg_latency_ms: 520, success_rate: 0.97, invocations_24h: 23 },
  { agent_name: 'ResourceOptimizer', tier: 4, available: true, avg_latency_ms: 410, success_rate: 0.95, invocations_24h: 56 },
  { agent_name: 'Explainer', tier: 5, available: true, avg_latency_ms: 680, success_rate: 0.93, invocations_24h: 112 },
  { agent_name: 'FeedbackLearner', tier: 5, available: true, avg_latency_ms: 340, success_rate: 0.97, invocations_24h: 78 },
];

// Sample pipeline health data
const SAMPLE_PIPELINE_HEALTH: PipelineHealthType[] = [
  { pipeline_name: 'TRx Data Ingestion', last_run: new Date().toISOString(), last_success: new Date().toISOString(), rows_processed: 125000, freshness_hours: 2.5, status: PipelineStatus.HEALTHY },
  { pipeline_name: 'Feature Store Sync', last_run: new Date().toISOString(), last_success: new Date().toISOString(), rows_processed: 45000, freshness_hours: 1.2, status: PipelineStatus.HEALTHY },
  { pipeline_name: 'Model Retraining', last_run: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(), last_success: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(), rows_processed: 8500, freshness_hours: 24, status: PipelineStatus.STALE },
  { pipeline_name: 'Causal Graph Update', last_run: new Date().toISOString(), last_success: new Date().toISOString(), rows_processed: 12000, freshness_hours: 4.5, status: PipelineStatus.HEALTHY },
];

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function mapHealthToStatus(health: string): StatusType {
  switch (health) {
    case 'healthy':
      return 'healthy';
    case 'warning':
      return 'warning';
    case 'critical':
    case 'error':
      return 'error';
    default:
      return 'unknown';
  }
}

function mapAlertSeverity(severity: string): AlertSeverity {
  switch (severity.toLowerCase()) {
    case 'critical':
      return 'critical';
    case 'high':
      return 'error';
    case 'medium':
      return 'warning';
    default:
      return 'info';
  }
}

function formatDate(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function formatRelativeTime(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  const days = Math.floor(diff / (24 * 60 * 60 * 1000));

  if (days === 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days} days ago`;
  if (days < 30) return `${Math.floor(days / 7)} weeks ago`;
  return formatDate(d);
}

// =============================================================================
// COMPONENT
// =============================================================================

function SystemHealth() {
  const [lastRefresh, setLastRefresh] = useState(new Date());
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  // Fetch alerts from API
  const {
    data: alertsData,
    isLoading: isLoadingAlerts,
    refetch: refetchAlerts,
  } = useAlerts({ status: AlertStatus.ACTIVE, limit: 10 }, {
    refetchInterval: 30000, // Auto-refresh every 30s
  });

  // Fetch monitoring runs
  const {
    data: _runsData,
    isLoading: isLoadingRuns,
    refetch: refetchRuns,
  } = useMonitoringRuns({ days: 7, limit: 5 });

  // Health Score API hooks
  const {
    data: quickHealthData,
    refetch: refetchHealth,
  } = useQuickHealthCheck({ refetchInterval: 30000 });

  const { data: agentHealthData } = useAgentHealth({ refetchInterval: 60000 });
  const { data: pipelineHealthData } = usePipelineHealth({ refetchInterval: 60000 });
  const { data: healthHistoryData } = useHealthHistory(20, { refetchInterval: 120000 });

  // In production, fetch from API. Using sample data for now.
  const services = SAMPLE_SERVICES;
  const models = SAMPLE_MODELS;

  // Use API data or fallback to samples
  const healthScore = quickHealthData?.overall_health_score ?? 92;
  const healthGrade = quickHealthData?.health_grade ?? HealthGrade.A;
  const agents = agentHealthData?.agents ?? SAMPLE_AGENT_HEALTH;
  const pipelines = pipelineHealthData?.pipelines ?? SAMPLE_PIPELINE_HEALTH;
  const healthHistory = healthHistoryData?.checks ?? SAMPLE_HISTORY;
  const healthTrend = healthHistoryData?.trend ?? 'stable';

  // Group agents by tier
  const agentsByTier = useMemo(() => {
    const grouped: Record<number, AgentHealth[]> = {};
    agents.forEach(agent => {
      if (!grouped[agent.tier]) grouped[agent.tier] = [];
      grouped[agent.tier].push(agent);
    });
    return grouped;
  }, [agents]);

  // Prepare chart data
  const historyChartData = useMemo(() => {
    return healthHistory.map(item => ({
      date: formatDate(item.timestamp),
      score: item.overall_health_score,
      grade: item.health_grade,
    }));
  }, [healthHistory]);

  const componentScoreData = useMemo(() => {
    if (!quickHealthData) {
      return [
        { name: 'Components', score: 95, fill: '#10b981' },
        { name: 'Models', score: 88, fill: '#3b82f6' },
        { name: 'Pipelines', score: 82, fill: '#8b5cf6' },
        { name: 'Agents', score: 92, fill: '#f59e0b' },
      ];
    }
    return [
      { name: 'Components', score: Math.round(quickHealthData.component_health_score * 100), fill: '#10b981' },
      { name: 'Models', score: Math.round(quickHealthData.model_health_score * 100), fill: '#3b82f6' },
      { name: 'Pipelines', score: Math.round(quickHealthData.pipeline_health_score * 100), fill: '#8b5cf6' },
      { name: 'Agents', score: Math.round(quickHealthData.agent_health_score * 100), fill: '#f59e0b' },
    ];
  }, [quickHealthData]);

  // Convert API alerts to AlertCard format
  const alerts = useMemo(() => {
    if (!alertsData?.alerts) return [];
    return alertsData.alerts.map((alert: AlertItem) => ({
      severity: mapAlertSeverity(alert.severity),
      title: alert.title,
      message: alert.description,
      timestamp: alert.triggered_at,
      source: alert.model_version,
      isNew: alert.status === AlertStatus.ACTIVE,
    }));
  }, [alertsData?.alerts]);

  // Calculate overall health stats
  const healthStats = useMemo(() => {
    const healthyServices = services.filter(s => s.status === 'healthy').length;
    const totalServices = services.length;
    const avgLatency = services.reduce((sum, s) => sum + (s.latencyMs || 0), 0) / services.length;

    const healthyModels = models.filter(m => m.status === 'healthy').length;
    const warningModels = models.filter(m => m.status === 'warning').length;
    const criticalModels = models.filter(m => m.status === 'critical').length;

    return {
      healthyServices,
      totalServices,
      serviceHealth: (healthyServices / totalServices) * 100,
      avgLatency: Math.round(avgLatency),
      healthyModels,
      warningModels,
      criticalModels,
      totalAlerts: alertsData?.active_count || 0,
    };
  }, [services, models, alertsData?.active_count]);

  // Refresh handler
  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true);
    await Promise.all([refetchAlerts(), refetchRuns(), refetchHealth()]);
    setLastRefresh(new Date());
    setIsRefreshing(false);
  }, [refetchAlerts, refetchRuns, refetchHealth]);

  // Auto-refresh timestamp update
  useEffect(() => {
    const interval = setInterval(() => {
      setLastRefresh(new Date());
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  const isLoading = isLoadingAlerts || isLoadingRuns;

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Page Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">System Health</h1>
          <p className="text-[var(--color-muted-foreground)]">
            Comprehensive system monitoring with health scores
          </p>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-sm text-[var(--color-muted-foreground)]">
            Last updated: {lastRefresh.toLocaleTimeString()}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={isRefreshing}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Overall Health Score Card */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
        <Card className="md:col-span-1">
          <CardHeader className="pb-2">
            <CardDescription>Overall Health</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4">
              <div className="text-4xl font-bold">{healthScore}</div>
              <div className={`px-3 py-1 rounded-lg border text-xl font-bold ${GRADE_COLORS[healthGrade] || GRADE_COLORS[HealthGrade.C]}`}>
                {healthGrade}
              </div>
            </div>
            <div className="flex items-center gap-1 mt-2 text-sm text-[var(--color-muted-foreground)]">
              {healthTrend === 'improving' && <TrendingUp className="h-4 w-4 text-emerald-500" />}
              {healthTrend === 'declining' && <TrendingDown className="h-4 w-4 text-rose-500" />}
              {healthTrend === 'stable' && <Minus className="h-4 w-4 text-slate-500" />}
              {healthTrend.charAt(0).toUpperCase() + healthTrend.slice(1)}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Services</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {healthStats.healthyServices}/{healthStats.totalServices}
              <StatusDot status="healthy" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Avg latency: {healthStats.avgLatency}ms
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Models</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {healthStats.healthyModels} / {models.length}
              {healthStats.warningModels > 0 && (
                <Badge variant="outline" className="text-amber-600 border-amber-300">
                  {healthStats.warningModels} warn
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              {healthStats.criticalModels} critical
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Agents</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {agents.filter(a => a.available).length} / {agents.length}
              <Bot className="h-5 w-5 text-[var(--color-muted-foreground)]" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              {agents.filter(a => !a.available).length} unavailable
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Active Alerts</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {healthStats.totalAlerts}
              {healthStats.totalAlerts > 0 ? (
                <AlertCircle className="h-5 w-5 text-amber-500" />
              ) : (
                <CheckCircle2 className="h-5 w-5 text-emerald-500" />
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              {healthStats.totalAlerts === 0 ? 'All clear' : 'Requires attention'}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Tabs for different health views */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5 lg:w-auto lg:inline-flex">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="agents">Agents</TabsTrigger>
          <TabsTrigger value="pipelines">Pipelines</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          {/* Component Score Breakdown + Health History */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="h-5 w-5" />
                  Component Scores
                </CardTitle>
                <CardDescription>Health breakdown by system component</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={componentScoreData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" domain={[0, 100]} />
                    <YAxis type="category" dataKey="name" width={80} />
                    <Tooltip formatter={(value) => [`${value ?? 0}%`, 'Score']} />
                    <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                      {componentScoreData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5" />
                  Health Trend
                </CardTitle>
                <CardDescription>7-day health score history</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={historyChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis domain={[60, 100]} />
                    <Tooltip formatter={(value) => [`${value ?? 0}`, 'Health Score']} />
                    <Line
                      type="monotone"
                      dataKey="score"
                      stroke="#10b981"
                      strokeWidth={2}
                      dot={{ fill: '#10b981', strokeWidth: 2 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Services and Models Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Services Status */}
            <Card className="lg:col-span-1">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Server className="h-5 w-5" />
                  Service Status
                </CardTitle>
                <CardDescription>Infrastructure components</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {services.map((service) => {
                  const Icon = service.icon;
                  return (
                    <div
                      key={service.name}
                      className="flex items-center justify-between p-3 rounded-lg bg-[var(--color-muted)]/50"
                    >
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-[var(--color-background)]">
                          <Icon className="h-4 w-4 text-[var(--color-muted-foreground)]" />
                        </div>
                        <div>
                          <p className="font-medium text-sm">{service.name}</p>
                          {service.latencyMs !== undefined && (
                            <p className="text-xs text-[var(--color-muted-foreground)]">
                              {service.latencyMs}ms
                            </p>
                          )}
                        </div>
                      </div>
                      <StatusBadge status={mapHealthToStatus(service.status)} size="sm" />
                    </div>
                  );
                })}
              </CardContent>
            </Card>

            {/* Model Health Cards */}
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  Model Health
                </CardTitle>
                <CardDescription>ML model performance and drift status</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                  {models.map((model) => (
                    <div
                      key={model.modelId}
                      className="p-4 rounded-lg border border-[var(--color-border)] bg-[var(--color-card)]"
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <h4 className="font-semibold text-sm">{model.name}</h4>
                          <p className="text-xs text-[var(--color-muted-foreground)]">
                            {model.modelId}
                          </p>
                        </div>
                        <ProgressRing
                          value={model.healthScore}
                          size={48}
                          strokeWidth={4}
                          status={model.status}
                        />
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-[var(--color-muted-foreground)]">Drift</span>
                          <span className={model.driftScore > 0.3 ? 'text-amber-600' : ''}>
                            {(model.driftScore * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-[var(--color-muted-foreground)]">Trend</span>
                          <span className="flex items-center gap-1">
                            {model.performanceTrend === 'improving' && (
                              <TrendingUp className="h-3 w-3 text-emerald-500" />
                            )}
                            {model.performanceTrend === 'degrading' && (
                              <TrendingDown className="h-3 w-3 text-rose-500" />
                            )}
                            {model.performanceTrend === 'stable' && (
                              <Minus className="h-3 w-3 text-slate-500" />
                            )}
                            {model.performanceTrend}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Agents Tab */}
        <TabsContent value="agents" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bot className="h-5 w-5" />
                Agent Health by Tier
              </CardTitle>
              <CardDescription>18-agent tiered orchestration system status</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {Object.entries(agentsByTier).sort((a, b) => parseInt(a[0]) - parseInt(b[0])).map(([tier, tierAgents]) => (
                <div key={tier}>
                  <div className="flex items-center gap-2 mb-3">
                    <Badge variant="outline">Tier {tier}</Badge>
                    <span className="text-sm font-medium">{TIER_NAMES[parseInt(tier)] || 'Unknown'}</span>
                    <span className="text-xs text-[var(--color-muted-foreground)]">
                      ({tierAgents.filter(a => a.available).length}/{tierAgents.length} available)
                    </span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {tierAgents.map(agent => (
                      <div
                        key={agent.agent_name}
                        className={`p-4 rounded-lg border ${
                          agent.available
                            ? 'border-emerald-200 bg-emerald-50/50 dark:bg-emerald-950/20'
                            : 'border-rose-200 bg-rose-50/50 dark:bg-rose-950/20'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-sm">{agent.agent_name}</span>
                          <StatusDot status={agent.available ? 'healthy' : 'error'} />
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs text-[var(--color-muted-foreground)]">
                          <div>
                            <p>Latency</p>
                            <p className="font-medium text-[var(--color-foreground)]">{agent.avg_latency_ms}ms</p>
                          </div>
                          <div>
                            <p>Success</p>
                            <p className="font-medium text-[var(--color-foreground)]">{(agent.success_rate * 100).toFixed(0)}%</p>
                          </div>
                          <div className="col-span-2">
                            <p>24h Invocations: <span className="font-medium text-[var(--color-foreground)]">{agent.invocations_24h}</span></p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Pipelines Tab */}
        <TabsContent value="pipelines" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Workflow className="h-5 w-5" />
                Data Pipeline Health
              </CardTitle>
              <CardDescription>ETL and data processing pipeline status</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {pipelines.map(pipeline => (
                  <div
                    key={pipeline.pipeline_name}
                    className="flex items-center justify-between p-4 rounded-lg border border-[var(--color-border)]"
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium">{pipeline.pipeline_name}</span>
                        <StatusBadge
                          status={
                            pipeline.status === 'healthy' ? 'healthy' :
                            pipeline.status === 'stale' ? 'warning' : 'error'
                          }
                          size="sm"
                        />
                      </div>
                      <p className="text-sm text-[var(--color-muted-foreground)]">
                        Last run: {formatRelativeTime(pipeline.last_run)} | Rows: {pipeline.rows_processed.toLocaleString()}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium">
                        {pipeline.freshness_hours < 1
                          ? `${Math.round(pipeline.freshness_hours * 60)}m`
                          : `${pipeline.freshness_hours.toFixed(1)}h`} fresh
                      </p>
                      <p className="text-xs text-[var(--color-muted-foreground)]">Data freshness</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Health Score History
              </CardTitle>
              <CardDescription>Historical health check records</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={historyChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis domain={[60, 100]} />
                  <Tooltip
                    formatter={(value, name) => [value ?? 0, name === 'score' ? 'Health Score' : name]}
                    labelFormatter={(label) => `Date: ${label}`}
                  />
                  <Line
                    type="monotone"
                    dataKey="score"
                    stroke="#10b981"
                    strokeWidth={2}
                    dot={{ fill: '#10b981', strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
              <div className="mt-4 grid grid-cols-3 gap-4 text-center">
                <div className="p-3 rounded-lg bg-[var(--color-muted)]/50">
                  <p className="text-sm text-[var(--color-muted-foreground)]">Average</p>
                  <p className="text-2xl font-bold">{healthHistoryData?.avg_health_score?.toFixed(1) ?? '89.5'}</p>
                </div>
                <div className="p-3 rounded-lg bg-[var(--color-muted)]/50">
                  <p className="text-sm text-[var(--color-muted-foreground)]">Trend</p>
                  <p className="text-2xl font-bold flex items-center justify-center gap-1">
                    {healthTrend === 'improving' && <TrendingUp className="h-5 w-5 text-emerald-500" />}
                    {healthTrend === 'declining' && <TrendingDown className="h-5 w-5 text-rose-500" />}
                    {healthTrend === 'stable' && <Minus className="h-5 w-5 text-slate-500" />}
                    {healthTrend.charAt(0).toUpperCase() + healthTrend.slice(1)}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-[var(--color-muted)]/50">
                  <p className="text-sm text-[var(--color-muted-foreground)]">Total Checks</p>
                  <p className="text-2xl font-bold">{healthHistoryData?.total_checks ?? healthHistory.length}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Alerts Tab */}
        <TabsContent value="alerts" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertCircle className="h-5 w-5" />
                Active Alerts
              </CardTitle>
              <CardDescription>Recent alerts requiring attention</CardDescription>
            </CardHeader>
            <CardContent>
              <AlertList
                alerts={alerts.length > 0 ? alerts : [
                  {
                    severity: 'warning' as AlertSeverity,
                    title: 'Data Drift Detected',
                    message: 'Churn model feature distribution has shifted significantly.',
                    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
                    source: 'Drift Monitor',
                    isNew: true,
                  },
                  {
                    severity: 'info' as AlertSeverity,
                    title: 'Model Retraining Scheduled',
                    message: 'Propensity model scheduled for retraining based on drift threshold.',
                    timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000),
                    source: 'Retraining Manager',
                    isNew: false,
                  },
                ]}
                compact={false}
                maxItems={10}
                isLoading={isLoading}
                emptyMessage="No active alerts - all systems operational"
              />
            </CardContent>
          </Card>

          {/* Issues and Recommendations from Health Check */}
          {quickHealthData && (quickHealthData.critical_issues?.length > 0 || quickHealthData.warnings?.length > 0) && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {quickHealthData.critical_issues?.length > 0 && (
                <Card className="border-rose-200">
                  <CardHeader>
                    <CardTitle className="text-rose-600">Critical Issues</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2">
                      {quickHealthData.critical_issues.map((issue, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm">
                          <AlertCircle className="h-4 w-4 text-rose-500 mt-0.5 flex-shrink-0" />
                          {issue}
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}
              {quickHealthData.warnings?.length > 0 && (
                <Card className="border-amber-200">
                  <CardHeader>
                    <CardTitle className="text-amber-600">Warnings</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2">
                      {quickHealthData.warnings.map((warning, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm">
                          <AlertCircle className="h-4 w-4 text-amber-500 mt-0.5 flex-shrink-0" />
                          {warning}
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}
            </div>
          )}

          {(quickHealthData?.recommendations?.length ?? 0) > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                  Recommendations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {quickHealthData?.recommendations?.map((rec, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm">
                      <CheckCircle2 className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                      {rec}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default SystemHealth;
