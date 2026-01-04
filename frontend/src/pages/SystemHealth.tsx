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
  Clock,
  Brain,
  TrendingUp,
  TrendingDown,
  Minus,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { useAlerts, useMonitoringRuns } from '@/hooks/api/use-monitoring';
import { AlertStatus } from '@/types/monitoring';
import type { AlertItem } from '@/types/monitoring';
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
    data: runsData,
    isLoading: isLoadingRuns,
    refetch: refetchRuns,
  } = useMonitoringRuns({ days: 7, limit: 5 });

  // In production, fetch from API. Using sample data for now.
  const services = SAMPLE_SERVICES;
  const models = SAMPLE_MODELS;

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
    await Promise.all([refetchAlerts(), refetchRuns()]);
    setLastRefresh(new Date());
    setIsRefreshing(false);
  }, [refetchAlerts, refetchRuns]);

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
            Service status, model health, and active alerts
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

      {/* Overview Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
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
            <CardDescription>Model Health</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {healthStats.healthyModels} healthy
              {healthStats.warningModels > 0 && (
                <Badge variant="outline" className="text-amber-600 border-amber-300">
                  {healthStats.warningModels} warning
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-1">
              <Badge variant="secondary" className="text-xs">
                {healthStats.criticalModels} critical
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Active Alerts</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {healthStats.totalAlerts}
              {healthStats.totalAlerts > 0 && (
                <AlertCircle className="h-5 w-5 text-amber-500" />
              )}
              {healthStats.totalAlerts === 0 && (
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

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Recent Runs</CardDescription>
            <CardTitle className="text-2xl">
              {runsData?.total_runs ?? '-'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Last 7 days
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Services Status */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Server className="h-5 w-5" />
              Service Status
            </CardTitle>
            <CardDescription>
              Infrastructure component health
            </CardDescription>
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
            <CardDescription>
              ML model performance and drift status
            </CardDescription>
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
                      <span className="text-[var(--color-muted-foreground)]">Alerts</span>
                      <span className={model.activeAlerts > 0 ? 'text-rose-600' : ''}>
                        {model.activeAlerts}
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
                    {model.lastRetrained && (
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-[var(--color-muted-foreground)]">Retrained</span>
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {formatRelativeTime(model.lastRetrained)}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Active Alerts Section */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5" />
            Active Alerts
          </CardTitle>
          <CardDescription>
            Recent alerts requiring attention
          </CardDescription>
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
            compact
            maxItems={5}
            isLoading={isLoading}
            emptyMessage="No active alerts - all systems operational"
          />
        </CardContent>
      </Card>
    </div>
  );
}

export default SystemHealth;
