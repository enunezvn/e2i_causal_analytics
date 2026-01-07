/**
 * System Health Score Component
 * =============================
 *
 * Displays overall system health metrics and model status.
 * Provides at-a-glance operational health monitoring.
 *
 * @module components/insights/SystemHealthScore
 */

import { useState, useEffect } from 'react';
import {
  Activity,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Server,
  Database,
  Cpu,
  RefreshCw,
  Clock,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { useModelHealth, useMonitoringRuns, useAlerts } from '@/hooks/api/use-monitoring';
import { AlertStatus } from '@/types/monitoring';

// =============================================================================
// TYPES
// =============================================================================

interface SystemHealthScoreProps {
  className?: string;
  modelId?: string;
}

interface HealthMetric {
  id: string;
  name: string;
  value: number;
  status: 'healthy' | 'warning' | 'critical';
  description: string;
  icon: React.ReactNode;
}

interface SystemSummary {
  overallScore: number;
  status: 'healthy' | 'warning' | 'critical';
  lastCheck: Date;
  activeAlerts: number;
  modelsMonitored: number;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_METRICS: HealthMetric[] = [
  {
    id: 'model-drift',
    name: 'Model Drift',
    value: 92,
    status: 'healthy',
    description: 'All models within acceptable drift thresholds',
    icon: <Activity className="h-4 w-4" />,
  },
  {
    id: 'data-quality',
    name: 'Data Quality',
    value: 88,
    status: 'healthy',
    description: 'Data pipelines operating normally',
    icon: <Database className="h-4 w-4" />,
  },
  {
    id: 'api-latency',
    name: 'API Latency',
    value: 76,
    status: 'warning',
    description: 'P95 latency slightly elevated (245ms)',
    icon: <Server className="h-4 w-4" />,
  },
  {
    id: 'inference-throughput',
    name: 'Inference Throughput',
    value: 95,
    status: 'healthy',
    description: '1,247 predictions/min',
    icon: <Cpu className="h-4 w-4" />,
  },
];

const SAMPLE_SUMMARY: SystemSummary = {
  overallScore: 87,
  status: 'healthy',
  lastCheck: new Date(),
  activeAlerts: 2,
  modelsMonitored: 8,
};

// =============================================================================
// HELPERS
// =============================================================================

function getStatusConfig(status: HealthMetric['status']) {
  const config = {
    healthy: {
      label: 'Healthy',
      className: 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20',
      icon: <CheckCircle2 className="h-4 w-4 text-emerald-500" />,
      progressColor: 'bg-emerald-500',
    },
    warning: {
      label: 'Warning',
      className: 'bg-amber-500/10 text-amber-600 border-amber-500/20',
      icon: <AlertTriangle className="h-4 w-4 text-amber-500" />,
      progressColor: 'bg-amber-500',
    },
    critical: {
      label: 'Critical',
      className: 'bg-rose-500/10 text-rose-600 border-rose-500/20',
      icon: <XCircle className="h-4 w-4 text-rose-500" />,
      progressColor: 'bg-rose-500',
    },
  };
  return config[status];
}

function getOverallStatusIcon(status: SystemSummary['status']) {
  switch (status) {
    case 'healthy':
      return <CheckCircle2 className="h-8 w-8 text-emerald-500" />;
    case 'warning':
      return <AlertTriangle className="h-8 w-8 text-amber-500" />;
    case 'critical':
      return <XCircle className="h-8 w-8 text-rose-500" />;
  }
}

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

function MetricRow({ metric }: { metric: HealthMetric }) {
  const statusConfig = getStatusConfig(metric.status);

  return (
    <div className="flex items-center gap-3 p-3 rounded-lg bg-[var(--color-muted)]/20">
      {/* Icon */}
      <div
        className={cn(
          'flex-shrink-0 p-2 rounded-lg',
          metric.status === 'healthy'
            ? 'bg-emerald-500/10'
            : metric.status === 'warning'
              ? 'bg-amber-500/10'
              : 'bg-rose-500/10'
        )}
      >
        <div
          className={cn(
            metric.status === 'healthy'
              ? 'text-emerald-500'
              : metric.status === 'warning'
                ? 'text-amber-500'
                : 'text-rose-500'
          )}
        >
          {metric.icon}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm font-medium text-[var(--color-foreground)]">{metric.name}</span>
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold">{metric.value}%</span>
            {statusConfig.icon}
          </div>
        </div>
        <Progress
          value={metric.value}
          className={cn(
            'h-1.5',
            metric.status === 'healthy' && '[&>div]:bg-emerald-500',
            metric.status === 'warning' && '[&>div]:bg-amber-500',
            metric.status === 'critical' && '[&>div]:bg-rose-500'
          )}
        />
        <p className="text-xs text-[var(--color-muted-foreground)] mt-1">{metric.description}</p>
      </div>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function SystemHealthScore({ className, modelId = 'propensity_v2.1.0' }: SystemHealthScoreProps) {
  const [metrics, setMetrics] = useState<HealthMetric[]>(SAMPLE_METRICS);
  const [summary, setSummary] = useState<SystemSummary>(SAMPLE_SUMMARY);

  // Fetch model health
  const { data: healthData, isLoading: healthLoading, refetch: refetchHealth } = useModelHealth(modelId);

  // Fetch monitoring runs
  const { data: runsData, isLoading: runsLoading } = useMonitoringRuns({ days: 7, limit: 10 });

  // Fetch active alerts count
  const { data: alertsData } = useAlerts({ status: AlertStatus.ACTIVE, limit: 100 });

  // Update summary from API data
  useEffect(() => {
    if (healthData) {
      const statusMap: Record<string, SystemSummary['status']> = {
        healthy: 'healthy',
        warning: 'warning',
        critical: 'critical',
      };

      setSummary((prev) => ({
        ...prev,
        status: statusMap[healthData.overall_health] || 'healthy',
        overallScore:
          healthData.overall_health === 'healthy'
            ? 95
            : healthData.overall_health === 'warning'
              ? 75
              : 45,
        activeAlerts: healthData.active_alerts,
        lastCheck: healthData.last_check ? new Date(healthData.last_check) : new Date(),
      }));

      // Update drift metric from health data
      setMetrics((prev) =>
        prev.map((m) =>
          m.id === 'model-drift'
            ? {
                ...m,
                value: Math.round((1 - healthData.drift_score) * 100),
                status:
                  healthData.drift_score < 0.1
                    ? 'healthy'
                    : healthData.drift_score < 0.3
                      ? 'warning'
                      : 'critical',
              }
            : m
        )
      );
    }
  }, [healthData]);

  // Update models monitored from runs data
  useEffect(() => {
    if (runsData) {
      setSummary((prev) => ({
        ...prev,
        modelsMonitored: runsData.total_runs > 0 ? Math.min(runsData.total_runs, 12) : prev.modelsMonitored,
      }));
    }
  }, [runsData]);

  // Update active alerts
  useEffect(() => {
    if (alertsData) {
      setSummary((prev) => ({
        ...prev,
        activeAlerts: alertsData.active_count,
      }));
    }
  }, [alertsData]);

  const handleRefresh = () => {
    refetchHealth();
  };

  const isLoading = healthLoading || runsLoading;
  const statusConfig = getStatusConfig(summary.status);

  return (
    <Card className={cn('bg-[var(--color-card)] border-[var(--color-border)]', className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-emerald-500/10">
              <Activity className="h-5 w-5 text-emerald-500" />
            </div>
            <div>
              <CardTitle className="text-base font-semibold">System Health Score</CardTitle>
              <p className="text-xs text-[var(--color-muted-foreground)]">
                Operational metrics and model status
              </p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={handleRefresh}
            disabled={isLoading}
            className="h-8 w-8"
          >
            <RefreshCw className={cn('h-4 w-4', isLoading && 'animate-spin')} />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Overall Score */}
        <div
          className={cn(
            'p-4 rounded-lg border',
            summary.status === 'healthy'
              ? 'border-emerald-500/30 bg-emerald-500/5'
              : summary.status === 'warning'
                ? 'border-amber-500/30 bg-amber-500/5'
                : 'border-rose-500/30 bg-rose-500/5'
          )}
        >
          <div className="flex items-center gap-4">
            {getOverallStatusIcon(summary.status)}
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-2xl font-bold text-[var(--color-foreground)]">
                  {summary.overallScore}%
                </span>
                <Badge variant="outline" className={cn('text-xs', statusConfig.className)}>
                  {statusConfig.label}
                </Badge>
              </div>
              <p className="text-sm text-[var(--color-muted-foreground)]">
                Overall System Health Score
              </p>
            </div>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-3 gap-3">
          <div className="p-3 rounded-lg bg-[var(--color-muted)]/30 text-center">
            <div className="text-lg font-bold text-[var(--color-foreground)]">
              {summary.modelsMonitored}
            </div>
            <div className="text-xs text-[var(--color-muted-foreground)]">Models</div>
          </div>
          <div className="p-3 rounded-lg bg-[var(--color-muted)]/30 text-center">
            <div
              className={cn(
                'text-lg font-bold',
                summary.activeAlerts > 0 ? 'text-amber-600' : 'text-emerald-600'
              )}
            >
              {summary.activeAlerts}
            </div>
            <div className="text-xs text-[var(--color-muted-foreground)]">Active Alerts</div>
          </div>
          <div className="p-3 rounded-lg bg-[var(--color-muted)]/30 text-center">
            <div className="text-lg font-bold text-[var(--color-foreground)]">
              {summary.lastCheck.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
            <div className="text-xs text-[var(--color-muted-foreground)]">Last Check</div>
          </div>
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="flex items-center justify-center py-4">
            <div className="flex items-center gap-3 text-[var(--color-muted-foreground)]">
              <RefreshCw className="h-5 w-5 animate-spin" />
              <span className="text-sm">Loading health metrics...</span>
            </div>
          </div>
        )}

        {/* Metrics List */}
        {!isLoading && (
          <div className="space-y-2">
            {metrics.map((metric) => (
              <MetricRow key={metric.id} metric={metric} />
            ))}
          </div>
        )}

        {/* Recommendations */}
        {healthData?.recommendations && healthData.recommendations.length > 0 && (
          <div className="p-3 rounded-lg bg-blue-500/5 border border-blue-500/20">
            <div className="flex items-center gap-2 mb-2">
              <Clock className="h-4 w-4 text-blue-500" />
              <span className="text-xs font-medium text-blue-600">Recommendations</span>
            </div>
            <ul className="space-y-1">
              {healthData.recommendations.slice(0, 3).map((rec, idx) => (
                <li key={idx} className="text-xs text-[var(--color-muted-foreground)]">
                  • {rec}
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default SystemHealthScore;
