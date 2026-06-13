/**
 * System Health Score Component
 * =============================
 *
 * Displays the REAL measured system health from the Tier-3 Health Score
 * agent (`GET /api/health-score/full`):
 * - overall_health_score (0-100, measured) + letter grade + provenance
 * - the four health dimensions (component / model / pipeline / agent,
 *   0-1 scores; null dimensions render an em-dash, never a fake number)
 * - critical issues / recommendations
 * Active alert count comes from the monitoring alerts API, and an optional
 * per-model drift row from `GET /api/monitoring/health/{modelId}`.
 *
 * Fabricated SAMPLE_METRICS / SAMPLE_SUMMARY (87% score, "1,247
 * predictions/min", invented healthy->95 / warning->75 / critical->45
 * mapping) were DELETED. Honest states only: real data, explicit empty,
 * labeled error.
 *
 * @module components/insights/SystemHealthScore
 */

import {
  Activity,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Server,
  Database,
  Cpu,
  Bot,
  RefreshCw,
  Clock,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { EmptyState } from '@/components/ui/EmptyState';
import { useFullHealthCheck } from '@/hooks/api/use-health-score';
import { useModelHealth, useAlerts } from '@/hooks/api/use-monitoring';
import { AlertStatus } from '@/types/monitoring';

// =============================================================================
// TYPES
// =============================================================================

interface SystemHealthScoreProps {
  className?: string;
  /** Optional model id for the per-model drift row. */
  modelId?: string;
}

type HealthStatus = 'healthy' | 'warning' | 'critical';

interface DimensionRow {
  id: string;
  name: string;
  /** Real 0-1 score from the API, or null when unmeasured. */
  score: number | null;
  icon: React.ReactNode;
}

// =============================================================================
// HELPERS
// =============================================================================

function statusFromScore(score: number | null): HealthStatus | null {
  if (score === null) return null;
  if (score >= 0.8) return 'healthy';
  if (score >= 0.5) return 'warning';
  return 'critical';
}

function overallStatus(score: number): HealthStatus {
  if (score >= 80) return 'healthy';
  if (score >= 50) return 'warning';
  return 'critical';
}

function getStatusConfig(status: HealthStatus) {
  const config = {
    healthy: {
      label: 'Healthy',
      className: 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20',
      icon: <CheckCircle2 className="h-4 w-4 text-emerald-500" />,
    },
    warning: {
      label: 'Warning',
      className: 'bg-amber-500/10 text-amber-600 border-amber-500/20',
      icon: <AlertTriangle className="h-4 w-4 text-amber-500" />,
    },
    critical: {
      label: 'Critical',
      className: 'bg-rose-500/10 text-rose-600 border-rose-500/20',
      icon: <XCircle className="h-4 w-4 text-rose-500" />,
    },
  };
  return config[status];
}

function getOverallStatusIcon(status: HealthStatus) {
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

function DimensionRowView({ row }: { row: DimensionRow }) {
  const status = statusFromScore(row.score);
  const statusConfig = status ? getStatusConfig(status) : null;
  const pct = row.score === null ? null : Math.round(row.score * 100);

  return (
    <div className="flex items-center gap-3 p-3 rounded-lg bg-[var(--color-muted)]/20">
      <div
        className={cn(
          'flex-shrink-0 p-2 rounded-lg',
          status === 'healthy'
            ? 'bg-emerald-500/10'
            : status === 'warning'
              ? 'bg-amber-500/10'
              : status === 'critical'
                ? 'bg-rose-500/10'
                : 'bg-[var(--color-muted)]/40'
        )}
      >
        <div
          className={cn(
            status === 'healthy'
              ? 'text-emerald-500'
              : status === 'warning'
                ? 'text-amber-500'
                : status === 'critical'
                  ? 'text-rose-500'
                  : 'text-[var(--color-muted-foreground)]'
          )}
        >
          {row.icon}
        </div>
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm font-medium text-[var(--color-foreground)]">{row.name}</span>
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold">{pct === null ? '—' : `${pct}%`}</span>
            {statusConfig?.icon}
          </div>
        </div>
        {pct !== null ? (
          <Progress
            value={pct}
            className={cn(
              'h-1.5',
              status === 'healthy' && '[&>div]:bg-emerald-500',
              status === 'warning' && '[&>div]:bg-amber-500',
              status === 'critical' && '[&>div]:bg-rose-500'
            )}
          />
        ) : (
          <p className="text-xs text-[var(--color-muted-foreground)]">
            Not measured in this check
          </p>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function SystemHealthScore({ className, modelId }: SystemHealthScoreProps) {
  // Real measured system health (Tier-3 health-score agent).
  const {
    data: health,
    isLoading,
    isError,
    error,
    refetch,
  } = useFullHealthCheck();

  // Real active alert count.
  const { data: alertsData } = useAlerts({ status: AlertStatus.ACTIVE, limit: 100 });

  // Optional per-model drift (real measured drift score for one model).
  const { data: modelHealth } = useModelHealth(modelId ?? '');

  const handleRefresh = () => {
    refetch();
  };

  const asScore = (v: number | null | undefined): number | null =>
    typeof v === 'number' ? v : null;

  const dimensions: DimensionRow[] = health
    ? [
        {
          id: 'component',
          name: 'Component Health',
          score: asScore(health.component_health_score),
          icon: <Server className="h-4 w-4" />,
        },
        {
          id: 'model',
          name: 'Model Health',
          score: asScore(health.model_health_score),
          icon: <Cpu className="h-4 w-4" />,
        },
        {
          id: 'pipeline',
          name: 'Pipeline Health',
          score: asScore(health.pipeline_health_score),
          icon: <Database className="h-4 w-4" />,
        },
        {
          id: 'agent',
          name: 'Agent Health',
          score: asScore(health.agent_health_score),
          icon: <Bot className="h-4 w-4" />,
        },
        ...(modelId && modelHealth
          ? [
              {
                id: 'model-drift',
                name: `Drift — ${modelId}`,
                score: asScore(
                  typeof modelHealth.drift_score === 'number'
                    ? 1 - modelHealth.drift_score
                    : null
                ),
                icon: <Activity className="h-4 w-4" />,
              },
            ]
          : []),
      ]
    : [];

  const score = health ? Math.round(health.overall_health_score) : null;
  const status: HealthStatus | null = score !== null ? overallStatus(score) : null;
  const statusConfig = status ? getStatusConfig(status) : null;
  const activeAlerts = alertsData?.active_count ?? null;
  const provenance = (health as { data_provenance?: string } | undefined)?.data_provenance;

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
                Measured by the health-score service
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
        {/* Loading State */}
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <div className="flex items-center gap-3 text-[var(--color-muted-foreground)]">
              <RefreshCw className="h-5 w-5 animate-spin" />
              <span className="text-sm">Loading health metrics...</span>
            </div>
          </div>
        )}

        {/* Error State */}
        {!isLoading && isError && (
          <div className="flex items-start gap-2 p-3 rounded-lg bg-rose-500/5 border border-rose-500/20">
            <AlertTriangle className="h-4 w-4 text-rose-500 mt-0.5" />
            <div className="text-xs text-[var(--color-muted-foreground)]">
              <span className="font-medium text-rose-600">Unable to load system health:</span>{' '}
              {error?.message ?? 'Health-score service unavailable'}
            </div>
          </div>
        )}

        {/* Honest empty state */}
        {!isLoading && !isError && !health && (
          <EmptyState
            title="No health data available"
            description="The health-score service has not returned a check result yet."
          />
        )}

        {/* Real health data */}
        {!isLoading && !isError && health && score !== null && status && statusConfig && (
          <>
            {/* Overall Score */}
            <div
              className={cn(
                'p-4 rounded-lg border',
                status === 'healthy'
                  ? 'border-emerald-500/30 bg-emerald-500/5'
                  : status === 'warning'
                    ? 'border-amber-500/30 bg-amber-500/5'
                    : 'border-rose-500/30 bg-rose-500/5'
              )}
            >
              <div className="flex items-center gap-4">
                {getOverallStatusIcon(status)}
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1 flex-wrap">
                    <span className="text-2xl font-bold text-[var(--color-foreground)]">
                      {score}%
                    </span>
                    <Badge variant="outline" className={cn('text-xs', statusConfig.className)}>
                      Grade {health.health_grade}
                    </Badge>
                    {provenance && provenance !== 'measured' && (
                      <Badge
                        variant="outline"
                        className="text-xs bg-amber-500/10 text-amber-600 border-amber-500/20"
                      >
                        provenance: {provenance}
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm text-[var(--color-muted-foreground)]">
                    {health.health_summary}
                  </p>
                </div>
              </div>
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-3 gap-3">
              <div className="p-3 rounded-lg bg-[var(--color-muted)]/30 text-center">
                <div className="text-lg font-bold text-[var(--color-foreground)]">
                  {health.model_metrics?.length ?? '—'}
                </div>
                <div className="text-xs text-[var(--color-muted-foreground)]">Models</div>
              </div>
              <div className="p-3 rounded-lg bg-[var(--color-muted)]/30 text-center">
                <div
                  className={cn(
                    'text-lg font-bold',
                    (activeAlerts ?? 0) > 0 ? 'text-amber-600' : 'text-emerald-600'
                  )}
                >
                  {activeAlerts ?? '—'}
                </div>
                <div className="text-xs text-[var(--color-muted-foreground)]">Active Alerts</div>
              </div>
              <div className="p-3 rounded-lg bg-[var(--color-muted)]/30 text-center">
                <div className="text-lg font-bold text-[var(--color-foreground)]">
                  {new Date(health.timestamp).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </div>
                <div className="text-xs text-[var(--color-muted-foreground)]">Last Check</div>
              </div>
            </div>

            {/* Dimension Rows */}
            <div className="space-y-2">
              {dimensions.map((row) => (
                <DimensionRowView key={row.id} row={row} />
              ))}
            </div>

            {/* Critical issues */}
            {health.critical_issues.length > 0 && (
              <div className="p-3 rounded-lg bg-rose-500/5 border border-rose-500/20">
                <div className="flex items-center gap-2 mb-2">
                  <XCircle className="h-4 w-4 text-rose-500" />
                  <span className="text-xs font-medium text-rose-600">Critical Issues</span>
                </div>
                <ul className="space-y-1">
                  {health.critical_issues.slice(0, 3).map((issue, idx) => (
                    <li key={idx} className="text-xs text-[var(--color-muted-foreground)]">
                      • {issue}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Recommendations */}
            {health.recommendations.length > 0 && (
              <div className="p-3 rounded-lg bg-blue-500/5 border border-blue-500/20">
                <div className="flex items-center gap-2 mb-2">
                  <Clock className="h-4 w-4 text-blue-500" />
                  <span className="text-xs font-medium text-blue-600">Recommendations</span>
                </div>
                <ul className="space-y-1">
                  {health.recommendations.slice(0, 3).map((rec, idx) => (
                    <li key={idx} className="text-xs text-[var(--color-muted-foreground)]">
                      • {rec}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

export default SystemHealthScore;
