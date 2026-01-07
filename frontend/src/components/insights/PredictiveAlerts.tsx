/**
 * Predictive Alerts Component
 * ===========================
 *
 * Displays forecasting warnings and predictive alerts from the monitoring system.
 * Shows drift detection, performance alerts, and proactive recommendations.
 *
 * @module components/insights/PredictiveAlerts
 */

import { AlertTriangle, Bell, TrendingDown, Activity, Clock, ExternalLink } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { useAlerts } from '@/hooks/api/use-monitoring';
import { AlertStatus } from '@/types/monitoring';
import type { AlertItem } from '@/types/monitoring';

// =============================================================================
// TYPES
// =============================================================================

interface PredictiveAlertsProps {
  className?: string;
  modelId?: string;
}

interface AlertDisplay {
  id: string;
  title: string;
  description: string;
  severity: 'critical' | 'warning' | 'info';
  type: string;
  triggeredAt: string;
  modelVersion: string;
  icon: React.ReactNode;
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_ALERTS: AlertDisplay[] = [
  {
    id: 'alert-1',
    title: 'Model Drift Detected - SE Region',
    description:
      'Feature distribution shift detected in prior authorization rates. P-value: 0.003. Recommend retraining within 14 days.',
    severity: 'critical',
    type: 'drift',
    triggeredAt: '2 hours ago',
    modelVersion: 'propensity_v2.1.0',
    icon: <TrendingDown className="h-4 w-4" />,
  },
  {
    id: 'alert-2',
    title: 'Performance Degradation Warning',
    description:
      'Conversion prediction accuracy dropped 3.2% in last 7 days. Current AUC: 0.82 (threshold: 0.85).',
    severity: 'warning',
    type: 'performance',
    triggeredAt: '6 hours ago',
    modelVersion: 'conversion_v1.3.0',
    icon: <Activity className="h-4 w-4" />,
  },
  {
    id: 'alert-3',
    title: 'New High-Value Segment Identified',
    description:
      'Clustering analysis discovered untapped HCP segment with 2.1x higher predicted conversion rate.',
    severity: 'info',
    type: 'insight',
    triggeredAt: '1 day ago',
    modelVersion: 'segmentation_v3.0.0',
    icon: <Bell className="h-4 w-4" />,
  },
];

// =============================================================================
// HELPERS
// =============================================================================

function getSeverityConfig(severity: AlertDisplay['severity']) {
  const config = {
    critical: {
      label: 'Critical',
      className: 'bg-rose-500/10 text-rose-600 border-rose-500/20',
      iconColor: 'text-rose-500',
    },
    warning: {
      label: 'Warning',
      className: 'bg-amber-500/10 text-amber-600 border-amber-500/20',
      iconColor: 'text-amber-500',
    },
    info: {
      label: 'Info',
      className: 'bg-blue-500/10 text-blue-600 border-blue-500/20',
      iconColor: 'text-blue-500',
    },
  };
  return config[severity];
}

function transformAlertToDisplay(alert: AlertItem): AlertDisplay {
  const severityMap: Record<string, AlertDisplay['severity']> = {
    critical: 'critical',
    warning: 'warning',
    info: 'info',
  };

  return {
    id: alert.id,
    title: alert.title,
    description: alert.description ?? '',
    severity: severityMap[alert.severity] ?? 'info',
    type: alert.alert_type,
    triggeredAt: alert.triggered_at
      ? new Date(alert.triggered_at).toLocaleString()
      : 'Unknown',
    modelVersion: alert.model_version,
    icon:
      alert.alert_type === 'drift' ? (
        <TrendingDown className="h-4 w-4" />
      ) : alert.alert_type === 'performance' ? (
        <Activity className="h-4 w-4" />
      ) : (
        <Bell className="h-4 w-4" />
      ),
  };
}

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

function AlertCard({ alert }: { alert: AlertDisplay }) {
  const severityConfig = getSeverityConfig(alert.severity);

  return (
    <div
      className={cn(
        'p-4 rounded-lg border',
        alert.severity === 'critical'
          ? 'border-rose-500/30 bg-rose-500/5'
          : alert.severity === 'warning'
            ? 'border-amber-500/30 bg-amber-500/5'
            : 'border-[var(--color-border)] bg-[var(--color-card)]'
      )}
    >
      <div className="flex items-start gap-3">
        {/* Icon */}
        <div
          className={cn(
            'flex-shrink-0 p-2 rounded-lg',
            alert.severity === 'critical'
              ? 'bg-rose-500/10'
              : alert.severity === 'warning'
                ? 'bg-amber-500/10'
                : 'bg-blue-500/10'
          )}
        >
          <div className={severityConfig.iconColor}>{alert.icon}</div>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2 mb-1">
            <h4 className="text-sm font-medium text-[var(--color-foreground)]">{alert.title}</h4>
            <Badge variant="outline" className={cn('text-xs flex-shrink-0', severityConfig.className)}>
              {severityConfig.label}
            </Badge>
          </div>

          <p className="text-sm text-[var(--color-muted-foreground)] mb-2">{alert.description}</p>

          {/* Metadata Row */}
          <div className="flex items-center gap-4 text-xs text-[var(--color-muted-foreground)]">
            <div className="flex items-center gap-1">
              <Clock className="h-3 w-3" />
              <span>{alert.triggeredAt}</span>
            </div>
            <div className="flex items-center gap-1">
              <Activity className="h-3 w-3" />
              <span>{alert.modelVersion}</span>
            </div>
            <Badge variant="secondary" className="text-xs">
              {alert.type}
            </Badge>
          </div>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function PredictiveAlerts({ className, modelId }: PredictiveAlertsProps) {
  // Fetch alerts from API
  const { data: alertsResponse, isLoading } = useAlerts({
    model_id: modelId,
    status: AlertStatus.ACTIVE,
    limit: 10,
  });

  // Transform API alerts or use sample data
  const alerts: AlertDisplay[] =
    alertsResponse?.alerts && alertsResponse.alerts.length > 0
      ? alertsResponse.alerts.slice(0, 5).map(transformAlertToDisplay)
      : SAMPLE_ALERTS;

  const criticalCount = alerts.filter((a) => a.severity === 'critical').length;
  const warningCount = alerts.filter((a) => a.severity === 'warning').length;

  return (
    <Card className={cn('bg-[var(--color-card)] border-[var(--color-border)]', className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-amber-500/10">
              <AlertTriangle className="h-5 w-5 text-amber-500" />
            </div>
            <div>
              <CardTitle className="text-base font-semibold">Predictive Alerts</CardTitle>
              <p className="text-xs text-[var(--color-muted-foreground)]">
                Proactive warnings and insights
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {criticalCount > 0 && (
              <Badge className="bg-rose-500 text-white">{criticalCount} Critical</Badge>
            )}
            {warningCount > 0 && (
              <Badge className="bg-amber-500 text-white">{warningCount} Warning</Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Loading State */}
        {isLoading && (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-24 bg-[var(--color-muted)] animate-pulse rounded-lg" />
            ))}
          </div>
        )}

        {/* Alert Cards */}
        {!isLoading && (
          <>
            {alerts.map((alert) => (
              <AlertCard key={alert.id} alert={alert} />
            ))}

            {/* View All Button */}
            <Button variant="outline" className="w-full mt-2">
              <span>View All Alerts</span>
              <ExternalLink className="h-4 w-4 ml-2" />
            </Button>
          </>
        )}
      </CardContent>
    </Card>
  );
}

export default PredictiveAlerts;
