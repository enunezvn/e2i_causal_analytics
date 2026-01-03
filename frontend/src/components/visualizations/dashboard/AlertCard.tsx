/**
 * Alert Card Component
 * ====================
 *
 * Card component for displaying alerts, notifications, and system messages
 * with severity levels and optional actions.
 *
 * @module components/visualizations/dashboard/AlertCard
 */

import * as React from 'react';
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Info,
  X,
  ArrowRight,
  Clock,
  Bell,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';

// =============================================================================
// TYPES
// =============================================================================

export type AlertSeverity = 'info' | 'success' | 'warning' | 'error' | 'critical';

export interface AlertAction {
  /** Action label */
  label: string;
  /** Action callback */
  onClick: () => void;
  /** Whether this is the primary action */
  primary?: boolean;
}

export interface AlertCardProps {
  /** Alert severity level */
  severity: AlertSeverity;
  /** Alert title */
  title: string;
  /** Alert message/description */
  message?: string;
  /** Timestamp of the alert */
  timestamp?: Date | string;
  /** Source/component that generated the alert */
  source?: string;
  /** Available actions */
  actions?: AlertAction[];
  /** Whether the alert can be dismissed */
  dismissible?: boolean;
  /** Callback when alert is dismissed */
  onDismiss?: () => void;
  /** Whether the alert is unread/new */
  isNew?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Compact mode for lists */
  compact?: boolean;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

interface SeverityConfig {
  icon: React.ElementType;
  bgColor: string;
  borderColor: string;
  iconColor: string;
  titleColor: string;
}

function getSeverityConfig(severity: AlertSeverity): SeverityConfig {
  switch (severity) {
    case 'info':
      return {
        icon: Info,
        bgColor: 'bg-blue-50 dark:bg-blue-900/20',
        borderColor: 'border-blue-200 dark:border-blue-800',
        iconColor: 'text-blue-500',
        titleColor: 'text-blue-700 dark:text-blue-400',
      };
    case 'success':
      return {
        icon: CheckCircle2,
        bgColor: 'bg-emerald-50 dark:bg-emerald-900/20',
        borderColor: 'border-emerald-200 dark:border-emerald-800',
        iconColor: 'text-emerald-500',
        titleColor: 'text-emerald-700 dark:text-emerald-400',
      };
    case 'warning':
      return {
        icon: AlertTriangle,
        bgColor: 'bg-amber-50 dark:bg-amber-900/20',
        borderColor: 'border-amber-200 dark:border-amber-800',
        iconColor: 'text-amber-500',
        titleColor: 'text-amber-700 dark:text-amber-400',
      };
    case 'error':
      return {
        icon: AlertCircle,
        bgColor: 'bg-rose-50 dark:bg-rose-900/20',
        borderColor: 'border-rose-200 dark:border-rose-800',
        iconColor: 'text-rose-500',
        titleColor: 'text-rose-700 dark:text-rose-400',
      };
    case 'critical':
      return {
        icon: AlertCircle,
        bgColor: 'bg-red-100 dark:bg-red-900/30',
        borderColor: 'border-red-300 dark:border-red-700',
        iconColor: 'text-red-600',
        titleColor: 'text-red-800 dark:text-red-300',
      };
    default:
      return getSeverityConfig('info');
  }
}

function formatTimestamp(timestamp: Date | string): string {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  const now = new Date();
  const diff = now.getTime() - date.getTime();

  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;

  return date.toLocaleDateString();
}

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * AlertCard displays an alert with severity, message, and optional actions.
 *
 * @example
 * ```tsx
 * <AlertCard
 *   severity="warning"
 *   title="Model Drift Detected"
 *   message="Churn model accuracy has dropped below threshold."
 *   timestamp={new Date()}
 *   source="Drift Monitor"
 *   actions={[
 *     { label: 'Retrain Model', onClick: handleRetrain, primary: true },
 *     { label: 'View Details', onClick: handleViewDetails },
 *   ]}
 *   dismissible
 *   onDismiss={handleDismiss}
 * />
 * ```
 */
export const AlertCard = React.forwardRef<HTMLDivElement, AlertCardProps>(
  (
    {
      severity,
      title,
      message,
      timestamp,
      source,
      actions,
      dismissible = false,
      onDismiss,
      isNew = false,
      className,
      compact = false,
    },
    ref
  ) => {
    const config = getSeverityConfig(severity);
    const Icon = config.icon;

    if (compact) {
      return (
        <div
          ref={ref}
          className={cn(
            'flex items-center gap-3 py-2 px-3 rounded-md border',
            config.bgColor,
            config.borderColor,
            className
          )}
        >
          {/* Icon */}
          <Icon className={cn('h-4 w-4 flex-shrink-0', config.iconColor)} />

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className={cn('text-sm font-medium truncate', config.titleColor)}>
                {title}
              </span>
              {isNew && (
                <span className="flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-2 w-2 rounded-full bg-blue-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500" />
                </span>
              )}
            </div>
            {timestamp && (
              <span className="text-xs text-[var(--color-muted-foreground)]">
                {formatTimestamp(timestamp)}
              </span>
            )}
          </div>

          {/* Actions */}
          {actions && actions.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={actions[0].onClick}
              className="flex-shrink-0"
            >
              <ArrowRight className="h-4 w-4" />
            </Button>
          )}

          {/* Dismiss */}
          {dismissible && (
            <button
              onClick={onDismiss}
              className="flex-shrink-0 p-1 rounded hover:bg-black/5 dark:hover:bg-white/5"
            >
              <X className="h-3.5 w-3.5 text-[var(--color-muted-foreground)]" />
            </button>
          )}
        </div>
      );
    }

    return (
      <div
        ref={ref}
        className={cn(
          'relative rounded-lg border p-4',
          config.bgColor,
          config.borderColor,
          className
        )}
      >
        {/* New indicator */}
        {isNew && (
          <span className="absolute -top-1 -right-1 flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-3 w-3 bg-blue-500" />
          </span>
        )}

        {/* Header */}
        <div className="flex items-start gap-3">
          {/* Icon */}
          <div className={cn('flex-shrink-0 mt-0.5', config.iconColor)}>
            <Icon className="h-5 w-5" />
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-start justify-between gap-2">
              <div>
                <h4 className={cn('font-semibold', config.titleColor)}>
                  {title}
                </h4>
                {source && (
                  <div className="flex items-center gap-1 mt-0.5 text-xs text-[var(--color-muted-foreground)]">
                    <Bell className="h-3 w-3" />
                    <span>{source}</span>
                  </div>
                )}
              </div>
              {dismissible && (
                <button
                  onClick={onDismiss}
                  className="flex-shrink-0 p-1 rounded hover:bg-black/5 dark:hover:bg-white/5"
                >
                  <X className="h-4 w-4 text-[var(--color-muted-foreground)]" />
                </button>
              )}
            </div>

            {/* Message */}
            {message && (
              <p className="mt-2 text-sm text-[var(--color-foreground)]">
                {message}
              </p>
            )}

            {/* Footer */}
            <div className="flex items-center justify-between mt-3">
              {/* Timestamp */}
              {timestamp && (
                <div className="flex items-center gap-1 text-xs text-[var(--color-muted-foreground)]">
                  <Clock className="h-3 w-3" />
                  <span>{formatTimestamp(timestamp)}</span>
                </div>
              )}

              {/* Actions */}
              {actions && actions.length > 0 && (
                <div className="flex items-center gap-2">
                  {actions.map((action, index) => (
                    <Button
                      key={index}
                      variant={action.primary ? 'default' : 'outline'}
                      size="sm"
                      onClick={action.onClick}
                    >
                      {action.label}
                    </Button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }
);

AlertCard.displayName = 'AlertCard';

// =============================================================================
// ALERT LIST
// =============================================================================

export interface AlertListProps {
  /** Array of alerts */
  alerts: Omit<AlertCardProps, 'compact'>[];
  /** Whether to use compact mode */
  compact?: boolean;
  /** Maximum number of alerts to show */
  maxItems?: number;
  /** Whether data is loading */
  isLoading?: boolean;
  /** Empty state message */
  emptyMessage?: string;
  /** Additional CSS classes */
  className?: string;
}

/**
 * AlertList displays a list of alerts.
 *
 * @example
 * ```tsx
 * <AlertList
 *   alerts={alertData}
 *   compact
 *   maxItems={5}
 * />
 * ```
 */
export const AlertList = React.forwardRef<HTMLDivElement, AlertListProps>(
  (
    {
      alerts,
      compact = false,
      maxItems,
      isLoading = false,
      emptyMessage = 'No alerts',
      className,
    },
    ref
  ) => {
    const displayAlerts = maxItems ? alerts.slice(0, maxItems) : alerts;

    if (isLoading) {
      return (
        <div ref={ref} className={cn('space-y-2', className)}>
          {[1, 2, 3].map((i) => (
            <div key={i} className="animate-pulse">
              <div className="h-16 bg-[var(--color-muted)] rounded-lg" />
            </div>
          ))}
        </div>
      );
    }

    if (alerts.length === 0) {
      return (
        <div
          ref={ref}
          className={cn(
            'flex items-center justify-center py-8 text-[var(--color-muted-foreground)]',
            className
          )}
        >
          <div className="text-center">
            <CheckCircle2 className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p>{emptyMessage}</p>
          </div>
        </div>
      );
    }

    return (
      <div ref={ref} className={cn('space-y-2', className)}>
        {displayAlerts.map((alert, index) => (
          <AlertCard key={index} {...alert} compact={compact} />
        ))}
        {maxItems && alerts.length > maxItems && (
          <p className="text-sm text-center text-[var(--color-muted-foreground)] py-2">
            +{alerts.length - maxItems} more alerts
          </p>
        )}
      </div>
    );
  }
);

AlertList.displayName = 'AlertList';

export default AlertCard;
