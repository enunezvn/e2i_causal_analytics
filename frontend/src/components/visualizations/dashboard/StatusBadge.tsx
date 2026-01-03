/**
 * Status Badge Component
 * ======================
 *
 * Badge component for displaying system/component status with
 * consistent styling and optional pulse animation.
 *
 * @module components/visualizations/dashboard/StatusBadge
 */

import * as React from 'react';
import { CheckCircle2, AlertCircle, XCircle, Clock, Loader2, HelpCircle, Pause, Play } from 'lucide-react';
import { cn } from '@/lib/utils';

// =============================================================================
// TYPES
// =============================================================================

export type StatusType =
  | 'healthy'
  | 'success'
  | 'warning'
  | 'error'
  | 'critical'
  | 'pending'
  | 'loading'
  | 'paused'
  | 'active'
  | 'inactive'
  | 'unknown';

export interface StatusBadgeProps {
  /** Status type */
  status: StatusType;
  /** Optional custom label (defaults to status name) */
  label?: string;
  /** Whether to show the status icon */
  showIcon?: boolean;
  /** Whether to show pulse animation for active statuses */
  pulse?: boolean;
  /** Badge size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Badge style variant */
  variant?: 'solid' | 'outline' | 'subtle';
  /** Additional CSS classes */
  className?: string;
  /** Optional tooltip text */
  tooltip?: string;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

interface StatusConfig {
  icon: React.ElementType;
  label: string;
  bgColor: string;
  textColor: string;
  borderColor: string;
  dotColor: string;
}

function getStatusConfig(status: StatusType): StatusConfig {
  switch (status) {
    case 'healthy':
    case 'success':
      return {
        icon: CheckCircle2,
        label: status === 'healthy' ? 'Healthy' : 'Success',
        bgColor: 'bg-emerald-100 dark:bg-emerald-900/30',
        textColor: 'text-emerald-700 dark:text-emerald-400',
        borderColor: 'border-emerald-300 dark:border-emerald-700',
        dotColor: 'bg-emerald-500',
      };
    case 'warning':
      return {
        icon: AlertCircle,
        label: 'Warning',
        bgColor: 'bg-amber-100 dark:bg-amber-900/30',
        textColor: 'text-amber-700 dark:text-amber-400',
        borderColor: 'border-amber-300 dark:border-amber-700',
        dotColor: 'bg-amber-500',
      };
    case 'error':
    case 'critical':
      return {
        icon: XCircle,
        label: status === 'error' ? 'Error' : 'Critical',
        bgColor: 'bg-rose-100 dark:bg-rose-900/30',
        textColor: 'text-rose-700 dark:text-rose-400',
        borderColor: 'border-rose-300 dark:border-rose-700',
        dotColor: 'bg-rose-500',
      };
    case 'pending':
      return {
        icon: Clock,
        label: 'Pending',
        bgColor: 'bg-blue-100 dark:bg-blue-900/30',
        textColor: 'text-blue-700 dark:text-blue-400',
        borderColor: 'border-blue-300 dark:border-blue-700',
        dotColor: 'bg-blue-500',
      };
    case 'loading':
      return {
        icon: Loader2,
        label: 'Loading',
        bgColor: 'bg-sky-100 dark:bg-sky-900/30',
        textColor: 'text-sky-700 dark:text-sky-400',
        borderColor: 'border-sky-300 dark:border-sky-700',
        dotColor: 'bg-sky-500',
      };
    case 'paused':
      return {
        icon: Pause,
        label: 'Paused',
        bgColor: 'bg-slate-100 dark:bg-slate-900/30',
        textColor: 'text-slate-700 dark:text-slate-400',
        borderColor: 'border-slate-300 dark:border-slate-700',
        dotColor: 'bg-slate-500',
      };
    case 'active':
      return {
        icon: Play,
        label: 'Active',
        bgColor: 'bg-emerald-100 dark:bg-emerald-900/30',
        textColor: 'text-emerald-700 dark:text-emerald-400',
        borderColor: 'border-emerald-300 dark:border-emerald-700',
        dotColor: 'bg-emerald-500',
      };
    case 'inactive':
      return {
        icon: Pause,
        label: 'Inactive',
        bgColor: 'bg-gray-100 dark:bg-gray-900/30',
        textColor: 'text-gray-700 dark:text-gray-400',
        borderColor: 'border-gray-300 dark:border-gray-700',
        dotColor: 'bg-gray-400',
      };
    case 'unknown':
    default:
      return {
        icon: HelpCircle,
        label: 'Unknown',
        bgColor: 'bg-gray-100 dark:bg-gray-900/30',
        textColor: 'text-gray-700 dark:text-gray-400',
        borderColor: 'border-gray-300 dark:border-gray-700',
        dotColor: 'bg-gray-400',
      };
  }
}

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * StatusBadge displays a status indicator with icon and label.
 *
 * @example
 * ```tsx
 * <StatusBadge status="healthy" />
 * <StatusBadge status="warning" label="Drift Detected" pulse />
 * <StatusBadge status="loading" size="sm" />
 * ```
 */
export const StatusBadge = React.forwardRef<HTMLSpanElement, StatusBadgeProps>(
  (
    {
      status,
      label: customLabel,
      showIcon = true,
      pulse = false,
      size = 'md',
      variant = 'subtle',
      className,
      tooltip,
    },
    ref
  ) => {
    const config = getStatusConfig(status);
    const Icon = config.icon;
    const displayLabel = customLabel ?? config.label;

    // Size styles
    const sizeStyles = {
      sm: {
        badge: 'px-1.5 py-0.5 text-xs gap-1',
        icon: 'h-3 w-3',
        dot: 'h-1.5 w-1.5',
      },
      md: {
        badge: 'px-2 py-1 text-sm gap-1.5',
        icon: 'h-3.5 w-3.5',
        dot: 'h-2 w-2',
      },
      lg: {
        badge: 'px-3 py-1.5 text-base gap-2',
        icon: 'h-4 w-4',
        dot: 'h-2.5 w-2.5',
      },
    }[size];

    // Variant styles
    const variantStyles = {
      solid: cn(config.bgColor, config.textColor),
      outline: cn('bg-transparent border', config.borderColor, config.textColor),
      subtle: cn(config.bgColor, config.textColor),
    }[variant];

    // Should pulse for certain statuses
    const shouldPulse = pulse || status === 'loading' || status === 'pending';

    return (
      <span
        ref={ref}
        className={cn(
          'inline-flex items-center rounded-full font-medium',
          sizeStyles.badge,
          variantStyles,
          className
        )}
        title={tooltip}
      >
        {showIcon && (
          <span className="relative flex items-center">
            {/* Pulse animation */}
            {shouldPulse && (
              <span
                className={cn(
                  'absolute inline-flex h-full w-full rounded-full opacity-75 animate-ping',
                  config.dotColor
                )}
              />
            )}
            <Icon
              className={cn(
                sizeStyles.icon,
                status === 'loading' && 'animate-spin'
              )}
            />
          </span>
        )}
        <span>{displayLabel}</span>
      </span>
    );
  }
);

StatusBadge.displayName = 'StatusBadge';

// =============================================================================
// STATUS DOT (Minimal variant)
// =============================================================================

export interface StatusDotProps {
  /** Status type */
  status: StatusType;
  /** Whether to show pulse animation */
  pulse?: boolean;
  /** Dot size */
  size?: 'sm' | 'md' | 'lg';
  /** Additional CSS classes */
  className?: string;
  /** Optional tooltip text */
  tooltip?: string;
}

/**
 * StatusDot is a minimal status indicator (just a dot).
 *
 * @example
 * ```tsx
 * <StatusDot status="healthy" />
 * <StatusDot status="critical" pulse />
 * ```
 */
export const StatusDot = React.forwardRef<HTMLSpanElement, StatusDotProps>(
  ({ status, pulse = false, size = 'md', className, tooltip }, ref) => {
    const config = getStatusConfig(status);

    const sizeStyles = {
      sm: 'h-1.5 w-1.5',
      md: 'h-2 w-2',
      lg: 'h-3 w-3',
    }[size];

    const shouldPulse = pulse || status === 'loading' || status === 'pending';

    return (
      <span
        ref={ref}
        className={cn('relative inline-flex', className)}
        title={tooltip || config.label}
      >
        {shouldPulse && (
          <span
            className={cn(
              'absolute inline-flex rounded-full opacity-75 animate-ping',
              sizeStyles,
              config.dotColor
            )}
          />
        )}
        <span
          className={cn(
            'relative inline-flex rounded-full',
            sizeStyles,
            config.dotColor
          )}
        />
      </span>
    );
  }
);

StatusDot.displayName = 'StatusDot';

export default StatusBadge;
