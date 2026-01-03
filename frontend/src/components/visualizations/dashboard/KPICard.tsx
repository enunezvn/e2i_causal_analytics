/**
 * KPI Card Component
 * ==================
 *
 * Dashboard card for displaying key performance indicators with
 * trend indicators, sparklines, and status.
 *
 * @module components/visualizations/dashboard/KPICard
 */

import * as React from 'react';
import { useMemo } from 'react';
import { TrendingUp, TrendingDown, Minus, ArrowRight, Info } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer } from 'recharts';
import { cn } from '@/lib/utils';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

// =============================================================================
// TYPES
// =============================================================================

export type KPIStatus = 'healthy' | 'warning' | 'critical' | 'neutral';

export interface KPICardProps {
  /** KPI title/name */
  title: string;
  /** Current value */
  value: number | string;
  /** Optional unit (e.g., '%', '$', 'K') */
  unit?: string;
  /** Prefix for value (e.g., '$') */
  prefix?: string;
  /** Previous period value for comparison */
  previousValue?: number;
  /** Target value */
  target?: number;
  /** Historical data for sparkline */
  sparklineData?: number[];
  /** KPI status */
  status?: KPIStatus;
  /** Optional description/tooltip */
  description?: string;
  /** Whether data is loading */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Callback when card is clicked */
  onClick?: () => void;
  /** Whether to show the target indicator */
  showTarget?: boolean;
  /** Whether higher is better (affects trend color) */
  higherIsBetter?: boolean;
  /** Card size variant */
  size?: 'sm' | 'md' | 'lg';
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_SPARKLINE = [45, 52, 48, 55, 60, 58, 62, 65, 63, 68];

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function getStatusColor(status: KPIStatus): string {
  switch (status) {
    case 'healthy':
      return 'bg-emerald-500';
    case 'warning':
      return 'bg-amber-500';
    case 'critical':
      return 'bg-rose-500';
    default:
      return 'bg-gray-400';
  }
}

function getStatusBorderColor(status: KPIStatus): string {
  switch (status) {
    case 'healthy':
      return 'border-l-emerald-500';
    case 'warning':
      return 'border-l-amber-500';
    case 'critical':
      return 'border-l-rose-500';
    default:
      return 'border-l-gray-400';
  }
}

function formatValue(value: number | string, prefix?: string, unit?: string): string {
  if (typeof value === 'string') return `${prefix || ''}${value}${unit || ''}`;

  let formatted: string;
  if (Math.abs(value) >= 1000000) {
    formatted = `${(value / 1000000).toFixed(1)}M`;
  } else if (Math.abs(value) >= 1000) {
    formatted = `${(value / 1000).toFixed(1)}K`;
  } else if (Number.isInteger(value)) {
    formatted = value.toString();
  } else {
    formatted = value.toFixed(2);
  }

  return `${prefix || ''}${formatted}${unit || ''}`;
}

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * KPICard displays a key performance indicator with trend and sparkline.
 *
 * @example
 * ```tsx
 * <KPICard
 *   title="Total Revenue"
 *   value={1250000}
 *   prefix="$"
 *   previousValue={1180000}
 *   target={1300000}
 *   sparklineData={revenueHistory}
 *   status="healthy"
 *   higherIsBetter
 * />
 * ```
 */
export const KPICard = React.forwardRef<HTMLDivElement, KPICardProps>(
  (
    {
      title,
      value,
      unit,
      prefix,
      previousValue,
      target,
      sparklineData: propSparkline,
      status = 'neutral',
      description,
      isLoading = false,
      className,
      onClick,
      showTarget = true,
      higherIsBetter = true,
      size = 'md',
    },
    ref
  ) => {
    // Use provided sparkline or sample
    const sparklineData = propSparkline ?? SAMPLE_SPARKLINE;

    // Calculate trend
    const trend = useMemo(() => {
      if (previousValue === undefined || typeof value !== 'number') {
        return null;
      }

      const change = value - previousValue;
      const changePercent = previousValue !== 0 ? (change / previousValue) * 100 : 0;
      const direction: 'up' | 'down' | 'stable' =
        Math.abs(changePercent) < 0.5 ? 'stable' : change > 0 ? 'up' : 'down';

      const isPositive = higherIsBetter ? change >= 0 : change <= 0;

      return {
        direction,
        changePercent,
        isPositive,
      };
    }, [value, previousValue, higherIsBetter]);

    // Calculate target progress
    const targetProgress = useMemo(() => {
      if (target === undefined || typeof value !== 'number') {
        return null;
      }

      const progress = (value / target) * 100;
      const isOnTrack = higherIsBetter ? value >= target : value <= target;

      return {
        progress: Math.min(100, progress),
        isOnTrack,
      };
    }, [value, target, higherIsBetter]);

    // Get trend icon
    const TrendIcon = trend?.direction === 'up' ? TrendingUp : trend?.direction === 'down' ? TrendingDown : Minus;

    // Size styles
    const sizeStyles = {
      sm: {
        card: 'p-3',
        title: 'text-xs',
        value: 'text-xl',
        sparkHeight: 30,
      },
      md: {
        card: 'p-4',
        title: 'text-sm',
        value: 'text-2xl',
        sparkHeight: 40,
      },
      lg: {
        card: 'p-5',
        title: 'text-base',
        value: 'text-3xl',
        sparkHeight: 50,
      },
    }[size];

    // Loading skeleton
    if (isLoading) {
      return (
        <div
          ref={ref}
          className={cn(
            'animate-pulse bg-[var(--color-card)] rounded-lg border border-[var(--color-border)]',
            sizeStyles.card,
            className
          )}
        >
          <div className="h-4 w-24 bg-[var(--color-muted)] rounded mb-2" />
          <div className="h-8 w-32 bg-[var(--color-muted)] rounded mb-3" />
          <div className="h-10 bg-[var(--color-muted)] rounded" />
        </div>
      );
    }

    return (
      <div
        ref={ref}
        className={cn(
          'bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] border-l-4',
          getStatusBorderColor(status),
          sizeStyles.card,
          onClick && 'cursor-pointer hover:shadow-md transition-shadow',
          className
        )}
        onClick={onClick}
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-2 mb-1">
          <h3 className={cn('font-medium text-[var(--color-muted-foreground)]', sizeStyles.title)}>
            {title}
          </h3>
          <div className="flex items-center gap-1">
            {/* Status indicator */}
            <div className={cn('w-2 h-2 rounded-full', getStatusColor(status))} />
            {/* Info tooltip */}
            {description && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Info className="h-3.5 w-3.5 text-[var(--color-muted-foreground)] cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs text-sm">{description}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
          </div>
        </div>

        {/* Value and trend */}
        <div className="flex items-baseline gap-2 mb-2">
          <span className={cn('font-bold text-[var(--color-foreground)]', sizeStyles.value)}>
            {formatValue(value, prefix, unit)}
          </span>
          {trend && (
            <div
              className={cn(
                'flex items-center gap-0.5 text-sm',
                trend.isPositive ? 'text-emerald-500' : 'text-rose-500'
              )}
            >
              <TrendIcon className="h-3.5 w-3.5" />
              <span>
                {trend.changePercent > 0 ? '+' : ''}
                {trend.changePercent.toFixed(1)}%
              </span>
            </div>
          )}
        </div>

        {/* Target progress */}
        {showTarget && target !== undefined && targetProgress && (
          <div className="mb-2">
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-[var(--color-muted-foreground)]">
                Target: {formatValue(target, prefix, unit)}
              </span>
              <span
                className={cn(
                  'font-medium',
                  targetProgress.isOnTrack ? 'text-emerald-500' : 'text-amber-500'
                )}
              >
                {targetProgress.progress.toFixed(0)}%
              </span>
            </div>
            <div className="h-1.5 bg-[var(--color-muted)] rounded-full overflow-hidden">
              <div
                className={cn(
                  'h-full rounded-full transition-all',
                  targetProgress.isOnTrack ? 'bg-emerald-500' : 'bg-amber-500'
                )}
                style={{ width: `${targetProgress.progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Sparkline */}
        {sparklineData.length > 0 && (
          <div className="mt-2">
            <ResponsiveContainer width="100%" height={sizeStyles.sparkHeight}>
              <LineChart data={sparklineData.map((v, i) => ({ value: v, index: i }))}>
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke={status === 'critical' ? '#ef4444' : status === 'warning' ? '#f59e0b' : '#10b981'}
                  strokeWidth={1.5}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Click indicator */}
        {onClick && (
          <div className="flex items-center justify-end mt-2 text-xs text-[var(--color-muted-foreground)]">
            <span>View details</span>
            <ArrowRight className="h-3 w-3 ml-1" />
          </div>
        )}
      </div>
    );
  }
);

KPICard.displayName = 'KPICard';

export default KPICard;
