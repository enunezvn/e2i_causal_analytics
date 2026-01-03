/**
 * Progress Ring Component
 * =======================
 *
 * Circular progress indicator with customizable appearance.
 * Shows percentage completion with optional center content.
 *
 * @module components/visualizations/dashboard/ProgressRing
 */

import * as React from 'react';
import { useMemo } from 'react';
import { cn } from '@/lib/utils';

// =============================================================================
// TYPES
// =============================================================================

export interface ProgressRingProps {
  /** Progress value (0-100) */
  value: number;
  /** Maximum value (default: 100) */
  max?: number;
  /** Ring size in pixels */
  size?: number;
  /** Stroke width in pixels */
  strokeWidth?: number;
  /** Color of the progress arc */
  color?: string;
  /** Color of the background arc */
  trackColor?: string;
  /** Whether to show the percentage label */
  showLabel?: boolean;
  /** Custom label (overrides percentage) */
  label?: React.ReactNode;
  /** Whether to animate the progress */
  animated?: boolean;
  /** Status-based coloring */
  status?: 'healthy' | 'warning' | 'critical' | 'neutral';
  /** Whether data is loading */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Center content (replaces default label) */
  children?: React.ReactNode;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function getStatusColor(status: string): string {
  switch (status) {
    case 'healthy':
      return '#10b981'; // Emerald-500
    case 'warning':
      return '#f59e0b'; // Amber-500
    case 'critical':
      return '#ef4444'; // Rose-500
    default:
      return 'hsl(var(--chart-1))';
  }
}

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * ProgressRing displays a circular progress indicator.
 *
 * @example
 * ```tsx
 * <ProgressRing value={75} status="healthy" />
 * <ProgressRing value={45} color="#3b82f6">
 *   <span className="text-lg font-bold">45%</span>
 * </ProgressRing>
 * ```
 */
export const ProgressRing = React.forwardRef<HTMLDivElement, ProgressRingProps>(
  (
    {
      value,
      max = 100,
      size = 80,
      strokeWidth = 8,
      color: propColor,
      trackColor = 'var(--color-muted)',
      showLabel = true,
      label,
      animated = true,
      status = 'neutral',
      isLoading = false,
      className,
      children,
    },
    ref
  ) => {
    // Calculate progress
    const normalizedValue = useMemo(() => {
      const clamped = Math.min(Math.max(value, 0), max);
      return (clamped / max) * 100;
    }, [value, max]);

    // Calculate SVG dimensions
    const center = size / 2;
    const radius = center - strokeWidth / 2;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (normalizedValue / 100) * circumference;

    // Get color based on status or prop
    const color = propColor ?? getStatusColor(status);

    // Loading skeleton
    if (isLoading) {
      return (
        <div
          ref={ref}
          className={cn('animate-pulse', className)}
          style={{ width: size, height: size }}
        >
          <div
            className="rounded-full bg-[var(--color-muted)]"
            style={{ width: size, height: size }}
          />
        </div>
      );
    }

    return (
      <div
        ref={ref}
        className={cn('relative inline-flex items-center justify-center', className)}
        style={{ width: size, height: size }}
      >
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          className="transform -rotate-90"
        >
          {/* Background track */}
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke={trackColor}
            strokeWidth={strokeWidth}
          />
          {/* Progress arc */}
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className={cn(animated && 'transition-all duration-500 ease-out')}
          />
        </svg>

        {/* Center content */}
        <div className="absolute inset-0 flex items-center justify-center">
          {children ?? (
            showLabel && (
              <span className="text-sm font-semibold text-[var(--color-foreground)]">
                {label ?? `${Math.round(normalizedValue)}%`}
              </span>
            )
          )}
        </div>
      </div>
    );
  }
);

ProgressRing.displayName = 'ProgressRing';

// =============================================================================
// PROGRESS RING GROUP
// =============================================================================

export interface ProgressRingGroupProps {
  /** Array of progress items */
  items: Array<{
    label: string;
    value: number;
    max?: number;
    color?: string;
    status?: 'healthy' | 'warning' | 'critical' | 'neutral';
  }>;
  /** Ring size */
  size?: number;
  /** Gap between items */
  gap?: number;
  /** Whether data is loading */
  isLoading?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * ProgressRingGroup displays multiple progress rings with labels.
 *
 * @example
 * ```tsx
 * <ProgressRingGroup
 *   items={[
 *     { label: 'Accuracy', value: 92, status: 'healthy' },
 *     { label: 'Precision', value: 88, status: 'healthy' },
 *     { label: 'Recall', value: 75, status: 'warning' },
 *   ]}
 * />
 * ```
 */
export const ProgressRingGroup = React.forwardRef<HTMLDivElement, ProgressRingGroupProps>(
  ({ items, size = 60, gap = 16, isLoading = false, className }, ref) => {
    return (
      <div
        ref={ref}
        className={cn('flex flex-wrap', className)}
        style={{ gap }}
      >
        {items.map((item, index) => (
          <div key={index} className="flex flex-col items-center gap-1">
            <ProgressRing
              value={item.value}
              max={item.max}
              size={size}
              color={item.color}
              status={item.status}
              isLoading={isLoading}
            />
            <span className="text-xs text-[var(--color-muted-foreground)] text-center">
              {item.label}
            </span>
          </div>
        ))}
      </div>
    );
  }
);

ProgressRingGroup.displayName = 'ProgressRingGroup';

export default ProgressRing;
