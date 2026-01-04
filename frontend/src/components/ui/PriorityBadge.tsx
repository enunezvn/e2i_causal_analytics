/**
 * PriorityBadge Component
 * =======================
 *
 * Displays priority indicators for alerts and notifications.
 * Uses color-coded badges to indicate priority levels.
 */

import * as React from 'react';
import { AlertTriangle, AlertCircle, Info, Bell } from 'lucide-react';
import { cn } from '@/lib/utils';

export type Priority = 'critical' | 'high' | 'medium' | 'low' | 'info';

interface PriorityBadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  priority: Priority;
  showIcon?: boolean;
  showLabel?: boolean;
  size?: 'sm' | 'md' | 'lg';
  pulsing?: boolean;
}

const priorityConfig: Record<Priority, { label: string; icon: React.ElementType; bgClass: string; textClass: string; borderClass: string }> = {
  critical: {
    label: 'Critical',
    icon: AlertCircle,
    bgClass: 'bg-red-100 dark:bg-red-900/30',
    textClass: 'text-red-800 dark:text-red-400',
    borderClass: 'border-red-200 dark:border-red-800',
  },
  high: {
    label: 'High',
    icon: AlertTriangle,
    bgClass: 'bg-orange-100 dark:bg-orange-900/30',
    textClass: 'text-orange-800 dark:text-orange-400',
    borderClass: 'border-orange-200 dark:border-orange-800',
  },
  medium: {
    label: 'Medium',
    icon: Bell,
    bgClass: 'bg-yellow-100 dark:bg-yellow-900/30',
    textClass: 'text-yellow-800 dark:text-yellow-400',
    borderClass: 'border-yellow-200 dark:border-yellow-800',
  },
  low: {
    label: 'Low',
    icon: Info,
    bgClass: 'bg-blue-100 dark:bg-blue-900/30',
    textClass: 'text-blue-800 dark:text-blue-400',
    borderClass: 'border-blue-200 dark:border-blue-800',
  },
  info: {
    label: 'Info',
    icon: Info,
    bgClass: 'bg-gray-100 dark:bg-gray-800/30',
    textClass: 'text-gray-800 dark:text-gray-400',
    borderClass: 'border-gray-200 dark:border-gray-700',
  },
};

const sizeConfig = {
  sm: {
    padding: 'px-1.5 py-0.5',
    text: 'text-xs',
    iconSize: 'h-3 w-3',
    gap: 'gap-1',
  },
  md: {
    padding: 'px-2 py-1',
    text: 'text-xs',
    iconSize: 'h-3.5 w-3.5',
    gap: 'gap-1.5',
  },
  lg: {
    padding: 'px-2.5 py-1.5',
    text: 'text-sm',
    iconSize: 'h-4 w-4',
    gap: 'gap-2',
  },
};

export function PriorityBadge({
  priority,
  showIcon = true,
  showLabel = true,
  size = 'md',
  pulsing = false,
  className,
  ...props
}: PriorityBadgeProps) {
  const config = priorityConfig[priority];
  const sizeStyles = sizeConfig[size];
  const Icon = config.icon;

  return (
    <div
      className={cn(
        'inline-flex items-center rounded-full border font-medium transition-colors',
        sizeStyles.padding,
        sizeStyles.text,
        sizeStyles.gap,
        config.bgClass,
        config.textClass,
        config.borderClass,
        pulsing && priority === 'critical' && 'animate-pulse',
        className
      )}
      {...props}
    >
      {showIcon && <Icon className={sizeStyles.iconSize} />}
      {showLabel && <span>{config.label}</span>}
    </div>
  );
}

// Dot-only variant for compact displays
export function PriorityDot({ priority, className }: { priority: Priority; className?: string }) {
  const dotColors: Record<Priority, string> = {
    critical: 'bg-red-500',
    high: 'bg-orange-500',
    medium: 'bg-yellow-500',
    low: 'bg-blue-500',
    info: 'bg-gray-500',
  };

  return (
    <span
      className={cn(
        'inline-block h-2 w-2 rounded-full',
        dotColors[priority],
        priority === 'critical' && 'animate-pulse',
        className
      )}
      aria-label={`Priority: ${priority}`}
    />
  );
}

export default PriorityBadge;
