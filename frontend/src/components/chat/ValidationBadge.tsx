/**
 * Validation Badge Component
 * ==========================
 *
 * Displays validation status badges for AI responses and actions.
 * Used to indicate confidence levels and required user review.
 *
 * Features:
 * - Three states: PROCEED, REVIEW, BLOCK
 * - Color-coded indicators
 * - Optional confidence percentage
 * - Tooltip explanations
 *
 * @module components/chat/ValidationBadge
 */

import * as React from 'react';
import { CheckCircle, AlertTriangle, XCircle, Info } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

// =============================================================================
// TYPES
// =============================================================================

export type ValidationStatus = 'proceed' | 'review' | 'block';

export interface ValidationBadgeProps {
  /** Validation status */
  status: ValidationStatus;
  /** Optional confidence score (0-100) */
  confidence?: number;
  /** Short label override */
  label?: string;
  /** Detailed explanation for tooltip */
  explanation?: string;
  /** Size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Additional CSS classes */
  className?: string;
  /** Show confidence percentage */
  showConfidence?: boolean;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function getStatusConfig(status: ValidationStatus) {
  switch (status) {
    case 'proceed':
      return {
        icon: CheckCircle,
        label: 'PROCEED',
        color: 'text-emerald-600 dark:text-emerald-400',
        bgColor: 'bg-emerald-100 dark:bg-emerald-900/30',
        borderColor: 'border-emerald-200 dark:border-emerald-800',
        description: 'Action is validated and safe to proceed',
      };
    case 'review':
      return {
        icon: AlertTriangle,
        label: 'REVIEW',
        color: 'text-amber-600 dark:text-amber-400',
        bgColor: 'bg-amber-100 dark:bg-amber-900/30',
        borderColor: 'border-amber-200 dark:border-amber-800',
        description: 'Requires human review before proceeding',
      };
    case 'block':
      return {
        icon: XCircle,
        label: 'BLOCK',
        color: 'text-rose-600 dark:text-rose-400',
        bgColor: 'bg-rose-100 dark:bg-rose-900/30',
        borderColor: 'border-rose-200 dark:border-rose-800',
        description: 'Action blocked due to validation failure',
      };
  }
}

function getSizeClasses(size: 'sm' | 'md' | 'lg') {
  switch (size) {
    case 'sm':
      return {
        badge: 'px-1.5 py-0.5 text-[10px]',
        icon: 'h-3 w-3',
        gap: 'gap-1',
      };
    case 'md':
      return {
        badge: 'px-2 py-1 text-xs',
        icon: 'h-3.5 w-3.5',
        gap: 'gap-1.5',
      };
    case 'lg':
      return {
        badge: 'px-3 py-1.5 text-sm',
        icon: 'h-4 w-4',
        gap: 'gap-2',
      };
  }
}

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * ValidationBadge displays a status badge for AI validation results.
 *
 * @example
 * ```tsx
 * <ValidationBadge status="proceed" confidence={95} />
 * <ValidationBadge status="review" explanation="Requires analyst approval" />
 * <ValidationBadge status="block" label="DENIED" />
 * ```
 */
export function ValidationBadge({
  status,
  confidence,
  label,
  explanation,
  size = 'md',
  className,
  showConfidence = true,
}: ValidationBadgeProps) {
  const config = getStatusConfig(status);
  const sizeClasses = getSizeClasses(size);
  const StatusIcon = config.icon;

  const tooltipContent = explanation || config.description;
  const displayLabel = label || config.label;

  const badgeContent = (
    <Badge
      variant="outline"
      className={cn(
        'font-semibold uppercase tracking-wide border',
        config.bgColor,
        config.color,
        config.borderColor,
        sizeClasses.badge,
        sizeClasses.gap,
        'flex items-center',
        className
      )}
    >
      <StatusIcon className={sizeClasses.icon} />
      <span>{displayLabel}</span>
      {showConfidence && confidence !== undefined && (
        <span className="opacity-70">({confidence}%)</span>
      )}
    </Badge>
  );

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>{badgeContent}</TooltipTrigger>
        <TooltipContent side="top" className="max-w-[200px]">
          <div className="flex items-start gap-2">
            <Info className="h-3.5 w-3.5 mt-0.5 shrink-0" />
            <p className="text-xs">{tooltipContent}</p>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// =============================================================================
// PRESET COMPONENTS
// =============================================================================

/**
 * Quick preset for PROCEED status
 */
export function ProceedBadge(props: Omit<ValidationBadgeProps, 'status'>) {
  return <ValidationBadge status="proceed" {...props} />;
}

/**
 * Quick preset for REVIEW status
 */
export function ReviewBadge(props: Omit<ValidationBadgeProps, 'status'>) {
  return <ValidationBadge status="review" {...props} />;
}

/**
 * Quick preset for BLOCK status
 */
export function BlockBadge(props: Omit<ValidationBadgeProps, 'status'>) {
  return <ValidationBadge status="block" {...props} />;
}

export default ValidationBadge;
