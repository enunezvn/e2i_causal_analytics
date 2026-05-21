/**
 * EmptyState Component
 * ====================
 *
 * Generic empty-state placeholder used in place of hardcoded SAMPLE_*
 * data when an API hook returns `undefined`. Surfaces the absence of
 * real data instead of silently displaying fabricated plausible values.
 *
 * @module components/ui/EmptyState
 */

import * as React from 'react';
import { Info } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface EmptyStateProps {
  /** Primary message (default: "No data available") */
  title?: string;
  /** Optional secondary description / call-to-action */
  description?: string;
  /** Optional action button or other element rendered below the description */
  action?: React.ReactNode;
  /** Optional icon override (default: Info) */
  icon?: React.ReactNode;
  /** Additional CSS classes for the container */
  className?: string;
}

/**
 * Renders an empty-state placeholder.
 *
 * Use this when an API hook returns `undefined` and you would otherwise
 * have fallen back to hardcoded SAMPLE_ data.
 *
 * @example
 * ```tsx
 * {data ? <Results data={data} /> : (
 *   <EmptyState
 *     title="No analysis available"
 *     description="Run an analysis to see results."
 *   />
 * )}
 * ```
 */
export function EmptyState({
  title = 'No data available',
  description,
  action,
  icon,
  className,
}: EmptyStateProps): React.ReactElement {
  return (
    <div
      role="status"
      data-testid="empty-state"
      className={cn(
        'flex flex-col items-center justify-center rounded-lg border border-dashed',
        'border-[var(--color-border)] bg-[var(--color-muted)]/30 p-8 text-center',
        className,
      )}
    >
      <div className="mb-3 text-[var(--color-muted-foreground)]">
        {icon ?? <Info className="h-8 w-8" aria-hidden="true" />}
      </div>
      <h3 className="mb-1 text-base font-medium text-[var(--color-foreground)]">
        {title}
      </h3>
      {description && (
        <p className="mb-4 max-w-md text-sm text-[var(--color-muted-foreground)]">
          {description}
        </p>
      )}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

export default EmptyState;
