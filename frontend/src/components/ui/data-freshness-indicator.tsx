/**
 * Data Freshness Indicator Component
 * ===================================
 *
 * UI component showing when data was last updated and its freshness level.
 * Integrates with the useDataFreshness hook.
 *
 * Features:
 * - Color-coded freshness levels (fresh, aging, stale)
 * - Compact or detailed display modes
 * - Refresh button integration
 * - Tooltip with exact timestamp
 *
 * @module components/ui/data-freshness-indicator
 */

import * as React from 'react';
import { RefreshCw, CheckCircle, Clock, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import type { DataFreshnessResult } from '@/hooks/use-data-freshness';
import { getFreshnessClassName } from '@/hooks/use-data-freshness';

// =============================================================================
// TYPES
// =============================================================================

export interface DataFreshnessIndicatorProps extends Partial<DataFreshnessResult> {
  /** Whether to show the refresh button */
  showRefreshButton?: boolean;
  /** Callback when refresh button is clicked */
  onRefresh?: () => void;
  /** Whether refresh is in progress */
  isRefreshing?: boolean;
  /** Display mode: 'compact' shows just the icon, 'full' shows text */
  mode?: 'compact' | 'full';
  /** Additional CSS classes */
  className?: string;
  /** Prefix text (default: "Updated") */
  prefix?: string;
}

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

function FreshnessIcon({ level }: { level: 'fresh' | 'aging' | 'stale' }) {
  const iconClassName = 'h-3.5 w-3.5';

  switch (level) {
    case 'fresh':
      return <CheckCircle className={cn(iconClassName, 'text-emerald-600 dark:text-emerald-400')} />;
    case 'aging':
      return <Clock className={cn(iconClassName, 'text-amber-600 dark:text-amber-400')} />;
    case 'stale':
      return <AlertTriangle className={cn(iconClassName, 'text-rose-600 dark:text-rose-400')} />;
    default:
      return <Clock className={iconClassName} />;
  }
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

/**
 * DataFreshnessIndicator shows users when data was last updated.
 *
 * @example
 * ```tsx
 * const { data, dataUpdatedAt, refetch, isFetching } = useQuery(...);
 * const freshness = useDataFreshness(dataUpdatedAt);
 *
 * return (
 *   <DataFreshnessIndicator
 *     {...freshness}
 *     showRefreshButton
 *     onRefresh={refetch}
 *     isRefreshing={isFetching}
 *   />
 * );
 * ```
 */
export function DataFreshnessIndicator({
  lastUpdatedFormatted = 'Never updated',
  freshnessLevel = 'stale',
  dataUpdatedAt,
  isStale,
  showRefreshButton = false,
  onRefresh,
  isRefreshing = false,
  mode = 'full',
  className,
  prefix = 'Updated',
}: DataFreshnessIndicatorProps) {
  // Format exact timestamp for tooltip
  const exactTimestamp = React.useMemo(() => {
    if (!dataUpdatedAt) return 'Never';
    return new Date(dataUpdatedAt).toLocaleString();
  }, [dataUpdatedAt]);

  const content = (
    <div
      className={cn(
        'flex items-center gap-1.5 text-xs',
        getFreshnessClassName(freshnessLevel),
        className
      )}
    >
      <FreshnessIcon level={freshnessLevel} />
      {mode === 'full' && (
        <span>
          {prefix} {lastUpdatedFormatted}
        </span>
      )}
      {showRefreshButton && onRefresh && (
        <Button
          variant="ghost"
          size="icon"
          className="h-5 w-5 ml-1"
          onClick={onRefresh}
          disabled={isRefreshing}
          aria-label="Refresh data"
        >
          <RefreshCw
            className={cn('h-3 w-3', isRefreshing && 'animate-spin')}
          />
        </Button>
      )}
    </div>
  );

  // Wrap in tooltip for exact timestamp
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          {content}
        </TooltipTrigger>
        <TooltipContent side="bottom" className="text-xs">
          <p>Last updated: {exactTimestamp}</p>
          {isStale && (
            <p className="text-amber-500 mt-1">
              Data may be outdated. Click refresh to update.
            </p>
          )}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

/**
 * Compact variant showing only an icon with tooltip.
 */
export function DataFreshnessIcon({
  freshnessLevel = 'stale',
  dataUpdatedAt,
  lastUpdatedFormatted,
  isStale,
  className,
}: Pick<DataFreshnessIndicatorProps, 'freshnessLevel' | 'dataUpdatedAt' | 'lastUpdatedFormatted' | 'isStale' | 'className'>) {
  return (
    <DataFreshnessIndicator
      freshnessLevel={freshnessLevel}
      dataUpdatedAt={dataUpdatedAt}
      lastUpdatedFormatted={lastUpdatedFormatted}
      isStale={isStale}
      mode="compact"
      className={className}
    />
  );
}

export default DataFreshnessIndicator;
