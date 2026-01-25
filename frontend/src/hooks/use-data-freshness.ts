/**
 * Data Freshness Hook
 * ===================
 *
 * Provides stale data indicators for TanStack Query results.
 * Shows users when data was last updated and whether it might be stale.
 *
 * Features:
 * - Human-readable "X ago" timestamps
 * - Staleness detection based on configured thresholds
 * - Auto-updating timestamps (updates every minute)
 * - Integration with TanStack Query dataUpdatedAt
 *
 * @module hooks/use-data-freshness
 */

import { useState, useEffect, useMemo } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export interface DataFreshnessOptions {
  /** Threshold in milliseconds after which data is considered stale (default: 5 minutes) */
  staleThreshold?: number;
  /** Whether to auto-update the formatted timestamp (default: true) */
  autoUpdate?: boolean;
  /** How often to update the formatted timestamp in ms (default: 60000 = 1 minute) */
  updateInterval?: number;
}

export interface DataFreshnessResult {
  /** Timestamp when data was last updated (from TanStack Query) */
  dataUpdatedAt: number | undefined;
  /** Whether the data is considered stale */
  isStale: boolean;
  /** Human-readable "X ago" string */
  lastUpdatedFormatted: string;
  /** Time elapsed since last update in milliseconds */
  elapsedMs: number | null;
  /** Staleness level: 'fresh' | 'aging' | 'stale' */
  freshnessLevel: 'fresh' | 'aging' | 'stale';
}

// =============================================================================
// CONSTANTS
// =============================================================================

const DEFAULT_STALE_THRESHOLD = 5 * 60 * 1000; // 5 minutes
const AGING_THRESHOLD_RATIO = 0.5; // Data is "aging" after 50% of stale threshold
const DEFAULT_UPDATE_INTERVAL = 60 * 1000; // 1 minute

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Format elapsed time as human-readable "X ago" string
 */
export function formatTimeAgo(timestamp: number | undefined): string {
  if (!timestamp) {
    return 'Never updated';
  }

  const now = Date.now();
  const elapsed = now - timestamp;

  if (elapsed < 0) {
    return 'Just now';
  }

  const seconds = Math.floor(elapsed / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (seconds < 10) {
    return 'Just now';
  }

  if (seconds < 60) {
    return `${seconds} seconds ago`;
  }

  if (minutes === 1) {
    return '1 minute ago';
  }

  if (minutes < 60) {
    return `${minutes} minutes ago`;
  }

  if (hours === 1) {
    return '1 hour ago';
  }

  if (hours < 24) {
    return `${hours} hours ago`;
  }

  if (days === 1) {
    return 'Yesterday';
  }

  return `${days} days ago`;
}

/**
 * Determine freshness level based on elapsed time and threshold
 */
function getFreshnessLevel(
  elapsedMs: number | null,
  staleThreshold: number
): 'fresh' | 'aging' | 'stale' {
  if (elapsedMs === null) {
    return 'stale';
  }

  if (elapsedMs > staleThreshold) {
    return 'stale';
  }

  if (elapsedMs > staleThreshold * AGING_THRESHOLD_RATIO) {
    return 'aging';
  }

  return 'fresh';
}

// =============================================================================
// HOOK IMPLEMENTATION
// =============================================================================

/**
 * Hook to track data freshness from TanStack Query results.
 *
 * @param dataUpdatedAt - The dataUpdatedAt timestamp from a TanStack Query result
 * @param options - Configuration options
 * @returns Data freshness information
 *
 * @example
 * ```tsx
 * const { data, dataUpdatedAt } = useQuery(...);
 * const { lastUpdatedFormatted, isStale, freshnessLevel } = useDataFreshness(dataUpdatedAt);
 *
 * return (
 *   <div>
 *     <DataCard data={data} />
 *     <span className={cn(
 *       'text-xs',
 *       freshnessLevel === 'fresh' && 'text-green-600',
 *       freshnessLevel === 'aging' && 'text-amber-600',
 *       freshnessLevel === 'stale' && 'text-red-600',
 *     )}>
 *       Updated {lastUpdatedFormatted}
 *     </span>
 *   </div>
 * );
 * ```
 */
export function useDataFreshness(
  dataUpdatedAt: number | undefined,
  options: DataFreshnessOptions = {}
): DataFreshnessResult {
  const {
    staleThreshold = DEFAULT_STALE_THRESHOLD,
    autoUpdate = true,
    updateInterval = DEFAULT_UPDATE_INTERVAL,
  } = options;

  // Track current time for re-renders
  const [now, setNow] = useState(Date.now());

  // Auto-update the timestamp display
  useEffect(() => {
    if (!autoUpdate) return;

    const interval = setInterval(() => {
      setNow(Date.now());
    }, updateInterval);

    return () => clearInterval(interval);
  }, [autoUpdate, updateInterval]);

  // Calculate elapsed time
  const elapsedMs = useMemo(() => {
    if (!dataUpdatedAt) return null;
    return now - dataUpdatedAt;
  }, [dataUpdatedAt, now]);

  // Format timestamp
  const lastUpdatedFormatted = useMemo(() => {
    return formatTimeAgo(dataUpdatedAt);
  }, [dataUpdatedAt, now]); // eslint-disable-line react-hooks/exhaustive-deps

  // Determine if stale
  const isStale = useMemo(() => {
    if (elapsedMs === null) return true;
    return elapsedMs > staleThreshold;
  }, [elapsedMs, staleThreshold]);

  // Determine freshness level
  const freshnessLevel = useMemo(() => {
    return getFreshnessLevel(elapsedMs, staleThreshold);
  }, [elapsedMs, staleThreshold]);

  return {
    dataUpdatedAt,
    isStale,
    lastUpdatedFormatted,
    elapsedMs,
    freshnessLevel,
  };
}

/**
 * Hook variant that accepts a TanStack Query result directly.
 *
 * @param queryResult - The result from useQuery
 * @param options - Configuration options
 * @returns Data freshness information
 *
 * @example
 * ```tsx
 * const queryResult = useQuery(...);
 * const freshness = useQueryFreshness(queryResult, { staleThreshold: 2 * 60 * 1000 });
 *
 * return (
 *   <DataFreshnessIndicator {...freshness} />
 * );
 * ```
 */
export function useQueryFreshness(
  queryResult: { dataUpdatedAt: number },
  options?: DataFreshnessOptions
): DataFreshnessResult {
  return useDataFreshness(queryResult.dataUpdatedAt, options);
}

// =============================================================================
// COMPONENT HELPERS
// =============================================================================

/**
 * Get CSS class names for freshness level (compatible with Tailwind)
 */
export function getFreshnessClassName(level: 'fresh' | 'aging' | 'stale'): string {
  switch (level) {
    case 'fresh':
      return 'text-emerald-600 dark:text-emerald-400';
    case 'aging':
      return 'text-amber-600 dark:text-amber-400';
    case 'stale':
      return 'text-rose-600 dark:text-rose-400';
    default:
      return '';
  }
}

/**
 * Get icon name for freshness level (compatible with lucide-react)
 */
export function getFreshnessIconName(level: 'fresh' | 'aging' | 'stale'): 'check-circle' | 'clock' | 'alert-triangle' {
  switch (level) {
    case 'fresh':
      return 'check-circle';
    case 'aging':
      return 'clock';
    case 'stale':
      return 'alert-triangle';
    default:
      return 'clock';
  }
}

export default useDataFreshness;
