/**
 * E2I Filters Hook
 * ================
 *
 * Provides access to dashboard filter state including brand, territory,
 * date range, and HCP segment filters.
 *
 * @module hooks/use-e2i-filters
 */

import * as React from 'react';
import { useE2ICopilot, useCopilotEnabled } from '@/providers/E2ICopilotProvider';
import type { E2IFilters } from '@/providers/E2ICopilotProvider';

// =============================================================================
// TYPES
// =============================================================================

export interface UseE2IFiltersReturn {
  /** Current filters */
  filters: E2IFilters;
  /** Whether CopilotKit is enabled */
  enabled: boolean;
  /** Set brand filter */
  setBrand: (brand: E2IFilters['brand']) => void;
  /** Set territory filter */
  setTerritory: (territory: string | null) => void;
  /** Set date range */
  setDateRange: (start: string, end: string) => void;
  /** Set HCP segment */
  setHcpSegment: (segment: string | null) => void;
  /** Reset all filters to defaults */
  resetFilters: () => void;
  /** Get filter summary for display */
  getFilterSummary: () => string;
}

// =============================================================================
// DEFAULT VALUES
// =============================================================================

const DEFAULT_FILTERS: E2IFilters = {
  brand: 'Remibrutinib',
  territory: null,
  dateRange: {
    start: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    end: new Date().toISOString().split('T')[0],
  },
  hcpSegment: null,
};

// =============================================================================
// HOOK
// =============================================================================

/**
 * Hook for managing E2I dashboard filters.
 *
 * @example
 * ```tsx
 * const { filters, setBrand, setDateRange } = useE2IFilters();
 *
 * // Change brand filter
 * setBrand('Kisqali');
 *
 * // Set date range
 * setDateRange('2024-01-01', '2024-06-30');
 * ```
 */
export function useE2IFilters(): UseE2IFiltersReturn {
  const enabled = useCopilotEnabled();

  // When CopilotKit is disabled, use local state
  const [localFilters, setLocalFilters] = React.useState<E2IFilters>(DEFAULT_FILTERS);

  // Try to get context, but don't throw if not available
  let contextFilters: E2IFilters | null = null;
  let setContextFilters: React.Dispatch<React.SetStateAction<E2IFilters>> | null = null;

  try {
    const context = useE2ICopilot();
    contextFilters = context.filters;
    setContextFilters = context.setFilters;
  } catch {
    // Context not available, will use local state
  }

  const filters = contextFilters || localFilters;
  const setFilters = setContextFilters || setLocalFilters;

  const setBrand = React.useCallback(
    (brand: E2IFilters['brand']) => {
      setFilters((prev) => ({ ...prev, brand }));
    },
    [setFilters]
  );

  const setTerritory = React.useCallback(
    (territory: string | null) => {
      setFilters((prev) => ({ ...prev, territory }));
    },
    [setFilters]
  );

  const setDateRange = React.useCallback(
    (start: string, end: string) => {
      setFilters((prev) => ({
        ...prev,
        dateRange: { start, end },
      }));
    },
    [setFilters]
  );

  const setHcpSegment = React.useCallback(
    (segment: string | null) => {
      setFilters((prev) => ({ ...prev, hcpSegment: segment }));
    },
    [setFilters]
  );

  const resetFilters = React.useCallback(() => {
    setFilters(DEFAULT_FILTERS);
  }, [setFilters]);

  const getFilterSummary = React.useCallback((): string => {
    const parts: string[] = [filters.brand];

    if (filters.territory) {
      parts.push(filters.territory);
    }

    if (filters.hcpSegment) {
      parts.push(filters.hcpSegment);
    }

    const dateStr = `${filters.dateRange.start} - ${filters.dateRange.end}`;
    parts.push(dateStr);

    return parts.join(' | ');
  }, [filters]);

  return {
    filters,
    enabled,
    setBrand,
    setTerritory,
    setDateRange,
    setHcpSegment,
    resetFilters,
    getFilterSummary,
  };
}

export default useE2IFilters;
