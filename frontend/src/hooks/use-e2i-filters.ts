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
  /** Set region filter (#1753) */
  setRegion: (region: E2IFilters['region']) => void;
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

// #1749: 'All' = no brand selected — never default to a specific brand.
// #1753: 'All US' = no region selected — same honest-sentinel design.
const DEFAULT_FILTERS: E2IFilters = {
  brand: 'All',
  region: 'All US',
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
  //
  // #1942: this try/catch is a rules-of-hooks *smell* (React explicitly warns
  // against calling hooks inside try/catch), but it is NOT a latent crash —
  // useE2ICopilot() calls exactly one hook (useContext) before its throw
  // point, so the hook count/order is identical on both paths, and whether it
  // throws is stable for a given render tree.
  //
  // The obvious-looking replacement — gate this call behind useCopilotEnabled()
  // instead — is NOT behavior-preserving and was rejected after measuring it:
  // RootLayout (frontend/src/router/index.tsx) mounts <E2ICopilotProvider>
  // unconditionally *inside* <CopilotKitWrapper enabled={env.copilotEnabled}>,
  // so the provider (and this real context) is present even when
  // useCopilotEnabled() reports false — e.g. dev mode (VITE_COPILOT_ENABLED
  // defaults to false outside prod) and the CI e2e build (explicitly
  // VITE_COPILOT_ENABLED=false). Gating on `enabled` there would silently
  // switch this hook to an isolated per-component local state instead of the
  // shared filters context that AIAgentInsights/E2IChatPopup/E2IChatSidebar
  // read directly via useE2ICopilot() — breaking cross-page brand/region
  // filter sync in exactly those environments. Confirmed with a render test
  // asserting a sibling reading useE2ICopilot() directly observes a setBrand()
  // made through this hook while wrapped only in
  // `<CopilotKitWrapper enabled={false}><E2ICopilotProvider>` (provider
  // mounted, copilot disabled) — it does today; it stops the moment this call
  // is gated on `enabled`. See use-e2i-filters.test.ts for the pinned
  // regression test. There is no way to distinguish "no provider mounted" from
  // "provider mounted, copilot disabled" using useCopilotEnabled() alone —
  // both report `false` — so a correct non-try/catch fix needs a safe,
  // non-throwing accessor exported from E2ICopilotProvider.tsx itself (out of
  // scope here).
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

  const setRegion = React.useCallback(
    (region: E2IFilters['region']) => {
      setFilters((prev) => ({ ...prev, region }));
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
    // The 'All' sentinel is not a brand — render it as prose (#1749).
    const parts: string[] = [filters.brand === 'All' ? 'All brands' : filters.brand];

    // Same for the region sentinel (#1753).
    parts.push(filters.region === 'All US' ? 'All US regions' : filters.region);

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
    setRegion,
    setTerritory,
    setDateRange,
    setHcpSegment,
    resetFilters,
    getFilterSummary,
  };
}

export default useE2IFilters;
