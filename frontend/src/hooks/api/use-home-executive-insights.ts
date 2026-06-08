/**
 * Home-local Executive Insights hook
 * ==================================
 *
 * A Home-SCOPED hook for `GET /api/executive-insights`, used by the Home page's
 * dual-source "Agent Insights" tile (executive insights + gap opportunities).
 *
 * CONFLICT-AVOIDANCE: PR #798 owns `src/api/executive-insights.ts`,
 * `src/hooks/api/use-executive-insights.ts`, `src/types/executive-insights.ts`,
 * and edits `src/lib/query-client.ts`. This file deliberately does NOT touch any
 * of those — it inlines its own literal queryKey and its own `get(...)` call so
 * there is nothing shared to conflict on.
 *
 * @module hooks/api/use-home-executive-insights
 */

import { useQuery, type UseQueryOptions } from '@tanstack/react-query';
import { get } from '@/lib/api-client';
import type { HomeExecutiveInsight } from '@/types/home-insights';

/**
 * Fetch crystallized executive insights for the Home tile.
 *
 * @param brand - Brand to filter by (omit / 'All' → unfiltered). When 'All',
 *   the brand param is not sent (the backend treats absence as portfolio-wide).
 * @param options - Extra react-query options (e.g. `enabled`).
 */
export function useHomeExecutiveInsights(
  brand?: string,
  options?: Omit<UseQueryOptions<HomeExecutiveInsight[], Error>, 'queryKey' | 'queryFn'>
) {
  const brandParam = brand && brand !== 'All' ? brand : undefined;
  return useQuery<HomeExecutiveInsight[], Error>({
    // Inline literal queryKey (no shared query-client.ts edit).
    queryKey: ['home', 'executive-insights', 'list', brandParam ?? ''],
    queryFn: () =>
      get<HomeExecutiveInsight[]>('/executive-insights', {
        brand: brandParam,
        limit: 10,
      }),
    staleTime: 60 * 1000,
    retry: false,
    ...options,
  });
}
