/**
 * Executive Insights React Query Hooks
 * ====================================
 *
 * @module hooks/api/use-executive-insights
 */

import { useQuery } from '@tanstack/react-query';
import type { UseQueryOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import type { ApiError } from '@/lib/api-client';
import { listExecutiveInsights } from '@/api/executive-insights';
import type { ExecutiveInsight } from '@/types/executive-insights';

/**
 * Fetch crystallized executive insights for a brand.
 * Disabled when `brand` is empty so the page doesn't fire an unscoped query.
 */
export function useExecutiveInsights(
  brand: string,
  options?: Omit<
    UseQueryOptions<ExecutiveInsight[], ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<ExecutiveInsight[], ApiError>({
    queryKey: queryKeys.executiveInsights.list(brand),
    queryFn: () => listExecutiveInsights({ brand }),
    enabled: !!brand,
    ...options,
  });
}
