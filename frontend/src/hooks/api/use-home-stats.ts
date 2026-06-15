/**
 * Home QUICK_STATS hooks
 * ======================
 *
 * React Query hooks for the two Home-tile sources backing the QUICK_STATS bar:
 *   - `useKpiSummary(brand)` → real Total TRx (MTD) / HCPs Reached rollup
 *   - `useActiveExperimentCount()` → Active Campaigns (running experiments)
 *
 * @module hooks/api/use-home-stats
 */

import { useQuery, type UseQueryOptions } from '@tanstack/react-query';
import {
  getKpiSummary,
  getActiveExperimentCount,
  type KpiSummaryResponse,
  type ActiveExperimentCountResponse,
} from '@/api/home-stats';

/** Hook for the real business_metrics KPI rollup.
 *  `region` is part of the query key so switching region refetches (and does
 *  not collide with the portfolio-wide cache entry). */
export function useKpiSummary(
  brand: string,
  region?: string,
  options?: Omit<UseQueryOptions<KpiSummaryResponse, Error>, 'queryKey' | 'queryFn'>
) {
  return useQuery<KpiSummaryResponse, Error>({
    queryKey: ['home', 'kpi-summary', brand, region ?? null],
    queryFn: () => getKpiSummary(brand, region),
    staleTime: 60 * 1000,
    retry: false,
    ...options,
  });
}

/** Hook for the count of running experiments (Active Campaigns). */
export function useActiveExperimentCount(
  options?: Omit<UseQueryOptions<ActiveExperimentCountResponse, Error>, 'queryKey' | 'queryFn'>
) {
  return useQuery<ActiveExperimentCountResponse, Error>({
    queryKey: ['home', 'active-experiment-count'],
    queryFn: () => getActiveExperimentCount(),
    staleTime: 60 * 1000,
    retry: false,
    ...options,
  });
}
