/**
 * Home QUICK_STATS API client
 * ===========================
 *
 * Thin clients for the two Home-tile sources that don't already have a hook:
 *   - the real business_metrics KPI rollup (`GET /api/copilotkit/kpis/summary`)
 *   - the active-experiments count (`GET /api/experiments/active-count`)
 *
 * @module api/home-stats
 */

import { get } from '@/lib/api-client';

/** Real business_metrics rollup returned by GET /copilotkit/kpis/summary. */
export interface KpiSummaryResponse {
  brand: string;
  period: string;
  metrics: {
    trx_volume?: number;
    nrx_volume?: number;
    market_share?: number;
    conversion_rate?: number;
    hcp_reach?: number;
    patient_starts?: number;
    [key: string]: number | string | string[] | undefined;
  };
  /** 'database' = real DB values; 'synthetic' = computed over synthetic-gold
   *  rows (E2I_KPI_INCLUDE_SYNTHETIC demo/review mode); 'fallback' = sample data. */
  data_source: 'database' | 'synthetic' | 'fallback' | string;
  /** Latest treatment_events prescription date (data-coverage end), ISO `YYYY-MM-DD`,
   *  or null when unavailable. Used to render "No recent activity — data through
   *  <date>" on empty (0/null) tiles with a dynamic date. */
  data_through?: string | null;
}

/** Fetch the real KPI rollup for a brand, optionally scoped to a region.
 *  `region` is omitted from the query when undefined (axios drops undefined
 *  params), so the portfolio-wide rollup is unchanged when no region is set. */
export async function getKpiSummary(
  brand: string,
  region?: string
): Promise<KpiSummaryResponse> {
  return get<KpiSummaryResponse>('/copilotkit/kpis/summary', { brand, region });
}

/** Count of currently-running experiments (Active Campaigns). */
export interface ActiveExperimentCountResponse {
  active_count: number;
}

/** Fetch the count of running experiments. */
export async function getActiveExperimentCount(): Promise<ActiveExperimentCountResponse> {
  return get<ActiveExperimentCountResponse>('/experiments/active-count');
}
