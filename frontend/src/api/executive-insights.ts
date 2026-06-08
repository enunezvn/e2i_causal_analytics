/**
 * Executive Insights API Client
 * =============================
 *
 * Reads crystallized cross-agent narratives from the backend
 * crystallization subsystem. Wraps the shared apiClient helpers.
 *
 * Endpoints:
 * - GET /api/executive-insights              (list, brand-filtered)
 * - GET /api/executive-insights/portfolio-summary
 *
 * @module api/executive-insights
 */

import { get } from '@/lib/api-client';
import type {
  ExecutiveInsight,
  ListExecutiveInsightsParams,
} from '@/types/executive-insights';

const EXEC_BASE = '/executive-insights';

/** List crystallized executive insights (brand filter strongly recommended). */
export async function listExecutiveInsights(
  params?: ListExecutiveInsightsParams
): Promise<ExecutiveInsight[]> {
  return get<ExecutiveInsight[]>(EXEC_BASE, params as Record<string, unknown>);
}
