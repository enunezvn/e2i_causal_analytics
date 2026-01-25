/**
 * Analytics API Client
 * ====================
 *
 * API functions for the analytics dashboard.
 *
 * @module api/analytics
 */

import apiClient from '@/lib/api-client';
import type {
  AnalyticsDashboardResponse,
  AgentMetrics,
  AgentPerformanceTrend,
  QueryMetricsSummary,
  AnalyticsPeriod,
  SummaryPeriod,
} from '@/types/analytics';

// =============================================================================
// API FUNCTIONS
// =============================================================================

/**
 * Fetch complete analytics dashboard data.
 *
 * @param period - Time period: 1d, 7d, 30d, 90d
 * @param brand - Optional brand filter
 * @returns Dashboard data with summary, agent metrics, and trends
 */
export async function getAnalyticsDashboard(
  period: AnalyticsPeriod = '7d',
  brand?: string
): Promise<AnalyticsDashboardResponse> {
  const params = new URLSearchParams({ period });
  if (brand) {
    params.append('brand', brand);
  }

  const response = await apiClient.get<AnalyticsDashboardResponse>(
    `/analytics/dashboard?${params.toString()}`
  );
  return response.data;
}

/**
 * Fetch metrics for a specific agent.
 *
 * @param agentName - Name of the agent
 * @param period - Time period
 * @param brand - Optional brand filter
 * @returns Agent metrics
 */
export async function getAgentMetrics(
  agentName: string,
  period: AnalyticsPeriod = '7d',
  brand?: string
): Promise<AgentMetrics> {
  const params = new URLSearchParams({ period });
  if (brand) {
    params.append('brand', brand);
  }

  const response = await apiClient.get<AgentMetrics>(
    `/analytics/agents/${encodeURIComponent(agentName)}?${params.toString()}`
  );
  return response.data;
}

/**
 * Fetch performance trend for a specific agent.
 *
 * @param agentName - Name of the agent
 * @param period - Time period
 * @param brand - Optional brand filter
 * @returns Agent performance trend data
 */
export async function getAgentTrend(
  agentName: string,
  period: AnalyticsPeriod = '7d',
  brand?: string
): Promise<AgentPerformanceTrend> {
  const params = new URLSearchParams({ period });
  if (brand) {
    params.append('brand', brand);
  }

  const response = await apiClient.get<AgentPerformanceTrend>(
    `/analytics/agents/${encodeURIComponent(agentName)}/trend?${params.toString()}`
  );
  return response.data;
}

/**
 * Fetch quick metrics summary (for status display).
 *
 * @param period - Time period: 1h, 6h, 24h, 7d
 * @returns Quick metrics summary
 */
export async function getMetricsSummary(
  period: SummaryPeriod = '24h'
): Promise<QueryMetricsSummary> {
  const response = await apiClient.get<QueryMetricsSummary>(
    `/analytics/summary?period=${period}`
  );
  return response.data;
}
