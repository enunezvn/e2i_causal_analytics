/**
 * Analytics Query Hooks
 * =====================
 *
 * TanStack Query hooks for the analytics dashboard.
 *
 * @module hooks/api/use-analytics
 */

import { useQuery } from '@tanstack/react-query';
import type { UseQueryOptions } from '@tanstack/react-query';
import {
  getAnalyticsDashboard,
  getAgentMetrics,
  getAgentTrend,
  getMetricsSummary,
} from '@/api/analytics';
import type {
  AnalyticsDashboardResponse,
  AgentMetrics,
  AgentPerformanceTrend,
  QueryMetricsSummary,
  AnalyticsPeriod,
  SummaryPeriod,
} from '@/types/analytics';
import type { ApiError } from '@/lib/api-client';

// =============================================================================
// QUERY KEYS
// =============================================================================

export const analyticsKeys = {
  all: ['analytics'] as const,
  dashboard: (period: AnalyticsPeriod, brand?: string) =>
    [...analyticsKeys.all, 'dashboard', period, brand] as const,
  agent: (agentName: string, period: AnalyticsPeriod, brand?: string) =>
    [...analyticsKeys.all, 'agent', agentName, period, brand] as const,
  agentTrend: (agentName: string, period: AnalyticsPeriod, brand?: string) =>
    [...analyticsKeys.all, 'agentTrend', agentName, period, brand] as const,
  summary: (period: SummaryPeriod) =>
    [...analyticsKeys.all, 'summary', period] as const,
};

// =============================================================================
// HOOKS
// =============================================================================

/**
 * Hook to fetch analytics dashboard data.
 *
 * @param period - Time period: 1d, 7d, 30d, 90d
 * @param brand - Optional brand filter
 * @param options - Additional query options
 *
 * @example
 * ```tsx
 * const { data, isLoading } = useAnalyticsDashboard('7d');
 * ```
 */
export function useAnalyticsDashboard(
  period: AnalyticsPeriod = '7d',
  brand?: string,
  options?: Omit<
    UseQueryOptions<AnalyticsDashboardResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<AnalyticsDashboardResponse, ApiError>({
    queryKey: analyticsKeys.dashboard(period, brand),
    queryFn: () => getAnalyticsDashboard(period, brand),
    staleTime: 2 * 60 * 1000, // 2 minutes
    refetchInterval: 5 * 60 * 1000, // Refetch every 5 minutes
    ...options,
  });
}

/**
 * Hook to fetch metrics for a specific agent.
 *
 * @param agentName - Name of the agent
 * @param period - Time period
 * @param brand - Optional brand filter
 * @param options - Additional query options
 *
 * @example
 * ```tsx
 * const { data } = useAgentMetrics('causal_impact', '7d');
 * ```
 */
export function useAgentMetrics(
  agentName: string,
  period: AnalyticsPeriod = '7d',
  brand?: string,
  options?: Omit<
    UseQueryOptions<AgentMetrics, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<AgentMetrics, ApiError>({
    queryKey: analyticsKeys.agent(agentName, period, brand),
    queryFn: () => getAgentMetrics(agentName, period, brand),
    staleTime: 2 * 60 * 1000,
    enabled: !!agentName,
    ...options,
  });
}

/**
 * Hook to fetch performance trend for a specific agent.
 *
 * @param agentName - Name of the agent
 * @param period - Time period
 * @param brand - Optional brand filter
 * @param options - Additional query options
 *
 * @example
 * ```tsx
 * const { data } = useAgentTrend('orchestrator', '7d');
 * ```
 */
export function useAgentTrend(
  agentName: string,
  period: AnalyticsPeriod = '7d',
  brand?: string,
  options?: Omit<
    UseQueryOptions<AgentPerformanceTrend, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<AgentPerformanceTrend, ApiError>({
    queryKey: analyticsKeys.agentTrend(agentName, period, brand),
    queryFn: () => getAgentTrend(agentName, period, brand),
    staleTime: 2 * 60 * 1000,
    enabled: !!agentName,
    ...options,
  });
}

/**
 * Hook to fetch quick metrics summary.
 *
 * @param period - Time period: 1h, 6h, 24h, 7d
 * @param options - Additional query options
 *
 * @example
 * ```tsx
 * const { data } = useMetricsSummary('24h');
 * ```
 */
export function useMetricsSummary(
  period: SummaryPeriod = '24h',
  options?: Omit<
    UseQueryOptions<QueryMetricsSummary, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<QueryMetricsSummary, ApiError>({
    queryKey: analyticsKeys.summary(period),
    queryFn: () => getMetricsSummary(period),
    staleTime: 60 * 1000, // 1 minute
    refetchInterval: 2 * 60 * 1000, // Refetch every 2 minutes
    ...options,
  });
}
