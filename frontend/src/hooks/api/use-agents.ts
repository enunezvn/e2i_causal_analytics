/**
 * Agent Orchestration Query Hooks
 * ===============================
 *
 * TanStack Query hooks for the agent-orchestration endpoints, backed by
 * `@/api/agents`. These replace the three inline `useQuery` bodies that lived
 * in `pages/AgentOrchestration.tsx` (#1941): the fetches themselves were
 * already validated, so this is a consistency refactor — the query options
 * below reproduce the page's previous ones exactly (30s poll, no retry, no
 * explicit staleTime).
 *
 * @module hooks/api/use-agents
 */

import { useQuery } from '@tanstack/react-query';
import type { UseQueryOptions } from '@tanstack/react-query';
import {
  getAgentStatus,
  getAgentActivity,
  getAgentTierMetrics,
} from '@/api/agents';
import type {
  AgentStatusResponseValidated,
  AgentActivityResponseValidated,
  TierMetricsResponseValidated,
} from '@/lib/api-schemas';

// =============================================================================
// QUERY KEYS
// =============================================================================

/**
 * Cache keys for the agent-orchestration queries.
 *
 * `status()` DELIBERATELY keeps the bare `['agent-status']` value it replaces:
 * Home, ExecutiveSummary and both chat sidebars each read `/agents/status`
 * under that same literal key today, so all five share one cache entry.
 * Namespacing it here would silently split that entry and make every one of
 * those consumers refetch — a behaviour change, not a consistency fix.
 * Centralising the *definition* is the point of #1941; changing the *value* is
 * a separate migration (point the other four call sites at `agentKeys.status()`
 * first, then rename in one step).
 *
 * `activity()` / `tierMetrics()` fold their result-affecting params into the
 * key so a different window cannot collide with the cached 24h read. Their
 * default-argument values reproduce the page's previous single call exactly,
 * and no other module reads those keys.
 */
export const agentKeys = {
  status: () => ['agent-status'] as const,
  activity: (hours: number = 24, limit: number = 50) =>
    ['agent-activity', hours, limit] as const,
  tierMetrics: (hours: number = 24) => ['tier-metrics', hours] as const,
};

// =============================================================================
// HOOKS
// =============================================================================

/**
 * Hook for the live agent roster (`GET /agents/status`).
 *
 * @param options - Additional query options
 *
 * @example
 * ```tsx
 * const { data: agentStatus, refetch } = useAgentStatus();
 * ```
 */
export function useAgentStatus(
  options?: Omit<
    UseQueryOptions<AgentStatusResponseValidated, Error>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<AgentStatusResponseValidated, Error>({
    queryKey: agentKeys.status(),
    queryFn: () => getAgentStatus(),
    refetchInterval: 30_000, // Refresh every 30 seconds
    retry: false,
    ...options,
  });
}

/**
 * Hook for the recent agent activity feed (`GET /agents/activity`).
 *
 * @param hours - Look-back window in hours (default 24)
 * @param limit - Maximum rows to return (default 50)
 * @param options - Additional query options
 *
 * @example
 * ```tsx
 * const { data: activityData } = useAgentActivity();
 * ```
 */
export function useAgentActivity(
  hours: number = 24,
  limit: number = 50,
  options?: Omit<
    UseQueryOptions<AgentActivityResponseValidated, Error>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<AgentActivityResponseValidated, Error>({
    queryKey: agentKeys.activity(hours, limit),
    queryFn: () => getAgentActivity(hours, limit),
    refetchInterval: 30_000,
    retry: false,
    ...options,
  });
}

/**
 * Hook for per-tier agent performance (`GET /analytics/tier-metrics`).
 *
 * @param hours - Look-back window in hours (default 24)
 * @param options - Additional query options
 *
 * @example
 * ```tsx
 * const { data: tierData } = useAgentTierMetrics();
 * ```
 */
export function useAgentTierMetrics(
  hours: number = 24,
  options?: Omit<
    UseQueryOptions<TierMetricsResponseValidated, Error>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<TierMetricsResponseValidated, Error>({
    queryKey: agentKeys.tierMetrics(hours),
    queryFn: () => getAgentTierMetrics(hours),
    refetchInterval: 30_000,
    retry: false,
    ...options,
  });
}
