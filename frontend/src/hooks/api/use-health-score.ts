/**
 * Health Score React Query Hooks
 * ==============================
 *
 * TanStack Query hooks for the Health Score API endpoints.
 * Provides typed query and mutation hooks for system health monitoring.
 * Integrates with the Tier 3 Health Score Fast Path agent.
 *
 * @module hooks/api/use-health-score
 */

import { useQuery, useMutation } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import { ApiError } from '@/lib/api-client';
import {
  runHealthCheck,
  quickHealthCheck,
  fullHealthCheck,
  getComponentHealth,
  getModelHealth,
  getPipelineHealth,
  getAgentHealth,
  getHealthHistory,
  getHealthServiceStatus,
  getComprehensiveHealth,
  getHealthDashboard,
} from '@/api/health-score';
import type {
  CheckScope,
  HealthScoreResponse,
  ComponentHealthResponse,
  ModelHealthResponse,
  PipelineHealthResponse,
  AgentHealthResponse,
  HealthHistoryResponse,
  HealthServiceStatus,
} from '@/types/health-score';

// =============================================================================
// HEALTH CHECK QUERY HOOKS
// =============================================================================

/**
 * Hook to run a quick health check (<1s target).
 *
 * @param options - Additional query options
 * @returns Query result with quick health check data
 *
 * @example
 * ```tsx
 * const { data } = useQuickHealthCheck({ refetchInterval: 30000 });
 * console.log(`Health: ${data?.health_grade}`);
 * ```
 */
export function useQuickHealthCheck(
  options?: Omit<UseQueryOptions<HealthScoreResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<HealthScoreResponse, ApiError>({
    queryKey: queryKeys.healthScore.quick(),
    queryFn: () => quickHealthCheck(),
    staleTime: 30 * 1000, // 30 seconds
    ...options,
  });
}

/**
 * Hook to run a full health check (<5s target).
 *
 * @param options - Additional query options
 * @returns Query result with comprehensive health check data
 *
 * @example
 * ```tsx
 * const { data } = useFullHealthCheck();
 * console.log(`Component: ${data?.component_health_score}`);
 * console.log(`Model: ${data?.model_health_score}`);
 * ```
 */
export function useFullHealthCheck(
  options?: Omit<UseQueryOptions<HealthScoreResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<HealthScoreResponse, ApiError>({
    queryKey: queryKeys.healthScore.full(),
    queryFn: () => fullHealthCheck(),
    staleTime: 60 * 1000, // 1 minute
    ...options,
  });
}

/**
 * Hook to run a scoped health check.
 *
 * @param scope - Check scope (full, quick, models, pipelines, agents)
 * @param options - Additional query options
 * @returns Query result with scoped health check data
 */
export function useScopedHealthCheck(
  scope: CheckScope | string,
  options?: Omit<UseQueryOptions<HealthScoreResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<HealthScoreResponse, ApiError>({
    queryKey: queryKeys.healthScore.check(scope),
    queryFn: () => runHealthCheck(scope),
    staleTime: 30 * 1000,
    ...options,
  });
}

// =============================================================================
// DIMENSION-SPECIFIC QUERY HOOKS
// =============================================================================

/**
 * Hook to get component health details.
 *
 * @param options - Additional query options
 * @returns Query result with component health data
 *
 * @example
 * ```tsx
 * const { data } = useComponentHealth();
 * data?.components.forEach(c => {
 *   console.log(`${c.component_name}: ${c.status} (${c.latency_ms}ms)`);
 * });
 * ```
 */
export function useComponentHealth(
  options?: Omit<UseQueryOptions<ComponentHealthResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<ComponentHealthResponse, ApiError>({
    queryKey: queryKeys.healthScore.components(),
    queryFn: () => getComponentHealth(),
    staleTime: 30 * 1000,
    ...options,
  });
}

/**
 * Hook to get model health details.
 *
 * @param options - Additional query options
 * @returns Query result with model health data
 *
 * @example
 * ```tsx
 * const { data } = useModelHealth();
 * data?.models.forEach(m => {
 *   console.log(`${m.model_name}: ${m.status}, Accuracy: ${m.accuracy?.toFixed(2)}`);
 * });
 * ```
 */
export function useModelHealth(
  options?: Omit<UseQueryOptions<ModelHealthResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<ModelHealthResponse, ApiError>({
    queryKey: queryKeys.healthScore.models(),
    queryFn: () => getModelHealth(),
    staleTime: 60 * 1000,
    ...options,
  });
}

/**
 * Hook to get pipeline health details.
 *
 * @param options - Additional query options
 * @returns Query result with pipeline health data
 *
 * @example
 * ```tsx
 * const { data } = usePipelineHealth();
 * data?.pipelines.forEach(p => {
 *   console.log(`${p.pipeline_name}: ${p.status}, Freshness: ${p.freshness_hours.toFixed(1)}h`);
 * });
 * ```
 */
export function usePipelineHealth(
  options?: Omit<UseQueryOptions<PipelineHealthResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<PipelineHealthResponse, ApiError>({
    queryKey: queryKeys.healthScore.pipelines(),
    queryFn: () => getPipelineHealth(),
    staleTime: 60 * 1000,
    ...options,
  });
}

/**
 * Hook to get agent health details.
 *
 * @param options - Additional query options
 * @returns Query result with agent health data
 *
 * @example
 * ```tsx
 * const { data } = useAgentHealth();
 * console.log(`Available: ${data?.available_count}/${data?.total_agents}`);
 * data?.agents.forEach(a => {
 *   console.log(`[Tier ${a.tier}] ${a.agent_name}: ${a.available ? 'UP' : 'DOWN'}`);
 * });
 * ```
 */
export function useAgentHealth(
  options?: Omit<UseQueryOptions<AgentHealthResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<AgentHealthResponse, ApiError>({
    queryKey: queryKeys.healthScore.agents(),
    queryFn: () => getAgentHealth(),
    staleTime: 30 * 1000,
    ...options,
  });
}

// =============================================================================
// HISTORY AND STATUS QUERY HOOKS
// =============================================================================

/**
 * Hook to get health check history.
 *
 * @param limit - Maximum number of records (default: 20)
 * @param options - Additional query options
 * @returns Query result with health history data
 *
 * @example
 * ```tsx
 * const { data } = useHealthHistory(10);
 * console.log(`Trend: ${data?.trend}`);
 * console.log(`Average: ${data?.avg_health_score.toFixed(1)}`);
 * ```
 */
export function useHealthHistory(
  limit: number = 20,
  options?: Omit<UseQueryOptions<HealthHistoryResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<HealthHistoryResponse, ApiError>({
    queryKey: queryKeys.healthScore.history(limit),
    queryFn: () => getHealthHistory(limit),
    staleTime: 60 * 1000,
    ...options,
  });
}

/**
 * Hook to get health service status.
 *
 * @param options - Additional query options
 * @returns Query result with service status data
 *
 * @example
 * ```tsx
 * const { data } = useHealthServiceStatus();
 * if (data?.status === 'healthy') {
 *   console.log(`${data.checks_24h} checks in last 24h`);
 * }
 * ```
 */
export function useHealthServiceStatus(
  options?: Omit<UseQueryOptions<HealthServiceStatus, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<HealthServiceStatus, ApiError>({
    queryKey: queryKeys.healthScore.status(),
    queryFn: () => getHealthServiceStatus(),
    staleTime: 30 * 1000,
    ...options,
  });
}

// =============================================================================
// COMPOSITE QUERY HOOKS
// =============================================================================

/**
 * Type for comprehensive health data
 */
export interface ComprehensiveHealthData {
  health: HealthScoreResponse;
  components: ComponentHealthResponse;
  models: ModelHealthResponse;
  pipelines: PipelineHealthResponse;
  agents: AgentHealthResponse;
}

/**
 * Type for dashboard health data
 */
export interface DashboardHealthData {
  current: HealthScoreResponse;
  status: HealthServiceStatus;
  history: HealthHistoryResponse;
}

/**
 * Hook to get comprehensive health status (all dimensions).
 *
 * @param options - Additional query options
 * @returns Query result with all health dimensions
 *
 * @example
 * ```tsx
 * const { data } = useComprehensiveHealth();
 * console.log(`Overall: ${data?.health.health_grade}`);
 * console.log(`Components: ${data?.components.healthy_count}/${data?.components.total_components}`);
 * ```
 */
export function useComprehensiveHealth(
  options?: Omit<UseQueryOptions<ComprehensiveHealthData, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<ComprehensiveHealthData, ApiError>({
    queryKey: [...queryKeys.healthScore.all(), 'comprehensive'],
    queryFn: () => getComprehensiveHealth(),
    staleTime: 60 * 1000,
    ...options,
  });
}

/**
 * Hook to get dashboard health data.
 *
 * @param options - Additional query options
 * @returns Query result with dashboard-ready data
 *
 * @example
 * ```tsx
 * const { data } = useHealthDashboard();
 * console.log(`Current: ${data?.current.health_grade}`);
 * console.log(`Trend: ${data?.history.trend}`);
 * console.log(`Service: ${data?.status.status}`);
 * ```
 */
export function useHealthDashboard(
  options?: Omit<UseQueryOptions<DashboardHealthData, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<DashboardHealthData, ApiError>({
    queryKey: [...queryKeys.healthScore.all(), 'dashboard'],
    queryFn: () => getHealthDashboard(),
    staleTime: 30 * 1000,
    ...options,
  });
}

// =============================================================================
// MUTATION HOOKS
// =============================================================================

/**
 * Hook to run a health check on demand.
 *
 * @param options - Mutation options
 * @returns Mutation object for triggering health checks
 *
 * @example
 * ```tsx
 * const { mutate: runCheck, isPending } = useRunHealthCheck();
 *
 * runCheck({ scope: 'full' });
 * ```
 */
export function useRunHealthCheck(
  options?: Omit<UseMutationOptions<HealthScoreResponse, ApiError, { scope?: CheckScope | string }>, 'mutationFn'>
) {
  return useMutation<HealthScoreResponse, ApiError, { scope?: CheckScope | string }>({
    mutationFn: ({ scope = 'full' }) => runHealthCheck(scope),
    ...options,
  });
}

// =============================================================================
// POLLING HOOKS
// =============================================================================

/**
 * Hook to continuously monitor health with auto-refresh.
 *
 * @param intervalMs - Refresh interval in milliseconds (default: 30000)
 * @param useQuick - Use quick check instead of full (default: true)
 * @param options - Additional query options
 * @returns Query result with auto-refreshing health data
 *
 * @example
 * ```tsx
 * const { data } = useHealthMonitor(60000, true);
 * console.log(`Health: ${data?.health_grade}`);
 * ```
 */
export function useHealthMonitor(
  intervalMs: number = 30000,
  useQuick: boolean = true,
  options?: Omit<UseQueryOptions<HealthScoreResponse, ApiError>, 'queryKey' | 'queryFn' | 'refetchInterval'>
) {
  return useQuery<HealthScoreResponse, ApiError>({
    queryKey: useQuick ? queryKeys.healthScore.quick() : queryKeys.healthScore.full(),
    queryFn: () => (useQuick ? quickHealthCheck() : fullHealthCheck()),
    refetchInterval: intervalMs,
    staleTime: intervalMs - 5000, // 5 seconds before interval
    ...options,
  });
}
