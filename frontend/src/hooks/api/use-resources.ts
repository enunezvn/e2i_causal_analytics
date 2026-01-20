/**
 * Resource Optimization React Query Hooks
 * ========================================
 *
 * TanStack Query hooks for the Resource Optimization API endpoints.
 * Provides typed query and mutation hooks for optimization operations.
 *
 * @module hooks/api/use-resources
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import { ApiError } from '@/lib/api-client';
import {
  runOptimization,
  getOptimization,
  listScenarios,
  getResourceHealth,
  runOptimizationAndWait,
  optimizeBudget,
  optimizeWithScenarios,
} from '@/api/resources';
import type {
  ListScenariosParams,
  OptimizationResponse,
  RunOptimizationRequest,
  ResourceHealthResponse,
  ScenarioListResponse,
} from '@/types/resources';

// =============================================================================
// QUERY HOOKS
// =============================================================================

/**
 * Hook to fetch an optimization result by ID.
 *
 * @param optimizationId - The unique optimization identifier
 * @param options - Additional query options
 * @returns Query result with optimization data
 *
 * @example
 * ```tsx
 * const { data, isLoading } = useOptimization('opt_abc123');
 * if (data?.status === 'completed') {
 *   console.log(`Objective value: ${data.objective_value}`);
 * }
 * ```
 */
export function useOptimization(
  optimizationId: string,
  options?: Omit<UseQueryOptions<OptimizationResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<OptimizationResponse, ApiError>({
    queryKey: queryKeys.resources.optimization(optimizationId),
    queryFn: () => getOptimization(optimizationId),
    enabled: !!optimizationId,
    ...options,
  });
}

/**
 * Hook to list scenario analyses.
 *
 * @param params - Optional filter parameters (min_roi, limit)
 * @param options - Additional query options
 * @returns Query result with scenario list
 *
 * @example
 * ```tsx
 * const { data } = useScenarios({ min_roi: 1.5, limit: 10 });
 * data?.scenarios.forEach(s => {
 *   console.log(`${s.scenario_name}: ROI ${s.roi}`);
 * });
 * ```
 */
export function useScenarios(
  params?: ListScenariosParams,
  options?: Omit<UseQueryOptions<ScenarioListResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<ScenarioListResponse, ApiError>({
    queryKey: [...queryKeys.resources.scenarios(), params?.min_roi, params?.limit],
    queryFn: () => listScenarios(params),
    staleTime: 60 * 1000,
    ...options,
  });
}

/**
 * Hook to get resource optimization service health.
 *
 * @param options - Additional query options
 * @returns Query result with service health status
 *
 * @example
 * ```tsx
 * const { data: health } = useResourceHealth();
 * if (health?.agent_available && health?.scipy_available) {
 *   console.log('Resource optimizer is ready');
 * }
 * ```
 */
export function useResourceHealth(
  options?: Omit<UseQueryOptions<ResourceHealthResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<ResourceHealthResponse, ApiError>({
    queryKey: queryKeys.resources.health(),
    queryFn: () => getResourceHealth(),
    staleTime: 30 * 1000,
    ...options,
  });
}

// =============================================================================
// MUTATION HOOKS
// =============================================================================

/**
 * Hook to run resource optimization.
 *
 * @param options - Mutation options
 * @returns Mutation object for triggering optimization
 *
 * @example
 * ```tsx
 * const { mutate: optimize, isPending } = useRunOptimization();
 *
 * optimize({
 *   request: {
 *     query: 'Optimize budget allocation',
 *     resource_type: 'budget',
 *     allocation_targets: targets,
 *     constraints: constraints,
 *   },
 *   asyncMode: true,
 * });
 * ```
 */
export function useRunOptimization(
  options?: Omit<
    UseMutationOptions<
      OptimizationResponse,
      ApiError,
      { request: RunOptimizationRequest; asyncMode?: boolean }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    OptimizationResponse,
    ApiError,
    { request: RunOptimizationRequest; asyncMode?: boolean }
  >({
    mutationFn: ({ request, asyncMode = true }) => runOptimization(request, asyncMode),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.resources.optimization(data.optimization_id), data);
      queryClient.invalidateQueries({ queryKey: queryKeys.resources.scenarios() });
    },
    ...options,
  });
}

/**
 * Hook to run optimization and wait for completion.
 *
 * @param options - Mutation options
 * @returns Mutation object for running optimization with polling
 */
export function useRunOptimizationAndWait(
  options?: Omit<
    UseMutationOptions<
      OptimizationResponse,
      ApiError,
      { request: RunOptimizationRequest; pollIntervalMs?: number; maxWaitMs?: number }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    OptimizationResponse,
    ApiError,
    { request: RunOptimizationRequest; pollIntervalMs?: number; maxWaitMs?: number }
  >({
    mutationFn: ({ request, pollIntervalMs, maxWaitMs }) =>
      runOptimizationAndWait(request, pollIntervalMs, maxWaitMs),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.resources.optimization(data.optimization_id), data);
      queryClient.invalidateQueries({ queryKey: queryKeys.resources.scenarios() });
    },
    ...options,
  });
}

/**
 * Hook to optimize budget allocation for maximum ROI.
 *
 * @param options - Mutation options
 * @returns Mutation object for budget optimization
 *
 * @example
 * ```tsx
 * const { mutate: optimizeBudgetAllocation } = useOptimizeBudget();
 *
 * optimizeBudgetAllocation({
 *   targets: [
 *     { entity_id: 'northeast', entity_type: 'territory', current_allocation: 50000, expected_response: 1.3 },
 *   ],
 *   totalBudget: 200000,
 *   runScenarios: true,
 * });
 * ```
 */
export function useOptimizeBudget(
  options?: Omit<
    UseMutationOptions<
      OptimizationResponse,
      ApiError,
      { targets: RunOptimizationRequest['allocation_targets']; totalBudget: number; runScenarios?: boolean }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    OptimizationResponse,
    ApiError,
    { targets: RunOptimizationRequest['allocation_targets']; totalBudget: number; runScenarios?: boolean }
  >({
    mutationFn: ({ targets, totalBudget, runScenarios }) => optimizeBudget(targets, totalBudget, runScenarios),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.resources.optimization(data.optimization_id), data);
      queryClient.invalidateQueries({ queryKey: queryKeys.resources.scenarios() });
    },
    ...options,
  });
}

/**
 * Hook to run optimization with scenario comparison.
 *
 * @param options - Mutation options
 * @returns Mutation object for optimization with scenarios
 */
export function useOptimizeWithScenarios(
  options?: Omit<
    UseMutationOptions<
      OptimizationResponse,
      ApiError,
      { request: RunOptimizationRequest; scenarioCount?: number }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    OptimizationResponse,
    ApiError,
    { request: RunOptimizationRequest; scenarioCount?: number }
  >({
    mutationFn: ({ request, scenarioCount }) => optimizeWithScenarios(request, scenarioCount),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.resources.optimization(data.optimization_id), data);
      queryClient.invalidateQueries({ queryKey: queryKeys.resources.scenarios() });
    },
    ...options,
  });
}

// =============================================================================
// POLLING HOOKS
// =============================================================================

/**
 * Hook to poll an optimization until completion.
 *
 * @param optimizationId - The optimization ID to poll
 * @param options - Query options
 * @returns Query result that updates until completion
 */
export function usePollOptimization(
  optimizationId: string,
  options?: Omit<UseQueryOptions<OptimizationResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<OptimizationResponse, ApiError>({
    queryKey: queryKeys.resources.optimization(optimizationId),
    queryFn: () => getOptimization(optimizationId),
    enabled: !!optimizationId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === 'completed' || status === 'failed') {
        return false;
      }
      return 2000;
    },
    ...options,
  });
}
