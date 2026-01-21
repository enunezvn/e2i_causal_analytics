/**
 * Digital Twin API Query Hooks
 * ============================
 *
 * TanStack Query hooks for the E2I Digital Twin simulation API.
 * Provides type-safe data fetching, caching, and state management
 * for intervention simulations.
 *
 * Features:
 * - Automatic caching and background refetching
 * - Loading and error states
 * - Query key management via queryKeys
 * - Optimistic updates for simulation runs
 *
 * @module hooks/api/use-digital-twin
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import {
  runSimulation,
  compareScenarios,
  getSimulation,
  getSimulationHistory,
  getDigitalTwinHealth,
} from '@/api/digital-twin';
import type {
  SimulateRequest,
  SimulationResponse,
  SimulationDetailResponse,
  ScenarioComparisonRequest,
  ScenarioComparisonResult,
  SimulationHistoryResponse,
  DigitalTwinHealthResponse,
} from '@/types/digital-twin';
import type { ApiError } from '@/lib/api-client';

// =============================================================================
// QUERY HOOKS
// =============================================================================

/**
 * Hook to fetch a specific simulation by ID.
 *
 * @param simulationId - The simulation identifier
 * @param options - Additional TanStack Query options
 * @returns Query result with simulation data
 *
 * @example
 * ```tsx
 * const { data: simulation, isLoading } = useSimulation('sim-123');
 * if (simulation) {
 *   console.log(`ATE: ${simulation.outcomes.ate.estimate}`);
 * }
 * ```
 */
export function useSimulation(
  simulationId: string,
  options?: Omit<
    UseQueryOptions<SimulationDetailResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<SimulationDetailResponse, ApiError>({
    queryKey: queryKeys.digitalTwin.simulation(simulationId),
    queryFn: () => getSimulation(simulationId),
    // Simulations are immutable once created
    staleTime: 60 * 60 * 1000, // 1 hour
    enabled: !!simulationId,
    ...options,
  });
}

/**
 * Hook to fetch simulation history.
 *
 * @param params - Filter and pagination parameters
 * @param options - Additional TanStack Query options
 * @returns Query result with simulation history
 *
 * @example
 * ```tsx
 * const { data: history } = useSimulationHistory({ brand: 'Remibrutinib', limit: 10 });
 * ```
 */
export function useSimulationHistory(
  params?: {
    limit?: number;
    offset?: number;
  },
  options?: Omit<
    UseQueryOptions<SimulationHistoryResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<SimulationHistoryResponse, ApiError>({
    queryKey: queryKeys.digitalTwin.history(),
    queryFn: () => getSimulationHistory(params),
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
}

/**
 * Hook to check digital twin service health.
 *
 * @param options - Additional TanStack Query options
 * @returns Query result with health status
 */
export function useDigitalTwinHealth(
  options?: Omit<
    UseQueryOptions<DigitalTwinHealthResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<DigitalTwinHealthResponse, ApiError>({
    queryKey: queryKeys.digitalTwin.health(),
    queryFn: getDigitalTwinHealth,
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 60 * 1000, // Refetch every minute
    ...options,
  });
}

// =============================================================================
// MUTATION HOOKS
// =============================================================================

/**
 * Hook to run a digital twin simulation.
 *
 * @param options - Additional TanStack mutation options
 * @returns Mutation result with mutate function
 *
 * @example
 * ```tsx
 * const { mutate: simulate, isPending, data } = useRunSimulation();
 *
 * const handleSimulate = () => {
 *   simulate({
 *     intervention_type: InterventionType.HCP_ENGAGEMENT,
 *     brand: 'Remibrutinib',
 *     sample_size: 1000,
 *     duration_days: 90,
 *   });
 * };
 * ```
 */
export function useRunSimulation(
  options?: Omit<
    UseMutationOptions<SimulationResponse, ApiError, SimulateRequest>,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<SimulationResponse, ApiError, SimulateRequest>({
    mutationFn: runSimulation,
    onSuccess: (data) => {
      // Add the new simulation to cache
      queryClient.setQueryData(
        queryKeys.digitalTwin.simulation(data.simulation_id),
        data
      );
      // Invalidate history to include the new simulation
      queryClient.invalidateQueries({
        queryKey: queryKeys.digitalTwin.history(),
      });
    },
    ...options,
  });
}

/**
 * Hook to compare multiple intervention scenarios.
 *
 * @param options - Additional TanStack mutation options
 * @returns Mutation result with mutate function
 *
 * @example
 * ```tsx
 * const { mutate: compare, isPending } = useCompareScenarios();
 *
 * const handleCompare = () => {
 *   compare({
 *     base_scenario: baseScenario,
 *     alternative_scenarios: [scenario1, scenario2],
 *   });
 * };
 * ```
 */
export function useCompareScenarios(
  options?: Omit<
    UseMutationOptions<ScenarioComparisonResult, ApiError, ScenarioComparisonRequest>,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<ScenarioComparisonResult, ApiError, ScenarioComparisonRequest>({
    mutationFn: compareScenarios,
    onSuccess: (data) => {
      // Cache each simulation result
      queryClient.setQueryData(
        queryKeys.digitalTwin.simulation(data.base_result.simulation_id),
        data.base_result
      );
      data.alternative_results.forEach((result) => {
        queryClient.setQueryData(
          queryKeys.digitalTwin.simulation(result.simulation_id),
          result
        );
      });
    },
    ...options,
  });
}

// =============================================================================
// PREFETCH HELPERS
// =============================================================================

/**
 * Prefetch simulation history for faster navigation.
 *
 * @param queryClient - The query client instance
 */
export async function prefetchSimulationHistory(
  queryClient: ReturnType<typeof useQueryClient>
) {
  await queryClient.prefetchQuery({
    queryKey: queryKeys.digitalTwin.history(),
    queryFn: () => getSimulationHistory({ limit: 10 }),
  });
}
