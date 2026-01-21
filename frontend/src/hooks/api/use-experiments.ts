/**
 * A/B Testing React Query Hooks
 * =============================
 *
 * TanStack Query hooks for the Experiments API endpoints.
 * Provides typed query and mutation hooks for experiment management.
 *
 * @module hooks/api/use-experiments
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import { ApiError } from '@/lib/api-client';
import {
  randomizeUnits,
  getAssignments,
  enrollUnit,
  withdrawUnit,
  getEnrollmentStats,
  triggerInterimAnalysis,
  listInterimAnalyses,
  getExperimentResults,
  getSegmentResults,
  getSRMChecks,
  runSRMCheck,
  getFidelityComparisons,
  updateFidelityComparison,
  triggerMonitoring,
  getExperimentHealth,
  getExperimentAlerts,
} from '@/api/experiments';
import type {
  RandomizeRequest,
  RandomizeResponse,
  AssignmentsListResponse,
  EnrollUnitRequest,
  EnrollmentResult,
  WithdrawRequest,
  WithdrawResponse,
  EnrollmentStatsResponse,
  InterimAnalysisResult,
  InterimAnalysesListResponse,
  ExperimentResults,
  SegmentResultsResponse,
  SRMChecksListResponse,
  SRMCheckResult,
  FidelityComparisonsResponse,
  FidelityComparison,
  MonitorResponse,
  ExperimentHealthSummary,
  ExperimentAlertsResponse,
  GetAssignmentsParams,
  TriggerInterimAnalysisRequest,
  TriggerMonitorRequest,
} from '@/types/experiments';

// =============================================================================
// ASSIGNMENT QUERY HOOKS
// =============================================================================

/**
 * Hook to fetch experiment assignments.
 *
 * @param experimentId - The experiment ID
 * @param params - Optional filter parameters
 * @param options - Additional query options
 * @returns Query result with assignment list
 *
 * @example
 * ```tsx
 * const { data } = useAssignments('exp_123', { variant: 'treatment' });
 * console.log(`${data?.total_assigned} units assigned`);
 * ```
 */
export function useAssignments(
  experimentId: string,
  params?: GetAssignmentsParams,
  options?: Omit<UseQueryOptions<AssignmentsListResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<AssignmentsListResponse, ApiError>({
    queryKey: [...queryKeys.experiments.assignments(experimentId), params],
    queryFn: () => getAssignments(experimentId, params),
    enabled: !!experimentId,
    staleTime: 60 * 1000,
    ...options,
  });
}

/**
 * Hook to fetch enrollment statistics.
 *
 * @param experimentId - The experiment ID
 * @param options - Additional query options
 * @returns Query result with enrollment stats
 *
 * @example
 * ```tsx
 * const { data } = useEnrollmentStats('exp_123');
 * console.log(`Control: ${data?.control_count}, Treatment: ${data?.treatment_count}`);
 * ```
 */
export function useEnrollmentStats(
  experimentId: string,
  options?: Omit<UseQueryOptions<EnrollmentStatsResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<EnrollmentStatsResponse, ApiError>({
    queryKey: queryKeys.experiments.enrollmentStats(experimentId),
    queryFn: () => getEnrollmentStats(experimentId),
    enabled: !!experimentId,
    staleTime: 30 * 1000,
    ...options,
  });
}

// =============================================================================
// ANALYSIS QUERY HOOKS
// =============================================================================

/**
 * Hook to list interim analyses for an experiment.
 *
 * @param experimentId - The experiment ID
 * @param options - Additional query options
 * @returns Query result with interim analysis list
 */
export function useInterimAnalyses(
  experimentId: string,
  options?: Omit<UseQueryOptions<InterimAnalysesListResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<InterimAnalysesListResponse, ApiError>({
    queryKey: queryKeys.experiments.interimAnalyses(experimentId),
    queryFn: () => listInterimAnalyses(experimentId),
    enabled: !!experimentId,
    staleTime: 60 * 1000,
    ...options,
  });
}

/**
 * Hook to fetch experiment results.
 *
 * @param experimentId - The experiment ID
 * @param options - Additional query options
 * @returns Query result with final experiment results
 *
 * @example
 * ```tsx
 * const { data } = useExperimentResults('exp_123');
 * if (data?.is_significant) {
 *   console.log(`Lift: ${data.lift_estimate}%`);
 * }
 * ```
 */
export function useExperimentResults(
  experimentId: string,
  options?: Omit<UseQueryOptions<ExperimentResults, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<ExperimentResults, ApiError>({
    queryKey: queryKeys.experiments.results(experimentId),
    queryFn: () => getExperimentResults(experimentId),
    enabled: !!experimentId,
    staleTime: 2 * 60 * 1000,
    ...options,
  });
}

/**
 * Hook to fetch segment-level experiment results.
 *
 * @param experimentId - The experiment ID
 * @param segmentVar - The segment variable name
 * @param options - Additional query options
 * @returns Query result with segment results
 */
export function useSegmentResults(
  experimentId: string,
  segments: string[],
  options?: Omit<UseQueryOptions<SegmentResultsResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<SegmentResultsResponse, ApiError>({
    queryKey: queryKeys.experiments.segmentResults(experimentId, segments.join(',')),
    queryFn: () => getSegmentResults(experimentId, segments),
    enabled: !!experimentId && segments.length > 0,
    staleTime: 2 * 60 * 1000,
    ...options,
  });
}

// =============================================================================
// VALIDATION QUERY HOOKS
// =============================================================================

/**
 * Hook to fetch SRM check results.
 *
 * @param experimentId - The experiment ID
 * @param options - Additional query options
 * @returns Query result with SRM checks
 *
 * @example
 * ```tsx
 * const { data } = useSRMChecks('exp_123');
 * if (data?.latest_check?.srm_detected) {
 *   console.warn('Sample Ratio Mismatch detected!');
 * }
 * ```
 */
export function useSRMChecks(
  experimentId: string,
  options?: Omit<UseQueryOptions<SRMChecksListResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<SRMChecksListResponse, ApiError>({
    queryKey: queryKeys.experiments.srmChecks(experimentId),
    queryFn: () => getSRMChecks(experimentId),
    enabled: !!experimentId,
    staleTime: 60 * 1000,
    ...options,
  });
}

/**
 * Hook to fetch fidelity comparisons.
 *
 * @param experimentId - The experiment ID
 * @param options - Additional query options
 * @returns Query result with fidelity comparisons
 */
export function useFidelityComparisons(
  experimentId: string,
  options?: Omit<UseQueryOptions<FidelityComparisonsResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<FidelityComparisonsResponse, ApiError>({
    queryKey: queryKeys.experiments.fidelityComparisons(experimentId),
    queryFn: () => getFidelityComparisons(experimentId),
    enabled: !!experimentId,
    staleTime: 2 * 60 * 1000,
    ...options,
  });
}

// =============================================================================
// HEALTH QUERY HOOKS
// =============================================================================

/**
 * Hook to get experiment service health.
 *
 * @param experimentId - The experiment ID
 * @param options - Additional query options
 * @returns Query result with service health status
 */
export function useExperimentHealth(
  experimentId: string,
  options?: Omit<UseQueryOptions<ExperimentHealthSummary, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<ExperimentHealthSummary, ApiError>({
    queryKey: queryKeys.experiments.health(experimentId),
    queryFn: () => getExperimentHealth(experimentId),
    enabled: !!experimentId,
    staleTime: 30 * 1000,
    ...options,
  });
}

/**
 * Hook to fetch experiment alerts.
 *
 * @param experimentId - The experiment ID
 * @param options - Additional query options
 * @returns Query result with experiment alerts
 */
export function useExperimentAlerts(
  experimentId: string,
  options?: Omit<UseQueryOptions<ExperimentAlertsResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<ExperimentAlertsResponse, ApiError>({
    queryKey: queryKeys.experiments.alerts(experimentId),
    queryFn: () => getExperimentAlerts(experimentId),
    enabled: !!experimentId,
    staleTime: 30 * 1000,
    ...options,
  });
}

// =============================================================================
// MUTATION HOOKS
// =============================================================================

/**
 * Hook to randomize units into experiment variants.
 *
 * @param options - Mutation options
 * @returns Mutation object for randomizing units
 *
 * @example
 * ```tsx
 * const { mutate: randomize } = useRandomizeUnits({
 *   onSuccess: (data) => {
 *     console.log(`${data.assignments_created} units randomized`);
 *   },
 * });
 *
 * randomize({
 *   experiment_id: 'exp_123',
 *   unit_ids: ['unit_1', 'unit_2'],
 *   stratification_vars: ['region'],
 * });
 * ```
 */
export function useRandomizeUnits(
  options?: Omit<UseMutationOptions<RandomizeResponse, ApiError, { experimentId: string; request: RandomizeRequest }>, 'mutationFn'>
) {
  const queryClient = useQueryClient();

  return useMutation<RandomizeResponse, ApiError, { experimentId: string; request: RandomizeRequest }>({
    mutationFn: ({ experimentId, request }) => randomizeUnits(experimentId, request),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.experiments.assignments(variables.experimentId),
      });
      queryClient.invalidateQueries({
        queryKey: queryKeys.experiments.enrollmentStats(variables.experimentId),
      });
    },
    ...options,
  });
}

/**
 * Hook to enroll a unit into an experiment.
 *
 * @param options - Mutation options
 * @returns Mutation object for enrollment
 */
export function useEnrollUnit(
  options?: Omit<
    UseMutationOptions<
      EnrollmentResult,
      ApiError,
      { experimentId: string; request: EnrollUnitRequest },
      { previousStats: EnrollmentStatsResponse | undefined }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    EnrollmentResult,
    ApiError,
    { experimentId: string; request: EnrollUnitRequest },
    { previousStats: EnrollmentStatsResponse | undefined }
  >({
    mutationFn: ({ experimentId, request }) => enrollUnit(experimentId, request),

    // Optimistic update: Increment enrollment counts before server responds
    onMutate: async (variables) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({
        queryKey: queryKeys.experiments.enrollmentStats(variables.experimentId),
      });

      // Snapshot previous state for rollback
      const previousStats = queryClient.getQueryData<EnrollmentStatsResponse>(
        queryKeys.experiments.enrollmentStats(variables.experimentId)
      );

      // Optimistically update enrollment stats
      if (previousStats) {
        const optimisticStats: EnrollmentStatsResponse = {
          ...previousStats,
          total_enrolled: previousStats.total_enrolled + 1,
          active_count: previousStats.active_count + 1,
        };
        queryClient.setQueryData(
          queryKeys.experiments.enrollmentStats(variables.experimentId),
          optimisticStats
        );
      }

      return { previousStats };
    },

    // Rollback on error
    onError: (_error, variables, context) => {
      if (context?.previousStats) {
        queryClient.setQueryData(
          queryKeys.experiments.enrollmentStats(variables.experimentId),
          context.previousStats
        );
      }
    },

    // Always refetch to ensure consistency
    onSettled: (_, __, variables) => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.experiments.assignments(variables.experimentId),
      });
      queryClient.invalidateQueries({
        queryKey: queryKeys.experiments.enrollmentStats(variables.experimentId),
      });
    },
    ...options,
  });
}

/**
 * Hook to withdraw a unit from an experiment.
 *
 * @param options - Mutation options
 * @returns Mutation object for withdrawal
 */
export function useWithdrawUnit(
  options?: Omit<
    UseMutationOptions<
      WithdrawResponse,
      ApiError,
      { experimentId: string; enrollmentId: string; request: WithdrawRequest },
      { previousStats: EnrollmentStatsResponse | undefined }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    WithdrawResponse,
    ApiError,
    { experimentId: string; enrollmentId: string; request: WithdrawRequest },
    { previousStats: EnrollmentStatsResponse | undefined }
  >({
    mutationFn: ({ experimentId, enrollmentId, request }) =>
      withdrawUnit(experimentId, enrollmentId, request),

    // Optimistic update: Update counts before server responds
    onMutate: async (variables) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({
        queryKey: queryKeys.experiments.enrollmentStats(variables.experimentId),
      });

      // Snapshot previous state for rollback
      const previousStats = queryClient.getQueryData<EnrollmentStatsResponse>(
        queryKeys.experiments.enrollmentStats(variables.experimentId)
      );

      // Optimistically update enrollment stats
      if (previousStats) {
        const optimisticStats: EnrollmentStatsResponse = {
          ...previousStats,
          active_count: Math.max(0, previousStats.active_count - 1),
          withdrawn_count: previousStats.withdrawn_count + 1,
        };
        queryClient.setQueryData(
          queryKeys.experiments.enrollmentStats(variables.experimentId),
          optimisticStats
        );
      }

      return { previousStats };
    },

    // Rollback on error
    onError: (_error, variables, context) => {
      if (context?.previousStats) {
        queryClient.setQueryData(
          queryKeys.experiments.enrollmentStats(variables.experimentId),
          context.previousStats
        );
      }
    },

    // Always refetch to ensure consistency
    onSettled: (_, __, variables) => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.experiments.assignments(variables.experimentId),
      });
      queryClient.invalidateQueries({
        queryKey: queryKeys.experiments.enrollmentStats(variables.experimentId),
      });
    },
    ...options,
  });
}

/**
 * Hook to trigger interim analysis.
 *
 * @param options - Mutation options
 * @returns Mutation object for triggering analysis
 *
 * @example
 * ```tsx
 * const { mutate: trigger } = useTriggerInterimAnalysis();
 * trigger({ experimentId: 'exp_123', analysisPurpose: 'scheduled' });
 * ```
 */
export function useTriggerInterimAnalysis(
  options?: Omit<
    UseMutationOptions<InterimAnalysisResult, ApiError, { experimentId: string; request: TriggerInterimAnalysisRequest }>,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<InterimAnalysisResult, ApiError, { experimentId: string; request: TriggerInterimAnalysisRequest }>({
    mutationFn: ({ experimentId, request }) => triggerInterimAnalysis(experimentId, request),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.experiments.interimAnalyses(variables.experimentId),
      });
      queryClient.invalidateQueries({
        queryKey: queryKeys.experiments.results(variables.experimentId),
      });
    },
    ...options,
  });
}

/**
 * Hook to run an SRM check.
 *
 * @param options - Mutation options
 * @returns Mutation object for SRM check
 */
export function useRunSRMCheck(
  options?: Omit<UseMutationOptions<SRMCheckResult, ApiError, string>, 'mutationFn'>
) {
  const queryClient = useQueryClient();

  return useMutation<SRMCheckResult, ApiError, string>({
    mutationFn: (experimentId) => runSRMCheck(experimentId),
    onSuccess: (_, experimentId) => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.experiments.srmChecks(experimentId),
      });
    },
    ...options,
  });
}

/**
 * Hook to update a fidelity comparison.
 *
 * @param options - Mutation options
 * @returns Mutation object for fidelity update
 */
export function useUpdateFidelityComparison(
  options?: Omit<
    UseMutationOptions<
      FidelityComparison,
      ApiError,
      { experimentId: string; twinSimulationId: string }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    FidelityComparison,
    ApiError,
    { experimentId: string; twinSimulationId: string }
  >({
    mutationFn: ({ experimentId, twinSimulationId }) =>
      updateFidelityComparison(experimentId, twinSimulationId),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.experiments.fidelityComparisons(variables.experimentId),
      });
    },
    ...options,
  });
}

/**
 * Hook to trigger experiment monitoring.
 *
 * @param options - Mutation options
 * @returns Mutation object for triggering monitoring
 */
export function useTriggerMonitoring(
  options?: Omit<UseMutationOptions<MonitorResponse, ApiError, TriggerMonitorRequest>, 'mutationFn'>
) {
  const queryClient = useQueryClient();

  return useMutation<MonitorResponse, ApiError, TriggerMonitorRequest>({
    mutationFn: (request) => triggerMonitoring(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.experiments.all() });
    },
    ...options,
  });
}
