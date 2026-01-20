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
  RandomizeUnitsRequest,
  RandomizationResponse,
  AssignmentListResponse,
  EnrollUnitRequest,
  EnrollmentResponse,
  WithdrawUnitRequest,
  EnrollmentStatsResponse,
  InterimAnalysisResponse,
  InterimAnalysisListResponse,
  ExperimentResultsResponse,
  SegmentResultsResponse,
  SRMCheckListResponse,
  SRMCheckResponse,
  FidelityComparisonListResponse,
  FidelityComparisonUpdateRequest,
  FidelityComparisonResponse,
  MonitoringResponse,
  ExperimentHealthResponse,
  ExperimentAlertListResponse,
  GetAssignmentsParams,
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
  options?: Omit<UseQueryOptions<AssignmentListResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<AssignmentListResponse, ApiError>({
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
  options?: Omit<UseQueryOptions<InterimAnalysisListResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<InterimAnalysisListResponse, ApiError>({
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
  options?: Omit<UseQueryOptions<ExperimentResultsResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<ExperimentResultsResponse, ApiError>({
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
  segmentVar: string,
  options?: Omit<UseQueryOptions<SegmentResultsResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<SegmentResultsResponse, ApiError>({
    queryKey: queryKeys.experiments.segmentResults(experimentId, segmentVar),
    queryFn: () => getSegmentResults(experimentId, segmentVar),
    enabled: !!experimentId && !!segmentVar,
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
  options?: Omit<UseQueryOptions<SRMCheckListResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<SRMCheckListResponse, ApiError>({
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
  options?: Omit<UseQueryOptions<FidelityComparisonListResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<FidelityComparisonListResponse, ApiError>({
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
 * @param options - Additional query options
 * @returns Query result with service health status
 */
export function useExperimentHealth(
  options?: Omit<UseQueryOptions<ExperimentHealthResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<ExperimentHealthResponse, ApiError>({
    queryKey: queryKeys.experiments.health(),
    queryFn: () => getExperimentHealth(),
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
  options?: Omit<UseQueryOptions<ExperimentAlertListResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<ExperimentAlertListResponse, ApiError>({
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
  options?: Omit<UseMutationOptions<RandomizationResponse, ApiError, RandomizeUnitsRequest>, 'mutationFn'>
) {
  const queryClient = useQueryClient();

  return useMutation<RandomizationResponse, ApiError, RandomizeUnitsRequest>({
    mutationFn: (request) => randomizeUnits(request),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.experiments.assignments(variables.experiment_id),
      });
      queryClient.invalidateQueries({
        queryKey: queryKeys.experiments.enrollmentStats(variables.experiment_id),
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
  options?: Omit<UseMutationOptions<EnrollmentResponse, ApiError, EnrollUnitRequest>, 'mutationFn'>
) {
  const queryClient = useQueryClient();

  return useMutation<EnrollmentResponse, ApiError, EnrollUnitRequest>({
    mutationFn: (request) => enrollUnit(request),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.experiments.assignments(variables.experiment_id),
      });
      queryClient.invalidateQueries({
        queryKey: queryKeys.experiments.enrollmentStats(variables.experiment_id),
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
  options?: Omit<UseMutationOptions<EnrollmentResponse, ApiError, WithdrawUnitRequest>, 'mutationFn'>
) {
  const queryClient = useQueryClient();

  return useMutation<EnrollmentResponse, ApiError, WithdrawUnitRequest>({
    mutationFn: (request) => withdrawUnit(request),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.experiments.assignments(variables.experiment_id),
      });
      queryClient.invalidateQueries({
        queryKey: queryKeys.experiments.enrollmentStats(variables.experiment_id),
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
    UseMutationOptions<InterimAnalysisResponse, ApiError, { experimentId: string; analysisPurpose?: string }>,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<InterimAnalysisResponse, ApiError, { experimentId: string; analysisPurpose?: string }>({
    mutationFn: ({ experimentId, analysisPurpose }) => triggerInterimAnalysis(experimentId, analysisPurpose),
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
  options?: Omit<UseMutationOptions<SRMCheckResponse, ApiError, string>, 'mutationFn'>
) {
  const queryClient = useQueryClient();

  return useMutation<SRMCheckResponse, ApiError, string>({
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
      FidelityComparisonResponse,
      ApiError,
      { experimentId: string; comparisonId: string; request: FidelityComparisonUpdateRequest }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    FidelityComparisonResponse,
    ApiError,
    { experimentId: string; comparisonId: string; request: FidelityComparisonUpdateRequest }
  >({
    mutationFn: ({ experimentId, comparisonId, request }) =>
      updateFidelityComparison(experimentId, comparisonId, request),
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
  options?: Omit<UseMutationOptions<MonitoringResponse, ApiError, { checkAllActive?: boolean }>, 'mutationFn'>
) {
  const queryClient = useQueryClient();

  return useMutation<MonitoringResponse, ApiError, { checkAllActive?: boolean }>({
    mutationFn: ({ checkAllActive }) => triggerMonitoring(checkAllActive),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.experiments.all() });
    },
    ...options,
  });
}
