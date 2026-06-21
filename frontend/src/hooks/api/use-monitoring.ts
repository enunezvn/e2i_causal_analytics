/**
 * Monitoring API Query Hooks
 * ==========================
 *
 * TanStack Query hooks for the E2I Model Monitoring API.
 * Provides type-safe data fetching, caching, and state management
 * for drift detection, alerts, model health, and performance tracking.
 *
 * Features:
 * - Automatic caching and background refetching
 * - Loading and error states
 * - Query key management via queryKeys
 * - Polling support for async tasks
 *
 * @module hooks/api/use-monitoring
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import {
  triggerDriftDetection,
  getDriftDetectionStatus,
  getLatestDriftStatus,
  getDriftHistory,
  listAlerts,
  getAlert,
  updateAlert,
  listMonitoringRuns,
  getModelHealth,
  recordPerformance,
  getPerformanceTrend,
  getPerformanceAlerts,
  compareModelPerformance,
  getConfusionMatrix,
  getRocCurve,
  getBrandModelSummary,
  triggerProductionSweep,
  evaluateRetrainingNeed,
  triggerRetraining,
  getRetrainingStatus,
  completeRetraining,
  rollbackRetraining,
  triggerRetrainingSweep,
} from '@/api/monitoring';
import type { BrandModelSummary } from '@/lib/api-schemas';
import type {
  AlertActionRequest,
  AlertItem,
  AlertListParams,
  AlertListResponse,
  CompleteRetrainingRequest,
  ConfusionMatrixResponse,
  DriftDetectionResponse,
  DriftHistoryParams,
  DriftHistoryResponse,
  ModelComparisonResponse,
  ModelHealthSummary,
  MonitoringRunsParams,
  MonitoringRunsResponse,
  PerformanceAlertsResponse,
  PerformanceRecordResponse,
  PerformanceTrendParams,
  PerformanceTrendResponse,
  ProductionSweepResponse,
  RecordPerformanceRequest,
  RetrainingDecisionResponse,
  RetrainingJobResponse,
  RocCurveResponse,
  RollbackRetrainingRequest,
  TaskStatusResponse,
  TriggerDriftDetectionRequest,
  TriggerRetrainingRequest,
} from '@/types/monitoring';
import { AlertStatus } from '@/types/monitoring';
import type { ApiError } from '@/lib/api-client';

// =============================================================================
// DRIFT DETECTION HOOKS
// =============================================================================

/**
 * Hook to get the latest drift status for a model.
 *
 * @param modelId - The model version/ID
 * @param limit - Maximum drift results to return (default: 10)
 * @param options - Additional TanStack Query options
 * @returns Query result with latest drift status
 *
 * @example
 * ```tsx
 * const { data: drift } = useLatestDriftStatus('propensity_v2.1.0');
 * console.log(`Drift score: ${drift?.overall_drift_score}`);
 * ```
 */
export function useLatestDriftStatus(
  modelId: string,
  limit: number = 10,
  options?: Omit<
    UseQueryOptions<DriftDetectionResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<DriftDetectionResponse, ApiError>({
    // `limit` changes how many drift results come back, so it must be part of
    // the cache key (otherwise a limit=5 read serves a limit=25 cache entry).
    // The mutation-side invalidation uses the bare driftLatest(modelId)
    // prefix, which still matches these limit-suffixed keys.
    queryKey: [...queryKeys.monitoring.driftLatest(modelId), limit] as const,
    queryFn: () => getLatestDriftStatus(modelId, limit),
    enabled: !!modelId,
    ...options,
  });
}

/**
 * Hook to get drift history for a model.
 *
 * @param params - Query parameters with model ID and filters
 * @param options - Additional TanStack Query options
 * @returns Query result with drift history
 *
 * @example
 * ```tsx
 * const { data: history } = useDriftHistory({
 *   model_id: 'propensity_v2.1.0',
 *   days: 30
 * });
 * ```
 */
export function useDriftHistory(
  params: DriftHistoryParams,
  options?: Omit<
    UseQueryOptions<DriftHistoryResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<DriftHistoryResponse, ApiError>({
    queryKey: [
      ...queryKeys.monitoring.driftHistory(params.model_id),
      params.feature_name,
      params.days,
      params.limit,
    ],
    queryFn: () => getDriftHistory(params),
    enabled: !!params.model_id,
    ...options,
  });
}

/**
 * Hook to poll drift detection task status.
 *
 * @param taskId - Celery task ID
 * @param options - Additional TanStack Query options
 * @returns Query result with task status
 *
 * @example
 * ```tsx
 * const { data: status } = useDriftTaskStatus('task_abc123', {
 *   refetchInterval: (data) => (data?.ready ? false : 2000)
 * });
 * ```
 */
export function useDriftTaskStatus(
  taskId: string | null,
  options?: Omit<
    UseQueryOptions<TaskStatusResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<TaskStatusResponse, ApiError>({
    queryKey: queryKeys.monitoring.driftStatus(taskId || ''),
    queryFn: () => getDriftDetectionStatus(taskId!),
    enabled: !!taskId,
    // Poll every 2 seconds until task is ready
    refetchInterval: (query) =>
      query.state.data?.ready ? false : 2000,
    ...options,
  });
}

/**
 * Hook for triggering drift detection.
 *
 * Uses mutation pattern since detection can be expensive.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: detectDrift, isPending } = useTriggerDriftDetection();
 *
 * detectDrift({
 *   model_id: 'propensity_v2.1.0',
 *   time_window: '7d'
 * });
 * ```
 */
export function useTriggerDriftDetection(
  options?: Omit<
    UseMutationOptions<
      DriftDetectionResponse,
      ApiError,
      { request: TriggerDriftDetectionRequest; asyncMode?: boolean }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();
  // #324 — compose caller's onSuccess with the hook's invalidation so
  // consumers (e.g. DataQuality.tsx) can add their own toasts without losing
  // the cache-invalidation contract this hook documents. Same idea for
  // onError so caller hooks don't silently override mutation-level error
  // handling either.
  const callerOnSuccess = options?.onSuccess;
  const callerOnError = options?.onError;

  return useMutation<
    DriftDetectionResponse,
    ApiError,
    { request: TriggerDriftDetectionRequest; asyncMode?: boolean }
  >({
    ...options,
    mutationFn: ({ request, asyncMode = true }) =>
      triggerDriftDetection(request, asyncMode),
    onSuccess: (data, variables, onMutateResult, context) => {
      // Invalidate drift queries for this model
      queryClient.invalidateQueries({
        queryKey: queryKeys.monitoring.driftLatest(variables.request.model_id),
      });
      queryClient.invalidateQueries({
        queryKey: queryKeys.monitoring.driftHistory(variables.request.model_id),
      });
      callerOnSuccess?.(data, variables, onMutateResult, context);
    },
    onError: (error, variables, onMutateResult, context) => {
      callerOnError?.(error, variables, onMutateResult, context);
    },
  });
}

// =============================================================================
// ALERT HOOKS
// =============================================================================

/**
 * Hook to list drift alerts.
 *
 * @param params - Optional filters for model, status, severity
 * @param options - Additional TanStack Query options
 * @returns Query result with alerts
 *
 * @example
 * ```tsx
 * const { data: alerts } = useAlerts({ status: AlertStatus.ACTIVE });
 * console.log(`Active alerts: ${alerts?.active_count}`);
 * ```
 */
export function useAlerts(
  params?: AlertListParams,
  options?: Omit<
    UseQueryOptions<AlertListResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<AlertListResponse, ApiError>({
    queryKey: [
      ...queryKeys.monitoring.alerts(),
      params?.model_id,
      params?.status,
      params?.severity,
      params?.limit,
    ],
    queryFn: () => listAlerts(params),
    // Alerts should be relatively fresh
    staleTime: 60 * 1000, // 1 minute
    ...options,
  });
}

/**
 * Hook to fetch the per-brand gold-standard model-performance summary.
 *
 * Drives the Home "Model Accuracy" tile (per-brand average accuracy). Pass no
 * brand (or 'All') for the all-12-model average; `available=false` => honest
 * empty (no gold-standard models recorded).
 */
export function useBrandModelSummary(
  brand?: string,
  options?: Omit<
    UseQueryOptions<BrandModelSummary, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<BrandModelSummary, ApiError>({
    queryKey: queryKeys.monitoring.brandSummary(brand ?? 'all'),
    queryFn: () => getBrandModelSummary(brand),
    staleTime: 5 * 60 * 1000, // 5 minutes — eval metrics are stable
    ...options,
  });
}

/**
 * Hook to get a specific alert.
 *
 * @param alertId - Alert UUID
 * @param options - Additional TanStack Query options
 * @returns Query result with alert details
 *
 * @example
 * ```tsx
 * const { data: alert } = useAlert('alert-uuid-123');
 * ```
 */
export function useAlert(
  alertId: string,
  options?: Omit<
    UseQueryOptions<AlertItem, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<AlertItem, ApiError>({
    queryKey: queryKeys.monitoring.alert(alertId),
    queryFn: () => getAlert(alertId),
    enabled: !!alertId,
    ...options,
  });
}

/**
 * Hook for updating an alert (acknowledge, resolve, snooze).
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: updateAlertAction } = useUpdateAlert();
 *
 * updateAlertAction({
 *   alertId: 'alert-uuid-123',
 *   request: { action: AlertAction.ACKNOWLEDGE, user_id: 'user_123' }
 * });
 * ```
 */
/**
 * Context captured during the optimistic update for rollback on error.
 *
 * `previousAlertsLists` snapshots EVERY param-suffixed alerts-list cache entry
 * (see {@link useAlerts}, which stores lists under
 * `[...alerts(), model_id, status, severity, limit]`), not just the bare
 * `alerts()` key — otherwise the optimistic update would miss the real cache
 * entries and the UI would show stale alert statuses.
 */
interface UpdateAlertContext {
  previousAlert: AlertItem | undefined;
  previousAlertsLists: Array<[readonly unknown[], AlertListResponse | undefined]>;
}

/**
 * Type guard distinguishing an alerts-LIST cache entry (AlertListResponse) from
 * an individual-ALERT cache entry (AlertItem). Both live under the
 * `[...alerts()]` key prefix, so a prefix match alone is ambiguous.
 */
function isAlertListResponse(value: unknown): value is AlertListResponse {
  return (
    typeof value === 'object' &&
    value !== null &&
    Array.isArray((value as AlertListResponse).alerts)
  );
}

export function useUpdateAlert(
  options?: Omit<
    UseMutationOptions<
      AlertItem,
      ApiError,
      { alertId: string; request: AlertActionRequest },
      UpdateAlertContext
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    AlertItem,
    ApiError,
    { alertId: string; request: AlertActionRequest },
    UpdateAlertContext
  >({
    mutationFn: ({ alertId, request }) => updateAlert(alertId, request),

    // Optimistic update: Immediately update the alert before the server responds
    onMutate: async (variables) => {
      // Cancel any outgoing refetches to prevent overwriting our optimistic
      // update. cancelQueries matches by key PREFIX, so a single call covers
      // all param-suffixed alerts lists; the individual alert is cancelled too.
      await queryClient.cancelQueries({
        queryKey: queryKeys.monitoring.alert(variables.alertId),
      });
      await queryClient.cancelQueries({
        queryKey: queryKeys.monitoring.alerts(),
      });

      // Snapshot the previous individual alert for rollback
      const previousAlert = queryClient.getQueryData<AlertItem>(
        queryKeys.monitoring.alert(variables.alertId)
      );

      // Snapshot EVERY alerts-list cache entry under the alerts() prefix. The
      // `alert(id)` cache shares this prefix, so filter to list responses only.
      const previousAlertsLists = queryClient
        .getQueriesData<AlertListResponse>({
          queryKey: queryKeys.monitoring.alerts(),
        })
        .filter(([, data]) => isAlertListResponse(data)) as Array<
        [readonly unknown[], AlertListResponse | undefined]
      >;

      // Determine new status based on action
      const statusMap: Record<string, AlertStatus> = {
        acknowledge: AlertStatus.ACKNOWLEDGED,
        resolve: AlertStatus.RESOLVED,
        snooze: AlertStatus.SNOOZED,
      };
      const newStatus = statusMap[variables.request.action];
      const now = new Date().toISOString();

      // Optimistically update individual alert
      if (previousAlert) {
        const optimisticAlert: AlertItem = {
          ...previousAlert,
          status: newStatus,
          ...(variables.request.action === 'acknowledge' && {
            acknowledged_at: now,
            acknowledged_by: variables.request.user_id,
          }),
          ...(variables.request.action === 'resolve' && {
            resolved_at: now,
            resolved_by: variables.request.user_id,
          }),
        };
        queryClient.setQueryData(
          queryKeys.monitoring.alert(variables.alertId),
          optimisticAlert
        );
      }

      // Optimistically update EVERY param-suffixed alerts list cache entry.
      for (const [queryKey, previousAlerts] of previousAlertsLists) {
        if (!previousAlerts) continue;
        const targetWasActive = previousAlerts.alerts.some(
          (alert) =>
            alert.id === variables.alertId &&
            alert.status === AlertStatus.ACTIVE
        );
        const optimisticAlerts: AlertListResponse = {
          ...previousAlerts,
          alerts: previousAlerts.alerts.map((alert) =>
            alert.id === variables.alertId
              ? {
                  ...alert,
                  status: newStatus,
                  ...(variables.request.action === 'acknowledge' && {
                    acknowledged_at: now,
                    acknowledged_by: variables.request.user_id,
                  }),
                  ...(variables.request.action === 'resolve' && {
                    resolved_at: now,
                    resolved_by: variables.request.user_id,
                  }),
                }
              : alert
          ),
          // Only decrement when this alert was actually counted as active and
          // is transitioning away from ACTIVE — keeps the count correct per list.
          active_count:
            targetWasActive && newStatus !== AlertStatus.ACTIVE
              ? Math.max(0, previousAlerts.active_count - 1)
              : previousAlerts.active_count,
        };
        queryClient.setQueryData(queryKey, optimisticAlerts);
      }

      // Return context with the snapshotted values
      return { previousAlert, previousAlertsLists };
    },

    // Rollback on error
    onError: (_error, variables, context) => {
      if (context?.previousAlert) {
        queryClient.setQueryData(
          queryKeys.monitoring.alert(variables.alertId),
          context.previousAlert
        );
      }
      // Restore every previously-snapshotted alerts-list cache entry.
      for (const [queryKey, previousAlerts] of context?.previousAlertsLists ??
        []) {
        queryClient.setQueryData(queryKey, previousAlerts);
      }
    },

    // Update cache with server response on success
    onSuccess: (data, variables) => {
      queryClient.setQueryData(
        queryKeys.monitoring.alert(variables.alertId),
        data
      );
    },

    // Always refetch after error or success to ensure consistency.
    // invalidateQueries matches by key prefix, so this covers all
    // param-suffixed alerts lists.
    onSettled: () => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.monitoring.alerts(),
      });
    },
    ...options,
  });
}

// =============================================================================
// MONITORING RUNS HOOKS
// =============================================================================

/**
 * Hook to list monitoring runs.
 *
 * @param params - Optional filters for model and time range
 * @param options - Additional TanStack Query options
 * @returns Query result with monitoring runs
 *
 * @example
 * ```tsx
 * const { data: runs } = useMonitoringRuns({ days: 7 });
 * console.log(`Total runs: ${runs?.total_runs}`);
 * ```
 */
export function useMonitoringRuns(
  params?: MonitoringRunsParams,
  options?: Omit<
    UseQueryOptions<MonitoringRunsResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<MonitoringRunsResponse, ApiError>({
    queryKey: [
      ...queryKeys.monitoring.runs(),
      params?.model_id,
      params?.days,
      params?.limit,
    ],
    queryFn: () => listMonitoringRuns(params),
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
}

// =============================================================================
// MODEL HEALTH HOOKS
// =============================================================================

/**
 * Hook to get model health summary.
 *
 * @param modelId - Model version/ID
 * @param options - Additional TanStack Query options
 * @returns Query result with health summary
 *
 * @example
 * ```tsx
 * const { data: health } = useModelHealth('propensity_v2.1.0');
 * if (health?.overall_health === 'critical') {
 *   console.log('Model needs attention:', health.recommendations);
 * }
 * ```
 */
export function useModelHealth(
  modelId: string,
  options?: Omit<
    UseQueryOptions<ModelHealthSummary, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<ModelHealthSummary, ApiError>({
    queryKey: queryKeys.monitoring.modelHealth(modelId),
    queryFn: () => getModelHealth(modelId),
    enabled: !!modelId,
    staleTime: 2 * 60 * 1000, // 2 minutes
    ...options,
  });
}

// =============================================================================
// PERFORMANCE TRACKING HOOKS
// =============================================================================

/**
 * Hook to get performance trend for a model.
 *
 * @param params - Query parameters with model ID and metric
 * @param options - Additional TanStack Query options
 * @returns Query result with performance trend
 *
 * @example
 * ```tsx
 * const { data: trend } = usePerformanceTrend({
 *   model_id: 'propensity_v2.1.0',
 *   metric_name: 'accuracy',
 *   days: 30
 * });
 * ```
 */
export function usePerformanceTrend(
  params: PerformanceTrendParams,
  options?: Omit<
    UseQueryOptions<PerformanceTrendResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<PerformanceTrendResponse, ApiError>({
    queryKey: [
      ...queryKeys.monitoring.performanceTrend(params.model_id),
      params.metric_name,
      params.days,
    ],
    queryFn: () => getPerformanceTrend(params),
    enabled: !!params.model_id,
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
}

/**
 * Hook to get performance alerts for a model.
 *
 * @param modelId - Model version/ID
 * @param options - Additional TanStack Query options
 * @returns Query result with performance alerts
 *
 * @example
 * ```tsx
 * const { data: alerts } = usePerformanceAlerts('propensity_v2.1.0');
 * ```
 */
export function usePerformanceAlerts(
  modelId: string,
  options?: Omit<
    UseQueryOptions<PerformanceAlertsResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<PerformanceAlertsResponse, ApiError>({
    queryKey: queryKeys.monitoring.performanceAlerts(modelId),
    queryFn: () => getPerformanceAlerts(modelId),
    enabled: !!modelId,
    staleTime: 2 * 60 * 1000, // 2 minutes
    ...options,
  });
}

/**
 * Hook to compare performance between two models.
 *
 * @param modelId - First model version
 * @param otherModelId - Second model version
 * @param metricName - Metric to compare (default: 'accuracy')
 * @param options - Additional TanStack Query options
 * @returns Query result with comparison
 *
 * @example
 * ```tsx
 * const { data: comparison } = useModelComparison(
 *   'propensity_v2.1.0',
 *   'propensity_v2.0.0',
 *   'accuracy'
 * );
 * ```
 */
export function useModelComparison(
  modelId: string,
  otherModelId: string,
  metricName: string = 'accuracy',
  options?: Omit<
    UseQueryOptions<ModelComparisonResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<ModelComparisonResponse, ApiError>({
    queryKey: [
      ...queryKeys.monitoring.performanceCompare(modelId, otherModelId),
      metricName,
    ],
    queryFn: () => compareModelPerformance(modelId, otherModelId, metricName),
    enabled: !!modelId && !!otherModelId,
    staleTime: 10 * 60 * 1000, // 10 minutes
    ...options,
  });
}

/**
 * Hook to get the latest holdout confusion matrix for a model.
 *
 * Returns `available: false` (honest empty state) when none is recorded yet.
 *
 * @example
 * ```tsx
 * const { data: cm } = useConfusionMatrix('initiation_goldstd');
 * ```
 */
export function useConfusionMatrix(
  modelId: string,
  options?: Omit<UseQueryOptions<ConfusionMatrixResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<ConfusionMatrixResponse, ApiError>({
    queryKey: queryKeys.monitoring.performanceConfusion(modelId),
    queryFn: () => getConfusionMatrix(modelId),
    enabled: !!modelId,
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
}

/**
 * Hook to get the latest holdout ROC curve for a model.
 *
 * Returns `available: false` (honest empty state) when none is recorded yet.
 *
 * @example
 * ```tsx
 * const { data: roc } = useRocCurve('initiation_goldstd');
 * ```
 */
export function useRocCurve(
  modelId: string,
  options?: Omit<UseQueryOptions<RocCurveResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<RocCurveResponse, ApiError>({
    queryKey: queryKeys.monitoring.performanceRoc(modelId),
    queryFn: () => getRocCurve(modelId),
    enabled: !!modelId,
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
}

/**
 * Hook for recording performance metrics.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: record } = useRecordPerformance();
 *
 * record({
 *   request: {
 *     model_id: 'propensity_v2.1.0',
 *     predictions: [1, 0, 1],
 *     actuals: [1, 0, 0]
 *   }
 * });
 * ```
 */
export function useRecordPerformance(
  options?: Omit<
    UseMutationOptions<
      PerformanceRecordResponse,
      ApiError,
      { request: RecordPerformanceRequest; asyncMode?: boolean }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    PerformanceRecordResponse,
    ApiError,
    { request: RecordPerformanceRequest; asyncMode?: boolean }
  >({
    mutationFn: ({ request, asyncMode = true }) =>
      recordPerformance(request, asyncMode),
    onSuccess: (_data, variables) => {
      // Invalidate performance queries for this model
      queryClient.invalidateQueries({
        queryKey: queryKeys.monitoring.performanceTrend(
          variables.request.model_id
        ),
      });
      queryClient.invalidateQueries({
        queryKey: queryKeys.monitoring.performanceAlerts(
          variables.request.model_id
        ),
      });
    },
    ...options,
  });
}

// =============================================================================
// PRODUCTION SWEEP HOOKS
// =============================================================================

/**
 * Hook for triggering production model sweep.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: sweep } = useProductionSweep();
 * sweep({ timeWindow: '7d' });
 * ```
 */
export function useProductionSweep(
  options?: Omit<
    UseMutationOptions<ProductionSweepResponse, ApiError, { timeWindow?: string }>,
    'mutationFn'
  >
) {
  return useMutation<ProductionSweepResponse, ApiError, { timeWindow?: string }>({
    mutationFn: ({ timeWindow = '7d' }) => triggerProductionSweep(timeWindow),
    ...options,
  });
}

// =============================================================================
// RETRAINING HOOKS
// =============================================================================

/**
 * Hook to evaluate retraining need for a model.
 *
 * Uses mutation pattern since evaluation involves analysis.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: evaluate, data: decision } = useEvaluateRetraining();
 * evaluate({ modelId: 'propensity_v2.1.0' });
 * ```
 */
export function useEvaluateRetraining(
  options?: Omit<
    UseMutationOptions<RetrainingDecisionResponse, ApiError, { modelId: string }>,
    'mutationFn'
  >
) {
  return useMutation<RetrainingDecisionResponse, ApiError, { modelId: string }>({
    mutationFn: ({ modelId }) => evaluateRetrainingNeed(modelId),
    ...options,
  });
}

/**
 * Hook for triggering model retraining.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: retrain } = useTriggerRetraining();
 *
 * retrain({
 *   modelId: 'propensity_v2.1.0',
 *   request: { reason: TriggerReason.DATA_DRIFT }
 * });
 * ```
 */
export function useTriggerRetraining(
  options?: Omit<
    UseMutationOptions<
      RetrainingJobResponse,
      ApiError,
      { modelId: string; request: TriggerRetrainingRequest; triggeredBy?: string }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    RetrainingJobResponse,
    ApiError,
    { modelId: string; request: TriggerRetrainingRequest; triggeredBy?: string }
  >({
    mutationFn: ({ modelId, request, triggeredBy = 'ui_user' }) =>
      triggerRetraining(modelId, request, triggeredBy),
    onSuccess: (_data, variables) => {
      // Invalidate model health since retraining affects it
      queryClient.invalidateQueries({
        queryKey: queryKeys.monitoring.modelHealth(variables.modelId),
      });
    },
    ...options,
  });
}

/**
 * Hook to get retraining job status.
 *
 * @param jobId - Retraining job ID
 * @param options - Additional TanStack Query options
 * @returns Query result with job status
 *
 * @example
 * ```tsx
 * const { data: job } = useRetrainingStatus('job-uuid-123');
 * ```
 */
export function useRetrainingStatus(
  jobId: string | null,
  options?: Omit<
    UseQueryOptions<RetrainingJobResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<RetrainingJobResponse, ApiError>({
    queryKey: queryKeys.monitoring.retrainingStatus(jobId || ''),
    queryFn: () => getRetrainingStatus(jobId!),
    enabled: !!jobId,
    // Poll every 5 seconds for in-progress jobs
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === 'pending' || status === 'in_progress') {
        return 5000;
      }
      return false;
    },
    ...options,
  });
}

/**
 * Hook for completing a retraining job.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: complete } = useCompleteRetraining();
 *
 * complete({
 *   jobId: 'job-uuid-123',
 *   request: { performance_after: 0.92, success: true }
 * });
 * ```
 */
export function useCompleteRetraining(
  options?: Omit<
    UseMutationOptions<
      RetrainingJobResponse,
      ApiError,
      { jobId: string; request: CompleteRetrainingRequest }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    RetrainingJobResponse,
    ApiError,
    { jobId: string; request: CompleteRetrainingRequest }
  >({
    mutationFn: ({ jobId, request }) => completeRetraining(jobId, request),
    onSuccess: (data, variables) => {
      // Update cache for this job
      queryClient.setQueryData(
        queryKeys.monitoring.retrainingStatus(variables.jobId),
        data
      );
      // Invalidate model health
      queryClient.invalidateQueries({
        queryKey: queryKeys.monitoring.modelHealth(data.model_version),
      });
    },
    ...options,
  });
}

/**
 * Hook for rolling back a retraining.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: rollback } = useRollbackRetraining();
 *
 * rollback({
 *   jobId: 'job-uuid-123',
 *   request: { reason: 'Performance degradation' }
 * });
 * ```
 */
export function useRollbackRetraining(
  options?: Omit<
    UseMutationOptions<
      RetrainingJobResponse,
      ApiError,
      { jobId: string; request: RollbackRetrainingRequest }
    >,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<
    RetrainingJobResponse,
    ApiError,
    { jobId: string; request: RollbackRetrainingRequest }
  >({
    mutationFn: ({ jobId, request }) => rollbackRetraining(jobId, request),
    onSuccess: (data, variables) => {
      // Update cache for this job
      queryClient.setQueryData(
        queryKeys.monitoring.retrainingStatus(variables.jobId),
        data
      );
      // Invalidate model health
      queryClient.invalidateQueries({
        queryKey: queryKeys.monitoring.modelHealth(data.model_version),
      });
    },
    ...options,
  });
}

/**
 * Hook for triggering retraining sweep for all models.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: sweep } = useRetrainingSweep();
 * sweep();
 * ```
 */
export function useRetrainingSweep(
  options?: Omit<
    UseMutationOptions<ProductionSweepResponse, ApiError, void>,
    'mutationFn'
  >
) {
  return useMutation<ProductionSweepResponse, ApiError, void>({
    mutationFn: () => triggerRetrainingSweep(),
    ...options,
  });
}

// =============================================================================
// PREFETCH HELPERS
// =============================================================================

/**
 * Prefetch model health data.
 *
 * @param modelId - Model version/ID
 *
 * @example
 * ```tsx
 * const queryClient = useQueryClient();
 * prefetchModelHealth(queryClient, 'propensity_v2.1.0');
 * ```
 */
export async function prefetchModelHealth(
  queryClient: ReturnType<typeof useQueryClient>,
  modelId: string
) {
  await queryClient.prefetchQuery({
    queryKey: queryKeys.monitoring.modelHealth(modelId),
    queryFn: () => getModelHealth(modelId),
  });
}

/**
 * Prefetch alerts list.
 *
 * @example
 * ```tsx
 * const queryClient = useQueryClient();
 * prefetchAlerts(queryClient);
 * ```
 */
export async function prefetchAlerts(
  queryClient: ReturnType<typeof useQueryClient>,
  params?: AlertListParams
) {
  await queryClient.prefetchQuery({
    queryKey: [
      ...queryKeys.monitoring.alerts(),
      params?.model_id,
      params?.status,
      params?.severity,
      params?.limit,
    ],
    queryFn: () => listAlerts(params),
  });
}

/**
 * Prefetch latest drift status.
 *
 * @param modelId - Model version/ID
 *
 * @example
 * ```tsx
 * const queryClient = useQueryClient();
 * prefetchLatestDriftStatus(queryClient, 'propensity_v2.1.0');
 * ```
 */
export async function prefetchLatestDriftStatus(
  queryClient: ReturnType<typeof useQueryClient>,
  modelId: string,
  limit: number = 10
) {
  await queryClient.prefetchQuery({
    // Must mirror the useLatestDriftStatus read key (model + limit) so the
    // prefetched data is actually consumed by the hook.
    queryKey: [...queryKeys.monitoring.driftLatest(modelId), limit] as const,
    queryFn: () => getLatestDriftStatus(modelId, limit),
  });
}
