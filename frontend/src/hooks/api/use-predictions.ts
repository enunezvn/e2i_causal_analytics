/**
 * Model Predictions React Query Hooks
 * ====================================
 *
 * TanStack Query hooks for interacting with the E2I Model Predictions API.
 * Provides caching, deduplication, and mutation support.
 *
 * Features:
 * - Single and batch predictions
 * - Model health monitoring
 * - Model metadata retrieval
 * - Status dashboard support
 *
 * @module hooks/api/use-predictions
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import {
  predict,
  predictBatch,
  getModelHealth,
  getModelInfo,
  getModelsStatus,
} from '@/api/predictions';
import { queryKeys, queryClient as globalQueryClient } from '@/lib/query-client';
import type {
  BatchPredictionRequest,
  BatchPredictionResponse,
  ModelEndpointHealth,
  ModelInfoResponse,
  ModelsStatusResponse,
  PredictionRequest,
  PredictionResponse,
} from '@/types/predictions';

// =============================================================================
// QUERY HOOKS - HEALTH & STATUS
// =============================================================================

/**
 * Hook to fetch health status of a specific model.
 *
 * @param modelName - Name of the model to check
 * @param options - Additional React Query options
 * @returns Query result with model health status
 *
 * @example
 * ```tsx
 * const { data: health, isLoading } = useModelHealth('churn_model');
 *
 * return (
 *   <StatusBadge
 *     status={health?.status}
 *     error={health?.error}
 *   />
 * );
 * ```
 */
export function useModelHealth(
  modelName: string,
  options?: Omit<UseQueryOptions<ModelEndpointHealth, Error>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.predictions.modelHealth(modelName),
    queryFn: () => getModelHealth(modelName),
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 60 * 1000, // Auto-refresh every minute
    enabled: !!modelName,
    ...options,
  });
}

/**
 * Hook to fetch model metadata/info.
 *
 * @param modelName - Name of the model
 * @param options - Additional React Query options
 * @returns Query result with model metadata
 *
 * @example
 * ```tsx
 * const { data: info } = useModelInfo('churn_model');
 *
 * return (
 *   <ModelCard
 *     name={info?.name}
 *     version={info?.version}
 *     type={info?.type}
 *     metrics={info?.metrics}
 *   />
 * );
 * ```
 */
export function useModelInfo(
  modelName: string,
  options?: Omit<UseQueryOptions<ModelInfoResponse, Error>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.predictions.modelInfo(modelName),
    queryFn: () => getModelInfo(modelName),
    staleTime: 5 * 60 * 1000, // 5 minutes - model info rarely changes
    enabled: !!modelName,
    ...options,
  });
}

/**
 * Hook to fetch status of all models.
 *
 * @param models - Optional list of specific models to check
 * @param options - Additional React Query options
 * @returns Query result with all models status
 *
 * @example
 * ```tsx
 * const { data: status } = useModelsStatus();
 *
 * return (
 *   <StatusDashboard
 *     total={status?.total_models}
 *     healthy={status?.healthy_count}
 *     unhealthy={status?.unhealthy_count}
 *     models={status?.models}
 *   />
 * );
 * ```
 */
export function useModelsStatus(
  models?: string[],
  options?: Omit<UseQueryOptions<ModelsStatusResponse, Error>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.predictions.modelsStatus(),
    queryFn: () => getModelsStatus(models),
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 60 * 1000, // Auto-refresh every minute
    ...options,
  });
}

// =============================================================================
// MUTATION HOOKS - PREDICTIONS
// =============================================================================

/**
 * Hook interface for prediction request with model name.
 */
interface PredictMutationVariables {
  modelName: string;
  request: PredictionRequest;
}

/**
 * Hook to make a single prediction.
 *
 * @param options - Additional React Query mutation options
 * @returns Mutation object for making predictions
 *
 * @example
 * ```tsx
 * const { mutate: makePrediction, isPending, data } = usePredict({
 *   onSuccess: (result) => {
 *     toast.success(`Prediction: ${result.prediction}`);
 *   }
 * });
 *
 * const handlePredict = () => {
 *   makePrediction({
 *     modelName: 'churn_model',
 *     request: {
 *       features: { hcp_id: 'HCP001', territory: 'Northeast' },
 *       return_probabilities: true,
 *     },
 *   });
 * };
 * ```
 */
export function usePredict(
  options?: Omit<UseMutationOptions<PredictionResponse, Error, PredictMutationVariables>, 'mutationFn'>
) {
  return useMutation({
    mutationFn: ({ modelName, request }: PredictMutationVariables) =>
      predict(modelName, request),
    ...options,
  });
}

/**
 * Hook interface for batch prediction request with model name.
 */
interface BatchPredictMutationVariables {
  modelName: string;
  request: BatchPredictionRequest;
}

/**
 * Hook to make batch predictions.
 *
 * @param options - Additional React Query mutation options
 * @returns Mutation object for batch predictions
 *
 * @example
 * ```tsx
 * const { mutate: batchPredict, isPending } = useBatchPredict({
 *   onSuccess: (result) => {
 *     toast.success(`Processed ${result.success_count}/${result.total_count} predictions`);
 *   }
 * });
 *
 * const handleBatchPredict = () => {
 *   batchPredict({
 *     modelName: 'churn_model',
 *     request: {
 *       instances: selectedHCPs.map(hcp => ({
 *         features: { hcp_id: hcp.id, territory: hcp.territory },
 *       })),
 *     },
 *   });
 * };
 * ```
 */
export function useBatchPredict(
  options?: Omit<UseMutationOptions<BatchPredictionResponse, Error, BatchPredictMutationVariables>, 'mutationFn'>
) {
  return useMutation({
    mutationFn: ({ modelName, request }: BatchPredictMutationVariables) =>
      predictBatch(modelName, request),
    ...options,
  });
}

// =============================================================================
// PREFETCH HELPERS
// =============================================================================

/**
 * Prefetch model health for faster navigation.
 *
 * @param modelName - Model name to prefetch health for
 *
 * @example
 * ```tsx
 * <ModelRow
 *   onMouseEnter={() => prefetchModelHealth(model.name)}
 * />
 * ```
 */
export function prefetchModelHealth(modelName: string): Promise<void> {
  return globalQueryClient.prefetchQuery({
    queryKey: queryKeys.predictions.modelHealth(modelName),
    queryFn: () => getModelHealth(modelName),
    staleTime: 30 * 1000,
  });
}

/**
 * Prefetch model info for faster detail view loading.
 *
 * @param modelName - Model name to prefetch info for
 *
 * @example
 * ```tsx
 * <ModelCard
 *   onMouseEnter={() => prefetchModelInfo(model.name)}
 * />
 * ```
 */
export function prefetchModelInfo(modelName: string): Promise<void> {
  return globalQueryClient.prefetchQuery({
    queryKey: queryKeys.predictions.modelInfo(modelName),
    queryFn: () => getModelInfo(modelName),
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Prefetch all models status.
 *
 * @example
 * ```tsx
 * useEffect(() => {
 *   prefetchModelsStatus();
 * }, []);
 * ```
 */
export function prefetchModelsStatus(): Promise<void> {
  return globalQueryClient.prefetchQuery({
    queryKey: queryKeys.predictions.modelsStatus(),
    queryFn: () => getModelsStatus(),
    staleTime: 30 * 1000,
  });
}

// =============================================================================
// COMBINED HOOKS
// =============================================================================

/**
 * Hook to get complete model info (health + metadata).
 *
 * @param modelName - Model name
 * @returns Combined health and info data
 *
 * @example
 * ```tsx
 * const { health, info, isLoading } = useModelDetail('churn_model');
 *
 * return (
 *   <ModelDetailCard
 *     status={health?.status}
 *     version={info?.version}
 *     metrics={info?.metrics}
 *     loading={isLoading}
 *   />
 * );
 * ```
 */
export function useModelDetail(modelName: string) {
  const healthQuery = useModelHealth(modelName);
  const infoQuery = useModelInfo(modelName);

  return {
    health: healthQuery.data,
    info: infoQuery.data,
    isLoading: healthQuery.isLoading || infoQuery.isLoading,
    error: healthQuery.error || infoQuery.error,
    isHealthLoading: healthQuery.isLoading,
    isInfoLoading: infoQuery.isLoading,
    refetch: () => {
      healthQuery.refetch();
      infoQuery.refetch();
    },
  };
}

/**
 * Hook to invalidate model-related caches.
 *
 * Useful after model updates or retraining.
 *
 * @returns Function to invalidate model caches
 *
 * @example
 * ```tsx
 * const invalidateModels = useInvalidateModelCache();
 *
 * const handleRetrainComplete = () => {
 *   invalidateModels(); // Refresh all model data
 *   toast.success('Model retrained and cache refreshed');
 * };
 * ```
 */
export function useInvalidateModelCache() {
  const queryClient = useQueryClient();

  return (modelName?: string) => {
    if (modelName) {
      // Invalidate specific model
      queryClient.invalidateQueries({
        queryKey: queryKeys.predictions.modelHealth(modelName),
      });
      queryClient.invalidateQueries({
        queryKey: queryKeys.predictions.modelInfo(modelName),
      });
    } else {
      // Invalidate all predictions-related queries
      queryClient.invalidateQueries({
        queryKey: queryKeys.predictions.all(),
      });
    }
  };
}
