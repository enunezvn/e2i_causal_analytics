/**
 * Explain API Query Hooks
 * =======================
 *
 * TanStack Query hooks for the E2I Model Interpretability API.
 * Provides type-safe data fetching, caching, and state management
 * for SHAP explanations.
 *
 * Features:
 * - Automatic caching and background refetching
 * - Loading and error states
 * - Query key management via queryKeys
 *
 * @module hooks/api/use-explain
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import {
  getExplanation,
  getBatchExplanations,
  getExplanationHistory,
  listExplainableModels,
  getExplainHealth,
} from '@/api/explain';
import type {
  BatchExplainRequest,
  BatchExplainResponse,
  ExplainHealthResponse,
  ExplainRequest,
  ExplainResponse,
  ExplanationHistoryParams,
  ExplanationHistoryResponse,
  ListExplainableModelsResponse,
  ModelType,
} from '@/types/explain';
import type { ApiError } from '@/lib/api-client';

// =============================================================================
// QUERY HOOKS
// =============================================================================

/**
 * Hook to list all explainable models.
 *
 * @param options - Additional TanStack Query options
 * @returns Query result with available models
 *
 * @example
 * ```tsx
 * const { data: models } = useExplainableModels();
 * console.log(`Available models: ${models?.total_models}`);
 * ```
 */
export function useExplainableModels(
  options?: Omit<
    UseQueryOptions<ListExplainableModelsResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<ListExplainableModelsResponse, ApiError>({
    queryKey: queryKeys.explain.models(),
    queryFn: listExplainableModels,
    // Model list doesn't change often
    staleTime: 30 * 60 * 1000, // 30 minutes
    ...options,
  });
}

/**
 * Hook to fetch explanation history for a patient.
 *
 * @param patientId - The patient identifier
 * @param modelType - Optional model type filter
 * @param limit - Maximum results (default 10)
 * @param options - Additional TanStack Query options
 * @returns Query result with historical explanations
 *
 * @example
 * ```tsx
 * const { data: history } = useExplanationHistory('patient_123', ModelType.PROPENSITY);
 * ```
 */
export function useExplanationHistory(
  patientId: string,
  modelType?: ModelType,
  limit?: number,
  options?: Omit<
    UseQueryOptions<ExplanationHistoryResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  const params: ExplanationHistoryParams = {
    patient_id: patientId,
    model_type: modelType,
    limit,
  };

  return useQuery<ExplanationHistoryResponse, ApiError>({
    queryKey: [...queryKeys.explain.history(patientId), modelType, limit],
    queryFn: () => getExplanationHistory(params),
    enabled: !!patientId,
    ...options,
  });
}

/**
 * Hook to check interpretability service health.
 *
 * @param options - Additional TanStack Query options
 * @returns Query result with health status
 *
 * @example
 * ```tsx
 * const { data: health } = useExplainHealth();
 * const isHealthy = health?.status === 'healthy';
 * ```
 */
export function useExplainHealth(
  options?: Omit<
    UseQueryOptions<ExplainHealthResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<ExplainHealthResponse, ApiError>({
    queryKey: queryKeys.explain.health(),
    queryFn: getExplainHealth,
    // Health checks should be fresh
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 60 * 1000, // Refetch every minute
    ...options,
  });
}

// =============================================================================
// MUTATION HOOKS
// =============================================================================

/**
 * Hook for getting real-time SHAP explanations.
 *
 * Uses mutation pattern since explanations can be expensive to compute
 * and we want explicit control over when they run.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: explain, data: explanation, isPending } = useExplain();
 *
 * explain({
 *   patient_id: 'patient_123',
 *   model_type: ModelType.PROPENSITY,
 *   format: ExplanationFormat.TOP_K,
 *   top_k: 5
 * });
 * ```
 */
export function useExplain(
  options?: Omit<
    UseMutationOptions<ExplainResponse, ApiError, ExplainRequest>,
    'mutationFn'
  >
) {
  return useMutation<ExplainResponse, ApiError, ExplainRequest>({
    mutationFn: getExplanation,
    ...options,
  });
}

/**
 * Hook for batch SHAP explanations.
 *
 * Processes multiple patients in a single request for efficiency.
 *
 * @param options - Additional TanStack Mutation options
 * @returns Mutation function and state
 *
 * @example
 * ```tsx
 * const { mutate: explainBatch, data: results, isPending } = useBatchExplain();
 *
 * explainBatch({
 *   requests: [
 *     { patient_id: 'p1', model_type: ModelType.PROPENSITY },
 *     { patient_id: 'p2', model_type: ModelType.PROPENSITY }
 *   ],
 *   parallel: true
 * });
 * ```
 */
export function useBatchExplain(
  options?: Omit<
    UseMutationOptions<BatchExplainResponse, ApiError, BatchExplainRequest>,
    'mutationFn'
  >
) {
  return useMutation<BatchExplainResponse, ApiError, BatchExplainRequest>({
    mutationFn: getBatchExplanations,
    ...options,
  });
}

// =============================================================================
// PREFETCH HELPERS
// =============================================================================

/**
 * Prefetch explainable models list.
 *
 * @example
 * ```tsx
 * const queryClient = useQueryClient();
 * prefetchExplainableModels(queryClient);
 * ```
 */
export async function prefetchExplainableModels(
  queryClient: ReturnType<typeof useQueryClient>
) {
  await queryClient.prefetchQuery({
    queryKey: queryKeys.explain.models(),
    queryFn: listExplainableModels,
  });
}

/**
 * Prefetch explanation history for a patient.
 *
 * @param patientId - The patient ID to prefetch history for
 *
 * @example
 * ```tsx
 * prefetchExplanationHistory(queryClient, 'patient_123');
 * ```
 */
export async function prefetchExplanationHistory(
  queryClient: ReturnType<typeof useQueryClient>,
  patientId: string,
  modelType?: ModelType,
  limit?: number
) {
  const params: ExplanationHistoryParams = {
    patient_id: patientId,
    model_type: modelType,
    limit,
  };

  await queryClient.prefetchQuery({
    queryKey: [...queryKeys.explain.history(patientId), modelType, limit],
    queryFn: () => getExplanationHistory(params),
  });
}
