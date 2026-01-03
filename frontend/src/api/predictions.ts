/**
 * Model Predictions API Client
 * ============================
 *
 * TypeScript API client functions for the E2I Model Predictions endpoints.
 * Uses the shared apiClient for consistent error handling and interceptors.
 *
 * Endpoints:
 * - Single and batch predictions via BentoML
 * - Model health checks
 * - Model metadata and status
 *
 * @module api/predictions
 */

import { get, post } from '@/lib/api-client';
import type {
  BatchPredictionRequest,
  BatchPredictionResponse,
  ModelHealthResponse,
  ModelInfoResponse,
  ModelsStatusResponse,
  PredictionRequest,
  PredictionResponse,
} from '@/types/predictions';

// =============================================================================
// PREDICTIONS API ENDPOINTS
// =============================================================================

const MODELS_BASE = '/models';

// =============================================================================
// PREDICTION ENDPOINTS
// =============================================================================

/**
 * Make a single prediction using a model.
 *
 * @param modelName - Name of the model to use
 * @param request - Prediction request with features
 * @returns Prediction result with confidence and metadata
 *
 * @example
 * ```typescript
 * const result = await predict('churn_model', {
 *   features: {
 *     hcp_id: 'HCP001',
 *     territory: 'Northeast',
 *     specialty: 'Oncology',
 *   },
 *   return_probabilities: true,
 * });
 * console.log(`Prediction: ${result.prediction}, Confidence: ${result.confidence}`);
 * ```
 */
export async function predict(
  modelName: string,
  request: PredictionRequest
): Promise<PredictionResponse> {
  return post<PredictionResponse, PredictionRequest>(
    `${MODELS_BASE}/predict/${encodeURIComponent(modelName)}`,
    request
  );
}

/**
 * Make batch predictions using a model.
 *
 * @param modelName - Name of the model to use
 * @param request - Batch request with multiple instances
 * @returns Batch prediction results with success/failure counts
 *
 * @example
 * ```typescript
 * const result = await predictBatch('churn_model', {
 *   instances: [
 *     { features: { hcp_id: 'HCP001', territory: 'Northeast' } },
 *     { features: { hcp_id: 'HCP002', territory: 'Southwest' } },
 *   ],
 * });
 * console.log(`Success: ${result.success_count}/${result.total_count}`);
 * ```
 */
export async function predictBatch(
  modelName: string,
  request: BatchPredictionRequest
): Promise<BatchPredictionResponse> {
  return post<BatchPredictionResponse, BatchPredictionRequest>(
    `${MODELS_BASE}/predict/${encodeURIComponent(modelName)}/batch`,
    request
  );
}

// =============================================================================
// HEALTH & STATUS ENDPOINTS
// =============================================================================

/**
 * Check health of a specific model.
 *
 * @param modelName - Name of the model to check
 * @returns Model health status
 *
 * @example
 * ```typescript
 * const health = await getModelHealth('churn_model');
 * if (health.status !== 'healthy') {
 *   console.warn(`Model unhealthy: ${health.error}`);
 * }
 * ```
 */
export async function getModelHealth(
  modelName: string
): Promise<ModelHealthResponse> {
  return get<ModelHealthResponse>(
    `${MODELS_BASE}/${encodeURIComponent(modelName)}/health`
  );
}

/**
 * Get metadata/info for a specific model.
 *
 * @param modelName - Name of the model
 * @returns Model metadata including version, type, and metrics
 *
 * @example
 * ```typescript
 * const info = await getModelInfo('churn_model');
 * console.log(`Model v${info.version}, Type: ${info.type}`);
 * console.log(`Metrics:`, info.metrics);
 * ```
 */
export async function getModelInfo(
  modelName: string
): Promise<ModelInfoResponse> {
  return get<ModelInfoResponse>(
    `${MODELS_BASE}/${encodeURIComponent(modelName)}/info`
  );
}

/**
 * Get status of all registered models.
 *
 * @param models - Optional list of specific models to check
 * @returns Status of all models with healthy/unhealthy counts
 *
 * @example
 * ```typescript
 * const status = await getModelsStatus();
 * console.log(`Healthy: ${status.healthy_count}/${status.total_models}`);
 *
 * // Check specific models
 * const specificStatus = await getModelsStatus(['churn_model', 'conversion_model']);
 * ```
 */
export async function getModelsStatus(
  models?: string[]
): Promise<ModelsStatusResponse> {
  return get<ModelsStatusResponse>(`${MODELS_BASE}/status`, {
    models: models?.join(','),
  });
}
