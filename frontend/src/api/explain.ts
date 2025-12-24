/**
 * Explain API Client
 * ==================
 *
 * TypeScript API client functions for the E2I Model Interpretability endpoints.
 * Uses the shared apiClient for consistent error handling and interceptors.
 *
 * Endpoints:
 * - Real-time SHAP explanations
 * - Batch explanations
 * - Explanation history
 * - Model listing
 * - Health checks
 *
 * @module api/explain
 */

import { get, post } from '@/lib/api-client';
import type {
  BatchExplainRequest,
  BatchExplainResponse,
  ExplainHealthResponse,
  ExplainRequest,
  ExplainResponse,
  ExplanationHistoryParams,
  ExplanationHistoryResponse,
  ListExplainableModelsResponse,
} from '@/types/explain';

// =============================================================================
// EXPLAIN API ENDPOINTS
// =============================================================================

const EXPLAIN_BASE = '/explain';

// =============================================================================
// EXPLANATION ENDPOINTS
// =============================================================================

/**
 * Get real-time SHAP explanation for a patient prediction.
 *
 * Computes feature contributions and optionally generates
 * natural language explanations.
 *
 * @param request - Explanation request with patient and model info
 * @returns SHAP explanation with feature contributions
 *
 * @example
 * ```typescript
 * const explanation = await getExplanation({
 *   patient_id: 'patient_123',
 *   model_type: ModelType.PROPENSITY,
 *   format: ExplanationFormat.TOP_K,
 *   top_k: 5
 * });
 * ```
 */
export async function getExplanation(
  request: ExplainRequest
): Promise<ExplainResponse> {
  return post<ExplainResponse, ExplainRequest>(`${EXPLAIN_BASE}`, request);
}

/**
 * Get batch SHAP explanations for multiple patients.
 *
 * Processes up to 50 patients per batch with optional parallelism.
 *
 * @param request - Batch request with multiple patient/model pairs
 * @returns Batch results with successful and failed explanations
 *
 * @example
 * ```typescript
 * const batch = await getBatchExplanations({
 *   requests: [
 *     { patient_id: 'p1', model_type: ModelType.PROPENSITY },
 *     { patient_id: 'p2', model_type: ModelType.PROPENSITY }
 *   ],
 *   parallel: true
 * });
 * console.log(`Successful: ${batch.successful}, Failed: ${batch.failed}`);
 * ```
 */
export async function getBatchExplanations(
  request: BatchExplainRequest
): Promise<BatchExplainResponse> {
  return post<BatchExplainResponse, BatchExplainRequest>(
    `${EXPLAIN_BASE}/batch`,
    request
  );
}

// =============================================================================
// HISTORY ENDPOINTS
// =============================================================================

/**
 * Get explanation history for a patient.
 *
 * Retrieves past SHAP explanations stored for audit trail.
 *
 * @param params - Query parameters with patient ID and optional filters
 * @returns Historical explanations for the patient
 *
 * @example
 * ```typescript
 * const history = await getExplanationHistory({
 *   patient_id: 'patient_123',
 *   model_type: ModelType.PROPENSITY,
 *   limit: 10
 * });
 * ```
 */
export async function getExplanationHistory(
  params: ExplanationHistoryParams
): Promise<ExplanationHistoryResponse> {
  return get<ExplanationHistoryResponse>(
    `${EXPLAIN_BASE}/history/${encodeURIComponent(params.patient_id)}`,
    {
      model_type: params.model_type,
      limit: params.limit,
    }
  );
}

// =============================================================================
// MODEL LISTING ENDPOINTS
// =============================================================================

/**
 * List all explainable models.
 *
 * Returns model types, versions, and explainer configurations.
 *
 * @returns List of available explainable models
 *
 * @example
 * ```typescript
 * const models = await listExplainableModels();
 * console.log(`Available models: ${models.total_models}`);
 * ```
 */
export async function listExplainableModels(): Promise<ListExplainableModelsResponse> {
  return get<ListExplainableModelsResponse>(`${EXPLAIN_BASE}/models`);
}

// =============================================================================
// HEALTH ENDPOINT
// =============================================================================

/**
 * Health check for the interpretability service.
 *
 * Returns status of BentoML, Feast, SHAP explainer, and database.
 *
 * @returns Service health status with dependency checks
 *
 * @example
 * ```typescript
 * const health = await getExplainHealth();
 * if (health.status === 'healthy') {
 *   console.log('Interpretability service is operational');
 * }
 * ```
 */
export async function getExplainHealth(): Promise<ExplainHealthResponse> {
  return get<ExplainHealthResponse>(`${EXPLAIN_BASE}/health`);
}
