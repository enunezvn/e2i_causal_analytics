/**
 * Model Prediction Types
 * ======================
 *
 * TypeScript interfaces for the E2I Model Predictions API.
 * Based on src/api/routes/predictions.py backend schemas.
 *
 * @module types/predictions
 */

// =============================================================================
// ENUMS
// =============================================================================

/**
 * Prediction time horizon
 */
export enum TimeHorizon {
  SHORT_TERM = 'short_term',
  MEDIUM_TERM = 'medium_term',
  LONG_TERM = 'long_term',
}

/**
 * Model health status
 */
export enum ModelHealthStatus {
  HEALTHY = 'healthy',
  UNHEALTHY = 'unhealthy',
  UNKNOWN = 'unknown',
}

// =============================================================================
// REQUEST MODELS
// =============================================================================

/**
 * Request schema for model prediction
 */
export interface PredictionRequest {
  /** Feature dictionary for prediction */
  features: Record<string, unknown>;
  /** Entity ID for feature store lookup (if features not provided) */
  entity_id?: string;
  /** Prediction time horizon */
  time_horizon?: TimeHorizon | string;
  /** Return class probabilities (classification models) */
  return_probabilities?: boolean;
  /** Return prediction intervals (regression models) */
  return_intervals?: boolean;
}

/**
 * Request schema for batch predictions
 */
export interface BatchPredictionRequest {
  /** List of prediction requests */
  instances: PredictionRequest[];
}

// =============================================================================
// RESPONSE MODELS
// =============================================================================

/**
 * Prediction interval for regression models
 */
export interface PredictionInterval {
  /** Lower bound of interval */
  lower: number;
  /** Upper bound of interval */
  upper: number;
}

/**
 * Response schema for model prediction
 */
export interface PredictionResponse {
  /** Name of the model used */
  model_name: string;
  /** Model prediction value */
  prediction: unknown;
  /** Prediction confidence score (0-1) */
  confidence?: number;
  /** Class probabilities (classification only) */
  probabilities?: Record<string, number>;
  /** Prediction interval (regression only) */
  prediction_interval?: PredictionInterval;
  /** Feature importance scores for this prediction */
  feature_importance?: Record<string, number>;
  /** Prediction latency in milliseconds */
  latency_ms: number;
  /** Model version used */
  model_version?: string;
  /** Prediction timestamp (ISO 8601) */
  timestamp: string;
}

/**
 * Response schema for batch predictions
 */
export interface BatchPredictionResponse {
  /** Name of the model used */
  model_name: string;
  /** List of prediction results */
  predictions: PredictionResponse[];
  /** Total number of predictions */
  total_count: number;
  /** Number of successful predictions */
  success_count: number;
  /** Number of failed predictions */
  failed_count: number;
  /** Total processing time in milliseconds */
  total_latency_ms: number;
  /** Batch processing timestamp (ISO 8601) */
  timestamp: string;
}

/**
 * Response schema for single model endpoint health check
 * Note: Named ModelEndpointHealth to avoid conflict with aggregate ModelHealthResponse in health-score.ts
 */
export interface ModelEndpointHealth {
  /** Name of the model */
  model_name: string;
  /** Health status: healthy, unhealthy, unknown */
  status: ModelHealthStatus | string;
  /** Model endpoint URL */
  endpoint: string;
  /** Last health check timestamp (ISO 8601) */
  last_check: string;
  /** Error message if unhealthy */
  error?: string;
}

/**
 * Response schema for all models status
 */
export interface ModelsStatusResponse {
  /** Total number of registered models */
  total_models: number;
  /** Number of healthy models */
  healthy_count: number;
  /** Number of unhealthy models */
  unhealthy_count: number;
  /** Individual model statuses */
  models: ModelEndpointHealth[];
  /** Status check timestamp (ISO 8601) */
  timestamp: string;
}

/**
 * Model info/metadata response
 */
export interface ModelInfoResponse {
  /** Model name */
  name: string;
  /** Model version */
  version?: string;
  /** Model type (classification, regression, etc.) */
  type?: string;
  /** Model description */
  description?: string;
  /** Input feature schema */
  input_schema?: Record<string, unknown>;
  /** Output schema */
  output_schema?: Record<string, unknown>;
  /** Training date */
  trained_at?: string;
  /** Performance metrics */
  metrics?: Record<string, number>;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}
