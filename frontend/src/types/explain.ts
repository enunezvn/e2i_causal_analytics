/**
 * Model Interpretability Types
 * ============================
 *
 * TypeScript interfaces for the E2I Real-Time Model Interpretability API.
 * Based on src/api/routes/explain.py backend schemas.
 *
 * @module types/explain
 */

// =============================================================================
// ENUMS
// =============================================================================

/**
 * Supported model types for SHAP explanation
 */
export enum ModelType {
  PROPENSITY = 'propensity',
  RISK_STRATIFICATION = 'risk_stratification',
  NEXT_BEST_ACTION = 'next_best_action',
  CHURN_PREDICTION = 'churn_prediction',
}

/**
 * Output format for SHAP explanations
 */
export enum ExplanationFormat {
  /** All SHAP values + metadata */
  FULL = 'full',
  /** Only top K contributing features */
  TOP_K = 'top_k',
  /** NL explanation (requires Claude) */
  NARRATIVE = 'narrative',
  /** Prediction + top 3 features only */
  MINIMAL = 'minimal',
}

// =============================================================================
// FEATURE CONTRIBUTION MODELS
// =============================================================================

/**
 * Single feature's contribution to prediction
 */
export interface FeatureContribution {
  /** Name of the feature */
  feature_name: string;
  /** Actual value of feature for this instance */
  feature_value: unknown;
  /** SHAP contribution to prediction */
  shap_value: number;
  /** positive or negative */
  contribution_direction: 'positive' | 'negative';
  /** Rank by absolute SHAP value */
  contribution_rank: number;
}

// =============================================================================
// REQUEST MODELS
// =============================================================================

/**
 * Request payload for real-time explanation
 */
export interface ExplainRequest {
  /** Patient identifier */
  patient_id: string;
  /** HCP context for the prediction */
  hcp_id?: string;
  /** Type of model to explain */
  model_type: ModelType;
  /** Specific model version (latest if not specified) */
  model_version_id?: string;
  /** Pre-computed features (fetched from Feast if not provided) */
  features?: Record<string, unknown>;
  /** Output format (default: top_k) */
  format?: ExplanationFormat;
  /** Number of top features to return (1-20, default 5) */
  top_k?: number;
  /** Include model's base prediction value (default true) */
  include_base_value?: boolean;
  /** Store explanation in ml_shap_analyses for compliance (default true) */
  store_for_audit?: boolean;
}

/**
 * Batch explanation request for multiple patients
 */
export interface BatchExplainRequest {
  /** Up to 50 patients per batch */
  requests: ExplainRequest[];
  /** Process in parallel (default true) */
  parallel?: boolean;
}

// =============================================================================
// RESPONSE MODELS
// =============================================================================

/**
 * Response payload with prediction + SHAP explanation
 */
export interface ExplainResponse {
  /** Unique ID for this explanation (for audit trail) */
  explanation_id: string;
  /** When request was received (ISO 8601) */
  request_timestamp: string;
  /** Patient identifier */
  patient_id: string;
  /** Model type used */
  model_type: ModelType;
  /** Model version used */
  model_version_id: string;
  /** Predicted class label */
  prediction_class: string;
  /** Prediction confidence [0-1] */
  prediction_probability: number;
  /** Model's expected value (average prediction) */
  base_value?: number;
  /** Top contributing features */
  top_features: FeatureContribution[];
  /** Sum of all SHAP values (should equal prediction - base_value) */
  shap_sum: number;
  /** Natural language explanation (if format=narrative) */
  narrative_explanation?: string;
  /** Time to compute explanation in ms */
  computation_time_ms: number;
  /** Whether explanation was stored for compliance */
  audit_stored: boolean;
}

/**
 * Batch explanation response
 */
export interface BatchExplainResponse {
  /** Batch identifier */
  batch_id: string;
  /** Total number of requests */
  total_requests: number;
  /** Number of successful explanations */
  successful: number;
  /** Number of failed explanations */
  failed: number;
  /** Successful explanations */
  explanations: ExplainResponse[];
  /** Errors for failed explanations */
  errors: Array<{
    patient_id: string;
    error: string;
  }>;
  /** Total processing time in ms */
  total_time_ms: number;
}

// =============================================================================
// HISTORY MODELS
// =============================================================================

/**
 * Request parameters for explanation history
 */
export interface ExplanationHistoryParams {
  /** Patient ID to get history for */
  patient_id: string;
  /** Filter by model type */
  model_type?: ModelType;
  /** Maximum number of results (default 10) */
  limit?: number;
}

/**
 * Response for explanation history
 */
export interface ExplanationHistoryResponse {
  /** Patient ID */
  patient_id: string;
  /** Total number of explanations */
  total_explanations: number;
  /** Historical explanations */
  explanations: ExplainResponse[];
  /** Status message */
  message?: string;
}

// =============================================================================
// MODELS INFO
// =============================================================================

/**
 * Information about an explainable model
 */
export interface ExplainableModelInfo {
  /** Model type */
  model_type: ModelType | string;
  /** Latest version */
  latest_version: string;
  /** Type of SHAP explainer used */
  explainer_type: 'TreeExplainer' | 'KernelExplainer';
  /** Average latency in milliseconds */
  avg_latency_ms: number;
}

/**
 * Response for listing explainable models
 */
export interface ListExplainableModelsResponse {
  /** Available models */
  supported_models: ExplainableModelInfo[];
  /** Total number of models */
  total_models: number;
}

// =============================================================================
// HEALTH CHECK
// =============================================================================

/**
 * Health check response for interpretability service
 */
export interface ExplainHealthResponse {
  /** Service status */
  status: 'healthy' | 'degraded' | 'unhealthy';
  /** Service name */
  service: string;
  /** Service version */
  version: string;
  /** Timestamp (ISO 8601) */
  timestamp: string;
  /** Dependency statuses */
  dependencies: {
    bentoml: 'connected' | 'disconnected';
    feast: 'connected' | 'disconnected';
    shap_explainer: 'loaded' | 'not_loaded';
    ml_shap_analyses_db: 'connected' | 'disconnected';
  };
}
