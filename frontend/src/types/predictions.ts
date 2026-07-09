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
  /**
   * Populate `feature_importance` with REAL per-prediction SHAP contributions
   * for this exact input. The backend delegates to the BentoML `/shap` endpoint
   * (LinearExplainer over the routed model). Off by default; the Predictive
   * Analytics page sets it true so the result card shows Feature Contributions.
   */
  return_feature_importance?: boolean;
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
  /**
   * Telemetry tag describing where the prediction's features came from:
   * 'feast_online' when the route fetched features from the Feast online store,
   * or 'user_provided' when the caller supplied them directly. Undefined for
   * paths that predate this contract.
   */
  feature_source?: string;
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
  /**
   * Authoritative ENCODED feature columns the model scores — one-hot
   * expansions + `__isna` flags (e.g. `geographic_region_south`,
   * `academic_hcp__isna`). Returned verbatim by the live BentoML `/model_info`.
   * Used to detect which `keep_columns` are categorical and their options.
   */
  feature_columns?: string[];
  /**
   * RAW human-meaningful covariates the served FeatureBuilder encodes (e.g.
   * `disease_severity`, `academic_hcp`, `geographic_region`). The Predictive
   * Analytics form collects THESE; the backend forwards them as `raw_features`.
   */
  keep_columns?: string[];
  /** Output schema */
  output_schema?: Record<string, unknown>;
  /** Training date */
  trained_at?: string;
  /** Performance metrics */
  metrics?: Record<string, number>;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

// =============================================================================
// COHORT SCORING (data-driven population view)
// =============================================================================

/** One scored entity from a model's holdout cohort. */
export interface CohortScoredRow {
  /** Real entity id (patient_id / hcp_id) from the holdout split */
  entity_id: string;
  /** Predicted positive-class probability */
  probability: number;
  /** The RAW covariates scored for this entity (feed the drill-down what-if) */
  covariates: Record<string, unknown>;
}

/** One cohort-level SHAP driver: mean |SHAP| over the sampled top targets. */
export interface CohortDriver {
  /** Encoded feature name (matches the drill-down SHAP names) */
  feature: string;
  /** Mean |SHAP| across the sampled top-ranked rows (log-odds scale) */
  importance: number;
  /** 'increases' | 'decreases' | 'mixed' (per-row contributions cancel out) */
  direction: string;
}

/** Probability distribution over ALL scored rows (fixed [0,1] bins). */
export interface CohortScoreDistribution {
  n: number;
  mean: number;
  /** Histogram bin edges (length = bin_counts.length + 1) */
  bin_edges: number[];
  /** Row count per probability bin */
  bin_counts: number[];
}

/**
 * Async cohort-scoring job (submit -> poll): the data-driven population view.
 * Replaces hand-typed input features — score the model's OWN out-of-sample
 * holdout cohort and rank targets.
 */
export interface CohortScoreResponse {
  job_id: string;
  /** pending | running | completed | failed */
  status: string;
  model_name: string;
  cohort?: string | null;
  brand?: string | null;
  /** Data split scored (out-of-sample) */
  split: string;
  out_of_sample: boolean;
  /** Provenance, e.g. 'holdout_synthetic' */
  feature_source: string;
  n_scored: number;
  top_n: number;
  /** Highest-probability entities, ranked desc, capped at top_n */
  top_rows: CohortScoredRow[];
  distribution?: CohortScoreDistribution | null;
  /** Cohort-level SHAP drivers over the top-ranked rows (best-effort; may be empty) */
  top_drivers?: CohortDriver[];
  /** How many top-ranked rows the driver aggregation actually sampled */
  drivers_from_top_n?: number;
  error?: string | null;
  latency_ms: number;
}
