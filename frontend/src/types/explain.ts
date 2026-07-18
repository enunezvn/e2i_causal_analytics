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
 * Supported model types for SHAP explanation.
 *
 * The first four are the legacy demonstration taxonomy (no deployed model
 * today). The four gold-standard cohort families back the REAL deployed
 * per-brand `*_goldstd_lr_v1` models (#39) and are the ones the page surfaces.
 */
export enum ModelType {
  PROPENSITY = 'propensity',
  RISK_STRATIFICATION = 'risk_stratification',
  NEXT_BEST_ACTION = 'next_best_action',
  CHURN_PREDICTION = 'churn_prediction',
  // Gold-standard cohort families (#39) — real deployed per-brand models.
  INITIATION = 'initiation',
  PERSISTENCE = 'persistence',
  DISCONTINUATION = 'discontinuation',
  HCP_ADOPTION = 'hcp_adoption',
}

/**
 * Brands the gold-standard per-brand cohort models are registered for (#39).
 * Mirrors `GOLDSTD_BRANDS` in src/api/routes/explain.py.
 */
export const GOLDSTD_BRANDS = ['Remibrutinib', 'Fabhalta', 'Kisqali'] as const;
export type GoldStandardBrand = (typeof GOLDSTD_BRANDS)[number];

/** Cohort families served by the real per-brand gold-standard models. */
export const GOLD_STANDARD_COHORTS: readonly ModelType[] = [
  ModelType.INITIATION,
  ModelType.PERSISTENCE,
  ModelType.DISCONTINUATION,
  ModelType.HCP_ADOPTION,
];

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
  /**
   * Brand for the gold-standard per-brand cohort models (#39/#967):
   * Remibrutinib | Fabhalta | Kisqali. Selects which per-brand model to
   * explain (serving name `f"{cohort}_{brand}_goldstd_lr_v1"`). Defaults to
   * Remibrutinib server-side when omitted for a gold-standard cohort; ignored
   * by the legacy single-model cohorts. See src/api/routes/explain.py.
   */
  brand?: GoldStandardBrand | string;
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
  /**
   * Latest registered version, or null when the registry has no row for this
   * model type (the backend emits null for the legacy demo types and, today,
   * for hcp_adoption — see src/api/routes/explain.py).
   */
  latest_version: string | null;
  /** Type of SHAP explainer used */
  explainer_type: 'TreeExplainer' | 'KernelExplainer' | 'LinearExplainer';
  /**
   * Whether this is a gold-standard per-brand cohort model (#39/#967).
   * When true the FE offers a brand selector (Remibrutinib | Fabhalta |
   * Kisqali) so all per-brand serving bundles are reachable. Emitted by the
   * backend `/api/explain/models` handler (src/api/routes/explain.py).
   */
  is_gold_standard?: boolean;
  /** Human-readable description emitted by `/api/explain/models`. */
  description?: string;
  /**
   * Raw covariate names (`keep_columns`) for gold-standard cohort families,
   * null for legacy types and when BentoML is unavailable (best-effort). SHAP
   * runs over the ENCODED vector (one-hot `geographic_region_west` + missingness
   * `disease_severity__isna` columns); these parent names let the UI group those
   * encoded columns back to the covariate the user thinks in terms of, so a
   * single covariate no longer reads as many duplicate rows. Emitted by
   * `/api/explain/models` (src/api/routes/explain.py). Brand-invariant within a
   * family.
   */
  keep_columns?: string[] | null;
  /**
   * Average latency in milliseconds.
   * Optional: the backend `/api/explain/models` handler does not emit this
   * field (no per-model latency telemetry source) — see src/api/routes/explain.py.
   */
  avg_latency_ms?: number;
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
// COHORT-LEVEL (GLOBAL) FEATURE IMPORTANCE (#39 — option 2)
// =============================================================================

/** One feature's SHAP importance aggregated over a cohort sample. */
export interface GlobalImportanceFeature {
  feature_name: string;
  /** Mean |SHAP| across the sample (global importance ranking). */
  mean_abs_shap: number;
  /** Mean signed SHAP across the sample (net direction). */
  mean_shap: number;
  /** Mean feature value across the sample (numeric features only). */
  mean_feature_value: number | null;
  /** Rank by mean_abs_shap (1 = most important). */
  contribution_rank: number;
}

/** One entity's signed SHAP for one feature — a real beeswarm dot. */
export interface GlobalImportancePoint {
  feature_name: string;
  shap_value: number;
  feature_value: number | null;
}

/** Cohort-level (global) SHAP feature importance for one per-brand model. */
export interface GlobalFeatureImportanceResponse {
  model_type: ModelType | string;
  brand: string;
  /** Resolved serving name, e.g. initiation_kisqali_goldstd_lr_v1. */
  model_name: string;
  /** Mean model base value across the sample. */
  base_value: number | null;
  /** Entities successfully explained (honest n_succeeded). */
  sample_size: number;
  /** Minimum sample size requested (adaptive sizing may exceed it). */
  requested_sample_size: number;
  /** SHAP explainer used. */
  computation_method: string;
  /** When the aggregate was computed (ISO 8601). */
  computed_at: string;
  /** True when read from a stored precomputed row. */
  cached: boolean;
  /** 'random' (uniform draw) or 'prefix_fallback'; null/absent on legacy cached rows. */
  sampling_method?: string | null;
  /** True when the ranking (per stability_criterion) separated beyond sampling noise. */
  stability_achieved?: boolean | null;
  /**
   * Which ranking the stability verdict certifies: 'covariate_group' (the
   * displayed covariate ranking) or 'encoded_feature' (fallback when the
   * covariate schema is unavailable); null/absent on legacy cached rows.
   */
  stability_criterion?: string | null;
  /** stable | max_sample_size_reached | candidates_exhausted. */
  stopping_reason?: string | null;
  /** Features ranked desc by mean_abs_shap. */
  features: GlobalImportanceFeature[];
  /** Per-entity SHAP points for the top features (real beeswarm distribution). */
  points: GlobalImportancePoint[];
}

/** Query params for the global feature-importance endpoint. */
export interface GlobalFeatureImportanceParams {
  model_type: ModelType | string;
  brand?: GoldStandardBrand | string;
  /** Minimum entities to explain; omit to use the server default. */
  sample_size?: number;
  /** Hard cap for adaptive stability sizing; omit to use the server default. */
  max_sample_size?: number;
  max_points?: number;
  refresh?: boolean;
}

/** Real entity IDs for the per-entity SHAP picker. */
export interface SampleEntitiesResponse {
  model_type: ModelType | string;
  /** 'patient' or 'hcp'. */
  grain: 'patient' | 'hcp' | string;
  /** patient_id | hcp_id. */
  id_field: string;
  entities: string[];
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
