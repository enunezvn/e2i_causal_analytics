/**
 * Causal Inference API Types
 * ==========================
 *
 * TypeScript interfaces for the E2I Causal Inference API.
 * Based on src/api/schemas/causal.py backend schemas.
 *
 * Supports:
 * - Hierarchical CATE analysis (EconML within CausalML segments)
 * - Library routing (DoWhy, EconML, CausalML, NetworkX)
 * - Sequential and parallel multi-library pipelines
 * - Cross-library validation
 *
 * @module types/causal
 */

// =============================================================================
// ENUMS
// =============================================================================

/**
 * Supported causal inference libraries
 */
export enum CausalLibrary {
  DOWHY = 'dowhy',
  ECONML = 'econml',
  CAUSALML = 'causalml',
  NETWORKX = 'networkx',
  NONE = 'none',
}

/**
 * Types of causal questions for routing
 */
export enum CausalQuestionType {
  /** "Does X cause Y?" → DoWhy */
  CAUSAL_EFFECT = 'causal_effect',
  /** "How does effect vary?" → EconML */
  EFFECT_HETEROGENEITY = 'effect_heterogeneity',
  /** "Who should we target?" → CausalML */
  TARGETING = 'targeting',
  /** "How does impact flow?" → NetworkX */
  SYSTEM_DEPENDENCIES = 'system_dependencies',
  /** All libraries */
  COMPREHENSIVE = 'comprehensive',
}

/**
 * Available causal estimators
 */
export enum EstimatorType {
  // EconML
  CAUSAL_FOREST = 'causal_forest',
  LINEAR_DML = 'linear_dml',
  ORTHO_FOREST = 'ortho_forest',
  DR_LEARNER = 'dr_learner',
  X_LEARNER = 'x_learner',
  T_LEARNER = 't_learner',
  S_LEARNER = 's_learner',
  OLS = 'ols',
  // CausalML
  UPLIFT_RANDOM_FOREST = 'uplift_random_forest',
  UPLIFT_GRADIENT_BOOSTING = 'uplift_gradient_boosting',
  // DoWhy
  PROPENSITY_SCORE_MATCHING = 'propensity_score_matching',
  INVERSE_PROPENSITY_WEIGHTING = 'inverse_propensity_weighting',
  REGRESSION_DISCONTINUITY = 'regression_discontinuity',
  INSTRUMENTAL_VARIABLE = 'instrumental_variable',
}

/**
 * Segmentation methods for hierarchical analysis
 */
export enum SegmentationMethod {
  QUANTILE = 'quantile',
  KMEANS = 'kmeans',
  THRESHOLD = 'threshold',
  TREE = 'tree',
}

/**
 * Aggregation methods for nested confidence intervals
 */
export enum AggregationMethod {
  VARIANCE_WEIGHTED = 'variance_weighted',
  SAMPLE_WEIGHTED = 'sample_weighted',
  EQUAL = 'equal',
  BOOTSTRAP = 'bootstrap',
}

/**
 * Pipeline execution mode
 */
export enum PipelineMode {
  SEQUENTIAL = 'sequential',
  PARALLEL = 'parallel',
}

/**
 * Analysis status
 */
export enum CausalAnalysisStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
}

// =============================================================================
// HIERARCHICAL ANALYSIS TYPES
// =============================================================================

/**
 * Request for hierarchical CATE analysis
 */
export interface HierarchicalAnalysisRequest {
  /** Treatment variable name */
  treatment_var: string;
  /** Outcome variable name */
  outcome_var: string;
  /** Variables that modify treatment effect */
  effect_modifiers?: string[];
  /** Data source identifier */
  data_source?: string;
  /** Data filters */
  filters?: Record<string, unknown>;
  /** Number of uplift segments (2-10) */
  n_segments?: number;
  /** Method for creating segments */
  segmentation_method?: SegmentationMethod;
  /** EconML estimator for segment-level CATE */
  estimator_type?: EstimatorType;
  /** Minimum samples per segment */
  min_segment_size?: number;
  /** Confidence level for CIs (0.80-0.99) */
  confidence_level?: number;
  /** Method for aggregating segment CATEs */
  aggregation_method?: AggregationMethod;
  /** Maximum execution time in seconds */
  timeout_seconds?: number;
}

/**
 * CATE result for a single segment
 */
export interface SegmentCATEResult {
  /** Segment identifier */
  segment_id: number;
  /** Segment name (e.g., 'high_uplift') */
  segment_name: string;
  /** Number of samples in segment */
  n_samples: number;
  /** Uplift score range [min, max] */
  uplift_range: [number, number];
  /** Mean CATE estimate */
  cate_mean?: number;
  /** CATE standard deviation */
  cate_std?: number;
  /** CATE CI lower bound */
  cate_ci_lower?: number;
  /** CATE CI upper bound */
  cate_ci_upper?: number;
  /** Whether estimation succeeded */
  success: boolean;
  /** Error if failed */
  error_message?: string;
}

/**
 * Nested confidence interval aggregation result
 */
export interface NestedCIResult {
  /** Aggregate ATE from segments */
  aggregate_ate: number;
  /** Aggregate CI lower bound */
  aggregate_ci_lower: number;
  /** Aggregate CI upper bound */
  aggregate_ci_upper: number;
  /** Aggregate standard error */
  aggregate_std: number;
  /** Confidence level used */
  confidence_level: number;
  /** Aggregation method used */
  aggregation_method: string;
  /** Weight contribution from each segment */
  segment_contributions: Record<string, number>;
  /** I² heterogeneity statistic (0-100) */
  i_squared?: number;
  /** τ² between-segment variance */
  tau_squared?: number;
  /** Segments included in aggregate */
  n_segments_included: number;
  /** Total samples across segments */
  total_sample_size: number;
}

/**
 * Response from hierarchical CATE analysis
 */
export interface HierarchicalAnalysisResponse {
  /** Unique analysis identifier */
  analysis_id: string;
  /** Analysis status */
  status: CausalAnalysisStatus;
  /** Per-segment CATE results */
  segment_results: SegmentCATEResult[];
  /** Nested CI aggregation */
  nested_ci?: NestedCIResult;
  /** Overall ATE estimate */
  overall_ate?: number;
  /** Overall CI lower */
  overall_ci_lower?: number;
  /** Overall CI upper */
  overall_ci_upper?: number;
  /** Heterogeneity score (I²) */
  segment_heterogeneity?: number;
  /** Number of segments analyzed */
  n_segments_analyzed: number;
  /** Segmentation method used */
  segmentation_method: string;
  /** EconML estimator used */
  estimator_type: string;
  /** Execution time in milliseconds */
  latency_ms: number;
  /** Analysis timestamp */
  created_at: string;
  /** Warnings */
  warnings: string[];
  /** Errors */
  errors: string[];
}

// =============================================================================
// LIBRARY ROUTING TYPES
// =============================================================================

/**
 * Request to route a causal query to appropriate library
 */
export interface RouteQueryRequest {
  /** Natural language causal question */
  query: string;
  /** Treatment variable if known */
  treatment_var?: string;
  /** Outcome variable if known */
  outcome_var?: string;
  /** Additional context for routing */
  context?: Record<string, unknown>;
  /** Preferred library (optional override) */
  prefer_library?: CausalLibrary;
}

/**
 * Response from query routing
 */
export interface RouteQueryResponse {
  /** Original query */
  query: string;
  /** Classified question type */
  question_type: CausalQuestionType;
  /** Recommended primary library */
  primary_library: CausalLibrary;
  /** Recommended secondary libraries */
  secondary_libraries: CausalLibrary[];
  /** Recommended estimators */
  recommended_estimators: string[];
  /** Confidence in routing decision (0-1) */
  routing_confidence: number;
  /** Explanation for routing decision */
  routing_rationale: string;
  /** Suggested pipeline mode */
  suggested_pipeline?: PipelineMode;
}

// =============================================================================
// PIPELINE TYPES
// =============================================================================

/**
 * Configuration for a pipeline stage
 */
export interface PipelineStageConfig {
  /** Library for this stage */
  library: CausalLibrary;
  /** Specific estimator */
  estimator?: string;
  /** Stage parameters */
  parameters?: Record<string, unknown>;
  /** Stage timeout in seconds (10-300) */
  timeout_seconds?: number;
}

/**
 * Request for sequential multi-library pipeline
 */
export interface SequentialPipelineRequest {
  /** Treatment variable */
  treatment_var: string;
  /** Outcome variable */
  outcome_var: string;
  /** Covariate variables */
  covariates?: string[];
  /** Data source */
  data_source?: string;
  /** Data filters */
  filters?: Record<string, unknown>;
  /** Pipeline stages in order (2-4 stages) */
  stages: PipelineStageConfig[];
  /** Propagate results between stages */
  propagate_state?: boolean;
  /** Stop pipeline on stage failure */
  stop_on_failure?: boolean;
  /** Minimum agreement threshold for validation (0.5-1.0) */
  validation_threshold?: number;
}

/**
 * Result from a single pipeline stage
 */
export interface PipelineStageResult {
  /** Stage position (1-indexed) */
  stage_number: number;
  /** Library used */
  library: string;
  /** Estimator used */
  estimator?: string;
  /** Stage status */
  status: CausalAnalysisStatus;
  /** Effect estimate */
  effect_estimate?: number;
  /** CI lower bound */
  ci_lower?: number;
  /** CI upper bound */
  ci_upper?: number;
  /** P-value */
  p_value?: number;
  /** Library-specific results */
  additional_results: Record<string, unknown>;
  /** Stage execution time in milliseconds */
  latency_ms: number;
  /** Error message if failed */
  error?: string;
}

/**
 * Response from sequential pipeline execution
 */
export interface SequentialPipelineResponse {
  /** Unique pipeline identifier */
  pipeline_id: string;
  /** Overall pipeline status */
  status: CausalAnalysisStatus;
  /** Number of stages completed */
  stages_completed: number;
  /** Total number of stages */
  stages_total: number;
  /** Results from each stage */
  stage_results: PipelineStageResult[];
  /** Confidence-weighted consensus effect */
  consensus_effect?: number;
  /** Consensus CI lower */
  consensus_ci_lower?: number;
  /** Consensus CI upper */
  consensus_ci_upper?: number;
  /** Agreement between libraries (0-1) */
  library_agreement_score?: number;
  /** Variance across library estimates */
  effect_estimate_variance?: number;
  /** Total pipeline execution time in milliseconds */
  total_latency_ms: number;
  /** Pipeline start timestamp */
  created_at: string;
  /** Warnings */
  warnings: string[];
}

/**
 * Request for parallel multi-library analysis
 */
export interface ParallelPipelineRequest {
  /** Treatment variable */
  treatment_var: string;
  /** Outcome variable */
  outcome_var: string;
  /** Covariate variables */
  covariates?: string[];
  /** Data source */
  data_source?: string;
  /** Data filters */
  filters?: Record<string, unknown>;
  /** Libraries to run in parallel (2-4) */
  libraries: CausalLibrary[];
  /** Specific estimator per library */
  estimators?: Record<string, string>;
  /** Method for consensus computation */
  consensus_method?: string;
  /** Overall timeout in seconds (30-300) */
  timeout_seconds?: number;
}

/**
 * Response from parallel pipeline execution
 */
export interface ParallelPipelineResponse {
  /** Unique pipeline identifier */
  pipeline_id: string;
  /** Overall status */
  status: CausalAnalysisStatus;
  /** Libraries that succeeded */
  libraries_succeeded: string[];
  /** Libraries that failed */
  libraries_failed: string[];
  /** Results per library */
  library_results: Record<string, Record<string, unknown>>;
  /** Consensus effect */
  consensus_effect?: number;
  /** Consensus CI lower */
  consensus_ci_lower?: number;
  /** Consensus CI upper */
  consensus_ci_upper?: number;
  /** Agreement score (0-1) */
  library_agreement_score?: number;
  /** Consensus method used */
  consensus_method: string;
  /** Total execution time in milliseconds */
  total_latency_ms: number;
  /** Analysis timestamp */
  created_at: string;
  /** Warnings */
  warnings: string[];
}

// =============================================================================
// CROSS-VALIDATION TYPES
// =============================================================================

/**
 * Request for cross-library validation
 */
export interface CrossValidationRequest {
  /** Treatment variable */
  treatment_var: string;
  /** Outcome variable */
  outcome_var: string;
  /** Covariate variables */
  covariates?: string[];
  /** Data source */
  data_source?: string;
  /** Primary library for validation */
  primary_library: CausalLibrary;
  /** Library to validate against */
  validation_library: CausalLibrary;
  /** Minimum agreement threshold (0.5-1.0) */
  agreement_threshold?: number;
  /** Bootstrap iterations for CI comparison (10-1000) */
  bootstrap_iterations?: number;
}

/**
 * Response from cross-library validation
 */
export interface CrossValidationResponse {
  /** Unique validation identifier */
  validation_id: string;
  /** Primary library */
  primary_library: string;
  /** Validation library */
  validation_library: string;
  /** Effect from primary library */
  primary_effect: number;
  /** Primary CI [lower, upper] */
  primary_ci: [number, number];
  /** Effect from validation library */
  validation_effect: number;
  /** Validation CI [lower, upper] */
  validation_ci: [number, number];
  /** Absolute difference in effects */
  effect_difference: number;
  /** Relative difference percentage */
  relative_difference: number;
  /** CI overlap ratio (0-1) */
  ci_overlap_ratio: number;
  /** Overall agreement score (0-1) */
  agreement_score: number;
  /** Whether validation threshold met */
  validation_passed: boolean;
  /** Threshold used */
  agreement_threshold: number;
  /** Validation execution time in milliseconds */
  latency_ms: number;
  /** Validation timestamp */
  created_at: string;
  /** Recommendations based on results */
  recommendations: string[];
}

// =============================================================================
// ESTIMATOR INFO TYPES
// =============================================================================

/**
 * Information about a causal estimator
 */
export interface EstimatorInfo {
  /** Estimator name */
  name: string;
  /** Source library */
  library: CausalLibrary;
  /** Type (CATE, uplift, identification, etc.) */
  estimator_type: string;
  /** Brief description */
  description: string;
  /** Best use cases */
  best_for: string[];
  /** Key parameters */
  parameters: string[];
  /** Whether CI is supported */
  supports_confidence_intervals: boolean;
  /** Whether HTE is supported */
  supports_heterogeneous_effects: boolean;
}

/**
 * Response listing available estimators
 */
export interface EstimatorListResponse {
  /** Available estimators */
  estimators: EstimatorInfo[];
  /** Total estimators */
  total: number;
  /** Estimators grouped by library */
  by_library: Record<string, string[]>;
}

// =============================================================================
// HEALTH CHECK TYPES
// =============================================================================

/**
 * Health check response for causal engine
 */
export interface CausalHealthResponse {
  /** Overall health status */
  status: string;
  /** Availability of each library */
  libraries_available: Record<string, boolean>;
  /** Number of estimators loaded */
  estimators_loaded: number;
  /** Whether pipeline orchestrator is ready */
  pipeline_orchestrator_ready: boolean;
  /** Whether hierarchical analyzer is ready */
  hierarchical_analyzer_ready: boolean;
  /** Timestamp of last analysis */
  last_analysis?: string;
  /** Analyses run in last 24 hours */
  analysis_count_24h: number;
  /** Average analysis latency in milliseconds */
  average_latency_ms?: number;
  /** Error message if unhealthy */
  error?: string;
}
