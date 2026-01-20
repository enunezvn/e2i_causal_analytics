/**
 * Segment Analysis Types
 * ======================
 *
 * TypeScript interfaces for the E2I Segment Analysis & Heterogeneous Optimization API.
 * Based on src/api/routes/segments.py backend schemas.
 *
 * @module types/segments
 */

// =============================================================================
// ENUMS
// =============================================================================

/**
 * Types of treatment responders
 */
export enum ResponderType {
  HIGH = 'high',
  LOW = 'low',
  AVERAGE = 'average',
}

// SegmentationMethod is exported from causal.ts to avoid duplicate export conflict

/**
 * Status of segment analysis
 */
export enum SegmentAnalysisStatus {
  PENDING = 'pending',
  ESTIMATING = 'estimating',
  ANALYZING = 'analyzing',
  OPTIMIZING = 'optimizing',
  COMPLETED = 'completed',
  FAILED = 'failed',
}

/**
 * Type of analysis question for library routing
 */
export enum QuestionType {
  EFFECT_HETEROGENEITY = 'effect_heterogeneity',
  TARGETING = 'targeting',
  SEGMENT_OPTIMIZATION = 'segment_optimization',
  COMPREHENSIVE = 'comprehensive',
}

// =============================================================================
// REQUEST MODELS
// =============================================================================

/**
 * Request to run segment analysis
 */
export interface RunSegmentAnalysisRequest {
  /** Natural language query describing the analysis */
  query: string;
  /** Treatment variable name (e.g., 'rep_visits', 'email_campaigns') */
  treatment_var: string;
  /** Outcome variable name (e.g., 'trx', 'conversion') */
  outcome_var: string;
  /** Variables to segment by (e.g., ['region', 'specialty']) */
  segment_vars: string[];
  /** Variables that modify treatment effect */
  effect_modifiers?: string[];
  /** Data source identifier */
  data_source?: string;
  /** Additional filters */
  filters?: Record<string, unknown>;
  /** Causal Forest trees (10-1000) */
  n_estimators?: number;
  /** Minimum samples per leaf (1-100) */
  min_samples_leaf?: number;
  /** Significance level for CI calculation (0-0.5) */
  significance_level?: number;
  /** Number of top segments to return (1-50) */
  top_segments_count?: number;
  /** Analysis question type for library routing */
  question_type?: QuestionType;
}

/**
 * Parameters for listing policies
 */
export interface ListPoliciesParams {
  /** Minimum expected lift threshold */
  min_lift?: number;
  /** Minimum confidence threshold */
  min_confidence?: number;
  /** Maximum number of results */
  limit?: number;
}

// =============================================================================
// RESPONSE MODELS
// =============================================================================

/**
 * CATE estimation result for a segment
 */
export interface CATEResult {
  /** Segment dimension name */
  segment_name: string;
  /** Segment value */
  segment_value: string;
  /** Conditional Average Treatment Effect */
  cate_estimate: number;
  /** 95% CI lower bound */
  cate_ci_lower: number;
  /** 95% CI upper bound */
  cate_ci_upper: number;
  /** Number of observations in segment */
  sample_size: number;
  /** Whether effect is statistically significant */
  statistical_significance: boolean;
}

/**
 * Profile of a high/low responder segment
 */
export interface SegmentProfile {
  /** Unique segment identifier */
  segment_id: string;
  /** Responder classification */
  responder_type: ResponderType;
  /** CATE for this segment */
  cate_estimate: number;
  /** Features that define this segment */
  defining_features: Array<Record<string, unknown>>;
  /** Segment size (observations) */
  size: number;
  /** Percentage of total population */
  size_percentage: number;
  /** Targeting recommendation */
  recommendation: string;
}

/**
 * Treatment allocation recommendation
 */
export interface PolicyRecommendation {
  /** Segment identifier */
  segment: string;
  /** Current treatment rate (0-1) */
  current_treatment_rate: number;
  /** Recommended treatment rate (0-1) */
  recommended_treatment_rate: number;
  /** Expected incremental outcome from change */
  expected_incremental_outcome: number;
  /** Recommendation confidence (0-1) */
  confidence: number;
}

/**
 * Uplift modeling metrics
 */
export interface UpliftMetrics {
  /** Area Under Uplift Curve (0-1) */
  overall_auuc: number;
  /** Qini coefficient */
  overall_qini: number;
  /** How well model targets responders (0-1) */
  targeting_efficiency: number;
  /** Model type (random_forest, gradient_boosting) */
  model_type_used: string;
}

/**
 * Response from segment analysis
 */
export interface SegmentAnalysisResponse {
  /** Unique analysis identifier */
  analysis_id: string;
  /** Analysis status */
  status: SegmentAnalysisStatus;
  /** Question type used for routing */
  question_type?: QuestionType;

  // CATE results
  /** CATE results grouped by segment variable */
  cate_by_segment: Record<string, CATEResult[]>;
  /** Overall Average Treatment Effect */
  overall_ate?: number;
  /** Treatment effect heterogeneity (0-1) */
  heterogeneity_score?: number;
  /** Feature importance for CATE */
  feature_importance?: Record<string, number>;

  // Uplift results
  /** Uplift modeling metrics */
  uplift_metrics?: UpliftMetrics;

  // Segment discovery
  /** High responder segments */
  high_responders: SegmentProfile[];
  /** Low responder segments */
  low_responders: SegmentProfile[];

  // Policy recommendations
  /** Targeting recommendations */
  policy_recommendations: PolicyRecommendation[];
  /** Expected lift from optimal allocation */
  expected_total_lift?: number;
  /** Summary of optimal allocation */
  optimal_allocation_summary?: string;

  // Summary
  /** Executive-level summary */
  executive_summary?: string;
  /** Key findings */
  key_insights: string[];

  // Multi-library support
  /** Causal libraries used */
  libraries_used?: string[];
  /** Agreement between libraries (0-1) */
  library_agreement_score?: number;
  /** Whether cross-validation passed */
  validation_passed?: boolean;

  // Metadata
  /** CATE estimation time (ms) */
  estimation_latency_ms: number;
  /** Segment analysis time (ms) */
  analysis_latency_ms: number;
  /** Total workflow time (ms) */
  total_latency_ms: number;
  /** Analysis timestamp */
  timestamp: string;
  /** Analysis warnings */
  warnings: string[];
  /** Overall analysis confidence (0-1) */
  confidence: number;
}

/**
 * Response for listing policy recommendations
 */
export interface PolicyListResponse {
  /** Total recommendations */
  total_count: number;
  /** Policy recommendations */
  recommendations: PolicyRecommendation[];
  /** Total expected lift if all policies adopted */
  expected_total_lift: number;
}

/**
 * Health check response for segment analysis service
 */
export interface SegmentHealthResponse {
  /** Service status */
  status: string;
  /** Heterogeneous Optimizer agent status */
  agent_available: boolean;
  /** EconML availability */
  econml_available: boolean;
  /** CausalML availability */
  causalml_available: boolean;
  /** Last analysis timestamp */
  last_analysis?: string;
  /** Analyses in last 24 hours */
  analyses_24h: number;
}
