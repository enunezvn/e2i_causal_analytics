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
  /**
   * Optional cohort FILTER (data-driven dropdown, like /causal/brands). Scopes
   * the gold-standard load to one brand server-side; it is a row subset, NOT a
   * causal variable. `undefined` => all brands.
   */
  brand?: string;
  /**
   * Treatment variable name (curated). Optional — the backend defaults to
   * `treatment_arm` and enforces the patient_journeys allowlist server-side.
   */
  treatment_var?: string;
  /**
   * Outcome variable name (curated). Optional — the backend defaults to
   * `persistent_180d` and enforces the patient_journeys allowlist server-side.
   */
  outcome_var?: string;
  /**
   * Variables to segment by. Optional — for the patient_journeys path the
   * backend FIXES the clinical segment set server-side; any value here is
   * overridden.
   */
  segment_vars?: string[];
  /** Variables that modify treatment effect (fixed server-side for the clinical path). */
  effect_modifiers?: string[];
  /**
   * Confounders to adjust for. Routed into the DML nuisance model (W) and
   * residualized out, so the reported per-segment CATE is de-confounded
   * (reflects the true treatment effect, not selection bias). Distinct from
   * segment_vars (reporting grouping) and effect_modifiers (heterogeneity).
   */
  confounders?: string[];
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
  /** CI lower bound at the response's confidence_level (default 0.95) */
  cate_ci_lower: number;
  /** CI upper bound at the response's confidence_level (default 0.95) */
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
/**
 * On-/off-label verdict for a recommendation, from the label-segmentation gater.
 * Populated only when the request enabled `label_segmentation`; absent otherwise.
 */
export type LabelVerdict = 'on_label' | 'off_label' | 'mixed' | 'indeterminate';

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
  /**
   * Whether this segment is off-label (population/use outside the FDA label).
   * Off-label items are de-prioritized (sunk to the bottom of the ranking) by
   * the backend. Populated only when label_segmentation was enabled.
   */
  off_label?: boolean;
  /** Human-readable reason the segment was judged off-label. */
  off_label_reason?: string;
  /** Structured verdict: on_label | off_label | mixed | indeterminate. */
  label_verdict?: LabelVerdict;
  /** True when the verdict was confirmed against the FDA drug label. */
  label_evidence_confirmed?: boolean;
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
  /**
   * Confidence level the CATE CIs (cate_by_segment[*].cate_ci_lower/upper)
   * are computed at, e.g. 0.95 => a 95% CI. Derived from the request's
   * significance_level (confidence_level = 1 - significance_level). Optional
   * for backward-compatibility with older responses; backend always sets it.
   */
  confidence_level?: number;
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
  /**
   * Mid (average) responder segments — |CATE| in the band between the low and
   * high thresholds (responder_type 'average'). Defaults to `[]` when none
   * qualify, so callers never need to null-check it.
   */
  mid_responders: SegmentProfile[];
  /** Low responder segments */
  low_responders: SegmentProfile[];

  // Policy recommendations
  /** Targeting recommendations */
  policy_recommendations: PolicyRecommendation[];
  /** Expected lift as a COUNT of incremental outcomes on the best single axis (secondary). */
  expected_total_lift?: number;
  /** HEADLINE: best-axis lift as a percentage-point change in the outcome rate (fraction 0..1). */
  expected_lift_pp?: number;
  /** Summary of optimal allocation */
  optimal_allocation_summary?: string;

  // Summary
  /** Executive-level summary */
  executive_summary?: string;
  /**
   * 3-tier business narrative (who responds, why, expected lift) from the
   * profile_generator node. Multi-paragraph; render with whitespace-pre-line.
   * Was silently dropped at the route before the clinical-HTE rebuild.
   */
  strategic_interpretation?: string;
  /** Key findings */
  key_insights: string[];

  // Hierarchical / heterogeneity (mapped from the final graph state)
  /** High/mid/low comparison summary (effect_ratio, counts) from segment_analyzer. */
  segment_comparison?: Record<string, unknown>;
  /** Between-segment heterogeneity (I^2, 0-100) from the hierarchical analyzer. */
  segment_heterogeneity?: number;
  /** Number of segments analyzed by the hierarchical analyzer. */
  n_segments_analyzed?: number;
  /** Segmentation method used (quantile/kmeans/threshold/tree). */
  segmentation_method_used?: string;
  /** Aggregate ATE from the hierarchical (nested-CI) analysis. */
  overall_hierarchical_ate?: number;
  /** Per-segment hierarchical CATE results. */
  hierarchical_segment_results?: Array<Record<string, unknown>>;
  /** Uplift scores grouped by segment dimension. */
  uplift_by_segment?: Record<string, unknown>;

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
  /**
   * Analyses-store backing: 'durable' (Redis, shared across workers) or
   * 'degraded' (process-local in-memory fallback — Redis unreachable from
   * this worker, so cross-worker reads may 404). Optional for older responses.
   */
  storage_mode?: string;
}

/**
 * Curated config options for the agent-driven Segment Analysis page.
 * From GET /segments/datasets — drives the data-driven config dropdowns.
 */
export interface SegmentDatasetsResponse {
  /** Curated selectable treatment columns (default: first entry). */
  treatments: string[];
  /** Curated selectable outcome columns (default: first entry). */
  outcomes: string[];
  /** Distinct brands present in the gold-standard cohort (filter dropdown). */
  brands: string[];
  /**
   * Human-readable display labels keyed by column name (treatment + outcome), e.g.
   * `low_gap_180d` -> "Low refill gap (≤30d)". The dropdowns must render these
   * instead of a raw title-cased column name; absent keys fall back to titleCase.
   */
  labels?: Record<string, string>;
  /**
   * Outcomes with a modeled causal edge from each offered treatment (causal_paths
   * SSOT, scoped to `brand`). The Outcome dropdown must offer ONLY these for the
   * selected treatment. Empty/absent when `options_source` is the curated
   * fallback — then the flat `outcomes` list applies.
   */
  outcomes_by_treatment?: Record<string, string[]>;
  /** Brand the options are scoped to (null/absent = all brands). */
  brand?: string | null;
  /**
   * 'causal_paths' when derived from the causal-path registry; 'curated_fallback'
   * when the registry was unavailable and the flat allowlists were returned.
   */
  options_source?: 'causal_paths' | 'curated_fallback';
}
