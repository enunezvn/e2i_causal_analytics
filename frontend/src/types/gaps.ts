/**
 * Gap Analysis Types
 * ==================
 *
 * TypeScript interfaces for the E2I Gap Analysis API.
 * Based on src/api/routes/gaps.py backend schemas.
 *
 * @module types/gaps
 */

import type { LabelVerdict } from './segments';

// =============================================================================
// ENUMS
// =============================================================================

/**
 * Types of performance gaps
 */
export enum GapType {
  VS_TARGET = 'vs_target',
  VS_BENCHMARK = 'vs_benchmark',
  VS_POTENTIAL = 'vs_potential',
  TEMPORAL = 'temporal',
  ALL = 'all',
}

/**
 * Difficulty levels for closing a gap
 */
export enum ImplementationDifficulty {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
}

/**
 * Status of a gap analysis
 */
export enum AnalysisStatus {
  PENDING = 'pending',
  DETECTING = 'detecting',
  CALCULATING = 'calculating',
  PRIORITIZING = 'prioritizing',
  COMPLETED = 'completed',
  FAILED = 'failed',
}

// =============================================================================
// REQUEST MODELS
// =============================================================================

/**
 * Request to run gap analysis
 */
export interface RunGapAnalysisRequest {
  /** Natural language query describing the analysis */
  query: string;
  /** Brand identifier (e.g., 'kisqali', 'fabhalta') */
  brand: string;
  /** KPIs to analyze */
  metrics?: string[];
  /** Segmentation dimensions */
  segments?: string[];
  /**
   * Analysis period. Accepted forms (anything else is a 422): 'current_quarter'
   * (quarter start to today), 'previous_quarter' / 'last_quarter' (the preceding
   * full calendar quarter), 'Q3_2026' or '2026-Q3' (an explicit calendar quarter),
   * 'YTD', 'MTD', or an explicit inclusive range 'YYYY-MM-DD_YYYY-MM-DD'. Relative
   * forms resolve on the SERVER's UTC calendar date, not the browser's; the window
   * actually compared comes back as `resolved_period` (#1834).
   */
  time_period?: string;
  /** Type of gaps to detect */
  gap_type?: GapType;
  /** Minimum gap percentage to report */
  min_gap_threshold?: number;
  /** Maximum opportunities to return */
  max_opportunities?: number;
  /** Additional filters */
  filters?: Record<string, unknown>;
}

/**
 * Parameters for listing opportunities
 */
export interface ListOpportunitiesParams {
  /** Filter by brand */
  brand?: string;
  /** Minimum ROI threshold */
  min_roi?: number;
  /** Filter by implementation difficulty */
  difficulty?: ImplementationDifficulty;
  /** Maximum number of results */
  limit?: number;
}

// =============================================================================
// RESPONSE MODELS
// =============================================================================

/**
 * Individual performance gap identified
 */
export interface PerformanceGap {
  /** Unique gap identifier */
  gap_id: string;
  /** KPI name */
  metric: string;
  /** Segmentation dimension */
  segment: string;
  /** Specific segment value */
  segment_value: string;
  /** Current performance value */
  current_value: number;
  /** Target/benchmark value */
  target_value: number;
  /** Absolute gap (target - current) */
  gap_size: number;
  /** Gap as percentage */
  gap_percentage: number;
  /** Type of comparison */
  gap_type: string;
}

/**
 * Bootstrap confidence interval for ROI estimates
 */
export interface ConfidenceInterval {
  /** 2.5th percentile */
  lower_bound: number;
  /** 50th percentile */
  median: number;
  /** 97.5th percentile */
  upper_bound: number;
  /** P(ROI > 1x) */
  probability_positive: number;
  /** P(ROI > target) */
  probability_target: number;
}

/**
 * ROI estimate for closing a performance gap
 */
export interface ROIEstimate {
  /** References gap identifier */
  gap_id: string;
  /** Annual revenue impact (USD) */
  estimated_revenue_impact: number;
  /** One-time cost (USD) */
  estimated_cost_to_close: number;
  /** Base ROI ratio */
  expected_roi: number;
  /** ROI after risk adjustment */
  risk_adjusted_roi: number;
  /** Months to recoup investment */
  payback_period_months: number;
  /** 95% confidence interval */
  confidence_interval?: ConfidenceInterval;
  /** Attribution level */
  attribution_level: string;
  /** Attribution rate (0-1) */
  attribution_rate: number;
  /** Estimate confidence (0-1) */
  confidence: number;
  /**
   * Whether this opportunity targets an off-label use (outside the FDA label).
   * Off-label bets are de-prioritized (sunk to the bottom of the ranking) by the
   * backend. Populated only when label_segmentation was enabled.
   */
  off_label?: boolean;
  /** Human-readable reason the opportunity was judged off-label. */
  off_label_reason?: string;
  /** Structured verdict: on_label | off_label | mixed | indeterminate. */
  label_verdict?: LabelVerdict;
  /** True when the verdict was confirmed against the FDA drug label. */
  label_evidence_confirmed?: boolean;
  /**
   * Market landscape (#1056) — surface-only curated competitor density for the
   * brand's indication, computed per bet by the backend ROI node. INFORMATIONAL:
   * it does NOT affect the ROI value or the ranking.
   */
  competitor_products_count?: number;
  /** Market saturation label: limited | moderate | crowded | unknown. */
  competitor_density_label?: string;
  /** Names of the competing products (curated, not FDA-sourced). */
  competitor_drug_names?: string[];
  /**
   * ROI rationale (T6) — the backend ROI node computes these for transparency;
   * surfaced for the opportunity drill-down. Previously dropped (the #1056
   * pattern). Present only on runs after the T6 deploy.
   */
  /** Combined risk-adjustment factor applied to the ROI (0-1). */
  total_risk_adjustment?: number;
  /** Revenue impact broken down by value driver (USD). */
  value_by_driver?: Record<string, number>;
  /** Economic assumptions behind the ROI estimate. */
  assumptions?: string[];
}

/** The 3-bucket opportunity category (T6). No residual "other". */
export type OpportunityCategory = 'quick_win' | 'steady_play' | 'strategic_bet';

/**
 * Prioritized gap with ROI estimate and action recommendation
 */
export interface PrioritizedOpportunity {
  /** Priority rank (1 = highest) */
  rank: number;
  /** The identified gap */
  gap: PerformanceGap;
  /** ROI analysis */
  roi_estimate: ROIEstimate;
  /** Specific action to close gap */
  recommended_action: string;
  /** Difficulty level */
  implementation_difficulty: ImplementationDifficulty;
  /** Expected time to results */
  time_to_impact: string;
  /** 3-bucket list-view category (set by the list endpoint). */
  category?: OpportunityCategory;
  /** Human-readable explanation of the difficulty rating (T6 drill-down). */
  difficulty_rationale?: string;
}

/**
 * Response from gap analysis
 */
/**
 * The concrete inclusive windows a gap analysis compared (#1834).
 * `time_period` is the label that was requested (e.g. 'current_quarter');
 * the four ISO dates are what it resolved to on the server's UTC calendar date
 * when the analysis ran (relative forms roll over at 00:00 UTC).
 */
export interface ResolvedPeriod {
  /** The requested time_period label */
  time_period: string;
  /** Current window start (YYYY-MM-DD, inclusive) */
  period_start: string;
  /** Current window end (YYYY-MM-DD, inclusive) */
  period_end: string;
  /** Comparison window start (YYYY-MM-DD, inclusive) */
  prior_start: string;
  /** Comparison window end (YYYY-MM-DD, inclusive) */
  prior_end: string;
}

export interface GapAnalysisResponse {
  /** Unique analysis identifier */
  analysis_id: string;
  /** Analysis status */
  status: AnalysisStatus;
  /** Brand analyzed */
  brand: string;
  /** KPIs analyzed */
  metrics_analyzed: string[];
  /** Number of segments */
  segments_analyzed: number;
  /**
   * The current/prior windows the requested time_period resolved to (#1834).
   * Absent/null while pending, when the run failed before resolution, or on
   * analyses persisted before the field existed.
   */
  resolved_period?: ResolvedPeriod | null;

  /** All opportunities ranked by ROI */
  prioritized_opportunities: PrioritizedOpportunity[];
  /** Low difficulty, high ROI (top 5) */
  quick_wins: PrioritizedOpportunity[];
  /** High impact, high difficulty (top 5) */
  strategic_bets: PrioritizedOpportunity[];

  /** Total potential revenue impact */
  total_addressable_value: number;
  /** Sum of all gap sizes */
  total_gap_value: number;

  /** Executive-level summary */
  executive_summary: string;
  /** Key findings */
  key_insights: string[];

  /** Causal libraries used */
  libraries_used?: string[];
  /** Agreement between libraries */
  library_agreement_score?: number;

  /** Detection time (ms) */
  detection_latency_ms: number;
  /** ROI calculation time (ms) */
  roi_latency_ms: number;
  /** Total workflow time (ms) */
  total_latency_ms: number;
  /** Analysis timestamp */
  timestamp: string;
  /** Analysis warnings */
  warnings: string[];
}

/**
 * Response for listing opportunities
 */
export interface OpportunityListResponse {
  /** Total opportunities */
  total_count: number;
  /** Number of quick wins */
  quick_wins_count: number;
  /** Number of steady plays — the meaningful middle bucket (T6). */
  steady_plays_count?: number;
  /** Number of strategic bets */
  strategic_bets_count: number;
  /** Low-value opportunities hidden (ROI at or below break-even) (T6). */
  suppressed_count?: number;
  /** List of opportunities */
  opportunities: PrioritizedOpportunity[];
  /** Total potential value */
  total_addressable_value: number;
}

/**
 * Health check response for gap analysis service
 */
export interface GapHealthResponse {
  /** Service status */
  status: string;
  /** Gap Analyzer agent status */
  agent_available: boolean;
  /** Last analysis timestamp */
  last_analysis?: string;
  /** Analyses in last 24 hours */
  analyses_24h: number;
}
