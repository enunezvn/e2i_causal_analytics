/**
 * Gap Analysis Types
 * ==================
 *
 * TypeScript interfaces for the E2I Gap Analysis API.
 * Based on src/api/routes/gaps.py backend schemas.
 *
 * @module types/gaps
 */

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
  /** Analysis period (e.g., 'current_quarter', '2024-Q3') */
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
}

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
}

/**
 * Response from gap analysis
 */
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
  /** Number of strategic bets */
  strategic_bets_count: number;
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
