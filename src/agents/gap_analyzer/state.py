"""State definitions for Gap Analyzer Agent.

This module defines the LangGraph state and all associated TypedDict structures
for the gap analyzer workflow: gap detection → ROI calculation → prioritization.
"""

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict


class PerformanceGap(TypedDict):
    """Individual performance gap identified in segment analysis.

    A gap represents the difference between current performance and a target
    (which could be a predefined target, peer benchmark, potential, or prior period).
    """

    gap_id: str  # Unique identifier: "{segment}_{segment_value}_{metric}"
    metric: str  # KPI name (e.g., "trx", "market_share", "conversion_rate")
    segment: str  # Segmentation dimension (e.g., "region", "specialty")
    segment_value: str  # Specific value (e.g., "Northeast", "Oncology")
    current_value: float  # Actual current performance
    target_value: float  # Comparison target
    gap_size: float  # Absolute gap (target - current)
    gap_percentage: float  # Percentage gap ((target - current) / target * 100)
    gap_type: Literal["vs_target", "vs_benchmark", "vs_potential", "temporal"]


class ConfidenceIntervalDict(TypedDict):
    """Bootstrap confidence interval for ROI estimates."""

    lower_bound: float  # 2.5th percentile
    median: float  # 50th percentile
    upper_bound: float  # 97.5th percentile
    probability_positive: float  # P(ROI > 1x)
    probability_target: float  # P(ROI > target)


class ROIEstimate(TypedDict):
    """ROI estimate for closing a specific performance gap.

    Uses pharmaceutical-specific economics from ROI methodology:
    - 6 value drivers (TRx lift, patient ID, action rate, ITP, data quality, drift)
    - Bootstrap confidence intervals (1,000 simulations)
    - Causal attribution framework
    - Risk adjustment

    Reference: docs/roi_methodology.md
    """

    gap_id: str  # References PerformanceGap.gap_id
    estimated_revenue_impact: float  # Annual revenue impact (USD)
    estimated_cost_to_close: float  # One-time cost to close gap (USD)
    expected_roi: float  # Base ROI ratio ((revenue - cost) / cost)
    risk_adjusted_roi: float  # ROI after risk adjustment
    payback_period_months: int  # Months to recoup investment (1-24)

    # Confidence interval from bootstrap
    confidence_interval: Optional[ConfidenceIntervalDict]  # 95% CI

    # Attribution
    attribution_level: str  # "full", "partial", "shared", "minimal"
    attribution_rate: float  # 0.0-1.0

    # Risk factors
    total_risk_adjustment: float  # Combined risk adjustment (0.0-1.0)

    # Value breakdown by driver
    value_by_driver: Optional[Dict[str, float]]  # e.g., {"trx_lift": 850000}

    # Legacy fields for backwards compatibility
    confidence: float  # Legacy confidence in estimate (0.0-1.0)
    assumptions: List[str]  # Economic assumptions made


class PrioritizedOpportunity(TypedDict):
    """Prioritized gap with ROI estimate and action recommendation.

    Combines gap detection and ROI estimation with actionable recommendations,
    difficulty assessment, and time-to-impact forecasting.
    """

    rank: int  # Priority rank (1 = highest ROI)
    gap: PerformanceGap  # The identified gap
    roi_estimate: ROIEstimate  # ROI analysis
    recommended_action: str  # Specific action to close gap
    implementation_difficulty: Literal["low", "medium", "high"]
    time_to_impact: str  # Expected time to see results (e.g., "1-3 months")


class GapAnalyzerState(TypedDict):
    """Complete LangGraph state for Gap Analyzer agent workflow.

    Workflow: gap_detector → roi_calculator → prioritizer → (formatter)

    The state accumulates:
    1. Input parameters (query, metrics, segments, configuration)
    2. Gap detection results (gaps_detected, gaps_by_segment)
    3. ROI calculations (roi_estimates, total_addressable_value)
    4. Prioritization (ranked opportunities, quick wins, strategic bets)
    5. Execution metadata (latencies, status, errors)
    """

    # === INPUT ===
    query: str  # Natural language query
    metrics: List[str]  # KPIs to analyze (e.g., ["trx", "market_share"])
    segments: List[str]  # Segmentation dimensions (e.g., ["region", "specialty"])
    brand: str  # Brand identifier (e.g., "kisqali")
    time_period: str  # Analysis period (e.g., "current_quarter", "2024-Q3")
    filters: Optional[Dict[str, Any]]  # Additional filters

    # === CONFIGURATION ===
    gap_type: Literal["vs_target", "vs_benchmark", "vs_potential", "temporal", "all"]
    min_gap_threshold: float  # Minimum gap % to report (e.g., 5.0)
    max_opportunities: int  # Maximum opportunities to return (e.g., 10)

    # === UPLIFT CONTEXT (from heterogeneous_optimizer, optional) ===
    # When uplift analysis is available, it enhances ROI calculations
    uplift_auuc: Optional[float]  # Area Under Uplift Curve (0-1)
    uplift_qini: Optional[float]  # Qini coefficient
    uplift_targeting_efficiency: Optional[float]  # Targeting efficiency (0-1)
    uplift_by_segment: Optional[Dict[str, Any]]  # Segment-level uplift scores

    # === DETECTION OUTPUTS (from gap_detector node) ===
    gaps_detected: Optional[List[PerformanceGap]]  # All gaps above threshold
    gaps_by_segment: Optional[Dict[str, List[PerformanceGap]]]  # Gaps grouped by segment
    total_gap_value: Optional[float]  # Sum of all gap sizes

    # === ROI OUTPUTS (from roi_calculator node) ===
    roi_estimates: Optional[List[ROIEstimate]]  # ROI for each gap
    total_addressable_value: Optional[float]  # Total potential revenue impact

    # === PRIORITIZATION OUTPUTS (from prioritizer node) ===
    prioritized_opportunities: Optional[List[PrioritizedOpportunity]]  # All opportunities ranked
    quick_wins: Optional[List[PrioritizedOpportunity]]  # Low difficulty, high ROI (top 5)
    strategic_bets: Optional[List[PrioritizedOpportunity]]  # High impact, high difficulty (top 5)

    # === SUMMARY (from formatter node or final output) ===
    executive_summary: Optional[str]  # Executive-level summary
    key_insights: Optional[List[str]]  # 3-5 key findings

    # === EXECUTION METADATA ===
    detection_latency_ms: int  # Gap detection time
    roi_latency_ms: int  # ROI calculation time
    total_latency_ms: int  # Total workflow time
    segments_analyzed: int  # Number of segments analyzed

    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]  # Accumulated errors
    warnings: Annotated[List[str], operator.add]  # Accumulated warnings
    status: Literal["pending", "detecting", "calculating", "prioritizing", "completed", "failed"]

    # ========================================================================
    # B7.4: Multi-Library Support (Pipeline-Aware ROI & Confidence)
    # ========================================================================

    # Library execution plan
    library_execution_plan: Optional[List[str]]  # e.g., ["networkx", "dowhy", "econml", "causalml"]
    library_execution_mode: Optional[Literal["sequential", "parallel"]]
    libraries_executed: Optional[List[str]]  # Actually executed libraries
    libraries_skipped: Optional[List[str]]  # Skipped due to validation or errors

    # Multi-library confidence scoring
    library_confidence_scores: Optional[Dict[str, float]]  # Per-library confidence (0-1)
    library_agreement_score: Optional[float]  # Overall agreement between libraries (0-1)
    library_consensus_effect: Optional[float]  # Confidence-weighted consensus effect
    effect_estimate_variance: Optional[float]  # Variance across library effect estimates

    # Pipeline-aware ROI estimates
    pipeline_roi_adjustment: Optional[float]  # ROI adjustment factor from pipeline confidence
    cross_validated_roi: Optional[bool]  # Whether ROI was cross-validated across libraries
    roi_confidence_source: Optional[Literal[
        "single_library",  # ROI from single library
        "multi_library_consensus",  # ROI from consensus of multiple libraries
        "cross_validated",  # ROI cross-validated (DoWhy ↔ CausalML)
        "pipeline_orchestrated",  # Full pipeline orchestration
    ]]

    # Causal library results feeding into ROI
    dowhy_effect_estimate: Optional[float]  # ATE from DoWhy
    dowhy_effect_confidence: Optional[float]  # Confidence from DoWhy (0-1)
    econml_cate_estimate: Optional[float]  # CATE from EconML
    econml_cate_confidence: Optional[float]  # Confidence from EconML (0-1)
    causalml_uplift_estimate: Optional[float]  # Uplift from CausalML
    causalml_uplift_confidence: Optional[float]  # Confidence from CausalML (0-1)
    networkx_graph_confidence: Optional[float]  # Graph structure confidence from NetworkX (0-1)

    # Cross-library validation
    cross_library_validation: Optional[Dict[str, Any]]  # Validation results between libraries
    validation_passed: Optional[bool]  # Whether cross-validation passed thresholds

    # Multi-library routing metadata
    question_type: Optional[Literal[
        "performance_gap",  # Gap analysis → DoWhy/EconML primary
        "roi_optimization",  # ROI focus → CausalML primary
        "system_analysis",  # Impact flow → NetworkX primary
        "comprehensive",  # All libraries
    ]]
    routing_confidence: Optional[float]  # Confidence in library routing decision
    routing_rationale: Optional[str]  # Why this library routing was chosen


# Type aliases for output contract compliance
GapAnalyzerOutput = TypedDict(
    "GapAnalyzerOutput",
    {
        "prioritized_opportunities": List[PrioritizedOpportunity],
        "quick_wins": List[PrioritizedOpportunity],
        "strategic_bets": List[PrioritizedOpportunity],
        "total_addressable_value": float,
        "total_gap_value": float,
        "segments_analyzed": int,
        "executive_summary": str,
        "key_insights": List[str],
        "detection_latency_ms": int,
        "roi_latency_ms": int,
        "total_latency_ms": int,
        "confidence": float,
        "warnings": List[str],
        "requires_further_analysis": bool,
        "suggested_next_agent": Optional[str],
        # B7.4: Multi-Library Support Output
        "libraries_used": Optional[List[str]],
        "library_agreement_score": Optional[float],
        "library_consensus_effect": Optional[float],
        "cross_validated_roi": Optional[bool],
        "roi_confidence_source": Optional[str],
        "question_type": Optional[str],
    },
)
