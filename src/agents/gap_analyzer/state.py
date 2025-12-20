"""State definitions for Gap Analyzer Agent.

This module defines the LangGraph state and all associated TypedDict structures
for the gap analyzer workflow: gap detection → ROI calculation → prioritization.
"""

from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
import operator


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


class ROIEstimate(TypedDict):
    """ROI estimate for closing a specific performance gap.

    Uses pharmaceutical-specific economics to estimate revenue impact,
    cost to close, expected ROI, and payback period.
    """

    gap_id: str  # References PerformanceGap.gap_id
    estimated_revenue_impact: float  # Annual revenue impact (USD)
    estimated_cost_to_close: float  # One-time cost to close gap (USD)
    expected_roi: float  # ROI ratio ((revenue - cost) / cost)
    payback_period_months: int  # Months to recoup investment (1-24)
    confidence: float  # Confidence in estimate (0.0-1.0)
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
    },
)
