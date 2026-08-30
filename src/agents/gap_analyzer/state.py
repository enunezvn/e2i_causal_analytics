"""State definitions for Gap Analyzer Agent.

This module defines the LangGraph state and all associated TypedDict structures
for the gap analyzer workflow: gap detection → ROI calculation → prioritization.
"""

import operator
from typing import Annotated, Any, Dict, List, Literal, NotRequired, Optional, TypedDict
from uuid import UUID


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
    # market_share gaps only: the brand's current TRx in this segment (same
    # window/aggregation as current_value). ROI valuation converts share points
    # to TRx-equivalents via relative share growth x this volume; without it a
    # share gap cannot be valued in dollars (share points are NOT scripts) and
    # the ROI calculator fails closed to $0. Attached by GapDetectorNode, the
    # only place the segment's trx and market_share sit on the same frame row.
    segment_trx: NotRequired[float]


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

    # Market landscape (INFORMATIONAL, surface-only — added 2026-06-20). Curated
    # competitor density for the brand's indication; does NOT alter the ROI or the
    # ranking (the prioritizer still sorts on the unchanged risk_adjusted_roi).
    competitor_products_count: NotRequired[int]
    competitor_density_label: NotRequired[str]  # limited / moderate / crowded / unknown
    competitor_drug_names: NotRequired[List[str]]

    # Value breakdown by driver
    value_by_driver: Optional[Dict[str, float]]  # e.g., {"trx_lift": 850000}

    # Legacy fields for backwards compatibility
    confidence: float  # Legacy confidence in estimate (0.0-1.0)
    assumptions: List[str]  # Economic assumptions made

    # V4.4: Causal-evidence ROI adjustment (written by PrioritizerNode when causal
    # evidence is present). Made explicit here to stop the silent-extra-key drift;
    # V4.4 historically wrote these onto a dict(roi_estimate) and re-cast to ROIEstimate.
    causal_adjustment_factor: NotRequired[float]
    causal_adjustment_reason: NotRequired[Optional[str]]

    # #357: Instrument-availability ROI adjustment (written by PrioritizerNode when a
    # STRONG instrument is available for the opportunity's feature). Independent of and
    # additive to the V4.4 causal adjustment above (compounded — see prioritizer).
    instrument_adjustment_factor: NotRequired[float]
    instrument_adjustment_reason: NotRequired[Optional[str]]

    # Label-gater: written by PrioritizerNode when label_segmentation is enabled and a
    # brand indicated-population is resolvable. ``off_label`` is True ONLY for a
    # label-evidenced violation; off-label opportunities are RANK-DEMOTED (sink below
    # on-label), never deleted, and the ROI values above are NEVER altered (rank-only).
    off_label: NotRequired[bool]
    off_label_reason: NotRequired[str]
    label_verdict: NotRequired[str]  # on_label | off_label | mixed | indeterminate
    label_evidence_confirmed: NotRequired[bool]


class InstrumentSpec(TypedDict):
    """#357 P-2 producer input: per-feature instrument specification for the IV first stage.

    Maps to the columns needed to run a real first-stage regression against the tier0
    frame for a given opportunity feature (feature_name matches gap["metric"]).
    """

    treatment_col: str  # D: endogenous treatment column in the tier0 frame
    instrument_cols: List[str]  # Z: one or more instrument columns (>= 1)
    outcome_col: NotRequired[str]  # Y: outcome; defaults to the feature's metric column
    covariate_cols: NotRequired[List[str]]  # X: exogenous controls (optional)


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

    # T6: 3-bucket category (quick_win | steady_play | strategic_bet) stamped by
    # the prioritizer via the shared opportunity_classification SSOT. NO "other".
    category: NotRequired[str]
    # T6: human-readable explanation of the difficulty rating (the cost/gap-size
    # factors that drove the score) — previously computed-then-discarded; now
    # surfaced for the drill-down drawer.
    difficulty_rationale: NotRequired[str]


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
    # #1743 (sibling of #1734): the tier0 passthrough frame itself must NEVER
    # ride graph state. On the chat path this graph runs nested under the
    # streamed chatbot graph, and every node's on_chain_start/on_chain_end
    # event re-serializes the full state to the client (measured on het: one
    # 377.6 MB chat turn, eval 4.4) — besides leaking patient-level rows past
    # the aggregates-only frontend contract and making state unserializable for
    # any checkpointer (#1351 class). Callers stash the frame in
    # src.utils.frame_registry and pass this opaque handle; gap_detector and
    # instrument_analyzer resolve it via resolve_state_frame().
    tier0_frame_ref: NotRequired[Optional[str]]

    # #357: per-feature instrument specification for the IV first stage (producer input).
    # Maps feature_name (== gap["metric"]) -> InstrumentSpec describing the treatment,
    # instrument, outcome, and covariate columns to slice out of the tier0 frame
    # (resolved via tier0_frame_ref, #1743). Absent => the IV step is a no-op and the
    # instrument-availability bonus never fires.
    instrument_specs: Optional[Dict[str, "InstrumentSpec"]]

    # === CONFIGURATION ===
    gap_type: Literal["vs_target", "vs_benchmark", "vs_potential", "temporal", "all"]
    min_gap_threshold: float  # Minimum gap % to report (e.g., 5.0)
    max_opportunities: int  # Maximum opportunities to return (e.g., 10)

    # #874: per-RUN synthetic-substrate opt-in (the #851/#872 provenance plumb).
    # The agent factory registers a real-mode instance (include_synthetic=False baked
    # into the compiled graph), so a per-dispatch opt-in must travel through the run
    # input: gap_detector resolves a per-run connector pair honoring this flag.
    # Absent => the agent's constructor flag governs (backward compatible).
    include_synthetic: NotRequired[bool]

    # Label-gater (opt-in): when ``label_segmentation`` is truthy AND ``brand`` is
    # present, PrioritizerNode resolves the brand's FDA-indicated population and flags
    # opportunities whose gap segment falls outside it as off_label (rank-demoted, ROI
    # untouched). Absent/falsey => unchanged behaviour (no gating). ``indication``
    # scopes the lookup; absent => the brand's primary indication (gap_analyzer loads
    # business_metrics, not patient_journeys, so diagnosis-based resolution is N/A here).
    indication: NotRequired[str]
    label_segmentation: NotRequired[bool]

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
    # T6: the meaningful middle ground — solid earners that are neither quick wins
    # nor strategic bets (medium effort, or high-effort-but-modest bets).
    steady_plays: NotRequired[Optional[List[PrioritizedOpportunity]]]
    strategic_bets: Optional[List[PrioritizedOpportunity]]  # High impact, high difficulty (top 5)
    # T6: count of opportunities suppressed as low-value noise (ROI <= break-even);
    # surfaced for transparency so an empty/short list never looks broken.
    suppressed_count: NotRequired[int]

    # === SUMMARY (from formatter node or final output) ===
    executive_summary: Optional[str]  # Executive-level summary
    key_insights: Optional[List[str]]  # 3-5 key findings

    # === EXECUTION METADATA ===
    detection_latency_ms: int  # Gap detection time
    roi_latency_ms: int  # ROI calculation time
    total_latency_ms: int  # Total workflow time
    segments_analyzed: int  # Number of segments analyzed
    # #1834: the concrete window the analysis compared, resolved by gap_detector from
    # ``time_period`` via the shared grammar (src.utils.gap_time_period):
    # {"time_period", "period_start", "period_end", "prior_start", "prior_end"} as
    # ISO dates. Absent/None when gap_detector failed before resolving it.
    resolved_period: NotRequired[Optional[Dict[str, str]]]

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
    roi_confidence_source: Optional[
        Literal[
            "single_library",  # ROI from single library
            "multi_library_consensus",  # ROI from consensus of multiple libraries
            "cross_validated",  # ROI cross-validated (DoWhy ↔ CausalML)
            "pipeline_orchestrated",  # Full pipeline orchestration
        ]
    ]

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
    question_type: Optional[
        Literal[
            "performance_gap",  # Gap analysis → DoWhy/EconML primary
            "roi_optimization",  # ROI focus → CausalML primary
            "system_analysis",  # Impact flow → NetworkX primary
            "comprehensive",  # All libraries
        ]
    ]
    routing_confidence: Optional[float]  # Confidence in library routing decision
    routing_rationale: Optional[str]  # Why this library routing was chosen

    # Audit chain (tamper-evident logging)
    audit_workflow_id: Optional[UUID]

    # ========================================================================
    # V4.4: Causal Discovery Integration
    # ========================================================================

    # Causal rankings from Feature Analyzer (DriverRanker output)
    causal_rankings: Optional[List[Dict[str, Any]]]  # List of FeatureRanking dicts
    # Each dict: feature_name, causal_rank, predictive_rank, causal_score,
    #            predictive_score, rank_difference, is_direct_cause, path_length

    # Discovery gate decision from upstream agent
    discovery_gate_decision: Optional[Literal["accept", "review", "reject", "augment"]]
    discovery_gate_confidence: Optional[float]  # Gate confidence [0, 1]

    # Causal importance for gap features
    causal_importance: Optional[Dict[str, float]]  # {feature_name: causal_score}

    # Feature categorization from causal analysis
    direct_cause_features: Optional[List[str]]  # Features with direct edge to target
    divergent_features: Optional[List[str]]  # Features with |rank_diff| > threshold
    causal_only_features: Optional[List[str]]  # Causal signal, no predictive
    predictive_only_features: Optional[List[str]]  # Predictive signal, no causal

    # Causal prioritization outputs
    causal_evidence_warnings: Optional[List[str]]  # Warnings about causal evidence

    # ========================================================================
    # #357: Instrument-availability (IV first-stage strength) integration
    # ========================================================================

    # Per-feature instrument strength from a REAL IV first-stage F-test
    # (src/causal_engine/iv), produced by InstrumentAnalyzerNode from instrument_specs
    # + the tier0 frame (resolved via tier0_frame_ref, #1743). Maps feature_name ->
    # IVDiagnostics.to_dict() output, i.e. keys:
    #   "instrument_strength" (str: "strong"|"moderate"|"weak"|"very_weak"),
    #   "is_weak_instrument" (bool), "first_stage_f_stat" (float).
    # Absent/None => no instrument analysis available => no bonus (fail-closed, like V4.4).
    instrument_strength_by_feature: Optional[Dict[str, Dict[str, Any]]]


# Type aliases for output contract compliance
GapAnalyzerOutput = TypedDict(
    "GapAnalyzerOutput",
    {
        "prioritized_opportunities": List[PrioritizedOpportunity],
        "quick_wins": List[PrioritizedOpportunity],
        # T6: the surfaced middle bucket and the count of value-destroying
        # candidates hidden by break-even suppression. Both are always emitted by
        # _build_output, so the typed contract must declare them.
        "steady_plays": List[PrioritizedOpportunity],
        "strategic_bets": List[PrioritizedOpportunity],
        "suppressed_count": int,
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
