"""State definitions for multi-library causal pipeline.

This module defines TypedDict structures for cross-library state propagation
in sequential, parallel, and hierarchical pipeline flows.
"""

import operator
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict


class PipelineStage(str, Enum):
    """Pipeline execution stages."""

    PENDING = "pending"
    ROUTING = "routing"
    GRAPH_ANALYSIS = "graph_analysis"  # NetworkX
    CAUSAL_VALIDATION = "causal_validation"  # DoWhy
    EFFECT_ESTIMATION = "effect_estimation"  # EconML
    UPLIFT_MODELING = "uplift_modeling"  # CausalML
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


class LibraryExecutionResult(TypedDict):
    """Result from a single library execution."""

    library: str  # "networkx", "dowhy", "econml", "causalml"
    success: bool
    latency_ms: int
    result: Optional[Dict[str, Any]]  # Library-specific result
    error: Optional[str]
    confidence: float  # 0.0-1.0
    warnings: List[str]


class PipelineConfig(TypedDict):
    """Configuration for pipeline execution."""

    # Execution mode
    mode: Literal["sequential", "parallel", "validation_loop", "hierarchical"]

    # Library selection
    libraries_enabled: List[str]  # ["networkx", "dowhy", "econml", "causalml"]
    primary_library: Optional[str]  # Primary library for this query

    # Timeouts
    stage_timeout_ms: int  # Timeout per stage (default: 30000)
    total_timeout_ms: int  # Total pipeline timeout (default: 120000)

    # Validation settings
    cross_validate: bool  # Whether to cross-validate between libraries
    min_agreement_threshold: float  # Min agreement for consensus (0.85)

    # Parallel execution settings
    max_parallel_libraries: int  # Max libraries to run in parallel (default: 4)
    fail_fast: bool  # Stop on first failure in parallel mode

    # Hierarchical settings
    segment_by_uplift: bool  # Use CausalML segments for EconML CATE
    nested_ci_level: float  # Confidence level for nested CIs (default: 0.95)


class PipelineState(TypedDict):
    """Complete state for multi-library causal pipeline.

    This state flows through the pipeline orchestrator and accumulates
    results from each library stage.

    Reference: docs/Data Architecture & Integration.html
    """

    # === INPUT ===
    query: str  # Natural language query
    question_type: Optional[str]  # Classified question type
    treatment_var: Optional[str]  # Treatment variable
    outcome_var: Optional[str]  # Outcome variable
    confounders: Optional[List[str]]  # Confounding variables
    effect_modifiers: Optional[List[str]]  # Effect modifiers
    data_source: str  # Data source identifier
    filters: Optional[Dict[str, Any]]  # Query filters

    # === CONFIGURATION ===
    config: PipelineConfig

    # === ROUTING ===
    routed_libraries: List[str]  # Libraries selected by router
    routing_confidence: float  # Confidence in routing decision
    routing_rationale: Optional[str]  # Explanation for routing

    # === LIBRARY RESULTS ===
    # NetworkX results (graph analysis)
    networkx_result: Optional[LibraryExecutionResult]
    causal_graph: Optional[Dict[str, Any]]  # DAG structure
    graph_metrics: Optional[Dict[str, float]]  # Centrality, paths, etc.

    # DoWhy results (causal validation)
    dowhy_result: Optional[LibraryExecutionResult]
    causal_effect: Optional[float]  # Estimated causal effect
    refutation_results: Optional[Dict[str, Any]]  # Refutation test results
    identification_method: Optional[str]  # How effect was identified

    # EconML results (heterogeneous effects)
    econml_result: Optional[LibraryExecutionResult]
    cate_by_segment: Optional[Dict[str, Any]]  # CATE per segment
    overall_ate: Optional[float]  # Average treatment effect
    heterogeneity_score: Optional[float]  # Effect heterogeneity

    # CausalML results (uplift modeling)
    causalml_result: Optional[LibraryExecutionResult]
    uplift_scores: Optional[Dict[str, Any]]  # Uplift by segment
    auuc: Optional[float]  # Area Under Uplift Curve
    qini: Optional[float]  # Qini coefficient
    targeting_recommendations: Optional[List[Dict[str, Any]]]

    # === AGGREGATED OUTPUTS ===
    # Consensus results (for parallel/validation modes)
    consensus_effect: Optional[float]  # Weighted consensus effect
    consensus_confidence: Optional[float]  # Agreement-based confidence
    library_agreement: Optional[Dict[str, float]]  # Pairwise agreement

    # Hierarchical results (nested analysis)
    nested_cate: Optional[Dict[str, Any]]  # CATE within uplift segments
    segment_confidence_intervals: Optional[Dict[str, Any]]

    # Final output
    executive_summary: Optional[str]
    key_insights: Optional[List[str]]
    recommended_actions: Optional[List[str]]

    # === EXECUTION METADATA ===
    current_stage: PipelineStage
    stage_latencies: Dict[str, int]  # Latency per stage
    total_latency_ms: int
    libraries_executed: List[str]
    libraries_skipped: List[str]

    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal["pending", "running", "completed", "failed", "partial"]


class PipelineInput(TypedDict):
    """Input contract for pipeline orchestrator."""

    query: str
    treatment_var: Optional[str]
    outcome_var: Optional[str]
    confounders: Optional[List[str]]
    effect_modifiers: Optional[List[str]]
    data_source: str
    filters: Optional[Dict[str, Any]]

    # Optional configuration overrides
    mode: Optional[Literal["sequential", "parallel", "validation_loop", "hierarchical"]]
    libraries_enabled: Optional[List[str]]
    cross_validate: Optional[bool]


class PipelineOutput(TypedDict):
    """Output contract for pipeline orchestrator."""

    # Core results
    question_type: str
    primary_result: Dict[str, Any]  # Result from primary library
    libraries_used: List[str]

    # Consensus (if applicable)
    consensus_effect: Optional[float]
    consensus_confidence: Optional[float]

    # Summary
    executive_summary: str
    key_insights: List[str]
    recommended_actions: List[str]

    # Metadata
    total_latency_ms: int
    status: Literal["completed", "failed", "partial"]
    warnings: List[str]
    errors: List[Dict[str, Any]]
