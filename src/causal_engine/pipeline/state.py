"""State definitions for multi-library causal pipeline.

This module defines TypedDict structures for cross-library state propagation
in sequential, parallel, and hierarchical pipeline flows.
"""

import operator
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    NotRequired,
    Optional,
    TypedDict,
)

if TYPE_CHECKING:
    import pandas as pd


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

    # === DATA (first-class DataFrame slot, #458) ===
    # Canonical in-state DataFrame for estimation. Promoted from the Wave-1
    # per-executor key sprawl (`state["filters"]["estimation_data"]`,
    # `state["data_cache"]["estimation_data"]`, `state["filters"]["dataframe"]`)
    # to a single first-class field. `resolve_estimation_dataframe()` prefers
    # this slot and emits a `DeprecationWarning` when falling back to the
    # legacy locations. `data_cache` remains available below for ancillary
    # cached artifacts that are NOT the estimation DataFrame itself.
    estimation_data: NotRequired[Optional["pd.DataFrame"]]
    # Ancillary cached artifacts (keyed bag). Retained for back-compat; the
    # estimation DataFrame should now travel via `estimation_data` above.
    data_cache: NotRequired[Optional[Dict[str, Any]]]

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
    # H8/H9: how the consensus effect was combined ("inverse_variance" when all
    # libraries report a positive SE, else "confidence") + the real mean pairwise
    # agreement score (None when no pairwise agreement could be computed).
    consensus_weighting: NotRequired[str]
    library_agreement_score: NotRequired[Optional[float]]

    # === C-6 EXTRACTED CHANNELS (added phase C-6 of GH #354) ===
    # All three are NotRequired[Optional[...]] so callers that
    # constructed PipelineState as a strict total TypedDict (e.g. the
    # round-trip fixtures in `tests/.../test_state.py:251`) remain
    # type-valid without listing these new keys. This makes the C-6
    # extension fully additive at the static-typing level (closes the
    # codex iter-0 MEDIUM: `Optional[...]` alone does NOT make a key
    # optional in a `total=True` TypedDict — `NotRequired` does).
    #
    # NetworkX structural-quality summary (extracted from networkx_result):
    # {n_nodes, n_edges, is_dag, has_treatment_outcome_path,
    #  structural_quality (0..1), n_cycles}.
    # Distinct from `graph_metrics` (centrality only) and `causal_graph`
    # (full payload); this channel feeds the consensus-confidence
    # modulator without forcing downstream consumers to dig through the
    # full NetworkX result.
    graph_quality: NotRequired[Optional[Dict[str, Any]]]

    # CausalML uplift-channel summary (extracted from causalml_result):
    # {auuc, qini, ate, ate_ci_lower, ate_ci_upper, n_samples,
    #  treatment_groups, control_name}. Carried SEPARATELY from
    # `consensus_effect` because uplift answers a different question
    # (population-targeting quality, not effect magnitude).
    uplift_summary: NotRequired[Optional[Dict[str, Any]]]

    # Per-library metric-type bookkeeping for the aggregator:
    # {"dowhy": "ate", "econml": "ate", "causalml": "ate"}. Lets the
    # aggregator distinguish ATE-track contributions from uplift-track
    # contributions when future executors emit non-ATE metrics.
    library_metric_types: NotRequired[Optional[Dict[str, str]]]

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

    # First-class DataFrame field (#458). The orchestrator copies this into
    # `PipelineState["estimation_data"]` so executors can resolve it via
    # `resolve_estimation_dataframe()` without per-executor key drift.
    estimation_data: NotRequired[Optional["pd.DataFrame"]]

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
