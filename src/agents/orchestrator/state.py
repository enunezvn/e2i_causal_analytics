"""State definition for orchestrator agent."""

import operator
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict


class ParsedEntity(TypedDict, total=False):
    """Entity extracted from user query."""

    type: Literal["brand", "region", "kpi", "time_period", "hcp_id", "patient_id"]
    value: str
    confidence: float  # 0.0 - 1.0
    source: Literal["exact", "fuzzy", "inferred"]


class ParsedQuery(TypedDict, total=False):
    """Output from NLP layer, input to Orchestrator."""

    query_id: str
    raw_query: str
    intent: Literal[
        "causal_impact",  # What is the effect of X on Y?
        "gap_analysis",  # Where are the gaps/opportunities?
        "heterogeneous",  # Who responds best to X?
        "experiment_design",  # Design an experiment for X
        "prediction",  # Predict Y for entity Z
        "explanation",  # Explain the analysis
        "health_check",  # System health status
        "drift_check",  # Model/data drift status
        "resource_optimize",  # Budget/resource allocation
        "ml_training",  # Train a new model (Tier 0)
        "feature_analysis",  # Feature importance (Tier 0)
        "model_deploy",  # Deploy model (Tier 0)
    ]
    entities: List[ParsedEntity]
    confidence: float
    ambiguity_flag: bool
    context: Optional[Dict[str, Any]]  # RAG-retrieved context
    timestamp: datetime


class IntentClassification(TypedDict, total=False):
    """Fast intent classification result."""

    primary_intent: Literal[
        "causal_effect",  # → Causal Impact Agent
        "performance_gap",  # → Gap Analyzer Agent
        "segment_analysis",  # → Heterogeneous Optimizer
        "experiment_design",  # → Experiment Designer
        "prediction",  # → Prediction Synthesizer
        "resource_allocation",  # → Resource Optimizer
        "explanation",  # → Explainer Agent
        "system_health",  # → Health Score Agent
        "drift_check",  # → Drift Monitor Agent
        "feedback",  # → Feedback Learner
        "general",  # → Direct response
    ]
    confidence: float
    secondary_intents: List[str]
    requires_multi_agent: bool


class LibraryRoutingDecision(TypedDict, total=False):
    """Multi-library routing decision from pipeline router.

    B7.4: Multi-Library Support
    Routes causal queries to appropriate library based on question type.
    """

    primary_library: Literal["networkx", "dowhy", "econml", "causalml"]
    secondary_libraries: List[str]
    execution_mode: Literal["sequential", "parallel"]
    routing_confidence: float
    question_type: Literal[
        "causal_relationship",  # "Does X cause Y?" → DoWhy
        "effect_heterogeneity",  # "How does effect vary?" → EconML
        "targeting",  # "Who should we target?" → CausalML
        "system_analysis",  # "How does impact flow?" → NetworkX
        "comprehensive",  # All four libraries
    ]
    routing_rationale: str


class AgentDispatch(TypedDict, total=False):
    """Agent dispatch specification."""

    agent_name: str
    priority: Literal["low", "medium", "high", "critical"]  # Contract: priority levels
    parameters: Dict[str, Any]
    timeout_ms: int
    fallback_agent: Optional[str]
    dispatch_id: Optional[str]  # Contract: unique dispatch identifier
    execution_mode: Optional[Literal["sequential", "parallel"]]  # Contract: execution mode


class AgentResult(TypedDict, total=False):
    """Result from dispatched agent."""

    agent_name: str
    success: bool
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    latency_ms: int


class Citation(TypedDict, total=False):
    """Source citation for response."""

    source_type: Literal["causal_path", "agent_activity", "business_metric", "ml_model"]
    source_id: str
    relevance: float


class OrchestratorState(TypedDict, total=False):
    """Complete state for Orchestrator agent.

    This is the main coordination agent that routes queries to specialized
    agents and synthesizes their responses.
    """

    # ========================================================================
    # INPUT FIELDS (From NLP layer or direct query)
    # ========================================================================

    # Query information
    query: str
    query_id: Optional[str]
    parsed_query: Optional[ParsedQuery]

    # User context
    user_id: Optional[str]
    session_id: Optional[str]
    user_context: Dict[str, Any]  # expertise level, preferences
    conversation_history: Optional[List[Dict]]

    # ========================================================================
    # NODE 1 OUTPUT: Intent Classification
    # ========================================================================

    # Classification result
    intent: Optional[IntentClassification]
    entities_extracted: Optional[Dict[str, List[str]]]

    # Timing
    classification_latency_ms: int

    # ========================================================================
    # NODE 1.5 OUTPUT: RAG Context Retrieval (Optional)
    # ========================================================================

    # RAG context
    rag_context: Optional[Dict[str, Any]]  # Retrieved context from RAG

    # Timing
    rag_latency_ms: int

    # ========================================================================
    # NODE 2 OUTPUT: Routing
    # ========================================================================

    # Dispatch plan
    dispatch_plan: Optional[List[AgentDispatch]]
    parallel_groups: Optional[List[List[str]]]  # Agents that can run in parallel

    # Timing
    routing_latency_ms: int

    # ========================================================================
    # NODE 2.5 OUTPUT: Library Routing (B7.4 Multi-Library Support)
    # ========================================================================

    # Library routing for causal queries
    library_routing: Optional[LibraryRoutingDecision]
    libraries_to_execute: Optional[List[str]]  # Ordered library execution plan
    library_execution_mode: Optional[Literal["sequential", "parallel"]]
    library_routing_latency_ms: int

    # ========================================================================
    # NODE 3 OUTPUT: Dispatching
    # ========================================================================

    # Agent execution results
    agent_results: Annotated[List[AgentResult], operator.add]

    # Timing
    dispatch_latency_ms: int

    # ========================================================================
    # NODE 4 OUTPUT: Synthesis
    # ========================================================================

    # Synthesized response
    synthesized_response: Optional[str]
    response_confidence: float
    recommendations: Optional[List[Dict[str, Any]]]
    follow_up_suggestions: Optional[List[str]]

    # Citations and visualizations
    citations: Optional[List[Citation]]
    visualizations: Optional[List[Dict[str, Any]]]

    # Timing
    synthesis_latency_ms: int

    # ========================================================================
    # WORKFLOW STATE
    # ========================================================================

    # Current phase
    current_phase: Literal["classifying", "routing", "dispatching", "synthesizing", "complete"]

    # Overall status
    status: Literal["pending", "processing", "completed", "failed"]

    # ========================================================================
    # METADATA
    # ========================================================================

    # Timing
    start_time: Optional[str]
    total_latency_ms: int

    # Agents used
    agents_dispatched: List[str]

    # ========================================================================
    # ERROR HANDLING
    # ========================================================================

    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    fallback_used: bool
    error: Optional[str]
    error_type: Optional[str]
