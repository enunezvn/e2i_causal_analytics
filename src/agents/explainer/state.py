"""
E2I Explainer Agent - State Definitions
Version: 4.2
Purpose: LangGraph state for natural language explanations
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, NotRequired, Optional, TypedDict
from uuid import UUID


class AnalysisContext(TypedDict):
    """Context extracted from prior analysis."""

    source_agent: str
    analysis_type: str
    key_findings: List[str]
    data_summary: Dict[str, Any]
    confidence: float
    warnings: List[str]


class Insight(TypedDict):
    """Extracted insight from analysis."""

    insight_id: str
    category: Literal["finding", "recommendation", "warning", "opportunity"]
    statement: str
    supporting_evidence: List[str]
    confidence: float
    priority: int
    actionability: Literal["immediate", "short_term", "long_term", "informational"]


class NarrativeSection(TypedDict):
    """Section of generated narrative."""

    section_type: str
    title: str
    content: str
    supporting_data: Optional[Dict[str, Any]]


class ExplainerState(TypedDict):
    """Complete state for Explainer agent."""

    # === INPUT (NotRequired - provided by caller) ===
    query: NotRequired[str]
    analysis_results: NotRequired[List[Dict[str, Any]]]
    user_expertise: NotRequired[Literal["executive", "analyst", "data_scientist"]]
    output_format: NotRequired[Literal["narrative", "structured", "presentation", "brief"]]
    focus_areas: NotRequired[List[str]]

    # === CONTEXT ===
    analysis_context: NotRequired[List[AnalysisContext]]
    user_context: NotRequired[Dict[str, Any]]
    conversation_history: NotRequired[List[Dict[str, Any]]]

    # === MEMORY INTEGRATION ===
    session_id: NotRequired[str]  # For memory correlation
    memory_config: NotRequired[Dict[str, Any]]  # Memory configuration
    episodic_context: NotRequired[List[Dict[str, Any]]]  # Retrieved past explanations
    semantic_context: NotRequired[Dict[str, Any]]  # Knowledge graph entities
    working_memory_messages: NotRequired[List[Dict[str, Any]]]  # Cached conversation

    # === REASONING OUTPUTS (Required) ===
    extracted_insights: List[Insight]
    narrative_structure: NotRequired[List[str]]
    key_themes: NotRequired[List[str]]

    # === NARRATIVE OUTPUTS (Required) ===
    executive_summary: str
    detailed_explanation: str
    narrative_sections: NotRequired[List[NarrativeSection]]

    # === SUPPLEMENTARY OUTPUTS ===
    visual_suggestions: NotRequired[List[Dict[str, Any]]]
    follow_up_questions: NotRequired[List[str]]
    related_analyses: NotRequired[List[str]]

    # === EXECUTION METADATA (NotRequired - populated during execution) ===
    assembly_latency_ms: NotRequired[int]
    reasoning_latency_ms: NotRequired[int]
    generation_latency_ms: NotRequired[int]
    total_latency_ms: NotRequired[int]
    model_used: NotRequired[str]

    # === ERROR HANDLING (Required outputs) ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal["pending", "assembling", "reasoning", "generating", "completed", "failed"]

    # === AUDIT CHAIN ===
    audit_workflow_id: NotRequired[UUID]

    # ========================================================================
    # V4.4: Causal Discovery Integration
    # ========================================================================

    # Discovered DAG from causal discovery module
    discovered_dag_adjacency: Optional[List[List[int]]]  # Adjacency matrix
    discovered_dag_nodes: Optional[List[str]]  # Node names
    discovered_dag_edge_types: Optional[Dict[str, str]]  # Edge types (DIRECTED, BIDIRECTED)

    # Discovery gate decision
    discovery_gate_decision: Optional[Literal["accept", "review", "reject", "augment"]]
    discovery_gate_confidence: Optional[float]  # Gate confidence [0, 1]

    # Causal vs predictive rankings from Feature Analyzer
    causal_rankings: Optional[List[Dict[str, Any]]]  # FeatureRanking dicts
    predictive_rankings: Optional[List[Dict[str, Any]]]
    rank_correlation: Optional[float]  # Spearman correlation between rankings
    divergent_features: Optional[List[str]]  # Features with high rank difference
    causal_only_features: Optional[List[str]]  # Features in causal but not predictive
    predictive_only_features: Optional[List[str]]  # Features in predictive but not causal
    concordant_features: Optional[List[str]]  # Features with similar rankings
