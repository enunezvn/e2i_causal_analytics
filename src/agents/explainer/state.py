"""
E2I Explainer Agent - State Definitions
Version: 4.2
Purpose: LangGraph state for natural language explanations
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
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

    # === INPUT ===
    query: str
    analysis_results: List[Dict[str, Any]]
    user_expertise: Literal["executive", "analyst", "data_scientist"]
    output_format: Literal["narrative", "structured", "presentation", "brief"]
    focus_areas: Optional[List[str]]

    # === CONTEXT ===
    analysis_context: Optional[List[AnalysisContext]]
    user_context: Optional[Dict[str, Any]]
    conversation_history: Optional[List[Dict[str, Any]]]

    # === MEMORY INTEGRATION ===
    session_id: Optional[str]  # For memory correlation
    memory_config: Optional[Dict[str, Any]]  # Memory configuration
    episodic_context: Optional[List[Dict[str, Any]]]  # Retrieved past explanations
    semantic_context: Optional[Dict[str, Any]]  # Knowledge graph entities
    working_memory_messages: Optional[List[Dict[str, Any]]]  # Cached conversation

    # === REASONING OUTPUTS ===
    # Note: Required output from reasoning node
    extracted_insights: List[Insight]
    narrative_structure: Optional[List[str]]
    key_themes: Optional[List[str]]

    # === NARRATIVE OUTPUTS ===
    # Note: Required outputs from narrative generation
    executive_summary: str
    detailed_explanation: str
    narrative_sections: Optional[List[NarrativeSection]]

    # === SUPPLEMENTARY OUTPUTS ===
    visual_suggestions: Optional[List[Dict[str, Any]]]
    follow_up_questions: Optional[List[str]]
    related_analyses: Optional[List[str]]

    # === EXECUTION METADATA ===
    assembly_latency_ms: int
    reasoning_latency_ms: int
    generation_latency_ms: int
    total_latency_ms: int
    model_used: Optional[str]

    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal["pending", "assembling", "reasoning", "generating", "completed", "failed"]

    # === AUDIT CHAIN ===
    audit_workflow_id: Optional[UUID]

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
