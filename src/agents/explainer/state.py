"""
E2I Explainer Agent - State Definitions
Version: 4.2
Purpose: LangGraph state for natural language explanations
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict


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

    # === REASONING OUTPUTS ===
    extracted_insights: Optional[List[Insight]]
    narrative_structure: Optional[List[str]]
    key_themes: Optional[List[str]]

    # === NARRATIVE OUTPUTS ===
    executive_summary: Optional[str]
    detailed_explanation: Optional[str]
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
    status: Literal[
        "pending", "assembling", "reasoning", "generating", "completed", "failed"
    ]
