# src/e2i/agents/orchestrator/classifier/schemas.py
"""
Pydantic models for the classification pipeline.

This module defines all data structures used across the 4-stage
classification pipeline:
- Stage 1: Feature Extraction
- Stage 2: Domain Mapping
- Stage 3: Dependency Detection
- Stage 4: Pattern Selection
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# ENUMS
# =============================================================================


class Domain(str, Enum):
    """Agent capability domains."""

    CAUSAL_ANALYSIS = "CAUSAL_ANALYSIS"
    HETEROGENEITY = "HETEROGENEITY"
    GAP_ANALYSIS = "GAP_ANALYSIS"
    EXPERIMENTATION = "EXPERIMENTATION"
    PREDICTION = "PREDICTION"
    MONITORING = "MONITORING"
    EXPLANATION = "EXPLANATION"
    COHORT_DEFINITION = "COHORT_DEFINITION"  # Patient/HCP cohort construction


class RoutingPattern(str, Enum):
    """Available routing patterns."""

    SINGLE_AGENT = "SINGLE_AGENT"
    PARALLEL_DELEGATION = "PARALLEL_DELEGATION"
    TOOL_COMPOSER = "TOOL_COMPOSER"
    CLARIFICATION_NEEDED = "CLARIFICATION_NEEDED"


class DependencyType(str, Enum):
    """Types of dependencies between sub-questions."""

    REFERENCE_CHAIN = "REFERENCE_CHAIN"  # "that", "those" referring back
    CONDITIONAL = "CONDITIONAL"  # "if X then Y"
    LOGICAL_SEQUENCE = "LOGICAL_SEQUENCE"  # Natural ordering required
    ENTITY_TRANSFORMATION = "ENTITY_TRANSFORMATION"  # Filtered entity set


# =============================================================================
# STAGE 1: FEATURE EXTRACTION MODELS
# =============================================================================


class StructuralFeatures(BaseModel):
    """Features extracted from query structure."""

    question_count: int = Field(default=1, ge=0)
    clause_count: int = Field(default=1, ge=1)
    has_conditional: bool = False
    has_comparison: bool = False
    has_sequence: bool = False
    word_count: int = Field(default=0, ge=0)
    connector_density: float = Field(default=0.0, ge=0.0, le=1.0)


class TemporalFeatures(BaseModel):
    """Features related to time references."""

    time_references: list[str] = Field(default_factory=list)
    time_span_count: int = Field(default=0, ge=0)
    has_future: bool = False
    has_past: bool = False


class EntityFeatures(BaseModel):
    """Features related to entities mentioned."""

    entity_types: list[str] = Field(default_factory=list)
    entity_mentions: list[str] = Field(default_factory=list)
    entity_type_count: int = Field(default=0, ge=0)


class IntentSignals(BaseModel):
    """Keyword signals indicating intent."""

    causal_keywords: list[str] = Field(default_factory=list)
    exploration_keywords: list[str] = Field(default_factory=list)
    prediction_keywords: list[str] = Field(default_factory=list)
    design_keywords: list[str] = Field(default_factory=list)
    explanation_keywords: list[str] = Field(default_factory=list)
    monitoring_keywords: list[str] = Field(default_factory=list)
    cohort_keywords: list[str] = Field(default_factory=list)  # Cohort construction signals


class ExtractedFeatures(BaseModel):
    """Complete feature set from Stage 1."""

    structural: StructuralFeatures
    temporal: TemporalFeatures
    entities: EntityFeatures
    intent_signals: IntentSignals
    raw_query: str


# =============================================================================
# STAGE 2: DOMAIN MAPPING MODELS
# =============================================================================


class DomainMatch(BaseModel):
    """A single domain match with confidence."""

    domain: Domain
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class DomainMapping(BaseModel):
    """Complete domain mapping from Stage 2."""

    domains_detected: list[DomainMatch]
    domain_count: int = Field(ge=0)
    primary_domain: Optional[Domain] = None
    is_multi_domain: bool = False


# =============================================================================
# STAGE 3: DEPENDENCY DETECTION MODELS
# =============================================================================


class SubQuestion(BaseModel):
    """A decomposed sub-question."""

    id: str
    text: str
    domains: list[Domain]
    primary_domain: Domain


class Dependency(BaseModel):
    """A dependency between two sub-questions."""

    from_id: str = Field(alias="from")
    to_id: str = Field(alias="to")
    dependency_type: DependencyType
    reason: str

    model_config = ConfigDict(populate_by_name=True)


class DependencyAnalysis(BaseModel):
    """Complete dependency analysis from Stage 3."""

    sub_questions: list[SubQuestion]
    dependencies: list[Dependency]
    has_dependencies: bool = False
    is_parallelizable: bool = True
    dependency_depth: int = Field(default=0, ge=0)


# =============================================================================
# STAGE 4: PATTERN SELECTION / FINAL OUTPUT
# =============================================================================


class ClassificationResult(BaseModel):
    """Final classification output."""

    routing_pattern: RoutingPattern
    target_agents: list[str] = Field(default_factory=list)
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    dependencies: list[Dependency] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

    # Metadata
    is_followup: bool = False
    context_source: Optional[str] = None
    complexity_warning: Optional[str] = None
    consultation_hints: list[str] = Field(default_factory=list)

    # Performance tracking
    classification_latency_ms: float = Field(default=0.0, ge=0.0)
    used_llm_layer: bool = False
