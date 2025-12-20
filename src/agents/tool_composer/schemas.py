# src/e2i/agents/tool_composer/schemas.py
"""
Pydantic models for the Tool Composer.

This module defines data structures for:
- Tool definitions and registry
- Composition requests and results
- Execution plans and steps
- Phase outputs
"""

from enum import Enum
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# ENUMS
# =============================================================================

class ToolCategory(str, Enum):
    """Tool capability categories."""
    CAUSAL = "CAUSAL"
    SEGMENTATION = "SEGMENTATION"
    GAP = "GAP"
    EXPERIMENT = "EXPERIMENT"
    PREDICTION = "PREDICTION"
    MONITORING = "MONITORING"


class CompositionStatus(str, Enum):
    """Status of a composition execution."""
    PENDING = "PENDING"
    DECOMPOSING = "DECOMPOSING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    SYNTHESIZING = "SYNTHESIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


# =============================================================================
# TOOL REGISTRY MODELS
# =============================================================================

@dataclass
class ToolSchema:
    """
    Schema for a composable tool.
    
    Tools are stateless functions exposed by agents that can be
    composed by the Tool Composer.
    """
    name: str
    description: str
    category: ToolCategory
    source_agent: str
    
    # JSON Schema format
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    
    # The actual callable (set at runtime)
    fn: Optional[Callable] = None
    
    # Composition flags
    composable: bool = True
    
    # Performance baselines
    avg_latency_ms: float = 500.0
    success_rate: float = 0.95
    
    # Dependencies: tools whose output this tool can consume
    can_consume_from: list[str] = field(default_factory=list)


# =============================================================================
# COMPOSITION REQUEST/RESULT MODELS
# =============================================================================

class SubQuestionInput(BaseModel):
    """A sub-question from the classifier."""
    id: str
    text: str
    primary_domain: str
    domains: list[str] = Field(default_factory=list)


class DependencyInput(BaseModel):
    """A dependency between sub-questions."""
    from_id: str = Field(alias="from")
    to_id: str = Field(alias="to")
    dependency_type: str
    reason: str

    model_config = ConfigDict(populate_by_name=True)


class CompositionRequest(BaseModel):
    """Request to compose a multi-faceted query."""
    query: str
    sub_questions: list[SubQuestionInput] = Field(default_factory=list)
    dependencies: list[DependencyInput] = Field(default_factory=list)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    context: dict = Field(default_factory=dict)


class CompositionResult(BaseModel):
    """Result of a tool composition."""
    composition_id: str
    query: str
    status: CompositionStatus
    
    # Phase outputs
    sub_questions: list[SubQuestionInput] = Field(default_factory=list)
    execution_plan: Optional["ExecutionPlan"] = None
    tool_outputs: dict[str, Any] = Field(default_factory=dict)
    response: Optional[str] = None
    
    # Timing
    total_latency_ms: float = 0.0
    decompose_latency_ms: float = 0.0
    plan_latency_ms: float = 0.0
    execute_latency_ms: float = 0.0
    synthesize_latency_ms: float = 0.0
    
    # Error info
    error_message: Optional[str] = None


# =============================================================================
# EXECUTION PLAN MODELS
# =============================================================================

class ExecutionStep(BaseModel):
    """A single step in the execution plan."""
    step_id: str
    step_number: int
    tool_name: str
    tool_id: Optional[str] = None
    
    # Input configuration
    input_params: dict = Field(default_factory=dict)
    input_from_steps: list[str] = Field(default_factory=list)
    
    # Dependencies
    depends_on: list[int] = Field(default_factory=list)  # Step numbers
    serves_sub_question: Optional[str] = None  # e.g., "Q1"
    
    # Execution state
    status: CompositionStatus = CompositionStatus.PENDING
    output: Optional[dict] = None
    error: Optional[str] = None
    latency_ms: float = 0.0


class ExecutionPlan(BaseModel):
    """Complete execution plan for a composition."""
    composition_id: str
    session_id: Optional[str] = None
    
    steps: list[ExecutionStep] = Field(default_factory=list)
    
    # Parallel execution groups (steps that can run concurrently)
    parallelizable_groups: list[list[str]] = Field(default_factory=list)
    
    # Estimated performance
    estimated_latency_ms: float = 0.0
    estimated_from_cache: bool = False
    similar_composition_id: Optional[str] = None


# =============================================================================
# PHASE OUTPUT MODELS
# =============================================================================

class DecomposeOutput(BaseModel):
    """Output from Phase 1: Decompose."""
    sub_questions: list[SubQuestionInput]
    decomposition_method: str  # "pre_decomposed", "llm", "rule_based"
    latency_ms: float = 0.0


class PlanOutput(BaseModel):
    """Output from Phase 2: Plan."""
    execution_plan: ExecutionPlan
    used_episodic_memory: bool = False
    latency_ms: float = 0.0


class ExecuteOutput(BaseModel):
    """Output from Phase 3: Execute."""
    tool_outputs: dict[str, Any]
    steps_completed: int
    steps_failed: int
    latency_ms: float = 0.0


class SynthesizeOutput(BaseModel):
    """Output from Phase 4: Synthesize."""
    response: str
    sources_cited: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0


# Forward reference resolution
CompositionResult.model_rebuild()
