"""
E2I Tool Composer - Data Models
Version: 4.2
Purpose: Pydantic models for tool composition planning and execution
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ============================================================================
# ENUMS
# ============================================================================


class CompositionPhase(str, Enum):
    """Phases of tool composition"""

    DECOMPOSE = "decompose"
    PLAN = "plan"
    EXECUTE = "execute"
    SYNTHESIZE = "synthesize"


class ExecutionStatus(str, Enum):
    """Status of tool execution"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CompositionStatus(str, Enum):
    """Contract-compliant status for AgentDispatchResponse compatibility"""

    SUCCESS = "success"
    PARTIAL = "partial"  # Some tools succeeded, some failed
    FAILED = "failed"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"  # Blocked by dependency or circuit breaker


class DependencyType(str, Enum):
    """Type of dependency between execution steps"""

    SEQUENTIAL = "sequential"  # Must wait for predecessor
    PARALLEL = "parallel"  # Can run concurrently
    CONDITIONAL = "conditional"  # Run only if condition met


# ============================================================================
# PHASE 1: DECOMPOSITION MODELS
# ============================================================================


class SubQuestion(BaseModel):
    """A single decomposed sub-question"""

    id: str = Field(default_factory=lambda: f"sq_{uuid4().hex[:8]}")
    question: str = Field(..., description="The atomic sub-question")
    intent: str = Field(..., description="Inferred intent (CAUSAL, COMPARATIVE, etc.)")
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    depends_on: List[str] = Field(
        default_factory=list, description="IDs of prerequisite sub-questions"
    )

    model_config = ConfigDict(frozen=True)


class DecompositionResult(BaseModel):
    """Result of Phase 1: Decomposition"""

    original_query: str
    sub_questions: List[SubQuestion]
    decomposition_reasoning: str = Field(..., description="LLM's reasoning for decomposition")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def question_count(self) -> int:
        return len(self.sub_questions)

    def get_root_questions(self) -> List[SubQuestion]:
        """Get questions with no dependencies (can start execution)"""
        return [sq for sq in self.sub_questions if not sq.depends_on]


# ============================================================================
# PHASE 2: PLANNING MODELS
# ============================================================================


class ToolMapping(BaseModel):
    """Mapping of a sub-question to a tool"""

    sub_question_id: str
    tool_name: str
    source_agent: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class ExecutionStep(BaseModel):
    """A single step in the execution plan"""

    step_id: str = Field(default_factory=lambda: f"step_{uuid4().hex[:8]}")
    sub_question_id: str
    tool_name: str
    source_agent: str

    # Execution configuration
    input_mapping: Dict[str, Any] = Field(
        default_factory=dict,
        description="How to construct tool inputs (may reference prior step outputs)",
    )
    dependency_type: DependencyType = DependencyType.SEQUENTIAL
    depends_on_steps: List[str] = Field(default_factory=list)

    # Execution state (updated during Phase 3)
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(use_enum_values=True)


class ExecutionPlan(BaseModel):
    """Result of Phase 2: Planning"""

    plan_id: str = Field(default_factory=lambda: f"plan_{uuid4().hex[:8]}")
    decomposition: DecompositionResult
    steps: List[ExecutionStep]
    tool_mappings: List[ToolMapping]

    # Plan metadata
    estimated_duration_ms: int = Field(default=0)
    parallel_groups: List[List[str]] = Field(
        default_factory=list, description="Groups of step_ids that can execute in parallel"
    )
    planning_reasoning: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def get_step(self, step_id: str) -> Optional[ExecutionStep]:
        """Get step by ID"""
        return next((s for s in self.steps if s.step_id == step_id), None)

    def get_ready_steps(self) -> List[ExecutionStep]:
        """Get steps that are ready to execute (dependencies satisfied)"""
        completed_ids = {s.step_id for s in self.steps if s.status == ExecutionStatus.COMPLETED}
        return [
            step
            for step in self.steps
            if step.status == ExecutionStatus.PENDING
            and all(dep in completed_ids for dep in step.depends_on_steps)
        ]

    def get_execution_order(self) -> List[List[str]]:
        """Get steps grouped by execution wave (parallel groups)"""
        if self.parallel_groups:
            return self.parallel_groups
        # Fallback: sequential execution
        return [[step.step_id] for step in self.steps]


# ============================================================================
# PHASE 3: EXECUTION MODELS
# ============================================================================


class ToolInput(BaseModel):
    """Input to a tool execution"""

    tool_name: str
    parameters: Dict[str, Any]
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context from prior steps"
    )


class ToolOutput(BaseModel):
    """Output from a tool execution"""

    tool_name: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: int = 0

    @property
    def is_success(self) -> bool:
        return self.success and self.result is not None


class StepResult(BaseModel):
    """Result of executing a single step"""

    step_id: str
    sub_question_id: str
    tool_name: str

    # Execution details
    input: ToolInput
    output: ToolOutput
    status: ExecutionStatus

    # Timing
    started_at: datetime
    completed_at: datetime
    duration_ms: int = 0

    @field_validator("duration_ms", mode="before")
    @classmethod
    def calculate_duration(cls, v, info):
        if v == 0 and "started_at" in info.data and "completed_at" in info.data:
            delta = info.data["completed_at"] - info.data["started_at"]
            return int(delta.total_seconds() * 1000)
        return v


class ExecutionTrace(BaseModel):
    """Complete execution trace for Phase 3"""

    plan_id: str
    step_results: List[StepResult] = Field(default_factory=list)

    # Aggregate metrics
    total_duration_ms: int = 0
    tools_executed: int = 0
    tools_succeeded: int = 0
    tools_failed: int = 0
    parallel_executions: int = 0

    # Timing
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    def add_result(self, result: StepResult) -> None:
        """Add a step result and update metrics"""
        self.step_results.append(result)
        self.tools_executed += 1
        if result.output.is_success:
            self.tools_succeeded += 1
        else:
            self.tools_failed += 1
        self.total_duration_ms += result.duration_ms

    def get_result(self, step_id: str) -> Optional[StepResult]:
        """Get result for a specific step"""
        return next((r for r in self.step_results if r.step_id == step_id), None)

    def get_all_outputs(self) -> Dict[str, Any]:
        """Get all successful outputs keyed by step_id"""
        return {r.step_id: r.output.result for r in self.step_results if r.output.is_success}


# ============================================================================
# PHASE 4: SYNTHESIS MODELS
# ============================================================================


class SynthesisInput(BaseModel):
    """Input to the synthesis phase"""

    original_query: str
    decomposition: DecompositionResult
    execution_trace: ExecutionTrace

    def get_context_for_synthesis(self) -> Dict[str, Any]:
        """Prepare context for LLM synthesis"""
        return {
            "query": self.original_query,
            "sub_questions": [
                {"id": sq.id, "question": sq.question, "intent": sq.intent}
                for sq in self.decomposition.sub_questions
            ],
            "results": [
                {
                    "step_id": r.step_id,
                    "tool": r.tool_name,
                    "success": r.output.is_success,
                    "output": r.output.result if r.output.is_success else None,
                    "error": r.output.error if not r.output.is_success else None,
                }
                for r in self.execution_trace.step_results
            ],
        }


class ComposedResponse(BaseModel):
    """Final composed response from Tool Composer"""

    response_id: str = Field(default_factory=lambda: f"resp_{uuid4().hex[:8]}")

    # The synthesized answer
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)

    # Supporting evidence
    supporting_data: Dict[str, Any] = Field(
        default_factory=dict, description="Key data points from tool outputs"
    )
    citations: List[str] = Field(
        default_factory=list, description="References to source tools/steps"
    )

    # Warnings and caveats
    caveats: List[str] = Field(default_factory=list)
    failed_components: List[str] = Field(
        default_factory=list, description="Sub-questions that couldn't be fully answered"
    )

    # Metadata
    synthesis_reasoning: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# TOP-LEVEL COMPOSITION RESULT
# ============================================================================


class CompositionResult(BaseModel):
    """Complete result of a tool composition operation.

    Contract-compliant with AgentDispatchResponse (orchestrator-contracts.md).
    Maintains backwards compatibility with existing `success` and `error` fields.
    """

    composition_id: str = Field(default_factory=lambda: f"comp_{uuid4().hex[:8]}")

    # Contract-compliant identifiers (AgentDispatchResponse fields)
    session_id: Optional[str] = Field(
        default=None, description="Session identifier from dispatch request"
    )
    query_id: Optional[str] = Field(
        default=None, description="Query identifier from dispatch request"
    )
    agent_name: str = Field(
        default="tool_composer", description="Agent identifier for contract compliance"
    )

    # Original query
    query: str

    # Phase outputs
    decomposition: DecompositionResult
    plan: ExecutionPlan
    execution: ExecutionTrace
    response: ComposedResponse

    # Overall metrics
    total_duration_ms: int = 0
    phase_durations: Dict[str, int] = Field(default_factory=dict)

    # Contract-compliant status (AgentDispatchResponse.status enum)
    status: CompositionStatus = Field(
        default=CompositionStatus.SUCCESS,
        description="Contract-compliant status enum",
    )
    # Legacy status field for backwards compatibility
    success: bool = True
    # Contract-compliant errors list (AgentDispatchResponse.errors)
    errors: List[str] = Field(
        default_factory=list, description="List of errors for contract compliance"
    )
    # Legacy error field for backwards compatibility
    error: Optional[str] = None

    # Observability fields (Gap G5 - Opik integration)
    span_id: Optional[str] = Field(default=None, description="Opik span ID for tracing")
    trace_id: Optional[str] = Field(default=None, description="Opik trace ID for tracing")

    # DSPy/GEPA training signals (Gap G4)
    training_signals: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Training signals for DSPy/GEPA optimization",
    )

    # Timestamps
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    def to_summary(self) -> Dict[str, Any]:
        """Generate a summary for logging/observability"""
        return {
            "composition_id": self.composition_id,
            "session_id": self.session_id,
            "query_id": self.query_id,
            "agent_name": self.agent_name,
            "query_length": len(self.query),
            "sub_questions": self.decomposition.question_count,
            "tools_executed": self.execution.tools_executed,
            "tools_succeeded": self.execution.tools_succeeded,
            "total_duration_ms": self.total_duration_ms,
            "status": self.status.value,
            "success": self.success,
            "confidence": self.response.confidence,
            "span_id": self.span_id,
            "trace_id": self.trace_id,
        }

    def to_dispatch_response(self) -> Dict[str, Any]:
        """Convert to AgentDispatchResponse-compatible format.

        Maps CompositionResult fields to the contract-defined AgentDispatchResponse
        structure as specified in orchestrator-contracts.md.
        """
        return {
            "dispatch_id": self.composition_id,
            "session_id": self.session_id,
            "query_id": self.query_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "agent_result": {
                "query": self.query,
                "answer": self.response.answer,
                "confidence": self.response.confidence,
                "sub_questions": self.decomposition.question_count,
                "tools_executed": self.execution.tools_executed,
                "tools_succeeded": self.execution.tools_succeeded,
                "phase_durations": self.phase_durations,
            },
            "confidence": self.response.confidence,
            "latency_ms": self.total_duration_ms,
            "errors": self.errors,
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "training_signals": self.training_signals,
            "metadata": {
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "legacy_success": self.success,
                "legacy_error": self.error,
            },
        }
