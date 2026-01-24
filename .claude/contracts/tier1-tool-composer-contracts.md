# Tool Composer Contracts

**Purpose**: Define interfaces and expectations for the Tool Composer agent's 4-phase pipeline and integration with the Orchestrator.

**Version**: 1.0
**Last Updated**: 2026-01-24
**Owner**: E2I Development Team

---

## Overview

The Tool Composer (Tier 1) is a specialized orchestration agent that:
- Decomposes multi-faceted queries into atomic sub-questions
- Maps sub-questions to available tools
- Executes tools in dependency order with resilience patterns
- Synthesizes results into coherent responses

This contract ensures consistent, predictable behavior across the 4-phase pipeline.

---

## 4-Phase Pipeline Contract

### Phase Overview

| Phase | Name | Target Duration | Description |
|-------|------|-----------------|-------------|
| 1 | DECOMPOSE | 10s | Break query into 2-6 atomic sub-questions |
| 2 | PLAN | 15s | Map sub-questions to tools, create DAG |
| 3 | EXECUTE | 120s | Run tools with retries and circuit breaker |
| 4 | SYNTHESIZE | 30s | Combine results into coherent response |

**Total SLA**: 180 seconds (with parallelization)

---

## Phase 1: Decomposition Contract

### DecompositionResult Structure

```python
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID, uuid4

class SubQuestion(BaseModel):
    """
    Atomic sub-question extracted from multi-faceted query.
    """

    # === IDENTIFICATION ===
    id: str = Field(
        default_factory=lambda: f"sq_{uuid4().hex[:8]}",
        description="Unique sub-question identifier"
    )

    # === CONTENT ===
    question: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="The atomic sub-question text"
    )

    intent: Literal[
        "CAUSAL",
        "COMPARATIVE",
        "PREDICTIVE",
        "DESCRIPTIVE",
        "EXPERIMENTAL"
    ] = Field(..., description="Sub-question intent type")

    # === ENTITIES ===
    entities: List[str] = Field(
        default_factory=list,
        description="Extracted entities (brands, KPIs, regions, etc.)"
    )

    # === DEPENDENCIES ===
    depends_on: List[str] = Field(
        default_factory=list,
        description="IDs of sub-questions this depends on (DAG)"
    )

    # === METADATA ===
    priority: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Execution priority (1=highest)"
    )


class DecompositionResult(BaseModel):
    """
    Result from Phase 1: Query Decomposition.
    """

    # === IDENTIFICATION ===
    decomposition_id: str = Field(
        default_factory=lambda: f"dec_{uuid4().hex[:12]}",
        description="Unique decomposition identifier"
    )

    # === INPUT ===
    original_query: str = Field(..., description="Original user query")

    # === OUTPUT ===
    sub_questions: List[SubQuestion] = Field(
        ...,
        min_length=2,
        max_length=6,
        description="Decomposed sub-questions (2-6)"
    )

    decomposition_reasoning: str = Field(
        ...,
        description="Reasoning behind the decomposition"
    )

    # === COMPUTED ===
    @property
    def question_count(self) -> int:
        return len(self.sub_questions)

    @property
    def has_dependencies(self) -> bool:
        return any(sq.depends_on for sq in self.sub_questions)

    # === METADATA ===
    phase_duration_ms: int = Field(default=0, description="Decomposition duration")
    model_used: Optional[str] = Field(None, description="LLM model used")
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

### Decomposition Validation Rules

```python
def validate_decomposition(result: DecompositionResult) -> None:
    """
    Validate decomposition result before proceeding to planning.

    Raises:
        ValueError: If validation fails
    """
    # Must have 2-6 sub-questions
    if not (2 <= len(result.sub_questions) <= 6):
        raise ValueError(f"Expected 2-6 sub-questions, got {len(result.sub_questions)}")

    # No duplicate IDs
    ids = [sq.id for sq in result.sub_questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sub-question IDs detected")

    # Dependencies must reference valid IDs
    valid_ids = set(ids)
    for sq in result.sub_questions:
        for dep in sq.depends_on:
            if dep not in valid_ids:
                raise ValueError(f"Invalid dependency: {dep}")

    # No circular dependencies (DAG check)
    if has_cycles(result.sub_questions):
        raise ValueError("Circular dependencies detected")

    # Each sub-question must have meaningful content
    for sq in result.sub_questions:
        if len(sq.question.strip()) < 10:
            raise ValueError(f"Sub-question too short: {sq.id}")
```

---

## Phase 2: Planning Contract

### ExecutionPlan Structure

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from uuid import uuid4

class ToolMapping(BaseModel):
    """
    Mapping from sub-question to tool.
    """

    sub_question_id: str = Field(..., description="SubQuestion.id")
    tool_name: str = Field(..., description="Tool to execute")
    tool_category: str = Field(..., description="Tool category (CAUSAL, etc.)")
    agent_name: str = Field(..., description="Agent providing this tool")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in tool selection"
    )


class ExecutionStep(BaseModel):
    """
    Single step in the execution plan.
    """

    # === IDENTIFICATION ===
    step_id: str = Field(
        default_factory=lambda: f"step_{uuid4().hex[:8]}",
        description="Unique step identifier"
    )

    # === MAPPING ===
    sub_question_id: str = Field(..., description="SubQuestion.id")
    tool_name: str = Field(..., description="Tool to execute")

    # === INPUT/OUTPUT ===
    input_mapping: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input parameter mapping (may include $step_X.field refs)"
    )

    expected_output_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="Expected output schema for validation"
    )

    # === DEPENDENCIES ===
    depends_on_steps: List[str] = Field(
        default_factory=list,
        description="Step IDs this step depends on"
    )

    # === EXECUTION CONFIG ===
    timeout_seconds: int = Field(default=60, ge=1, le=300)
    max_retries: int = Field(default=2, ge=0, le=5)
    priority: int = Field(default=1, ge=1, le=10)


class ExecutionPlan(BaseModel):
    """
    Result from Phase 2: Planning.
    """

    # === IDENTIFICATION ===
    plan_id: str = Field(
        default_factory=lambda: f"plan_{uuid4().hex[:12]}",
        description="Unique plan identifier"
    )

    # === INPUT ===
    decomposition: DecompositionResult = Field(..., description="Source decomposition")

    # === OUTPUT ===
    tool_mappings: List[ToolMapping] = Field(
        ...,
        description="Sub-question to tool mappings"
    )

    steps: List[ExecutionStep] = Field(
        ...,
        description="Ordered execution steps"
    )

    parallel_groups: List[List[str]] = Field(
        default_factory=list,
        description="Groups of step_ids that can run in parallel"
    )

    planning_reasoning: str = Field(
        ...,
        description="Reasoning behind the plan"
    )

    # === COMPUTED ===
    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def tool_count(self) -> int:
        return len(set(s.tool_name for s in self.steps))

    # === METADATA ===
    phase_duration_ms: int = Field(default=0)
    model_used: Optional[str] = Field(None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

### Planning Validation Rules

```python
def validate_execution_plan(plan: ExecutionPlan) -> None:
    """
    Validate execution plan before proceeding to execution.

    Raises:
        ValueError: If validation fails
    """
    # Every sub-question must have a tool mapping
    sq_ids = {sq.id for sq in plan.decomposition.sub_questions}
    mapped_ids = {tm.sub_question_id for tm in plan.tool_mappings}

    if sq_ids != mapped_ids:
        missing = sq_ids - mapped_ids
        raise ValueError(f"Sub-questions without tool mapping: {missing}")

    # All tools must be registered
    from src.tool_registry.registry import ToolRegistry
    registry = ToolRegistry()

    for step in plan.steps:
        if not registry.has_tool(step.tool_name):
            raise ValueError(f"Unknown tool: {step.tool_name}")

    # Step dependencies must be valid
    step_ids = {s.step_id for s in plan.steps}
    for step in plan.steps:
        for dep in step.depends_on_steps:
            if dep not in step_ids:
                raise ValueError(f"Invalid step dependency: {dep}")

    # No cycles in step dependencies
    if has_step_cycles(plan.steps):
        raise ValueError("Circular step dependencies detected")

    # Parallel groups must contain valid step IDs
    for group in plan.parallel_groups:
        for step_id in group:
            if step_id not in step_ids:
                raise ValueError(f"Invalid step in parallel group: {step_id}")
```

---

## Phase 3: Execution Contract

### ExecutionTrace Structure

```python
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import uuid4

class StepResult(BaseModel):
    """
    Result from executing a single step.
    """

    # === IDENTIFICATION ===
    step_id: str = Field(..., description="ExecutionStep.step_id")
    tool_name: str = Field(..., description="Tool that was executed")

    # === STATUS ===
    status: Literal[
        "PENDING",
        "RUNNING",
        "COMPLETED",
        "FAILED",
        "TIMEOUT",
        "SKIPPED",
        "CANCELLED"
    ] = Field(..., description="Step execution status")

    success: bool = Field(..., description="Whether step succeeded")

    # === OUTPUT ===
    output: Optional[Dict[str, Any]] = Field(
        None,
        description="Tool output (if successful)"
    )

    error: Optional[str] = Field(
        None,
        description="Error message (if failed)"
    )

    # === EXECUTION DETAILS ===
    duration_ms: int = Field(default=0, description="Execution duration")
    retry_count: int = Field(default=0, description="Number of retries attempted")

    # === CIRCUIT BREAKER ===
    circuit_state: Optional[Literal["CLOSED", "OPEN", "HALF_OPEN"]] = Field(
        None,
        description="Circuit breaker state after execution"
    )

    # === METADATA ===
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)


class ExecutionTrace(BaseModel):
    """
    Result from Phase 3: Execution.
    """

    # === IDENTIFICATION ===
    trace_id: str = Field(
        default_factory=lambda: f"trace_{uuid4().hex[:12]}",
        description="Unique trace identifier"
    )

    plan_id: str = Field(..., description="Source ExecutionPlan.plan_id")

    # === RESULTS ===
    step_results: List[StepResult] = Field(
        default_factory=list,
        description="Results from each step"
    )

    # === AGGREGATES ===
    @property
    def tools_executed(self) -> int:
        return len([r for r in self.step_results if r.status != "SKIPPED"])

    @property
    def tools_succeeded(self) -> int:
        return len([r for r in self.step_results if r.success])

    @property
    def tools_failed(self) -> int:
        return len([r for r in self.step_results if not r.success and r.status != "SKIPPED"])

    @property
    def success_rate(self) -> float:
        executed = self.tools_executed
        return self.tools_succeeded / executed if executed > 0 else 0.0

    # === STATUS ===
    overall_status: Literal["SUCCESS", "PARTIAL", "FAILED"] = Field(
        default="FAILED",
        description="Overall execution status"
    )

    failure_summary: Optional[str] = Field(
        None,
        description="Summary of failures (if any)"
    )

    # === METADATA ===
    total_duration_ms: int = Field(default=0)
    total_retries: int = Field(default=0)
    circuit_breaker_trips: int = Field(default=0)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
```

### Resilience Patterns Contract

```python
class ExponentialBackoffConfig(BaseModel):
    """
    Configuration for exponential backoff retry strategy.
    """

    base_delay_seconds: float = Field(default=1.0, ge=0.1, le=10.0)
    max_delay_seconds: float = Field(default=30.0, ge=1.0, le=120.0)
    factor: float = Field(default=2.0, ge=1.0, le=4.0)
    jitter: float = Field(default=0.1, ge=0.0, le=0.5)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt (0-indexed)."""
        import random
        delay = min(self.max_delay_seconds, self.base_delay_seconds * (self.factor ** attempt))
        if self.jitter > 0:
            jitter_range = delay * self.jitter
            delay += random.uniform(-jitter_range, jitter_range)
        return max(0, delay)


class CircuitBreakerConfig(BaseModel):
    """
    Configuration for circuit breaker pattern.
    """

    failure_threshold: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Failures before opening circuit"
    )

    reset_timeout_seconds: float = Field(
        default=60.0,
        ge=10.0,
        le=300.0,
        description="Seconds before attempting half-open"
    )

    half_open_max_calls: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Max calls allowed in half-open state"
    )


class CircuitBreakerState(BaseModel):
    """
    Current state of a circuit breaker.
    """

    tool_name: str = Field(..., description="Tool this breaker protects")
    state: Literal["CLOSED", "OPEN", "HALF_OPEN"] = Field(default="CLOSED")
    failure_count: int = Field(default=0)
    success_count: int = Field(default=0)
    last_failure_at: Optional[datetime] = Field(None)
    opened_at: Optional[datetime] = Field(None)

    def should_allow_request(self, config: CircuitBreakerConfig) -> bool:
        """Determine if request should be allowed."""
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            # Check if reset timeout has passed
            if self.opened_at:
                elapsed = (datetime.utcnow() - self.opened_at).total_seconds()
                if elapsed >= config.reset_timeout_seconds:
                    # Transition to half-open
                    self.state = "HALF_OPEN"
                    return True
            return False

        # HALF_OPEN: allow limited calls
        return self.success_count < config.half_open_max_calls
```

---

## Phase 4: Synthesis Contract

### ComposedResponse Structure

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class ComposedResponse(BaseModel):
    """
    Result from Phase 4: Synthesis.
    """

    # === PRIMARY OUTPUT ===
    answer: str = Field(
        ...,
        min_length=50,
        description="Natural language response to original query"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the response"
    )

    # === SUPPORTING DATA ===
    supporting_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key data points supporting the answer"
    )

    citations: List[str] = Field(
        default_factory=list,
        description="Step IDs that contributed to the answer"
    )

    # === CAVEATS ===
    caveats: List[str] = Field(
        default_factory=list,
        description="Limitations or caveats about the answer"
    )

    failed_components: List[str] = Field(
        default_factory=list,
        description="Sub-questions that could not be answered"
    )

    # === REASONING ===
    reasoning: Optional[str] = Field(
        None,
        description="Synthesis reasoning (how results were combined)"
    )

    # === METADATA ===
    phase_duration_ms: int = Field(default=0)
    model_used: Optional[str] = Field(None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

---

## CompositionResult (Top-Level Output)

```python
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import uuid4

class CompositionStatus(str, Enum):
    """Overall composition status."""
    PENDING = "PENDING"
    DECOMPOSING = "DECOMPOSING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    SYNTHESIZING = "SYNTHESIZING"
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


class CompositionResult(BaseModel):
    """
    Top-level result from Tool Composer execution.

    Combines outputs from all 4 phases.
    """

    # === IDENTIFICATION ===
    composition_id: str = Field(
        default_factory=lambda: f"comp_{uuid4().hex[:12]}",
        description="Unique composition identifier"
    )

    # === INPUT ===
    query: str = Field(..., description="Original user query")

    # === PHASE OUTPUTS ===
    decomposition: Optional[DecompositionResult] = Field(
        None,
        description="Phase 1 output"
    )

    plan: Optional[ExecutionPlan] = Field(
        None,
        description="Phase 2 output"
    )

    execution: Optional[ExecutionTrace] = Field(
        None,
        description="Phase 3 output"
    )

    response: Optional[ComposedResponse] = Field(
        None,
        description="Phase 4 output (final response)"
    )

    # === STATUS ===
    status: CompositionStatus = Field(
        default=CompositionStatus.PENDING,
        description="Current composition status"
    )

    success: bool = Field(
        default=False,
        description="Whether composition succeeded"
    )

    # === ERRORS ===
    errors: List[str] = Field(
        default_factory=list,
        description="Errors encountered during composition"
    )

    error: Optional[str] = Field(
        None,
        description="Primary error message (if failed)"
    )

    # === TIMING ===
    phase_durations: Dict[str, int] = Field(
        default_factory=dict,
        description="Duration per phase in ms"
    )

    total_duration_ms: Optional[int] = Field(
        None,
        description="Total composition duration"
    )

    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
```

---

## Integration with Orchestrator

### Orchestrator → Tool Composer Dispatch

```python
class ToolComposerDispatch(BaseModel):
    """
    Dispatch request from Orchestrator to Tool Composer.

    Sent when query is classified as MULTI_FACETED.
    """

    # === IDENTIFICATION ===
    dispatch_id: str = Field(..., description="Unique dispatch ID")
    session_id: str = Field(..., description="Session identifier")
    query_id: str = Field(..., description="Query identifier")

    # === INPUT ===
    query: str = Field(..., description="Multi-faceted query")

    extracted_entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Entities extracted by NLP layer"
    )

    user_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="User context (filters, permissions)"
    )

    # === EXECUTION CONFIG ===
    timeout_seconds: int = Field(
        default=180,
        description="Maximum execution time"
    )

    parallel_limit: int = Field(
        default=3,
        description="Maximum parallel tool executions"
    )
```

### Tool Composer → Orchestrator Response

```python
class ToolComposerResponse(BaseModel):
    """
    Response from Tool Composer to Orchestrator.
    """

    # === STATUS ===
    success: bool = Field(..., description="Whether composition succeeded")

    # === OUTPUT ===
    response: str = Field(..., description="Natural language response")
    confidence: float = Field(..., ge=0.0, le=1.0)

    supporting_data: Dict[str, Any] = Field(default_factory=dict)
    citations: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)

    # === METADATA ===
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Includes composition_id, sub_questions, tools_executed, etc."
    )

    # === ERRORS ===
    error: Optional[str] = Field(None, description="Error message if failed")
```

---

## Tool Registry Contract

### Tool Registration

```python
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field
from enum import Enum

class ToolCategory(str, Enum):
    """Categories of tools available to Tool Composer."""
    CAUSAL = "CAUSAL"           # DoWhy/EconML causal inference
    SEGMENTATION = "SEGMENTATION"   # Heterogeneous analysis
    GAP = "GAP"                 # Opportunity detection
    EXPERIMENT = "EXPERIMENT"   # A/B test design
    PREDICTION = "PREDICTION"   # ML predictions
    MONITORING = "MONITORING"   # Drift detection
    EXPLANATION = "EXPLANATION" # Natural language explanation


class ToolSchema(BaseModel):
    """
    Schema for a registered tool.
    """

    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="What the tool does")
    category: ToolCategory = Field(..., description="Tool category")
    agent_name: str = Field(..., description="Agent providing this tool")

    # === INPUT/OUTPUT ===
    input_schema: Dict[str, Any] = Field(
        ...,
        description="JSON Schema for input parameters"
    )

    output_schema: Dict[str, Any] = Field(
        ...,
        description="JSON Schema for output"
    )

    # === EXECUTION ===
    supports_chaining: bool = Field(
        default=True,
        description="Can receive output from other tools"
    )

    timeout_seconds: int = Field(default=60, ge=1, le=300)

    # === EXAMPLES ===
    examples: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Example invocations"
    )


class ToolRegistryInterface:
    """
    Interface for the Tool Registry.
    """

    def register_tool(
        self,
        schema: ToolSchema,
        handler: Callable
    ) -> None:
        """Register a tool with its handler."""
        pass

    def get_tool(self, name: str) -> Optional[ToolSchema]:
        """Get tool schema by name."""
        pass

    def find_tools_by_capability(
        self,
        intent: str,
        entities: List[str]
    ) -> List[ToolSchema]:
        """Find tools matching intent and entities."""
        pass

    def get_tools_by_category(
        self,
        category: ToolCategory
    ) -> List[ToolSchema]:
        """Get all tools in a category."""
        pass

    def has_tool(self, name: str) -> bool:
        """Check if tool exists."""
        pass

    @property
    def tool_count(self) -> int:
        """Total number of registered tools."""
        pass
```

---

## Memory Integration Contract

### Memory Contribution

```python
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class MemoryContributionResult(BaseModel):
    """
    Result from contributing composition to memory.
    """

    episodic_stored: bool = Field(
        default=False,
        description="Whether stored in episodic memory"
    )

    procedural_stored: bool = Field(
        default=False,
        description="Whether stored in procedural memory"
    )

    working_cached: bool = Field(
        default=False,
        description="Whether cached in working memory"
    )

    errors: List[str] = Field(default_factory=list)


class MemoryHooksInterface:
    """
    Interface for memory system integration.
    """

    async def store_episodic(
        self,
        composition_id: str,
        result: Dict[str, Any],
        session_id: Optional[str] = None,
        brand: Optional[str] = None,
        region: Optional[str] = None
    ) -> bool:
        """Store composition in episodic memory for future retrieval."""
        pass

    async def store_procedural(
        self,
        composition_id: str,
        pattern: Dict[str, Any],
        success_rate: float
    ) -> bool:
        """Store successful pattern in procedural memory."""
        pass

    async def cache_working(
        self,
        composition_id: str,
        result: Dict[str, Any],
        ttl_seconds: int = 3600
    ) -> bool:
        """Cache result in working memory (Redis)."""
        pass

    async def retrieve_similar_compositions(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve similar past compositions for planning hints."""
        pass
```

---

## Testing Requirements

### Unit Test Coverage

| Component | Required Coverage | Test File |
|-----------|------------------|-----------|
| QueryDecomposer | 90% | `test_decomposer.py` |
| ToolPlanner | 90% | `test_planner.py` |
| PlanExecutor | 95% | `test_executor.py` |
| ResponseSynthesizer | 90% | `test_synthesizer.py` |
| ToolComposer | 85% | `test_composer.py` |
| Models | 100% | `test_models.py` |
| Cache | 85% | `test_cache.py` |

### Integration Test Requirements

1. **End-to-end pipeline**: Query → Decomposition → Planning → Execution → Synthesis
2. **Parallel execution**: Multiple tools executing simultaneously
3. **Circuit breaker**: Failure threshold triggering circuit open
4. **Retry logic**: Transient failures with exponential backoff
5. **Memory integration**: Successful patterns stored and retrieved

---

## Compliance Checklist

Before deploying Tool Composer changes:

- [ ] DecompositionResult validates (2-6 sub-questions, no cycles)
- [ ] ExecutionPlan validates (all tools registered, no step cycles)
- [ ] ExecutionTrace captures all step results
- [ ] ComposedResponse includes citations and caveats
- [ ] CompositionResult has correct status
- [ ] Resilience patterns working (backoff, circuit breaker)
- [ ] Memory contribution successful
- [ ] Opik tracing enabled
- [ ] Audit chain entries recorded
- [ ] Unit tests passing (85%+ coverage)
- [ ] Integration tests passing

---

**End of Tool Composer Contracts**
