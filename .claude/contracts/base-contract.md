# Base Agent Contracts

**Purpose**: Define foundational interfaces and expectations for all agents in the E2I Causal Analytics system to ensure consistent behavior and integration.

**Version**: 1.0
**Last Updated**: 2025-12-18
**Owner**: E2I Development Team

---

## Overview

This contract defines the base structures that ALL agents must implement. These contracts ensure:
- Consistent state management across all 18 agents
- Standardized configuration patterns
- Uniform error handling
- Predictable agent lifecycle
- Type-safe inter-agent communication

---

## Base Structures

### AgentState Contract

All agents use LangGraph state management with a common base structure.

```python
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
from datetime import datetime
import operator

class BaseAgentState(TypedDict):
    """
    Base state structure that ALL agent states must extend.

    This provides common fields for tracking, error handling,
    and inter-agent communication.
    """

    # === IDENTIFICATION ===
    session_id: str                           # Unique session identifier
    query_id: str                             # Unique query identifier
    agent_name: str                           # Name of current agent

    # === INPUT ===
    query: str                                # Original user query
    parsed_query: Optional[Dict[str, Any]]    # Parsed query from NLP layer

    # === CONTEXT ===
    rag_context: Optional[Dict[str, Any]]     # Retrieved context from RAG
    memory_context: Optional[Dict[str, Any]]  # Memory context (working, episodic, etc.)
    upstream_results: Optional[List[Dict]]    # Results from upstream agents

    # === OUTPUTS ===
    agent_result: Optional[Dict[str, Any]]    # Primary agent output
    analysis_summary: Optional[str]           # Human-readable summary
    confidence: Optional[float]               # Confidence score (0.0-1.0)

    # === METADATA ===
    started_at: Optional[datetime]            # Agent start timestamp
    completed_at: Optional[datetime]          # Agent completion timestamp
    execution_time_ms: Optional[int]          # Execution duration
    tokens_used: Optional[int]                # LLM tokens used (if applicable)
    model_used: Optional[str]                 # Model identifier (if LLM used)

    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]  # Error accumulator
    warnings: Annotated[List[str], operator.add]           # Warning accumulator
    status: Literal["pending", "in_progress", "completed", "failed", "cancelled"]

    # === ROUTING ===
    next_agent: Optional[str]                 # Next agent to route to (if any)
    requires_human: bool                      # Flag for human-in-the-loop

    # === HANDOFF ===
    handoff: Optional[Dict[str, Any]]         # Structured handoff to next agent
```

**Validation Rules**:
```python
# Required fields (must be present and non-null)
REQUIRED_FIELDS = ["session_id", "query_id", "agent_name", "query", "status"]

# Field constraints
CONSTRAINTS = {
    "session_id": {
        "type": str,
        "format": r"^sess_[a-z0-9]{16}$"
    },
    "query_id": {
        "type": str,
        "format": r"^qry_[a-z0-9]{16}$"
    },
    "agent_name": {
        "type": str,
        "allowed_values": [
            # Tier 0
            "scope_definer", "data_preparer", "model_selector",
            "model_trainer", "feature_analyzer", "model_deployer",
            "observability_connector",
            # Tier 1
            "orchestrator",
            # Tier 2
            "causal_impact", "gap_analyzer", "heterogeneous_optimizer",
            # Tier 3
            "experiment_designer", "drift_monitor", "health_score",
            # Tier 4
            "prediction_synthesizer", "resource_optimizer",
            # Tier 5
            "explainer", "feedback_learner"
        ]
    },
    "confidence": {
        "type": float,
        "min_value": 0.0,
        "max_value": 1.0
    },
    "status": {
        "type": str,
        "allowed_values": ["pending", "in_progress", "completed", "failed", "cancelled"]
    }
}
```

---

### AgentConfig Contract

All agents must be initialized with a configuration object.

```python
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, validator

class AgentConfig(BaseModel):
    """
    Base configuration structure for all agents.

    Loaded from config/agent_config.yaml at runtime.
    """

    # === IDENTITY ===
    agent_name: str = Field(..., description="Unique agent identifier")
    tier: int = Field(..., ge=0, le=5, description="Agent tier (0-5)")
    tier_name: str = Field(..., description="Tier name (e.g., 'ml_foundation')")

    # === CLASSIFICATION ===
    agent_type: Literal["standard", "hybrid", "deep"] = Field(
        default="standard",
        description="Agent type: standard (no LLM), hybrid (compute + LLM), deep (extended reasoning)"
    )

    # === PERFORMANCE ===
    sla_seconds: Optional[float] = Field(
        default=None,
        description="Service level agreement in seconds (None = no hard limit)"
    )
    timeout_seconds: Optional[float] = Field(
        default=None,
        description="Hard timeout (defaults to 2x SLA)"
    )

    # === MEMORY ===
    memory_types: List[Literal["working", "episodic", "procedural", "semantic"]] = Field(
        default=["working"],
        description="Memory types this agent uses"
    )

    # === TOOLS ===
    tools: List[str] = Field(
        default_factory=list,
        description="MLOps tools this agent integrates with"
    )

    # === LLM CONFIGURATION (for hybrid/deep agents) ===
    primary_model: Optional[str] = Field(
        default="claude-sonnet-4-20250514",
        description="Primary LLM model"
    )
    fallback_models: Optional[List[str]] = Field(
        default_factory=lambda: ["claude-haiku-4-20250414"],
        description="Fallback models in priority order"
    )
    max_tokens: Optional[int] = Field(
        default=4096,
        description="Maximum tokens for LLM calls"
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="LLM temperature"
    )

    # === RETRY/FALLBACK ===
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts on failure"
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.0,
        description="Initial retry delay (exponential backoff)"
    )

    # === DATABASE ===
    database_url: Optional[str] = Field(
        default=None,
        description="Database connection string (defaults to environment variable)"
    )

    # === OBSERVABILITY ===
    enable_tracing: bool = Field(
        default=True,
        description="Enable Opik tracing"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )

    @validator("timeout_seconds", always=True)
    def set_timeout(cls, v, values):
        """Set timeout to 2x SLA if not specified."""
        if v is None and "sla_seconds" in values and values["sla_seconds"] is not None:
            return values["sla_seconds"] * 2
        return v

    class Config:
        extra = "forbid"  # Raise error on unknown fields
        validate_assignment = True
```

---

### BaseAgent Interface Contract

All agents must implement this interface.

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import asyncio
import time
from datetime import datetime

class BaseAgent(ABC):
    """
    Abstract base class that all E2I agents must extend.

    Provides common functionality:
    - State initialization
    - Error handling
    - Timeout management
    - Retry logic
    - Memory integration
    - Observability
    """

    # === CLASS ATTRIBUTES (must be set by subclass) ===
    tier: int                    # Agent tier (0-5)
    tier_name: str              # Tier name
    agent_type: str             # "standard", "hybrid", "deep"
    sla_seconds: Optional[float] # SLA target

    def __init__(self, config: AgentConfig):
        """
        Initialize agent with configuration.

        Args:
            config: AgentConfig instance
        """
        self.config = config
        self.agent_name = config.agent_name

        # Validate class attributes match config
        assert self.agent_name == config.agent_name, "Agent name mismatch"
        assert self.tier == config.tier, f"Tier mismatch: {self.tier} != {config.tier}"

        # Initialize memory backends
        self._init_memory()

        # Initialize observability
        self._init_observability()

    @abstractmethod
    async def execute(self, state: BaseAgentState) -> BaseAgentState:
        """
        Main execution method. Must be implemented by all agents.

        Args:
            state: Current agent state

        Returns:
            Updated agent state

        Raises:
            AgentError: On execution failure
            TimeoutError: On timeout
        """
        pass

    async def run(self, state: BaseAgentState) -> BaseAgentState:
        """
        Run the agent with timeout, retry, and error handling.

        This is the public interface. It wraps execute() with:
        - Timeout management
        - Retry logic
        - Error handling
        - Observability

        Args:
            state: Input state

        Returns:
            Output state
        """
        start_time = time.time()

        # Initialize state
        state = self._init_state(state)

        # Create span for observability
        with self._create_span() as span:
            try:
                # Execute with timeout
                if self.config.timeout_seconds:
                    result = await asyncio.wait_for(
                        self.execute(state),
                        timeout=self.config.timeout_seconds
                    )
                else:
                    result = await self.execute(state)

                # Finalize state
                result = self._finalize_state(result, start_time)

                # Update observability
                span.set_success()

                return result

            except asyncio.TimeoutError:
                error_state = self._handle_timeout(state, start_time)
                span.set_error("Timeout")
                return error_state

            except Exception as e:
                error_state = self._handle_error(state, e, start_time)
                span.set_error(str(e))
                return error_state

    def _init_state(self, state: BaseAgentState) -> BaseAgentState:
        """Initialize state with agent metadata."""
        return {
            **state,
            "agent_name": self.agent_name,
            "status": "in_progress",
            "started_at": datetime.utcnow(),
            "errors": state.get("errors", []),
            "warnings": state.get("warnings", [])
        }

    def _finalize_state(self, state: BaseAgentState, start_time: float) -> BaseAgentState:
        """Finalize state with timing and status."""
        execution_time = int((time.time() - start_time) * 1000)

        return {
            **state,
            "completed_at": datetime.utcnow(),
            "execution_time_ms": execution_time,
            "status": "completed" if state.get("status") != "failed" else "failed"
        }

    def _handle_timeout(self, state: BaseAgentState, start_time: float) -> BaseAgentState:
        """Handle timeout error."""
        return {
            **state,
            "status": "failed",
            "errors": state.get("errors", []) + [{
                "agent": self.agent_name,
                "error_type": "timeout",
                "message": f"Agent exceeded timeout of {self.config.timeout_seconds}s",
                "timestamp": datetime.utcnow().isoformat()
            }],
            "completed_at": datetime.utcnow(),
            "execution_time_ms": int((time.time() - start_time) * 1000)
        }

    def _handle_error(self, state: BaseAgentState, error: Exception, start_time: float) -> BaseAgentState:
        """Handle execution error."""
        return {
            **state,
            "status": "failed",
            "errors": state.get("errors", []) + [{
                "agent": self.agent_name,
                "error_type": type(error).__name__,
                "message": str(error),
                "timestamp": datetime.utcnow().isoformat()
            }],
            "completed_at": datetime.utcnow(),
            "execution_time_ms": int((time.time() - start_time) * 1000)
        }

    def _init_memory(self):
        """Initialize memory backends based on config."""
        # To be implemented based on memory_types in config
        pass

    def _init_observability(self):
        """Initialize observability tracing."""
        # To be implemented with Opik integration
        pass

    def _create_span(self):
        """Create observability span for this execution."""
        # To be implemented with Opik integration
        pass
```

---

### AgentResult Contract

Standard result structure returned by all agents.

```python
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class AgentResult(BaseModel):
    """
    Standard result structure from agent execution.

    This is the 'agent_result' field in AgentState.
    """

    # === IDENTIFICATION ===
    agent_name: str = Field(..., description="Agent that produced this result")
    result_type: str = Field(..., description="Type of result (agent-specific)")

    # === PRIMARY OUTPUT ===
    data: Dict[str, Any] = Field(..., description="Primary result data")

    # === METADATA ===
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    method_used: Optional[str] = Field(None, description="Method/algorithm used")

    # === QUALITY INDICATORS ===
    assumptions: List[str] = Field(
        default_factory=list,
        description="Assumptions made during analysis"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Known limitations of results"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings about result interpretation"
    )

    # === RECOMMENDATIONS ===
    next_steps: Optional[List[str]] = Field(
        None,
        description="Suggested next steps"
    )
    requires_validation: bool = Field(
        default=False,
        description="Whether results require human validation"
    )

    # === SUPPORTING DATA ===
    supporting_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional supporting data"
    )
    visualizations: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Suggested visualizations"
    )

    # === PROVENANCE ===
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources used"
    )
    computation_time_ms: Optional[int] = Field(
        None,
        description="Computation time"
    )

    class Config:
        extra = "allow"  # Allow agent-specific fields
```

---

## Error Handling Contracts

### Error Structure

```python
from typing import Any, Dict, Optional
from pydantic import BaseModel
from datetime import datetime

class AgentError(BaseModel):
    """Standard error structure."""

    agent: str                              # Agent that raised error
    error_type: str                         # Error type/class
    message: str                            # Human-readable message
    timestamp: str                          # ISO timestamp
    severity: str = "error"                 # "error" | "warning"
    recoverable: bool = False               # Whether error is recoverable
    context: Optional[Dict[str, Any]] = None # Additional context
    stack_trace: Optional[str] = None       # Stack trace (dev only)
```

### Error Types

```python
class AgentErrorType:
    """Standard error types."""

    # Configuration errors
    INVALID_CONFIG = "invalid_config"
    MISSING_DEPENDENCY = "missing_dependency"

    # Execution errors
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    INVALID_INPUT = "invalid_input"
    INVALID_STATE = "invalid_state"

    # Data errors
    DATA_NOT_FOUND = "data_not_found"
    DATA_QUALITY_FAILED = "data_quality_failed"
    DATA_LEAKAGE_DETECTED = "data_leakage_detected"

    # Model errors
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_LOAD_FAILED = "model_load_failed"
    INFERENCE_FAILED = "inference_failed"

    # Integration errors
    DATABASE_ERROR = "database_error"
    API_ERROR = "api_error"
    LLM_ERROR = "llm_error"
    TOOL_ERROR = "tool_error"

    # Logic errors
    ASSERTION_FAILED = "assertion_failed"
    VALIDATION_FAILED = "validation_failed"
    COMPUTATION_ERROR = "computation_error"
```

### Retry Strategy

```python
from typing import List, Callable
import asyncio

class RetryConfig(BaseModel):
    """Retry configuration."""

    max_retries: int = 3
    base_delay: float = 1.0          # Base delay in seconds
    max_delay: float = 30.0          # Max delay in seconds
    exponential_base: float = 2.0    # Exponential backoff base
    retryable_errors: List[str] = [
        AgentErrorType.TIMEOUT,
        AgentErrorType.RESOURCE_EXHAUSTED,
        AgentErrorType.API_ERROR,
        AgentErrorType.DATABASE_ERROR
    ]

async def retry_with_backoff(
    func: Callable,
    config: RetryConfig,
    *args,
    **kwargs
) -> Any:
    """
    Execute function with exponential backoff retry.

    Args:
        func: Async function to execute
        config: Retry configuration
        *args, **kwargs: Arguments to pass to func

    Returns:
        Function result

    Raises:
        Last exception if all retries exhausted
    """
    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            # Check if error is retryable
            error_type = type(e).__name__
            if error_type not in config.retryable_errors:
                raise

            # Calculate delay
            if attempt < config.max_retries:
                delay = min(
                    config.base_delay * (config.exponential_base ** attempt),
                    config.max_delay
                )
                await asyncio.sleep(delay)

    # All retries exhausted
    raise last_exception
```

### Fallback Strategy

```python
from typing import List, Any

class FallbackChain:
    """
    Fallback chain for graceful degradation.

    Example:
        chain = FallbackChain([
            "claude-sonnet-4-20250514",
            "claude-haiku-4-20250414",
            "template_response"
        ])
    """

    def __init__(self, options: List[str]):
        self.options = options
        self.current_index = 0

    def get_next(self) -> Optional[str]:
        """Get next fallback option."""
        if self.current_index < len(self.options):
            option = self.options[self.current_index]
            self.current_index += 1
            return option
        return None

    def reset(self):
        """Reset to first option."""
        self.current_index = 0
```

---

## State Transition Rules

### Valid State Transitions

```python
VALID_TRANSITIONS = {
    "pending": ["in_progress", "cancelled"],
    "in_progress": ["completed", "failed", "cancelled"],
    "completed": [],  # Terminal state
    "failed": [],     # Terminal state
    "cancelled": []   # Terminal state
}

def validate_state_transition(
    current_status: str,
    new_status: str
) -> bool:
    """Validate state transition is allowed."""
    return new_status in VALID_TRANSITIONS.get(current_status, [])
```

### Status Semantics

| Status | Meaning | Terminal | Can Retry |
|--------|---------|----------|-----------|
| `pending` | Agent queued, not started | No | N/A |
| `in_progress` | Agent currently executing | No | N/A |
| `completed` | Agent succeeded | Yes | No |
| `failed` | Agent failed (retriable or not) | Yes | Maybe |
| `cancelled` | Agent cancelled by user/system | Yes | No |

---

## Memory Integration Contract

All agents that use memory must implement these methods.

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class MemoryIntegration(ABC):
    """
    Memory integration interface.

    Agents declare which memory types they use in config.
    """

    @abstractmethod
    async def load_working_memory(self, session_id: str) -> Dict[str, Any]:
        """Load working memory for session."""
        pass

    @abstractmethod
    async def save_working_memory(self, session_id: str, data: Dict[str, Any]):
        """Save working memory for session."""
        pass

    @abstractmethod
    async def load_episodic_memory(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Load relevant episodic memories."""
        pass

    @abstractmethod
    async def save_episodic_memory(self, event: Dict[str, Any]):
        """Save event to episodic memory."""
        pass

    @abstractmethod
    async def load_procedural_memory(
        self,
        task_type: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Load relevant procedural memories (patterns)."""
        pass

    @abstractmethod
    async def save_procedural_memory(self, pattern: Dict[str, Any]):
        """Save successful pattern to procedural memory."""
        pass

    @abstractmethod
    async def query_semantic_memory(
        self,
        query: str,
        relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query semantic memory graph."""
        pass

    @abstractmethod
    async def update_semantic_memory(self, relationships: List[Dict[str, Any]]):
        """Update semantic memory graph."""
        pass
```

---

## Validation Rules

### Pre-Execution Validation

All agents must validate input state before execution.

```python
def validate_input_state(state: BaseAgentState) -> None:
    """
    Validate input state meets requirements.

    Raises:
        ValueError: If validation fails
    """
    # Required fields
    for field in REQUIRED_FIELDS:
        if field not in state or state[field] is None:
            raise ValueError(f"Missing required field: {field}")

    # Field constraints
    for field, constraints in CONSTRAINTS.items():
        if field in state and state[field] is not None:
            value = state[field]

            # Type check
            if "type" in constraints and not isinstance(value, constraints["type"]):
                raise ValueError(f"Invalid type for {field}: expected {constraints['type']}")

            # Format check
            if "format" in constraints:
                import re
                if not re.match(constraints["format"], str(value)):
                    raise ValueError(f"Invalid format for {field}")

            # Range check
            if "min_value" in constraints and value < constraints["min_value"]:
                raise ValueError(f"{field} below minimum: {value} < {constraints['min_value']}")

            if "max_value" in constraints and value > constraints["max_value"]:
                raise ValueError(f"{field} above maximum: {value} > {constraints['max_value']}")

            # Allowed values check
            if "allowed_values" in constraints and value not in constraints["allowed_values"]:
                raise ValueError(f"Invalid value for {field}: {value}")
```

### Post-Execution Validation

All agents must validate output state after execution.

```python
def validate_output_state(state: BaseAgentState) -> None:
    """
    Validate output state is valid.

    Raises:
        ValueError: If validation fails
    """
    # Status must be terminal or valid intermediate
    if state["status"] not in ["completed", "failed", "in_progress"]:
        raise ValueError(f"Invalid output status: {state['status']}")

    # If completed, must have result
    if state["status"] == "completed" and not state.get("agent_result"):
        raise ValueError("Completed status requires agent_result")

    # If failed, must have errors
    if state["status"] == "failed" and not state.get("errors"):
        raise ValueError("Failed status requires errors")

    # Timing must be present
    if not state.get("started_at") or not state.get("completed_at"):
        raise ValueError("Missing timing information")

    # Execution time must be positive
    if state.get("execution_time_ms", 0) < 0:
        raise ValueError("Execution time cannot be negative")
```

---

## Change Management

### Breaking Changes

Changes that require code updates in ALL agents:
1. Adding required fields to `BaseAgentState`
2. Changing field types in `BaseAgentState`
3. Removing fields from `BaseAgentState`
4. Changing `BaseAgent` interface methods
5. Modifying error handling behavior

### Non-Breaking Changes

Changes that don't require agent updates:
1. Adding optional fields to `BaseAgentState`
2. Adding new error types to `AgentErrorType`
3. Relaxing validation rules
4. Adding new memory integration methods
5. Updating documentation

### Deprecation Process

1. **Announce**: 30-day notice with deprecation warning
2. **Mark**: Add `@deprecated` decorator to deprecated items
3. **Migrate**: Provide migration guide and examples
4. **Remove**: Remove after grace period

### Versioning

This contract follows semantic versioning:
- **Major**: Breaking changes
- **Minor**: Non-breaking additions
- **Patch**: Bug fixes and clarifications

Current version: **1.0.0**

---

## Testing Requirements

All agents must implement tests that verify:

1. **State Validation**:
   - Input state validation works correctly
   - Output state validation works correctly
   - Invalid states are rejected

2. **Error Handling**:
   - All error types can be raised and handled
   - Retry logic works as expected
   - Fallback logic works as expected

3. **Timeout**:
   - Timeouts are enforced
   - Timeout errors are properly handled
   - State is properly finalized on timeout

4. **Memory Integration**:
   - Memory operations work correctly
   - Memory failures are handled gracefully

5. **Contract Compliance**:
   - Agent follows BaseAgent interface
   - Config validation works
   - State transitions are valid

---

## Examples

### Minimal Agent Implementation

```python
from src.agents.base_agent import BaseAgent
from .contracts.base_contract import BaseAgentState, AgentConfig, AgentResult

class MinimalAgent(BaseAgent):
    """Minimal agent implementation."""

    tier = 1
    tier_name = "orchestration"
    agent_type = "standard"
    sla_seconds = 2.0

    def __init__(self, config: AgentConfig):
        super().__init__(config)

    async def execute(self, state: BaseAgentState) -> BaseAgentState:
        """Main execution logic."""

        # Validate input
        validate_input_state(state)

        # Perform work
        result = AgentResult(
            agent_name=self.agent_name,
            result_type="minimal_result",
            data={"message": "Hello from minimal agent"},
            confidence=1.0
        )

        # Update state
        updated_state = {
            **state,
            "agent_result": result.dict(),
            "status": "completed"
        }

        # Validate output
        validate_output_state(updated_state)

        return updated_state
```

### Error Handling Example

```python
async def execute(self, state: BaseAgentState) -> BaseAgentState:
    """Execute with error handling."""

    try:
        # Attempt primary method
        result = await self._primary_method(state)

    except DataQualityError as e:
        # Handle recoverable error
        return {
            **state,
            "status": "failed",
            "errors": state.get("errors", []) + [{
                "agent": self.agent_name,
                "error_type": "data_quality_failed",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "recoverable": False
            }]
        }

    except Exception as e:
        # Unhandled error
        return {
            **state,
            "status": "failed",
            "errors": state.get("errors", []) + [{
                "agent": self.agent_name,
                "error_type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "recoverable": False
            }]
        }

    return result
```

---

## Compliance Checklist

Before deploying an agent, verify:

- [ ] Extends `BaseAgent`
- [ ] Implements `execute()` method
- [ ] Has correct tier and tier_name class attributes
- [ ] Has correct agent_type class attribute
- [ ] Has realistic sla_seconds (or None)
- [ ] Config loaded from agent_config.yaml
- [ ] Input state validated
- [ ] Output state validated
- [ ] Errors properly structured
- [ ] Memory integration implemented (if applicable)
- [ ] Observability spans created
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Documentation complete

---

**End of Base Contracts**
