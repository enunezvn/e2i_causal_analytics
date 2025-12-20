# Base Integration Contracts

## Overview

This document defines the base contracts that all E2I agents must implement for inter-agent communication. All contracts use Pydantic BaseModel for validation.

## Core Types

```python
# src/contracts/base.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum

# === ENUMS ===

class AgentName(str, Enum):
    """All available agents"""
    ORCHESTRATOR = "orchestrator"
    CAUSAL_IMPACT = "causal_impact"
    GAP_ANALYZER = "gap_analyzer"
    HETEROGENEOUS_OPTIMIZER = "heterogeneous_optimizer"
    EXPERIMENT_DESIGNER = "experiment_designer"
    DRIFT_MONITOR = "drift_monitor"
    HEALTH_SCORE = "health_score"
    PREDICTION_SYNTHESIZER = "prediction_synthesizer"
    RESOURCE_OPTIMIZER = "resource_optimizer"
    EXPLAINER = "explainer"
    FEEDBACK_LEARNER = "feedback_learner"

class AgentTier(int, Enum):
    """Agent tier levels"""
    ORCHESTRATION = 1
    CAUSAL_INFERENCE = 2
    DESIGN_MONITORING = 3
    ML_PREDICTIONS = 4
    SELF_IMPROVEMENT = 5

class ExecutionStatus(str, Enum):
    """Execution status codes"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    PARTIAL = "partial"

class ErrorSeverity(str, Enum):
    """Error severity levels"""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

# === BASE MODELS ===

class AgentError(BaseModel):
    """Standardized error structure"""
    error_id: str
    node: str
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: datetime = Field(default_factory=datetime.now)
    recoverable: bool = True
    context: Optional[Dict[str, Any]] = None

class ExecutionMetadata(BaseModel):
    """Execution metadata for all agents"""
    agent: AgentName
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    latency_ms: Optional[int] = None
    model_used: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None

class BaseAgentInput(BaseModel):
    """Base input contract for all agents"""
    query: str
    execution_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    brand: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "allow"  # Allow additional fields

class BaseAgentOutput(BaseModel):
    """Base output contract for all agents"""
    status: ExecutionStatus
    execution_metadata: ExecutionMetadata
    errors: List[AgentError] = []
    warnings: List[str] = []
    
    class Config:
        extra = "allow"

# === HANDOFF STRUCTURES ===

class HandoffPayload(BaseModel):
    """Standardized handoff between agents"""
    source_agent: AgentName
    target_agent: AgentName
    handoff_type: Literal["request", "response", "error"]
    priority: int = Field(default=5, ge=1, le=10)
    timeout_ms: int = 30000
    payload: Dict[str, Any]
    metadata: ExecutionMetadata

class AgentDispatch(BaseModel):
    """Dispatch instruction from Orchestrator"""
    agent: AgentName
    priority: int
    timeout_ms: int
    input_mapping: Dict[str, str]  # Maps query fields to agent input fields
    fallback_agent: Optional[AgentName] = None
    parallel_group: Optional[int] = None  # For parallel execution

class AgentResult(BaseModel):
    """Result from dispatched agent"""
    agent: AgentName
    status: ExecutionStatus
    output: Dict[str, Any]
    latency_ms: int
    errors: List[AgentError] = []

# === UTILITY FUNCTIONS ===

def create_handoff(
    source: AgentName,
    target: AgentName,
    payload: Dict[str, Any],
    execution_id: str,
    priority: int = 5,
    timeout_ms: int = 30000
) -> HandoffPayload:
    """Create a standardized handoff payload"""
    
    return HandoffPayload(
        source_agent=source,
        target_agent=target,
        handoff_type="request",
        priority=priority,
        timeout_ms=timeout_ms,
        payload=payload,
        metadata=ExecutionMetadata(
            agent=source,
            execution_id=execution_id,
            start_time=datetime.now()
        )
    )

def create_error_response(
    agent: AgentName,
    execution_id: str,
    error: Exception,
    node: str = "unknown"
) -> BaseAgentOutput:
    """Create a standardized error response"""
    
    return BaseAgentOutput(
        status=ExecutionStatus.FAILED,
        execution_metadata=ExecutionMetadata(
            agent=agent,
            execution_id=execution_id,
            start_time=datetime.now(),
            end_time=datetime.now()
        ),
        errors=[
            AgentError(
                error_id=str(uuid4())[:8],
                node=node,
                error_type=type(error).__name__,
                message=str(error),
                severity=ErrorSeverity.ERROR
            )
        ]
    )
```

## Contract Validation

```python
# src/contracts/validation.py

from pydantic import ValidationError
from typing import Type, TypeVar, Dict, Any

T = TypeVar('T', bound=BaseModel)

def validate_input(data: Dict[str, Any], contract: Type[T]) -> T:
    """Validate input against contract"""
    try:
        return contract(**data)
    except ValidationError as e:
        raise ContractViolation(f"Input validation failed: {e}")

def validate_output(data: Dict[str, Any], contract: Type[T]) -> T:
    """Validate output against contract"""
    try:
        return contract(**data)
    except ValidationError as e:
        raise ContractViolation(f"Output validation failed: {e}")

class ContractViolation(Exception):
    """Raised when a contract is violated"""
    pass
```

## YAML Handoff Format

For human-readable handoffs and debugging:

```yaml
# Standard handoff format
handoff:
  source_agent: <agent_name>
  target_agent: <agent_name>
  handoff_type: request|response|error
  priority: 1-10
  timeout_ms: <milliseconds>
  
  payload:
    query: <original query>
    # Agent-specific fields
    
  metadata:
    execution_id: <uuid>
    start_time: <iso timestamp>
    latency_ms: <if response>
    model_used: <if applicable>
    
  # For responses
  key_findings:
    - <finding 1>
    - <finding 2>
  
  recommendations:
    - <recommendation 1>
  
  requires_further_analysis: true|false
  suggested_next_agent: <agent_name>
```
