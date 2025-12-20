# Orchestrator Integration Contracts

## Overview

The Orchestrator is the central hub for all agent communication. All requests flow through the Orchestrator, and all responses return through it.

## Input Contracts

### User Query Input

```python
# src/contracts/orchestrator.py

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from .base import BaseAgentInput

class OrchestratorInput(BaseAgentInput):
    """Input to the Orchestrator from user interface"""
    
    query: str = Field(..., description="User's natural language query")
    brand: Optional[str] = Field(None, description="Brand context (Remibrutinib, Fabhalta, Kisqali)")
    user_expertise: Literal["executive", "analyst", "data_scientist"] = "analyst"
    output_format: Literal["narrative", "structured", "brief"] = "narrative"
    max_latency_ms: Optional[int] = Field(None, description="Maximum acceptable latency")
    
    # Optional context
    conversation_history: Optional[List[dict]] = None
    prior_analysis_ids: Optional[List[str]] = None
```

### Intent Classification Output

```python
class IntentClassification(BaseModel):
    """Result of intent classification"""
    
    primary_intent: str
    confidence: float = Field(..., ge=0, le=1)
    secondary_intents: List[str] = []
    entities_extracted: dict = {}
    classification_method: Literal["pattern", "llm"] = "pattern"
    classification_latency_ms: int
```

## Dispatch Contracts

### Agent Dispatch Request

```python
class AgentDispatchRequest(BaseModel):
    """Request to dispatch to a specific agent"""
    
    agent: AgentName
    priority: int = Field(5, ge=1, le=10)
    timeout_ms: int = 30000
    
    # Input transformation
    input_payload: dict
    
    # Fallback configuration
    fallback_agent: Optional[AgentName] = None
    fallback_on: List[str] = ["timeout", "error"]
    
    # Parallel execution
    parallel_group: Optional[int] = None
    wait_for_group: bool = True
```

### Agent Dispatch Response

```python
class AgentDispatchResponse(BaseModel):
    """Response from a dispatched agent"""
    
    agent: AgentName
    status: ExecutionStatus
    
    # Timing
    dispatch_latency_ms: int
    execution_latency_ms: int
    total_latency_ms: int
    
    # Results
    output: dict
    key_findings: List[str] = []
    
    # Errors
    errors: List[AgentError] = []
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
```

## Routing Rules

### Intent to Agent Mapping

```python
INTENT_TO_AGENTS: Dict[str, List[AgentDispatchRequest]] = {
    "causal_effect": [
        AgentDispatchRequest(
            agent=AgentName.CAUSAL_IMPACT,
            priority=1,
            timeout_ms=30000,
            input_payload={"analysis_type": "effect_estimation"}
        )
    ],
    
    "segment_optimization": [
        AgentDispatchRequest(
            agent=AgentName.HETEROGENEOUS_OPTIMIZER,
            priority=1,
            timeout_ms=25000,
            input_payload={}
        )
    ],
    
    "opportunity_analysis": [
        AgentDispatchRequest(
            agent=AgentName.GAP_ANALYZER,
            priority=1,
            timeout_ms=20000,
            input_payload={}
        )
    ],
    
    "experiment_design": [
        AgentDispatchRequest(
            agent=AgentName.EXPERIMENT_DESIGNER,
            priority=1,
            timeout_ms=60000,
            input_payload={}
        )
    ],
    
    "prediction": [
        AgentDispatchRequest(
            agent=AgentName.PREDICTION_SYNTHESIZER,
            priority=1,
            timeout_ms=15000,
            input_payload={}
        )
    ],
    
    "resource_allocation": [
        AgentDispatchRequest(
            agent=AgentName.RESOURCE_OPTIMIZER,
            priority=1,
            timeout_ms=20000,
            input_payload={}
        )
    ],
    
    "system_health": [
        AgentDispatchRequest(
            agent=AgentName.HEALTH_SCORE,
            priority=1,
            timeout_ms=5000,
            input_payload={}
        ),
        AgentDispatchRequest(
            agent=AgentName.DRIFT_MONITOR,
            priority=2,
            timeout_ms=10000,
            parallel_group=1,
            input_payload={}
        )
    ],
    
    "explanation": [
        AgentDispatchRequest(
            agent=AgentName.EXPLAINER,
            priority=1,
            timeout_ms=45000,
            input_payload={}
        )
    ],
    
    "comprehensive_analysis": [
        # Multiple agents in sequence/parallel
        AgentDispatchRequest(
            agent=AgentName.CAUSAL_IMPACT,
            priority=1,
            timeout_ms=30000,
            parallel_group=1,
            input_payload={}
        ),
        AgentDispatchRequest(
            agent=AgentName.GAP_ANALYZER,
            priority=1,
            timeout_ms=20000,
            parallel_group=1,
            input_payload={}
        ),
        AgentDispatchRequest(
            agent=AgentName.EXPLAINER,
            priority=2,
            timeout_ms=45000,
            input_payload={}
        )
    ]
}
```

## Output Contracts

### Final Response

```python
class OrchestratorOutput(BaseAgentOutput):
    """Final output from Orchestrator to user"""
    
    # Response content
    response: str
    response_type: Literal["direct", "synthesized", "explanation"]
    
    # Agent execution summary
    agents_invoked: List[AgentName]
    agent_results: List[AgentDispatchResponse]
    
    # Timing
    total_latency_ms: int
    breakdown: dict = {}  # Latency by component
    
    # Suggestions
    follow_up_questions: List[str] = []
    related_analyses: List[str] = []
    
    # Classification info
    intent_classification: IntentClassification
```

## Error Contracts

### Orchestrator Error Response

```python
class OrchestratorErrorResponse(BaseAgentOutput):
    """Error response from Orchestrator"""
    
    status: ExecutionStatus = ExecutionStatus.FAILED
    
    error_message: str
    error_category: Literal[
        "classification_failed",
        "routing_failed",
        "all_agents_failed",
        "timeout",
        "synthesis_failed"
    ]
    
    # Partial results if available
    partial_results: Optional[List[AgentDispatchResponse]] = None
    
    # Recovery suggestions
    retry_recommended: bool = False
    alternative_queries: List[str] = []
```

## Handoff YAML Examples

### Request to Agent

```yaml
orchestrator_to_agent:
  source_agent: orchestrator
  target_agent: causal_impact
  handoff_type: request
  priority: 1
  timeout_ms: 30000
  
  payload:
    query: "What's the impact of increased sampling on Kisqali prescriptions?"
    brand: "Kisqali"
    treatment: "sampling_frequency"
    outcome: "new_rx"
    user_expertise: "analyst"
    
  metadata:
    execution_id: "abc123"
    start_time: "2025-01-15T10:30:00Z"
    intent: "causal_effect"
    confidence: 0.95
```

### Response from Agent

```yaml
agent_to_orchestrator:
  source_agent: causal_impact
  target_agent: orchestrator
  handoff_type: response
  
  payload:
    ate: 2.3
    confidence_interval: [1.8, 2.8]
    p_value: 0.001
    robustness_passed: true
    interpretation: "Increased sampling is associated with 2.3 additional prescriptions per HCP..."
    
  metadata:
    execution_id: "abc123"
    start_time: "2025-01-15T10:30:00Z"
    end_time: "2025-01-15T10:30:15Z"
    latency_ms: 15234
    model_used: "claude-sonnet-4-20250514"
    
  key_findings:
    - "Positive causal effect of 2.3 additional Rx per HCP"
    - "Effect is statistically significant (p < 0.001)"
    - "Robust to placebo and subset refutation tests"
    
  recommendations:
    - "Consider increasing sampling budget for high-potential HCPs"
    
  requires_further_analysis: false
```

### Multi-Agent Synthesis

```yaml
orchestrator_synthesis:
  source_agents:
    - causal_impact
    - gap_analyzer
  target_agent: explainer
  handoff_type: request
  
  payload:
    query: "Summarize the findings for executive presentation"
    analysis_results:
      - agent: causal_impact
        key_findings:
          - "2.3 additional Rx per HCP from sampling"
      - agent: gap_analyzer
        key_findings:
          - "Top 3 territories have 40% gap vs potential"
    user_expertise: "executive"
    output_format: "brief"
```
