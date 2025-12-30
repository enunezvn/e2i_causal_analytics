# Orchestrator Contracts

**Purpose**: Define interfaces and expectations for communication between the Orchestrator agent and all other agents in the E2I Causal Analytics system.

**Version**: 1.1
**Last Updated**: 2025-12-30
**Owner**: E2I Development Team

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-18 | E2I Team | Initial version |
| 1.1 | 2025-12-30 | E2I Team | V4.4: Added discovery fields to AgentSelectionCriteria (enable_discovery, discovery_config) |

---

## Overview

The Orchestrator (Tier 1) is the central coordinator that:
- Receives parsed queries from NLP layer
- Selects appropriate agents to handle queries
- Dispatches work to agents (sequential or parallel)
- Aggregates results from multiple agents
- Synthesizes final responses

This contract ensures consistent, predictable communication between orchestrator and all 17 downstream agents.

---

## Orchestrator → Agent Dispatch

### Dispatch Request Structure

```python
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class AgentDispatchRequest(BaseModel):
    """
    Request structure when orchestrator dispatches to an agent.

    This is passed to agent.run(state) where state includes this dispatch.
    """

    # === IDENTIFICATION ===
    dispatch_id: str = Field(
        ...,
        description="Unique dispatch identifier",
        regex=r"^disp_[a-z0-9]{16}$"
    )
    session_id: str = Field(..., description="Session identifier")
    query_id: str = Field(..., description="Query identifier")

    # === ROUTING ===
    target_agent: str = Field(..., description="Agent to execute")
    dispatch_reason: str = Field(..., description="Why this agent was selected")
    priority: Literal["low", "medium", "high", "critical"] = Field(
        default="medium",
        description="Dispatch priority"
    )

    # === INPUT DATA ===
    query: str = Field(..., description="Original user query")
    parsed_query: Dict[str, Any] = Field(..., description="Parsed query from NLP")
    rag_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Retrieved RAG context"
    )

    # === EXECUTION CONTEXT ===
    execution_mode: Literal["sequential", "parallel"] = Field(
        default="sequential",
        description="How this agent fits in execution plan"
    )
    upstream_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Results from previously executed agents"
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="Agent names this dispatch depends on"
    )

    # === CONSTRAINTS ===
    timeout_seconds: Optional[float] = Field(
        None,
        description="Custom timeout for this dispatch (overrides agent default)"
    )
    max_cost_dollars: Optional[float] = Field(
        None,
        description="Maximum cost allowed (for LLM agents)"
    )

    # === METADATA ===
    dispatched_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Dispatch timestamp"
    )
    user_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional user context"
    )
```

### Dispatch Validation

```python
def validate_dispatch_request(request: AgentDispatchRequest) -> None:
    """
    Validate dispatch request before sending to agent.

    Raises:
        ValueError: If validation fails
    """
    # Target agent must exist
    VALID_AGENTS = [
        "scope_definer", "data_preparer", "model_selector", "model_trainer",
        "feature_analyzer", "model_deployer", "observability_connector",
        "causal_impact", "gap_analyzer", "heterogeneous_optimizer",
        "experiment_designer", "drift_monitor", "health_score",
        "prediction_synthesizer", "resource_optimizer",
        "explainer", "feedback_learner"
    ]

    if request.target_agent not in VALID_AGENTS:
        raise ValueError(f"Invalid target agent: {request.target_agent}")

    # Dependencies must be valid agents
    for dep in request.depends_on:
        if dep not in VALID_AGENTS:
            raise ValueError(f"Invalid dependency: {dep}")

    # Timeout must be positive
    if request.timeout_seconds is not None and request.timeout_seconds <= 0:
        raise ValueError("Timeout must be positive")

    # Cost limit must be positive
    if request.max_cost_dollars is not None and request.max_cost_dollars <= 0:
        raise ValueError("Cost limit must be positive")
```

---

## Agent → Orchestrator Response

### Response Structure

```python
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class AgentDispatchResponse(BaseModel):
    """
    Response structure when agent completes and returns to orchestrator.

    This wraps the AgentState returned by agent.run()
    """

    # === IDENTIFICATION ===
    dispatch_id: str = Field(..., description="Original dispatch ID")
    session_id: str = Field(..., description="Session identifier")
    query_id: str = Field(..., description="Query identifier")
    agent_name: str = Field(..., description="Agent that executed")

    # === STATUS ===
    status: Literal["completed", "failed", "timeout", "cancelled"] = Field(
        ...,
        description="Execution status"
    )

    # === RESULTS (if completed) ===
    agent_result: Optional[Dict[str, Any]] = Field(
        None,
        description="Primary agent result"
    )
    analysis_summary: Optional[str] = Field(
        None,
        description="Human-readable summary"
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Result confidence"
    )

    # === ERRORS (if failed) ===
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Errors encountered"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings raised"
    )

    # === ROUTING ===
    next_agent: Optional[str] = Field(
        None,
        description="Agent recommends routing to this agent next"
    )
    requires_human: bool = Field(
        default=False,
        description="Requires human-in-the-loop"
    )

    # === HANDOFF ===
    handoff: Optional[Dict[str, Any]] = Field(
        None,
        description="Structured handoff for next agent"
    )

    # === METADATA ===
    execution_time_ms: int = Field(..., description="Execution duration")
    tokens_used: Optional[int] = Field(None, description="LLM tokens used")
    model_used: Optional[str] = Field(None, description="Model used (if LLM)")
    started_at: datetime = Field(..., description="Start timestamp")
    completed_at: datetime = Field(..., description="Completion timestamp")

    # === OBSERVABILITY ===
    span_id: Optional[str] = Field(None, description="Opik span ID")
    trace_id: Optional[str] = Field(None, description="Opik trace ID")
```

### Response Validation

```python
def validate_dispatch_response(response: AgentDispatchResponse) -> None:
    """
    Validate agent response before orchestrator processes it.

    Raises:
        ValueError: If validation fails
    """
    # Completed status requires result
    if response.status == "completed":
        if not response.agent_result:
            raise ValueError("Completed status requires agent_result")

    # Failed status requires errors
    if response.status == "failed":
        if not response.errors:
            raise ValueError("Failed status requires errors")

    # Execution time must be positive
    if response.execution_time_ms < 0:
        raise ValueError("Execution time cannot be negative")

    # Timestamps must be in order
    if response.completed_at < response.started_at:
        raise ValueError("Completion time before start time")

    # Next agent must be valid if specified
    if response.next_agent:
        VALID_AGENTS = [
            "scope_definer", "data_preparer", "model_selector", "model_trainer",
            "feature_analyzer", "model_deployer", "observability_connector",
            "causal_impact", "gap_analyzer", "heterogeneous_optimizer",
            "experiment_designer", "drift_monitor", "health_score",
            "prediction_synthesizer", "resource_optimizer",
            "explainer", "feedback_learner"
        ]
        if response.next_agent not in VALID_AGENTS:
            raise ValueError(f"Invalid next_agent: {response.next_agent}")
```

---

## Agent Selection Contract

### Selection Criteria

```python
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field

class AgentSelectionCriteria(BaseModel):
    """
    Criteria used by orchestrator to select agents.
    """

    # === QUERY CLASSIFICATION ===
    intent_type: Literal[
        "causal",
        "exploratory",
        "comparative",
        "trend",
        "what_if",
        "ml_training",
        "model_deployment",
        "monitoring"
    ] = Field(..., description="Primary intent type")

    entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Extracted entities (brands, kpis, regions, etc.)"
    )

    # === COMPLEXITY INDICATORS ===
    complexity: Literal["simple", "moderate", "complex"] = Field(
        default="moderate",
        description="Query complexity"
    )

    requires_ml: bool = Field(
        default=False,
        description="Requires ML model training or inference"
    )

    requires_causal_inference: bool = Field(
        default=False,
        description="Requires causal analysis"
    )

    requires_experimentation: bool = Field(
        default=False,
        description="Requires experiment design"
    )

    # === DISCOVERY (V4.4+) ===
    enable_discovery: bool = Field(
        default=False,
        description="Enable automatic DAG structure learning for causal intents"
    )

    discovery_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Discovery configuration: algorithms=['ges','pc'], ensemble_threshold (0.5), alpha (0.05)"
    )

    # === CONSTRAINTS ===
    time_budget_seconds: Optional[float] = Field(
        None,
        description="Maximum time budget for response"
    )

    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable confidence"
    )

    # === AGENT PREFERENCES ===
    preferred_agents: List[str] = Field(
        default_factory=list,
        description="User or system preferences for agents"
    )

    excluded_agents: List[str] = Field(
        default_factory=list,
        description="Agents to exclude"
    )
```

### Selection Algorithm Contract

```python
from typing import List, Tuple

class AgentSelector:
    """
    Agent selection interface.

    Orchestrator uses this to determine which agents to dispatch.
    """

    def select_agents(
        self,
        criteria: AgentSelectionCriteria,
        available_agents: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Select agents based on criteria.

        Args:
            criteria: Selection criteria
            available_agents: List of available agent names

        Returns:
            List of (agent_name, priority_score) tuples, sorted by priority

        Priority score range: 0.0 (lowest) to 1.0 (highest)
        """
        pass

    def can_execute_parallel(
        self,
        agent_list: List[str]
    ) -> bool:
        """
        Determine if agents can execute in parallel.

        Args:
            agent_list: List of agent names

        Returns:
            True if agents can execute in parallel
        """
        pass

    def get_execution_order(
        self,
        agent_list: List[str]
    ) -> List[List[str]]:
        """
        Determine execution order for agents.

        Args:
            agent_list: List of agent names

        Returns:
            List of agent groups, where each group can execute in parallel

        Example:
            [["causal_impact"], ["gap_analyzer", "heterogeneous_optimizer"], ["explainer"]]
            - First execute causal_impact
            - Then execute gap_analyzer and heterogeneous_optimizer in parallel
            - Then execute explainer
        """
        pass
```

---

## Result Aggregation Contract

### Aggregation Strategy

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class AgentResultAggregation(BaseModel):
    """
    Aggregated results from multiple agents.
    """

    # === SOURCE AGENTS ===
    agents_executed: List[str] = Field(
        ...,
        description="Agents that contributed to result"
    )

    execution_order: List[str] = Field(
        ...,
        description="Order in which agents executed"
    )

    # === AGGREGATED DATA ===
    combined_results: Dict[str, Any] = Field(
        ...,
        description="Combined agent results"
    )

    overall_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence (min of all agents)"
    )

    # === SYNTHESIS ===
    synthesis: str = Field(
        ...,
        description="Natural language synthesis of results"
    )

    key_insights: List[str] = Field(
        default_factory=list,
        description="Key insights from aggregated results"
    )

    # === CONSISTENCY ===
    conflicts_detected: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Conflicts between agent results"
    )

    consistency_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How consistent results are (1.0 = fully consistent)"
    )

    # === METADATA ===
    total_execution_time_ms: int = Field(
        ...,
        description="Total time for all agents"
    )

    total_tokens_used: Optional[int] = Field(
        None,
        description="Total LLM tokens used"
    )
```

### Aggregation Rules

```python
class AggregationRules:
    """Rules for combining agent results."""

    @staticmethod
    def combine_confidence(confidences: List[float]) -> float:
        """
        Combine confidence scores from multiple agents.

        Strategy: Use minimum confidence (conservative approach)
        """
        return min(confidences) if confidences else 0.0

    @staticmethod
    def detect_conflicts(
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts between agent results.

        Returns:
            List of conflicts with agent names and conflicting values
        """
        pass

    @staticmethod
    def resolve_conflicts(
        conflicts: List[Dict[str, Any]],
        resolution_strategy: Literal["highest_confidence", "most_recent", "weighted_vote"]
    ) -> Dict[str, Any]:
        """
        Resolve conflicts using specified strategy.
        """
        pass
```

---

## Error Propagation Rules

### Error Handling Strategy

```python
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field

class ErrorPropagationPolicy(BaseModel):
    """
    Policy for handling errors from agents.
    """

    # === SEVERITY HANDLING ===
    on_warning: Literal["continue", "log_only"] = Field(
        default="continue",
        description="Action when agent returns warning"
    )

    on_error: Literal["fail_fast", "continue", "fallback"] = Field(
        default="fail_fast",
        description="Action when agent returns error"
    )

    on_timeout: Literal["fail_fast", "skip_agent", "use_partial"] = Field(
        default="skip_agent",
        description="Action when agent times out"
    )

    # === FALLBACK STRATEGY ===
    enable_fallback: bool = Field(
        default=True,
        description="Whether to use fallback agents on error"
    )

    fallback_agents: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Fallback agents for each primary agent"
    )

    # === PARTIAL RESULTS ===
    allow_partial_results: bool = Field(
        default=True,
        description="Whether to return partial results if some agents fail"
    )

    min_success_rate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of agents that must succeed"
    )
```

### Error Propagation Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    ERROR PROPAGATION FLOW                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Agent Error                                                 │
│       │                                                      │
│       ├─► Warning? ──► Log & Continue                        │
│       │                                                      │
│       ├─► Recoverable? ──┬─► Retry with backoff             │
│       │                  └─► Fallback agent (if configured)  │
│       │                                                      │
│       ├─► Timeout? ──┬─► Use partial results (if available) │
│       │              └─► Skip agent & continue               │
│       │                                                      │
│       └─► Critical? ──► Fail entire query                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Routing Decision Logic

### Intent-Based Routing

```python
from typing import List
from pydantic import BaseModel, Field

class IntentRoutingRule(BaseModel):
    """
    Routing rule based on query intent.
    """

    intent_type: str = Field(..., description="Intent type")
    primary_agents: List[str] = Field(..., description="Primary agents for this intent")
    optional_agents: List[str] = Field(default_factory=list, description="Optional agents")
    sequence_required: bool = Field(default=False, description="Must execute in sequence")

# Default routing rules
INTENT_ROUTING_RULES = {
    "causal": IntentRoutingRule(
        intent_type="causal",
        primary_agents=["causal_impact"],
        optional_agents=["gap_analyzer", "heterogeneous_optimizer", "explainer"],
        sequence_required=True
    ),

    "what_if": IntentRoutingRule(
        intent_type="what_if",
        primary_agents=["experiment_designer"],
        optional_agents=["causal_impact", "explainer"],
        sequence_required=True
    ),

    "trend": IntentRoutingRule(
        intent_type="trend",
        primary_agents=["drift_monitor"],
        optional_agents=["explainer"],
        sequence_required=False
    ),

    "ml_training": IntentRoutingRule(
        intent_type="ml_training",
        primary_agents=[
            "scope_definer",
            "data_preparer",
            "model_selector",
            "model_trainer",
            "feature_analyzer"
        ],
        optional_agents=["model_deployer"],
        sequence_required=True  # Must execute Tier 0 pipeline in order
    ),

    "model_deployment": IntentRoutingRule(
        intent_type="model_deployment",
        primary_agents=["model_deployer"],
        optional_agents=["health_score"],
        sequence_required=False
    ),

    "monitoring": IntentRoutingRule(
        intent_type="monitoring",
        primary_agents=["health_score", "drift_monitor"],
        optional_agents=["explainer"],
        sequence_required=False
    ),

    "exploratory": IntentRoutingRule(
        intent_type="exploratory",
        primary_agents=["gap_analyzer"],
        optional_agents=["causal_impact", "explainer"],
        sequence_required=False
    ),

    "comparative": IntentRoutingRule(
        intent_type="comparative",
        primary_agents=["gap_analyzer", "heterogeneous_optimizer"],
        optional_agents=["explainer"],
        sequence_required=False
    )
}
```

### KPI-Based Routing

```python
# KPI → Agent mapping
KPI_ROUTING_MAP = {
    # Prescription metrics
    "NRx": ["causal_impact", "gap_analyzer"],
    "TRx": ["causal_impact", "gap_analyzer"],
    "NBRx": ["causal_impact", "heterogeneous_optimizer"],

    # Model performance
    "model_accuracy": ["drift_monitor", "health_score"],
    "model_f1": ["drift_monitor", "health_score"],

    # Data quality
    "match_rate": ["health_score", "drift_monitor"],
    "data_freshness": ["health_score"],

    # Engagement
    "MAU": ["gap_analyzer"],
    "DAU": ["gap_analyzer"],

    # Operational
    "time_to_release": ["resource_optimizer"],
    "inference_latency": ["health_score", "resource_optimizer"]
}
```

---

## Parallel Execution Contracts

### Parallel Execution Rules

```python
from typing import List, Set
from pydantic import BaseModel, Field

class ParallelExecutionGroup(BaseModel):
    """
    Group of agents that can execute in parallel.
    """

    group_id: str = Field(..., description="Group identifier")
    agents: List[str] = Field(..., description="Agents in this group")
    shared_resources: Set[str] = Field(
        default_factory=set,
        description="Shared resources (database tables, models, etc.)"
    )
    max_concurrency: int = Field(
        default=10,
        description="Maximum concurrent executions"
    )
```

### Parallelization Rules

Agents can execute in parallel if:
1. **No shared mutable state**: Agents don't modify same data
2. **No dependencies**: Neither agent depends on other's output
3. **Resource availability**: Sufficient compute/memory for parallel execution

```python
# Agents that can NEVER execute in parallel (due to shared state)
EXCLUSIVE_PAIRS = [
    ("data_preparer", "model_trainer"),      # QC gate dependency
    ("model_trainer", "feature_analyzer"),   # Model must finish first
    ("feature_analyzer", "model_deployer"),  # SHAP before deploy
    ("scope_definer", "data_preparer"),      # Scope must be defined first
]

# Agents that CAN execute in parallel (independent)
PARALLEL_COMPATIBLE = [
    ("gap_analyzer", "heterogeneous_optimizer"),       # Both Tier 2, independent
    ("drift_monitor", "health_score"),                 # Both Tier 3, independent
    ("prediction_synthesizer", "resource_optimizer"),  # Both Tier 4, independent
    ("explainer", "feedback_learner"),                 # Both Tier 5, independent (usually)
]
```

---

## Orchestrator State Management

### Orchestrator Internal State

```python
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class OrchestratorState(BaseModel):
    """
    Orchestrator's internal state during query processing.

    NOT exposed to agents - internal only.
    """

    # === IDENTIFICATION ===
    session_id: str
    query_id: str

    # === EXECUTION PLAN ===
    execution_plan: List[List[str]] = Field(
        ...,
        description="Agent execution plan (groups of parallel agents)"
    )
    current_stage: int = Field(
        default=0,
        description="Current stage in execution plan"
    )

    # === DISPATCHES ===
    dispatched_agents: Dict[str, AgentDispatchRequest] = Field(
        default_factory=dict,
        description="All dispatched agents (agent_name -> dispatch)"
    )

    completed_agents: Dict[str, AgentDispatchResponse] = Field(
        default_factory=dict,
        description="Completed agents (agent_name -> response)"
    )

    failed_agents: Dict[str, AgentDispatchResponse] = Field(
        default_factory=dict,
        description="Failed agents (agent_name -> response)"
    )

    # === TIMING ===
    started_at: datetime
    updated_at: datetime
    total_execution_time_ms: int = 0

    # === ERRORS ===
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # === STATUS ===
    status: Literal["planning", "executing", "aggregating", "completed", "failed"]
```

---

## DSPy Hub Role

The Orchestrator serves as the **Hub** in the DSPy optimization architecture (from E2I DSPy Feedback Learner Architecture V2).

### Hub Responsibilities

```python
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class DSPyHubInterface:
    """
    DSPy Hub role for Orchestrator.

    The Hub coordinates the DSPy optimization loop:
    1. Collect training signals from Sender agents
    2. Trigger optimization when thresholds met
    3. Distribute optimized prompts to Recipients
    """

    async def collect_training_signal(
        self,
        signal: "TrainingSignal"
    ) -> None:
        """
        Collect training signal from a Sender agent.

        Called by: causal_impact, gap_analyzer, heterogeneous_optimizer,
                   drift_monitor, experiment_designer, prediction_synthesizer
        """
        pass

    async def check_optimization_trigger(self) -> bool:
        """
        Check if optimization should be triggered.

        Returns True if:
        - min_signals_for_optimization (100) reached, OR
        - optimization_interval_hours (24) exceeded
        """
        pass

    async def coordinate_optimization_cycle(
        self,
        signals: List["TrainingSignal"],
        target_signatures: List[str]
    ) -> "OptimizationResult":
        """
        Coordinate DSPy optimization with feedback_learner.

        Delegates to feedback_learner for actual MIPROv2 optimization.
        """
        pass

    async def distribute_optimized_prompts(
        self,
        prompts: Dict[str, str],
        recipient_agents: List[str]
    ) -> "DistributionResult":
        """
        Distribute optimized prompts to Recipient agents.

        Recipients: health_score, resource_optimizer, explainer
        Hybrids also receive: tool_composer, feedback_learner
        """
        pass

    def get_signal_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about collected signals.

        Returns counts by signature, quality metrics, optimization history.
        """
        pass


class OptimizationResult(BaseModel):
    """Result from DSPy optimization cycle."""
    optimization_id: str
    signatures_optimized: List[str]
    signals_used: int
    improvement_scores: Dict[str, float]  # signature -> improvement
    new_prompts: Dict[str, str]  # signature -> optimized prompt
    latency_ms: int
    success: bool
    errors: List[str] = Field(default_factory=list)


class DistributionResult(BaseModel):
    """Result from distributing optimized prompts."""
    distribution_id: str
    prompts_distributed: int
    recipients_updated: List[str]
    failures: List[Dict[str, str]] = Field(default_factory=list)
    latency_ms: int
```

### Hub Integration with Agent Dispatch

When dispatching to agents, the Hub includes DSPy context:

```python
class AgentDispatchRequest(BaseModel):
    # ... existing fields ...

    # === DSPy CONTEXT (Hub provides) ===
    optimized_prompts: Optional[Dict[str, str]] = Field(
        None,
        description="DSPy-optimized prompts for this agent (if recipient)"
    )
    collect_signals: bool = Field(
        default=True,
        description="Whether this agent should collect training signals"
    )
    signal_signature: Optional[str] = Field(
        None,
        description="Primary DSPy signature this agent uses"
    )
```

### Signal Collection in Agent Response

Agents return collected signals in their response:

```python
class AgentDispatchResponse(BaseModel):
    # ... existing fields ...

    # === DSPy SIGNALS (Senders provide) ===
    training_signals: List["TrainingSignal"] = Field(
        default_factory=list,
        description="Training signals collected during execution"
    )
    signal_quality: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Average quality of collected signals"
    )
```

---

## Testing Requirements

All orchestrator integrations must verify:

1. **Dispatch Validation**:
   - All required fields present
   - Valid agent names
   - Valid dependencies

2. **Response Validation**:
   - All required fields present
   - Status matches result content
   - Timing is logical

3. **Agent Selection**:
   - Correct agents selected for each intent
   - Parallelization rules followed
   - Dependencies respected

4. **Aggregation**:
   - Results combined correctly
   - Conflicts detected
   - Confidence calculated correctly

5. **Error Propagation**:
   - Errors handled per policy
   - Fallbacks triggered appropriately
   - Partial results returned when allowed

---

## Examples

### Sequential Execution

```python
# User query: "What caused the drop in NRx for Kisqali in Midwest?"

# Orchestrator execution plan:
execution_plan = [
    ["causal_impact"],              # Stage 1: Analyze causal relationships
    ["gap_analyzer"],               # Stage 2: Identify opportunities
    ["explainer"]                   # Stage 3: Generate explanation
]

# Dispatch to causal_impact
dispatch_1 = AgentDispatchRequest(
    dispatch_id="disp_abc123",
    session_id="sess_xyz",
    query_id="qry_789",
    target_agent="causal_impact",
    dispatch_reason="Causal intent detected",
    query="What caused the drop in NRx for Kisqali in Midwest?",
    parsed_query={
        "intent": "causal",
        "brands": ["Kisqali"],
        "kpis": ["NRx"],
        "regions": ["Midwest"],
        "time_period": "recent"
    },
    execution_mode="sequential",
    upstream_results=[]
)

# After causal_impact completes, dispatch to gap_analyzer with upstream results
dispatch_2 = AgentDispatchRequest(
    dispatch_id="disp_def456",
    session_id="sess_xyz",
    query_id="qry_789",
    target_agent="gap_analyzer",
    dispatch_reason="Identify improvement opportunities",
    query="What caused the drop in NRx for Kisqali in Midwest?",
    parsed_query={...},
    execution_mode="sequential",
    upstream_results=[
        {
            "agent": "causal_impact",
            "result": {...}
        }
    ],
    depends_on=["causal_impact"]
)
```

### Parallel Execution

```python
# User query: "Show me market trends and opportunities for Kisqali"

# Orchestrator execution plan:
execution_plan = [
    ["drift_monitor", "gap_analyzer"],  # Stage 1: Both can run in parallel
    ["explainer"]                       # Stage 2: Synthesize results
]

# Dispatch both agents simultaneously
dispatches = [
    AgentDispatchRequest(
        dispatch_id="disp_111",
        target_agent="drift_monitor",
        execution_mode="parallel",
        ...
    ),
    AgentDispatchRequest(
        dispatch_id="disp_222",
        target_agent="gap_analyzer",
        execution_mode="parallel",
        ...
    )
]

# Wait for both to complete, then dispatch explainer
```

---

## Compliance Checklist

Before deploying orchestrator integration:

- [ ] Dispatch requests validated
- [ ] Agent responses validated
- [ ] Agent selection logic correct
- [ ] Parallel execution rules enforced
- [ ] Dependencies respected
- [ ] Result aggregation tested
- [ ] Error propagation works correctly
- [ ] Fallback strategy implemented
- [ ] Timeout handling works
- [ ] Observability spans created
- [ ] Unit tests passing
- [ ] Integration tests with all 17 agents passing

---

**End of Orchestrator Contracts**
