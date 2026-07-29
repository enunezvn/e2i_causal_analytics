# Tier 1 Contracts: Orchestrator Agent

> Integration contracts for the Orchestrator agent (Tier 1).
> Defines input/output schemas, dispatch protocols, and handoff formats.

## Overview

The Orchestrator is the sole Tier 1 agent responsible for:
- Intent classification and routing
- Multi-agent coordination
- Response synthesis
- Cross-tier communication

**Agent Type:** Standard (Fast)  
**Model:** Haiku/Sonnet  
**Latency Budget:** <2s (strict)  
**Critical Path:** Yes

---

## Input Contract

### OrchestratorInput

The orchestrator receives parsed queries from the NLP layer.

```python
from typing import TypedDict, List, Optional, Literal
from datetime import datetime

class ParsedEntity(TypedDict):
    """Entity extracted from user query."""
    type: Literal["brand", "region", "kpi", "time_period", "hcp_id", "patient_id"]
    value: str
    confidence: float  # 0.0 - 1.0
    source: Literal["exact", "fuzzy", "inferred"]

class ParsedQuery(TypedDict):
    """Output from NLP layer, input to Orchestrator."""
    query_id: str
    raw_query: str
    intent: Literal[
        "causal_impact",      # What is the effect of X on Y?
        "gap_analysis",       # Where are the gaps/opportunities?
        "heterogeneous",      # Who responds best to X?
        "experiment_design",  # Design an experiment for X
        "prediction",         # Predict Y for entity Z
        "explanation",        # Explain the analysis
        "health_check",       # System health status
        "drift_check",        # Model/data drift status
        "resource_optimize",  # Budget/resource allocation
        "ml_training",        # Train a new model (Tier 0)
        "feature_analysis",   # Feature importance (Tier 0)
        "model_deploy"        # Deploy model (Tier 0)
    ]
    entities: List[ParsedEntity]
    confidence: float
    ambiguity_flag: bool
    context: Optional[dict]  # RAG-retrieved context
    timestamp: datetime

class OrchestratorInput(TypedDict):
    """Complete input to Orchestrator agent."""
    parsed_query: ParsedQuery
    user_id: str
    session_id: str
    conversation_history: List[dict]
    preferences: Optional[dict]
```

### Validation Rules

| Field | Rule | Error Action |
|-------|------|--------------|
| `query_id` | UUID format | Reject with 400 |
| `intent` | Must be valid IntentType | Fallback to `explanation` |
| `confidence` | 0.0 - 1.0 | Clamp to bounds |
| `entities` | Max 20 entities | Truncate with warning |

---

## Output Contract

### OrchestratorOutput

```python
from typing import TypedDict, List, Optional, Literal, Any
from datetime import datetime

class AgentResult(TypedDict):
    """Result from a dispatched agent."""
    agent_id: str
    agent_tier: int
    status: Literal["success", "partial", "failed", "timeout"]
    result: Any
    confidence: float
    latency_ms: int
    error: Optional[str]

class Citation(TypedDict):
    """Source citation for response."""
    source_type: Literal["causal_path", "agent_activity", "business_metric", "ml_model"]
    source_id: str
    relevance: float

class OrchestratorOutput(TypedDict):
    """Complete output from Orchestrator agent."""
    query_id: str
    status: Literal["success", "partial", "failed"]
    
    # Synthesized response
    response_text: str
    response_confidence: float
    
    # Agent execution details
    agents_dispatched: List[str]
    agent_results: List[AgentResult]
    
    # Metadata
    citations: List[Citation]
    visualizations: List[dict]  # Chart specifications
    follow_up_suggestions: List[str]
    
    # Performance
    total_latency_ms: int
    timestamp: datetime
```

---

## Dispatch Contract

### AgentDispatch

Format for dispatching work to Tier 0-5 agents.

```yaml
# orchestrator-dispatch.yaml
dispatch:
  dispatch_id: string  # UUID
  query_id: string     # From input
  target_agent: string # Agent identifier
  target_tier: int     # 0-5
  
  payload:
    intent: string
    entities: list
    context: object
    constraints:
      max_latency_ms: int
      required_confidence: float
      
  routing:
    priority: int      # 1=highest
    timeout_ms: int
    retry_count: int
    fallback_agent: string | null
    
  metadata:
    dispatched_at: datetime
    correlation_id: string
```

### Dispatch Examples

#### Causal Impact Query
```yaml
dispatch:
  dispatch_id: "d-001"
  query_id: "q-123"
  target_agent: "causal_impact"
  target_tier: 2
  
  payload:
    intent: "causal_impact"
    entities:
      - type: "brand"
        value: "Kisqali"
      - type: "kpi"
        value: "conversion_rate"
    context:
      treatment: "hcp_engagement_increase"
      outcome: "patient_conversion"
    constraints:
      max_latency_ms: 120000
      required_confidence: 0.8
      
  routing:
    priority: 1
    timeout_ms: 120000
    retry_count: 2
    fallback_agent: "explainer"
```

#### ML Training Query (Tier 0)
```yaml
dispatch:
  dispatch_id: "d-002"
  query_id: "q-456"
  target_agent: "scope_definer"
  target_tier: 0
  
  payload:
    intent: "ml_training"
    entities:
      - type: "brand"
        value: "Fabhalta"
    context:
      model_type: "conversion_prediction"
      target_variable: "converted_30d"
    constraints:
      max_latency_ms: 300000  # 5 min for full pipeline
      required_confidence: 0.75
      
  routing:
    priority: 2
    timeout_ms: 300000
    retry_count: 1
    fallback_agent: null  # No fallback for ML training
```

---

## Handoff Contract

### AgentHandoff

Format for agents returning results to Orchestrator.

```yaml
# agent-handoff.yaml
handoff:
  dispatch_id: string    # From dispatch
  agent_id: string
  agent_tier: int
  
  execution:
    status: "success" | "partial" | "failed" | "timeout" | "blocked"
    started_at: datetime
    completed_at: datetime
    latency_ms: int
    
  result:
    primary_output: object   # Agent-specific
    confidence: float
    evidence: list
    caveats: list
    
  artifacts:
    visualizations: list
    tables: list
    models: list           # For Tier 0 agents
    
  errors:
    error_type: string | null
    error_message: string | null
    recoverable: bool
    
  metadata:
    model_used: string
    tokens_consumed: int
    cache_hit: bool
```

### Handoff Examples

#### Causal Impact Result
```yaml
handoff:
  dispatch_id: "d-001"
  agent_id: "causal_impact"
  agent_tier: 2
  
  execution:
    status: "success"
    started_at: "2025-12-08T10:00:00Z"
    completed_at: "2025-12-08T10:01:45Z"
    latency_ms: 105000
    
  result:
    primary_output:
      treatment: "hcp_engagement_increase"
      outcome: "patient_conversion"
      ate: 0.12
      confidence_interval: [0.08, 0.16]
      p_value: 0.003
      refutation_passed: true
    confidence: 0.87
    evidence:
      - "Placebo test: effect near zero (0.002)"
      - "Random common cause: estimate stable"
    caveats:
      - "Effect strongest in oncology segment"
      
  artifacts:
    visualizations:
      - type: "causal_graph"
        data: {...}
      - type: "effect_distribution"
        data: {...}
```

#### Tier 0 Training Result
```yaml
handoff:
  dispatch_id: "d-002"
  agent_id: "model_trainer"
  agent_tier: 0
  
  execution:
    status: "success"
    started_at: "2025-12-08T10:00:00Z"
    completed_at: "2025-12-08T10:15:30Z"
    latency_ms: 930000
    
  result:
    primary_output:
      model_id: "e2i-fabhalta-conversion-v1"
      model_version: "1.0.0"
      metrics:
        rmse: 0.142
        r2: 0.78
        auc: 0.84
      best_params:
        n_estimators: 500
        max_depth: 7
        learning_rate: 0.05
    confidence: 0.82
    evidence:
      - "Cross-validation: 5-fold, mean AUC 0.83"
      - "QC gate passed: 98% expectations met"
    caveats:
      - "Limited training data for new patients"
      
  artifacts:
    models:
      - mlflow_uri: "models:/e2i-fabhalta-conversion/1"
        stage: "Staging"
```

---

## Routing Rules

### Intent to Agent Mapping

| Intent | Primary Agent | Tier | Supporting Agents |
|--------|--------------|------|-------------------|
| `causal_impact` | causal_impact | 2 | explainer |
| `gap_analysis` | gap_analyzer | 2 | resource_optimizer |
| `heterogeneous` | heterogeneous_optimizer | 2 | prediction_synthesizer |
| `experiment_design` | experiment_designer | 3 | causal_impact |
| `prediction` | prediction_synthesizer | 4 | explainer, model_deployer |
| `explanation` | explainer | 5 | - |
| `health_check` | health_score | 3 | drift_monitor, observability_connector |
| `drift_check` | drift_monitor | 3 | health_score, data_preparer |
| `resource_optimize` | resource_optimizer | 4 | gap_analyzer |
| `ml_training` | scope_definer → model_trainer | 0 | data_preparer, model_selector |
| `feature_analysis` | feature_analyzer | 0 | explainer |
| `model_deploy` | model_deployer | 0 | model_trainer |

### Priority Rules

1. Lower tier = higher priority
2. Critical path agents get priority scheduling
3. Tier 0 requests may involve pipeline orchestration (multi-agent sequence)

### Timeout Configuration

| Tier | Default Timeout | Max Retries | Fallback Strategy |
|------|-----------------|-------------|-------------------|
| 0 | Variable | 1 | None (fail) |
| 1 | 2,000ms | 0 | N/A |
| 2 | 120,000ms | 2 | explainer |
| 3 | 60,000ms | 2 | health_score |
| 4 | 20,000ms | 3 | explainer |
| 5 | 180,000ms | 1 | template response |

---

## Multi-Agent Coordination

### Sequential Dispatch (Tier 0 Pipeline)

For ML training requests, orchestrator manages sequential dispatch:

```
scope_definer → data_preparer → model_selector → model_trainer
                     │
                     ▼
              [QC Gate Check]
                     │
            ┌────────┴────────┐
            │                 │
         PASS              FAIL
            │                 │
            ▼                 ▼
    Continue Pipeline    Return Error
```

### Parallel Dispatch

For complex queries requiring multiple perspectives:

```python
# Example: "What's causing conversion drops and how do we fix it?"
parallel_dispatches = [
    {"agent": "causal_impact", "intent": "identify_causes"},
    {"agent": "gap_analyzer", "intent": "find_opportunities"},
    {"agent": "resource_optimizer", "intent": "suggest_allocation"}
]

# Orchestrator waits for all, then synthesizes
results = await asyncio.gather(*[dispatch(d) for d in parallel_dispatches])
synthesized = await synthesize_results(results)
```

---

## Error Handling

### Error Categories

```python
class OrchestratorError(Enum):
    INVALID_INPUT = "invalid_input"
    ROUTING_FAILED = "routing_failed"
    AGENT_TIMEOUT = "agent_timeout"
    AGENT_ERROR = "agent_error"
    SYNTHESIS_FAILED = "synthesis_failed"
    QC_GATE_BLOCKED = "qc_gate_blocked"  # Tier 0 specific
```

### Recovery Actions

| Error | Recovery Action |
|-------|-----------------|
| `INVALID_INPUT` | Return 400 with validation details |
| `ROUTING_FAILED` | Fallback to `explainer` agent |
| `AGENT_TIMEOUT` | Retry with extended timeout, then fallback |
| `AGENT_ERROR` | Log, retry if recoverable, fallback otherwise |
| `SYNTHESIS_FAILED` | Return partial results with caveat |
| `QC_GATE_BLOCKED` | Return blocked status with QC report |

---

## Validation Checklist

### Pre-Dispatch Validation

- [ ] Query ID is valid UUID
- [ ] Intent is recognized
- [ ] Required entities present for intent
- [ ] User has permission for requested operation
- [ ] Target agent is available

### Post-Handoff Validation

- [ ] Dispatch ID matches
- [ ] Status is valid
- [ ] Latency within budget (warning if exceeded)
- [ ] Confidence meets threshold
- [ ] Required output fields present

---

---

## DSPy Hub Role

### Overview

The Orchestrator is a **DSPy Hub** agent that coordinates DSPy optimization across all agents.

```python
# dspy_type identification
dspy_type: Literal["hub"] = "hub"
```

### DSPy Signatures

#### AgentRoutingSignature

Routes queries to appropriate E2I agents.

```python
class AgentRoutingSignature(dspy.Signature):
    """Route queries to appropriate E2I agents."""

    query: str = dspy.InputField(desc="User query to route")
    query_pattern: str = dspy.InputField(desc="Classified query type")
    entities: str = dspy.InputField(desc="Extracted entities from query")
    available_agents: str = dspy.InputField(desc="List of available agents")

    primary_agent: str = dspy.OutputField(desc="Primary agent to handle query")
    secondary_agents: list = dspy.OutputField(desc="Secondary agents for context")
    routing_confidence: float = dspy.OutputField(desc="Confidence in routing (0-1)")
    routing_rationale: str = dspy.OutputField(desc="Explanation of routing decision")
```

#### IntentClassificationSignature

Classifies query intent for routing.

```python
class IntentClassificationSignature(dspy.Signature):
    """Classify query intent for routing."""

    query: str = dspy.InputField(desc="User query")
    conversation_context: str = dspy.InputField(desc="Recent conversation history")

    intent: str = dspy.OutputField(desc="Intent classification")
    sub_intent: str = dspy.OutputField(desc="More specific intent")
    confidence: float = dspy.OutputField(desc="Confidence in classification (0-1)")
```

### Training Signal Contract

```python
@dataclass
class RoutingTrainingSignal:
    """Training signal for AgentRoutingSignature optimization."""

    # Input Context
    signal_id: str
    session_id: str
    query: str
    query_pattern: str
    intent: str
    entities_extracted: List[str]

    # Routing Decision
    agents_selected: List[str]
    routing_confidence: float
    routing_rationale: str

    # Execution Outcome
    agents_succeeded: int
    agents_failed: int
    total_latency_ms: float

    # Quality Metrics (Delayed)
    user_satisfaction: Optional[float]  # 1-5 rating
    answer_quality: Optional[float]  # 0-1 score
    was_rerouted: bool

    def compute_reward(self) -> float:
        """
        Compute reward for MIPROv2 optimization.

        Weighting:
        - routing_accuracy: 0.35 (agents succeeded / selected)
        - efficiency: 0.25 (latency penalty)
        - no_rerouting: 0.20 (got it right first time)
        - user_satisfaction: 0.20 (if available)
        """
        ...
```

### Hub Coordination

```python
class OrchestratorDSPyHub:
    """DSPy Hub coordination for the Orchestrator."""

    dspy_type: Literal["hub"] = "hub"

    async def request_optimization(
        self,
        agent_name: str,
        signature_name: str,
        training_signals: List[Dict[str, Any]],
        priority: Literal["low", "medium", "high"] = "medium",
    ) -> str:
        """Request optimization for an agent's DSPy signature."""
        ...

    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """Get pending optimization requests."""
        ...
```

### Signal Collection

```python
class OrchestratorSignalCollector:
    """Collects training signals from orchestrator routing decisions."""

    def collect_routing_signal(...) -> RoutingTrainingSignal:
        """Collect training signal at routing decision time."""
        ...

    def update_with_outcome(...) -> RoutingTrainingSignal:
        """Update signal with execution outcome."""
        ...

    def update_with_feedback(...) -> RoutingTrainingSignal:
        """Update signal with user feedback (delayed)."""
        ...

    def get_signals_for_training(...) -> List[Dict[str, Any]]:
        """Get signals suitable for DSPy training."""
        ...
```

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-23 | V5: Added DSPy Hub role specification |
| 2025-12-08 | V4: Added Tier 0 dispatch patterns |
| 2025-12-08 | V4: Added QC gate blocking status |
| 2025-12-08 | V4: Added ML training sequential dispatch |
| 2025-12-04 | Initial creation for V3 |
