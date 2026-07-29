# Tier 5: Self-Improvement Contracts

## Overview

Tier 5 agents handle explanation generation and continuous learning from feedback. These agents operate with extended reasoning (Deep pattern) and may run asynchronously.

**Agents Covered:**
- `explainer` - Natural language explanations of analyses
- `feedback_learner` - Async learning from user feedback

**Latency Budgets:**
- Explainer: <45s
- Feedback Learner: Async (no real-time limit)

**Model Tier:** Opus preferred (complex reasoning required)

---

## Explainer Contract

### Purpose
Generates natural language explanations of causal analyses, predictions, and recommendations for non-technical stakeholders.

### Input Contract

```yaml
# explainer_input.yaml
explanation_request:
  request_id: string              # UUID
  timestamp: datetime             # ISO 8601
  
  # Source analysis to explain
  source:
    agent: string                 # Source agent name
    agent_tier: int               # 2, 3, or 4
    analysis_type: string         # "causal_effect" | "gap_analysis" | "prediction" | "optimization"
    analysis_results: object      # Full results from source agent
    
  # Original query context
  query_context:
    original_query: string        # User's original question
    intent: string                # Classified intent
    entities: object              # Extracted entities
    
  # Explanation parameters
  explanation_config:
    audience: enum                # "executive" | "analyst" | "field_rep" | "technical"
    detail_level: enum            # "summary" | "standard" | "detailed"
    format: enum                  # "narrative" | "bullet_points" | "structured"
    max_length: int               # Maximum characters (optional)
    include_caveats: bool         # Include methodology caveats (default: true)
    include_next_steps: bool      # Include recommended actions (default: true)
    
  # Brand context
  brand_context:
    brand: string                 # "Remibrutinib" | "Fabhalta" | "Kisqali"
    therapeutic_area: string
    key_messages: list[string]    # Brand-specific messaging guidelines
```

### Output Contract

```yaml
# explainer_output.yaml
explanation_response:
  request_id: string              # Echo from input
  timestamp: datetime
  processing_time_ms: int
  
  # Primary explanation
  explanation:
    # Main narrative
    summary: string               # 1-2 sentence executive summary
    main_explanation: string      # Full explanation text
    
    # Structured components
    key_findings:
      - finding: string
        confidence: enum          # "high" | "medium" | "low"
        evidence: string          # Supporting data point
        
    # Causal narrative (if applicable)
    causal_story:
      cause: string
      effect: string
      mechanism: string           # How cause leads to effect
      magnitude: string           # Plain language effect size
      
    # Uncertainty communication
    caveats:
      - caveat: string
        severity: enum            # "minor" | "moderate" | "significant"
        
    # Actionable insights
    next_steps:
      - action: string
        priority: enum            # "high" | "medium" | "low"
        expected_impact: string
        
    # Visual suggestions
    suggested_visualizations:
      - type: string              # "causal_graph" | "waterfall" | "bar" | "trend"
        title: string
        description: string
        
  # Audience adaptation
  audience_metadata:
    target_audience: string
    reading_level: string         # "executive" | "technical"
    jargon_used: list[string]     # Technical terms included
    
  # Source attribution
  source_attribution:
    source_agent: string
    source_confidence: float
    key_assumptions: list[string]
    
  # Quality metrics
  quality_metrics:
    explanation_confidence: float # Confidence in explanation quality
    completeness: float           # Coverage of source analysis
    clarity_score: float          # Estimated readability
    
  # Status
  status: enum                    # "success" | "partial" | "failed"
  warnings: list[string]
  errors: list[object]
```

### Required Output Keys (Contract 2 Compliance)

```python
REQUIRED_KEYS = ["explanation"]
```

### Validation Rules

1. **Accuracy**: Explanation must faithfully represent source analysis
2. **No Hallucination**: All claims must be traceable to source data
3. **Audience Appropriate**: Language complexity matches audience level
4. **Caveat Inclusion**: Methodology limitations must be communicated
5. **Actionability**: Next steps must be concrete and achievable

---

## Feedback Learner Contract

### Purpose
Asynchronously processes user feedback to improve system performance, optimize RAG retrieval weights, and identify patterns for model refinement.

### Input Contract

```yaml
# feedback_learner_input.yaml
feedback_batch:
  batch_id: string                # UUID
  timestamp: datetime             # ISO 8601
  
  # Feedback items
  feedback_items:
    - feedback_id: string
      timestamp: datetime
      
      # User feedback
      feedback_type: enum         # "thumbs_up" | "thumbs_down" | "rating" | "correction" | "comment"
      feedback_value: any         # true/false, 1-5, string, etc.
      feedback_text: string       # Optional user comment
      
      # Context of feedback
      context:
        conversation_id: string
        message_id: string
        original_query: string
        response_text: string
        agents_used: list[string]
        
      # Analysis that was evaluated
      analysis_context:
        primary_agent: string
        analysis_type: string
        confidence: float
        key_results: object
        
      # RAG context (for retrieval optimization)
      rag_context:
        retrieved_chunks: list[object]
        retrieval_scores: list[float]
        
  # Learning parameters
  learning_config:
    learning_rate: float          # Weight update rate (default: 0.01)
    min_feedback_count: int       # Minimum feedback before learning (default: 10)
    pattern_threshold: float      # Confidence for pattern detection (default: 0.8)
```

### Output Contract

```yaml
# feedback_learner_output.yaml
learning_response:
  batch_id: string                # Echo from input
  timestamp: datetime
  processing_time_ms: int
  
  # Learned patterns
  learned_patterns:
    # Query patterns
    query_patterns:
      - pattern_id: string
        pattern_type: enum        # "positive" | "negative" | "neutral"
        query_characteristics: object
        success_rate: float
        sample_size: int
        confidence: float
        
    # Agent performance patterns
    agent_patterns:
      - agent: string
        performance_trend: enum   # "improving" | "stable" | "degrading"
        common_failures: list[string]
        success_factors: list[string]
        
    # RAG retrieval patterns
    retrieval_patterns:
      - source_type: string
        relevance_score: float
        user_satisfaction: float
        recommended_weight_adjustment: float
        
  # Weight updates (for RAG optimization)
  weight_updates:
    dense_weight_delta: float
    sparse_weight_delta: float
    graph_weight_delta: float
    new_weights:
      dense: float
      sparse: float
      graph: float
      
  # Recommendations
  recommendations:
    # Model recommendations
    model_recommendations:
      - recommendation: string
        priority: enum            # "high" | "medium" | "low"
        expected_impact: string
        evidence: string
        
    # Prompt recommendations
    prompt_recommendations:
      - agent: string
        recommendation: string
        sample_queries: list[string]
        
    # Data recommendations
    data_recommendations:
      - recommendation: string
        data_source: string
        rationale: string
        
  # Knowledge store updates
  knowledge_updates:
    facts_added: int
    facts_updated: int
    facts_deprecated: int
    new_facts:
      - fact_id: string
        fact_type: string
        content: string
        confidence: float
        source_feedback_ids: list[string]
        
  # Learning metrics
  learning_metrics:
    feedback_processed: int
    patterns_detected: int
    weights_updated: bool
    knowledge_updated: bool
    
  # Status
  status: enum                    # "success" | "partial" | "insufficient_data" | "failed"
  warnings: list[string]
  errors: list[object]
```

### Required Output Keys (Contract 2 Compliance)

```python
REQUIRED_KEYS = ["learned_patterns"]
```

### Validation Rules

1. **Minimum Data**: Require minimum feedback count before learning
2. **Pattern Confidence**: Only report patterns above confidence threshold
3. **Weight Bounds**: RAG weight updates must keep weights in valid range
4. **Audit Trail**: All updates must be traceable to source feedback
5. **No Overfit**: Patterns must be validated across multiple feedback sources

---

## Inter-Agent Communication

### Tier 5 → Orchestrator Handoff

```yaml
# tier5_handoff.yaml
handoff:
  source_agent: string            # "explainer" | "feedback_learner"
  source_tier: 5
  
  # Results summary
  summary:
    primary_result: string        # One-line summary
    confidence: float
    
  # Full results
  analysis_results:
    # Agent-specific keys as defined above
    
  # Narrative
  narrative: string               # Natural language explanation
  
  # Metadata
  processing_time_ms: int
  model_used: string              # "opus" | "sonnet"
  reasoning_depth: enum           # "standard" | "extended"
```

### Tier 5 ← All Tiers Dependencies

```yaml
# tier5_dependencies.yaml
explainer:
  required_inputs:
    - source: "any_agent"
      data: analysis_results
      usage: "Source material for explanation"
  optional_inputs:
    - source: orchestrator
      data: query_context
      usage: "Original user intent"
      
feedback_learner:
  required_inputs:
    - source: "user_feedback_queue"
      data: feedback_batch
      usage: "Raw feedback to process"
  optional_inputs:
    - source: rag
      data: retrieval_context
      usage: "RAG performance data"
    - source: "all_agents"
      data: analysis_results
      usage: "Agent performance data"
```

### Feedback Learner → RAG Integration

```yaml
# feedback_to_rag.yaml
rag_update:
  update_type: "weight_adjustment"
  
  current_weights:
    dense: float
    sparse: float
    graph: float
    
  new_weights:
    dense: float
    sparse: float
    graph: float
    
  rationale: string
  confidence: float
  effective_date: datetime
  rollback_available: bool
```

### Feedback Learner → Experiment Knowledge Store

```yaml
# feedback_to_knowledge.yaml
knowledge_update:
  update_type: "fact_insertion" | "fact_update" | "fact_deprecation"
  
  facts:
    - fact_id: string
      category: string            # "experiment_outcome" | "query_pattern" | "agent_behavior"
      content: string
      confidence: float
      metadata:
        source_feedback_count: int
        first_observed: datetime
        last_confirmed: datetime
```

---

## Async Processing (Feedback Learner)

### Queue Contract

```yaml
# feedback_queue.yaml
queue_config:
  queue_name: "e2i_feedback_queue"
  batch_size: 50                  # Process in batches
  batch_interval_seconds: 300     # Every 5 minutes
  max_batch_wait_seconds: 3600    # Max 1 hour between processing
  
message_format:
  message_id: string
  timestamp: datetime
  feedback_item: object           # Single feedback item
  priority: enum                  # "high" | "normal" | "low"
```

### Async Response Contract

```yaml
# feedback_async_response.yaml
async_response:
  batch_id: string
  status: enum                    # "queued" | "processing" | "completed" | "failed"
  
  # If completed
  completion_timestamp: datetime
  results_location: string        # S3 URI or API endpoint
  
  # Progress (if processing)
  progress:
    items_processed: int
    items_total: int
    estimated_completion: datetime
```

---

## Error Handling

### Explainer Errors

| Error Code | Description | Recovery |
|------------|-------------|----------|
| `EX_001` | Source analysis incomplete | Generate partial explanation with caveat |
| `EX_002` | Audience level unclear | Default to "analyst" level |
| `EX_003` | Explanation too long | Summarize and offer detailed version |
| `EX_004` | Causal claim unverifiable | Exclude claim with note |

### Feedback Learner Errors

| Error Code | Description | Recovery |
|------------|-------------|----------|
| `FL_001` | Insufficient feedback | Skip learning, report minimum needed |
| `FL_002` | Conflicting feedback patterns | Report uncertainty, no weight update |
| `FL_003` | Weight update out of bounds | Clamp to valid range with warning |
| `FL_004` | Knowledge store unavailable | Queue updates for retry |

---

## Quality Assurance

### Explainer Quality Checks

```python
class ExplainerQualityContract:
    """Quality checks for explanations."""
    
    @staticmethod
    def validate_faithfulness(explanation: str, source: dict) -> bool:
        """Ensure explanation accurately represents source."""
        pass
    
    @staticmethod
    def validate_no_hallucination(explanation: str, source: dict) -> bool:
        """Ensure no claims beyond source data."""
        pass
    
    @staticmethod
    def validate_readability(explanation: str, audience: str) -> bool:
        """Ensure appropriate reading level."""
        pass
```

### Feedback Learner Quality Checks

```python
class FeedbackLearnerQualityContract:
    """Quality checks for learning outputs."""
    
    @staticmethod
    def validate_pattern_significance(pattern: dict) -> bool:
        """Ensure pattern has statistical significance."""
        pass
    
    @staticmethod
    def validate_weight_stability(old: dict, new: dict) -> bool:
        """Ensure weight changes are not too drastic."""
        pass
    
    @staticmethod
    def validate_audit_trail(update: dict) -> bool:
        """Ensure all updates are traceable."""
        pass
```

---

---

## DSPy Role Specifications

### Overview

Tier 5 agents have different DSPy roles:
- **Recipient**: explainer (receives optimized prompts)
- **Hybrid**: feedback_learner (receives signals, runs optimization, distributes prompts)

---

## Explainer - DSPy Recipient Role

### Overview

Explainer is a **Recipient** agent that receives optimized prompts from feedback_learner
but does not generate training signals for optimization.

### DSPy Signatures

```python
class ExplanationSynthesisSignature(dspy.Signature):
    """Synthesize explanation from agent results."""

    agent_results: str = dspy.InputField(desc="Results from other agents")
    query_context: str = dspy.InputField(desc="Original query context")
    audience_level: str = dspy.InputField(desc="Target audience expertise level")

    explanation: str = dspy.OutputField(desc="Synthesized explanation")
    key_points: list = dspy.OutputField(desc="Key takeaway points")
    confidence_level: str = dspy.OutputField(desc="Explanation confidence")

class InsightExtractionSignature(dspy.Signature):
    """Extract insights from analysis results."""

    analysis_results: str = dspy.InputField(desc="Analysis output")
    business_context: str = dspy.InputField(desc="Business context")

    insights: list = dspy.OutputField(desc="Extracted insights")
    actionable_items: list = dspy.OutputField(desc="Actionable recommendations")
    supporting_evidence: list = dspy.OutputField(desc="Supporting data points")

class NarrativeStructureSignature(dspy.Signature):
    """Structure explanation as narrative."""

    raw_content: str = dspy.InputField(desc="Raw explanation content")
    narrative_style: str = dspy.InputField(desc="Desired narrative style")

    structured_narrative: str = dspy.OutputField(desc="Structured narrative")
    sections: list = dspy.OutputField(desc="Narrative sections")
    flow_coherence: float = dspy.OutputField(desc="Narrative coherence score")

class QueryRewriteForExplanationSignature(dspy.Signature):
    """Rewrite query for explanation generation."""

    original_query: str = dspy.InputField(desc="User's original query")
    agent_outputs: str = dspy.InputField(desc="Available agent outputs")

    rewritten_query: str = dspy.OutputField(desc="Query optimized for explanation")
    focus_areas: list = dspy.OutputField(desc="Key areas to explain")
```

### Recipient Configuration

```python
class ExplainerRecipient:
    """DSPy Recipient for Explainer agent."""

    dspy_type: Literal["recipient"] = "recipient"

    # Prompt optimization settings
    prompt_refresh_interval_hours: int = 24

    # Signatures that can receive optimized prompts
    optimizable_signatures: List[str] = [
        "ExplanationSynthesisSignature",
        "InsightExtractionSignature",
        "NarrativeStructureSignature",
        "QueryRewriteForExplanationSignature",
    ]

    def apply_optimized_prompt(
        self,
        signature_name: str,
        optimized_prompt: str,
        version: str,
    ) -> bool: ...

    def get_current_prompt_version(self, signature_name: str) -> str: ...

    def report_prompt_performance(
        self,
        signature_name: str,
        success_rate: float,
        latency_ms: float,
    ) -> None: ...
```

---

## Feedback Learner - DSPy Hybrid Role

### Overview

Feedback Learner is a **Hybrid** agent that:
1. Receives training signals from all Sender agents
2. Runs MIPROv2 prompt optimization
3. Distributes optimized prompts to Recipient agents
4. Generates its own training signals for meta-optimization

### Training Signal Contract (From All Agents)

```python
class AgentTrainingSignal(TypedDict):
    """Training signal emitted by any E2I agent for DSPy optimization."""

    # Signal metadata
    signal_id: str
    source_agent: str
    timestamp: str

    # Input context
    input_context: Dict[str, Any]

    # Agent output
    output: Dict[str, Any]

    # Ground truth / feedback
    user_feedback: Optional[Dict[str, Any]]
    outcome_observed: Optional[Dict[str, Any]]

    # Pre-computed metrics
    latency_ms: float
    token_count: Optional[int]

    # Phase information (for CognitiveRAG integration)
    cognitive_phase: Optional[str]
```

### Feedback Learner's Own Training Signal

```python
@dataclass
class FeedbackLearnerTrainingSignal:
    """Training signal for MIPROv2 optimization of feedback learning itself."""

    # Identity
    signal_id: str = ""
    session_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Input: Learning Cycle
    signals_processed: int = 0
    agents_represented: List[str] = field(default_factory=list)
    time_window_hours: float = 0.0

    # Pattern Detection
    patterns_detected: int = 0
    pattern_categories: Dict[str, int] = field(default_factory=dict)
    cross_agent_correlations: int = 0

    # Optimization
    optimizations_triggered: int = 0
    prompts_updated: int = 0
    improvement_magnitude: float = 0.0

    # Knowledge Updates
    knowledge_entries_created: int = 0
    rag_updates: int = 0

    # Outcome
    total_latency_ms: float = 0.0
    optimization_success: Optional[bool] = None
    user_satisfaction: Optional[float] = None

    def compute_reward(self) -> float:
        """
        Compute reward for meta-optimization.

        Weighting:
        - pattern_coverage: 0.25 (patterns / signals ratio)
        - optimization_impact: 0.30 (improvement magnitude)
        - knowledge_generation: 0.20 (entries + RAG updates)
        - efficiency: 0.15 (latency)
        - success_validation: 0.10 (if available)
        """
        ...
```

### DSPy Signatures

```python
class PatternDetectionSignature(dspy.Signature):
    """Detect patterns in collected training signals."""

    training_signals: str = dspy.InputField(desc="Batch of training signals")
    historical_patterns: str = dspy.InputField(desc="Previously detected patterns")
    cognitive_context: str = dspy.InputField(desc="CognitiveRAG context")

    new_patterns: list = dspy.OutputField(desc="Newly detected patterns")
    pattern_strengths: dict = dspy.OutputField(desc="Pattern confidence scores")
    cross_agent_insights: list = dspy.OutputField(desc="Cross-agent correlations")
    optimization_candidates: list = dspy.OutputField(desc="Signatures needing optimization")

class RecommendationGenerationSignature(dspy.Signature):
    """Generate optimization recommendations."""

    detected_patterns: str = dspy.InputField(desc="Detected patterns")
    agent_performance: str = dspy.InputField(desc="Agent performance metrics")
    optimization_history: str = dspy.InputField(desc="Previous optimization outcomes")

    recommendations: list = dspy.OutputField(desc="Prioritized optimization recommendations")
    expected_improvements: dict = dspy.OutputField(desc="Expected improvement per recommendation")
    risk_assessment: dict = dspy.OutputField(desc="Risk of each recommendation")
    implementation_order: list = dspy.OutputField(desc="Recommended implementation order")

class KnowledgeUpdateSignature(dspy.Signature):
    """Determine knowledge base updates."""

    learning_outcomes: str = dspy.InputField(desc="Outcomes from learning cycle")
    current_knowledge: str = dspy.InputField(desc="Current knowledge state")
    validation_results: str = dspy.InputField(desc="Validation results")

    updates_needed: list = dspy.OutputField(desc="Knowledge entries to update")
    new_entries: list = dspy.OutputField(desc="New knowledge entries to create")
    deprecations: list = dspy.OutputField(desc="Entries to deprecate")
    rag_sync_actions: list = dspy.OutputField(desc="RAG synchronization actions")

class LearningSummarySignature(dspy.Signature):
    """Generate learning cycle summary."""

    cycle_results: str = dspy.InputField(desc="Complete learning cycle results")
    before_metrics: str = dspy.InputField(desc="Pre-optimization metrics")
    after_metrics: str = dspy.InputField(desc="Post-optimization metrics")

    summary: str = dspy.OutputField(desc="Executive summary of learning cycle")
    key_improvements: list = dspy.OutputField(desc="Key improvements achieved")
    remaining_gaps: list = dspy.OutputField(desc="Gaps requiring attention")
    next_cycle_focus: list = dspy.OutputField(desc="Focus areas for next cycle")
```

### Hybrid Coordinator Contract

```python
class FeedbackLearnerHybridCoordinator:
    """DSPy Hybrid Coordinator for Feedback Learner agent."""

    dspy_type: Literal["hybrid"] = "hybrid"

    # Signal ingestion configuration
    signal_batch_size: int = 100
    signal_retention_hours: int = 168  # 1 week

    # Optimization configuration
    min_signals_for_optimization: int = 100
    optimization_interval_hours: int = 24
    min_improvement_threshold: float = 0.05

    # Distribution configuration
    recipient_agents: List[str] = [
        "health_score",
        "resource_optimizer",
        "explainer",
    ]

    async def ingest_signal(self, signal: AgentTrainingSignal) -> None:
        """Receive training signal from sender agent."""
        ...

    async def run_optimization_cycle(self) -> Dict[str, Any]:
        """Run MIPROv2 optimization on accumulated signals."""
        ...

    async def distribute_optimized_prompts(
        self,
        optimization_results: Dict[str, Any],
    ) -> Dict[str, bool]:
        """Distribute optimized prompts to recipient agents."""
        ...

    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get statistics on collected signals."""
        ...

    def get_optimization_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent optimization history."""
        ...
```

### MIPROv2 Integration

```python
class MIPROv2Integration:
    """MIPROv2 optimizer integration for prompt optimization."""

    # Optimization parameters
    num_candidates: int = 10
    num_iterations: int = 5
    metric: str = "reward"

    async def optimize_signature(
        self,
        signature_name: str,
        training_examples: List[Dict[str, Any]],
        current_prompt: str,
    ) -> Dict[str, Any]:
        """
        Optimize a DSPy signature using MIPROv2.

        Returns:
            {
                "optimized_prompt": str,
                "improvement": float,
                "new_version": str,
                "metrics": Dict[str, float],
            }
        """
        ...

    def validate_optimization(
        self,
        signature_name: str,
        optimized_prompt: str,
        holdout_examples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate optimization on holdout set."""
        ...
```

---

## Signal Flow

### Complete DSPy Signal Flow

```
Tier 2 Senders                Tier 3 Senders
(causal_impact,              (drift_monitor,
 gap_analyzer,                experiment_designer)
 heterogeneous_optimizer)            │
        │                            │
        └────────────┬───────────────┘
                     │
                     ▼
             Tier 4 Sender
         (prediction_synthesizer)
                     │
                     ▼
            ┌────────────────┐
            │feedback_learner│ (Hybrid)
            │                │
            │ ┌────────────┐ │
            │ │  Ingest    │ │
            │ │  Signals   │ │
            │ └─────┬──────┘ │
            │       ▼        │
            │ ┌────────────┐ │
            │ │   MIPROv2  │ │
            │ │Optimization│ │
            │ └─────┬──────┘ │
            │       ▼        │
            │ ┌────────────┐ │
            │ │ Distribute │ │
            │ │  Prompts   │ │
            │ └─────┬──────┘ │
            └───────┼────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   health_score  resource_    explainer
   (Recipient)   optimizer   (Recipient)
                (Recipient)
```

---

## Validation Tests

```bash
# Explainer contracts
pytest tests/integration/test_tier5_contracts.py::test_explainer_input
pytest tests/integration/test_tier5_contracts.py::test_explainer_output
pytest tests/integration/test_tier5_contracts.py::test_explainer_faithfulness

# Feedback Learner contracts
pytest tests/integration/test_tier5_contracts.py::test_feedback_learner_input
pytest tests/integration/test_tier5_contracts.py::test_feedback_learner_output
pytest tests/integration/test_tier5_contracts.py::test_weight_updates

# Async processing
pytest tests/integration/test_tier5_contracts.py::test_feedback_queue
pytest tests/integration/test_tier5_contracts.py::test_async_response

# Inter-agent communication
pytest tests/integration/test_tier5_contracts.py::test_tier5_handoff
pytest tests/integration/test_tier5_contracts.py::test_rag_integration
```

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-23 | V2: Added DSPy Role specifications for Tier 5 agents |
| 2025-12-08 | Initial creation |
