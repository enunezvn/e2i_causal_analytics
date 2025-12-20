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
| 2025-12-08 | Initial creation |
