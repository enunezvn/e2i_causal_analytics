# Tier 5 Contracts: Self-Improvement Agents

**Version**: 1.0
**Last Updated**: 2025-12-18
**Status**: Active

## Overview

This document defines integration contracts for **Tier 5: Self-Improvement** agents in the E2I Causal Analytics platform. These agents handle natural language explanation generation and self-improvement through feedback learning.

### Tier 5 Agents

| Agent | Type | Responsibility | Primary Methods |
|-------|------|----------------|-----------------|
| **Explainer** | Deep (Extended Reasoning) | Natural language explanation synthesis | Deep reasoning, narrative generation |
| **Feedback Learner** | Deep (Async) | Self-improvement from feedback | Pattern analysis, knowledge updates |

---

## 1. Shared Types

### 1.1 Common Enums

```python
from typing import Literal

# User expertise levels
ExpertiseLevel = Literal["executive", "analyst", "data_scientist"]

# Output formats
OutputFormat = Literal["narrative", "structured", "presentation", "brief"]

# Insight categories
InsightCategory = Literal["finding", "recommendation", "warning", "opportunity"]

# Actionability levels
ActionabilityLevel = Literal["immediate", "short_term", "long_term", "informational"]

# Feedback types
FeedbackType = Literal["rating", "correction", "outcome", "explicit"]

# Pattern types
PatternType = Literal["accuracy_issue", "latency_issue", "relevance_issue", "format_issue", "coverage_gap"]

# Severity levels
SeverityLevel = Literal["low", "medium", "high", "critical"]
```

### 1.2 Common Input Fields

All Tier 5 agents accept these common fields:

```python
from pydantic import BaseModel, Field
from typing import Optional

class Tier5CommonInput(BaseModel):
    """Common input fields for all Tier 5 agents"""
    query: Optional[str] = Field(None, description="User's natural language query (if applicable)")
```

### 1.3 Common Output Fields

All Tier 5 agents return these common fields:

```python
from typing import List

class Tier5CommonOutput(BaseModel):
    """Common output fields for all Tier 5 agents"""
    total_latency_ms: int = Field(..., description="Total processing time in milliseconds")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    timestamp: str = Field(..., description="ISO 8601 timestamp of completion")
    model_used: str = Field(..., description="Model used for deep reasoning")
```

---

## 2. Explainer Agent

**Agent Type**: Deep (Extended Reasoning)
**Primary Models**: Claude Sonnet 4, Claude Opus 4.5 (for complex synthesis)
**Latency**: Up to 45s

### 2.1 Input Contract

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ExplainerInput(BaseModel):
    """Input contract for Explainer Agent"""

    # Required fields
    query: str = Field(..., description="Original user query")
    analysis_results: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        description="Analysis results from upstream agents to explain"
    )

    # Optional fields
    user_expertise: ExpertiseLevel = Field(
        "analyst",
        description="Target audience expertise level"
    )
    output_format: OutputFormat = Field(
        "narrative",
        description="Desired explanation format"
    )
    focus_areas: Optional[List[str]] = Field(
        None,
        description="Specific areas to focus on in explanation"
    )

    # Model configuration
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is the causal effect of HCP engagement on TRx?",
                "analysis_results": [
                    {
                        "agent": "causal_impact",
                        "analysis_type": "causal_effect_estimation",
                        "key_findings": [
                            "ATE: 0.23 (95% CI: 0.18-0.28)",
                            "Statistically significant (p < 0.001)",
                            "Passed robustness checks"
                        ],
                        "ate_estimate": 0.23,
                        "confidence_interval": [0.18, 0.28],
                        "confidence": 0.85
                    }
                ],
                "user_expertise": "executive",
                "output_format": "narrative",
                "focus_areas": ["business_impact", "recommendations"]
            }
        }
    }
```

### 2.2 Output Contract

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Insight(BaseModel):
    """Extracted insight"""
    insight_id: str
    category: InsightCategory
    statement: str
    supporting_evidence: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    priority: int = Field(..., ge=1, le=5)
    actionability: ActionabilityLevel

class NarrativeSection(BaseModel):
    """Section of generated narrative"""
    section_type: str
    title: str
    content: str
    supporting_data: Optional[Dict[str, Any]] = None

class VisualSuggestion(BaseModel):
    """Suggestion for visual representation"""
    visual_type: str  # "chart", "table", "diagram"
    title: str
    data_source: str
    description: str

class ExplainerOutput(BaseModel):
    """Output contract for Explainer Agent"""

    # Core narrative
    executive_summary: str = Field(..., description="High-level summary (2-3 sentences)")
    detailed_explanation: str = Field(..., description="Full explanation narrative")

    # Structured outputs
    extracted_insights: List[Insight] = Field(..., description="Key insights extracted")
    narrative_sections: List[NarrativeSection] = Field(..., description="Structured sections")
    key_themes: List[str] = Field(..., description="Overarching themes")

    # Supplementary
    visual_suggestions: List[VisualSuggestion] = Field(
        default_factory=list,
        description="Suggested visualizations"
    )
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions"
    )
    related_analyses: List[str] = Field(
        default_factory=list,
        description="Related analyses to explore"
    )

    # Metadata
    assembly_latency_ms: int
    reasoning_latency_ms: int
    generation_latency_ms: int
    total_latency_ms: int
    model_used: str
    timestamp: str
    warnings: List[str] = Field(default_factory=list)
```

### 2.3 State Definition

```python
from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
import operator

class AnalysisContext(TypedDict):
    """Context from prior analysis"""
    source_agent: str
    analysis_type: str
    key_findings: List[str]
    data_summary: Dict[str, Any]
    confidence: float
    warnings: List[str]

class ExplainerState(TypedDict):
    """Complete LangGraph state for Explainer Agent"""

    # === INPUT ===
    query: str
    analysis_results: List[Dict[str, Any]]
    user_expertise: Literal["executive", "analyst", "data_scientist"]
    output_format: Literal["narrative", "structured", "presentation", "brief"]
    focus_areas: Optional[List[str]]

    # === CONTEXT ===
    analysis_context: Optional[List[Dict[str, Any]]]  # AnalysisContext
    user_context: Optional[Dict[str, Any]]
    conversation_history: Optional[List[Dict]]

    # === REASONING OUTPUTS ===
    extracted_insights: Optional[List[Dict[str, Any]]]  # Insight
    narrative_structure: Optional[List[str]]
    key_themes: Optional[List[str]]

    # === NARRATIVE OUTPUTS ===
    executive_summary: Optional[str]
    detailed_explanation: Optional[str]
    narrative_sections: Optional[List[Dict[str, Any]]]  # NarrativeSection

    # === SUPPLEMENTARY OUTPUTS ===
    visual_suggestions: Optional[List[Dict[str, Any]]]  # VisualSuggestion
    follow_up_questions: Optional[List[str]]
    related_analyses: Optional[List[str]]

    # === EXECUTION METADATA ===
    assembly_latency_ms: int
    reasoning_latency_ms: int
    generation_latency_ms: int
    total_latency_ms: int
    model_used: str

    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal["pending", "assembling", "reasoning", "generating", "completed", "failed"]
```

### 2.4 Handoff Format

```yaml
explainer_handoff:
  agent: explainer
  analysis_type: explanation
  status: completed

  executive_summary: |
    Increasing HCP engagement frequency by 1 visit per month causes a 23% increase
    in prescription rates (95% CI: 18-28%). This effect is statistically significant
    and robust across multiple validation tests. We recommend prioritizing engagement
    in high-potential territories, with expected ROI of 2.8x.

  key_insights:
    - insight_id: "1"
      category: finding
      statement: "HCP engagement has a strong causal effect on TRx (ATE: 0.23)"
      confidence: 0.85
      priority: 1
      actionability: immediate
    - insight_id: "2"
      category: recommendation
      statement: "Increase engagement frequency in high-potential territories"
      confidence: 0.80
      priority: 1
      actionability: immediate
    - insight_id: "3"
      category: warning
      statement: "Effect may vary by HCP specialty - consider segment analysis"
      confidence: 0.70
      priority: 2
      actionability: short_term

  key_themes:
    - "Causal relationship between engagement and outcomes"
    - "Actionable opportunities for ROI improvement"
    - "Need for segment-specific strategies"

  visual_suggestions:
    - visual_type: "chart"
      title: "Causal Effect Estimate with Confidence Intervals"
      data_source: "causal_impact.ate_estimate"
      description: "Bar chart showing ATE with 95% CI error bars"
    - visual_type: "table"
      title: "Robustness Check Results"
      data_source: "causal_impact.refutation_results"
      description: "Table showing placebo test, random cause test results"

  follow_up_questions:
    - "Which HCP segments respond best to increased engagement?"
    - "What is the optimal engagement frequency by territory?"
    - "How long does the effect persist after engagement?"

  related_analyses:
    - "heterogeneous_optimizer: Segment-specific effect analysis"
    - "gap_analyzer: Identify territories with largest opportunity"
    - "experiment_designer: Design A/B test to validate"

  model_used: "claude-sonnet-4-20250514"
  reasoning_latency_ms: 12500
  total_latency_ms: 18200

  warnings:
    - "Analysis based on observational data - experimental validation recommended"

  requires_further_analysis: false
  suggested_next_agent: null
```

---

## 3. Feedback Learner Agent

**Agent Type**: Deep (Async)
**Primary Models**: Claude Sonnet 4, Claude Opus 4.5 (for complex pattern analysis)
**Latency**: No real-time constraint (async processing)

### 3.1 Input Contract

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class FeedbackLearnerInput(BaseModel):
    """Input contract for Feedback Learner Agent"""

    # Required fields
    batch_id: str = Field(..., description="Unique identifier for this feedback batch")
    time_range_start: str = Field(..., description="ISO 8601 start timestamp")
    time_range_end: str = Field(..., description="ISO 8601 end timestamp")

    # Optional fields
    focus_agents: Optional[List[str]] = Field(
        None,
        description="Specific agents to analyze, or None for all"
    )

    # Model configuration
    model_config = {
        "json_schema_extra": {
            "example": {
                "batch_id": "batch_2025_12_18",
                "time_range_start": "2025-12-11T00:00:00Z",
                "time_range_end": "2025-12-18T00:00:00Z",
                "focus_agents": ["causal_impact", "gap_analyzer", "prediction_synthesizer"]
            }
        }
    }
```

### 3.2 Output Contract

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class DetectedPattern(BaseModel):
    """Detected pattern in feedback"""
    pattern_id: str
    pattern_type: PatternType
    description: str
    frequency: int = Field(..., ge=1)
    severity: SeverityLevel
    affected_agents: List[str]
    example_feedback_ids: List[str]
    root_cause_hypothesis: str

class LearningRecommendation(BaseModel):
    """Recommendation for improvement"""
    recommendation_id: str
    category: Literal["prompt_update", "model_retrain", "data_update", "config_change", "new_capability"]
    description: str
    affected_agents: List[str]
    expected_impact: str
    implementation_effort: Literal["low", "medium", "high"]
    priority: int = Field(..., ge=1, le=5)
    proposed_change: Optional[str] = None

class KnowledgeUpdate(BaseModel):
    """Update to knowledge base"""
    update_id: str
    knowledge_type: Literal["experiment", "baseline", "agent_config", "prompt", "threshold"]
    key: str
    old_value: Any
    new_value: Any
    justification: str
    effective_date: str

class FeedbackLearnerOutput(BaseModel):
    """Output contract for Feedback Learner Agent"""

    # Feedback summary
    feedback_summary: Dict[str, Any] = Field(..., description="Summary of feedback collected")

    # Pattern analysis
    detected_patterns: List[DetectedPattern] = Field(..., description="Detected patterns")
    pattern_clusters: Dict[str, List[str]] = Field(..., description="Clustered patterns")

    # Learning outputs
    learning_recommendations: List[LearningRecommendation] = Field(
        ...,
        description="Recommended improvements"
    )
    priority_improvements: List[str] = Field(..., description="Top priority improvements")

    # Knowledge updates
    proposed_updates: List[KnowledgeUpdate] = Field(..., description="Proposed knowledge updates")
    applied_updates: List[str] = Field(..., description="Update IDs that were applied")

    # Summary
    learning_summary: str = Field(..., description="Human-readable summary of learning")
    metrics_before: Optional[Dict[str, float]] = Field(None, description="Metrics before updates")
    metrics_after: Optional[Dict[str, float]] = Field(None, description="Metrics after updates")

    # Metadata
    collection_latency_ms: int
    analysis_latency_ms: int
    extraction_latency_ms: int
    update_latency_ms: int
    total_latency_ms: int
    model_used: str
    timestamp: str
    warnings: List[str] = Field(default_factory=list)
```

### 3.3 State Definition

```python
from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
import operator

class FeedbackItem(TypedDict):
    """Individual feedback item"""
    feedback_id: str
    timestamp: str
    feedback_type: Literal["rating", "correction", "outcome", "explicit"]
    source_agent: str
    query: str
    agent_response: str
    user_feedback: Any
    metadata: Dict[str, Any]

class FeedbackLearnerState(TypedDict):
    """Complete LangGraph state for Feedback Learner Agent"""

    # === INPUT ===
    batch_id: str
    time_range_start: str
    time_range_end: str
    focus_agents: Optional[List[str]]

    # === FEEDBACK DATA ===
    feedback_items: Optional[List[Dict[str, Any]]]  # FeedbackItem
    feedback_summary: Optional[Dict[str, Any]]

    # === PATTERN ANALYSIS ===
    detected_patterns: Optional[List[Dict[str, Any]]]  # DetectedPattern
    pattern_clusters: Optional[Dict[str, List[str]]]

    # === LEARNING OUTPUTS ===
    learning_recommendations: Optional[List[Dict[str, Any]]]  # LearningRecommendation
    priority_improvements: Optional[List[str]]

    # === KNOWLEDGE UPDATES ===
    proposed_updates: Optional[List[Dict[str, Any]]]  # KnowledgeUpdate
    applied_updates: Optional[List[str]]

    # === SUMMARY ===
    learning_summary: Optional[str]
    metrics_before: Optional[Dict[str, float]]
    metrics_after: Optional[Dict[str, float]]

    # === EXECUTION METADATA ===
    collection_latency_ms: int
    analysis_latency_ms: int
    extraction_latency_ms: int
    update_latency_ms: int
    total_latency_ms: int
    model_used: str

    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal["pending", "collecting", "analyzing", "extracting", "updating", "completed", "failed"]
```

### 3.4 Handoff Format

```yaml
feedback_learner_handoff:
  agent: feedback_learner
  analysis_type: feedback_learning
  status: completed
  batch_id: "batch_2025_12_18"

  feedback_summary:
    total_count: 247
    by_type:
      rating: 185
      correction: 42
      outcome: 20
    by_agent:
      causal_impact: 95
      gap_analyzer: 78
      prediction_synthesizer: 74
    average_rating: 4.2

  detected_patterns:
    - pattern_id: "P1"
      pattern_type: accuracy_issue
      description: "Causal Impact agent overestimates effects in small sample scenarios"
      frequency: 12
      severity: high
      affected_agents: ["causal_impact"]
      root_cause_hypothesis: "Insufficient sample size warnings not prominent enough"
    - pattern_id: "P2"
      pattern_type: format_issue
      description: "Gap Analyzer output too technical for executive users"
      frequency: 8
      severity: medium
      affected_agents: ["gap_analyzer"]
      root_cause_hypothesis: "Lack of audience adaptation in output generation"
    - pattern_id: "P3"
      pattern_type: latency_issue
      description: "Prediction Synthesizer exceeds target latency on large feature sets"
      frequency: 6
      severity: medium
      affected_agents: ["prediction_synthesizer"]
      root_cause_hypothesis: "No timeout or feature selection optimization"

  learning_recommendations:
    - recommendation_id: "R1"
      category: prompt_update
      description: "Add prominent sample size warnings to Causal Impact output template"
      affected_agents: ["causal_impact"]
      expected_impact: "Reduce overconfidence in small-sample estimates"
      implementation_effort: low
      priority: 1
      proposed_change: "Update prompt to include: 'WARNING: Sample size < 100. Results may be unstable.'"
    - recommendation_id: "R2"
      category: config_change
      description: "Add executive-mode output formatting to Gap Analyzer"
      affected_agents: ["gap_analyzer"]
      expected_impact: "Improve executive user satisfaction"
      implementation_effort: medium
      priority: 2
    - recommendation_id: "R3"
      category: model_retrain
      description: "Retrain Prediction Synthesizer models with feature selection"
      affected_agents: ["prediction_synthesizer"]
      expected_impact: "Reduce latency by 30%"
      implementation_effort: high
      priority: 3

  priority_improvements:
    - "R1: Add sample size warnings to Causal Impact"
    - "R2: Add executive formatting to Gap Analyzer"
    - "R3: Optimize Prediction Synthesizer feature selection"

  proposed_updates:
    - update_id: "U1"
      knowledge_type: threshold
      key: "causal_impact.min_sample_size_warning"
      old_value: null
      new_value: 100
      justification: "Pattern P1 shows overconfidence below n=100"
      effective_date: "2025-12-19T00:00:00Z"
    - update_id: "U2"
      knowledge_type: prompt
      key: "gap_analyzer.executive_template"
      old_value: null
      new_value: "Focus on ROI and actionable opportunities. Minimize statistical jargon."
      justification: "Pattern P2 shows executive users prefer simplified output"
      effective_date: "2025-12-19T00:00:00Z"

  applied_updates:
    - "U1"
    - "U2"

  learning_summary: |
    Analyzed 247 feedback items from 7-day period. Identified 3 high-priority patterns
    affecting causal_impact, gap_analyzer, and prediction_synthesizer agents.
    Applied 2 immediate fixes (sample size warnings, executive formatting).
    Recommended model retraining for prediction_synthesizer to address latency issues.

  metrics_before:
    average_rating: 4.2
    causal_impact_accuracy: 0.78
    gap_analyzer_satisfaction: 3.8
    prediction_synthesizer_latency_p95: 18500

  metrics_after:
    average_rating: null  # Will measure in next batch
    causal_impact_accuracy: null
    gap_analyzer_satisfaction: null
    prediction_synthesizer_latency_p95: null

  model_used: "claude-sonnet-4-20250514"
  analysis_latency_ms: 45200
  total_latency_ms: 62800

  warnings:
    - "Metrics_after will be available after 7-day observation period"
    - "Update U2 requires manual review before production deployment"

  requires_further_analysis: false
  suggested_next_agent: null
```

---

## 4. Inter-Agent Communication

### 4.1 Orchestrator → Tier 5 Dispatch

**Note**: Tier 5 agents are typically NOT dispatched directly by the orchestrator in real-time. They are:
- **Explainer**: Dispatched after primary analysis completes
- **Feedback Learner**: Runs asynchronously on schedule (nightly/weekly batches)

However, they can receive dispatches via the standard `AgentDispatchRequest` (see `orchestrator-contracts.md`) when triggered.

### 4.2 Tier 5 → Orchestrator Response

Both agents return via the standard `AgentDispatchResponse` when triggered.

### 4.3 Tier 5 Inter-Agent Handoffs

#### Any Agent → Explainer

Any agent can trigger Explainer for explanation:

```python
# Any agent sets in response
next_agent = "explainer"
handoff_context = {
    "upstream_agent": state["agent_name"],
    "analysis_results": [
        {
            "agent": state["agent_name"],
            "analysis_type": state["analysis_type"],
            "key_findings": state["key_findings"],
            # ... rest of analysis output
        }
    ],
    "user_expertise": state.get("user_expertise", "analyst"),
    "reason": "Generate stakeholder-friendly explanation"
}
```

#### Feedback Learner → System (Async)

Feedback Learner typically runs asynchronously and updates knowledge stores directly rather than triggering other agents. However, it may generate reports for human review:

```python
# Feedback Learner outputs
learning_report = {
    "batch_id": state["batch_id"],
    "patterns_detected": len(state["detected_patterns"]),
    "recommendations": state["learning_recommendations"],
    "updates_applied": state["applied_updates"],
    "requires_human_review": True if high_severity_patterns else False
}
```

---

## 5. Validation Rules

### 5.1 Input Validation

**Explainer:**
```python
def validate_explainer_input(state: ExplainerState) -> List[str]:
    """Validation for Explainer inputs"""
    errors = []

    # Required fields
    if not state.get("query"):
        errors.append("query is required")
    if not state.get("analysis_results") or len(state["analysis_results"]) == 0:
        errors.append("At least one analysis result is required")

    # User expertise validation
    valid_expertise = ["executive", "analyst", "data_scientist"]
    if state.get("user_expertise") and state["user_expertise"] not in valid_expertise:
        errors.append(f"user_expertise must be one of {valid_expertise}")

    # Output format validation
    valid_formats = ["narrative", "structured", "presentation", "brief"]
    if state.get("output_format") and state["output_format"] not in valid_formats:
        errors.append(f"output_format must be one of {valid_formats}")

    # Analysis results structure
    for i, result in enumerate(state.get("analysis_results", [])):
        if "agent" not in result:
            errors.append(f"analysis_results[{i}] missing 'agent' field")
        if "analysis_type" not in result:
            errors.append(f"analysis_results[{i}] missing 'analysis_type' field")

    return errors
```

**Feedback Learner:**
```python
def validate_feedback_learner_input(state: FeedbackLearnerState) -> List[str]:
    """Validation for Feedback Learner inputs"""
    errors = []

    # Required fields
    if not state.get("batch_id"):
        errors.append("batch_id is required")
    if not state.get("time_range_start"):
        errors.append("time_range_start is required")
    if not state.get("time_range_end"):
        errors.append("time_range_end is required")

    # Time range validation
    try:
        from datetime import datetime
        start = datetime.fromisoformat(state["time_range_start"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(state["time_range_end"].replace("Z", "+00:00"))
        if start >= end:
            errors.append("time_range_start must be before time_range_end")

        # Reasonable time range (not more than 30 days)
        delta = (end - start).days
        if delta > 30:
            errors.append("time_range cannot exceed 30 days")
    except (ValueError, KeyError):
        errors.append("Invalid time_range format (use ISO 8601)")

    return errors
```

### 5.2 Output Validation

```python
def validate_tier5_output(output: Dict[str, Any], agent_name: str) -> List[str]:
    """Validate Tier 5 agent outputs"""
    errors = []

    # Common required fields
    required = ["total_latency_ms", "warnings", "timestamp", "model_used"]
    for field in required:
        if field not in output:
            errors.append(f"{agent_name} output missing required field: {field}")

    # Agent-specific validation
    if agent_name == "explainer":
        if "executive_summary" not in output or not output["executive_summary"]:
            errors.append("executive_summary is required and cannot be empty")
        if "detailed_explanation" not in output or not output["detailed_explanation"]:
            errors.append("detailed_explanation is required and cannot be empty")
        if "extracted_insights" in output:
            for insight in output["extracted_insights"]:
                if "confidence" in insight:
                    if not (0.0 <= insight["confidence"] <= 1.0):
                        errors.append(f"Insight confidence must be between 0.0 and 1.0")

    elif agent_name == "feedback_learner":
        if "feedback_summary" not in output:
            errors.append("feedback_summary is required")
        if "detected_patterns" not in output:
            errors.append("detected_patterns is required")
        if "learning_recommendations" not in output:
            errors.append("learning_recommendations is required")

    return errors
```

---

## 6. Error Handling

### 6.1 Explainer Error Patterns

**No Analysis Results:**
```python
if not state.get("analysis_results") or len(state["analysis_results"]) == 0:
    return {
        **state,
        "errors": [{"node": "input_validation", "error": "No analysis results to explain"}],
        "status": "failed"
    }
```

**Reasoning Timeout (Fallback):**
```python
try:
    response = await asyncio.wait_for(
        self.llm.ainvoke(reasoning_prompt),
        timeout=120
    )
    model_used = "claude-sonnet-4-20250514"
except asyncio.TimeoutError:
    # Fallback to simpler model
    response = await self.fallback_llm.ainvoke(simplified_prompt)
    model_used = "claude-haiku-4-20250414 (fallback)"
    state = {**state, "warnings": ["Used fallback model due to timeout"]}
```

**Incomplete Reasoning (Graceful Degradation):**
```python
# If reasoning fails, still provide basic explanation
if not extracted_insights:
    return {
        **state,
        "extracted_insights": [],
        "executive_summary": "Analysis completed. See detailed results for full information.",
        "warnings": ["Could not extract structured insights - providing basic summary"],
        "status": "generating"  # Continue with generation
    }
```

### 6.2 Feedback Learner Error Patterns

**No Feedback Available:**
```python
if not feedback_items:
    return {
        **state,
        "feedback_items": [],
        "feedback_summary": {"total_count": 0},
        "detected_patterns": [],
        "learning_recommendations": [],
        "learning_summary": "No feedback available for this time period",
        "status": "completed"  # Not an error
    }
```

**Pattern Analysis Failure (Non-Fatal):**
```python
try:
    patterns = await pattern_analyzer.analyze(feedback_items)
except Exception as e:
    # Continue without pattern analysis
    return {
        **state,
        "detected_patterns": [],
        "warnings": [f"Pattern analysis failed: {str(e)}"],
        "status": "extracting"  # Continue to next stage
    }
```

**Knowledge Update Failure:**
```python
# Track which updates succeeded/failed
applied_updates = []
for update in proposed_updates:
    try:
        await knowledge_store.apply_update(update)
        applied_updates.append(update["update_id"])
    except Exception as e:
        state = {
            **state,
            "warnings": state.get("warnings", []) + [f"Update {update['update_id']} failed: {str(e)}"]
        }

return {
    **state,
    "applied_updates": applied_updates,
    "status": "completed"  # Partial success is OK
}
```

---

## 7. Performance Requirements

| Agent | Target Latency | Max Latency | Throughput | LLM Calls |
|-------|----------------|-------------|------------|-----------|
| **Explainer** | 20-30s | 45s | 2 queries/min | 2 (reasoning + generation) |
| **Feedback Learner** | N/A (async) | No limit | 1 batch/day | 1 (pattern analysis) |

### 7.1 Latency Breakdown

**Explainer:**
- Context assembly: 1-2s
- Deep reasoning: 10-15s
- Narrative generation: 8-12s
- Total: 20-30s

**Feedback Learner (Async):**
- Feedback collection: 5-10s
- Pattern analysis (deep): 30-60s
- Learning extraction: 10-20s
- Knowledge updates: 5-10s
- Total: 50-100s (no real-time constraint)

---

## 8. Testing Requirements

### 8.1 Unit Tests

**Explainer:**
- Context extraction from analysis results
- Insight extraction and categorization
- Narrative structure planning
- Audience adaptation (executive vs technical)
- Visual suggestion generation

**Feedback Learner:**
- Feedback collection from multiple sources
- Pattern detection algorithms
- Learning recommendation generation
- Knowledge update application

### 8.2 Integration Tests

```python
async def test_tier5_agent_integration(agent_name: str):
    """Test full agent execution"""

    # Setup
    state = create_test_input(agent_name)
    graph = build_agent_graph(agent_name)

    # Execute
    result = await graph.ainvoke(state)

    # Validate output contract
    errors = validate_tier5_output(result, agent_name)
    assert len(errors) == 0, f"Output validation failed: {errors}"

    # Check required fields
    assert result["status"] == "completed"
    assert result["total_latency_ms"] > 0
    assert result["timestamp"]
    assert result["model_used"]

    # Agent-specific checks
    if agent_name == "explainer":
        assert result["executive_summary"]
        assert result["detailed_explanation"]
        assert len(result["extracted_insights"]) > 0
    elif agent_name == "feedback_learner":
        assert result["feedback_summary"]
        assert isinstance(result["detected_patterns"], list)
        assert isinstance(result["learning_recommendations"], list)
```

---

## 9. Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-12-18 | Initial Tier 5 contracts | Claude |

---

## 10. DSPy Signal Contracts

### 10.0 DSPy Role Assignments

| Agent | DSPy Role | Primary Signature | Behavior |
|-------|-----------|-------------------|----------|
| **Explainer** | Recipient | N/A (consumes optimized prompts) | Uses QueryRewriteSignature for query clarification |
| **Feedback Learner** | Hybrid | All 14 Cognitive RAG DSPy components (11 signatures + 4 modules) + 36 agent signatures | Collects training signals AND consumes/distributes prompts |

### 10.0.1 DSPy Recipient Mixin (Explainer)

```python
from abc import ABC, abstractmethod
from typing import Dict, Optional

class DSPyRecipientMixin(ABC):
    """Mixin for agents that consume DSPy-optimized prompts (Recipients)."""

    _optimized_prompts: Dict[str, str] = {}
    _last_update_timestamp: Optional[str] = None

    async def load_optimized_prompts(self, prompts: Dict[str, str]) -> None:
        """Load optimized prompts from Feedback Learner via Hub."""
        from datetime import datetime
        self._optimized_prompts = prompts
        self._last_update_timestamp = datetime.utcnow().isoformat()

    def get_optimized_prompt(self, signature_name: str, default: str = "") -> str:
        """Get optimized prompt for a signature, falling back to default."""
        return self._optimized_prompts.get(signature_name, default)

    def has_optimized_prompts(self) -> bool:
        """Check if agent has loaded optimized prompts."""
        return len(self._optimized_prompts) > 0
```

### 10.0.2 DSPy Hybrid Mixin (Feedback Learner)

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime

class DSPyHybridMixin(ABC):
    """Mixin for agents that both generate AND consume DSPy signals (Hybrids).

    Feedback Learner is the primary Hybrid agent - it:
    1. Collects training signals from all Sender agents
    2. Coordinates MIPROv2 optimization cycles
    3. Distributes optimized prompts to Recipient agents
    4. Itself uses optimized prompts for pattern analysis
    """

    _signals_buffer: List["TrainingSignal"] = []
    _optimized_prompts: Dict[str, str] = {}

    def collect_training_signal(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        quality_score: float,
        signature_name: str,
        source_agent: str,
        confidence: float = 0.8,
        latency_ms: int = 0,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "TrainingSignal":
        """Collect training signal from any agent execution."""
        import uuid
        signal = TrainingSignal(
            signal_id=f"{source_agent}_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.utcnow(),
            source_agent=source_agent,
            source_type="sender",  # Collected from Sender agents
            signature_name=signature_name,
            input_data=input_data,
            output_data=output_data,
            ground_truth=None,
            quality_score=quality_score,
            confidence=confidence,
            latency_ms=latency_ms,
            user_feedback=None,
            session_id=session_id or f"session_{uuid.uuid4().hex[:8]}",
            metadata=metadata or {},
        )
        self._signals_buffer.append(signal)
        return signal

    def get_collected_signals(self) -> List["TrainingSignal"]:
        """Get all collected training signals."""
        return self._signals_buffer.copy()

    async def trigger_optimization_cycle(
        self,
        target_signatures: Optional[List[str]] = None,
        optimizer: Literal["miprov2", "bootstrap_fewshot", "copro"] = "miprov2",
    ) -> "OptimizationResult":
        """Trigger MIPROv2 optimization cycle for specified signatures."""
        # Implemented by FeedbackLearner
        raise NotImplementedError("Subclass must implement optimization cycle")

    async def distribute_optimized_prompts(
        self,
        prompts: Dict[str, str],
        recipient_agents: List[str],
    ) -> Dict[str, bool]:
        """Distribute optimized prompts to Recipient agents."""
        # Implemented by FeedbackLearner
        raise NotImplementedError("Subclass must implement prompt distribution")

    def get_optimized_prompt(self, signature_name: str, default: str = "") -> str:
        """Get optimized prompt for own use."""
        return self._optimized_prompts.get(signature_name, default)
```

### 10.0.3 Implementation Requirements

**Explainer Agent** (Recipient):
- MUST inherit from `DSPyRecipientMixin`
- MUST load optimized prompts at initialization
- SHOULD use `QueryRewriteSignature` optimized prompt for query clarification
- Prompt update frequency: Daily (via Feedback Learner distribution)

**Feedback Learner Agent** (Hybrid):
- MUST inherit from `DSPyHybridMixin`
- MUST implement `trigger_optimization_cycle()` using MIPROv2
- MUST implement `distribute_optimized_prompts()` to all Recipients
- MUST maintain signal buffer with configurable size (default: 10,000 signals)
- Optimization trigger conditions:
  - Signal count >= 100 (min_signals_for_optimization)
  - Time since last optimization >= 24h (optimization_interval_hours)
  - Any signature quality_score < 0.6 (min_signal_quality threshold)

---

## 10.1 DSPy Integration Contracts

The Feedback Learner agent serves as the central hub for DSPy optimization across the E2I system.

### 10.1 DSPy Signal Contract

```python
from typing import TypedDict, Optional, Dict, Any, Literal, List
from pydantic import BaseModel, Field

class DSPySignal(TypedDict):
    """Signal emitted by DSPy-powered components"""
    signal_id: str
    timestamp: str
    source_component: str  # Agent or RAG phase
    signature_name: str    # Which DSPy signature
    prediction: Any        # What was predicted
    ground_truth: Optional[Any]  # Actual outcome (if available)
    confidence: float      # Model confidence
    latency_ms: int        # Execution time
    user_feedback: Optional[int]  # 1-5 rating if available
    metadata: Dict[str, Any]

class DSPySignalBatch(BaseModel):
    """Batch of signals for optimization"""
    batch_id: str = Field(..., description="Unique batch identifier")
    signals: List[DSPySignal] = Field(..., description="Signals in this batch")
    source_phase: Literal["summarizer", "investigator", "agent", "reflector"] = Field(
        ..., description="Cognitive RAG phase"
    )
    time_range_start: str = Field(..., description="Batch start time (ISO 8601)")
    time_range_end: str = Field(..., description="Batch end time (ISO 8601)")
```

### 10.2 Signal Aggregation Contract

```python
class SignalAggregation(TypedDict):
    """Aggregated signals for pattern detection"""
    signature_name: str
    signal_count: int
    avg_confidence: float
    avg_latency_ms: float
    avg_user_rating: Optional[float]
    accuracy: Optional[float]  # If ground truth available
    trend: Literal["improving", "stable", "degrading"]
    recommended_action: Literal["optimize", "monitor", "none"]

class AggregationResult(BaseModel):
    """Result of signal aggregation"""
    aggregations: List[SignalAggregation] = Field(..., description="Aggregated signals by signature")
    signatures_requiring_optimization: List[str] = Field(..., description="Signatures needing recompilation")
    optimization_priority: Dict[str, int] = Field(..., description="Priority order for optimization")
```

### 10.3 DSPy Optimization Request/Response

```python
class DSPyOptimizationRequest(BaseModel):
    """Request to optimize a DSPy signature"""
    signature_name: str = Field(..., description="Signature to optimize")
    optimizer: Literal["miprov2", "bootstrap_fewshot", "copro"] = Field(
        default="miprov2",
        description="Optimizer to use"
    )
    training_examples: List[Dict[str, Any]] = Field(..., description="Training examples")
    metric: str = Field(..., description="Optimization metric name")
    max_iterations: int = Field(default=100, ge=10, le=1000, description="Max optimization iterations")
    early_stopping_patience: int = Field(default=10, ge=5, le=50, description="Early stopping patience")

class DSPyOptimizationResult(BaseModel):
    """Result of DSPy signature optimization"""
    signature_name: str = Field(..., description="Optimized signature name")
    optimizer_used: Literal["miprov2", "bootstrap_fewshot", "copro"] = Field(..., description="Optimizer used")
    before_metric: float = Field(..., ge=0.0, le=1.0, description="Metric before optimization")
    after_metric: float = Field(..., ge=0.0, le=1.0, description="Metric after optimization")
    improvement_pct: float = Field(..., description="Improvement percentage")
    iterations: int = Field(..., description="Iterations run")
    converged: bool = Field(..., description="Whether optimization converged")
    new_prompt: Optional[str] = Field(None, description="New optimized prompt (if applicable)")
    optimization_time_ms: int = Field(..., description="Total optimization time")
```

### 10.4 Optimizable Agents Registry

```python
OPTIMIZABLE_AGENTS = {
    "causal_impact": {
        "signatures": ["EffectEstimationSignature", "RobustnessCheckSignature"],
        "optimizers": ["miprov2", "bootstrap_fewshot"],
        "metrics": ["effect_accuracy", "assumption_validity"],
        "optimization_schedule": "daily"
    },
    "gap_analyzer": {
        "signatures": ["OpportunityDetectionSignature", "PriorityRankingSignature"],
        "optimizers": ["miprov2"],
        "metrics": ["gap_precision", "priority_correlation"],
        "optimization_schedule": "daily"
    },
    "heterogeneous_optimizer": {
        "signatures": ["SegmentAnalysisSignature", "CATEEstimationSignature"],
        "optimizers": ["miprov2"],
        "metrics": ["segment_accuracy", "cate_mse"],
        "optimization_schedule": "weekly"
    },
    "drift_monitor": {
        "signatures": ["DriftDetectionSignature", "AlertPrioritySignature"],
        "optimizers": ["bootstrap_fewshot"],
        "metrics": ["detection_recall", "false_positive_rate"],
        "optimization_schedule": "hourly"
    },
    "experiment_designer": {
        "signatures": ["DesignSpecSignature", "PowerAnalysisSignature"],
        "optimizers": ["miprov2"],
        "metrics": ["design_validity", "power_accuracy"],
        "optimization_schedule": "weekly"
    },
    "health_score": {
        "signatures": ["HealthAssessmentSignature", "ComponentScoreSignature"],
        "optimizers": ["bootstrap_fewshot"],
        "metrics": ["health_accuracy", "component_correlation"],
        "optimization_schedule": "hourly"
    },
    "prediction_synthesizer": {
        "signatures": ["PredictionAggregationSignature", "ConfidenceEstimationSignature"],
        "optimizers": ["miprov2"],
        "metrics": ["prediction_mape", "calibration_score"],
        "optimization_schedule": "daily"
    },
    "resource_optimizer": {
        "signatures": ["ResourceAllocationSignature", "ConstraintSatisfactionSignature"],
        "optimizers": ["miprov2"],
        "metrics": ["allocation_optimality", "constraint_satisfaction"],
        "optimization_schedule": "daily"
    },
    "explainer": {
        "signatures": ["ExplanationGenerationSignature", "VisualizationSelectionSignature"],
        "optimizers": ["copro"],
        "metrics": ["explanation_clarity", "user_satisfaction"],
        "optimization_schedule": "daily"
    },
    "cognitive_rag": {
        "signatures": [
            "QueryRewriteSignature", "EntityExtractionSignature",
            "IntentClassificationSignature", "InvestigationPlanSignature",
            "HopDecisionSignature", "EvidenceRelevanceSignature",
            "EvidenceSynthesisSignature", "AgentRoutingSignature",
            "VisualizationConfigSignature", "MemoryWorthinessSignature",
            "ProcedureLearningSignature"
        ],
        "optimizers": ["miprov2", "bootstrap_fewshot", "copro"],
        "metrics": ["retrieval_relevance", "hop_efficiency", "response_quality"],
        "optimization_schedule": "continuous"
    }
}
```

### 10.5 Optimization Schedule Contract

```python
class OptimizationSchedule(BaseModel):
    """Schedule for DSPy optimization cycles"""
    schedule_type: Literal["continuous", "hourly", "daily", "weekly"] = Field(
        ..., description="Schedule frequency"
    )
    trigger: str = Field(..., description="Trigger expression (cron or event)")
    signatures: List[str] = Field(..., description="Signatures to optimize")
    optimizer: Literal["miprov2", "bootstrap_fewshot", "copro"] = Field(
        ..., description="Default optimizer"
    )
    min_examples: int = Field(default=50, ge=10, description="Minimum examples before optimization")
    skip_if_metric_above: float = Field(default=0.95, ge=0.5, le=1.0, description="Skip if metric already high")

OPTIMIZATION_SCHEDULES = {
    "continuous": OptimizationSchedule(
        schedule_type="continuous",
        trigger="session_end",
        signatures=["QueryRewriteSignature", "EntityExtractionSignature"],
        optimizer="bootstrap_fewshot",
        min_examples=10,
        skip_if_metric_above=0.98
    ),
    "hourly": OptimizationSchedule(
        schedule_type="hourly",
        trigger="cron(0 * 9-17 * * *)",
        signatures=["EvidenceRelevanceSignature", "AgentRoutingSignature", "DriftDetectionSignature"],
        optimizer="bootstrap_fewshot",
        min_examples=25,
        skip_if_metric_above=0.95
    ),
    "daily": OptimizationSchedule(
        schedule_type="daily",
        trigger="cron(0 2 * * *)",
        signatures=["all_miprov2_signatures"],
        optimizer="miprov2",
        min_examples=100,
        skip_if_metric_above=0.95
    ),
    "weekly": OptimizationSchedule(
        schedule_type="weekly",
        trigger="cron(0 3 * * 0)",
        signatures=["all_signatures"],
        optimizer="miprov2",
        min_examples=500,
        skip_if_metric_above=0.98
    )
}
```

### 10.6 Optimization Metrics Target

```python
OPTIMIZATION_TARGETS = {
    # Agent-specific targets
    "causal_impact": {"effect_accuracy": 0.90, "assumption_validity": 0.90},
    "gap_analyzer": {"gap_precision": 0.85, "priority_correlation": 0.85},
    "heterogeneous_optimizer": {"segment_accuracy": 0.85, "cate_mse": 0.15},
    "drift_monitor": {"detection_recall": 0.95, "false_positive_rate": 0.05},
    "experiment_designer": {"design_validity": 0.90, "power_accuracy": 0.90},
    "health_score": {"health_accuracy": 0.90, "component_correlation": 0.90},
    "prediction_synthesizer": {"prediction_mape": 0.10, "calibration_score": 0.85},
    "resource_optimizer": {"allocation_optimality": 0.85, "constraint_satisfaction": 1.0},
    "explainer": {"user_satisfaction": 4.0, "explanation_clarity": 0.80},

    # Cognitive RAG targets by phase
    "cognitive_rag.summarizer": {"entity_recall": 0.95, "intent_accuracy": 0.95},
    "cognitive_rag.investigator": {"hop_efficiency": 0.90, "evidence_relevance": 0.90},
    "cognitive_rag.agent": {"response_quality": 0.90, "routing_accuracy": 0.95},
    "cognitive_rag.reflector": {"memory_precision": 0.90, "procedure_quality": 0.85}
}
```

---

## 11. Related Documents

- `base-contract.md` - Base agent structures
- `orchestrator-contracts.md` - Orchestrator communication
- `tier0-contracts.md` - ML Foundation contracts
- `tier2-contracts.md` - Causal Inference contracts
- `tier3-contracts.md` - Design & Monitoring contracts
- `tier4-contracts.md` - ML Predictions contracts
- `integration-contracts.md` - System integration contracts (includes Cognitive RAG)
- `agent-handoff.yaml` - Standard handoff format examples
- `.claude/specialists/Agent_Specialists_Tiers 1-5/explainer.md` - Explainer specialist
- `.claude/specialists/Agent_Specialists_Tiers 1-5/feedback-learner.md` - Feedback Learner specialist
- `.claude/specialists/system/rag.md` - RAG system specialist with Cognitive RAG DSPy
