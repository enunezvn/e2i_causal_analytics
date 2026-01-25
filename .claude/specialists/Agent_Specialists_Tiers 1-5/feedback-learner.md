# Tier 5: Feedback Learner Agent Specialist

## Agent Classification

| Property | Value |
|----------|-------|
| **Tier** | 5 (Self-Improvement) |
| **Agent Type** | Deep (Extended Reasoning, Async) |
| **Model Tier** | Opus |
| **Latency Tolerance** | High (async - no real-time requirement) |
| **Critical Path** | No - runs asynchronously |

## Domain Scope

You are the specialist for the Tier 5 Feedback Learner Agent:
- `src/agents/feedback_learner/` - Self-improvement from feedback

This is a **Deep Async Agent** for:
- Learning from user feedback
- Improving agent performance over time
- Identifying systematic issues
- Updating organizational knowledge

## Design Principles

### Async Learning Pipeline
The Feedback Learner operates asynchronously:
- Processes feedback batches, not real-time
- Deep analysis of patterns
- Updates to knowledge stores
- No user-facing latency constraints

### Responsibilities
1. **Feedback Processing** - Ingest and categorize feedback
2. **Pattern Detection** - Identify systematic issues
3. **Learning Extraction** - Generate actionable improvements
4. **Knowledge Update** - Update organizational knowledge bases

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEEDBACK LEARNER AGENT                        │
│                    (Deep Async Pattern)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                FEEDBACK COLLECTOR                        │    │
│  │   • User ratings  • Corrections  • Outcome data         │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               PATTERN ANALYZER                           │    │
│  │   • Deep reasoning for pattern detection                 │    │
│  │   • Systematic issue identification                      │    │
│  │   • Root cause analysis                                  │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              LEARNING EXTRACTOR                          │    │
│  │   • Generate improvement recommendations                 │    │
│  │   • Propose prompt updates                               │    │
│  │   • Suggest model retraining                             │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              KNOWLEDGE UPDATER                           │    │
│  │   • Update experiment knowledge base                     │    │
│  │   • Update agent configurations                          │    │
│  │   • Update baseline assumptions                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
feedback_learner/
├── agent.py              # Main FeedbackLearnerAgent class
├── state.py              # LangGraph state definitions (v4.4: discovery feedback)
├── graph.py              # LangGraph assembly
├── scheduler.py          # Async scheduler for learning cycles (v4.3)
├── dspy_integration.py   # DSPy signals + GEPA trigger (v4.3)
├── mlflow_tracker.py     # MLflow experiment tracking
├── config/
│   └── loader.py         # Configuration loading
├── evaluation/
│   ├── rubric_evaluator.py  # AI-as-judge evaluation
│   └── criteria.py       # Evaluation criteria
├── nodes/
│   ├── feedback_collector.py     # Gather feedback data
│   ├── pattern_analyzer.py       # Deep pattern analysis
│   ├── learning_extractor.py     # Generate improvements
│   ├── knowledge_updater.py      # Apply updates
│   ├── rubric_node.py           # Rubric evaluation (v4.4)
│   └── discovery_feedback_node.py # Discovery feedback loop (v4.4)
└── feedback_types.py     # Feedback type definitions
```

## Async Scheduler (v4.3)

The Feedback Learner now includes an async scheduler for periodic learning cycles:

```python
from src.agents.feedback_learner import (
    FeedbackLearnerAgent,
    FeedbackLearnerScheduler,
    SchedulerConfig,
    create_scheduler,
)

# Create agent and scheduler
agent = FeedbackLearnerAgent()
scheduler = create_scheduler(
    agent,
    interval_hours=6.0,        # Run every 6 hours
    min_feedback_threshold=10,  # Minimum feedback to trigger
)

# Start scheduler (runs in background)
await scheduler.start()

# Run cycle manually
result = await scheduler.run_cycle_now(force=True)

# Stop gracefully
await scheduler.stop()
```

### Scheduler Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `interval_hours` | 6.0 | Hours between learning cycles |
| `min_feedback_threshold` | 10 | Minimum feedback items to trigger |
| `cycle_timeout_seconds` | 300 | Timeout per cycle (5 min) |
| `cooldown_hours` | - | Implicit via interval |

## GEPA Optimization Trigger (v4.3)

The agent includes explicit trigger conditions for GEPA prompt optimization:

```python
from src.agents.feedback_learner.dspy_integration import GEPAOptimizationTrigger

trigger = GEPAOptimizationTrigger(
    min_signals=100,           # Minimum training signals
    min_reward_delta=0.05,     # Minimum reward change
    cooldown_hours=24,         # Hours between optimizations
)

should_trigger, reason = trigger.should_trigger(
    signal_count=150,
    current_reward=0.75,
    baseline_reward=0.65,
    last_optimization=last_opt_time,
    has_critical_patterns=False,
)

if should_trigger:
    budget = trigger.get_recommended_budget(signal_count, hours_since)
    # budget: "light", "medium", or "heavy"
```

## Discovery Feedback Loop (v4.4)

Specialized feedback processing for causal discovery results:

```python
# Discovery feedback types
- user_correction: User corrects discovered edges
- expert_review: Domain expert validates/rejects DAG
- outcome_validation: Observed outcomes vs predicted
- gate_override: User overrides gate decision

# Tracked metrics by algorithm
- acceptance_rate, rejection_rate, modification_rate
- edge_precision, edge_recall
- average_accuracy
```

## LangGraph State Definition

```python
# src/agents/feedback_learner/state.py

from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
from datetime import datetime
import operator

class FeedbackItem(TypedDict):
    """Individual feedback item"""
    feedback_id: str
    timestamp: str
    feedback_type: Literal["rating", "correction", "outcome", "explicit"]
    source_agent: str
    query: str
    agent_response: str
    user_feedback: Any  # Rating, correction text, or outcome
    metadata: Dict[str, Any]

class DetectedPattern(TypedDict):
    """Detected pattern in feedback"""
    pattern_id: str
    pattern_type: Literal["accuracy_issue", "latency_issue", "relevance_issue", "format_issue", "coverage_gap"]
    description: str
    frequency: int
    severity: Literal["low", "medium", "high", "critical"]
    affected_agents: List[str]
    example_feedback_ids: List[str]
    root_cause_hypothesis: str

class LearningRecommendation(TypedDict):
    """Recommendation for improvement"""
    recommendation_id: str
    category: Literal["prompt_update", "model_retrain", "data_update", "config_change", "new_capability"]
    description: str
    affected_agents: List[str]
    expected_impact: str
    implementation_effort: Literal["low", "medium", "high"]
    priority: int
    proposed_change: Optional[str]

class KnowledgeUpdate(TypedDict):
    """Update to knowledge base"""
    update_id: str
    knowledge_type: Literal["experiment", "baseline", "agent_config", "prompt", "threshold"]
    key: str
    old_value: Any
    new_value: Any
    justification: str
    effective_date: str

class FeedbackLearnerState(TypedDict):
    """Complete state for Feedback Learner agent"""
    
    # === INPUT ===
    batch_id: str
    time_range_start: str
    time_range_end: str
    focus_agents: Optional[List[str]]
    
    # === FEEDBACK DATA ===
    feedback_items: Optional[List[FeedbackItem]]
    feedback_summary: Optional[Dict[str, Any]]
    
    # === PATTERN ANALYSIS ===
    detected_patterns: Optional[List[DetectedPattern]]
    pattern_clusters: Optional[Dict[str, List[str]]]
    
    # === LEARNING OUTPUTS ===
    learning_recommendations: Optional[List[LearningRecommendation]]
    priority_improvements: Optional[List[str]]
    
    # === KNOWLEDGE UPDATES ===
    proposed_updates: Optional[List[KnowledgeUpdate]]
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

## Node Implementations

### Feedback Collector Node

```python
# src/agents/feedback_learner/nodes/feedback_collector.py

import time
from typing import List, Dict, Any
from datetime import datetime

from ..state import FeedbackLearnerState, FeedbackItem

class FeedbackCollectorNode:
    """
    Collect feedback from various sources
    Prepares data for pattern analysis
    """
    
    def __init__(self, feedback_store, outcome_store):
        self.feedback_store = feedback_store
        self.outcome_store = outcome_store
    
    async def execute(self, state: FeedbackLearnerState) -> FeedbackLearnerState:
        start_time = time.time()
        
        try:
            # Collect explicit user feedback
            user_feedback = await self._collect_user_feedback(state)
            
            # Collect outcome-based feedback (predictions vs actuals)
            outcome_feedback = await self._collect_outcome_feedback(state)
            
            # Collect implicit feedback (engagement, follow-ups)
            implicit_feedback = await self._collect_implicit_feedback(state)
            
            # Combine all feedback
            all_feedback = user_feedback + outcome_feedback + implicit_feedback
            
            # Generate summary
            summary = self._generate_summary(all_feedback)
            
            collection_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "feedback_items": all_feedback,
                "feedback_summary": summary,
                "collection_latency_ms": collection_time,
                "status": "analyzing"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "feedback_collector", "error": str(e)}],
                "status": "failed"
            }
    
    async def _collect_user_feedback(self, state: FeedbackLearnerState) -> List[FeedbackItem]:
        """Collect explicit user feedback (ratings, corrections)"""
        
        raw_feedback = await self.feedback_store.get_feedback(
            start_time=state["time_range_start"],
            end_time=state["time_range_end"],
            agents=state.get("focus_agents")
        )
        
        items = []
        for fb in raw_feedback:
            items.append(FeedbackItem(
                feedback_id=fb["id"],
                timestamp=fb["timestamp"],
                feedback_type="rating" if "rating" in fb else "correction",
                source_agent=fb["agent"],
                query=fb["query"],
                agent_response=fb["response"],
                user_feedback=fb.get("rating") or fb.get("correction"),
                metadata=fb.get("metadata", {})
            ))
        
        return items
    
    async def _collect_outcome_feedback(self, state: FeedbackLearnerState) -> List[FeedbackItem]:
        """Collect outcome-based feedback (predictions vs actuals)"""
        
        outcomes = await self.outcome_store.get_outcomes(
            start_time=state["time_range_start"],
            end_time=state["time_range_end"]
        )
        
        items = []
        for outcome in outcomes:
            items.append(FeedbackItem(
                feedback_id=f"outcome_{outcome['id']}",
                timestamp=outcome["timestamp"],
                feedback_type="outcome",
                source_agent=outcome["agent"],
                query=outcome.get("original_query", ""),
                agent_response=str(outcome.get("prediction")),
                user_feedback={
                    "predicted": outcome.get("prediction"),
                    "actual": outcome.get("actual"),
                    "error": outcome.get("actual", 0) - outcome.get("prediction", 0)
                },
                metadata=outcome.get("metadata", {})
            ))
        
        return items
    
    async def _collect_implicit_feedback(self, state: FeedbackLearnerState) -> List[FeedbackItem]:
        """Collect implicit feedback from user behavior"""
        # Could include: follow-up questions (confusion), session abandonment, etc.
        return []
    
    def _generate_summary(self, feedback: List[FeedbackItem]) -> Dict[str, Any]:
        """Generate feedback summary statistics"""
        
        if not feedback:
            return {"total_count": 0}
        
        by_type = {}
        by_agent = {}
        ratings = []
        
        for item in feedback:
            # By type
            fb_type = item["feedback_type"]
            by_type[fb_type] = by_type.get(fb_type, 0) + 1
            
            # By agent
            agent = item["source_agent"]
            by_agent[agent] = by_agent.get(agent, 0) + 1
            
            # Ratings
            if fb_type == "rating" and isinstance(item["user_feedback"], (int, float)):
                ratings.append(item["user_feedback"])
        
        return {
            "total_count": len(feedback),
            "by_type": by_type,
            "by_agent": by_agent,
            "average_rating": sum(ratings) / len(ratings) if ratings else None,
            "rating_count": len(ratings)
        }
```

### Pattern Analyzer Node (Deep)

```python
# src/agents/feedback_learner/nodes/pattern_analyzer.py

import asyncio
import time
import json
from typing import List, Dict, Any
from langchain_anthropic import ChatAnthropic

from ..state import FeedbackLearnerState, DetectedPattern

class PatternAnalyzerNode:
    """
    Deep reasoning for pattern detection in feedback
    Identifies systematic issues requiring attention
    """
    
    def __init__(self):
        # Use Opus for deep pattern analysis
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",  # Or Opus for production
            max_tokens=8192,
            timeout=180
        )
    
    async def execute(self, state: FeedbackLearnerState) -> FeedbackLearnerState:
        start_time = time.time()
        
        if state.get("status") == "failed":
            return state
        
        try:
            feedback_items = state["feedback_items"]
            
            if not feedback_items:
                return {
                    **state,
                    "detected_patterns": [],
                    "pattern_clusters": {},
                    "analysis_latency_ms": 0,
                    "status": "extracting"
                }
            
            prompt = self._build_analysis_prompt(state)
            
            response = await asyncio.wait_for(
                self.llm.ainvoke(prompt),
                timeout=180
            )
            
            patterns = self._parse_patterns(response.content)
            clusters = self._cluster_patterns(patterns)
            
            analysis_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "detected_patterns": patterns,
                "pattern_clusters": clusters,
                "analysis_latency_ms": analysis_time,
                "model_used": "claude-sonnet-4-20250514",
                "status": "extracting"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "pattern_analyzer", "error": str(e)}],
                "status": "failed"
            }
    
    def _build_analysis_prompt(self, state: FeedbackLearnerState) -> str:
        """Build deep analysis prompt"""
        
        feedback_items = state["feedback_items"]
        summary = state["feedback_summary"]
        
        # Sample feedback for analysis (avoid token limits)
        sample_size = min(50, len(feedback_items))
        sampled = feedback_items[:sample_size]
        
        feedback_str = "\n\n".join([
            f"**Feedback {i+1}** (Type: {fb['feedback_type']}, Agent: {fb['source_agent']})\n"
            f"Query: {fb['query'][:200]}\n"
            f"Response: {fb['agent_response'][:300]}\n"
            f"Feedback: {fb['user_feedback']}"
            for i, fb in enumerate(sampled)
        ])
        
        return f"""You are an expert in analyzing AI system feedback to identify systematic issues and improvement opportunities.

## Feedback Summary
- Total feedback items: {summary['total_count']}
- By type: {json.dumps(summary['by_type'])}
- By agent: {json.dumps(summary['by_agent'])}
- Average rating: {summary.get('average_rating', 'N/A')}

## Sample Feedback Items

{feedback_str}

---

## Your Task

Analyze this feedback to identify systematic patterns requiring attention.

### Pattern Categories
1. **accuracy_issue**: Agent provides incorrect or inaccurate information
2. **latency_issue**: Agent takes too long to respond
3. **relevance_issue**: Agent response doesn't address the query
4. **format_issue**: Agent response is poorly formatted or unclear
5. **coverage_gap**: Agent lacks knowledge in certain areas

### Analysis Steps
1. Look for recurring issues across multiple feedback items
2. Identify root causes (not just symptoms)
3. Assess severity based on frequency and impact
4. Note which agents are affected

### Output Format (JSON)

```json
{{
  "patterns": [
    {{
      "pattern_id": "P1",
      "pattern_type": "accuracy_issue|latency_issue|relevance_issue|format_issue|coverage_gap",
      "description": "Clear description of the pattern",
      "frequency": <number of occurrences>,
      "severity": "low|medium|high|critical",
      "affected_agents": ["agent1", "agent2"],
      "example_feedback_ids": ["id1", "id2"],
      "root_cause_hypothesis": "Hypothesis about why this is happening"
    }}
  ]
}}
```

Focus on actionable patterns that could improve system performance."""

    def _parse_patterns(self, content: str) -> List[DetectedPattern]:
        """Parse detected patterns from response"""
        import re
        
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return [DetectedPattern(**p) for p in data.get("patterns", [])]
            except (json.JSONDecodeError, TypeError):
                pass
        
        return []
    
    def _cluster_patterns(self, patterns: List[DetectedPattern]) -> Dict[str, List[str]]:
        """Cluster patterns by type and agent"""
        
        clusters = {}
        
        for pattern in patterns:
            # By type
            ptype = pattern["pattern_type"]
            if ptype not in clusters:
                clusters[ptype] = []
            clusters[ptype].append(pattern["pattern_id"])
        
        return clusters
```

### Learning Extractor Node

```python
# src/agents/feedback_learner/nodes/learning_extractor.py

import asyncio
import time
import json
from typing import List, Dict
from langchain_anthropic import ChatAnthropic

from ..state import FeedbackLearnerState, LearningRecommendation

class LearningExtractorNode:
    """
    Extract actionable learnings from detected patterns
    Generate improvement recommendations
    """
    
    def __init__(self):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            timeout=120
        )
    
    async def execute(self, state: FeedbackLearnerState) -> FeedbackLearnerState:
        start_time = time.time()
        
        if state.get("status") == "failed":
            return state
        
        try:
            patterns = state.get("detected_patterns", [])
            
            if not patterns:
                return {
                    **state,
                    "learning_recommendations": [],
                    "priority_improvements": [],
                    "extraction_latency_ms": 0,
                    "status": "updating"
                }
            
            prompt = self._build_extraction_prompt(state)
            
            response = await asyncio.wait_for(
                self.llm.ainvoke(prompt),
                timeout=120
            )
            
            recommendations = self._parse_recommendations(response.content)
            priorities = self._prioritize(recommendations)
            
            extraction_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "learning_recommendations": recommendations,
                "priority_improvements": priorities,
                "extraction_latency_ms": extraction_time,
                "status": "updating"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "learning_extractor", "error": str(e)}],
                "status": "failed"
            }
    
    def _build_extraction_prompt(self, state: FeedbackLearnerState) -> str:
        """Build learning extraction prompt"""
        
        patterns = state["detected_patterns"]
        
        patterns_str = "\n\n".join([
            f"**{p['pattern_id']}**: {p['description']}\n"
            f"Type: {p['pattern_type']}, Severity: {p['severity']}\n"
            f"Affected agents: {', '.join(p['affected_agents'])}\n"
            f"Root cause hypothesis: {p['root_cause_hypothesis']}"
            for p in patterns
        ])
        
        return f"""Based on these detected patterns, generate actionable improvement recommendations.

## Detected Patterns

{patterns_str}

---

## Recommendation Categories

1. **prompt_update**: Changes to agent prompts
2. **model_retrain**: Model retraining or fine-tuning
3. **data_update**: Updates to training data or knowledge bases
4. **config_change**: Configuration parameter changes
5. **new_capability**: New features or capabilities needed

## Output Format (JSON)

```json
{{
  "recommendations": [
    {{
      "recommendation_id": "R1",
      "category": "prompt_update|model_retrain|data_update|config_change|new_capability",
      "description": "Clear description of the improvement",
      "affected_agents": ["agent1"],
      "expected_impact": "Description of expected improvement",
      "implementation_effort": "low|medium|high",
      "priority": 1-5,
      "proposed_change": "Specific change to implement (if applicable)"
    }}
  ]
}}
```

Focus on high-impact, actionable improvements."""

    def _parse_recommendations(self, content: str) -> List[LearningRecommendation]:
        """Parse recommendations from response"""
        import re
        
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return [LearningRecommendation(**r) for r in data.get("recommendations", [])]
            except (json.JSONDecodeError, TypeError):
                pass
        
        return []
    
    def _prioritize(self, recommendations: List[LearningRecommendation]) -> List[str]:
        """Get prioritized list of improvements"""
        
        # Sort by priority (lower = higher priority) and impact
        sorted_recs = sorted(
            recommendations,
            key=lambda r: (r["priority"], {"low": 3, "medium": 2, "high": 1}.get(r["implementation_effort"], 2))
        )
        
        return [r["description"] for r in sorted_recs[:5]]
```

### Knowledge Updater Node

```python
# src/agents/feedback_learner/nodes/knowledge_updater.py

import time
from typing import List
from datetime import datetime

from ..state import FeedbackLearnerState, KnowledgeUpdate

class KnowledgeUpdaterNode:
    """
    Apply learnings to knowledge bases
    Updates organizational knowledge
    """
    
    def __init__(self, knowledge_stores: dict):
        self.stores = knowledge_stores
    
    async def execute(self, state: FeedbackLearnerState) -> FeedbackLearnerState:
        start_time = time.time()
        
        if state.get("status") == "failed":
            return state
        
        try:
            recommendations = state.get("learning_recommendations", [])
            
            # Generate proposed updates
            proposed_updates = self._generate_updates(recommendations)
            
            # Apply updates (with validation)
            applied = []
            for update in proposed_updates:
                success = await self._apply_update(update)
                if success:
                    applied.append(update["update_id"])
            
            # Generate summary
            summary = self._generate_summary(state, proposed_updates, applied)
            
            update_time = int((time.time() - start_time) * 1000)
            total_time = (
                state.get("collection_latency_ms", 0) +
                state.get("analysis_latency_ms", 0) +
                state.get("extraction_latency_ms", 0) +
                update_time
            )
            
            return {
                **state,
                "proposed_updates": proposed_updates,
                "applied_updates": applied,
                "learning_summary": summary,
                "update_latency_ms": update_time,
                "total_latency_ms": total_time,
                "status": "completed"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "knowledge_updater", "error": str(e)}],
                "status": "failed"
            }
    
    def _generate_updates(self, recommendations: List) -> List[KnowledgeUpdate]:
        """Generate knowledge updates from recommendations"""
        
        updates = []
        
        for rec in recommendations:
            if rec["category"] == "data_update":
                updates.append(KnowledgeUpdate(
                    update_id=f"U_{rec['recommendation_id']}",
                    knowledge_type="baseline",
                    key=rec.get("affected_agents", ["unknown"])[0],
                    old_value=None,
                    new_value=rec.get("proposed_change"),
                    justification=rec["description"],
                    effective_date=datetime.now().isoformat()
                ))
            
            elif rec["category"] == "config_change":
                updates.append(KnowledgeUpdate(
                    update_id=f"U_{rec['recommendation_id']}",
                    knowledge_type="agent_config",
                    key=rec.get("affected_agents", ["unknown"])[0],
                    old_value=None,
                    new_value=rec.get("proposed_change"),
                    justification=rec["description"],
                    effective_date=datetime.now().isoformat()
                ))
        
        return updates
    
    async def _apply_update(self, update: KnowledgeUpdate) -> bool:
        """Apply a single update to knowledge store"""
        
        knowledge_type = update["knowledge_type"]
        
        if knowledge_type not in self.stores:
            return False
        
        store = self.stores[knowledge_type]
        
        try:
            await store.update(
                key=update["key"],
                value=update["new_value"],
                justification=update["justification"]
            )
            return True
        except Exception:
            return False
    
    def _generate_summary(
        self, 
        state: FeedbackLearnerState,
        proposed: List,
        applied: List
    ) -> str:
        """Generate learning summary"""
        
        feedback_count = len(state.get("feedback_items", []))
        pattern_count = len(state.get("detected_patterns", []))
        rec_count = len(state.get("learning_recommendations", []))
        
        summary = f"Learning cycle complete. "
        summary += f"Processed {feedback_count} feedback items. "
        summary += f"Detected {pattern_count} patterns. "
        summary += f"Generated {rec_count} recommendations. "
        summary += f"Applied {len(applied)} of {len(proposed)} proposed updates."
        
        return summary
```

## Graph Assembly

```python
# src/agents/feedback_learner/graph.py

from langgraph.graph import StateGraph, END

from .state import FeedbackLearnerState
from .nodes.feedback_collector import FeedbackCollectorNode
from .nodes.pattern_analyzer import PatternAnalyzerNode
from .nodes.learning_extractor import LearningExtractorNode
from .nodes.knowledge_updater import KnowledgeUpdaterNode

def build_feedback_learner_graph(
    feedback_store,
    outcome_store,
    knowledge_stores
):
    """
    Build the Feedback Learner agent graph
    
    Architecture:
        [collect] → [analyze] → [extract] → [update] → END
    
    Note: This runs asynchronously, not in user request path
    """
    
    # Initialize nodes
    collector = FeedbackCollectorNode(feedback_store, outcome_store)
    analyzer = PatternAnalyzerNode()
    extractor = LearningExtractorNode()
    updater = KnowledgeUpdaterNode(knowledge_stores)
    
    # Build graph
    workflow = StateGraph(FeedbackLearnerState)
    
    # Add nodes
    workflow.add_node("collect", collector.execute)
    workflow.add_node("analyze", analyzer.execute)
    workflow.add_node("extract", extractor.execute)
    workflow.add_node("update", updater.execute)
    workflow.add_node("error_handler", error_handler_node)
    
    # Flow
    workflow.set_entry_point("collect")
    
    workflow.add_conditional_edges(
        "collect",
        lambda s: "error" if s.get("status") == "failed" else "analyze",
        {"analyze": "analyze", "error": "error_handler"}
    )
    
    workflow.add_conditional_edges(
        "analyze",
        lambda s: "error" if s.get("status") == "failed" else "extract",
        {"extract": "extract", "error": "error_handler"}
    )
    
    workflow.add_conditional_edges(
        "extract",
        lambda s: "error" if s.get("status") == "failed" else "update",
        {"update": "update", "error": "error_handler"}
    )
    
    workflow.add_edge("update", END)
    workflow.add_edge("error_handler", END)
    
    return workflow.compile()

async def error_handler_node(state: FeedbackLearnerState) -> FeedbackLearnerState:
    return {
        **state,
        "learning_summary": "Learning cycle failed. See errors for details.",
        "status": "failed"
    }
```

## Async Execution Pattern

```python
# src/agents/feedback_learner/scheduler.py

import asyncio
from datetime import datetime, timedelta

class FeedbackLearnerScheduler:
    """
    Schedule async feedback learning cycles
    """
    
    def __init__(self, agent, interval_hours: int = 24):
        self.agent = agent
        self.interval = timedelta(hours=interval_hours)
        self.running = False
    
    async def start(self):
        """Start the learning loop"""
        self.running = True
        
        while self.running:
            try:
                # Run learning cycle
                end_time = datetime.now()
                start_time = end_time - self.interval
                
                result = await self.agent.learn(
                    time_range_start=start_time.isoformat(),
                    time_range_end=end_time.isoformat()
                )
                
                # Log results
                print(f"Learning cycle complete: {result.get('learning_summary')}")
                
            except Exception as e:
                print(f"Learning cycle failed: {e}")
            
            # Wait for next cycle
            await asyncio.sleep(self.interval.total_seconds())
    
    def stop(self):
        """Stop the learning loop"""
        self.running = False
```

## Integration Contracts

### Input Contract
```python
class FeedbackLearnerInput(BaseModel):
    batch_id: str
    time_range_start: str
    time_range_end: str
    focus_agents: Optional[List[str]] = None
```

### Output Contract
```python
class FeedbackLearnerOutput(BaseModel):
    detected_patterns: List[DetectedPattern]
    learning_recommendations: List[LearningRecommendation]
    priority_improvements: List[str]
    applied_updates: List[str]
    learning_summary: str
    total_latency_ms: int
```

## Handoff Format

```yaml
feedback_learner_handoff:
  agent: feedback_learner
  analysis_type: learning_cycle
  key_findings:
    - feedback_processed: <count>
    - patterns_detected: <count>
    - recommendations: <count>
    - updates_applied: <count>
  patterns:
    - type: <pattern type>
      severity: <severity>
      affected_agents: [<agent 1>]
  top_recommendations:
    - <recommendation 1>
    - <recommendation 2>
  summary: <learning summary>
  next_cycle: <scheduled time>
```

## Integration with Experiment Designer

The Feedback Learner forms a closed loop with the Experiment Designer:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENTAL LEARNING CYCLE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────────┐         ┌──────────────────┐                 │
│   │   EXPERIMENT     │────────►│   CAUSAL         │                 │
│   │   DESIGNER       │         │   IMPACT         │                 │
│   │  Design spec     │         │  Effect estimate │                 │
│   │  DAG template    │         │  Robustness      │                 │
│   └────────▲─────────┘         └────────┬─────────┘                 │
│            │                            │                            │
│            │   ┌──────────────────┐    │                            │
│            └───│   FEEDBACK       │◄───┘                            │
│                │   LEARNER        │                                  │
│                │  What worked?    │                                  │
│                │  Assumption      │                                  │
│                │  violations?     │                                  │
│                └──────────────────┘                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

Lessons learned feed back into the Experiment Designer's organizational context.

## DSPy MIPROv2 Integration

The Feedback Learner is the central hub for DSPy optimization across the E2I system.

### DSPy Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   DSPy FEEDBACK LEARNER ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────┐                         ┌─────────────────────────┐    │
│  │  SIGNAL SOURCES │                         │   TARGET AGENTS          │    │
│  │  (12 sources)   │                         │   (10 agents)            │    │
│  │                 │                         │                          │    │
│  │  • Orchestrator │   ┌─────────────────┐   │  • Causal Impact        │    │
│  │  • Causal Impact│──►│ FEEDBACK LEARNER│──►│  • Gap Analyzer         │    │
│  │  • Gap Analyzer │   │                 │   │  • Heterogeneous Opt    │    │
│  │  • Drift Monitor│   │ MIPROv2 Optim.  │   │  • Drift Monitor        │    │
│  │  • Health Score │   │ BootstrapFewShot│   │  • Experiment Designer  │    │
│  │  • Prediction S.│   │ COPRO           │   │  • Health Score         │    │
│  │  • Explainer    │   │                 │   │  • Prediction Synth.    │    │
│  │  • RAG Phases:  │   └─────────────────┘   │  • Resource Optimizer   │    │
│  │    - Summarizer │                         │  • Explainer            │    │
│  │    - Investigator                         │  • Cognitive RAG        │    │
│  │    - Agent      │                         │                          │    │
│  │    - Reflector  │                         │                          │    │
│  └─────────────────┘                         └─────────────────────────┘    │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 12 Signal Sources

| Source | Signal Type | What It Provides |
|--------|-------------|------------------|
| **Orchestrator** | Routing decisions | Which agents were selected, latency |
| **Causal Impact** | Effect estimates | Causal effect accuracy, confidence |
| **Gap Analyzer** | Opportunity scores | Gap detection precision |
| **Drift Monitor** | Drift alerts | Drift detection accuracy |
| **Health Score** | System health | Health metric accuracy |
| **Prediction Synth** | Predictions | Prediction accuracy vs actuals |
| **Explainer** | Explanations | Explanation quality ratings |
| **RAG Summarizer** | Query processing | Entity extraction, intent classification |
| **RAG Investigator** | Evidence gathering | Hop efficiency, evidence relevance |
| **RAG Agent** | Response generation | Response quality, routing accuracy |
| **RAG Reflector** | Memory decisions | Memory worthiness, procedure learning |
| **User Feedback** | Direct feedback | Ratings, corrections, outcomes |

### 10 Target Agents for Optimization

```python
OPTIMIZABLE_AGENTS = {
    "causal_impact": {
        "signatures": ["EffectEstimationSignature", "RobustnessCheckSignature"],
        "optimizers": ["MIPROv2", "BootstrapFewShot"],
        "metrics": ["effect_accuracy", "assumption_validity"]
    },
    "gap_analyzer": {
        "signatures": ["OpportunityDetectionSignature", "PriorityRankingSignature"],
        "optimizers": ["MIPROv2"],
        "metrics": ["gap_precision", "priority_correlation"]
    },
    "heterogeneous_optimizer": {
        "signatures": ["SegmentAnalysisSignature", "CATEEstimationSignature"],
        "optimizers": ["MIPROv2"],
        "metrics": ["segment_accuracy", "cate_mse"]
    },
    "drift_monitor": {
        "signatures": ["DriftDetectionSignature", "AlertPrioritySignature"],
        "optimizers": ["BootstrapFewShot"],
        "metrics": ["detection_recall", "false_positive_rate"]
    },
    "experiment_designer": {
        "signatures": ["DesignSpecSignature", "PowerAnalysisSignature"],
        "optimizers": ["MIPROv2"],
        "metrics": ["design_validity", "power_accuracy"]
    },
    "health_score": {
        "signatures": ["HealthAssessmentSignature", "ComponentScoreSignature"],
        "optimizers": ["BootstrapFewShot"],
        "metrics": ["health_accuracy", "component_correlation"]
    },
    "prediction_synthesizer": {
        "signatures": ["PredictionAggregationSignature", "ConfidenceEstimationSignature"],
        "optimizers": ["MIPROv2"],
        "metrics": ["prediction_mape", "calibration_score"]
    },
    "resource_optimizer": {
        "signatures": ["ResourceAllocationSignature", "ConstraintSatisfactionSignature"],
        "optimizers": ["MIPROv2"],
        "metrics": ["allocation_optimality", "constraint_satisfaction"]
    },
    "explainer": {
        "signatures": ["ExplanationGenerationSignature", "VisualizationSelectionSignature"],
        "optimizers": ["COPRO"],
        "metrics": ["explanation_clarity", "user_satisfaction"]
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
        "optimizers": ["MIPROv2", "BootstrapFewShot", "COPRO"],
        "metrics": ["retrieval_relevance", "hop_efficiency", "response_quality"]
    }
}
```

### DSPy Optimization Loop

```python
class DSPyFeedbackOptimizer:
    """
    Central DSPy optimization orchestrator.
    Coordinates MIPROv2, BootstrapFewShot, and COPRO optimizers.
    """

    def __init__(self, signal_store, signature_registry):
        self.signal_store = signal_store
        self.signatures = signature_registry
        self.optimizers = {
            "miprov2": MIPROv2(),
            "bootstrap": BootstrapFewShot(),
            "copro": COPRO()
        }

    async def optimization_cycle(self, time_range: tuple) -> OptimizationResult:
        """
        1. Aggregate signals from 12 sources
        2. Detect patterns requiring optimization
        3. Select signatures to optimize
        4. Run appropriate optimizer
        5. Update signature in registry
        6. Emit optimization signals for tracking
        """

        # Step 1: Aggregate signals
        signals = await self.signal_store.get_signals(
            start_time=time_range[0],
            end_time=time_range[1]
        )

        # Step 2: Detect patterns
        patterns = self._detect_optimization_patterns(signals)

        # Step 3: Select and optimize
        results = []
        for pattern in patterns:
            signature = self.signatures.get(pattern.signature_name)
            optimizer = self.optimizers[pattern.recommended_optimizer]

            # Step 4: Run optimization
            optimized = await optimizer.compile(
                signature,
                trainset=pattern.training_examples,
                metric=pattern.optimization_metric
            )

            # Step 5: Update registry
            await self.signatures.update(pattern.signature_name, optimized)

            results.append(OptimizationResult(
                signature=pattern.signature_name,
                improvement=pattern.expected_improvement,
                optimizer_used=pattern.recommended_optimizer
            ))

        return results
```

### Signal Aggregation Schema

```python
class DSPySignal(TypedDict):
    """Signal emitted by DSPy-powered components"""
    signal_id: str
    timestamp: str
    source_component: str  # Agent or RAG phase
    signature_name: str    # Which DSPy signature
    prediction: Any        # What was predicted
    ground_truth: Any      # Actual outcome (if available)
    confidence: float      # Model confidence
    latency_ms: int        # Execution time
    user_feedback: Optional[int]  # 1-5 rating if available
    metadata: Dict[str, Any]

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
```

### Optimization Metrics by Agent

| Agent | Primary Metric | Secondary Metric | Target |
|-------|----------------|------------------|--------|
| Causal Impact | Effect accuracy | Assumption validity | >0.90 |
| Gap Analyzer | Gap precision | Priority correlation | >0.85 |
| Heterogeneous Opt | Segment accuracy | CATE MSE | >0.85, <0.15 |
| Drift Monitor | Detection recall | False positive rate | >0.95, <0.05 |
| Experiment Designer | Design validity | Power accuracy | >0.90 |
| Health Score | Health accuracy | Component correlation | >0.90 |
| Prediction Synth | Prediction MAPE | Calibration score | <0.10, >0.85 |
| Resource Optimizer | Allocation optimality | Constraint satisfaction | >0.85, 1.0 |
| Explainer | User satisfaction | Explanation clarity | >4.0, >0.80 |
| Cognitive RAG | Response quality | Hop efficiency | >0.90, >0.85 |

### Integration with Cognitive RAG

The Feedback Learner receives signals from all 4 phases of the Cognitive RAG:

```python
# Signal collection from Cognitive RAG
RAG_PHASE_SIGNALS = {
    "summarizer": [
        "QueryRewriteSignature",
        "EntityExtractionSignature",
        "IntentClassificationSignature"
    ],
    "investigator": [
        "InvestigationPlanSignature",
        "HopDecisionSignature",
        "EvidenceRelevanceSignature"
    ],
    "agent": [
        "EvidenceSynthesisSignature",
        "AgentRoutingSignature",
        "VisualizationConfigSignature"
    ],
    "reflector": [
        "MemoryWorthinessSignature",
        "ProcedureLearningSignature"
    ]
}
```

### Optimization Scheduling

```python
OPTIMIZATION_SCHEDULE = {
    "continuous": {
        # Run after each user session
        "trigger": "session_end",
        "signatures": ["QueryRewriteSignature", "EntityExtractionSignature"],
        "optimizer": "BootstrapFewShot"
    },
    "hourly": {
        # Run every hour during business hours
        "trigger": "cron(0 * 9-17 * * *)",
        "signatures": ["EvidenceRelevanceSignature", "AgentRoutingSignature"],
        "optimizer": "BootstrapFewShot"
    },
    "daily": {
        # Run overnight
        "trigger": "cron(0 2 * * *)",
        "signatures": ["all_miprov2_signatures"],
        "optimizer": "MIPROv2"
    },
    "weekly": {
        # Full recompilation
        "trigger": "cron(0 3 * * 0)",
        "signatures": ["all_signatures"],
        "optimizer": "MIPROv2"
    }
}
```

### Expected Improvements

| Optimization Target | Before | After | Improvement |
|---------------------|--------|-------|-------------|
| Query Understanding | ~75% | ~95% | +20% |
| Evidence Retrieval | ~70% | ~95% | +25% |
| Response Quality | ~75% | ~95% | +20% |
| Agent Routing | ~80% | ~95% | +15% |
| Hop Efficiency | ~60% | ~90% | +30% |

---

## Discovery Feedback Loop (V4.4+)

The Feedback Learner tracks causal discovery outcomes and recommends parameter adjustments based on accumulated feedback patterns.

### Integration Pattern

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DISCOVERY FEEDBACK LOOP                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐       │
│  │ Discovery    │───▶│ Gate         │───▶│ User/Expert          │       │
│  │ Runner       │    │ Evaluation   │    │ Feedback             │       │
│  └──────────────┘    └──────────────┘    └──────────┬───────────┘       │
│                                                      │                   │
│                                                      ▼                   │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │                    FEEDBACK LEARNER                           │       │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │       │
│  │  │ Accuracy       │  │ Pattern        │  │ Parameter      │  │       │
│  │  │ Tracking       │  │ Detection      │  │ Recommender    │  │       │
│  │  └────────────────┘  └────────────────┘  └────────────────┘  │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ OUTPUTS:                                                      │       │
│  │ • Discovery accuracy metrics                                  │       │
│  │ • Failure pattern analysis                                    │       │
│  │ • Parameter adjustment recommendations                        │       │
│  │ • Algorithm selection guidance                                │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### State Fields for Discovery Feedback

```python
class FeedbackLearnerState(TypedDict):
    # ... existing fields ...

    # === DISCOVERY FEEDBACK (V4.4+) ===
    discovery_feedback_records: Optional[List[Dict[str, Any]]]  # Accumulated feedback
    discovery_accuracy_metrics: Optional[Dict[str, float]]  # Per-algorithm accuracy
    discovery_failure_patterns: Optional[List[Dict[str, Any]]]  # Identified patterns
    discovery_parameter_recommendations: Optional[Dict[str, Any]]  # Suggested adjustments
    discovery_algorithm_rankings: Optional[List[Tuple[str, float]]]  # Algorithm performance
```

### Feedback Collection

Collect feedback on discovery outcomes from user interactions and expert reviews:

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from uuid import UUID


class DiscoveryFeedbackType(Enum):
    """Types of discovery feedback."""
    USER_CORRECTION = "user_correction"  # User corrected a discovered edge
    EXPERT_REVIEW = "expert_review"  # Expert validated discovery
    OUTCOME_VALIDATION = "outcome_validation"  # Downstream effect confirmed/denied
    GATE_OVERRIDE = "gate_override"  # User overrode gate decision


@dataclass
class DiscoveryFeedbackRecord:
    """Single discovery feedback record."""
    feedback_id: UUID
    session_id: UUID
    discovery_id: UUID  # References ml.discovered_dags
    feedback_type: DiscoveryFeedbackType

    # Discovery context
    algorithms_used: List[str]
    ensemble_threshold: float
    gate_decision: str
    gate_confidence: float

    # Feedback specifics
    feedback_positive: bool  # True = discovery was correct
    corrected_edges: Optional[List[Dict[str, Any]]] = None  # Added/removed edges
    feedback_notes: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    feedback_source: str = "user"  # user, expert, automated


def collect_discovery_feedback(
    state: Dict[str, Any],
    feedback_type: DiscoveryFeedbackType,
    is_positive: bool,
    corrected_edges: Optional[List[Dict[str, Any]]] = None,
    notes: Optional[str] = None,
) -> DiscoveryFeedbackRecord:
    """
    Collect feedback on a discovery outcome.

    Args:
        state: Current state with discovery results
        feedback_type: Type of feedback being provided
        is_positive: Whether discovery was accurate
        corrected_edges: Any edge corrections made
        notes: Optional feedback notes

    Returns:
        DiscoveryFeedbackRecord for storage
    """
    import uuid

    discovery_result = state.get("discovery_result", {})
    gate_eval = state.get("discovery_gate_evaluation", {})

    return DiscoveryFeedbackRecord(
        feedback_id=uuid.uuid4(),
        session_id=state.get("session_id"),
        discovery_id=discovery_result.get("discovery_id"),
        feedback_type=feedback_type,
        algorithms_used=discovery_result.get("algorithms", ["ges", "pc"]),
        ensemble_threshold=discovery_result.get("ensemble_threshold", 0.5),
        gate_decision=gate_eval.get("decision", "unknown"),
        gate_confidence=gate_eval.get("confidence", 0.0),
        feedback_positive=is_positive,
        corrected_edges=corrected_edges,
        feedback_notes=notes,
    )
```

### Accuracy Tracking

Track discovery accuracy metrics over time by algorithm and configuration:

```python
from collections import defaultdict
from typing import Dict, List, Tuple


@dataclass
class DiscoveryAccuracyMetrics:
    """Accuracy metrics for discovery configurations."""
    total_discoveries: int = 0
    positive_feedback_count: int = 0
    negative_feedback_count: int = 0

    # Per-algorithm metrics
    algorithm_accuracy: Dict[str, float] = field(default_factory=dict)
    algorithm_counts: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # (positive, total)

    # Per-gate-decision metrics
    gate_decision_accuracy: Dict[str, float] = field(default_factory=dict)

    # Edge-level metrics
    edge_precision: float = 0.0  # Fraction of discovered edges that were correct
    edge_recall: float = 0.0  # Fraction of true edges that were discovered

    @property
    def overall_accuracy(self) -> float:
        """Overall discovery accuracy."""
        if self.total_discoveries == 0:
            return 0.0
        return self.positive_feedback_count / self.total_discoveries


def compute_discovery_accuracy(
    feedback_records: List[DiscoveryFeedbackRecord],
) -> DiscoveryAccuracyMetrics:
    """
    Compute accuracy metrics from accumulated feedback.

    Args:
        feedback_records: List of feedback records

    Returns:
        DiscoveryAccuracyMetrics with computed values
    """
    metrics = DiscoveryAccuracyMetrics()

    # Algorithm-level tracking
    algo_positive = defaultdict(int)
    algo_total = defaultdict(int)

    # Gate decision tracking
    gate_positive = defaultdict(int)
    gate_total = defaultdict(int)

    # Edge-level tracking
    total_discovered_edges = 0
    correct_edges = 0
    total_true_edges = 0
    discovered_true_edges = 0

    for record in feedback_records:
        metrics.total_discoveries += 1

        if record.feedback_positive:
            metrics.positive_feedback_count += 1
        else:
            metrics.negative_feedback_count += 1

        # Track by algorithm
        for algo in record.algorithms_used:
            algo_total[algo] += 1
            if record.feedback_positive:
                algo_positive[algo] += 1

        # Track by gate decision
        gate_total[record.gate_decision] += 1
        if record.feedback_positive:
            gate_positive[record.gate_decision] += 1

        # Track edge corrections for precision/recall
        if record.corrected_edges:
            for edge in record.corrected_edges:
                if edge.get("correction_type") == "removed":
                    # Edge was incorrectly discovered
                    total_discovered_edges += 1
                elif edge.get("correction_type") == "added":
                    # Edge was missed
                    total_true_edges += 1
                elif edge.get("correction_type") == "confirmed":
                    correct_edges += 1
                    discovered_true_edges += 1
                    total_discovered_edges += 1
                    total_true_edges += 1

    # Compute algorithm accuracy
    for algo in algo_total:
        metrics.algorithm_accuracy[algo] = (
            algo_positive[algo] / algo_total[algo] if algo_total[algo] > 0 else 0.0
        )
        metrics.algorithm_counts[algo] = (algo_positive[algo], algo_total[algo])

    # Compute gate decision accuracy
    for decision in gate_total:
        metrics.gate_decision_accuracy[decision] = (
            gate_positive[decision] / gate_total[decision] if gate_total[decision] > 0 else 0.0
        )

    # Compute edge precision/recall
    if total_discovered_edges > 0:
        metrics.edge_precision = correct_edges / total_discovered_edges
    if total_true_edges > 0:
        metrics.edge_recall = discovered_true_edges / total_true_edges

    return metrics
```

### Pattern Detection

Identify patterns in discovery failures to improve future runs:

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DiscoveryFailurePattern:
    """Identified pattern in discovery failures."""
    pattern_id: str
    pattern_type: str  # "data_characteristic", "configuration", "domain"
    description: str
    frequency: int  # Number of times observed
    confidence: float  # Confidence this is a real pattern

    # Pattern specifics
    trigger_conditions: Dict[str, Any]  # When this failure occurs
    affected_algorithms: List[str]  # Which algorithms are affected
    recommended_action: str  # What to do about it


def detect_failure_patterns(
    feedback_records: List[DiscoveryFeedbackRecord],
    min_frequency: int = 3,
    min_confidence: float = 0.7,
) -> List[DiscoveryFailurePattern]:
    """
    Detect patterns in discovery failures.

    Args:
        feedback_records: List of feedback records
        min_frequency: Minimum occurrences to consider a pattern
        min_confidence: Minimum confidence threshold

    Returns:
        List of identified failure patterns
    """
    patterns = []
    negative_records = [r for r in feedback_records if not r.feedback_positive]

    if len(negative_records) < min_frequency:
        return patterns

    # Pattern 1: Low ensemble threshold failures
    low_threshold_failures = [
        r for r in negative_records
        if r.ensemble_threshold < 0.5
    ]
    if len(low_threshold_failures) >= min_frequency:
        confidence = len(low_threshold_failures) / len(negative_records)
        if confidence >= min_confidence:
            patterns.append(DiscoveryFailurePattern(
                pattern_id="low_ensemble_threshold",
                pattern_type="configuration",
                description="Discovery failures correlate with low ensemble threshold (<0.5)",
                frequency=len(low_threshold_failures),
                confidence=confidence,
                trigger_conditions={"ensemble_threshold": "<0.5"},
                affected_algorithms=["all"],
                recommended_action="Increase ensemble_threshold to 0.6 or higher",
            ))

    # Pattern 2: Single algorithm failures
    single_algo_failures = [
        r for r in negative_records
        if len(r.algorithms_used) == 1
    ]
    if len(single_algo_failures) >= min_frequency:
        confidence = len(single_algo_failures) / len(negative_records)
        if confidence >= min_confidence:
            patterns.append(DiscoveryFailurePattern(
                pattern_id="single_algorithm",
                pattern_type="configuration",
                description="Discovery failures correlate with using only one algorithm",
                frequency=len(single_algo_failures),
                confidence=confidence,
                trigger_conditions={"algorithm_count": 1},
                affected_algorithms=list(set(
                    r.algorithms_used[0] for r in single_algo_failures
                )),
                recommended_action="Use ensemble of GES + PC for better edge voting",
            ))

    # Pattern 3: Algorithm-specific failures
    algo_failure_counts: Dict[str, int] = defaultdict(int)
    algo_total_counts: Dict[str, int] = defaultdict(int)

    for record in feedback_records:
        for algo in record.algorithms_used:
            algo_total_counts[algo] += 1
            if not record.feedback_positive:
                algo_failure_counts[algo] += 1

    for algo, failure_count in algo_failure_counts.items():
        if failure_count >= min_frequency:
            failure_rate = failure_count / algo_total_counts[algo]
            if failure_rate >= 0.5:  # >50% failure rate
                patterns.append(DiscoveryFailurePattern(
                    pattern_id=f"{algo}_high_failure",
                    pattern_type="algorithm",
                    description=f"{algo.upper()} algorithm has high failure rate ({failure_rate:.1%})",
                    frequency=failure_count,
                    confidence=failure_rate,
                    trigger_conditions={"algorithm": algo},
                    affected_algorithms=[algo],
                    recommended_action=f"Reduce weight of {algo} in ensemble or exclude",
                ))

    # Pattern 4: Gate override patterns
    overridden_accepts = [
        r for r in negative_records
        if r.gate_decision == "accept" and r.feedback_type == DiscoveryFeedbackType.GATE_OVERRIDE
    ]
    if len(overridden_accepts) >= min_frequency:
        avg_confidence = sum(r.gate_confidence for r in overridden_accepts) / len(overridden_accepts)
        patterns.append(DiscoveryFailurePattern(
            pattern_id="accept_threshold_too_low",
            pattern_type="configuration",
            description=f"ACCEPT gate decisions being overridden (avg confidence: {avg_confidence:.2f})",
            frequency=len(overridden_accepts),
            confidence=len(overridden_accepts) / len(negative_records),
            trigger_conditions={"gate_decision": "accept", "avg_confidence": avg_confidence},
            affected_algorithms=["all"],
            recommended_action=f"Increase ACCEPT threshold from 0.8 to {avg_confidence + 0.1:.2f}",
        ))

    return patterns
```

### Parameter Recommender

Generate recommendations for discovery parameter adjustments:

```python
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ParameterRecommendation:
    """Recommendation for parameter adjustment."""
    parameter: str
    current_value: Any
    recommended_value: Any
    confidence: float
    reasoning: str
    expected_improvement: str


def generate_parameter_recommendations(
    accuracy_metrics: DiscoveryAccuracyMetrics,
    failure_patterns: List[DiscoveryFailurePattern],
    current_config: Dict[str, Any],
) -> List[ParameterRecommendation]:
    """
    Generate parameter adjustment recommendations.

    Args:
        accuracy_metrics: Current accuracy metrics
        failure_patterns: Identified failure patterns
        current_config: Current discovery configuration

    Returns:
        List of parameter recommendations
    """
    recommendations = []

    # Recommendation 1: Ensemble threshold adjustment
    if accuracy_metrics.overall_accuracy < 0.7:
        current_threshold = current_config.get("ensemble_threshold", 0.5)
        recommendations.append(ParameterRecommendation(
            parameter="ensemble_threshold",
            current_value=current_threshold,
            recommended_value=min(current_threshold + 0.1, 0.8),
            confidence=0.8,
            reasoning=f"Overall accuracy ({accuracy_metrics.overall_accuracy:.1%}) below 70% target",
            expected_improvement="Reduce false positive edges by requiring higher agreement",
        ))

    # Recommendation 2: Algorithm selection based on performance
    best_algo = max(
        accuracy_metrics.algorithm_accuracy.items(),
        key=lambda x: x[1],
        default=(None, 0.0)
    )
    worst_algo = min(
        accuracy_metrics.algorithm_accuracy.items(),
        key=lambda x: x[1],
        default=(None, 1.0)
    )

    if best_algo[0] and worst_algo[0] and best_algo[1] - worst_algo[1] > 0.2:
        current_algos = current_config.get("algorithms", ["ges", "pc"])
        if worst_algo[0] in current_algos and len(current_algos) > 1:
            recommendations.append(ParameterRecommendation(
                parameter="algorithms",
                current_value=current_algos,
                recommended_value=[a for a in current_algos if a != worst_algo[0]],
                confidence=0.7,
                reasoning=f"{worst_algo[0].upper()} accuracy ({worst_algo[1]:.1%}) significantly below {best_algo[0].upper()} ({best_algo[1]:.1%})",
                expected_improvement=f"Remove low-performing algorithm to improve ensemble quality",
            ))

    # Recommendation 3: Gate threshold adjustments based on patterns
    for pattern in failure_patterns:
        if pattern.pattern_id == "accept_threshold_too_low":
            current_accept = current_config.get("gate_thresholds", {}).get("accept", 0.8)
            recommendations.append(ParameterRecommendation(
                parameter="gate_thresholds.accept",
                current_value=current_accept,
                recommended_value=min(current_accept + 0.05, 0.95),
                confidence=pattern.confidence,
                reasoning=pattern.description,
                expected_improvement="Reduce false ACCEPT decisions requiring expert review",
            ))

    # Recommendation 4: Alpha (significance level) for statistical tests
    if accuracy_metrics.edge_precision < 0.6:
        current_alpha = current_config.get("alpha", 0.05)
        recommendations.append(ParameterRecommendation(
            parameter="alpha",
            current_value=current_alpha,
            recommended_value=max(current_alpha / 2, 0.01),
            confidence=0.6,
            reasoning=f"Edge precision ({accuracy_metrics.edge_precision:.1%}) indicates too many false positives",
            expected_improvement="Stricter significance threshold reduces spurious edge detection",
        ))

    return recommendations
```

### Usage Example

```python
from src.agents.feedback_learner.discovery_feedback import (
    collect_discovery_feedback,
    compute_discovery_accuracy,
    detect_failure_patterns,
    generate_parameter_recommendations,
    DiscoveryFeedbackType,
)


async def process_discovery_feedback(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process discovery feedback and generate recommendations.

    Args:
        state: Current agent state with discovery results and feedback

    Returns:
        Updated state with accuracy metrics and recommendations
    """
    # Collect new feedback
    if state.get("new_discovery_feedback"):
        feedback = collect_discovery_feedback(
            state=state,
            feedback_type=DiscoveryFeedbackType(state["new_discovery_feedback"]["type"]),
            is_positive=state["new_discovery_feedback"]["positive"],
            corrected_edges=state["new_discovery_feedback"].get("corrections"),
            notes=state["new_discovery_feedback"].get("notes"),
        )

        # Store feedback
        existing_records = state.get("discovery_feedback_records", [])
        existing_records.append(feedback.__dict__)
        state["discovery_feedback_records"] = existing_records

    # Compute accuracy metrics
    records = state.get("discovery_feedback_records", [])
    if records:
        metrics = compute_discovery_accuracy(
            [DiscoveryFeedbackRecord(**r) for r in records]
        )
        state["discovery_accuracy_metrics"] = {
            "overall_accuracy": metrics.overall_accuracy,
            "algorithm_accuracy": metrics.algorithm_accuracy,
            "gate_decision_accuracy": metrics.gate_decision_accuracy,
            "edge_precision": metrics.edge_precision,
            "edge_recall": metrics.edge_recall,
        }

        # Detect failure patterns
        patterns = detect_failure_patterns(
            [DiscoveryFeedbackRecord(**r) for r in records]
        )
        state["discovery_failure_patterns"] = [p.__dict__ for p in patterns]

        # Generate recommendations
        current_config = state.get("discovery_config", {
            "ensemble_threshold": 0.5,
            "algorithms": ["ges", "pc"],
            "alpha": 0.05,
        })
        recommendations = generate_parameter_recommendations(
            metrics, patterns, current_config
        )
        state["discovery_parameter_recommendations"] = [r.__dict__ for r in recommendations]

        # Rank algorithms by performance
        state["discovery_algorithm_rankings"] = sorted(
            metrics.algorithm_accuracy.items(),
            key=lambda x: x[1],
            reverse=True,
        )

    return state
```

### Configuration

```yaml
# config/feedback_learner_discovery.yaml

discovery_feedback:
  # Minimum records before pattern detection
  min_records_for_patterns: 10

  # Pattern detection thresholds
  pattern_detection:
    min_frequency: 3
    min_confidence: 0.7

  # Accuracy targets
  accuracy_targets:
    overall: 0.8
    edge_precision: 0.75
    edge_recall: 0.7

  # Recommendation triggers
  recommendation_triggers:
    accuracy_below: 0.7
    precision_below: 0.6
    algorithm_gap: 0.2  # Difference between best and worst

  # Feedback retention
  retention:
    max_records: 1000
    max_age_days: 90
```

### Testing Requirements

```python
# tests/unit/test_agents/test_feedback_learner/test_discovery_feedback.py

class TestDiscoveryFeedbackCollection:
    """Test feedback collection functions."""

    def test_collect_user_correction_feedback(self):
        """Test collecting user correction feedback."""
        pass

    def test_collect_expert_review_feedback(self):
        """Test collecting expert review feedback."""
        pass

    def test_collect_gate_override_feedback(self):
        """Test collecting gate override feedback."""
        pass


class TestDiscoveryAccuracyMetrics:
    """Test accuracy computation."""

    def test_overall_accuracy_computation(self):
        """Test overall accuracy calculation."""
        pass

    def test_algorithm_accuracy_tracking(self):
        """Test per-algorithm accuracy tracking."""
        pass

    def test_edge_precision_recall(self):
        """Test edge-level precision/recall."""
        pass


class TestFailurePatternDetection:
    """Test pattern detection."""

    def test_low_threshold_pattern_detection(self):
        """Test detection of low ensemble threshold failures."""
        pass

    def test_single_algorithm_pattern_detection(self):
        """Test detection of single algorithm failures."""
        pass

    def test_algorithm_specific_pattern_detection(self):
        """Test algorithm-specific failure detection."""
        pass


class TestParameterRecommendations:
    """Test recommendation generation."""

    def test_threshold_increase_recommendation(self):
        """Test ensemble threshold increase recommendation."""
        pass

    def test_algorithm_removal_recommendation(self):
        """Test poor algorithm removal recommendation."""
        pass

    def test_alpha_adjustment_recommendation(self):
        """Test significance level adjustment recommendation."""
        pass
```

### Integration with Self-Improvement Loop

The discovery feedback loop integrates with the broader self-improvement system:

1. **Feedback Collection**: Automatically captured when users correct discovered DAGs
2. **Pattern Learning**: DSPy signatures learn from accumulated feedback patterns
3. **Parameter Tuning**: GEPA optimizer adjusts discovery parameters based on metrics
4. **Continuous Monitoring**: Accuracy metrics tracked via Opik dashboards

```python
# Integration with existing DSPy optimization
DISCOVERY_FEEDBACK_SIGNATURES = {
    "DiscoveryParameterSignature": {
        "input_fields": ["accuracy_metrics", "failure_patterns", "current_config"],
        "output_fields": ["recommended_config"],
        "description": "Recommend discovery configuration adjustments",
    },
    "FailurePatternSignature": {
        "input_fields": ["feedback_records", "discovery_metadata"],
        "output_fields": ["identified_patterns", "pattern_confidence"],
        "description": "Identify patterns in discovery failures",
    },
}
```
