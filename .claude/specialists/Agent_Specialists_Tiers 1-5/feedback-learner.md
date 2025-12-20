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
├── state.py              # LangGraph state definitions
├── graph.py              # LangGraph assembly
├── nodes/
│   ├── feedback_collector.py  # Gather feedback data
│   ├── pattern_analyzer.py    # Deep pattern analysis
│   ├── learning_extractor.py  # Generate improvements
│   └── knowledge_updater.py   # Apply updates
├── feedback_types.py     # Feedback type definitions
└── knowledge_stores.py   # Knowledge base interfaces
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
