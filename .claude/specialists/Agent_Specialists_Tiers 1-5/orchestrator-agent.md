# Tier 1: Orchestrator Agent Specialist

## Agent Classification

| Property | Value |
|----------|-------|
| **Tier** | 1 (Coordination) |
| **Agent Type** | Standard (Fast Path) |
| **Model Tier** | Haiku/Sonnet |
| **Latency Tolerance** | <2s (strict) |
| **Critical Path** | Yes - All queries pass through |

## Domain Scope

You are the specialist for the Tier 1 Orchestrator Agent:
- `src/agents/orchestrator/` - Central coordination hub

The Orchestrator is the **entry point for all queries**. It must be extremely fast and reliable.

## Design Principles

### Fast Path Optimization
The Orchestrator is optimized for speed over depth:
- Minimal reasoning tokens
- Simple classification heuristics
- No extended thinking
- Parallel agent dispatch where possible
- Sub-2-second response time requirement

### Responsibilities
1. **Intent Classification** - Determine query type and required agents
2. **Agent Routing** - Dispatch to appropriate specialist agents
3. **Context Assembly** - Prepare minimal context for downstream agents
4. **Response Synthesis** - Combine multi-agent outputs into coherent response
5. **Error Recovery** - Handle agent failures gracefully

## Agent Architecture Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR AGENT                          │
│                      (Fast Path - <2s)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   INTENT    │───►│   ROUTER    │───►│  DISPATCH   │         │
│  │   PARSER    │    │             │    │             │         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│                                               │                 │
│                          ┌────────────────────┼────────────┐    │
│                          ▼                    ▼            ▼    │
│                     [Tier 2]             [Tier 3]     [Tier 4]  │
│                                               │                 │
│                          └────────────────────┼────────────┘    │
│                                               ▼                 │
│                                    ┌─────────────────┐          │
│                                    │    SYNTHESIZER  │          │
│                                    └─────────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
orchestrator/
├── agent.py              # Main OrchestratorAgent class
├── intent_classifier.py  # Fast intent classification
├── router.py             # Agent routing logic
├── dispatcher.py         # Parallel agent dispatch
├── synthesizer.py        # Response combination
├── state.py              # LangGraph state definitions
├── graph.py              # LangGraph assembly
└── prompts.py            # Minimal prompts for classification
```

## LangGraph State Definition

```python
# src/agents/orchestrator/state.py

from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
from datetime import datetime
import operator

class IntentClassification(TypedDict):
    """Fast intent classification result"""
    primary_intent: Literal[
        "causal_effect",      # → Causal Impact Agent
        "performance_gap",    # → Gap Analyzer Agent
        "segment_analysis",   # → Heterogeneous Optimizer
        "experiment_design",  # → Experiment Designer
        "prediction",         # → Prediction Synthesizer
        "resource_allocation",# → Resource Optimizer
        "explanation",        # → Explainer Agent
        "system_health",      # → Health Score Agent
        "drift_check",        # → Drift Monitor Agent
        "feedback",           # → Feedback Learner
        "general"             # → Direct response
    ]
    confidence: float
    secondary_intents: List[str]
    requires_multi_agent: bool

class AgentDispatch(TypedDict):
    """Agent dispatch specification"""
    agent_name: str
    priority: int  # 1 = highest
    parameters: Dict[str, Any]
    timeout_ms: int
    fallback_agent: Optional[str]

class AgentResult(TypedDict):
    """Result from dispatched agent"""
    agent_name: str
    success: bool
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    latency_ms: int

class OrchestratorState(TypedDict):
    """Complete state for Orchestrator agent"""
    
    # === INPUT ===
    query: str
    user_context: Dict[str, Any]  # expertise level, preferences
    conversation_history: Optional[List[Dict]]
    
    # === CLASSIFICATION ===
    intent: Optional[IntentClassification]
    entities_extracted: Optional[Dict[str, List[str]]]
    
    # === ROUTING ===
    dispatch_plan: Optional[List[AgentDispatch]]
    parallel_groups: Optional[List[List[str]]]  # Agents that can run in parallel
    
    # === EXECUTION ===
    agent_results: Annotated[List[AgentResult], operator.add]
    current_phase: Literal["classifying", "routing", "dispatching", "synthesizing", "complete"]
    
    # === OUTPUT ===
    synthesized_response: Optional[str]
    recommendations: Optional[List[Dict[str, Any]]]
    follow_up_suggestions: Optional[List[str]]
    
    # === METADATA ===
    start_time: str
    total_latency_ms: int
    classification_latency_ms: int
    routing_latency_ms: int
    dispatch_latency_ms: int
    synthesis_latency_ms: int
    
    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    fallback_used: bool
    status: Literal["pending", "processing", "completed", "failed"]
```

## Node Implementations

### Intent Classifier (Fast)

```python
# src/agents/orchestrator/intent_classifier.py

import time
import re
from typing import Dict, Any, List
from langchain_anthropic import ChatAnthropic

from .state import OrchestratorState, IntentClassification

class IntentClassifierNode:
    """
    Fast intent classification - optimized for <500ms
    Uses pattern matching first, LLM only for ambiguous cases
    """
    
    # Pattern-based classification for common queries
    INTENT_PATTERNS = {
        "causal_effect": [
            r"what.*(caus|impact|effect|driv|lead|result)",
            r"why.*(increase|decrease|change|drop|rise)",
            r"how does.*affect",
            r"what drives",
            r"attribution"
        ],
        "performance_gap": [
            r"(gap|opportunit|underperform|potential|improve)",
            r"roi.*(opportun|analys)",
            r"where.*underperform",
            r"untapped"
        ],
        "segment_analysis": [
            r"(segment|cohort|group|heterogen)",
            r"which.*(respond|perform).*(best|better)",
            r"cate|treatment effect.*by",
            r"differentiat.*strategy"
        ],
        "experiment_design": [
            r"(design|run|plan).*(experiment|test|trial)",
            r"a/b test",
            r"sample size",
            r"hypothesis.*test"
        ],
        "prediction": [
            r"predict|forecast|project",
            r"what will|expected",
            r"likelihood|probability"
        ],
        "resource_allocation": [
            r"(allocat|optimi|distribut).*(resource|budget|rep)",
            r"where.*invest",
            r"prioriti"
        ],
        "explanation": [
            r"explain|clarify|what does.*mean",
            r"help.*understand",
            r"break down"
        ],
        "system_health": [
            r"system.*(health|status)",
            r"model.*perform",
            r"pipeline.*status"
        ],
        "drift_check": [
            r"drift|shift|distribution.*change",
            r"data quality",
            r"model.*degrad"
        ],
        "feedback": [
            r"feedback|learn.*from",
            r"improve.*based on"
        ]
    }
    
    def __init__(self):
        # Use Haiku for fast classification
        self.llm = ChatAnthropic(
            model="claude-haiku-4-20250414",
            max_tokens=256,
            timeout=2
        )
    
    async def execute(self, state: OrchestratorState) -> OrchestratorState:
        start_time = time.time()
        
        query = state["query"].lower()
        
        # Try pattern matching first (fastest)
        pattern_result = self._pattern_classify(query)
        
        if pattern_result["confidence"] >= 0.8:
            intent = pattern_result
        else:
            # Fall back to LLM for ambiguous cases
            intent = await self._llm_classify(state["query"])
        
        classification_time = int((time.time() - start_time) * 1000)
        
        return {
            **state,
            "intent": intent,
            "classification_latency_ms": classification_time,
            "current_phase": "routing"
        }
    
    def _pattern_classify(self, query: str) -> IntentClassification:
        """Fast pattern-based classification"""
        scores = {}
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
            scores[intent] = score / len(patterns) if patterns else 0
        
        if not scores or max(scores.values()) == 0:
            return IntentClassification(
                primary_intent="general",
                confidence=0.5,
                secondary_intents=[],
                requires_multi_agent=False
            )
        
        primary = max(scores, key=scores.get)
        confidence = scores[primary]
        
        # Get secondary intents
        secondary = [k for k, v in scores.items() if v > 0 and k != primary]
        
        return IntentClassification(
            primary_intent=primary,
            confidence=min(confidence * 1.5, 1.0),  # Scale up confidence
            secondary_intents=secondary[:2],
            requires_multi_agent=len(secondary) > 0 and scores[secondary[0]] > 0.3
        )
    
    async def _llm_classify(self, query: str) -> IntentClassification:
        """LLM-based classification for ambiguous cases"""
        
        prompt = f"""Classify this pharmaceutical analytics query into ONE primary intent.

Query: "{query}"

Intents:
- causal_effect: Questions about cause and effect, impact, attribution
- performance_gap: ROI opportunities, underperformance, potential improvements
- segment_analysis: Segment-specific effects, CATE, cohort analysis
- experiment_design: A/B tests, experiment planning, sample size
- prediction: Forecasting, projections, likelihood estimates
- resource_allocation: Budget/resource optimization, prioritization
- explanation: Clarifying results, interpreting findings
- system_health: Model/pipeline status, system performance
- drift_check: Data/model drift, distribution changes
- feedback: Learning from outcomes, improvement suggestions
- general: Other/unclear

Respond with ONLY a JSON object:
{{"primary_intent": "<intent>", "confidence": <0.0-1.0>, "requires_multi_agent": <bool>}}"""

        try:
            response = await self.llm.ainvoke(prompt)
            import json
            result = json.loads(response.content)
            return IntentClassification(
                primary_intent=result.get("primary_intent", "general"),
                confidence=result.get("confidence", 0.5),
                secondary_intents=[],
                requires_multi_agent=result.get("requires_multi_agent", False)
            )
        except Exception:
            return IntentClassification(
                primary_intent="general",
                confidence=0.3,
                secondary_intents=[],
                requires_multi_agent=False
            )
```

### Router Node

```python
# src/agents/orchestrator/router.py

import time
from typing import List, Dict, Any

from .state import OrchestratorState, AgentDispatch

class RouterNode:
    """
    Fast routing decisions based on intent classification
    No LLM calls - pure logic
    """
    
    # Agent capabilities mapping
    INTENT_TO_AGENTS = {
        "causal_effect": [
            AgentDispatch(
                agent_name="causal_impact",
                priority=1,
                parameters={"interpretation_depth": "standard"},
                timeout_ms=30000,
                fallback_agent="explainer"
            )
        ],
        "performance_gap": [
            AgentDispatch(
                agent_name="gap_analyzer",
                priority=1,
                parameters={},
                timeout_ms=20000,
                fallback_agent=None
            )
        ],
        "segment_analysis": [
            AgentDispatch(
                agent_name="heterogeneous_optimizer",
                priority=1,
                parameters={},
                timeout_ms=25000,
                fallback_agent="gap_analyzer"
            )
        ],
        "experiment_design": [
            AgentDispatch(
                agent_name="experiment_designer",
                priority=1,
                parameters={"preregistration_formality": "medium"},
                timeout_ms=60000,
                fallback_agent=None
            )
        ],
        "prediction": [
            AgentDispatch(
                agent_name="prediction_synthesizer",
                priority=1,
                parameters={},
                timeout_ms=15000,
                fallback_agent=None
            )
        ],
        "resource_allocation": [
            AgentDispatch(
                agent_name="resource_optimizer",
                priority=1,
                parameters={},
                timeout_ms=20000,
                fallback_agent=None
            )
        ],
        "explanation": [
            AgentDispatch(
                agent_name="explainer",
                priority=1,
                parameters={"depth": "standard"},
                timeout_ms=45000,
                fallback_agent=None
            )
        ],
        "system_health": [
            AgentDispatch(
                agent_name="health_score",
                priority=1,
                parameters={},
                timeout_ms=5000,
                fallback_agent=None
            )
        ],
        "drift_check": [
            AgentDispatch(
                agent_name="drift_monitor",
                priority=1,
                parameters={},
                timeout_ms=10000,
                fallback_agent=None
            )
        ],
        "feedback": [
            AgentDispatch(
                agent_name="feedback_learner",
                priority=1,
                parameters={},
                timeout_ms=30000,
                fallback_agent=None
            )
        ]
    }
    
    # Multi-agent patterns
    MULTI_AGENT_PATTERNS = {
        ("causal_effect", "segment_analysis"): [
            ("causal_impact", 1), ("heterogeneous_optimizer", 2)
        ],
        ("performance_gap", "resource_allocation"): [
            ("gap_analyzer", 1), ("resource_optimizer", 2)
        ],
        ("prediction", "explanation"): [
            ("prediction_synthesizer", 1), ("explainer", 2)
        ]
    }
    
    async def execute(self, state: OrchestratorState) -> OrchestratorState:
        start_time = time.time()
        
        intent = state["intent"]
        dispatch_plan = []
        parallel_groups = []
        
        # Check for multi-agent patterns
        if intent["requires_multi_agent"] and intent["secondary_intents"]:
            key = (intent["primary_intent"], intent["secondary_intents"][0])
            if key in self.MULTI_AGENT_PATTERNS:
                pattern = self.MULTI_AGENT_PATTERNS[key]
                for agent_name, priority in pattern:
                    dispatch_plan.append(
                        self._get_dispatch_for_agent(agent_name, priority)
                    )
                # Group by priority for parallel execution
                parallel_groups = self._group_by_priority(dispatch_plan)
        
        # Single agent dispatch
        if not dispatch_plan:
            primary = intent["primary_intent"]
            if primary in self.INTENT_TO_AGENTS:
                dispatch_plan = self.INTENT_TO_AGENTS[primary]
            else:
                # Default to explainer for general queries
                dispatch_plan = [
                    AgentDispatch(
                        agent_name="explainer",
                        priority=1,
                        parameters={"depth": "minimal"},
                        timeout_ms=30000,
                        fallback_agent=None
                    )
                ]
        
        routing_time = int((time.time() - start_time) * 1000)
        
        return {
            **state,
            "dispatch_plan": dispatch_plan,
            "parallel_groups": parallel_groups or [[d["agent_name"] for d in dispatch_plan]],
            "routing_latency_ms": routing_time,
            "current_phase": "dispatching"
        }
    
    def _get_dispatch_for_agent(self, agent_name: str, priority: int) -> AgentDispatch:
        """Get dispatch config for a specific agent"""
        for intent_agents in self.INTENT_TO_AGENTS.values():
            for dispatch in intent_agents:
                if dispatch["agent_name"] == agent_name:
                    return AgentDispatch(
                        **{**dispatch, "priority": priority}
                    )
        # Default dispatch
        return AgentDispatch(
            agent_name=agent_name,
            priority=priority,
            parameters={},
            timeout_ms=30000,
            fallback_agent=None
        )
    
    def _group_by_priority(self, dispatches: List[AgentDispatch]) -> List[List[str]]:
        """Group agents by priority for parallel execution"""
        from collections import defaultdict
        groups = defaultdict(list)
        for d in dispatches:
            groups[d["priority"]].append(d["agent_name"])
        return [groups[p] for p in sorted(groups.keys())]
```

### Dispatcher Node

```python
# src/agents/orchestrator/dispatcher.py

import asyncio
import time
from typing import Dict, Any, List

from .state import OrchestratorState, AgentResult, AgentDispatch

class DispatcherNode:
    """
    Parallel agent dispatch with timeout handling
    """
    
    def __init__(self, agent_registry: Dict[str, Any]):
        """
        Args:
            agent_registry: Dict mapping agent_name to agent instance
        """
        self.agents = agent_registry
    
    async def execute(self, state: OrchestratorState) -> OrchestratorState:
        start_time = time.time()
        
        dispatch_plan = state["dispatch_plan"]
        parallel_groups = state["parallel_groups"]
        all_results = []
        
        # Execute each parallel group sequentially
        for group in parallel_groups:
            group_dispatches = [
                d for d in dispatch_plan if d["agent_name"] in group
            ]
            
            # Run agents in parallel within group
            tasks = [
                self._dispatch_agent(d, state)
                for d in group_dispatches
            ]
            
            group_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for dispatch, result in zip(group_dispatches, group_results):
                if isinstance(result, Exception):
                    all_results.append(AgentResult(
                        agent_name=dispatch["agent_name"],
                        success=False,
                        result=None,
                        error=str(result),
                        latency_ms=0
                    ))
                    
                    # Try fallback if available
                    if dispatch["fallback_agent"]:
                        fallback_result = await self._dispatch_fallback(
                            dispatch["fallback_agent"], state
                        )
                        all_results.append(fallback_result)
                else:
                    all_results.append(result)
        
        dispatch_time = int((time.time() - start_time) * 1000)
        
        return {
            **state,
            "agent_results": all_results,
            "dispatch_latency_ms": dispatch_time,
            "current_phase": "synthesizing"
        }
    
    async def _dispatch_agent(
        self, 
        dispatch: AgentDispatch, 
        state: OrchestratorState
    ) -> AgentResult:
        """Dispatch to a single agent with timeout"""
        
        agent_name = dispatch["agent_name"]
        start_time = time.time()
        
        if agent_name not in self.agents:
            return AgentResult(
                agent_name=agent_name,
                success=False,
                result=None,
                error=f"Agent '{agent_name}' not found in registry",
                latency_ms=0
            )
        
        agent = self.agents[agent_name]
        timeout_ms = dispatch["timeout_ms"]
        
        try:
            # Prepare agent input
            agent_input = self._prepare_agent_input(state, dispatch)
            
            # Execute with timeout
            result = await asyncio.wait_for(
                agent.analyze(agent_input),
                timeout=timeout_ms / 1000
            )
            
            latency = int((time.time() - start_time) * 1000)
            
            return AgentResult(
                agent_name=agent_name,
                success=True,
                result=result,
                error=None,
                latency_ms=latency
            )
            
        except asyncio.TimeoutError:
            return AgentResult(
                agent_name=agent_name,
                success=False,
                result=None,
                error=f"Agent timed out after {timeout_ms}ms",
                latency_ms=timeout_ms
            )
        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            return AgentResult(
                agent_name=agent_name,
                success=False,
                result=None,
                error=str(e),
                latency_ms=latency
            )
    
    def _prepare_agent_input(
        self, 
        state: OrchestratorState, 
        dispatch: AgentDispatch
    ) -> Dict[str, Any]:
        """Prepare input for specific agent"""
        return {
            "query": state["query"],
            "user_context": state["user_context"],
            "parameters": dispatch["parameters"]
        }
    
    async def _dispatch_fallback(
        self, 
        agent_name: str, 
        state: OrchestratorState
    ) -> AgentResult:
        """Dispatch to fallback agent"""
        fallback_dispatch = AgentDispatch(
            agent_name=agent_name,
            priority=99,
            parameters={},
            timeout_ms=30000,
            fallback_agent=None
        )
        return await self._dispatch_agent(fallback_dispatch, state)
```

### Synthesizer Node

```python
# src/agents/orchestrator/synthesizer.py

import time
from typing import Dict, Any, List
from langchain_anthropic import ChatAnthropic

from .state import OrchestratorState, AgentResult

class SynthesizerNode:
    """
    Combine multi-agent results into coherent response
    Uses fast model for synthesis
    """
    
    def __init__(self):
        self.llm = ChatAnthropic(
            model="claude-haiku-4-20250414",
            max_tokens=1024,
            timeout=5
        )
    
    async def execute(self, state: OrchestratorState) -> OrchestratorState:
        start_time = time.time()
        
        results = state["agent_results"]
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        # Single agent - return directly
        if len(successful_results) == 1:
            synthesized = self._extract_response(successful_results[0])
        elif len(successful_results) > 1:
            # Multi-agent synthesis
            synthesized = await self._synthesize_multiple(successful_results)
        else:
            # All failed
            synthesized = self._generate_error_response(failed_results)
        
        synthesis_time = int((time.time() - start_time) * 1000)
        total_latency = (
            state.get("classification_latency_ms", 0) +
            state.get("routing_latency_ms", 0) +
            state.get("dispatch_latency_ms", 0) +
            synthesis_time
        )
        
        return {
            **state,
            "synthesized_response": synthesized["response"],
            "recommendations": synthesized.get("recommendations", []),
            "follow_up_suggestions": synthesized.get("follow_ups", []),
            "synthesis_latency_ms": synthesis_time,
            "total_latency_ms": total_latency,
            "current_phase": "complete",
            "status": "completed" if successful_results else "failed"
        }
    
    def _extract_response(self, result: AgentResult) -> Dict[str, Any]:
        """Extract response from single agent result"""
        agent_output = result["result"]
        
        return {
            "response": agent_output.get("narrative", agent_output.get("response", str(agent_output))),
            "recommendations": agent_output.get("recommendations", []),
            "follow_ups": agent_output.get("follow_up_suggestions", [])
        }
    
    async def _synthesize_multiple(self, results: List[AgentResult]) -> Dict[str, Any]:
        """Synthesize multiple agent results"""
        
        summaries = []
        all_recommendations = []
        
        for result in results:
            agent_output = result["result"]
            summaries.append(f"**{result['agent_name']}**: {agent_output.get('narrative', '')[:500]}")
            all_recommendations.extend(agent_output.get("recommendations", []))
        
        synthesis_prompt = f"""Synthesize these analysis results into a coherent response.
Be concise and actionable.

{chr(10).join(summaries)}

Provide a unified 2-3 paragraph response that:
1. Highlights the key findings
2. Connects insights across analyses
3. Provides clear recommendations"""

        try:
            response = await self.llm.ainvoke(synthesis_prompt)
            return {
                "response": response.content,
                "recommendations": all_recommendations[:5],
                "follow_ups": ["Explore segment-specific effects", "Design follow-up experiment"]
            }
        except Exception:
            # Fallback: concatenate responses
            return {
                "response": "\n\n".join([s.split(": ", 1)[1] for s in summaries]),
                "recommendations": all_recommendations[:5],
                "follow_ups": []
            }
    
    def _generate_error_response(self, failed_results: List[AgentResult]) -> Dict[str, Any]:
        """Generate error response when all agents fail"""
        
        errors = [f"- {r['agent_name']}: {r['error']}" for r in failed_results]
        
        return {
            "response": f"I was unable to complete the analysis due to the following errors:\n{chr(10).join(errors)}\n\nPlease try again or rephrase your question.",
            "recommendations": [],
            "follow_ups": ["Simplify your question", "Check system health"]
        }
```

## Graph Assembly

```python
# src/agents/orchestrator/graph.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import OrchestratorState
from .intent_classifier import IntentClassifierNode
from .router import RouterNode
from .dispatcher import DispatcherNode
from .synthesizer import SynthesizerNode

def build_orchestrator_graph(
    agent_registry: dict,
    enable_checkpointing: bool = False
):
    """
    Build the Orchestrator agent graph
    
    Architecture:
        [classify] → [route] → [dispatch] → [synthesize] → END
    
    Total latency target: <2 seconds for classification + routing
    """
    
    # Initialize nodes
    classifier = IntentClassifierNode()
    router = RouterNode()
    dispatcher = DispatcherNode(agent_registry)
    synthesizer = SynthesizerNode()
    
    # Build graph
    workflow = StateGraph(OrchestratorState)
    
    # Add nodes
    workflow.add_node("classify", classifier.execute)
    workflow.add_node("route", router.execute)
    workflow.add_node("dispatch", dispatcher.execute)
    workflow.add_node("synthesize", synthesizer.execute)
    
    # Linear flow (no conditionals for speed)
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "route")
    workflow.add_edge("route", "dispatch")
    workflow.add_edge("dispatch", "synthesize")
    workflow.add_edge("synthesize", END)
    
    # Compile
    if enable_checkpointing:
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    return workflow.compile()
```

## Integration Contracts

### Input Contract (from User/API)
```python
class OrchestratorInput(BaseModel):
    query: str
    user_id: str
    user_expertise: Literal["executive", "analyst", "data_scientist"] = "analyst"
    brand: Optional[Literal["Remibrutinib", "Fabhalta", "Kisqali"]] = None
    session_id: Optional[str] = None
```

### Output Contract (to User/API)
```python
class OrchestratorOutput(BaseModel):
    response: str
    recommendations: List[Dict[str, Any]]
    follow_up_suggestions: List[str]
    agents_used: List[str]
    total_latency_ms: int
    confidence: float
```

### Agent Registry Contract
```python
class AgentRegistryEntry(TypedDict):
    agent: BaseAgent  # Must implement analyze(input) -> output
    tier: int
    capabilities: List[str]
    max_timeout_ms: int
```

## Testing Requirements

```
tests/unit/test_agents/test_orchestrator/
├── test_intent_classifier.py   # Pattern matching and LLM classification
├── test_router.py              # Routing logic coverage
├── test_dispatcher.py          # Parallel dispatch and timeout handling
├── test_synthesizer.py         # Multi-agent synthesis
└── test_integration.py         # End-to-end orchestration
```

### Performance Requirements
- Intent classification: <500ms (pattern match) or <1500ms (LLM fallback)
- Routing: <50ms (pure logic)
- Total orchestration overhead: <2000ms (excluding agent execution)

### Test Cases
1. Single-intent queries route to correct agent
2. Multi-intent queries trigger multi-agent dispatch
3. Agent timeout triggers fallback
4. All-agent failure produces graceful error response
5. Parallel dispatch respects priority groups

## Handoff Format

```yaml
orchestrator_handoff:
  intent: <classified intent>
  confidence: <0.0-1.0>
  dispatched_agents:
    - agent: <agent_name>
      status: <success|failed|timeout>
      latency_ms: <int>
  synthesized_response: <combined response>
  total_latency_ms: <int>
  recommendations:
    - <recommendation 1>
    - <recommendation 2>
  follow_up_suggestions:
    - <suggestion 1>
```

## Performance Optimization Notes

1. **Pattern Matching First**: Use regex patterns before LLM for classification
2. **Haiku for Speed**: Always use Claude Haiku for orchestrator tasks
3. **Parallel Dispatch**: Agents at same priority level run concurrently
4. **Early Termination**: If first agent provides complete answer, skip synthesis
5. **Minimal Context**: Only pass essential information to downstream agents
6. **No Extended Thinking**: Orchestrator never uses extended reasoning

---

## Cognitive RAG DSPy Integration

The Orchestrator integrates with the Cognitive RAG system's 4-phase workflow. DSPy signatures optimize routing decisions through continuous learning.

### Phase 3: Agent Routing Integration

The Orchestrator receives routing decisions from CognitiveRAG's **Phase 3 (Agent Phase)**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COGNITIVE RAG → ORCHESTRATOR FLOW                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CognitiveRAG                           Orchestrator                     │
│  ┌─────────────────┐                    ┌─────────────────┐              │
│  │ Phase 3: Agent  │                    │                 │              │
│  │                 │   AgentRouting     │  Intent Match   │              │
│  │ AgentRouting    │───────────────────►│  Validation     │              │
│  │ Signature       │   Signal           │                 │              │
│  │                 │                    │  Dispatch Plan  │              │
│  └─────────────────┘                    └─────────────────┘              │
│         │                                       │                        │
│         ▼                                       ▼                        │
│  ┌─────────────────┐                    ┌─────────────────┐              │
│  │ Visualization   │                    │  Parallel       │              │
│  │ Config          │◄───────────────────│  Agent Dispatch │              │
│  │ Signature       │   Response         │                 │              │
│  └─────────────────┘                    └─────────────────┘              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### DSPy Signatures Consumed by Orchestrator

#### 1. AgentRoutingSignature (from CognitiveRAG Phase 3)

```python
class AgentRoutingSignature(dspy.Signature):
    """Determine which E2I agent should handle the synthesized query."""

    synthesized_query: str = dspy.InputField(desc="The user's refined query")
    detected_intent: str = dspy.InputField(desc="Classified intent from Phase 1")
    evidence_summary: str = dspy.InputField(desc="Synthesized evidence from retrieval")
    available_agents: List[str] = dspy.InputField(desc="List of available agent names")

    # Outputs consumed by Orchestrator
    primary_agent: str = dspy.OutputField(desc="Primary agent to route query to")
    secondary_agents: List[str] = dspy.OutputField(desc="Secondary agents if multi-agent needed")
    routing_confidence: float = dspy.OutputField(desc="Confidence in routing decision 0.0-1.0")
    parameters: Dict[str, Any] = dspy.OutputField(desc="Agent-specific parameters")
```

**Integration Point**: Orchestrator's `IntentClassifierNode` can optionally accept pre-classified intent from CognitiveRAG to bypass local classification.

```python
# src/agents/orchestrator/intent_classifier.py (integration mode)

async def execute(self, state: OrchestratorState) -> OrchestratorState:
    # Check for pre-routed query from CognitiveRAG
    if state.get("cognitive_routing"):
        routing = state["cognitive_routing"]
        return {
            **state,
            "intent": IntentClassification(
                primary_intent=routing["primary_agent"],
                confidence=routing["routing_confidence"],
                secondary_intents=routing.get("secondary_agents", []),
                requires_multi_agent=len(routing.get("secondary_agents", [])) > 0
            ),
            "classification_latency_ms": 0,  # Pre-computed
            "current_phase": "routing"
        }

    # Fall back to local classification
    return await self._local_classify(state)
```

### DSPy Training Signals from Orchestrator

The Orchestrator provides feedback signals for DSPy MIPROv2 optimization:

#### 1. Routing Accuracy Signals

```python
class OrchestratorTrainingSignal:
    """Training signal for CognitiveRAG's AgentRoutingSignature."""

    def __init__(self, state: OrchestratorState, result: OrchestratorOutput):
        self.query = state["query"]
        self.cognitive_routing = state.get("cognitive_routing")
        self.actual_agents_used = result.agents_used
        self.routing_success = self._compute_routing_success(result)
        self.user_satisfaction = None  # Set via feedback

    def _compute_routing_success(self, result) -> float:
        """Compute routing success score for DSPy optimization."""
        if not self.cognitive_routing:
            return 0.5  # No pre-routing, neutral signal

        predicted_agent = self.cognitive_routing["primary_agent"]
        actual_primary = result.agents_used[0] if result.agents_used else None

        if predicted_agent == actual_primary:
            return 1.0
        elif predicted_agent in result.agents_used:
            return 0.7  # Predicted was used, but not primary
        else:
            return 0.2  # Prediction mismatch

    def to_dspy_example(self) -> dspy.Example:
        """Convert to DSPy training example."""
        return dspy.Example(
            synthesized_query=self.query,
            detected_intent=self.cognitive_routing.get("detected_intent", ""),
            primary_agent=self.actual_agents_used[0],
            routing_confidence=self.routing_success
        ).with_inputs("synthesized_query", "detected_intent")
```

#### 2. Signal Collection for MIPROv2

```python
# Orchestrator emits training signals to feedback_learner

async def collect_training_signal(
    state: OrchestratorState,
    result: OrchestratorOutput
) -> Dict[str, Any]:
    """Collect training signal for DSPy optimization."""

    return {
        "signal_type": "orchestrator_routing",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": state["query"],
        "cognitive_routing": state.get("cognitive_routing"),
        "actual_dispatch": {
            "agents_used": result.agents_used,
            "total_latency_ms": result.total_latency_ms,
            "success": result.confidence > 0.5
        },
        "routing_accuracy": OrchestratorTrainingSignal(state, result).routing_success,
        # For MIPROv2 optimization
        "optimization_target": "agent_routing_accuracy"
    }
```

### CognitiveState Integration

When receiving queries from CognitiveRAG, the Orchestrator receives enriched context:

```python
class CognitiveEnrichedInput(TypedDict):
    """Input from CognitiveRAG to Orchestrator."""

    # Standard input
    query: str
    user_id: str

    # CognitiveRAG enrichments
    cognitive_state: CognitiveState
    routing_decision: Dict[str, Any]  # From AgentRoutingSignature
    evidence_context: List[Evidence]  # Retrieved evidence
    investigation_history: List[Dict]  # Multi-hop retrieval path
```

### Configuration

```yaml
# config/agents.yaml - Orchestrator DSPy integration

orchestrator:
  cognitive_rag_integration:
    enabled: true
    accept_pre_routing: true  # Accept routing from CognitiveRAG
    emit_training_signals: true  # Send signals to feedback_learner
    routing_confidence_threshold: 0.7  # Override if confidence < threshold

  training_signal_collection:
    buffer_size: 100
    flush_interval_seconds: 300
    target_agent: feedback_learner
```

### Testing Requirements for DSPy Integration

```
tests/unit/test_agents/test_orchestrator/
├── test_cognitive_routing_integration.py  # CognitiveRAG routing acceptance
├── test_training_signal_collection.py     # DSPy signal emission
└── test_routing_override.py               # Low confidence override logic
```

### Integration Test Cases

1. **Accept CognitiveRAG routing**: Orchestrator uses pre-computed routing when confidence > threshold
2. **Override low-confidence routing**: Falls back to local classification when confidence < 0.7
3. **Training signal emission**: Correctly emits routing accuracy signals for MIPROv2
4. **Multi-agent routing**: Handles secondary_agents from DSPy routing
5. **Visualization config passthrough**: Forwards VisualizationConfigSignature outputs to frontend
