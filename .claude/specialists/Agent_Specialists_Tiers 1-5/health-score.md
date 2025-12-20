# Tier 3: Health Score Agent Specialist

## Agent Classification

| Property | Value |
|----------|-------|
| **Tier** | 3 (Monitoring) |
| **Agent Type** | Standard (Fast Path) |
| **Model Tier** | Haiku |
| **Latency Tolerance** | Low (<5s) |
| **Critical Path** | No - monitoring agent |

## Domain Scope

You are the specialist for the Tier 3 Health Score Agent:
- `src/agents/health_score/` - System health metrics and monitoring

This is a **Fast Path Agent** optimized for:
- Quick health checks
- System status aggregation
- Dashboard metrics
- Zero LLM usage in critical path

## Design Principles

### Absolute Speed Priority
The Health Score agent must be the fastest:
- Pre-computed metrics where possible
- Simple aggregation logic
- No LLM calls
- Cached results with TTL

### Responsibilities
1. **Component Health** - Check status of all system components
2. **Model Health** - Monitor model performance metrics
3. **Data Pipeline Health** - Check data freshness and quality
4. **Composite Score** - Generate overall health score

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      HEALTH SCORE AGENT                          │
│                      (Fast Path - <5s)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐ │
│  │  COMPONENT │  │   MODEL    │  │   DATA     │  │   AGENT   │ │
│  │   HEALTH   │  │   HEALTH   │  │  PIPELINE  │  │  HEALTH   │ │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬─────┘ │
│        │               │               │               │        │
│        └───────────────┼───────────────┼───────────────┘        │
│                        ▼                                         │
│              ┌─────────────────┐                                │
│              │  SCORE COMPOSER │                                │
│              │  (Weighted Avg) │                                │
│              └─────────────────┘                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
health_score/
├── agent.py              # Main HealthScoreAgent class
├── state.py              # LangGraph state definitions
├── graph.py              # LangGraph assembly
├── nodes/
│   ├── component_health.py  # System component checks
│   ├── model_health.py      # Model performance metrics
│   ├── pipeline_health.py   # Data pipeline status
│   ├── agent_health.py      # Agent availability
│   └── score_composer.py    # Composite score calculation
└── metrics.py            # Metric definitions and thresholds
```

## LangGraph State Definition

```python
# src/agents/health_score/state.py

from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
from datetime import datetime
import operator

class ComponentStatus(TypedDict):
    """Status of a system component"""
    component_name: str
    status: Literal["healthy", "degraded", "unhealthy", "unknown"]
    latency_ms: Optional[int]
    last_check: str
    error_message: Optional[str]

class ModelMetrics(TypedDict):
    """Model performance metrics"""
    model_id: str
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    auc_roc: Optional[float]
    prediction_latency_p50_ms: Optional[int]
    prediction_latency_p99_ms: Optional[int]
    predictions_last_24h: int
    error_rate: float
    status: Literal["healthy", "degraded", "unhealthy"]

class PipelineStatus(TypedDict):
    """Data pipeline status"""
    pipeline_name: str
    last_run: str
    last_success: str
    rows_processed: int
    freshness_hours: float
    status: Literal["healthy", "stale", "failed"]

class AgentStatus(TypedDict):
    """Agent availability status"""
    agent_name: str
    tier: int
    available: bool
    avg_latency_ms: int
    success_rate: float
    last_invocation: str

class HealthScoreState(TypedDict):
    """Complete state for Health Score agent"""
    
    # === INPUT ===
    query: str
    check_scope: Literal["full", "quick", "models", "pipelines", "agents"]
    
    # === COMPONENT HEALTH ===
    component_statuses: Optional[List[ComponentStatus]]
    component_health_score: Optional[float]
    
    # === MODEL HEALTH ===
    model_metrics: Optional[List[ModelMetrics]]
    model_health_score: Optional[float]
    
    # === PIPELINE HEALTH ===
    pipeline_statuses: Optional[List[PipelineStatus]]
    pipeline_health_score: Optional[float]
    
    # === AGENT HEALTH ===
    agent_statuses: Optional[List[AgentStatus]]
    agent_health_score: Optional[float]
    
    # === COMPOSITE SCORE ===
    overall_health_score: Optional[float]  # 0-100
    health_grade: Optional[Literal["A", "B", "C", "D", "F"]]
    
    # === ISSUES ===
    critical_issues: Optional[List[str]]
    warnings: Optional[List[str]]
    
    # === SUMMARY ===
    health_summary: Optional[str]
    
    # === EXECUTION METADATA ===
    check_latency_ms: int
    timestamp: str
    
    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    status: Literal["pending", "checking", "completed", "failed"]
```

## Node Implementations

### Component Health Node

```python
# src/agents/health_score/nodes/component_health.py

import asyncio
import time
from typing import List
from datetime import datetime

from ..state import HealthScoreState, ComponentStatus

class ComponentHealthNode:
    """
    Check health of system components
    Fast parallel health checks
    """
    
    COMPONENTS = [
        {"name": "database", "endpoint": "/health/db"},
        {"name": "cache", "endpoint": "/health/cache"},
        {"name": "vector_store", "endpoint": "/health/vectors"},
        {"name": "api_gateway", "endpoint": "/health/api"},
        {"name": "message_queue", "endpoint": "/health/queue"},
    ]
    
    def __init__(self, health_client):
        self.health_client = health_client
        self.timeout_ms = 2000
    
    async def execute(self, state: HealthScoreState) -> HealthScoreState:
        start_time = time.time()
        
        try:
            # Parallel health checks
            tasks = [
                self._check_component(comp)
                for comp in self.COMPONENTS
            ]
            
            statuses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            component_statuses = []
            for comp, status in zip(self.COMPONENTS, statuses):
                if isinstance(status, Exception):
                    component_statuses.append(ComponentStatus(
                        component_name=comp["name"],
                        status="unknown",
                        latency_ms=None,
                        last_check=datetime.now().isoformat(),
                        error_message=str(status)
                    ))
                else:
                    component_statuses.append(status)
            
            # Calculate component health score
            healthy_count = sum(1 for s in component_statuses if s["status"] == "healthy")
            health_score = healthy_count / len(component_statuses) if component_statuses else 0
            
            check_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "component_statuses": component_statuses,
                "component_health_score": health_score,
                "check_latency_ms": check_time
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "component_health", "error": str(e)}],
                "component_health_score": 0.0
            }
    
    async def _check_component(self, component: dict) -> ComponentStatus:
        """Check single component health"""
        start = time.time()
        
        try:
            result = await asyncio.wait_for(
                self.health_client.check(component["endpoint"]),
                timeout=self.timeout_ms / 1000
            )
            
            latency = int((time.time() - start) * 1000)
            
            return ComponentStatus(
                component_name=component["name"],
                status="healthy" if result.get("ok") else "unhealthy",
                latency_ms=latency,
                last_check=datetime.now().isoformat(),
                error_message=result.get("error")
            )
            
        except asyncio.TimeoutError:
            return ComponentStatus(
                component_name=component["name"],
                status="unhealthy",
                latency_ms=self.timeout_ms,
                last_check=datetime.now().isoformat(),
                error_message="Health check timed out"
            )
```

### Model Health Node

```python
# src/agents/health_score/nodes/model_health.py

import asyncio
import time
from typing import List
from datetime import datetime, timedelta

from ..state import HealthScoreState, ModelMetrics

class ModelHealthNode:
    """
    Check health of deployed models
    Aggregates performance metrics
    """
    
    # Thresholds for health determination
    THRESHOLDS = {
        "min_accuracy": 0.7,
        "min_auc": 0.65,
        "max_error_rate": 0.05,
        "max_latency_p99_ms": 1000,
        "min_predictions_24h": 100
    }
    
    def __init__(self, metrics_store):
        self.metrics_store = metrics_store
    
    async def execute(self, state: HealthScoreState) -> HealthScoreState:
        start_time = time.time()
        
        if state.get("check_scope") not in ["full", "models"]:
            return {**state, "model_metrics": [], "model_health_score": 1.0}
        
        try:
            # Fetch all active models
            active_models = await self.metrics_store.get_active_models()
            
            # Fetch metrics for each model in parallel
            tasks = [
                self._get_model_metrics(model_id)
                for model_id in active_models
            ]
            
            metrics_list = await asyncio.gather(*tasks)
            
            # Calculate overall model health
            if metrics_list:
                healthy = sum(1 for m in metrics_list if m["status"] == "healthy")
                health_score = healthy / len(metrics_list)
            else:
                health_score = 1.0  # No models = healthy by default
            
            check_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "model_metrics": metrics_list,
                "model_health_score": health_score,
                "check_latency_ms": state.get("check_latency_ms", 0) + check_time
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "model_health", "error": str(e)}],
                "model_health_score": 0.5  # Unknown = degraded
            }
    
    async def _get_model_metrics(self, model_id: str) -> ModelMetrics:
        """Get metrics for a single model"""
        
        metrics = await self.metrics_store.get_model_metrics(
            model_id=model_id,
            time_window="24h"
        )
        
        # Determine health status
        status = self._determine_status(metrics)
        
        return ModelMetrics(
            model_id=model_id,
            accuracy=metrics.get("accuracy"),
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            f1_score=metrics.get("f1"),
            auc_roc=metrics.get("auc_roc"),
            prediction_latency_p50_ms=metrics.get("latency_p50"),
            prediction_latency_p99_ms=metrics.get("latency_p99"),
            predictions_last_24h=metrics.get("prediction_count", 0),
            error_rate=metrics.get("error_rate", 0),
            status=status
        )
    
    def _determine_status(self, metrics: dict) -> str:
        """Determine model health status from metrics"""
        
        issues = []
        
        if metrics.get("accuracy", 1) < self.THRESHOLDS["min_accuracy"]:
            issues.append("low_accuracy")
        
        if metrics.get("auc_roc", 1) < self.THRESHOLDS["min_auc"]:
            issues.append("low_auc")
        
        if metrics.get("error_rate", 0) > self.THRESHOLDS["max_error_rate"]:
            issues.append("high_error_rate")
        
        if metrics.get("latency_p99", 0) > self.THRESHOLDS["max_latency_p99_ms"]:
            issues.append("high_latency")
        
        if len(issues) >= 2:
            return "unhealthy"
        elif len(issues) == 1:
            return "degraded"
        else:
            return "healthy"
```

### Score Composer Node

```python
# src/agents/health_score/nodes/score_composer.py

import time
from datetime import datetime

from ..state import HealthScoreState

class ScoreComposerNode:
    """
    Compose overall health score from component scores
    Pure computation - no LLM
    """
    
    # Weights for each health dimension
    WEIGHTS = {
        "component": 0.30,
        "model": 0.30,
        "pipeline": 0.25,
        "agent": 0.15
    }
    
    # Grade thresholds
    GRADE_THRESHOLDS = {
        "A": 0.9,
        "B": 0.8,
        "C": 0.7,
        "D": 0.6,
        "F": 0.0
    }
    
    async def execute(self, state: HealthScoreState) -> HealthScoreState:
        start_time = time.time()
        
        try:
            # Collect scores
            scores = {
                "component": state.get("component_health_score", 1.0),
                "model": state.get("model_health_score", 1.0),
                "pipeline": state.get("pipeline_health_score", 1.0),
                "agent": state.get("agent_health_score", 1.0)
            }
            
            # Calculate weighted average
            overall_score = sum(
                scores[dim] * weight
                for dim, weight in self.WEIGHTS.items()
            )
            
            # Convert to 0-100 scale
            overall_score_100 = overall_score * 100
            
            # Determine grade
            grade = self._determine_grade(overall_score)
            
            # Identify issues
            critical_issues, warnings = self._identify_issues(state)
            
            # Generate summary
            summary = self._generate_summary(overall_score_100, grade, critical_issues)
            
            check_time = state.get("check_latency_ms", 0) + int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "overall_health_score": overall_score_100,
                "health_grade": grade,
                "critical_issues": critical_issues,
                "warnings": warnings,
                "health_summary": summary,
                "check_latency_ms": check_time,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "score_composer", "error": str(e)}],
                "status": "failed"
            }
    
    def _determine_grade(self, score: float) -> str:
        """Determine letter grade from score"""
        for grade, threshold in self.GRADE_THRESHOLDS.items():
            if score >= threshold:
                return grade
        return "F"
    
    def _identify_issues(self, state: HealthScoreState) -> tuple:
        """Identify critical issues and warnings"""
        
        critical = []
        warnings = []
        
        # Check components
        for comp in state.get("component_statuses", []):
            if comp["status"] == "unhealthy":
                critical.append(f"Component '{comp['component_name']}' is unhealthy")
            elif comp["status"] == "degraded":
                warnings.append(f"Component '{comp['component_name']}' is degraded")
        
        # Check models
        for model in state.get("model_metrics", []):
            if model["status"] == "unhealthy":
                critical.append(f"Model '{model['model_id']}' is unhealthy")
            elif model["status"] == "degraded":
                warnings.append(f"Model '{model['model_id']}' is degraded")
        
        # Check pipelines
        for pipeline in state.get("pipeline_statuses", []):
            if pipeline["status"] == "failed":
                critical.append(f"Pipeline '{pipeline['pipeline_name']}' has failed")
            elif pipeline["status"] == "stale":
                warnings.append(f"Pipeline '{pipeline['pipeline_name']}' data is stale")
        
        # Check agents
        for agent in state.get("agent_statuses", []):
            if not agent["available"]:
                critical.append(f"Agent '{agent['agent_name']}' is unavailable")
            elif agent["success_rate"] < 0.9:
                warnings.append(f"Agent '{agent['agent_name']}' has low success rate")
        
        return critical, warnings
    
    def _generate_summary(self, score: float, grade: str, issues: list) -> str:
        """Generate health summary"""
        
        if grade == "A":
            status = "excellent"
        elif grade == "B":
            status = "good"
        elif grade == "C":
            status = "fair"
        elif grade == "D":
            status = "poor"
        else:
            status = "critical"
        
        summary = f"System health is {status} (Grade: {grade}, Score: {score:.1f}/100)."
        
        if issues:
            summary += f" {len(issues)} critical issue(s) detected."
        else:
            summary += " All systems operational."
        
        return summary
```

## Graph Assembly

```python
# src/agents/health_score/graph.py

from langgraph.graph import StateGraph, END

from .state import HealthScoreState
from .nodes.component_health import ComponentHealthNode
from .nodes.model_health import ModelHealthNode
from .nodes.pipeline_health import PipelineHealthNode
from .nodes.agent_health import AgentHealthNode
from .nodes.score_composer import ScoreComposerNode

def build_health_score_graph(
    health_client,
    metrics_store,
    pipeline_store,
    agent_registry
):
    """
    Build the Health Score agent graph
    
    Architecture:
        [component] → [model] → [pipeline] → [agent] → [compose] → END
    """
    
    # Initialize nodes
    component = ComponentHealthNode(health_client)
    model = ModelHealthNode(metrics_store)
    pipeline = PipelineHealthNode(pipeline_store)
    agent = AgentHealthNode(agent_registry)
    composer = ScoreComposerNode()
    
    # Build graph
    workflow = StateGraph(HealthScoreState)
    
    # Add nodes
    workflow.add_node("component", component.execute)
    workflow.add_node("model", model.execute)
    workflow.add_node("pipeline", pipeline.execute)
    workflow.add_node("agent", agent.execute)
    workflow.add_node("compose", composer.execute)
    
    # Sequential flow for simplicity
    workflow.set_entry_point("component")
    workflow.add_edge("component", "model")
    workflow.add_edge("model", "pipeline")
    workflow.add_edge("pipeline", "agent")
    workflow.add_edge("agent", "compose")
    workflow.add_edge("compose", END)
    
    return workflow.compile()
```

## Integration Contracts

### Input Contract
```python
class HealthScoreInput(BaseModel):
    query: str
    check_scope: Literal["full", "quick", "models", "pipelines", "agents"] = "full"
```

### Output Contract
```python
class HealthScoreOutput(BaseModel):
    overall_health_score: float  # 0-100
    health_grade: str  # A-F
    component_health_score: float
    model_health_score: float
    pipeline_health_score: float
    agent_health_score: float
    critical_issues: List[str]
    warnings: List[str]
    health_summary: str
    check_latency_ms: int
```

## Handoff Format

```yaml
health_score_handoff:
  agent: health_score
  analysis_type: system_health
  key_findings:
    - overall_score: <0-100>
    - grade: <A-F>
    - critical_issues: <count>
  component_scores:
    component: <0-1>
    model: <0-1>
    pipeline: <0-1>
    agent: <0-1>
  issues:
    - <issue 1>
    - <issue 2>
  recommendations:
    - <recommendation 1>
  requires_further_analysis: <bool>
  suggested_next_agent: <drift_monitor>
```

## Testing Requirements

```
tests/unit/test_agents/test_health_score/
├── test_component_health.py  # Component checks
├── test_model_health.py      # Model metrics
├── test_pipeline_health.py   # Pipeline status
├── test_agent_health.py      # Agent availability
├── test_score_composer.py    # Score calculation
└── test_integration.py       # End-to-end flow
```

### Performance Requirements
- Total check: <5s for full scope
- Quick check: <1s
- No LLM calls
- Parallel component checks

---

## Cognitive RAG DSPy Integration

### Integration Flow (Non-LLM Pattern)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     HEALTH SCORE ↔ COGNITIVE RAG DSPY                        │
│                        (Fast Path - No LLM Usage)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────────────────────────────────────┐ │
│  │   HEALTH SCORE  │    │            COGNITIVE RAG DSPY                   │ │
│  │   (Fast Path)   │◄───│                                                 │ │
│  │                 │    │  ┌─────────────────────────────────────────┐   │ │
│  │ ┌─────────────┐ │    │  │  Baseline Context (Pre-fetched)         │   │ │
│  │ │  COMPONENT  │ │    │  │  ├─ baseline_thresholds: normal ranges  │   │ │
│  │ │   HEALTH    │◄├────│  │  ├─ seasonal_patterns: expected cycles  │   │ │
│  │ └─────────────┘ │    │  │  └─ maintenance_windows: planned events │   │ │
│  │       ↓         │    │  └─────────────────────────────────────────┘   │ │
│  │ ┌─────────────┐ │    │                                                 │ │
│  │ │    SCORE    │ │    │  ┌─────────────────────────────────────────┐   │ │
│  │ │  COMPOSER   │◄├────│  │  Anomaly Context (Cached Lookup)        │   │ │
│  │ │             │ │    │  │  ├─ similar_anomalies: past patterns    │   │ │
│  │ └─────────────┘ │    │  │  ├─ resolution_history: what fixed it   │   │ │
│  │       ↓         │    │  │  └─ impact_patterns: cascading effects  │   │ │
│  │       │         │    │  └─────────────────────────────────────────┘   │ │
│  │       ▼         │    │                                                 │ │
│  │ ┌─────────────┐ │    └─────────────────────────────────────────────────┘ │
│  │ │  TRAINING   │─┼───────────────────────────────────────────────────────►│
│  │ │  SIGNAL     │ │    MIPROv2 Optimizer (alert quality metrics)          │
│  │ └─────────────┘ │                                                        │
│  └─────────────────┘                                                        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  MEMORY CONTRIBUTION: health_events (EPISODIC)                          ││
│  │  ├─ Stores: component, status, duration, resolution, root_cause         ││
│  │  └─ Temporal: Events weighted by recency for pattern matching           ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  NOTE: Health Score is a Fast Path agent (No LLM). Cognitive context is     │
│  pre-fetched and cached to maintain <5s latency. No runtime DSPy calls.     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Cognitive Context (Pre-fetched)

The Health Score agent uses pre-fetched cognitive context to maintain fast path performance:

```python
# Cognitive context for Health Score (cached/pre-fetched)

class HealthCognitiveContext(TypedDict):
    """Pre-fetched cognitive context for health monitoring."""
    baseline_thresholds: Dict[str, Dict[str, float]]  # Component -> metric thresholds
    seasonal_patterns: Dict[str, List[float]]  # Component -> expected hourly patterns
    maintenance_windows: List[Dict[str, Any]]  # Planned maintenance events
    recent_anomalies: List[Dict[str, Any]]  # Recent anomaly patterns
    resolution_history: Dict[str, List[str]]  # Component -> successful resolutions
    last_refresh: str  # When context was last refreshed
```

### Score Composer with Cognitive Integration

```python
# src/agents/health_score/nodes/score_composer.py

from typing import Optional
import time
from datetime import datetime

from ..state import HealthScoreState


class HealthCognitiveContext(TypedDict):
    """Pre-fetched cognitive context for health monitoring."""
    baseline_thresholds: Dict[str, Dict[str, float]]
    seasonal_patterns: Dict[str, List[float]]
    maintenance_windows: List[Dict[str, Any]]
    recent_anomalies: List[Dict[str, Any]]
    resolution_history: Dict[str, List[str]]
    last_refresh: str


class ScoreComposerNode:
    """Score composer with cognitive context enrichment (no LLM)."""

    def __init__(self, context_cache: Optional[Dict] = None):
        """Initialize with optional pre-fetched context cache."""
        self._context_cache = context_cache or {}
        self._context_ttl_seconds = 300  # 5 minute cache TTL

    async def execute(
        self,
        state: HealthScoreState,
        cognitive_context: Optional[HealthCognitiveContext] = None
    ) -> HealthScoreState:
        """Execute score composition with cognitive enrichment."""
        start_time = time.time()

        try:
            # Collect scores
            scores = {
                "component": state.get("component_health_score", 1.0),
                "model": state.get("model_health_score", 1.0),
                "pipeline": state.get("pipeline_health_score", 1.0),
                "agent": state.get("agent_health_score", 1.0)
            }

            # Apply cognitive adjustments if available
            if cognitive_context:
                scores = self._apply_cognitive_adjustments(scores, state, cognitive_context)

            # Calculate weighted average
            overall_score = sum(
                scores[dim] * weight
                for dim, weight in self.WEIGHTS.items()
            )

            # Convert to 0-100 scale
            overall_score_100 = overall_score * 100

            # Determine grade
            grade = self._determine_grade(overall_score)

            # Identify issues with cognitive enrichment
            critical_issues, warnings = self._identify_issues(
                state, cognitive_context
            )

            # Generate summary
            summary = self._generate_summary(overall_score_100, grade, critical_issues)

            check_time = state.get("check_latency_ms", 0) + int((time.time() - start_time) * 1000)

            return {
                **state,
                "overall_health_score": overall_score_100,
                "health_grade": grade,
                "critical_issues": critical_issues,
                "warnings": warnings,
                "health_summary": summary,
                "check_latency_ms": check_time,
                "cognitive_context_used": cognitive_context is not None,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }

        except Exception as e:
            return {
                **state,
                "errors": [{"node": "score_composer", "error": str(e)}],
                "status": "failed"
            }

    def _apply_cognitive_adjustments(
        self,
        scores: Dict[str, float],
        state: HealthScoreState,
        context: HealthCognitiveContext
    ) -> Dict[str, float]:
        """Apply cognitive adjustments to scores (fast path, no LLM)."""

        adjusted = scores.copy()
        current_hour = datetime.now().hour

        # Check if within maintenance window
        for window in context.get("maintenance_windows", []):
            if self._is_in_maintenance_window(window):
                # During maintenance, don't penalize expected degradation
                for component in window.get("affected_components", []):
                    if component in adjusted:
                        # Boost score back up if degradation is expected
                        adjusted[component] = min(adjusted[component] + 0.2, 1.0)

        # Apply seasonal pattern adjustments
        for component, hourly_pattern in context.get("seasonal_patterns", {}).items():
            if component in adjusted and len(hourly_pattern) == 24:
                expected_baseline = hourly_pattern[current_hour]
                actual = adjusted[component]

                # If actual is close to expected seasonal pattern, boost slightly
                if abs(actual - expected_baseline) < 0.1:
                    adjusted[component] = min(adjusted[component] + 0.05, 1.0)

        return adjusted

    def _identify_issues(
        self,
        state: HealthScoreState,
        context: Optional[HealthCognitiveContext] = None
    ) -> tuple:
        """Identify issues with cognitive enrichment."""

        critical = []
        warnings = []

        # Standard issue identification
        for comp in state.get("component_statuses", []):
            if comp["status"] == "unhealthy":
                issue_text = f"Component '{comp['component_name']}' is unhealthy"

                # Enrich with resolution history if available
                if context and context.get("resolution_history"):
                    resolutions = context["resolution_history"].get(comp["component_name"], [])
                    if resolutions:
                        issue_text += f" (Past fixes: {', '.join(resolutions[:2])})"

                critical.append(issue_text)
            elif comp["status"] == "degraded":
                warnings.append(f"Component '{comp['component_name']}' is degraded")

        # Check for similar historical anomalies
        if context and context.get("recent_anomalies"):
            current_issues = set(c.split("'")[1] for c in critical if "'" in c)
            for anomaly in context["recent_anomalies"]:
                if anomaly.get("component") in current_issues:
                    impact = anomaly.get("cascade_impact", [])
                    if impact:
                        warnings.append(
                            f"Historical pattern: {anomaly['component']} issues may cascade to {', '.join(impact)}"
                        )

        # Standard checks for models, pipelines, agents...
        for model in state.get("model_metrics", []):
            if model["status"] == "unhealthy":
                critical.append(f"Model '{model['model_id']}' is unhealthy")
            elif model["status"] == "degraded":
                warnings.append(f"Model '{model['model_id']}' is degraded")

        for pipeline in state.get("pipeline_statuses", []):
            if pipeline["status"] == "failed":
                critical.append(f"Pipeline '{pipeline['pipeline_name']}' has failed")
            elif pipeline["status"] == "stale":
                warnings.append(f"Pipeline '{pipeline['pipeline_name']}' data is stale")

        for agent in state.get("agent_statuses", []):
            if not agent["available"]:
                critical.append(f"Agent '{agent['agent_name']}' is unavailable")
            elif agent["success_rate"] < 0.9:
                warnings.append(f"Agent '{agent['agent_name']}' has low success rate")

        return critical, warnings

    def _is_in_maintenance_window(self, window: Dict[str, Any]) -> bool:
        """Check if currently in a maintenance window."""
        from datetime import datetime
        now = datetime.now()
        start = datetime.fromisoformat(window.get("start", "2099-01-01"))
        end = datetime.fromisoformat(window.get("end", "2000-01-01"))
        return start <= now <= end
```

### Training Signal for MIPROv2

```python
# src/agents/health_score/training_signal.py

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class HealthScoreTrainingSignal:
    """Training signal for health monitoring quality."""

    # Alert quality metrics
    critical_issues_count: int
    warnings_count: int
    true_positives: int  # Confirmed actual issues
    false_positives: int  # Alerts that weren't real issues
    missed_issues: int  # Issues that should have been detected

    # Cognitive context usage
    cognitive_context_used: bool
    resolution_suggestions_provided: int
    cascade_warnings_issued: int

    # Performance metrics
    check_latency_ms: int
    max_latency_threshold_ms: int = 5000

    def compute_reward(self) -> float:
        """Compute reward for MIPROv2 optimization."""

        # Base reward for completing check
        base_reward = 0.3

        # Reward for detection accuracy
        total_alerts = self.true_positives + self.false_positives + self.missed_issues
        if total_alerts > 0:
            precision = self.true_positives / max(self.true_positives + self.false_positives, 1)
            recall = self.true_positives / max(self.true_positives + self.missed_issues, 1)
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                base_reward += 0.3 * f1
        else:
            # No issues detected and none missed = perfect
            base_reward += 0.3

        # Penalize false positives heavily (alert fatigue)
        if self.false_positives > 0:
            base_reward -= 0.05 * min(self.false_positives, 5)

        # Penalize missed issues even more heavily
        if self.missed_issues > 0:
            base_reward -= 0.1 * min(self.missed_issues, 5)

        # Reward cognitive enrichment
        if self.cognitive_context_used:
            base_reward += 0.1
            if self.resolution_suggestions_provided > 0:
                base_reward += 0.05
            if self.cascade_warnings_issued > 0:
                base_reward += 0.05

        # Penalize slow checks (must maintain fast path)
        if self.check_latency_ms > self.max_latency_threshold_ms:
            latency_penalty = (self.check_latency_ms - self.max_latency_threshold_ms) / 10000
            base_reward -= min(latency_penalty, 0.2)

        return max(0.0, min(base_reward, 1.0))
```

### Memory Contribution

```python
# Memory contribution for health events

async def contribute_to_memory(
    state: HealthScoreState,
    memory_backend: MemoryBackend
) -> None:
    """Store health events in organizational memory (EPISODIC)."""

    # Only store if there were issues
    if not state.get("critical_issues") and not state.get("warnings"):
        return

    health_event = {
        "event_id": f"health_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "overall_score": state.get("overall_health_score", 0),
        "grade": state.get("health_grade", ""),
        "critical_issues": state.get("critical_issues", []),
        "warnings": state.get("warnings", []),
        "component_statuses": state.get("component_statuses", []),
        "timestamp": state.get("timestamp", ""),
        "resolution": None,  # To be updated when resolved
        "duration_minutes": None,  # To be updated when resolved
    }

    await memory_backend.store(
        memory_type="EPISODIC",
        content=health_event,
        metadata={
            "agent": "health_score",
            "index": "health_events",
            "embedding_fields": ["critical_issues", "warnings"],
            "temporal_weight": 0.9  # Recent events weighted higher
        }
    )
```

### Cognitive Input TypedDict

```python
# src/agents/health_score/cognitive_input.py

from typing import TypedDict, List, Dict, Any, Optional, Literal


class HealthScoreCognitiveInput(TypedDict):
    """Full cognitive input for Health Score agent."""

    # Standard input
    query: str
    check_scope: Literal["full", "quick", "models", "pipelines", "agents"]

    # Cognitive context (pre-fetched, cached)
    cognitive_context: Optional[HealthCognitiveContext]
```

### Configuration

```yaml
# config/agents/health_score.yaml

health_score:
  tier: 3
  type: standard_fast_path

  performance:
    max_latency_ms: 5000
    quick_check_latency_ms: 1000
    no_llm_calls: true

  cognitive_rag:
    enabled: true
    mode: pre_fetched  # No runtime DSPy calls
    context_sources:
      - baseline_thresholds
      - seasonal_patterns
      - maintenance_windows
      - recent_anomalies
      - resolution_history
    cache_ttl_seconds: 300
    refresh_on_startup: true

  dspy:
    optimizer: MIPROv2
    training_signals:
      - alert_accuracy
      - latency_compliance
      - resolution_helpfulness
    optimization_target: detection_f1_score

  memory:
    contribution_enabled: true
    index: health_events
    memory_type: EPISODIC
    temporal_decay: true
    embedding_fields:
      - critical_issues
      - warnings
```

### Testing Requirements

```python
# tests/unit/test_agents/test_health_score/test_cognitive_integration.py

@pytest.mark.asyncio
async def test_health_check_with_cognitive_context():
    """Test health check with pre-fetched cognitive context."""
    agent = HealthScoreAgent()

    cognitive_context = HealthCognitiveContext(
        baseline_thresholds={
            "database": {"latency_ms": 100, "connections": 1000}
        },
        seasonal_patterns={
            "component": [0.9] * 24  # Flat pattern
        },
        maintenance_windows=[],
        recent_anomalies=[
            {"component": "database", "cascade_impact": ["cache", "api"]}
        ],
        resolution_history={
            "database": ["restart", "connection_pool_increase"]
        },
        last_refresh=datetime.now().isoformat()
    )

    result = await agent.check_health(
        check_scope="full",
        cognitive_context=cognitive_context
    )

    assert result.check_latency_ms < 5000  # Must maintain fast path
    assert result.cognitive_context_used is True


@pytest.mark.asyncio
async def test_cascade_warning_from_cognitive_context():
    """Test cascade warnings based on historical patterns."""
    agent = HealthScoreAgent()

    # Simulate database being unhealthy
    state = HealthScoreState(
        component_statuses=[
            ComponentStatus(
                component_name="database",
                status="unhealthy",
                latency_ms=5000,
                last_check=datetime.now().isoformat(),
                error_message="Connection timeout"
            )
        ]
    )

    cognitive_context = HealthCognitiveContext(
        recent_anomalies=[
            {"component": "database", "cascade_impact": ["cache", "api"]}
        ],
        # ... other fields
    )

    result = await agent._identify_issues(state, cognitive_context)
    critical, warnings = result

    # Should warn about cascade impact
    cascade_warning = [w for w in warnings if "cascade" in w.lower()]
    assert len(cascade_warning) > 0


def test_training_signal_penalizes_slow_checks():
    """Test that slow checks are penalized in training signal."""
    signal = HealthScoreTrainingSignal(
        critical_issues_count=1,
        warnings_count=0,
        true_positives=1,
        false_positives=0,
        missed_issues=0,
        cognitive_context_used=True,
        resolution_suggestions_provided=1,
        cascade_warnings_issued=0,
        check_latency_ms=8000,  # Exceeds 5000ms threshold
        max_latency_threshold_ms=5000
    )

    reward = signal.compute_reward()

    # Should be penalized for slow check
    fast_signal = HealthScoreTrainingSignal(
        **{**signal.__dict__, "check_latency_ms": 2000}
    )
    fast_reward = fast_signal.compute_reward()

    assert fast_reward > reward  # Fast check should have higher reward
```
