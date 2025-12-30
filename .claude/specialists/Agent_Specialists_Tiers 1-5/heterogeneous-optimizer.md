# Tier 2: Heterogeneous Optimizer Agent Specialist

## Agent Classification

| Property | Value |
|----------|-------|
| **Tier** | 2 (Causal Analytics) |
| **Agent Type** | Standard (Computational) |
| **Model Tier** | Sonnet |
| **Latency Tolerance** | Medium (up to 25s) |
| **Critical Path** | Core E2I mission agent |

## Domain Scope

You are the specialist for the Tier 2 Heterogeneous Optimizer Agent:
- `src/agents/heterogeneous_optimizer/` - Segment-level CATE analysis

This is a **Standard Agent** with EconML focus:
- Conditional Average Treatment Effect (CATE) estimation
- Segment-level heterogeneity analysis
- Optimal treatment allocation recommendations

## Design Principles

### EconML-Centric
The Heterogeneous Optimizer is built around EconML's Causal Forests:
- CausalForestDML for CATE estimation
- Double Machine Learning for debiasing
- Feature importance for segment discovery

### Responsibilities
1. **CATE Estimation** - Estimate treatment effects by segment
2. **Segment Discovery** - Identify high/low responder groups
3. **Policy Learning** - Recommend optimal treatment allocation
4. **Effect Visualization** - Generate interpretable effect profiles

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 HETEROGENEOUS OPTIMIZER AGENT                    │
│                     (Standard Pattern)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │    CATE     │───►│   SEGMENT   │───►│   POLICY    │         │
│  │  ESTIMATOR  │    │  ANALYZER   │    │   LEARNER   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                  │                   │                 │
│         ▼                  ▼                   ▼                 │
│  ┌─────────────────────────────────────────────────────┐        │
│  │              EFFECT PROFILE GENERATOR               │        │
│  │  [CATE Plots] [Feature Importance] [Segment Grid]   │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
heterogeneous_optimizer/
├── agent.py              # Main HeterogeneousOptimizerAgent class
├── state.py              # LangGraph state definitions
├── graph.py              # LangGraph assembly
├── nodes/
│   ├── cate_estimator.py # EconML CATE estimation
│   ├── segment_analyzer.py # Segment discovery
│   ├── policy_learner.py # Optimal allocation
│   └── profile_generator.py # Visualization data
├── segment_utils.py      # Segment handling utilities
└── prompts.py            # Summary generation prompts
```

## LangGraph State Definition

```python
# src/agents/heterogeneous_optimizer/state.py

from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
from datetime import datetime
import operator

class CATEResult(TypedDict):
    """CATE estimation result for a segment"""
    segment_name: str
    segment_value: str
    cate_estimate: float
    cate_ci_lower: float
    cate_ci_upper: float
    sample_size: int
    statistical_significance: bool

class SegmentProfile(TypedDict):
    """Profile of a high/low responder segment"""
    segment_id: str
    responder_type: Literal["high", "low", "average"]
    cate_estimate: float
    defining_features: List[Dict[str, Any]]
    size: int
    size_percentage: float
    recommendation: str

class PolicyRecommendation(TypedDict):
    """Treatment allocation recommendation"""
    segment: str
    current_treatment_rate: float
    recommended_treatment_rate: float
    expected_incremental_outcome: float
    confidence: float

class HeterogeneousOptimizerState(TypedDict):
    """Complete state for Heterogeneous Optimizer agent"""
    
    # === INPUT ===
    query: str
    treatment_var: str
    outcome_var: str
    segment_vars: List[str]  # Variables to segment by
    effect_modifiers: List[str]  # Variables that modify treatment effect
    data_source: str
    filters: Optional[Dict[str, Any]]
    
    # === CONFIGURATION ===
    n_estimators: int  # Causal Forest trees (default: 100)
    min_samples_leaf: int  # Minimum samples per leaf (default: 10)
    significance_level: float  # For CI calculation (default: 0.05)
    top_segments_count: int  # Number of top segments to return (default: 10)
    
    # === CATE OUTPUTS ===
    cate_by_segment: Optional[Dict[str, List[CATEResult]]]
    overall_ate: Optional[float]
    heterogeneity_score: Optional[float]  # 0-1, higher = more heterogeneity
    feature_importance: Optional[Dict[str, float]]
    
    # === SEGMENT DISCOVERY OUTPUTS ===
    high_responders: Optional[List[SegmentProfile]]
    low_responders: Optional[List[SegmentProfile]]
    segment_comparison: Optional[Dict[str, Any]]
    
    # === POLICY OUTPUTS ===
    policy_recommendations: Optional[List[PolicyRecommendation]]
    expected_total_lift: Optional[float]
    optimal_allocation_summary: Optional[str]
    
    # === VISUALIZATION DATA ===
    cate_plot_data: Optional[Dict[str, Any]]
    segment_grid_data: Optional[Dict[str, Any]]
    
    # === SUMMARY ===
    executive_summary: Optional[str]
    key_insights: Optional[List[str]]
    
    # === EXECUTION METADATA ===
    estimation_latency_ms: int
    analysis_latency_ms: int
    total_latency_ms: int
    
    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    status: Literal["pending", "estimating", "analyzing", "optimizing", "completed", "failed"]
```

## Node Implementations

### CATE Estimator Node

```python
# src/agents/heterogeneous_optimizer/nodes/cate_estimator.py

import asyncio
import time
import traceback
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from ..state import HeterogeneousOptimizerState, CATEResult

class CATEEstimatorNode:
    """
    Estimate Conditional Average Treatment Effects using EconML
    Core computation node
    """
    
    def __init__(self, data_connector):
        self.data_connector = data_connector
        self.timeout_seconds = 180
    
    async def execute(self, state: HeterogeneousOptimizerState) -> HeterogeneousOptimizerState:
        start_time = time.time()
        
        try:
            from econml.dml import CausalForestDML
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            
            # Fetch data
            df = await self._fetch_data(state)
            
            if df is None or len(df) < 100:
                return {
                    **state,
                    "errors": [{"node": "cate_estimator", "error": "Insufficient data (need >= 100 rows)"}],
                    "status": "failed"
                }
            
            # Prepare data
            Y = df[state["outcome_var"]].values
            T = df[state["treatment_var"]].values
            X = df[state["effect_modifiers"]].values
            W = df[state["segment_vars"]].values if state["segment_vars"] else None
            
            # Fit Causal Forest
            cf = CausalForestDML(
                model_y=RandomForestRegressor(n_estimators=50, random_state=42),
                model_t=RandomForestClassifier(n_estimators=50, random_state=42) if self._is_binary(T) else RandomForestRegressor(n_estimators=50, random_state=42),
                n_estimators=state.get("n_estimators", 100),
                min_samples_leaf=state.get("min_samples_leaf", 10),
                random_state=42
            )
            
            # Fit with timeout
            await asyncio.wait_for(
                asyncio.to_thread(cf.fit, Y, T, X=X, W=W),
                timeout=self.timeout_seconds
            )
            
            # Get overall ATE
            ate = cf.ate(X)
            
            # Get individual treatment effects
            cate_individual = cf.effect(X)
            
            # Calculate heterogeneity score
            heterogeneity = self._calculate_heterogeneity(cate_individual, ate)
            
            # Get feature importance
            feature_importance = dict(zip(
                state["effect_modifiers"],
                cf.feature_importances_.tolist() if hasattr(cf, 'feature_importances_') else [0] * len(state["effect_modifiers"])
            ))
            
            # Calculate CATE by segment
            cate_by_segment = await self._calculate_cate_by_segment(
                df, cf, state["segment_vars"], state["effect_modifiers"],
                state.get("significance_level", 0.05)
            )
            
            estimation_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "overall_ate": float(ate),
                "heterogeneity_score": heterogeneity,
                "feature_importance": feature_importance,
                "cate_by_segment": cate_by_segment,
                "estimation_latency_ms": estimation_time,
                "status": "analyzing"
            }
            
        except asyncio.TimeoutError:
            return {
                **state,
                "errors": [{"node": "cate_estimator", "error": f"Timed out after {self.timeout_seconds}s"}],
                "status": "failed"
            }
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "cate_estimator", "error": str(e), "traceback": traceback.format_exc()}],
                "status": "failed"
            }
    
    async def _fetch_data(self, state: HeterogeneousOptimizerState) -> pd.DataFrame:
        """Fetch data for CATE estimation"""
        columns = [state["treatment_var"], state["outcome_var"]] + \
                  state["effect_modifiers"] + state["segment_vars"]
        
        return await self.data_connector.query(
            source=state["data_source"],
            columns=list(set(columns)),
            filters=state.get("filters")
        )
    
    def _is_binary(self, T: np.ndarray) -> bool:
        """Check if treatment is binary"""
        unique_vals = np.unique(T)
        return len(unique_vals) == 2
    
    def _calculate_heterogeneity(self, cate_individual: np.ndarray, ate: float) -> float:
        """Calculate heterogeneity score (coefficient of variation)"""
        std = np.std(cate_individual)
        if ate == 0:
            return 0.0
        cv = std / abs(ate)
        # Normalize to 0-1 scale
        return min(cv / 2, 1.0)
    
    async def _calculate_cate_by_segment(
        self,
        df: pd.DataFrame,
        cf,
        segment_vars: List[str],
        effect_modifiers: List[str],
        alpha: float
    ) -> Dict[str, List[CATEResult]]:
        """Calculate CATE for each segment value"""
        
        cate_by_segment = {}
        
        for segment_var in segment_vars:
            segment_results = []
            
            for segment_value in df[segment_var].unique():
                mask = df[segment_var] == segment_value
                segment_df = df[mask]
                
                if len(segment_df) < 10:
                    continue
                
                X_segment = segment_df[effect_modifiers].values
                
                # Get CATE for segment
                cate = cf.effect(X_segment)
                cate_mean = float(np.mean(cate))
                
                # Get confidence interval
                try:
                    cate_interval = cf.effect_interval(X_segment, alpha=alpha)
                    ci_lower = float(np.mean(cate_interval[0]))
                    ci_upper = float(np.mean(cate_interval[1]))
                except Exception:
                    ci_lower = cate_mean - 1.96 * float(np.std(cate))
                    ci_upper = cate_mean + 1.96 * float(np.std(cate))
                
                # Determine statistical significance
                significant = (ci_lower > 0) or (ci_upper < 0)
                
                segment_results.append(CATEResult(
                    segment_name=segment_var,
                    segment_value=str(segment_value),
                    cate_estimate=cate_mean,
                    cate_ci_lower=ci_lower,
                    cate_ci_upper=ci_upper,
                    sample_size=len(segment_df),
                    statistical_significance=significant
                ))
            
            # Sort by CATE estimate
            segment_results.sort(key=lambda x: x["cate_estimate"], reverse=True)
            cate_by_segment[segment_var] = segment_results
        
        return cate_by_segment
```

### Segment Analyzer Node

```python
# src/agents/heterogeneous_optimizer/nodes/segment_analyzer.py

import time
from typing import List, Dict, Any

from ..state import HeterogeneousOptimizerState, SegmentProfile, CATEResult

class SegmentAnalyzerNode:
    """
    Analyze segments to identify high/low responders
    Pure computation - no LLM needed
    """
    
    def __init__(self):
        self.high_responder_threshold = 1.5  # 1.5x ATE
        self.low_responder_threshold = 0.5   # 0.5x ATE
    
    async def execute(self, state: HeterogeneousOptimizerState) -> HeterogeneousOptimizerState:
        start_time = time.time()
        
        if state.get("status") == "failed":
            return state
        
        try:
            ate = state["overall_ate"]
            cate_by_segment = state["cate_by_segment"]
            top_count = state.get("top_segments_count", 10)
            
            # Flatten all segment results
            all_segments = []
            total_size = 0
            
            for segment_var, results in cate_by_segment.items():
                for result in results:
                    total_size += result["sample_size"]
                    all_segments.append({
                        "segment_var": segment_var,
                        "result": result
                    })
            
            # Identify high responders
            high_responders = self._identify_responders(
                all_segments, 
                ate, 
                total_size,
                "high",
                self.high_responder_threshold
            )[:top_count]
            
            # Identify low responders
            low_responders = self._identify_responders(
                all_segments, 
                ate, 
                total_size,
                "low",
                self.low_responder_threshold
            )[:top_count]
            
            # Create segment comparison
            comparison = self._create_comparison(high_responders, low_responders, ate)
            
            analysis_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "high_responders": high_responders,
                "low_responders": low_responders,
                "segment_comparison": comparison,
                "analysis_latency_ms": analysis_time,
                "status": "optimizing"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "segment_analyzer", "error": str(e)}],
                "status": "failed"
            }
    
    def _identify_responders(
        self,
        all_segments: List[Dict],
        ate: float,
        total_size: int,
        responder_type: str,
        threshold: float
    ) -> List[SegmentProfile]:
        """Identify high or low responder segments"""
        
        profiles = []
        
        for seg in all_segments:
            result = seg["result"]
            cate = result["cate_estimate"]
            
            # Determine if segment qualifies
            if responder_type == "high":
                qualifies = ate > 0 and cate >= ate * threshold
            else:
                qualifies = ate > 0 and cate <= ate * threshold
            
            if not qualifies:
                continue
            
            profile = SegmentProfile(
                segment_id=f"{seg['segment_var']}_{result['segment_value']}",
                responder_type=responder_type,
                cate_estimate=cate,
                defining_features=[{
                    "variable": seg["segment_var"],
                    "value": result["segment_value"],
                    "effect_size": cate / ate if ate != 0 else 0
                }],
                size=result["sample_size"],
                size_percentage=result["sample_size"] / total_size * 100 if total_size > 0 else 0,
                recommendation=self._generate_recommendation(seg["segment_var"], result, responder_type)
            )
            profiles.append(profile)
        
        # Sort by CATE (descending for high, ascending for low)
        reverse = responder_type == "high"
        profiles.sort(key=lambda x: x["cate_estimate"], reverse=reverse)
        
        return profiles
    
    def _generate_recommendation(
        self, 
        segment_var: str, 
        result: CATEResult, 
        responder_type: str
    ) -> str:
        """Generate action recommendation for segment"""
        
        segment_value = result["segment_value"]
        cate = result["cate_estimate"]
        
        if responder_type == "high":
            return f"Prioritize treatment for {segment_var}={segment_value} (CATE: {cate:.3f}). High response expected."
        else:
            return f"De-prioritize treatment for {segment_var}={segment_value} (CATE: {cate:.3f}). Consider alternative interventions."
    
    def _create_comparison(
        self,
        high_responders: List[SegmentProfile],
        low_responders: List[SegmentProfile],
        ate: float
    ) -> Dict[str, Any]:
        """Create comparison summary between high and low responders"""
        
        high_avg_cate = sum(h["cate_estimate"] for h in high_responders) / len(high_responders) if high_responders else 0
        low_avg_cate = sum(l["cate_estimate"] for l in low_responders) / len(low_responders) if low_responders else 0
        
        return {
            "overall_ate": ate,
            "high_responder_avg_cate": high_avg_cate,
            "low_responder_avg_cate": low_avg_cate,
            "effect_ratio": high_avg_cate / low_avg_cate if low_avg_cate != 0 else float('inf'),
            "high_responder_count": len(high_responders),
            "low_responder_count": len(low_responders)
        }
```

### Policy Learner Node

```python
# src/agents/heterogeneous_optimizer/nodes/policy_learner.py

import time
from typing import List, Dict, Any

from ..state import HeterogeneousOptimizerState, PolicyRecommendation

class PolicyLearnerNode:
    """
    Learn optimal treatment allocation policy
    Uses CATE estimates to recommend allocation changes
    """
    
    def __init__(self):
        self.min_cate_for_treatment = 0.01  # Minimum CATE to recommend treatment
    
    async def execute(self, state: HeterogeneousOptimizerState) -> HeterogeneousOptimizerState:
        start_time = time.time()
        
        if state.get("status") == "failed":
            return state
        
        try:
            cate_by_segment = state["cate_by_segment"]
            high_responders = state["high_responders"]
            low_responders = state["low_responders"]
            ate = state["overall_ate"]
            
            # Generate policy recommendations
            recommendations = []
            
            for segment_var, results in cate_by_segment.items():
                for result in results:
                    rec = self._generate_recommendation(result, ate)
                    if rec:
                        recommendations.append(rec)
            
            # Sort by expected incremental outcome
            recommendations.sort(
                key=lambda x: x["expected_incremental_outcome"], 
                reverse=True
            )
            
            # Calculate total expected lift if policy is implemented
            total_lift = sum(r["expected_incremental_outcome"] for r in recommendations)
            
            # Generate summary
            summary = self._generate_allocation_summary(
                recommendations, high_responders, low_responders, ate
            )
            
            total_time = (
                state.get("estimation_latency_ms", 0) +
                state.get("analysis_latency_ms", 0) +
                int((time.time() - start_time) * 1000)
            )
            
            return {
                **state,
                "policy_recommendations": recommendations[:20],  # Top 20
                "expected_total_lift": total_lift,
                "optimal_allocation_summary": summary,
                "total_latency_ms": total_time,
                "status": "completed"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "policy_learner", "error": str(e)}],
                "status": "failed"
            }
    
    def _generate_recommendation(
        self, 
        result: Dict[str, Any],
        ate: float
    ) -> PolicyRecommendation:
        """Generate policy recommendation for a segment"""
        
        cate = result["cate_estimate"]
        segment_name = result["segment_name"]
        segment_value = result["segment_value"]
        sample_size = result["sample_size"]
        
        # Determine recommended treatment rate change
        if cate >= ate * 1.5:
            # High responder - increase treatment
            current_rate = 0.5  # Assume current 50% coverage
            recommended_rate = min(0.9, current_rate + 0.2)
        elif cate <= ate * 0.5:
            # Low responder - decrease treatment
            current_rate = 0.5
            recommended_rate = max(0.1, current_rate - 0.2)
        elif cate >= self.min_cate_for_treatment:
            # Average responder - maintain
            current_rate = 0.5
            recommended_rate = 0.5
        else:
            # Very low/negative responder - minimize treatment
            current_rate = 0.5
            recommended_rate = 0.1
        
        # Calculate expected incremental outcome
        rate_change = recommended_rate - current_rate
        expected_lift = rate_change * cate * sample_size
        
        # Confidence based on sample size and significance
        confidence = min(0.9, 0.5 + (sample_size / 1000) * 0.3)
        if result.get("statistical_significance"):
            confidence = min(confidence + 0.1, 0.95)
        
        return PolicyRecommendation(
            segment=f"{segment_name}={segment_value}",
            current_treatment_rate=current_rate,
            recommended_treatment_rate=recommended_rate,
            expected_incremental_outcome=expected_lift,
            confidence=confidence
        )
    
    def _generate_allocation_summary(
        self,
        recommendations: List[PolicyRecommendation],
        high_responders: List,
        low_responders: List,
        ate: float
    ) -> str:
        """Generate natural language summary of optimal allocation"""
        
        increase_recs = [r for r in recommendations if r["recommended_treatment_rate"] > r["current_treatment_rate"]]
        decrease_recs = [r for r in recommendations if r["recommended_treatment_rate"] < r["current_treatment_rate"]]
        
        summary_parts = [
            f"Treatment effect heterogeneity detected (ATE: {ate:.3f}).",
            f"Identified {len(high_responders)} high-responder segments and {len(low_responders)} low-responder segments.",
        ]
        
        if increase_recs:
            top_increase = increase_recs[0]
            summary_parts.append(
                f"Recommend increasing treatment in {len(increase_recs)} segments, "
                f"starting with {top_increase['segment']}."
            )
        
        if decrease_recs:
            summary_parts.append(
                f"Recommend decreasing treatment in {len(decrease_recs)} segments to optimize resource allocation."
            )
        
        total_lift = sum(r["expected_incremental_outcome"] for r in recommendations)
        summary_parts.append(f"Expected total outcome lift from reallocation: {total_lift:.1f} units.")
        
        return " ".join(summary_parts)
```

## Graph Assembly

```python
# src/agents/heterogeneous_optimizer/graph.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import HeterogeneousOptimizerState
from .nodes.cate_estimator import CATEEstimatorNode
from .nodes.segment_analyzer import SegmentAnalyzerNode
from .nodes.policy_learner import PolicyLearnerNode
from .nodes.profile_generator import ProfileGeneratorNode

def build_heterogeneous_optimizer_graph(
    data_connector,
    enable_checkpointing: bool = True
):
    """
    Build the Heterogeneous Optimizer agent graph
    
    Architecture:
        [estimate_cate] → [analyze_segments] → [learn_policy] → [generate_profiles] → END
    """
    
    # Initialize nodes
    cate_estimator = CATEEstimatorNode(data_connector)
    segment_analyzer = SegmentAnalyzerNode()
    policy_learner = PolicyLearnerNode()
    profile_generator = ProfileGeneratorNode()
    
    # Build graph
    workflow = StateGraph(HeterogeneousOptimizerState)
    
    # Add nodes
    workflow.add_node("estimate_cate", cate_estimator.execute)
    workflow.add_node("analyze_segments", segment_analyzer.execute)
    workflow.add_node("learn_policy", policy_learner.execute)
    workflow.add_node("generate_profiles", profile_generator.execute)
    workflow.add_node("error_handler", error_handler_node)
    
    # Entry point
    workflow.set_entry_point("estimate_cate")
    
    # Edges
    workflow.add_conditional_edges(
        "estimate_cate",
        lambda s: "error" if s.get("status") == "failed" else "analyze_segments",
        {"analyze_segments": "analyze_segments", "error": "error_handler"}
    )
    
    workflow.add_conditional_edges(
        "analyze_segments",
        lambda s: "error" if s.get("status") == "failed" else "learn_policy",
        {"learn_policy": "learn_policy", "error": "error_handler"}
    )
    
    workflow.add_edge("learn_policy", "generate_profiles")
    workflow.add_edge("generate_profiles", END)
    workflow.add_edge("error_handler", END)
    
    # Compile
    if enable_checkpointing:
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    return workflow.compile()

async def error_handler_node(state: HeterogeneousOptimizerState) -> HeterogeneousOptimizerState:
    """Handle errors gracefully"""
    errors = state.get("errors", [])
    
    return {
        **state,
        "executive_summary": "Heterogeneous effect analysis could not be completed.",
        "key_insights": [f"Error: {e.get('error', 'Unknown')}" for e in errors],
        "status": "failed"
    }
```

## Integration Contracts

### Input Contract (from Orchestrator)
```python
class HeterogeneousOptimizerInput(BaseModel):
    query: str
    treatment_var: str
    outcome_var: str
    segment_vars: List[str]
    effect_modifiers: List[str]
    data_source: str
    filters: Optional[Dict[str, Any]] = None
    n_estimators: int = 100
    min_samples_leaf: int = 10
    top_segments_count: int = 10
```

### Output Contract (to Orchestrator)
```python
class HeterogeneousOptimizerOutput(BaseModel):
    overall_ate: float
    heterogeneity_score: float
    high_responders: List[SegmentProfile]
    low_responders: List[SegmentProfile]
    policy_recommendations: List[PolicyRecommendation]
    expected_total_lift: float
    feature_importance: Dict[str, float]
    executive_summary: str
    key_insights: List[str]
    total_latency_ms: int
```

## Testing Requirements

```
tests/unit/test_agents/test_heterogeneous_optimizer/
├── test_cate_estimator.py   # EconML integration
├── test_segment_analyzer.py # Segment discovery
├── test_policy_learner.py   # Policy recommendations
└── test_integration.py      # End-to-end flow
```

### Performance Requirements
- CATE estimation: <120s for 100k rows
- Segment analysis: <5s
- Total latency: <150s

### Test Cases
1. CATE correctly estimated for known heterogeneous effects
2. High/low responders identified at correct thresholds
3. Policy recommendations align with CATE direction
4. Handles low sample size segments gracefully
5. Feature importance ranks correctly

## Handoff Format

```yaml
heterogeneous_optimizer_handoff:
  agent: heterogeneous_optimizer
  analysis_type: cate_estimation
  key_findings:
    - overall_ate: <float>
    - heterogeneity_score: <float>
    - top_high_responder: <segment description>
    - top_low_responder: <segment description>
  policy_recommendations:
    - segment: <segment>
      action: increase|decrease|maintain
      expected_lift: <float>
  feature_importance:
    - <feature>: <importance>
  recommendations:
    - <recommendation 1>
    - <recommendation 2>
  requires_further_analysis: <bool>
  suggested_next_agent: <experiment_designer|resource_optimizer>
```

## EconML Integration Notes

### Supported Estimators
1. `CausalForestDML` - Default, handles heterogeneity well
2. `LinearDML` - Faster, assumes linear effects
3. `SparseLinearDML` - For high-dimensional effect modifiers

### Key Parameters
- `n_estimators`: More trees = better estimates but slower
- `min_samples_leaf`: Higher = more stable but less granular
- `honest`: Use separate samples for growing and estimation (default: True)

### Confidence Intervals
- Uses bootstrap or asymptotic inference
- `effect_interval(X, alpha=0.05)` for 95% CI

---

## Cognitive RAG DSPy Integration

### Integration Overview

The Heterogeneous Optimizer integrates with the Cognitive RAG DSPy system to receive evidence about treatment effects and segment characteristics, contributing CATE insights back to the semantic memory.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            HETEROGENEOUS OPTIMIZER - COGNITIVE RAG INTEGRATION               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 2: INVESTIGATOR                    PHASE 3: AGENT                    │
│  ┌──────────────────────┐                ┌─────────────────────────────┐    │
│  │ EvidenceSynthesis    │                │  HETEROGENEOUS OPTIMIZER    │    │
│  │ Signature            │                │                             │    │
│  │  - evidence_items    │───────────────►│  1. Parse treatment context │    │
│  │  - user_query        │                │  2. Extract segment priors  │    │
│  │                      │                │  3. Run CATE estimation     │    │
│  │ OUTPUTS:             │                │  4. Validate against priors │    │
│  │  - synthesized_summary                │                             │    │
│  │  - treatment_context ─────────────────│──► Prior effect estimates   │    │
│  │  - segment_insights  ─────────────────│──► Known responder patterns │    │
│  │  - prior_cate_estimates ──────────────│──► Historical CATE values   │    │
│  └──────────────────────┘                └──────────────┬──────────────┘    │
│                                                         │                   │
│                                                         ▼                   │
│                                          ┌─────────────────────────────┐    │
│                                          │    TRAINING SIGNAL          │    │
│                                          │  HetOptTrainingSignal       │    │
│                                          │   - cate_accuracy           │    │
│                                          │   - segment_discovery_rate  │    │
│                                          │   - policy_adoption_rate    │    │
│                                          └──────────────┬──────────────┘    │
│                                                         │                   │
│  PHASE 4: REFLECTOR                                     ▼                   │
│  ┌──────────────────────┐                ┌─────────────────────────────┐    │
│  │ MemoryWorthiness     │◄───────────────│   MEMORY CONTRIBUTION       │    │
│  │ Signature            │                │   - segment_effects index   │    │
│  │   store_cate_results │                │   - responder_profiles      │    │
│  └──────────────────────┘                └─────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### DSPy Signatures Consumed

The Heterogeneous Optimizer consumes evidence with treatment and segment context:

```python
# From CognitiveRAG Phase 2 - Consumed by Heterogeneous Optimizer

class EvidenceSynthesisSignature(dspy.Signature):
    """Synthesize evidence from multi-hop retrieval into coherent context."""
    evidence_items: List[Evidence] = dspy.InputField(desc="Retrieved evidence items")
    user_query: str = dspy.InputField(desc="Original user query")
    investigation_path: List[Dict] = dspy.InputField(desc="Multi-hop retrieval path")

    synthesized_summary: str = dspy.OutputField(desc="Coherent synthesis of all evidence")
    treatment_context: str = dspy.OutputField(desc="Prior knowledge about treatment effects")
    segment_insights: List[Dict] = dspy.OutputField(desc="Known segment characteristics")
    prior_cate_estimates: Dict[str, float] = dspy.OutputField(desc="Historical CATE by segment")
    confidence_score: float = dspy.OutputField(desc="Confidence in synthesis 0.0-1.0")
```

### Integration in CATE Estimator Node

```python
# src/agents/heterogeneous_optimizer/nodes/cate_estimator.py

from typing import TypedDict, List, Dict, Any, Optional

class HetOptCognitiveContext(TypedDict):
    """Cognitive context from CognitiveRAG for CATE estimation."""
    synthesized_summary: str
    treatment_context: str
    segment_insights: List[Dict[str, Any]]
    prior_cate_estimates: Dict[str, float]
    evidence_confidence: float

class CATEEstimatorNode:
    """CATE Estimator with Cognitive RAG integration."""

    async def execute(self, state: HeterogeneousOptimizerState) -> HeterogeneousOptimizerState:
        start_time = time.time()

        try:
            # Extract cognitive context if available
            cognitive_context = self._extract_cognitive_context(state)

            # Fetch and prepare data
            df = await self._fetch_data(state)

            # Fit Causal Forest with prior-informed parameters
            cf = self._build_causal_forest(state, cognitive_context)

            await asyncio.wait_for(
                asyncio.to_thread(cf.fit, Y, T, X=X, W=W),
                timeout=self.timeout_seconds
            )

            # Get CATE estimates
            cate_by_segment = await self._calculate_cate_by_segment(
                df, cf, state, cognitive_context
            )

            # Validate against prior estimates if available
            validation_metrics = self._validate_against_priors(
                cate_by_segment, cognitive_context
            )

            return {
                **state,
                "cate_by_segment": cate_by_segment,
                "prior_validation": validation_metrics,
                "cognitive_enrichment_applied": cognitive_context is not None,
                "status": "analyzing"
            }

        except Exception as e:
            return {
                **state,
                "errors": [{"node": "cate_estimator", "error": str(e)}],
                "status": "failed"
            }

    def _extract_cognitive_context(
        self,
        state: HeterogeneousOptimizerState
    ) -> Optional[HetOptCognitiveContext]:
        """Extract cognitive context from state."""

        cognitive_input = state.get("cognitive_enrichment")
        if not cognitive_input:
            return None

        return HetOptCognitiveContext(
            synthesized_summary=cognitive_input.get("synthesized_summary", ""),
            treatment_context=cognitive_input.get("treatment_context", ""),
            segment_insights=cognitive_input.get("segment_insights", []),
            prior_cate_estimates=cognitive_input.get("prior_cate_estimates", {}),
            evidence_confidence=cognitive_input.get("confidence_score", 0.5)
        )

    def _build_causal_forest(
        self,
        state: HeterogeneousOptimizerState,
        cognitive_context: Optional[HetOptCognitiveContext]
    ) -> Any:
        """Build CausalForestDML with prior-informed parameters."""

        from econml.dml import CausalForestDML
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        n_estimators = state.get("n_estimators", 100)
        min_samples_leaf = state.get("min_samples_leaf", 10)

        # Adjust parameters based on prior knowledge
        if cognitive_context and cognitive_context["prior_cate_estimates"]:
            # If we have strong priors, we can use fewer estimators
            prior_confidence = cognitive_context["evidence_confidence"]
            if prior_confidence > 0.8:
                n_estimators = max(50, int(n_estimators * 0.7))

        return CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=50, random_state=42),
            model_t=RandomForestClassifier(n_estimators=50, random_state=42),
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

    def _validate_against_priors(
        self,
        cate_by_segment: Dict[str, List[CATEResult]],
        cognitive_context: Optional[HetOptCognitiveContext]
    ) -> Dict[str, Any]:
        """Validate CATE estimates against prior knowledge."""

        if not cognitive_context or not cognitive_context["prior_cate_estimates"]:
            return {"prior_validation_available": False}

        priors = cognitive_context["prior_cate_estimates"]
        comparisons = []

        for segment_var, results in cate_by_segment.items():
            for result in results:
                segment_key = f"{segment_var}_{result['segment_value']}"
                if segment_key in priors:
                    prior_cate = priors[segment_key]
                    current_cate = result["cate_estimate"]

                    comparisons.append({
                        "segment": segment_key,
                        "prior_cate": prior_cate,
                        "current_cate": current_cate,
                        "deviation": abs(current_cate - prior_cate) / max(abs(prior_cate), 0.001),
                        "consistent": abs(current_cate - prior_cate) < abs(prior_cate) * 0.5
                    })

        if comparisons:
            consistency_rate = len([c for c in comparisons if c["consistent"]]) / len(comparisons)
        else:
            consistency_rate = None

        return {
            "prior_validation_available": True,
            "segments_compared": len(comparisons),
            "consistency_rate": consistency_rate,
            "comparisons": comparisons
        }
```

### Training Signal for MIPROv2

```python
# src/agents/heterogeneous_optimizer/training_signals.py

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class HetOptTrainingSignal:
    """
    Training signal for CognitiveRAG's EvidenceSynthesisSignature.

    Emitted after CATE analysis to inform MIPROv2 about
    the quality of treatment context and segment insights.
    """

    # Identifiers
    analysis_id: str
    analysis_timestamp: datetime

    # CATE Quality Metrics
    segments_analyzed: int
    high_responders_found: int
    low_responders_found: int
    heterogeneity_score: float

    # Prior Validation
    prior_validation_available: bool
    prior_consistency_rate: Optional[float]

    # Cognitive Context Usage
    treatment_context_used: bool
    segment_insights_used: bool
    prior_cate_used: bool

    # User Validation (collected post-analysis)
    policy_adoption_rate: Optional[float] = None  # % of recommendations adopted
    cate_accuracy_feedback: Optional[float] = None  # validation against experiments

    def compute_reward(self) -> float:
        """Compute reward signal for MIPROv2 optimization."""

        base_reward = 0.5

        # Reward for heterogeneity discovery
        if self.heterogeneity_score > 0.3:
            base_reward += 0.1 * min(self.heterogeneity_score, 0.8)

        # Reward for finding actionable segments
        total_responders = self.high_responders_found + self.low_responders_found
        if total_responders > 0:
            base_reward += 0.1 * min(total_responders / 10, 1.0)

        # Reward for prior consistency (cognitive context quality)
        if self.prior_validation_available and self.prior_consistency_rate is not None:
            base_reward += 0.15 * self.prior_consistency_rate

        # User feedback rewards
        if self.policy_adoption_rate is not None:
            base_reward += 0.15 * self.policy_adoption_rate

        return min(base_reward, 1.0)

    def to_training_example(self) -> Dict[str, Any]:
        """Convert to DSPy training example format."""
        return {
            "input": {
                "query_type": "heterogeneous_effect_analysis",
                "treatment_context_available": self.treatment_context_used,
                "segment_insights_available": self.segment_insights_used,
            },
            "output": {
                "heterogeneity_score": self.heterogeneity_score,
                "actionable_segments": self.high_responders_found + self.low_responders_found,
                "prior_consistency": self.prior_consistency_rate,
            },
            "reward": self.compute_reward(),
            "metadata": {
                "analysis_id": self.analysis_id,
                "timestamp": self.analysis_timestamp.isoformat(),
            }
        }


async def emit_het_opt_signal(
    result: HeterogeneousOptimizerOutput,
    state: HeterogeneousOptimizerState,
    signal_store: Any
) -> None:
    """Emit training signal after CATE analysis completion."""

    cognitive_enrichment = state.get("cognitive_enrichment")
    prior_validation = state.get("prior_validation", {})

    signal = HetOptTrainingSignal(
        analysis_id=str(uuid.uuid4()),
        analysis_timestamp=datetime.now(),
        segments_analyzed=len(state.get("cate_by_segment", {})),
        high_responders_found=len(result.high_responders),
        low_responders_found=len(result.low_responders),
        heterogeneity_score=result.heterogeneity_score,
        prior_validation_available=prior_validation.get("prior_validation_available", False),
        prior_consistency_rate=prior_validation.get("consistency_rate"),
        treatment_context_used=cognitive_enrichment is not None and bool(cognitive_enrichment.get("treatment_context")),
        segment_insights_used=cognitive_enrichment is not None and len(cognitive_enrichment.get("segment_insights", [])) > 0,
        prior_cate_used=cognitive_enrichment is not None and len(cognitive_enrichment.get("prior_cate_estimates", {})) > 0,
    )

    await signal_store.store_signal(
        agent="heterogeneous_optimizer",
        signal=signal.to_training_example()
    )
```

### Memory Contribution

```python
# src/agents/heterogeneous_optimizer/memory_contribution.py

from typing import Any
from datetime import datetime

async def contribute_to_memory(
    result: HeterogeneousOptimizerOutput,
    state: HeterogeneousOptimizerState,
    memory_backend: Any
) -> None:
    """
    Contribute CATE analysis results to CognitiveRAG's semantic memory.

    Stores in 'segment_effects' index for future retrieval.
    """

    # Store high responder profiles
    for responder in result.high_responders[:5]:  # Top 5
        segment_effect_entry = {
            "segment_id": responder["segment_id"],
            "responder_type": responder["responder_type"],
            "cate_estimate": responder["cate_estimate"],
            "defining_features": responder["defining_features"],
            "size_percentage": responder["size_percentage"],
            "recommendation": responder["recommendation"],
            "treatment_var": state["treatment_var"],
            "outcome_var": state["outcome_var"],
            "analysis_date": datetime.now().isoformat(),
            "heterogeneity_score": result.heterogeneity_score,
        }

        await memory_backend.store(
            memory_type="SEMANTIC",
            content=segment_effect_entry,
            metadata={
                "agent": "heterogeneous_optimizer",
                "index": "segment_effects",
                "embedding_fields": ["segment_id", "recommendation", "defining_features"]
            }
        )

    # Store low responder profiles for contrast
    for responder in result.low_responders[:3]:  # Top 3
        segment_effect_entry = {
            "segment_id": responder["segment_id"],
            "responder_type": responder["responder_type"],
            "cate_estimate": responder["cate_estimate"],
            "defining_features": responder["defining_features"],
            "size_percentage": responder["size_percentage"],
            "recommendation": responder["recommendation"],
            "treatment_var": state["treatment_var"],
            "outcome_var": state["outcome_var"],
            "analysis_date": datetime.now().isoformat(),
        }

        await memory_backend.store(
            memory_type="SEMANTIC",
            content=segment_effect_entry,
            metadata={
                "agent": "heterogeneous_optimizer",
                "index": "segment_effects",
                "embedding_fields": ["segment_id", "recommendation"]
            }
        )
```

### Cognitive-Enriched Input TypedDict

```python
# src/agents/heterogeneous_optimizer/state.py (additions)

class HeterogeneousOptimizerCognitiveInput(TypedDict):
    """Extended input with cognitive enrichment from CognitiveRAG."""

    # Standard inputs
    query: str
    treatment_var: str
    outcome_var: str
    segment_vars: List[str]
    effect_modifiers: List[str]
    data_source: str

    # Cognitive enrichment from Phase 2
    cognitive_enrichment: Optional[Dict[str, Any]]
    # Contains:
    #   - synthesized_summary: str
    #   - treatment_context: str
    #   - segment_insights: List[Dict]
    #   - prior_cate_estimates: Dict[str, float]
    #   - confidence_score: float

    # Working memory context
    working_memory: Optional[Dict[str, Any]]
```

### Configuration

```yaml
# config/agents/heterogeneous_optimizer.yaml

heterogeneous_optimizer:
  # ... existing config ...

  cognitive_rag_integration:
    enabled: true

    # Evidence consumption
    consume_treatment_context: true
    consume_segment_insights: true
    consume_prior_cate: true

    # Prior-informed estimation
    adjust_estimator_params: true
    validate_against_priors: true
    prior_deviation_threshold: 0.5  # Flag if >50% deviation

    # Training signal emission
    emit_training_signals: true
    signal_destination: "feedback_learner"

    # Memory contribution
    contribute_to_memory: true
    memory_index: "segment_effects"
    max_responders_to_store: 8  # 5 high + 3 low
```

### Testing Requirements

```python
# tests/unit/test_agents/test_heterogeneous_optimizer/test_cognitive_integration.py

import pytest
from src.agents.heterogeneous_optimizer.nodes.cate_estimator import CATEEstimatorNode
from src.agents.heterogeneous_optimizer.training_signals import HetOptTrainingSignal

class TestHetOptCognitiveIntegration:
    """Tests for Heterogeneous Optimizer Cognitive RAG integration."""

    @pytest.mark.asyncio
    async def test_cognitive_context_extraction(self):
        """Test extraction of cognitive context from state."""
        pass

    @pytest.mark.asyncio
    async def test_prior_informed_estimation(self):
        """Test parameter adjustment based on prior confidence."""
        pass

    @pytest.mark.asyncio
    async def test_prior_validation(self):
        """Test validation of CATE against prior estimates."""
        pass

    @pytest.mark.asyncio
    async def test_training_signal_emission(self):
        """Test training signal generation and reward computation."""
        signal = HetOptTrainingSignal(
            analysis_id="test",
            analysis_timestamp=datetime.now(),
            segments_analyzed=5,
            high_responders_found=2,
            low_responders_found=1,
            heterogeneity_score=0.4,
            prior_validation_available=True,
            prior_consistency_rate=0.8,
            treatment_context_used=True,
            segment_insights_used=True,
            prior_cate_used=True,
        )

        reward = signal.compute_reward()
        assert 0.0 <= reward <= 1.0

    @pytest.mark.asyncio
    async def test_memory_contribution(self):
        """Test contributing segment effects to semantic memory."""
        pass

    @pytest.mark.asyncio
    async def test_without_cognitive_enrichment(self):
        """Test graceful operation without cognitive context."""
        pass
```

---

## Discovery Integration (V4.4+)

### Overview

The Heterogeneous Optimizer can optionally leverage discovered causal structures from the Causal Impact agent's auto-discovery capability (V4.4+). This integration validates treatment-outcome relationships and discovers effect modifiers from learned DAGs.

### Integration Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            HETEROGENEOUS OPTIMIZER - DISCOVERY INTEGRATION                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FROM CAUSAL IMPACT AGENT                   HETEROGENEOUS OPTIMIZER          │
│  ┌──────────────────────┐                  ┌─────────────────────────────┐   │
│  │ discovery_result     │                  │                             │   │
│  │   - ensemble_dag     │─────────────────►│  1. Extract confounders     │   │
│  │   - edge_confidences │                  │  2. Validate effect_modifiers│   │
│  │   - gate_decision    │                  │  3. Log discovery metadata  │   │
│  │                      │                  │                             │   │
│  │ gate_evaluation      │                  │                             │   │
│  │   - confidence       │─────────────────►│  4. Adjust CATE confidence  │   │
│  │   - high_conf_edges  │                  │                             │   │
│  └──────────────────────┘                  └─────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### State Fields for Discovery

When receiving discovered structures from Causal Impact agent:

```python
class HeterogeneousOptimizerState(TypedDict):
    # ... existing fields ...

    # === DISCOVERY INTEGRATION (V4.4+) ===
    discovered_dag: Optional[Dict[str, Any]]  # From Causal Impact agent
    discovered_confounders: Optional[List[str]]  # Extracted from DAG
    discovery_gate_decision: Optional[str]  # accept/augment/review/reject
    discovery_confidence: Optional[float]  # Gate confidence score
    effect_modifiers_validated: Optional[bool]  # True if validated against DAG
    discovery_metadata: Optional[Dict[str, Any]]  # Additional discovery info
```

### Confounder Extraction

Extract confounders from discovered DAG for CATE estimation:

```python
def extract_confounders_from_dag(
    discovered_dag: Dict[str, Any],
    treatment_var: str,
    outcome_var: str
) -> List[str]:
    """
    Extract confounders from discovered DAG.

    Confounders are nodes that:
    - Have edges TO both treatment and outcome, OR
    - Lie on backdoor paths between treatment and outcome
    """
    edges = discovered_dag.get("edges", [])
    nodes = discovered_dag.get("nodes", [])

    # Find nodes with edges to both treatment and outcome
    confounders = []
    for node in nodes:
        if node in [treatment_var, outcome_var]:
            continue

        edges_to_treatment = any(
            e[0] == node and e[1] == treatment_var for e in edges
        )
        edges_to_outcome = any(
            e[0] == node and e[1] == outcome_var for e in edges
        )

        if edges_to_treatment and edges_to_outcome:
            confounders.append(node)

    return confounders
```

### Effect Modifier Validation

Validate that effect_modifiers are consistent with discovered structure:

```python
def validate_effect_modifiers(
    effect_modifiers: List[str],
    discovered_dag: Dict[str, Any],
    treatment_var: str,
    outcome_var: str
) -> Dict[str, Any]:
    """
    Validate effect modifiers against discovered DAG.

    Returns validation results with recommendations.
    """
    edges = discovered_dag.get("edges", [])
    edge_confidences = discovered_dag.get("edge_confidences", {})

    validated = []
    warnings = []

    for modifier in effect_modifiers:
        # Check if modifier affects outcome
        affects_outcome = any(
            e[0] == modifier and e[1] == outcome_var for e in edges
        )

        # Check if modifier interacts with treatment
        interacts_with_treatment = any(
            (e[0] == modifier and e[1] == treatment_var) or
            (e[0] == treatment_var and e[1] == modifier)
            for e in edges
        )

        if affects_outcome:
            validated.append({
                "modifier": modifier,
                "affects_outcome": True,
                "confidence": edge_confidences.get(f"{modifier}->{outcome_var}", 0.5)
            })
        else:
            warnings.append(
                f"{modifier} not found to affect {outcome_var} in discovered DAG"
            )

    return {
        "validated_modifiers": validated,
        "warnings": warnings,
        "validation_rate": len(validated) / len(effect_modifiers) if effect_modifiers else 1.0
    }
```

### CATE Confidence Adjustment

Adjust CATE confidence based on discovery quality:

```python
def adjust_cate_confidence(
    base_confidence: float,
    discovery_gate_decision: str,
    discovery_confidence: float,
    effect_modifiers_validation_rate: float
) -> float:
    """
    Adjust CATE confidence based on discovery quality.

    Higher discovery confidence → Higher CATE confidence
    ACCEPT gate decision → Boost confidence
    REJECT gate decision → Reduce confidence
    """
    gate_multipliers = {
        "accept": 1.1,
        "augment": 1.0,
        "review": 0.9,
        "reject": 0.8
    }

    multiplier = gate_multipliers.get(discovery_gate_decision, 1.0)

    # Factor in discovery confidence
    discovery_factor = 0.5 + (discovery_confidence * 0.5)

    # Factor in modifier validation
    validation_factor = 0.7 + (effect_modifiers_validation_rate * 0.3)

    adjusted = base_confidence * multiplier * discovery_factor * validation_factor

    return min(max(adjusted, 0.0), 1.0)
```

### Usage in HeterogeneousOptimizerState

```python
# State with discovery integration
state = HeterogeneousOptimizerState(
    query="What segments respond best to HCP engagement?",
    treatment_var="hcp_engagement_level",
    outcome_var="patient_conversion_rate",
    effect_modifiers=["geographic_region", "hcp_specialty", "decile"],

    # Discovery context from Causal Impact agent (V4.4+)
    discovered_dag={
        "nodes": ["hcp_engagement_level", "patient_conversion_rate", "geographic_region", "market_size"],
        "edges": [
            ("hcp_engagement_level", "patient_conversion_rate"),
            ("geographic_region", "hcp_engagement_level"),
            ("geographic_region", "patient_conversion_rate"),
            ("market_size", "patient_conversion_rate")
        ],
        "edge_confidences": {
            "hcp_engagement_level->patient_conversion_rate": 0.92,
            "geographic_region->hcp_engagement_level": 0.85,
            "geographic_region->patient_conversion_rate": 0.88
        }
    },
    discovery_gate_decision="accept",
    discovery_confidence=0.87,
)
```

### Configuration

```yaml
# config/agents/heterogeneous_optimizer.yaml

heterogeneous_optimizer:
  # ... existing config ...

  discovery_integration:
    enabled: true

    # Confounder extraction
    extract_confounders_from_dag: true
    min_edge_confidence_for_confounder: 0.6

    # Effect modifier validation
    validate_effect_modifiers: true
    warn_on_unvalidated_modifiers: true

    # CATE confidence adjustment
    adjust_cate_confidence: true
    require_discovery_for_high_confidence: false

    # Logging
    log_discovery_metadata: true
```

### Testing Requirements

```python
# tests/unit/test_agents/test_heterogeneous_optimizer/test_discovery_integration.py

class TestHetOptDiscoveryIntegration:
    """Tests for Heterogeneous Optimizer discovery integration."""

    def test_confounder_extraction_from_dag(self):
        """Test extracting confounders from discovered DAG."""
        pass

    def test_effect_modifier_validation(self):
        """Test validating effect modifiers against DAG."""
        pass

    def test_cate_confidence_adjustment(self):
        """Test confidence adjustment based on discovery quality."""
        pass

    def test_without_discovery_context(self):
        """Test graceful operation without discovery context."""
        pass

    def test_reject_gate_reduces_confidence(self):
        """Test that REJECT gate decision reduces CATE confidence."""
        pass
```
