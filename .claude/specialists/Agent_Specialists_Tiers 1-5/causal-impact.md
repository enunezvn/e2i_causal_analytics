# Tier 2: Causal Impact Agent Specialist

## Agent Classification

| Property | Value |
|----------|-------|
| **Tier** | 2 (Causal Analytics) |
| **Agent Type** | Hybrid (Computation + Deep Reasoning) |
| **Model Tier** | Sonnet + Opus |
| **Latency Tolerance** | Medium (up to 30s) |
| **Critical Path** | Core E2I mission agent |

## Domain Scope

You are the specialist for the Tier 2 Causal Impact Agent:
- `src/agents/causal_impact/` - Causal chain tracing and effect estimation

This is a **Hybrid Agent** with a two-node pattern:
1. **Computation Node** - DoWhy/EconML execution (Standard)
2. **Interpretation Node** - Natural language synthesis (Deep Reasoning)

## Design Principles

### The Core Tension

| Phase | Nature | Example Output |
|-------|--------|----------------|
| Computation | Deterministic, tool-bound | ATE = 0.23, CI: [0.18, 0.28], p < 0.001 |
| Interpretation | Reasoning-intensive, contextual | "Increasing HCP engagement frequency causes a 23% lift in prescription rates..." |

The computation phase executes DoWhy/EconML. The interpretation phase requires understanding the pharmaceutical domain, user expertise level, and framing causal findings as actionable insights.

## Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAUSAL IMPACT AGENT                          │
│                    (Hybrid Pattern)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐      ┌──────────────────────┐        │
│  │  COMPUTATION NODE    │      │  INTERPRETATION NODE │        │
│  │  (Standard Agent)    │ ───► │  (Deep Reasoning)    │        │
│  │                      │      │                      │        │
│  │  • DoWhy estimation  │      │  • Causal narrative  │        │
│  │  • EconML CATE       │      │  • Assumption audit  │        │
│  │  • Refutation tests  │      │  • Action framing    │        │
│  │  • Sensitivity       │      │  • Uncertainty comm. │        │
│  └──────────────────────┘      └──────────────────────┘        │
│           │                              │                      │
│           ▼                              ▼                      │
│  ┌──────────────────────┐      ┌──────────────────────┐        │
│  │  Structured Results  │      │  Natural Language    │        │
│  │  (JSON/DataFrame)    │      │  Explanation         │        │
│  └──────────────────────┘      └──────────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
causal_impact/
├── agent.py              # Main CausalImpactAgent class
├── state.py              # LangGraph state definitions
├── graph.py              # LangGraph assembly
├── nodes/
│   ├── graph_builder.py  # Causal DAG construction
│   ├── estimation.py     # DoWhy effect estimation
│   ├── refutation.py     # Robustness tests
│   ├── sensitivity.py    # Sensitivity analysis
│   └── interpretation.py # Deep reasoning node
├── chain_tracer.py       # Causal chain identification
├── effect_narrator.py    # Natural language generation
└── prompts.py            # Claude prompts for interpretation
```

## LangGraph State Definition

```python
# src/agents/causal_impact/state.py

from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
from datetime import datetime
import operator

class CausalGraphSpec(TypedDict):
    """Specification for the causal DAG"""
    treatment: str
    outcome: str
    confounders: List[str]
    mediators: Optional[List[str]]
    effect_modifiers: Optional[List[str]]
    instruments: Optional[List[str]]
    edges: List[tuple]
    dot_string: str
    description: str

class RefutationResults(TypedDict):
    """Results from DoWhy refutation tests"""
    placebo_treatment: Dict[str, Any]
    random_common_cause: Dict[str, Any]
    data_subset: Dict[str, Any]
    unobserved_common_cause: Optional[Dict[str, Any]]

class SensitivityAnalysis(TypedDict):
    """Sensitivity analysis results"""
    e_value: Optional[float]
    robustness_value: Optional[float]
    confounder_strength_bounds: Optional[Dict[str, float]]
    interpretation: str

class CausalImpactState(TypedDict):
    """Complete state for Causal Impact hybrid agent"""
    
    # === INPUT ===
    query: str
    treatment_var: str
    outcome_var: str
    confounders: List[str]
    data_source: str  # Table or query reference
    filters: Optional[Dict[str, Any]]
    
    # === CONFIGURATION ===
    interpretation_depth: Literal["none", "minimal", "standard", "deep"]
    user_expertise: Literal["executive", "analyst", "data_scientist"]
    estimation_method: str  # Default: "backdoor.econml.dml.CausalForestDML"
    confidence_level: float  # Default: 0.95
    
    # === COMPUTATION OUTPUTS ===
    causal_graph: Optional[CausalGraphSpec]
    ate_estimate: Optional[float]
    confidence_interval: Optional[tuple]
    p_value: Optional[float]
    refutation_results: Optional[RefutationResults]
    sensitivity_analysis: Optional[SensitivityAnalysis]
    cate_by_segment: Optional[Dict[str, Dict[str, float]]]
    
    # === INTERPRETATION OUTPUTS ===
    causal_narrative: Optional[str]
    assumption_warnings: Optional[List[str]]
    actionable_recommendations: Optional[List[str]]
    executive_summary: Optional[str]
    
    # === EXECUTION METADATA ===
    computation_latency_ms: int
    interpretation_latency_ms: int
    model_used: str
    timestamp: str
    
    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    fallback_used: bool
    retry_count: int
    status: Literal["pending", "computing", "interpreting", "completed", "failed"]
```

## Node Implementations

### Computation Node: Effect Estimation

```python
# src/agents/causal_impact/nodes/estimation.py

import asyncio
import time
import traceback
from typing import Optional
import pandas as pd

from ..state import CausalImpactState

class CausalEstimationNode:
    """
    Core computation node - DoWhy/EconML execution
    Optimized for reliability with fallback methods
    """
    
    ESTIMATION_METHODS = [
        "backdoor.econml.dml.CausalForestDML",
        "backdoor.econml.dml.LinearDML",
        "backdoor.linear_regression",
        "backdoor.propensity_score_weighting"
    ]
    
    def __init__(self, data_connector):
        self.data_connector = data_connector
        self.timeout_seconds = 120
    
    async def execute(self, state: CausalImpactState) -> CausalImpactState:
        start_time = time.time()
        
        try:
            from dowhy import CausalModel
            
            # Fetch data
            df = await self._fetch_data(state)
            
            if df is None or len(df) == 0:
                return {
                    **state,
                    "errors": [{"node": "estimation", "error": "No data returned"}],
                    "status": "failed"
                }
            
            # Build DoWhy model
            model = CausalModel(
                data=df,
                treatment=state["treatment_var"],
                outcome=state["outcome_var"],
                common_causes=state["confounders"],
                graph=state["causal_graph"]["dot_string"]
            )
            
            # Identify estimand
            identified_estimand = model.identify_effect(
                proceed_when_unidentifiable=True
            )
            
            # Estimate with fallback chain
            estimate = await self._estimate_with_fallback(
                model, 
                identified_estimand,
                state.get("estimation_method", self.ESTIMATION_METHODS[0])
            )
            
            # Get confidence intervals
            ci = self._safe_get_ci(estimate, state.get("confidence_level", 0.95))
            
            computation_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "ate_estimate": float(estimate.value),
                "confidence_interval": ci,
                "p_value": getattr(estimate, 'p_value', None),
                "computation_latency_ms": computation_time,
                "status": "computing"
            }
            
        except asyncio.TimeoutError:
            return {
                **state,
                "errors": [{"node": "estimation", "error": f"Timed out after {self.timeout_seconds}s"}],
                "warnings": ["Consider reducing data size"],
                "status": "failed"
            }
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "estimation", "error": str(e), "traceback": traceback.format_exc()}],
                "status": "failed"
            }
    
    async def _estimate_with_fallback(self, model, estimand, preferred_method: str):
        """Try methods in order until one succeeds"""
        
        methods = [preferred_method] + [m for m in self.ESTIMATION_METHODS if m != preferred_method]
        
        for method in methods:
            try:
                estimate = await asyncio.wait_for(
                    asyncio.to_thread(
                        model.estimate_effect,
                        estimand,
                        method_name=method,
                        method_params={
                            "init_params": {"n_estimators": 100, "random_state": 42},
                            "fit_params": {}
                        }
                    ),
                    timeout=self.timeout_seconds
                )
                return estimate
            except Exception as e:
                continue
        
        raise Exception("All estimation methods failed")
    
    async def _fetch_data(self, state: CausalImpactState) -> pd.DataFrame:
        """Fetch data from configured source"""
        return await self.data_connector.query(
            source=state["data_source"],
            columns=[state["treatment_var"], state["outcome_var"]] + state["confounders"],
            filters=state.get("filters")
        )
    
    def _safe_get_ci(self, estimate, confidence_level: float) -> tuple:
        """Safely extract confidence interval"""
        try:
            ci = estimate.get_confidence_intervals(confidence_level=confidence_level)
            return (float(ci[0]), float(ci[1]))
        except Exception:
            val = estimate.value
            return (val - abs(val), val + abs(val))
```

### Computation Node: Refutation Tests

```python
# src/agents/causal_impact/nodes/refutation.py

import asyncio
from typing import Dict, Any

from ..state import CausalImpactState, RefutationResults

class RefutationNode:
    """
    Run robustness checks on causal estimates
    Parallel execution with individual timeouts
    """
    
    def __init__(self, data_connector):
        self.data_connector = data_connector
        self.timeout_per_test = 60
    
    async def execute(self, state: CausalImpactState) -> CausalImpactState:
        
        if state.get("status") == "failed":
            return state
        
        try:
            from dowhy import CausalModel
            
            df = await self.data_connector.query(
                source=state["data_source"],
                columns=[state["treatment_var"], state["outcome_var"]] + state["confounders"],
                filters=state.get("filters")
            )
            
            model = CausalModel(
                data=df,
                treatment=state["treatment_var"],
                outcome=state["outcome_var"],
                common_causes=state["confounders"],
                graph=state["causal_graph"]["dot_string"]
            )
            
            estimand = model.identify_effect(proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(estimand, method_name=state["estimation_method"])
            
            # Run refutation tests in parallel
            tasks = [
                self._run_refutation(model, estimand, estimate, "placebo_treatment_refuter"),
                self._run_refutation(model, estimand, estimate, "random_common_cause"),
                self._run_refutation(model, estimand, estimate, "data_subset_refuter", {"subset_fraction": 0.8}),
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            refutation_results = RefutationResults(
                placebo_treatment=self._format_result(results[0]),
                random_common_cause=self._format_result(results[1]),
                data_subset=self._format_result(results[2]),
                unobserved_common_cause=None
            )
            
            return {
                **state,
                "refutation_results": refutation_results
            }
            
        except Exception as e:
            return {
                **state,
                "warnings": [f"Refutation tests failed: {str(e)}"],
                "refutation_results": None
            }
    
    async def _run_refutation(self, model, estimand, estimate, method: str, params: dict = None):
        """Run single refutation with timeout"""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    model.refute_estimate,
                    estimand,
                    estimate,
                    method_name=method,
                    **(params or {})
                ),
                timeout=self.timeout_per_test
            )
        except asyncio.TimeoutError:
            return {"error": "timeout", "method": method}
        except Exception as e:
            return {"error": str(e), "method": method}
    
    def _format_result(self, result) -> Dict[str, Any]:
        """Format refutation result"""
        if isinstance(result, dict) and "error" in result:
            return result
        if isinstance(result, Exception):
            return {"error": str(result)}
        
        return {
            "refutation_result": getattr(result, 'refutation_result', None),
            "estimated_effect": getattr(result, 'estimated_effect', None),
            "new_effect": getattr(result, 'new_effect', None),
            "passed": True
        }
```

### Deep Reasoning Node: Interpretation

```python
# src/agents/causal_impact/nodes/interpretation.py

import asyncio
import time
from typing import Dict, Any
from langchain_anthropic import ChatAnthropic

from ..state import CausalImpactState

class InterpretationNode:
    """
    Deep reasoning node for causal interpretation
    This is where the hybrid pattern's value shows
    """
    
    def __init__(self):
        # Primary: Sonnet for interpretation
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            timeout=90
        )
        # Fallback: Haiku for speed
        self.fallback_llm = ChatAnthropic(
            model="claude-haiku-4-20250414",
            max_tokens=2048,
            timeout=30
        )
    
    async def execute(self, state: CausalImpactState) -> CausalImpactState:
        start_time = time.time()
        
        if state.get("status") == "failed":
            return state
        
        # Skip if not requested
        if state.get("interpretation_depth") == "none":
            return {**state, "status": "completed", "interpretation_latency_ms": 0}
        
        # Minimal interpretation - no LLM
        if state.get("interpretation_depth") == "minimal":
            narrative = self._minimal_narrative(state)
            return {
                **state,
                "causal_narrative": narrative,
                "status": "completed",
                "interpretation_latency_ms": int((time.time() - start_time) * 1000)
            }
        
        # Deep interpretation with LLM
        try:
            prompt = self._build_interpretation_prompt(state)
            
            try:
                response = await asyncio.wait_for(
                    self.llm.ainvoke(prompt),
                    timeout=90
                )
                model_used = "claude-sonnet-4-20250514"
            except (asyncio.TimeoutError, Exception) as e:
                response = await self.fallback_llm.ainvoke(
                    self._build_simplified_prompt(state)
                )
                model_used = "claude-haiku-4-20250414 (fallback)"
                state = {**state, "warnings": state.get("warnings", []) + [f"Used fallback: {str(e)}"]}
            
            parsed = self._parse_interpretation(response.content)
            interpretation_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "causal_narrative": parsed.get("narrative", response.content),
                "assumption_warnings": parsed.get("warnings", []),
                "actionable_recommendations": parsed.get("recommendations", []),
                "executive_summary": parsed.get("executive_summary"),
                "interpretation_latency_ms": interpretation_time,
                "model_used": model_used,
                "status": "completed"
            }
            
        except Exception as e:
            return {
                **state,
                "causal_narrative": self._minimal_narrative(state),
                "errors": [{"node": "interpretation", "error": str(e)}],
                "fallback_used": True,
                "status": "completed"
            }
    
    def _minimal_narrative(self, state: CausalImpactState) -> str:
        """Template-based narrative without LLM"""
        ate = state["ate_estimate"]
        ci = state["confidence_interval"]
        treatment = state["treatment_var"]
        outcome = state["outcome_var"]
        
        direction = "increases" if ate > 0 else "decreases"
        significant = ci[0] > 0 or ci[1] < 0
        
        return (
            f"The estimated causal effect of {treatment} on {outcome} is {ate:.3f} "
            f"(95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]). "
            f"{'This effect is statistically significant.' if significant else 'This effect is not statistically significant.'}"
        )
    
    def _build_interpretation_prompt(self, state: CausalImpactState) -> str:
        """Full interpretation prompt with user expertise framing"""
        
        expertise_framing = {
            "executive": "Explain in business terms, focus on ROI and strategic implications. Avoid technical jargon.",
            "analyst": "Balance technical accuracy with accessibility. Include key statistics but explain their meaning.",
            "data_scientist": "Full technical detail. Discuss methodology limitations, assumption violations, and statistical nuances."
        }
        
        refutation_summary = "Not performed"
        if state.get("refutation_results"):
            ref = state["refutation_results"]
            refutation_summary = f"""- Placebo Treatment: {ref.get('placebo_treatment', 'N/A')}
- Random Common Cause: {ref.get('random_common_cause', 'N/A')}
- Data Subset: {ref.get('data_subset', 'N/A')}"""
        
        sensitivity_summary = "Not performed"
        if state.get("sensitivity_analysis"):
            sens = state["sensitivity_analysis"]
            sensitivity_summary = f"""- E-value: {sens.get('e_value', 'N/A')}
- Robustness Value: {sens.get('robustness_value', 'N/A')}
- Interpretation: {sens.get('interpretation', 'N/A')}"""
        
        return f"""You are a pharmaceutical causal inference expert interpreting results for {state['user_expertise']}-level stakeholders.

## Causal Analysis Results

**Research Question:** What is the causal effect of {state['treatment_var']} on {state['outcome_var']}?

**Estimated Effect:**
- Average Treatment Effect (ATE): {state['ate_estimate']:.4f}
- 95% Confidence Interval: [{state['confidence_interval'][0]:.4f}, {state['confidence_interval'][1]:.4f}]
- P-value: {state.get('p_value', 'Not calculated')}

**Causal Graph:**
{state['causal_graph']['description']}
Confounders controlled: {', '.join(state['confounders'])}

**Robustness Checks:**
{refutation_summary}

**Sensitivity to Unobserved Confounding:**
{sensitivity_summary}

---

## Interpretation Guidelines

{expertise_framing[state['user_expertise']]}

---

## Your Task

### 1. EXECUTIVE SUMMARY (2-3 sentences)
The key takeaway for decision-makers.

### 2. CAUSAL NARRATIVE (2-3 paragraphs)
- State the causal claim using causal language ("causes", "leads to")
- Quantify the effect in meaningful units
- Explain the mechanism if identifiable
- Address confidence based on robustness checks

### 3. ASSUMPTION WARNINGS (bullet list)
- Which causal assumptions might be at risk?
- Sensitivity to unobserved confounding
- Data quality concerns

### 4. ACTIONABLE RECOMMENDATIONS (bullet list)
- What should stakeholders DO based on this?
- What additional evidence would strengthen the claim?
- Suggested next steps

Format with clear headers. Be direct and actionable."""

    def _build_simplified_prompt(self, state: CausalImpactState) -> str:
        """Simplified prompt for fallback model"""
        return f"""Interpret this causal analysis:

Treatment: {state['treatment_var']}
Outcome: {state['outcome_var']}
Effect: {state['ate_estimate']:.4f} (CI: {state['confidence_interval']})

Provide:
1. One paragraph explaining what this means
2. Key caveats (2-3 bullets)
3. Recommended actions (2-3 bullets)"""

    def _parse_interpretation(self, content: str) -> Dict[str, Any]:
        """Parse LLM response into structured components"""
        result = {
            "narrative": content,
            "warnings": [],
            "recommendations": [],
            "executive_summary": None
        }
        
        sections = content.split("###")
        
        for section in sections:
            section_lower = section.lower()
            
            if "executive summary" in section_lower:
                result["executive_summary"] = section.split("\n", 1)[-1].strip()
            elif "warning" in section_lower or "assumption" in section_lower:
                lines = section.split("\n")
                result["warnings"] = [
                    line.strip("- •").strip() 
                    for line in lines 
                    if line.strip().startswith(("-", "•", "*"))
                ]
            elif "recommendation" in section_lower or "action" in section_lower:
                lines = section.split("\n")
                result["recommendations"] = [
                    line.strip("- •").strip() 
                    for line in lines 
                    if line.strip().startswith(("-", "•", "*"))
                ]
        
        return result
```

## Graph Assembly

```python
# src/agents/causal_impact/graph.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import CausalImpactState
from .nodes.graph_builder import CausalGraphBuilderNode
from .nodes.estimation import CausalEstimationNode
from .nodes.refutation import RefutationNode
from .nodes.sensitivity import SensitivityAnalysisNode
from .nodes.interpretation import InterpretationNode

def build_causal_impact_graph(
    data_connector,
    domain_graphs: dict = None,
    enable_checkpointing: bool = True
):
    """
    Build the complete Causal Impact hybrid agent graph
    
    Architecture:
        [graph_builder] → [estimation] → [refutation] → [sensitivity] → [interpretation]
                           │                                              │
                           └──────── (on failure) ────────────────────────┘
    """
    
    # Initialize nodes
    graph_builder = CausalGraphBuilderNode(domain_graphs)
    estimation = CausalEstimationNode(data_connector)
    refutation = RefutationNode(data_connector)
    sensitivity = SensitivityAnalysisNode()
    interpretation = InterpretationNode()
    
    # Build graph
    workflow = StateGraph(CausalImpactState)
    
    # Add nodes
    workflow.add_node("graph_builder", graph_builder.execute)
    workflow.add_node("estimation", estimation.execute)
    workflow.add_node("refutation", refutation.execute)
    workflow.add_node("sensitivity", sensitivity.execute)
    workflow.add_node("interpretation", interpretation.execute)
    workflow.add_node("error_handler", error_handler_node)
    
    # Entry point
    workflow.set_entry_point("graph_builder")
    
    # Edges with error handling
    workflow.add_conditional_edges(
        "graph_builder",
        lambda s: "error" if s.get("status") == "failed" else "estimation",
        {"estimation": "estimation", "error": "error_handler"}
    )
    
    workflow.add_conditional_edges(
        "estimation",
        route_after_estimation,
        {
            "refutation": "refutation",
            "interpretation": "interpretation",
            "error": "error_handler"
        }
    )
    
    workflow.add_edge("refutation", "sensitivity")
    workflow.add_edge("sensitivity", "interpretation")
    workflow.add_edge("interpretation", END)
    workflow.add_edge("error_handler", END)
    
    # Compile
    if enable_checkpointing:
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    return workflow.compile()

def route_after_estimation(state: CausalImpactState) -> str:
    """Route after estimation - handle partial failures"""
    if state.get("status") == "failed":
        if state.get("ate_estimate") is not None:
            return "interpretation"  # Proceed with what we have
        return "error"
    return "refutation"

async def error_handler_node(state: CausalImpactState) -> CausalImpactState:
    """Generate user-friendly error response"""
    errors = state.get("errors", [])
    
    error_summary = []
    for error in errors:
        node = error.get("node", "unknown")
        msg = error.get("error", "Unknown error")
        
        if "timeout" in msg.lower():
            error_summary.append(f"The {node} step timed out. Consider reducing data size.")
        elif "no data" in msg.lower():
            error_summary.append("No data available. Check your filters.")
        else:
            error_summary.append(f"Error in {node}: {msg}")
    
    return {
        **state,
        "causal_narrative": "Analysis could not be completed due to errors.",
        "assumption_warnings": error_summary,
        "actionable_recommendations": [
            "Review error messages above",
            "Check data availability and quality",
            "Consider simplifying the causal model"
        ],
        "status": "failed"
    }
```

## Integration Contracts

### Input Contract (from Orchestrator)
```python
class CausalImpactInput(BaseModel):
    query: str
    treatment_var: str
    outcome_var: str
    confounders: List[str]
    data_source: str
    filters: Optional[Dict[str, Any]] = None
    interpretation_depth: Literal["none", "minimal", "standard", "deep"] = "standard"
    user_expertise: Literal["executive", "analyst", "data_scientist"] = "analyst"
```

### Output Contract (to Orchestrator)
```python
class CausalImpactOutput(BaseModel):
    ate_estimate: float
    confidence_interval: Tuple[float, float]
    causal_narrative: str
    assumption_warnings: List[str]
    actionable_recommendations: List[str]
    executive_summary: Optional[str]
    confidence: float  # Based on refutation results
    computation_latency_ms: int
    interpretation_latency_ms: int
```

## Testing Requirements

```
tests/unit/test_agents/test_causal_impact/
├── test_graph_builder.py     # DAG construction
├── test_estimation.py        # DoWhy estimation
├── test_refutation.py        # Robustness tests
├── test_sensitivity.py       # Sensitivity analysis
├── test_interpretation.py    # Deep reasoning
└── test_integration.py       # End-to-end flow
```

### Performance Requirements
- Computation phase: <60s
- Interpretation phase: <30s
- Total latency: <120s (excluding data fetch)
- Fallback activation: <5s after primary timeout

### Test Cases
1. Successful estimation with full interpretation
2. Estimation timeout triggers fallback method
3. Refutation failure is non-fatal (warning only)
4. Deep interpretation adapts to user expertise
5. Minimal interpretation skips LLM entirely

## Handoff Format

```yaml
causal_impact_handoff:
  agent: causal_impact
  analysis_type: causal_effect_estimation
  key_findings:
    - ate: <effect estimate>
    - confidence_interval: [<lower>, <upper>]
    - significant: <bool>
  robustness:
    placebo_passed: <bool>
    subset_stable: <bool>
    sensitivity_e_value: <float>
  narrative: <interpretation summary>
  recommendations:
    - action: <recommended action>
      expected_impact: <estimated effect>
      confidence: <0.0-1.0>
  requires_further_analysis: <bool>
  suggested_next_agent: <heterogeneous_optimizer|experiment_designer>
```

## DoWhy/EconML Integration Notes

### Supported Estimation Methods
1. `backdoor.econml.dml.CausalForestDML` - Default, handles heterogeneity
2. `backdoor.econml.dml.LinearDML` - Faster, assumes linearity
3. `backdoor.linear_regression` - Simplest fallback
4. `backdoor.propensity_score_weighting` - For observational data

### Refutation Tests
1. **Placebo Treatment** - Effect should disappear with random treatment
2. **Random Common Cause** - Effect should be stable with added noise
3. **Data Subset** - Effect should be consistent across subsets

### Sensitivity Analysis
- **E-value** - How strong must unmeasured confounding be to explain away effect?
- **Robustness Value** - E-value for confidence interval bound

---

## Cognitive RAG DSPy Integration

The Causal Impact agent integrates with CognitiveRAG's 4-phase cognitive workflow, receiving enriched context from the investigation phase and providing feedback signals for DSPy optimization.

### Evidence Flow from CognitiveRAG

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COGNITIVE RAG → CAUSAL IMPACT FLOW                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CognitiveRAG Phase 2                CognitiveRAG Phase 3                │
│  ┌─────────────────┐                 ┌─────────────────┐                 │
│  │ Investigator    │                 │ Agent Phase     │                 │
│  │                 │   Evidence      │                 │                 │
│  │ Multi-hop       │────────────────►│ Evidence        │                 │
│  │ Retrieval       │   Collection    │ Synthesis       │                 │
│  │                 │                 │                 │                 │
│  └─────────────────┘                 └────────┬────────┘                 │
│         │                                     │                          │
│         │                                     ▼                          │
│         │                            ┌─────────────────┐                 │
│         │ causal_paths               │ Causal Impact   │                 │
│         │ evidence                   │ Agent           │                 │
│         └───────────────────────────►│                 │                 │
│                                      │ • DAG context   │                 │
│                                      │ • Prior effects │                 │
│                                      │ • Confounders   │                 │
│                                      └─────────────────┘                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### DSPy Signatures Relevant to Causal Impact

#### 1. EvidenceSynthesisSignature (from CognitiveRAG Phase 3)

The Causal Impact agent receives synthesized evidence from this signature:

```python
class EvidenceSynthesisSignature(dspy.Signature):
    """Synthesize evidence from multi-hop retrieval into coherent context."""

    evidence_items: List[Evidence] = dspy.InputField(desc="Retrieved evidence items")
    user_query: str = dspy.InputField(desc="Original user query")
    investigation_path: List[Dict] = dspy.InputField(desc="Multi-hop retrieval path")

    # Outputs used by Causal Impact agent
    synthesized_summary: str = dspy.OutputField(desc="Coherent synthesis of all evidence")
    causal_context: str = dspy.OutputField(desc="Extracted causal relationships from evidence")
    confidence_score: float = dspy.OutputField(desc="Confidence in synthesis 0.0-1.0")
    suggested_confounders: List[str] = dspy.OutputField(desc="Suggested confounders from evidence")
```

**Integration Point**: The `causal_context` and `suggested_confounders` are passed to the CausalGraphBuilderNode:

```python
# src/agents/causal_impact/nodes/graph_builder.py (integration mode)

async def execute(self, state: CausalImpactState) -> CausalImpactState:
    # Check for CognitiveRAG evidence context
    if state.get("cognitive_evidence"):
        evidence = state["cognitive_evidence"]

        # Augment confounders with DSPy suggestions
        suggested = evidence.get("suggested_confounders", [])
        base_confounders = state["confounders"]
        augmented_confounders = list(set(base_confounders + suggested))

        # Use causal context to inform DAG structure
        causal_context = evidence.get("causal_context", "")
        domain_hints = self._extract_domain_hints(causal_context)

        return await self._build_informed_dag(
            state,
            confounders=augmented_confounders,
            domain_hints=domain_hints
        )

    # Fall back to standard DAG building
    return await self._build_standard_dag(state)
```

#### 2. InvestigationPlanSignature Evidence

Causal Impact uses evidence from the `causal_paths` memory type:

```python
class CausalPathEvidence(TypedDict):
    """Evidence retrieved from causal_paths index."""

    treatment: str
    outcome: str
    estimated_effect: float
    confidence_interval: Tuple[float, float]
    methodology: str
    confounders_used: List[str]
    timestamp: str
    source_query: str
```

### DSPy Training Signals from Causal Impact

The Causal Impact agent provides valuable feedback signals for DSPy optimization:

#### 1. Effect Estimation Quality Signals

```python
class CausalImpactTrainingSignal:
    """Training signal for CognitiveRAG's investigation optimization."""

    def __init__(
        self,
        state: CausalImpactState,
        result: CausalImpactOutput,
        cognitive_evidence: Optional[Dict] = None
    ):
        self.query = state["query"]
        self.cognitive_evidence = cognitive_evidence
        self.treatment = state["treatment_var"]
        self.outcome = state["outcome_var"]

        # Analysis quality metrics
        self.estimation_success = result.ate_estimate is not None
        self.refutation_passed = self._check_refutations(state)
        self.confidence_interval_width = self._compute_ci_width(result)

        # Evidence quality assessment
        self.evidence_useful = self._assess_evidence_utility(
            cognitive_evidence,
            state
        )

    def _assess_evidence_utility(
        self,
        evidence: Optional[Dict],
        state: CausalImpactState
    ) -> float:
        """Assess how useful CognitiveRAG evidence was."""
        if not evidence:
            return 0.0

        score = 0.0

        # Did suggested confounders improve model?
        suggested = evidence.get("suggested_confounders", [])
        if suggested and len(set(suggested) & set(state["confounders"])) > 0:
            score += 0.3

        # Was causal context informative?
        if evidence.get("causal_context"):
            score += 0.3

        # Did prior effects align with current estimate?
        prior_effect = evidence.get("prior_effect_estimate")
        if prior_effect and state.get("ate_estimate"):
            alignment = 1 - abs(prior_effect - state["ate_estimate"]) / max(
                abs(prior_effect), abs(state["ate_estimate"]), 0.01
            )
            score += 0.4 * max(0, alignment)

        return min(score, 1.0)

    def to_dspy_example(self) -> dspy.Example:
        """Convert to DSPy training example for MIPROv2."""
        return dspy.Example(
            query=self.query,
            treatment=self.treatment,
            outcome=self.outcome,
            estimation_success=self.estimation_success,
            evidence_useful=self.evidence_useful,
            analysis_quality=self._compute_quality_score()
        ).with_inputs("query", "treatment", "outcome")
```

#### 2. Signal Collection for Feedback Learner

```python
# src/agents/causal_impact/training_signals.py

async def collect_training_signal(
    state: CausalImpactState,
    result: CausalImpactOutput
) -> Dict[str, Any]:
    """Collect training signal for DSPy optimization via feedback_learner."""

    return {
        "signal_type": "causal_analysis_quality",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": state["query"],
        "analysis_context": {
            "treatment": state["treatment_var"],
            "outcome": state["outcome_var"],
            "confounders": state["confounders"],
            "methodology": state.get("estimation_method")
        },
        "cognitive_evidence": state.get("cognitive_evidence"),
        "results": {
            "ate_estimate": result.ate_estimate,
            "confidence_interval": result.confidence_interval,
            "refutation_passed": all([
                r.get("passed", False)
                for r in (state.get("refutation_results") or {}).values()
                if isinstance(r, dict)
            ])
        },
        "quality_metrics": {
            "estimation_success": result.ate_estimate is not None,
            "interpretation_quality": len(result.causal_narrative) > 100,
            "evidence_utility": CausalImpactTrainingSignal(
                state, result, state.get("cognitive_evidence")
            ).evidence_useful
        },
        # For MIPROv2 optimization
        "optimization_targets": [
            "evidence_synthesis_quality",
            "confounder_suggestion_accuracy",
            "causal_context_relevance"
        ]
    }
```

### Memory Contribution to CognitiveRAG

The Causal Impact agent contributes to the semantic memory (causal_paths index):

```python
# After successful analysis, store in semantic memory

async def contribute_to_memory(
    result: CausalImpactOutput,
    state: CausalImpactState,
    memory_backend: MemoryBackend
) -> None:
    """Contribute analysis results to CognitiveRAG's causal_paths memory."""

    causal_path_entry = {
        "type": "causal_effect",
        "treatment": state["treatment_var"],
        "outcome": state["outcome_var"],
        "effect": {
            "ate": result.ate_estimate,
            "ci_lower": result.confidence_interval[0],
            "ci_upper": result.confidence_interval[1],
            "significant": result.confidence_interval[0] * result.confidence_interval[1] > 0
        },
        "methodology": {
            "estimation_method": state.get("estimation_method"),
            "confounders": state["confounders"],
            "refutation_results": state.get("refutation_results")
        },
        "interpretation": result.executive_summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_query": state["query"]
    }

    await memory_backend.store(
        memory_type=MemoryType.SEMANTIC,
        content=causal_path_entry,
        metadata={"agent": "causal_impact", "index": "causal_paths"}
    )
```

### CognitiveState Integration

When receiving queries from CognitiveRAG, the agent receives enriched context:

```python
class CausalImpactCognitiveInput(TypedDict):
    """Enhanced input from CognitiveRAG."""

    # Standard input
    query: str
    treatment_var: str
    outcome_var: str
    confounders: List[str]

    # CognitiveRAG enrichments
    cognitive_state: CognitiveState
    cognitive_evidence: Dict[str, Any]  # From EvidenceSynthesisSignature
    prior_causal_paths: List[Dict]  # Previously discovered paths
    investigation_context: Dict  # Multi-hop retrieval context
```

### Configuration

```yaml
# config/agents.yaml - Causal Impact DSPy integration

causal_impact:
  cognitive_rag_integration:
    enabled: true
    use_suggested_confounders: true  # Accept confounder suggestions
    store_results_to_memory: true  # Contribute to causal_paths

  training_signal_collection:
    enabled: true
    signal_types:
      - causal_analysis_quality
      - evidence_utility
      - confounder_suggestion_accuracy
    target_agent: feedback_learner

  memory_contribution:
    index: causal_paths
    min_confidence_to_store: 0.6  # Only store confident results
```

### Testing Requirements for DSPy Integration

```
tests/unit/test_agents/test_causal_impact/
├── test_cognitive_evidence_integration.py  # Evidence from CognitiveRAG
├── test_confounder_suggestion.py          # DSPy-suggested confounders
├── test_training_signal_emission.py       # Quality signals for MIPROv2
└── test_memory_contribution.py            # Storing to causal_paths
```

### Integration Test Cases

1. **Evidence augmentation**: Confounders are augmented with DSPy suggestions when provided
2. **Causal context usage**: DAG building uses causal context from evidence synthesis
3. **Prior path alignment**: Results align with previously discovered causal paths
4. **Training signal quality**: Emits accurate quality signals for optimization
5. **Memory contribution**: Successfully stores results to causal_paths index
