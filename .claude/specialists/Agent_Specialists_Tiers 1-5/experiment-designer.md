# Tier 3: Experiment Designer Agent Specialist

## Agent Classification

| Property | Value |
|----------|-------|
| **Tier** | 3 (Monitoring & Design) |
| **Agent Type** | Hybrid (Deep Reasoning + Computation) |
| **Model Tier** | Sonnet + Opus |
| **Latency Tolerance** | High (up to 60s) |
| **Critical Path** | No - can run async |

## Domain Scope

You are the specialist for the Tier 3 Experiment Designer Agent:
- `src/agents/experiment_designer/` - A/B test and experiment design

This is a **Hybrid Agent** with multiple deep reasoning phases:
1. **Design Reasoning** - Strategic experiment design (Deep)
2. **Power Analysis** - Sample size calculation (Computation)
3. **Validity Audit** - Adversarial critique (Deep)
4. **Template Generation** - DoWhy code output (Computation)

## Design Principles

### Deep Reasoning for Strategy
Experiment design requires creative exploration:
- Multiple design options (RCT, cluster RCT, quasi-experimental)
- Trade-off analysis
- Anticipating validity threats
- Domain-specific considerations

### Responsibilities
1. **Design Strategy** - Recommend optimal experiment design
2. **Power Analysis** - Calculate required sample size
3. **Validity Audit** - Identify threats and mitigations
4. **Pre-registration** - Generate analysis plan and DoWhy code

## Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT DESIGNER AGENT                         │
│                       (Hybrid Pattern)                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────┐                                             │
│  │  DESIGN REASONING  │ ◄── Deep Agent (Sonnet/Opus)                │
│  │  NODE              │     Extended reasoning for:                 │
│  │                    │     • Hypothesis refinement                 │
│  │                    │     • Design space exploration              │
│  │                    │     • Stratification strategy               │
│  └─────────┬──────────┘                                             │
│            │                                                         │
│            ▼                                                         │
│  ┌────────────────────┐                                             │
│  │  POWER ANALYSIS    │ ◄── Standard Computation                    │
│  │  NODE              │     • Sample size calculation               │
│  │                    │     • MDE given constraints                 │
│  │                    │     • Duration estimation                   │
│  └─────────┬──────────┘                                             │
│            │                                                         │
│            ▼                                                         │
│  ┌────────────────────┐                                             │
│  │  VALIDITY AUDIT    │ ◄── Deep Agent (adversarial)                │
│  │  NODE              │     • Internal validity threats             │
│  │                    │     • External validity limits              │
│  │                    │     • Mitigation recommendations            │
│  └─────────┬──────────┘                                             │
│            │                                                         │
│       ┌────┴────┐                                                   │
│       │Redesign?│──── Yes ───► Back to Power Analysis               │
│       └────┬────┘                                                   │
│            │ No                                                      │
│            ▼                                                         │
│  ┌────────────────────┐                                             │
│  │  TEMPLATE          │ ◄── Structured Output                       │
│  │  GENERATOR         │     • DAG specification                     │
│  │                    │     • Analysis code template                │
│  │                    │     • Pre-registration document             │
│  └────────────────────┘                                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
experiment_designer/
├── agent.py              # Main ExperimentDesignerAgent class
├── state.py              # LangGraph state definitions
├── graph.py              # LangGraph assembly with redesign loop
├── nodes/
│   ├── context_loader.py # Organizational learning context
│   ├── design_reasoning.py # Deep reasoning for design
│   ├── power_analysis.py # Statistical power calculations
│   ├── validity_audit.py # Adversarial validity critique
│   ├── redesign.py       # Incorporate audit feedback
│   └── template_generator.py # DoWhy code generation
├── design_patterns.py    # Common experiment patterns
└── prompts.py            # LLM prompts for reasoning
```

## LangGraph State Definition

```python
# src/agents/experiment_designer/state.py

from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
from datetime import datetime
import operator

class TreatmentDefinition(TypedDict):
    """Treatment arm specification"""
    name: str
    levels: List[str]
    operationalization: str
    dosage_description: Optional[str]

class OutcomeDefinition(TypedDict):
    """Outcome variable specification"""
    primary: str
    secondary: List[str]
    measurement_timing: str
    measurement_method: str
    type: Literal["continuous", "binary", "count", "time_to_event"]

class ValidityThreat(TypedDict):
    """Identified validity threat"""
    threat_type: Literal["selection_bias", "confounding", "measurement", "contamination", "temporal", "attrition"]
    description: str
    likelihood: Literal["low", "medium", "high"]
    severity: Literal["low", "medium", "high"]
    mitigation: str

class MitigationRecommendation(TypedDict):
    """Recommendation to address validity threat"""
    threat_addressed: str
    recommendation: str
    implementation_difficulty: Literal["low", "medium", "high"]

class ExperimentDesignState(TypedDict):
    """Complete state for Experiment Designer hybrid agent"""
    
    # === INPUT ===
    business_question: str
    constraints: Dict[str, Any]  # budget, timeline, ethical, operational
    available_data: Dict[str, Any]  # existing variables
    
    # === ORGANIZATIONAL CONTEXT ===
    similar_experiments: Optional[List[Dict]]
    organizational_defaults: Optional[Dict[str, Any]]
    recent_assumption_violations: Optional[List[Dict]]
    
    # === DESIGN OUTPUTS ===
    refined_hypothesis: Optional[str]
    treatment_definition: Optional[TreatmentDefinition]
    outcome_definition: Optional[OutcomeDefinition]
    design_type: Optional[Literal["rct", "cluster_rct", "quasi_did", "quasi_rdd", "quasi_iv"]]
    design_rationale: Optional[str]
    stratification_vars: Optional[List[str]]
    blocking_strategy: Optional[str]
    anticipated_confounders: Optional[List[Dict[str, str]]]
    
    # === POWER ANALYSIS OUTPUTS ===
    required_sample_size: Optional[int]
    achievable_mde: Optional[float]
    power: Optional[float]
    estimated_duration_weeks: Optional[int]
    power_analysis_details: Optional[Dict[str, Any]]
    
    # === VALIDITY AUDIT OUTPUTS ===
    internal_validity_threats: Optional[List[ValidityThreat]]
    external_validity_limits: Optional[List[str]]
    statistical_concerns: Optional[List[str]]
    mitigation_recommendations: Optional[List[MitigationRecommendation]]
    validity_score: Optional[Literal["strong", "moderate", "weak"]]
    proceed_recommendation: Optional[Literal["proceed", "proceed_with_caution", "redesign_needed"]]
    
    # === DOWHY INTEGRATION OUTPUTS ===
    causal_dag_spec: Optional[Dict[str, Any]]
    analysis_code_template: Optional[str]
    preregistration_doc: Optional[str]
    
    # === EXECUTION METADATA ===
    design_latency_ms: int
    power_latency_ms: int
    validity_latency_ms: int
    template_latency_ms: int
    total_latency_ms: int
    models_used: List[str]
    redesign_iterations: int
    timestamp: str
    
    # === ERROR HANDLING ===
    errors: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    fallback_used: bool
    status: Literal["pending", "designing", "analyzing_power", "auditing", "redesigning", "generating", "completed", "failed"]
```

## Node Implementations

### Design Reasoning Node (Deep)

```python
# src/agents/experiment_designer/nodes/design_reasoning.py

import asyncio
import time
import json
import re
from typing import Dict, Any
from langchain_anthropic import ChatAnthropic

from ..state import ExperimentDesignState

class DesignReasoningNode:
    """
    Deep reasoning for experiment design strategy
    Creative exploration of design space
    """
    
    def __init__(self):
        # Use Sonnet for design reasoning
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            timeout=120
        )
        # Fallback to Haiku
        self.fallback_llm = ChatAnthropic(
            model="claude-haiku-4-20250414",
            max_tokens=4096,
            timeout=60
        )
    
    async def execute(self, state: ExperimentDesignState) -> ExperimentDesignState:
        start_time = time.time()
        
        try:
            prompt = self._build_design_prompt(state)
            
            try:
                response = await asyncio.wait_for(
                    self.llm.ainvoke(prompt),
                    timeout=120
                )
                model_used = "claude-sonnet-4-20250514"
            except (asyncio.TimeoutError, Exception) as e:
                response = await self.fallback_llm.ainvoke(
                    self._build_simplified_prompt(state)
                )
                model_used = "claude-haiku-4-20250414 (fallback)"
                state = {**state, "warnings": state.get("warnings", []) + [f"Design used fallback: {str(e)}"]}
            
            design = self._parse_design_response(response.content)
            design_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "refined_hypothesis": design.get("refined_hypothesis"),
                "treatment_definition": design.get("treatment_definition"),
                "outcome_definition": design.get("outcome_definition"),
                "design_type": design.get("design_type"),
                "design_rationale": design.get("design_rationale"),
                "stratification_vars": design.get("stratification_vars", []),
                "blocking_strategy": design.get("blocking_strategy"),
                "anticipated_confounders": design.get("anticipated_confounders", []),
                "design_latency_ms": design_time,
                "models_used": [model_used],
                "status": "analyzing_power"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "design_reasoning", "error": str(e)}],
                "status": "failed"
            }
    
    def _build_design_prompt(self, state: ExperimentDesignState) -> str:
        """Full design prompt with organizational context"""
        
        org_context = self._build_org_context(state)
        
        return f"""You are an expert in causal inference and experimental design for pharmaceutical commercial operations.

## Organizational Learning Context
{org_context}

---

## Current Design Request

**Business Question:** {state['business_question']}

**Constraints:**
{json.dumps(state['constraints'], indent=2)}

**Available Data:**
{json.dumps(state['available_data'], indent=2)}

---

## Your Task

Design a rigorous experiment to answer this question. Think through multiple design options.

### Step 1: Hypothesis Refinement
- What is the precise causal claim to test?
- What is the treatment? (Be specific about dose/intensity/timing)
- What is the primary outcome? Secondary outcomes?

### Step 2: Design Space Exploration

**Option A: Randomized Controlled Trial**
- Unit of randomization: Individual HCP? Territory? Region?
- Feasibility given constraints?
- Pros/Cons?

**Option B: Cluster Randomized Trial**  
- What is the cluster? Why?
- How many clusters available?
- Intra-cluster correlation concerns?

**Option C: Quasi-Experimental Design**
- Difference-in-differences? Regression discontinuity? Instrumental variable?
- What natural variation could be exploited?
- What assumptions required?

### Step 3: Recommended Design
Based on trade-offs, recommend ONE design with full specification.

### Step 4: Output Format (CRITICAL - Must be valid JSON)

```json
{{
  "refined_hypothesis": "Precise causal claim to test",
  "treatment_definition": {{
    "name": "variable_name",
    "levels": ["control", "treatment"],
    "operationalization": "How treatment is implemented",
    "dosage_description": "If applicable"
  }},
  "outcome_definition": {{
    "primary": "primary_outcome_variable",
    "secondary": ["secondary_1", "secondary_2"],
    "measurement_timing": "When measured",
    "measurement_method": "How measured",
    "type": "continuous|binary|count|time_to_event"
  }},
  "design_type": "rct|cluster_rct|quasi_did|quasi_rdd|quasi_iv",
  "design_rationale": "2-3 sentences explaining why",
  "stratification_vars": ["var1", "var2"],
  "blocking_strategy": "Description or null",
  "anticipated_confounders": [
    {{"name": "confounder", "how_addressed": "How design handles it"}}
  ]
}}
```"""

    def _build_org_context(self, state: ExperimentDesignState) -> str:
        """Build organizational learning context section"""
        parts = []
        
        if state.get("similar_experiments"):
            parts.append(f"### Similar Past Experiments\n{json.dumps(state['similar_experiments'][:3], indent=2)}")
        
        if state.get("recent_assumption_violations"):
            parts.append(f"### Recent Assumption Violations\n{json.dumps(state['recent_assumption_violations'][:5], indent=2)}")
        
        if state.get("organizational_defaults"):
            defaults = state["organizational_defaults"]
            parts.append(f"""### Organizational Defaults
- Default effect size: {defaults.get('effect_size', 0.3)}
- Default ICC for clusters: {defaults.get('icc', 0.05)}
- Standard confounders: {defaults.get('standard_confounders', [])}""")
        
        return "\n\n".join(parts) if parts else "No historical context available."
    
    def _build_simplified_prompt(self, state: ExperimentDesignState) -> str:
        """Simplified prompt for fallback"""
        return f"""Design an experiment for: {state['business_question']}

Constraints: {json.dumps(state['constraints'])}

Provide JSON with:
- refined_hypothesis
- treatment_definition (name, levels, operationalization)
- outcome_definition (primary, secondary, type)
- design_type (rct, cluster_rct, quasi_did, quasi_rdd, quasi_iv)
- design_rationale
- stratification_vars
- anticipated_confounders"""

    def _parse_design_response(self, content: str) -> Dict[str, Any]:
        """Extract JSON from LLM response"""
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass
        
        return {"design_rationale": content}
```

### Power Analysis Node (Computation)

```python
# src/agents/experiment_designer/nodes/power_analysis.py

import time
from typing import Dict, Any
import numpy as np
from scipy import stats

from ..state import ExperimentDesignState

class PowerAnalysisNode:
    """
    Statistical power analysis
    Pure computation - no LLM needed
    """
    
    async def execute(self, state: ExperimentDesignState) -> ExperimentDesignState:
        start_time = time.time()
        
        if state.get("status") == "failed":
            return state
        
        try:
            design_type = state.get("design_type", "rct")
            constraints = state.get("constraints", {})
            outcome_type = state.get("outcome_definition", {}).get("type", "continuous")
            
            effect_size = constraints.get("expected_effect_size", 0.3)
            alpha = constraints.get("alpha", 0.05)
            power_target = constraints.get("power", 0.80)
            
            if design_type == "cluster_rct":
                result = self._cluster_rct_power(state, effect_size, alpha, power_target)
            elif outcome_type == "binary":
                result = self._binary_outcome_power(state, effect_size, alpha, power_target)
            else:
                result = self._continuous_outcome_power(effect_size, alpha, power_target)
            
            accrual_rate = constraints.get("weekly_accrual", 50)
            duration_weeks = max(1, int(np.ceil(result["sample_size"] / accrual_rate)))
            
            power_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "required_sample_size": result["sample_size"],
                "achievable_mde": result["mde"],
                "power": power_target,
                "estimated_duration_weeks": duration_weeks,
                "power_analysis_details": result["details"],
                "power_latency_ms": power_time,
                "status": "auditing"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "power_analysis", "error": str(e)}],
                "warnings": ["Power analysis failed - using default sample size"],
                "required_sample_size": 500,
                "power": None,
                "status": "auditing"
            }
    
    def _continuous_outcome_power(self, effect_size: float, alpha: float, power: float) -> Dict[str, Any]:
        """Power for continuous outcome (two-sample t-test)"""
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n_per_arm = int(np.ceil(2 * ((z_alpha + z_beta) / effect_size) ** 2))
        total_n = n_per_arm * 2
        
        return {
            "sample_size": total_n,
            "mde": effect_size,
            "details": {
                "n_per_arm": n_per_arm,
                "analysis_type": "two_sample_t_test",
                "assumptions": "Equal variance, normal distribution"
            }
        }
    
    def _binary_outcome_power(self, state: ExperimentDesignState, effect_size: float, alpha: float, power: float) -> Dict[str, Any]:
        """Power for binary outcome (two proportions)"""
        p1 = state.get("constraints", {}).get("baseline_rate", 0.3)
        p2 = p1 + effect_size * p1
        
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        p_bar = (p1 + p2) / 2
        n_per_arm = int(np.ceil(
            2 * p_bar * (1 - p_bar) * ((z_alpha + z_beta) / (p2 - p1)) ** 2
        ))
        
        return {
            "sample_size": n_per_arm * 2,
            "mde": p2 - p1,
            "details": {
                "n_per_arm": n_per_arm,
                "baseline_rate": p1,
                "expected_treatment_rate": p2,
                "analysis_type": "two_proportions_z_test"
            }
        }
    
    def _cluster_rct_power(self, state: ExperimentDesignState, effect_size: float, alpha: float, power: float) -> Dict[str, Any]:
        """Power for cluster RCT with ICC adjustment"""
        constraints = state.get("constraints", {})
        icc = constraints.get("expected_icc", 0.05)
        cluster_size = constraints.get("cluster_size", 20)
        
        base_result = self._continuous_outcome_power(effect_size, alpha, power)
        base_n = base_result["sample_size"]
        
        design_effect = 1 + (cluster_size - 1) * icc
        adjusted_n = int(np.ceil(base_n * design_effect))
        n_clusters = int(np.ceil(adjusted_n / cluster_size))
        
        return {
            "sample_size": adjusted_n,
            "mde": effect_size,
            "details": {
                "n_clusters_total": n_clusters,
                "n_clusters_per_arm": n_clusters // 2,
                "cluster_size": cluster_size,
                "icc": icc,
                "design_effect": design_effect,
                "analysis_type": "cluster_rct_adjusted"
            }
        }
```

### Validity Audit Node (Deep)

```python
# src/agents/experiment_designer/nodes/validity_audit.py

import asyncio
import time
import json
import re
from typing import Dict, Any
from langchain_anthropic import ChatAnthropic

from ..state import ExperimentDesignState

class ValidityAuditNode:
    """
    Deep reasoning for adversarial validity assessment
    Red team the experiment design
    """
    
    def __init__(self):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            timeout=90
        )
    
    async def execute(self, state: ExperimentDesignState) -> ExperimentDesignState:
        start_time = time.time()
        
        if state.get("status") == "failed":
            return state
        
        try:
            prompt = self._build_audit_prompt(state)
            
            response = await asyncio.wait_for(
                self.llm.ainvoke(prompt),
                timeout=90
            )
            
            audit = self._parse_audit_response(response.content)
            validity_time = int((time.time() - start_time) * 1000)
            
            return {
                **state,
                "internal_validity_threats": audit.get("internal_validity_threats", []),
                "external_validity_limits": audit.get("external_validity_limits", []),
                "statistical_concerns": audit.get("statistical_concerns", []),
                "mitigation_recommendations": audit.get("mitigation_recommendations", []),
                "validity_score": audit.get("overall_validity_score", "moderate"),
                "proceed_recommendation": audit.get("proceed_recommendation", "proceed_with_caution"),
                "validity_latency_ms": validity_time,
                "models_used": state.get("models_used", []) + ["claude-sonnet-4-20250514"],
                "status": "generating"
            }
            
        except asyncio.TimeoutError:
            return {
                **state,
                "warnings": state.get("warnings", []) + ["Validity audit timed out"],
                "validity_score": "unknown",
                "proceed_recommendation": "proceed_with_caution",
                "status": "generating"
            }
        except Exception as e:
            return {
                **state,
                "warnings": state.get("warnings", []) + [f"Validity audit failed: {str(e)}"],
                "status": "generating"
            }
    
    def _build_audit_prompt(self, state: ExperimentDesignState) -> str:
        """Build adversarial audit prompt"""
        
        return f"""You are a methodological critic reviewing an experiment design. Your job is to find weaknesses.

## Proposed Experiment

**Hypothesis:** {state.get('refined_hypothesis', 'Not specified')}
**Design Type:** {state.get('design_type', 'Not specified')}
**Treatment:** {json.dumps(state.get('treatment_definition', {}), indent=2)}
**Outcome:** {json.dumps(state.get('outcome_definition', {}), indent=2)}
**Sample Size:** {state.get('required_sample_size', 'Not calculated')}
**Stratification:** {state.get('stratification_vars', [])}
**Anticipated Confounders:** {json.dumps(state.get('anticipated_confounders', []), indent=2)}

---

## Audit Checklist

### Internal Validity Threats
For each threat, assess likelihood (low/medium/high), severity, and mitigation:

1. **Selection Bias** - Is randomization truly random?
2. **Confounding** - What confounders might be MISSED?
3. **Measurement** - Could outcome measurement differ between arms?
4. **Contamination/Spillover** - Could control be exposed to treatment?
5. **Temporal** - History, maturation, regression to mean?
6. **Attrition** - Differential dropout?

### External Validity
- What populations does this generalize to?
- What contexts would NOT transfer?

### Statistical Concerns
- Is power analysis realistic?
- Multiple comparison issues?

---

## Output (Must be valid JSON)

```json
{{
  "internal_validity_threats": [
    {{
      "threat_type": "selection_bias|confounding|measurement|contamination|temporal|attrition",
      "description": "Specific concern",
      "likelihood": "low|medium|high",
      "severity": "low|medium|high",
      "mitigation": "How to address"
    }}
  ],
  "external_validity_limits": ["Limit 1", "Limit 2"],
  "statistical_concerns": ["Concern 1", "Concern 2"],
  "mitigation_recommendations": [
    {{
      "threat_addressed": "Which threat",
      "recommendation": "What to do",
      "implementation_difficulty": "low|medium|high"
    }}
  ],
  "overall_validity_score": "strong|moderate|weak",
  "proceed_recommendation": "proceed|proceed_with_caution|redesign_needed"
}}
```"""

    def _parse_audit_response(self, content: str) -> Dict[str, Any]:
        """Parse audit JSON from response"""
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        return {
            "overall_validity_score": "moderate",
            "proceed_recommendation": "proceed_with_caution",
            "internal_validity_threats": [],
            "external_validity_limits": ["Unable to fully assess"],
            "mitigation_recommendations": []
        }
```

### DoWhy Template Generator Node

```python
# src/agents/experiment_designer/nodes/template_generator.py

import time
from datetime import datetime
from typing import Dict, Any

from ..state import ExperimentDesignState

class TemplateGeneratorNode:
    """
    Generate DoWhy-compatible outputs
    Structured code generation
    """
    
    async def execute(self, state: ExperimentDesignState) -> ExperimentDesignState:
        start_time = time.time()
        
        if state.get("status") == "failed":
            return state
        
        try:
            dag_spec = self._build_dag_spec(state)
            code_template = self._generate_analysis_code(state, dag_spec)
            prereg = self._generate_preregistration(state)
            
            template_time = int((time.time() - start_time) * 1000)
            total_time = (
                state.get("design_latency_ms", 0) +
                state.get("power_latency_ms", 0) +
                state.get("validity_latency_ms", 0) +
                template_time
            )
            
            return {
                **state,
                "causal_dag_spec": dag_spec,
                "analysis_code_template": code_template,
                "preregistration_doc": prereg,
                "template_latency_ms": template_time,
                "total_latency_ms": total_time,
                "status": "completed"
            }
            
        except Exception as e:
            return {
                **state,
                "errors": [{"node": "template_generator", "error": str(e)}],
                "status": "failed"
            }
    
    def _build_dag_spec(self, state: ExperimentDesignState) -> Dict[str, Any]:
        """Build DoWhy-compatible DAG"""
        treatment = state.get("treatment_definition", {}).get("name", "treatment")
        outcome = state.get("outcome_definition", {}).get("primary", "outcome")
        confounders = [c.get("name", c) if isinstance(c, dict) else c
                       for c in state.get("anticipated_confounders", [])]
        stratification = state.get("stratification_vars", [])
        
        edges = [(treatment, outcome)]
        for conf in confounders:
            edges.extend([(conf, treatment), (conf, outcome)])
        for strat in stratification:
            if strat not in confounders:
                edges.append((strat, outcome))
        
        dot_lines = ["digraph {"]
        for src, tgt in edges:
            dot_lines.append(f'    "{src}" -> "{tgt}";')
        dot_lines.append("}")
        
        return {
            "treatment": treatment,
            "outcome": outcome,
            "confounders": confounders,
            "effect_modifiers": stratification,
            "edges": edges,
            "dot_string": "\n".join(dot_lines)
        }
    
    def _generate_analysis_code(self, state: ExperimentDesignState, dag_spec: Dict) -> str:
        """Generate Python analysis template"""
        return f'''"""
E2I Experiment Analysis Template
================================
Experiment: {state.get('refined_hypothesis', 'Not specified')}
Design: {state.get('design_type', 'rct')}
Generated: {datetime.now().isoformat()}
"""

import pandas as pd
from dowhy import CausalModel
from econml.dml import CausalForestDML

# === DATA LOADING ===
df = pd.read_parquet("experiment_results.parquet")

# === CAUSAL MODEL ===
model = CausalModel(
    data=df,
    treatment="{dag_spec['treatment']}",
    outcome="{dag_spec['outcome']}",
    common_causes={dag_spec['confounders']},
    effect_modifiers={dag_spec['effect_modifiers']},
    graph="""{dag_spec['dot_string']}"""
)

# === IDENTIFICATION ===
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# === PRIMARY ANALYSIS ===
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dml.CausalForestDML"
)
print(f"ATE: {{estimate.value:.4f}}")
print(f"95% CI: {{estimate.get_confidence_intervals()}}")

# === REFUTATION TESTS ===
refute_placebo = model.refute_estimate(
    identified_estimand, estimate,
    method_name="placebo_treatment_refuter"
)
print(f"Placebo test: {{refute_placebo}}")
'''

    def _generate_preregistration(self, state: ExperimentDesignState) -> str:
        """Generate pre-registration document"""
        return f"""# Experiment Pre-Registration

## Study Information
- **Title:** {state.get('refined_hypothesis', 'Experiment')}
- **Registration Date:** {datetime.now().strftime('%Y-%m-%d')}
- **Design Type:** {state.get('design_type', 'RCT')}

## Hypotheses
{state.get('refined_hypothesis', 'Not specified')}

## Design
- **Treatment:** {state.get('treatment_definition', {}).get('name', 'Not specified')}
- **Outcome:** {state.get('outcome_definition', {}).get('primary', 'Not specified')}
- **Sample Size:** {state.get('required_sample_size', 'Not calculated')}
- **Power:** {state.get('power', 0.80)}
- **Duration:** {state.get('estimated_duration_weeks', 'Unknown')} weeks

## Validity Assessment
- **Score:** {state.get('validity_score', 'Not assessed')}
- **Recommendation:** {state.get('proceed_recommendation', 'Not assessed')}

---
*Pre-registration auto-generated by E2I Experiment Designer*
"""
```

## Graph Assembly with Redesign Loop

```python
# src/agents/experiment_designer/graph.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import ExperimentDesignState
from .nodes.context_loader import OrganizationalContextNode
from .nodes.design_reasoning import DesignReasoningNode
from .nodes.power_analysis import PowerAnalysisNode
from .nodes.validity_audit import ValidityAuditNode
from .nodes.redesign import RedesignNode
from .nodes.template_generator import TemplateGeneratorNode

def build_experiment_designer_graph(
    knowledge_store=None,
    enable_checkpointing: bool = True,
    max_redesign_iterations: int = 2
):
    """
    Build the Experiment Designer hybrid agent graph with redesign loop
    """
    
    # Initialize nodes
    context_node = OrganizationalContextNode(knowledge_store)
    design_node = DesignReasoningNode()
    power_node = PowerAnalysisNode()
    validity_node = ValidityAuditNode()
    redesign_node = RedesignNode()
    template_node = TemplateGeneratorNode()
    
    # Build graph
    workflow = StateGraph(ExperimentDesignState)
    
    # Add nodes
    workflow.add_node("context", context_node.execute)
    workflow.add_node("design", design_node.execute)
    workflow.add_node("power", power_node.execute)
    workflow.add_node("validity", validity_node.execute)
    workflow.add_node("redesign", redesign_node.execute)
    workflow.add_node("template", template_node.execute)
    workflow.add_node("error_handler", error_handler_node)
    
    # Entry point
    workflow.set_entry_point("context")
    
    # Main flow
    workflow.add_edge("context", "design")
    
    workflow.add_conditional_edges(
        "design",
        lambda s: "error" if s.get("status") == "failed" else "power",
        {"power": "power", "error": "error_handler"}
    )
    
    workflow.add_edge("power", "validity")
    
    # Conditional redesign based on validity audit
    def should_redesign(state: ExperimentDesignState) -> str:
        if state.get("status") == "failed":
            return "error"
        
        if state.get("proceed_recommendation") == "redesign_needed":
            iterations = state.get("redesign_iterations", 0)
            if iterations < max_redesign_iterations:
                return "redesign"
        
        return "template"
    
    workflow.add_conditional_edges(
        "validity",
        should_redesign,
        {"redesign": "redesign", "template": "template", "error": "error_handler"}
    )
    
    workflow.add_edge("redesign", "power")
    workflow.add_edge("template", END)
    workflow.add_edge("error_handler", END)
    
    # Compile
    if enable_checkpointing:
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    return workflow.compile()
```

## Integration Contracts

### Input Contract
```python
class ExperimentDesignerInput(BaseModel):
    business_question: str
    constraints: Dict[str, Any]
    available_data: Dict[str, Any]
    preregistration_formality: Literal["light", "medium", "heavy"] = "medium"
```

### Output Contract
```python
class ExperimentDesignerOutput(BaseModel):
    refined_hypothesis: str
    design_type: str
    required_sample_size: int
    estimated_duration_weeks: int
    validity_score: str
    proceed_recommendation: str
    causal_dag_spec: Dict[str, Any]
    analysis_code_template: str
    preregistration_doc: str
    total_latency_ms: int
```

## Handoff Format

```yaml
experiment_designer_handoff:
  agent: experiment_designer
  analysis_type: experiment_design
  design:
    type: <rct|cluster_rct|quasi_*>
    hypothesis: <refined hypothesis>
    sample_size: <n>
    duration_weeks: <weeks>
  validity:
    score: <strong|moderate|weak>
    key_threats: [<threat 1>, <threat 2>]
    mitigations: [<mitigation 1>]
  outputs:
    dag_spec: <available>
    analysis_code: <available>
    preregistration: <available>
  recommendations:
    - <recommendation 1>
  requires_further_analysis: <bool>
  suggested_next_agent: <causal_impact>
```

---

## Cognitive RAG DSPy Integration

### Integration Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   EXPERIMENT DESIGNER ↔ COGNITIVE RAG DSPY                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────────────────────────────────────┐ │
│  │   EXPERIMENT    │    │            COGNITIVE RAG DSPY                   │ │
│  │   DESIGNER      │◄───│                                                 │ │
│  │                 │    │  ┌─────────────────────────────────────────┐   │ │
│  │ ┌─────────────┐ │    │  │  EvidenceSynthesisSignature             │   │ │
│  │ │   DESIGN    │ │    │  │  ├─ experiment_history: similar trials  │   │ │
│  │ │  REASONING  │◄├────│  │  ├─ hypothesis_patterns: refined tests  │   │ │
│  │ │   NODE      │ │    │  │  ├─ effect_size_priors: historical MDE  │   │ │
│  │ └─────────────┘ │    │  │  └─ validity_lessons: past issues       │   │ │
│  │       ↓         │    │  └─────────────────────────────────────────┘   │ │
│  │ ┌─────────────┐ │    │                                                 │ │
│  │ │   VALIDITY  │ │    │  ┌─────────────────────────────────────────┐   │ │
│  │ │   AUDIT     │◄├────│  │  PriorExperimentSignature (Optional)    │   │ │
│  │ │   NODE      │ │    │  │  ├─ similar_designs: past experiments   │   │ │
│  │ └─────────────┘ │    │  │  ├─ common_threats: historical issues   │   │ │
│  │       ↓         │    │  │  └─ successful_mitigations: what works  │   │ │
│  │       │         │    │  └─────────────────────────────────────────┘   │ │
│  │       ▼         │    │                                                 │ │
│  │ ┌─────────────┐ │    └─────────────────────────────────────────────────┘ │
│  │ │  TRAINING   │─┼───────────────────────────────────────────────────────►│
│  │ │  SIGNAL     │ │    MIPROv2 Optimizer (design quality metrics)         │
│  │ └─────────────┘ │                                                        │
│  └─────────────────┘                                                        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  MEMORY CONTRIBUTION: experiment_designs (SEMANTIC)                     ││
│  │  ├─ Stores: hypothesis, design_type, validity_score, outcome_achieved   ││
│  │  └─ Embedding: business_question + design_rationale + validity findings ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### DSPy Signature Consumption

The Experiment Designer agent consumes cognitive context via `EvidenceSynthesisSignature`:

```python
# Cognitive context fields consumed by Experiment Designer

class ExperimentCognitiveContext(TypedDict):
    """Cognitive context from CognitiveRAG for experiment design."""
    synthesized_summary: str  # Overall context synthesis
    experiment_history: List[Dict[str, Any]]  # Similar past experiments
    hypothesis_patterns: List[Dict[str, Any]]  # Successful hypothesis refinements
    effect_size_priors: Dict[str, float]  # Historical effect sizes by domain
    validity_lessons: List[Dict[str, Any]]  # Past validity threats and resolutions
    evidence_confidence: float  # Confidence in retrieved context
```

### Design Reasoning Node with Cognitive Integration

```python
# src/agents/experiment_designer/nodes/design_reasoning.py

from typing import Optional
from ..state import ExperimentDesignState

class ExperimentCognitiveContext(TypedDict):
    """Cognitive context from CognitiveRAG for experiment design."""
    synthesized_summary: str
    experiment_history: List[Dict[str, Any]]
    hypothesis_patterns: List[Dict[str, Any]]
    effect_size_priors: Dict[str, float]
    validity_lessons: List[Dict[str, Any]]
    evidence_confidence: float


class DesignReasoningNode:
    """Design reasoning with cognitive enrichment."""

    async def execute(
        self,
        state: ExperimentDesignState,
        cognitive_context: Optional[ExperimentCognitiveContext] = None
    ) -> ExperimentDesignState:
        """Execute design reasoning with optional cognitive enrichment."""

        # Build enhanced prompt with cognitive context
        prompt = self._build_design_prompt(state)

        if cognitive_context and cognitive_context.get("evidence_confidence", 0) > 0.3:
            prompt = self._enrich_with_cognitive_context(prompt, cognitive_context)

        # Execute LLM reasoning
        response = await self.llm.ainvoke(prompt)
        design = self._parse_design_response(response.content)

        # Validate against historical priors if available
        if cognitive_context and cognitive_context.get("effect_size_priors"):
            design = self._validate_effect_size(design, cognitive_context)

        return {
            **state,
            **design,
            "cognitive_enrichment_used": cognitive_context is not None,
            "status": "analyzing_power"
        }

    def _enrich_with_cognitive_context(
        self,
        prompt: str,
        context: ExperimentCognitiveContext
    ) -> str:
        """Enrich prompt with cognitive context."""

        enrichment_sections = []

        # Add similar experiment history
        if context.get("experiment_history"):
            history_text = self._format_experiment_history(context["experiment_history"][:3])
            enrichment_sections.append(f"""
### Historical Experiment Insights (from Organizational Memory)
{history_text}
""")

        # Add hypothesis refinement patterns
        if context.get("hypothesis_patterns"):
            patterns_text = self._format_hypothesis_patterns(context["hypothesis_patterns"][:3])
            enrichment_sections.append(f"""
### Successful Hypothesis Patterns
{patterns_text}
""")

        # Add effect size priors
        if context.get("effect_size_priors"):
            priors = context["effect_size_priors"]
            priors_text = "\n".join([f"- {domain}: {size:.3f}" for domain, size in priors.items()])
            enrichment_sections.append(f"""
### Historical Effect Size Priors
{priors_text}
Consider these when estimating expected effect size.
""")

        # Add validity lessons
        if context.get("validity_lessons"):
            lessons_text = self._format_validity_lessons(context["validity_lessons"][:5])
            enrichment_sections.append(f"""
### Organizational Validity Lessons
{lessons_text}
""")

        if enrichment_sections:
            cognitive_section = f"""
## Cognitive RAG Context (Confidence: {context.get('evidence_confidence', 0):.2f})
{"".join(enrichment_sections)}
---
"""
            # Insert after organizational context section
            if "## Current Design Request" in prompt:
                prompt = prompt.replace(
                    "## Current Design Request",
                    f"{cognitive_section}\n## Current Design Request"
                )

        return prompt

    def _validate_effect_size(
        self,
        design: Dict[str, Any],
        context: ExperimentCognitiveContext
    ) -> Dict[str, Any]:
        """Validate expected effect size against historical priors."""

        priors = context.get("effect_size_priors", {})
        design_domain = design.get("design_domain", "general")

        if design_domain in priors:
            historical_prior = priors[design_domain]
            expected = design.get("expected_effect_size", 0.3)

            # Flag if expected effect is much larger than historical
            if expected > historical_prior * 2:
                design["warnings"] = design.get("warnings", []) + [
                    f"Expected effect size ({expected:.3f}) is much larger than "
                    f"historical prior ({historical_prior:.3f}) for {design_domain}"
                ]

        return design
```

### Validity Audit Node with Cognitive Integration

```python
# src/agents/experiment_designer/nodes/validity_audit.py

class ValidityAuditNode:
    """Validity audit with cognitive enrichment from past lessons."""

    async def execute(
        self,
        state: ExperimentDesignState,
        cognitive_context: Optional[ExperimentCognitiveContext] = None
    ) -> ExperimentDesignState:
        """Execute validity audit with historical lessons."""

        prompt = self._build_audit_prompt(state)

        # Enrich with historical validity lessons
        if cognitive_context and cognitive_context.get("validity_lessons"):
            prompt = self._add_historical_lessons(prompt, cognitive_context["validity_lessons"])

        response = await self.llm.ainvoke(prompt)
        audit = self._parse_audit_response(response.content)

        # Cross-reference with known organizational issues
        if cognitive_context:
            audit = self._cross_reference_known_issues(audit, cognitive_context)

        return {
            **state,
            **audit,
            "status": "generating"
        }

    def _add_historical_lessons(
        self,
        prompt: str,
        lessons: List[Dict[str, Any]]
    ) -> str:
        """Add historical validity lessons to audit prompt."""

        lessons_section = """
## Historical Validity Lessons (Organization Memory)

The following validity issues have occurred in similar past experiments:

"""
        for lesson in lessons[:5]:
            lessons_section += f"""
### {lesson.get('experiment_type', 'Unknown')} Experiment
- **Issue**: {lesson.get('threat_encountered', 'Unknown')}
- **Severity**: {lesson.get('severity', 'Unknown')}
- **Resolution**: {lesson.get('resolution', 'Unknown')}
- **Lesson**: {lesson.get('lesson_learned', 'None recorded')}
"""

        lessons_section += """
**Consider whether similar issues could affect this design.**
"""

        return prompt + lessons_section

    def _cross_reference_known_issues(
        self,
        audit: Dict[str, Any],
        context: ExperimentCognitiveContext
    ) -> Dict[str, Any]:
        """Cross-reference audit findings with known organizational issues."""

        known_threats = set()
        for lesson in context.get("validity_lessons", []):
            known_threats.add(lesson.get("threat_type", "").lower())

        for threat in audit.get("internal_validity_threats", []):
            threat_type = threat.get("threat_type", "").lower()
            if threat_type in known_threats:
                threat["organizational_precedent"] = True
                threat["recommendation"] = (
                    threat.get("recommendation", "") +
                    " (Note: Similar issues occurred in past experiments - check organizational lessons)"
                )

        return audit
```

### Training Signal for MIPROv2 Optimization

```python
# src/agents/experiment_designer/training_signal.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentDesignerTrainingSignal:
    """Training signal for MIPROv2 optimization of experiment design."""

    # Design quality metrics
    design_completed: bool
    validity_score: str  # "strong", "moderate", "weak"
    redesign_iterations: int

    # Cognitive enrichment metrics
    cognitive_context_used: bool
    historical_lessons_applied: int
    effect_size_validated: bool

    # Output quality metrics
    hypothesis_refined: bool
    dag_generated: bool
    preregistration_complete: bool

    # Execution outcome (if available)
    experiment_executed: Optional[bool] = None
    actual_validity_issues: Optional[int] = None
    effect_size_achieved: Optional[float] = None

    def compute_reward(self) -> float:
        """Compute reward for MIPROv2 optimization."""

        if not self.design_completed:
            return 0.0

        base_reward = 0.3

        # Reward for validity score
        validity_rewards = {"strong": 0.25, "moderate": 0.15, "weak": 0.05}
        base_reward += validity_rewards.get(self.validity_score, 0.1)

        # Penalize excessive redesigns
        if self.redesign_iterations > 2:
            base_reward -= 0.1 * (self.redesign_iterations - 2)

        # Reward cognitive enrichment usage
        if self.cognitive_context_used:
            base_reward += 0.1
            if self.historical_lessons_applied > 0:
                base_reward += 0.05 * min(self.historical_lessons_applied, 3)
            if self.effect_size_validated:
                base_reward += 0.05

        # Reward completeness
        if self.hypothesis_refined:
            base_reward += 0.05
        if self.dag_generated:
            base_reward += 0.05
        if self.preregistration_complete:
            base_reward += 0.05

        # If experiment was executed, weight actual outcomes heavily
        if self.experiment_executed is not None:
            if self.experiment_executed:
                base_reward += 0.1

                # Reward if fewer validity issues than predicted
                if self.actual_validity_issues is not None and self.actual_validity_issues == 0:
                    base_reward += 0.15

                # Reward if effect size was close to expected
                if self.effect_size_achieved is not None:
                    # This would need expected effect size from state
                    base_reward += 0.1

        return min(base_reward, 1.0)
```

### Memory Contribution

The Experiment Designer contributes to organizational memory:

```python
# Memory contribution after experiment design completion

async def contribute_to_memory(
    state: ExperimentDesignState,
    memory_backend: MemoryBackend
) -> None:
    """Store experiment design in organizational memory."""

    design_record = {
        "experiment_id": state.get("experiment_id", ""),
        "business_question": state["business_question"],
        "hypothesis": state.get("refined_hypothesis", ""),
        "design_type": state.get("design_type", ""),
        "design_rationale": state.get("design_rationale", ""),
        "sample_size": state.get("required_sample_size", 0),
        "validity_score": state.get("validity_score", ""),
        "validity_threats": state.get("internal_validity_threats", []),
        "mitigations_applied": state.get("mitigation_recommendations", []),
        "effect_size_expected": state.get("constraints", {}).get("expected_effect_size", 0),
        "timestamp": state.get("timestamp", ""),
        "status": state.get("status", ""),
    }

    await memory_backend.store(
        memory_type="SEMANTIC",
        content=design_record,
        metadata={
            "agent": "experiment_designer",
            "index": "experiment_designs",
            "embedding_fields": [
                "business_question",
                "hypothesis",
                "design_rationale"
            ],
        }
    )
```

### Cognitive Input TypedDict

```python
# src/agents/experiment_designer/cognitive_input.py

from typing import TypedDict, List, Dict, Any, Optional


class ExperimentDesignerCognitiveInput(TypedDict):
    """Full cognitive input for Experiment Designer agent."""

    # Standard input
    business_question: str
    constraints: Dict[str, Any]
    available_data: Dict[str, Any]

    # Cognitive enrichment (optional)
    cognitive_context: Optional[ExperimentCognitiveContext]

    # Organizational context (from knowledge stores)
    similar_experiments: Optional[List[Dict]]
    organizational_defaults: Optional[Dict[str, Any]]
    recent_assumption_violations: Optional[List[Dict]]
```

### Configuration

```yaml
# config/agents/experiment_designer.yaml

experiment_designer:
  tier: 3
  type: hybrid

  cognitive_rag:
    enabled: true
    context_sources:
      - experiment_designs
      - validity_lessons
      - effect_size_priors
    min_confidence_threshold: 0.3
    max_history_items: 5

  dspy:
    optimizer: MIPROv2
    training_signals:
      - design_quality
      - validity_prediction
      - effect_size_accuracy
    optimization_target: design_validity_score

  memory:
    contribution_enabled: true
    index: experiment_designs
    memory_type: SEMANTIC
    embedding_fields:
      - business_question
      - hypothesis
      - design_rationale
```

### Testing Requirements

```python
# tests/unit/test_agents/test_experiment_designer/test_cognitive_integration.py

@pytest.mark.asyncio
async def test_design_with_cognitive_context():
    """Test design reasoning with cognitive enrichment."""
    agent = ExperimentDesignerAgent()

    cognitive_context = ExperimentCognitiveContext(
        synthesized_summary="Prior HCP engagement experiments showed...",
        experiment_history=[
            {"type": "rct", "hypothesis": "...", "outcome": "positive"}
        ],
        hypothesis_patterns=[
            {"pattern": "dose_response", "success_rate": 0.85}
        ],
        effect_size_priors={"hcp_engagement": 0.25},
        validity_lessons=[
            {"threat_type": "selection_bias", "severity": "high"}
        ],
        evidence_confidence=0.78
    )

    result = await agent.design(
        business_question="Does HCP email frequency affect Rx volume?",
        constraints={"budget": 50000, "duration_weeks": 12},
        cognitive_context=cognitive_context
    )

    assert result.cognitive_enrichment_used is True
    assert "selection_bias" in str(result.internal_validity_threats).lower()


@pytest.mark.asyncio
async def test_training_signal_computation():
    """Test training signal reward computation."""
    signal = ExperimentDesignerTrainingSignal(
        design_completed=True,
        validity_score="strong",
        redesign_iterations=1,
        cognitive_context_used=True,
        historical_lessons_applied=2,
        effect_size_validated=True,
        hypothesis_refined=True,
        dag_generated=True,
        preregistration_complete=True
    )

    reward = signal.compute_reward()
    assert reward > 0.7  # High-quality design should have high reward
```

---

## Discovered DAG Integration (V4.4+)

### Overview

The Experiment Designer can accept pre-discovered causal graphs from the Causal Impact agent's auto-discovery capability (V4.4+). This allows experiment designs to be informed by data-driven causal structure rather than relying solely on domain knowledge.

### Integration Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              EXPERIMENT DESIGNER - DISCOVERY INTEGRATION                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FROM CAUSAL IMPACT AGENT                     EXPERIMENT DESIGNER            │
│  ┌──────────────────────┐                    ┌─────────────────────────────┐ │
│  │ discovered_dag_spec  │                    │                             │ │
│  │   - nodes            │───────────────────►│  1. Extract treatment-      │ │
│  │   - edges            │                    │     outcome relationships   │ │
│  │   - confidences      │                    │                             │ │
│  │                      │                    │  2. Identify confounders    │ │
│  │ gate_evaluation      │                    │     for randomization       │ │
│  │   - decision         │───────────────────►│                             │ │
│  │   - confidence       │                    │  3. Validate design against │ │
│  │   - reasons          │                    │     discovered structure    │ │
│  └──────────────────────┘                    └─────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### State Fields for Discovery

```python
class ExperimentDesignerState(TypedDict):
    # ... existing fields ...

    # === DISCOVERED DAG INTEGRATION (V4.4+) ===
    discovered_dag_spec: Optional[Dict[str, Any]]  # From Causal Impact agent
    discovery_gate_decision: Optional[str]  # accept/augment/review/reject
    discovery_confidence: Optional[float]  # Gate confidence score
    discovered_confounders: Optional[List[str]]  # For stratification
    dag_validated_design: Optional[bool]  # True if design validated against DAG
    discovery_warnings: Optional[List[str]]  # Issues with discovered structure
```

### Confounder-Based Stratification

Use discovered confounders for experiment stratification:

```python
def design_stratification_from_dag(
    discovered_dag: Dict[str, Any],
    treatment_var: str,
    outcome_var: str,
    max_strata: int = 4
) -> Dict[str, Any]:
    """
    Design stratification based on discovered confounders.

    Confounders are variables that affect both treatment assignment
    and outcome - these should be used for stratified randomization.
    """
    edges = discovered_dag.get("edges", [])
    edge_confidences = discovered_dag.get("edge_confidences", {})

    # Identify confounders (affect both treatment and outcome)
    confounders = []
    for node in discovered_dag.get("nodes", []):
        if node in [treatment_var, outcome_var]:
            continue

        affects_treatment = any(
            e[0] == node and e[1] == treatment_var for e in edges
        )
        affects_outcome = any(
            e[0] == node and e[1] == outcome_var for e in edges
        )

        if affects_treatment and affects_outcome:
            # Calculate combined confidence
            conf_treatment = edge_confidences.get(f"{node}->{treatment_var}", 0.5)
            conf_outcome = edge_confidences.get(f"{node}->{outcome_var}", 0.5)
            combined_conf = (conf_treatment + conf_outcome) / 2

            confounders.append({
                "variable": node,
                "confidence": combined_conf,
                "stratification_priority": combined_conf
            })

    # Sort by priority and limit
    confounders.sort(key=lambda x: x["stratification_priority"], reverse=True)
    selected = confounders[:max_strata]

    return {
        "stratification_variables": [c["variable"] for c in selected],
        "variable_confidences": {c["variable"]: c["confidence"] for c in selected},
        "total_confounders_found": len(confounders),
        "recommendation": f"Stratify on {len(selected)} variables for balanced randomization"
    }
```

### Design Validation Against DAG

Validate experiment design against discovered structure:

```python
def validate_design_against_dag(
    design: ExperimentDesign,
    discovered_dag: Dict[str, Any],
    discovery_confidence: float
) -> Dict[str, Any]:
    """
    Validate experiment design against discovered causal structure.

    Checks:
    - Treatment → Outcome edge exists
    - Confounders are controlled for
    - No unexpected mediators
    """
    edges = discovered_dag.get("edges", [])
    treatment = design.treatment_variable
    outcome = design.outcome_variable

    validations = []
    warnings = []

    # Check treatment → outcome relationship
    treatment_outcome_edge = any(
        e[0] == treatment and e[1] == outcome for e in edges
    )
    if treatment_outcome_edge:
        validations.append({
            "check": "treatment_outcome_relationship",
            "passed": True,
            "message": f"Discovered DAG confirms {treatment} → {outcome} relationship"
        })
    else:
        warnings.append(
            f"Warning: No direct edge {treatment} → {outcome} in discovered DAG. "
            "Effect may be mediated or spurious."
        )

    # Check confounder control
    discovered_confounders = [
        node for node in discovered_dag.get("nodes", [])
        if node not in [treatment, outcome]
        and any(e[0] == node and e[1] == treatment for e in edges)
        and any(e[0] == node and e[1] == outcome for e in edges)
    ]

    controlled = set(design.stratification_variables or [])
    uncontrolled = set(discovered_confounders) - controlled

    if uncontrolled:
        warnings.append(
            f"Uncontrolled confounders: {list(uncontrolled)}. "
            "Consider adding to stratification."
        )
    else:
        validations.append({
            "check": "confounder_control",
            "passed": True,
            "message": "All discovered confounders are controlled"
        })

    # Flag low discovery confidence
    if discovery_confidence < 0.5:
        warnings.append(
            f"Discovery confidence is low ({discovery_confidence:.2f}). "
            "DAG structure may be unreliable."
        )

    return {
        "validations": validations,
        "warnings": warnings,
        "overall_valid": len(warnings) == 0,
        "recommendation": "Proceed" if len(warnings) == 0 else "Review warnings"
    }
```

### Expert Review Flagging

Flag low-confidence discoveries for expert review:

```python
def flag_for_expert_review(
    discovery_gate_decision: str,
    discovery_confidence: float,
    design_complexity: str
) -> Dict[str, Any]:
    """
    Determine if experiment design needs expert review based on discovery.

    Returns:
        Review requirements and priority
    """
    requires_review = False
    priority = "low"
    reasons = []

    # Flag based on gate decision
    if discovery_gate_decision in ["review", "reject"]:
        requires_review = True
        reasons.append(f"Discovery gate decision: {discovery_gate_decision}")
        priority = "high" if discovery_gate_decision == "reject" else "medium"

    # Flag low confidence
    if discovery_confidence < 0.5:
        requires_review = True
        reasons.append(f"Low discovery confidence: {discovery_confidence:.2f}")
        if priority == "low":
            priority = "medium"

    # Flag complex designs with uncertain structure
    if design_complexity == "complex" and discovery_confidence < 0.7:
        requires_review = True
        reasons.append("Complex design with uncertain causal structure")
        priority = "high"

    return {
        "requires_expert_review": requires_review,
        "priority": priority,
        "reasons": reasons,
        "recommended_reviewer": "causal_methods_expert" if requires_review else None
    }
```

### Usage Example

```python
# State with discovered DAG integration
state = ExperimentDesignerState(
    business_question="Does increased HCP call frequency improve TRx?",
    hypothesis="Higher call frequency → Higher TRx",
    treatment_variable="call_frequency",
    outcome_variable="trx_volume",

    # Discovered DAG from Causal Impact agent (V4.4+)
    discovered_dag_spec={
        "nodes": ["call_frequency", "trx_volume", "geographic_region", "hcp_specialty", "market_size"],
        "edges": [
            ("call_frequency", "trx_volume"),
            ("geographic_region", "call_frequency"),
            ("geographic_region", "trx_volume"),
            ("hcp_specialty", "trx_volume"),
            ("market_size", "trx_volume")
        ],
        "edge_confidences": {
            "call_frequency->trx_volume": 0.88,
            "geographic_region->call_frequency": 0.92,
            "geographic_region->trx_volume": 0.85
        }
    },
    discovery_gate_decision="accept",
    discovery_confidence=0.85,
)

# ExperimentDesignerNode will:
# 1. Extract confounders: ["geographic_region"]
# 2. Recommend stratification on geographic_region
# 3. Validate design against discovered structure
# 4. Proceed without expert review (high confidence)
```

### Configuration

```yaml
# config/agents/experiment_designer.yaml

experiment_designer:
  # ... existing config ...

  discovery_integration:
    enabled: true

    # DAG acceptance
    accept_discovered_dags: true
    min_confidence_for_auto_use: 0.7

    # Stratification
    use_discovered_confounders: true
    max_stratification_variables: 4

    # Validation
    validate_design_against_dag: true
    flag_uncontrolled_confounders: true

    # Expert review
    flag_low_confidence_for_review: true
    low_confidence_threshold: 0.5
    auto_flag_reject_decisions: true
```

### Testing Requirements

```python
# tests/unit/test_agents/test_experiment_designer/test_discovery_integration.py

class TestExperimentDesignerDiscoveryIntegration:
    """Tests for Experiment Designer discovery integration."""

    def test_stratification_from_discovered_dag(self):
        """Test designing stratification from discovered confounders."""
        pass

    def test_design_validation_against_dag(self):
        """Test validating experiment design against DAG."""
        pass

    def test_expert_review_flagging(self):
        """Test flagging low-confidence discoveries for review."""
        pass

    def test_without_discovery_context(self):
        """Test graceful operation without discovery context."""
        pass

    def test_reject_decision_triggers_review(self):
        """Test that REJECT gate decision triggers expert review."""
        pass

    def test_uncontrolled_confounder_warning(self):
        """Test warning for uncontrolled discovered confounders."""
        pass
```
