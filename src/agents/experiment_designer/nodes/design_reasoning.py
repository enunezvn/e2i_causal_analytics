"""Design Reasoning Node.

This node performs deep reasoning for experiment design strategy.
It uses an LLM (Claude Sonnet/Opus) to explore the design space and
recommend optimal experiment configurations.

Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md lines 204-415
Contract: .claude/contracts/tier3-contracts.md lines 82-142
"""

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from typing import Any

from src.agents.experiment_designer.state import (
    ExperimentDesignState,
    ErrorDetails,
    TreatmentDefinition,
    OutcomeDefinition,
)


# ===== MOCK LLM =====
# TODO: Replace with actual LangChain ChatAnthropic when API is configured
# Integration blocker documented in CONTRACT_VALIDATION.md
class MockLLM:
    """Mock LLM for testing design reasoning.

    CRITICAL: This is a temporary mock. Replace with:
        from langchain_anthropic import ChatAnthropic
        self.llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=8192)
    """

    async def ainvoke(self, prompt: str) -> "MockResponse":
        """Mock LLM invocation that returns structured design output."""
        # Simulate LLM latency
        await asyncio.sleep(0.1)

        # Return mock design response
        mock_response = {
            "refined_hypothesis": "Increasing rep visit frequency from 2x/month to 4x/month will improve HCP engagement scores by at least 15% within 8 weeks",
            "treatment_definition": {
                "name": "rep_visit_frequency",
                "description": "Number of rep visits per month to HCPs",
                "implementation_details": "Double visit frequency from baseline",
                "target_population": "High-value HCPs in target territories",
                "dosage_or_intensity": "4 visits/month vs 2 visits/month",
                "duration": "8 weeks",
                "delivery_mechanism": "Field sales force",
            },
            "outcome_definition": {
                "name": "hcp_engagement_score",
                "metric_type": "continuous",
                "measurement_method": "Composite engagement index (0-100)",
                "measurement_frequency": "Weekly",
                "baseline_value": 45.0,
                "expected_effect_size": 0.25,
                "minimum_detectable_effect": 0.15,
                "is_primary": True,
            },
            "design_type": "cluster_rct",
            "design_rationale": "Cluster RCT at territory level prevents contamination between treatment and control HCPs. Geographic clustering also accounts for regional variation in prescribing patterns.",
            "stratification_vars": ["region", "hcp_specialty", "prior_engagement_quartile"],
            "blocking_variables": ["territory_size"],
            "randomization_unit": "cluster",
            "randomization_method": "stratified_block_randomization",
            "causal_assumptions": [
                "SUTVA: No spillover between territories",
                "Ignorability: Randomization ensures treatment independence",
                "Positivity: All territories can receive either treatment level",
            ],
            "anticipated_confounders": [
                {"name": "competitive_activity", "how_addressed": "Stratification by market"},
                {"name": "seasonal_effects", "how_addressed": "Time-matched control"},
            ],
        }

        return MockResponse(json.dumps(mock_response))


class MockResponse:
    """Mock response from LLM."""

    def __init__(self, content: str):
        self.content = content


class DesignReasoningNode:
    """Deep reasoning for experiment design strategy.

    This node uses an LLM to:
    1. Refine the business hypothesis into a testable claim
    2. Explore the design space (RCT, cluster RCT, quasi-experimental)
    3. Recommend optimal design with stratification strategy
    4. Identify anticipated confounders

    Model: Claude Sonnet 4 (primary), Claude Haiku 4 (fallback)
    Performance Target: <30s for design reasoning
    """

    def __init__(self):
        """Initialize design reasoning node."""
        # TODO: Replace with actual LLM when API is configured
        self.llm = MockLLM()
        self.fallback_llm = MockLLM()
        self.model_name = "claude-sonnet-4-20250514"
        self.fallback_model_name = "claude-haiku-4-20250414"

    async def execute(self, state: ExperimentDesignState) -> ExperimentDesignState:
        """Execute design reasoning.

        Args:
            state: Current agent state with business_question and context

        Returns:
            Updated state with design outputs
        """
        start_time = time.time()

        # Skip if status is failed
        if state.get("status") == "failed":
            return state

        try:
            # Update status
            state["status"] = "reasoning"

            # Build prompt with organizational context
            prompt = self._build_design_prompt(state)

            # Invoke LLM with fallback
            try:
                response = await asyncio.wait_for(
                    self.llm.ainvoke(prompt),
                    timeout=120,
                )
                model_used = self.model_name
            except (asyncio.TimeoutError, Exception) as e:
                response = await self.fallback_llm.ainvoke(
                    self._build_simplified_prompt(state)
                )
                model_used = f"{self.fallback_model_name} (fallback)"
                state["warnings"] = state.get("warnings", []) + [
                    f"Design used fallback: {str(e)}"
                ]

            # Parse design response
            design = self._parse_design_response(response.content)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            node_latencies = state.get("node_latencies_ms", {})
            node_latencies["design_reasoning"] = latency_ms

            # Update state with design outputs
            state["design_type"] = design.get("design_type", "RCT")
            state["design_rationale"] = design.get("design_rationale", "")

            # Parse treatment definition
            if "treatment_definition" in design:
                treatment = design["treatment_definition"]
                state["treatments"] = [
                    TreatmentDefinition(
                        name=treatment.get("name", "treatment"),
                        description=treatment.get("description", ""),
                        implementation_details=treatment.get("implementation_details", ""),
                        target_population=treatment.get("target_population", ""),
                        dosage_or_intensity=treatment.get("dosage_or_intensity"),
                        duration=treatment.get("duration"),
                        delivery_mechanism=treatment.get("delivery_mechanism"),
                    )
                ]

            # Parse outcome definition
            if "outcome_definition" in design:
                outcome = design["outcome_definition"]
                state["outcomes"] = [
                    OutcomeDefinition(
                        name=outcome.get("name", "outcome"),
                        metric_type=outcome.get("metric_type", "continuous"),
                        measurement_method=outcome.get("measurement_method", ""),
                        measurement_frequency=outcome.get("measurement_frequency", ""),
                        baseline_value=outcome.get("baseline_value"),
                        expected_effect_size=outcome.get("expected_effect_size"),
                        minimum_detectable_effect=outcome.get("minimum_detectable_effect"),
                        is_primary=outcome.get("is_primary", True),
                    )
                ]

            state["randomization_unit"] = design.get("randomization_unit", "individual")
            state["randomization_method"] = design.get("randomization_method", "simple")
            state["blocking_variables"] = design.get("blocking_variables", [])
            state["stratification_variables"] = design.get("stratification_vars", [])
            state["causal_assumptions"] = design.get("causal_assumptions", [])

            # Update metadata
            state["node_latencies_ms"] = node_latencies
            state["total_llm_tokens_used"] = state.get("total_llm_tokens_used", 0) + 2000  # Estimate

            # Update status for next node
            state["status"] = "calculating"

        except Exception as e:
            error: ErrorDetails = {
                "node": "design_reasoning",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            state["errors"] = state.get("errors", []) + [error]
            state["status"] = "failed"

        return state

    def _build_design_prompt(self, state: ExperimentDesignState) -> str:
        """Build full design prompt with organizational context.

        Args:
            state: Current agent state

        Returns:
            Formatted prompt string
        """
        org_context = self._build_org_context(state)

        return f"""You are an expert in causal inference and experimental design for pharmaceutical commercial operations.

## Organizational Learning Context
{org_context}

---

## Current Design Request

**Business Question:** {state['business_question']}

**Constraints:**
{json.dumps(state.get('constraints', {}), indent=2)}

**Available Data:**
{json.dumps(state.get('available_data', {}), indent=2)}

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
    "description": "What the treatment is",
    "implementation_details": "How treatment is implemented",
    "target_population": "Who receives treatment",
    "dosage_or_intensity": "If applicable",
    "duration": "How long",
    "delivery_mechanism": "How delivered"
  }},
  "outcome_definition": {{
    "name": "primary_outcome_variable",
    "metric_type": "continuous|binary|count|time_to_event",
    "measurement_method": "How measured",
    "measurement_frequency": "When measured",
    "baseline_value": 0.0,
    "expected_effect_size": 0.0,
    "minimum_detectable_effect": 0.0,
    "is_primary": true
  }},
  "design_type": "RCT|quasi_experiment|difference_in_differences|regression_discontinuity|instrumental_variable|synthetic_control",
  "design_rationale": "2-3 sentences explaining why",
  "randomization_unit": "individual|cluster|time_period|geography",
  "randomization_method": "simple|stratified|block|stratified_block",
  "stratification_vars": ["var1", "var2"],
  "blocking_variables": ["var1"],
  "causal_assumptions": ["assumption1", "assumption2"],
  "anticipated_confounders": [
    {{"name": "confounder", "how_addressed": "How design handles it"}}
  ]
}}
```"""

    def _build_org_context(self, state: ExperimentDesignState) -> str:
        """Build organizational learning context section.

        Args:
            state: Current agent state

        Returns:
            Formatted context string
        """
        parts = []

        if state.get("historical_experiments"):
            experiments = state["historical_experiments"][:3]
            parts.append(f"### Similar Past Experiments\n{json.dumps(experiments, indent=2)}")

        if state.get("domain_knowledge"):
            domain = state["domain_knowledge"]
            if "organizational_defaults" in domain:
                defaults = domain["organizational_defaults"]
                parts.append(f"""### Organizational Defaults
- Default effect size: {defaults.get('effect_size', 0.25)}
- Default ICC for clusters: {defaults.get('icc', 0.05)}
- Standard confounders: {defaults.get('standard_confounders', [])}""")

        if state.get("warnings"):
            # Include recent assumption violations from warnings
            violations = [w for w in state["warnings"] if "Past violation" in w]
            if violations:
                parts.append(f"### Recent Assumption Violations\n" + "\n".join(f"- {v}" for v in violations[:3]))

        return "\n\n".join(parts) if parts else "No historical context available."

    def _build_simplified_prompt(self, state: ExperimentDesignState) -> str:
        """Build simplified prompt for fallback LLM.

        Args:
            state: Current agent state

        Returns:
            Simplified prompt string
        """
        return f"""Design an experiment for: {state['business_question']}

Constraints: {json.dumps(state.get('constraints', {}))}

Provide JSON with:
- refined_hypothesis
- treatment_definition (name, description, implementation_details, target_population)
- outcome_definition (name, metric_type, measurement_method, is_primary)
- design_type (RCT, quasi_experiment, difference_in_differences, regression_discontinuity)
- design_rationale
- randomization_unit
- stratification_vars
- anticipated_confounders"""

    def _parse_design_response(self, content: str) -> dict[str, Any]:
        """Extract JSON from LLM response.

        Args:
            content: Raw LLM response

        Returns:
            Parsed design dictionary
        """
        # Try to extract JSON from markdown code block
        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to extract bare JSON
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass

        # Fallback: return minimal design with raw content
        return {"design_rationale": content}
