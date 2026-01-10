"""Design Reasoning Node.

This node performs deep reasoning for experiment design strategy.
It uses an LLM (Claude Sonnet/Opus) to explore the design space and
recommend optimal experiment configurations.

Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md lines 204-415
Contract: .claude/contracts/tier3-contracts.md lines 82-142
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

from src.utils.llm_factory import get_chat_llm, get_fast_llm, get_llm_provider

from src.agents.experiment_designer.state import (
    ErrorDetails,
    ExperimentDesignState,
    OutcomeDefinition,
    TreatmentDefinition,
)

logger = logging.getLogger(__name__)


def _get_opik_connector():
    """Lazy import of OpikConnector to avoid circular imports."""
    try:
        from src.mlops.opik_connector import get_opik_connector

        return get_opik_connector()
    except ImportError:
        logger.debug("OpikConnector not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to get OpikConnector: {e}")
        return None


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
        """Initialize design reasoning node with LLM factory."""
        self._provider = get_llm_provider()
        # Set model names based on provider for tracing
        if self._provider == "openai":
            self.model_name = "gpt-4o"
            self.fallback_model_name = "gpt-4o-mini"
        else:
            self.model_name = "claude-sonnet-4-20250514"
            self.fallback_model_name = "claude-haiku-4-20250414"
        # Use reasoning tier for primary, fast tier for fallback
        self.llm = get_chat_llm(model_tier="reasoning", max_tokens=8192, timeout=120)
        self.fallback_llm = get_fast_llm(max_tokens=4096, timeout=60)

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

            # Get OpikConnector for LLM call tracing
            opik = _get_opik_connector()

            # Invoke LLM with fallback
            try:
                if opik and opik.is_enabled:
                    # Trace the LLM call with dynamic provider info
                    async with opik.trace_llm_call(
                        model=self.model_name,
                        provider=self._provider,
                        prompt_template="experiment_design_reasoning",
                        input_data={"prompt": prompt[:500], "business_question": state.get("business_question", "")},
                        metadata={"agent": "experiment_designer", "operation": "design_reasoning"},
                    ) as llm_span:
                        response = await asyncio.wait_for(
                            self.llm.ainvoke(prompt),
                            timeout=120,
                        )
                        # Log tokens from response metadata
                        usage = response.response_metadata.get("usage", {})
                        llm_span.log_tokens(
                            input_tokens=usage.get("input_tokens", 0),
                            output_tokens=usage.get("output_tokens", 0),
                        )
                else:
                    # Fallback: no tracing
                    response = await asyncio.wait_for(
                        self.llm.ainvoke(prompt),
                        timeout=120,
                    )
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Primary LLM failed, using fallback: {e}")
                # Use fallback LLM with tracing
                fallback_prompt = self._build_simplified_prompt(state)
                if opik and opik.is_enabled:
                    async with opik.trace_llm_call(
                        model=self.fallback_model_name,
                        provider=self._provider,
                        prompt_template="experiment_design_reasoning_fallback",
                        input_data={"prompt": fallback_prompt[:500]},
                        metadata={"agent": "experiment_designer", "operation": "design_reasoning_fallback"},
                    ) as llm_span:
                        response = await self.fallback_llm.ainvoke(fallback_prompt)
                        usage = response.response_metadata.get("usage", {})
                        llm_span.log_tokens(
                            input_tokens=usage.get("input_tokens", 0),
                            output_tokens=usage.get("output_tokens", 0),
                        )
                else:
                    response = await self.fallback_llm.ainvoke(fallback_prompt)
                state["warnings"] = state.get("warnings", []) + [f"Design used fallback: {str(e)}"]

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
            # Get actual token usage from response
            usage = response.response_metadata.get("usage", {})
            actual_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            state["total_llm_tokens_used"] = state.get("total_llm_tokens_used", 0) + actual_tokens

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
                parts.append(
                    f"""### Organizational Defaults
- Default effect size: {defaults.get('effect_size', 0.25)}
- Default ICC for clusters: {defaults.get('icc', 0.05)}
- Standard confounders: {defaults.get('standard_confounders', [])}"""
                )

        if state.get("warnings"):
            # Include recent assumption violations from warnings
            violations = [w for w in state["warnings"] if "Past violation" in w]
            if violations:
                parts.append(
                    "### Recent Assumption Violations\n"
                    + "\n".join(f"- {v}" for v in violations[:3])
                )

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
