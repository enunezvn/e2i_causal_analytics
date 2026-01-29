"""Validity Audit Node.

This node performs adversarial validity assessment of the experiment design.
It uses an LLM to red-team the proposed design and identify potential threats
to internal and external validity.

Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md lines 552-706
Contract: .claude/contracts/tier3-contracts.md lines 82-142

V4.4: Added DAG-aware validity validation.
V4.5: Added LangChain ChatAnthropic integration with graceful fallback.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, runtime_checkable

from src.agents.experiment_designer.state import (
    ErrorDetails,
    ExperimentDesignState,
    MitigationRecommendation,
    ValidityThreat,
)

logger = logging.getLogger(__name__)


# ===== LLM INTERFACE =====
@runtime_checkable
class LLMInterface(Protocol):
    """Protocol for LLM implementations."""

    async def ainvoke(self, prompt: str) -> Any:
        """Async invocation of LLM."""
        ...


def _get_validity_llm() -> tuple[Any, str, bool]:
    """Get LLM for validity audit.

    Attempts to use ChatAnthropic if available and configured,
    otherwise falls back to MockValidityLLM.

    Returns:
        Tuple of (llm_instance, model_name, is_real_llm)
    """
    # Check if API key is configured
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.info("ANTHROPIC_API_KEY not set, using mock LLM for validity audit")
        return MockValidityLLM(), "mock-validity-llm", False

    try:
        from langchain_anthropic import ChatAnthropic

        # Use Claude Sonnet 4 for validity assessment
        model_name = os.environ.get(
            "VALIDITY_AUDIT_MODEL", "claude-sonnet-4-20250514"
        )
        llm = ChatAnthropic(
            model=model_name,
            max_tokens=4096,
            temperature=0.3,  # Lower temperature for structured analysis
            api_key=api_key,
        )
        logger.info(f"Using ChatAnthropic ({model_name}) for validity audit")
        return llm, model_name, True

    except ImportError:
        logger.warning(
            "langchain_anthropic not installed, using mock LLM. "
            "Install with: pip install langchain-anthropic"
        )
        return MockValidityLLM(), "mock-validity-llm", False

    except Exception as e:
        logger.warning(f"Failed to initialize ChatAnthropic: {e}, using mock LLM")
        return MockValidityLLM(), "mock-validity-llm", False


class MockValidityLLM:
    """Mock LLM for testing validity audit.

    Used as fallback when ChatAnthropic is not available (missing API key or
    langchain-anthropic not installed). Returns realistic mock responses for
    testing and development.
    """

    async def ainvoke(self, prompt: str) -> "MockValidityResponse":
        """Mock LLM invocation that returns structured validity audit."""
        await asyncio.sleep(0.1)

        mock_response = {
            "internal_validity_threats": [
                {
                    "threat_type": "internal",
                    "threat_name": "selection_bias",
                    "description": "Non-random territory assignment may introduce systematic differences",
                    "severity": "medium",
                    "affected_outcomes": ["hcp_engagement_score"],
                    "mitigation_possible": True,
                    "mitigation_strategy": "Use stratified randomization by territory characteristics",
                },
                {
                    "threat_type": "internal",
                    "threat_name": "contamination",
                    "description": "HCPs in treatment territories may share information with control territories",
                    "severity": "low",
                    "affected_outcomes": ["hcp_engagement_score"],
                    "mitigation_possible": True,
                    "mitigation_strategy": "Ensure geographic separation between treatment and control",
                },
            ],
            "external_validity_limits": [
                "Results may not generalize to different therapeutic areas",
                "Seasonal effects may limit generalizability to other time periods",
            ],
            "statistical_concerns": [
                "ICC assumption of 0.05 may be optimistic for territory-level clustering",
            ],
            "mitigation_recommendations": [
                {
                    "threat_addressed": "selection_bias",
                    "strategy": "Implement covariate balance checks post-randomization",
                    "implementation_steps": [
                        "Calculate baseline covariate balance statistics",
                        "Re-randomize if balance criteria not met",
                        "Document final covariate distributions",
                    ],
                    "effectiveness_rating": "high",
                    "trade_offs": ["May require multiple randomization attempts"],
                },
            ],
            "overall_validity_score": 0.75,
            "validity_confidence": "medium",
            "proceed_recommendation": "proceed_with_caution",
            "redesign_needed": False,
            "redesign_recommendations": [],
        }

        return MockValidityResponse(json.dumps(mock_response))


class MockValidityResponse:
    """Mock response from LLM."""

    def __init__(self, content: str):
        self.content = content


class ValidityAuditNode:
    """Adversarial validity assessment for experiment design.

    This node uses an LLM to:
    1. Identify internal validity threats (selection, confounding, measurement, etc.)
    2. Assess external validity limitations
    3. Flag statistical concerns
    4. Recommend mitigations
    5. Determine if redesign is needed

    Model: Claude Sonnet 4 (primary), with graceful fallback to mock
    Performance Target: <30s for validity audit
    """

    def __init__(self):
        """Initialize validity audit node."""
        self.llm, self.model_name, self._using_real_llm = _get_validity_llm()

    async def execute(self, state: ExperimentDesignState) -> ExperimentDesignState:
        """Execute validity audit.

        Args:
            state: Current agent state with design and power analysis outputs

        Returns:
            Updated state with validity audit results
        """
        start_time = time.time()

        # Skip if status is failed
        if state.get("status") == "failed":
            return state

        # Skip if validity audit is disabled
        if not state.get("enable_validity_audit", True):
            state["warnings"] = state.get("warnings", []) + ["Validity audit skipped (disabled)"]
            state["validity_confidence"] = "low"
            state["redesign_needed"] = False
            state["status"] = "generating"
            return state

        try:
            # Update status
            state["status"] = "auditing"

            # Build audit prompt
            prompt = self._build_audit_prompt(state)

            # Invoke LLM with timeout
            try:
                response = await asyncio.wait_for(
                    self.llm.ainvoke(prompt),
                    timeout=90,
                )
            except asyncio.TimeoutError:
                state["warnings"] = state.get("warnings", []) + ["Validity audit timed out"]
                state["validity_confidence"] = "low"
                state["redesign_needed"] = False
                state["status"] = "generating"
                return state

            # Parse audit response
            audit = self._parse_audit_response(response.content)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            node_latencies = state.get("node_latencies_ms", {})
            node_latencies["validity_audit"] = latency_ms

            # Parse validity threats
            threats: list[ValidityThreat] = []
            for threat_data in audit.get("internal_validity_threats", []):
                threat = ValidityThreat(
                    threat_type=threat_data.get("threat_type", "internal"),
                    threat_name=threat_data.get("threat_name", "unknown"),
                    description=threat_data.get("description", ""),
                    severity=threat_data.get("severity", "medium"),
                    affected_outcomes=threat_data.get("affected_outcomes", []),
                    mitigation_possible=threat_data.get("mitigation_possible", True),
                    mitigation_strategy=threat_data.get("mitigation_strategy"),
                )
                threats.append(threat)

            # Parse mitigations
            mitigations: list[MitigationRecommendation] = []
            for mit_data in audit.get("mitigation_recommendations", []):
                mitigation = MitigationRecommendation(
                    threat_addressed=mit_data.get("threat_addressed", ""),
                    strategy=mit_data.get("strategy", ""),
                    implementation_steps=mit_data.get("implementation_steps", []),
                    cost_estimate=mit_data.get("cost_estimate"),
                    effectiveness_rating=mit_data.get("effectiveness_rating", "medium"),
                    trade_offs=mit_data.get("trade_offs", []),
                )
                mitigations.append(mitigation)

            # Update state with audit results
            state["validity_threats"] = threats
            state["mitigations"] = mitigations
            state["overall_validity_score"] = audit.get("overall_validity_score", 0.5)
            state["validity_confidence"] = audit.get("validity_confidence", "medium")
            state["redesign_needed"] = audit.get("redesign_needed", False)
            state["redesign_recommendations"] = audit.get("redesign_recommendations", [])

            # V4.4: DAG-aware validity validation
            if self._has_dag_evidence(state):
                dag_results, dag_warnings = self._perform_dag_validation(state)

                # Store DAG validation results in state
                state["dag_confounders_validated"] = dag_results.get("confounders_validated", [])
                state["dag_missing_confounders"] = dag_results.get("confounders_missing", [])
                state["dag_latent_confounders"] = dag_results.get("latent_confounders", [])
                state["dag_instrument_candidates"] = dag_results.get("instrument_candidates", [])
                state["dag_effect_modifiers"] = dag_results.get("effect_modifiers", [])
                state["dag_validation_warnings"] = dag_warnings

                # Add DAG warnings to overall warnings
                state["warnings"] = state.get("warnings", []) + dag_warnings

                # If missing confounders or latent confounders detected, flag for review
                if dag_results.get("confounders_missing") or dag_results.get("latent_confounders"):
                    # Adjust validity score down by 10% for each concern
                    penalty = 0.1 * (
                        (1 if dag_results.get("confounders_missing") else 0)
                        + (1 if dag_results.get("latent_confounders") else 0)
                    )
                    state["overall_validity_score"] = max(
                        0.0, state.get("overall_validity_score", 0.5) - penalty
                    )

            # Update metadata
            state["node_latencies_ms"] = node_latencies
            state["total_llm_tokens_used"] = state.get("total_llm_tokens_used", 0) + 1500

            # Determine next status based on redesign decision
            if state["redesign_needed"]:
                current_iteration = state.get("current_iteration", 0)
                max_iterations = state.get("max_redesign_iterations", 2)

                if current_iteration < max_iterations:
                    state["status"] = "redesigning"
                else:
                    state["warnings"] = state.get("warnings", []) + [
                        f"Max redesign iterations ({max_iterations}) reached. Proceeding with current design."
                    ]
                    state["status"] = "generating"
            else:
                state["status"] = "generating"

        except Exception as e:
            error: ErrorDetails = {
                "node": "validity_audit",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "recoverable": True,
            }
            state["errors"] = state.get("errors", []) + [error]
            state["warnings"] = state.get("warnings", []) + [f"Validity audit failed: {str(e)}"]
            # Set required output defaults on failure
            state["validity_threats"] = state.get("validity_threats", [])
            state["overall_validity_score"] = state.get("overall_validity_score", 0.0)
            state["validity_confidence"] = "low"
            state["redesign_needed"] = False
            state["status"] = "generating"

        return state

    def _build_audit_prompt(self, state: ExperimentDesignState) -> str:
        """Build adversarial audit prompt.

        Args:
            state: Current agent state

        Returns:
            Formatted prompt string
        """
        # Extract design details
        treatments = state.get("treatments", [])
        treatment_json = json.dumps([dict(t) for t in treatments], indent=2) if treatments else "{}"

        outcomes = state.get("outcomes", [])
        outcome_json = json.dumps([dict(o) for o in outcomes], indent=2) if outcomes else "{}"

        power_analysis = state.get("power_analysis", {})

        return f"""You are a methodological critic reviewing an experiment design. Your job is to find weaknesses.

## Proposed Experiment

**Design Type:** {state.get('design_type', 'Not specified')}
**Design Rationale:** {state.get('design_rationale', 'Not specified')}

**Treatment:**
{treatment_json}

**Outcome:**
{outcome_json}

**Sample Size:** {power_analysis.get('required_sample_size', 'Not calculated')}
**Randomization Unit:** {state.get('randomization_unit', 'individual')}
**Randomization Method:** {state.get('randomization_method', 'simple')}
**Stratification:** {state.get('stratification_variables', [])}
**Blocking Variables:** {state.get('blocking_variables', [])}
**Causal Assumptions:** {json.dumps(state.get('causal_assumptions', []), indent=2)}

---

## Audit Checklist

### Internal Validity Threats
For each threat, assess severity (low/medium/high/critical) and mitigation:

1. **Selection Bias** - Is randomization truly random? Any systematic differences?
2. **Confounding** - What confounders might be MISSED by the design?
3. **Measurement** - Could outcome measurement differ between arms?
4. **Contamination/Spillover** - Could control be exposed to treatment?
5. **Temporal** - History, maturation, regression to mean?
6. **Attrition** - Differential dropout expected?

### External Validity
- What populations does this generalize to?
- What contexts would NOT transfer?

### Statistical Concerns
- Is power analysis realistic?
- Multiple comparison issues?
- Assumption violations?

---

## Output (Must be valid JSON)

```json
{{
  "internal_validity_threats": [
    {{
      "threat_type": "internal",
      "threat_name": "selection_bias|confounding|measurement|contamination|temporal|attrition",
      "description": "Specific concern",
      "severity": "low|medium|high|critical",
      "affected_outcomes": ["outcome1"],
      "mitigation_possible": true,
      "mitigation_strategy": "How to address"
    }}
  ],
  "external_validity_limits": ["Limit 1", "Limit 2"],
  "statistical_concerns": ["Concern 1", "Concern 2"],
  "mitigation_recommendations": [
    {{
      "threat_addressed": "Which threat",
      "strategy": "What to do",
      "implementation_steps": ["Step 1", "Step 2"],
      "effectiveness_rating": "low|medium|high",
      "trade_offs": ["Trade-off 1"]
    }}
  ],
  "overall_validity_score": 0.75,
  "validity_confidence": "low|medium|high",
  "redesign_needed": false,
  "redesign_recommendations": [],
  "proceed_recommendation": "proceed|proceed_with_caution|redesign_needed"
}}
```"""

    def _parse_audit_response(self, content: str) -> dict[str, Any]:
        """Parse audit JSON from response.

        Args:
            content: Raw LLM response

        Returns:
            Parsed audit dictionary
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

        # Fallback: return default audit
        return {
            "overall_validity_score": 0.5,
            "validity_confidence": "low",
            "redesign_needed": False,
            "proceed_recommendation": "proceed_with_caution",
            "internal_validity_threats": [],
            "external_validity_limits": ["Unable to fully assess"],
            "mitigation_recommendations": [],
        }

    # =========================================================================
    # V4.4: DAG-Aware Validity Validation
    # =========================================================================

    def _has_dag_evidence(self, state: ExperimentDesignState) -> bool:
        """Check if DAG evidence is available for validation.

        Args:
            state: Current experiment design state

        Returns:
            True if DAG evidence is available and valid
        """
        dag_adjacency = state.get("discovered_dag_adjacency")
        dag_nodes = state.get("discovered_dag_nodes")
        discovery_gate_decision = state.get("discovery_gate_decision")

        # DAG evidence is available if:
        # 1. We have DAG adjacency matrix and nodes
        # 2. Discovery gate decision is accept or review (not reject)
        return (
            dag_adjacency is not None
            and dag_nodes is not None
            and len(dag_adjacency) > 0
            and len(dag_nodes) > 0
            and discovery_gate_decision in ("accept", "review")
        )

    def _validate_confounders_against_dag(
        self, state: ExperimentDesignState
    ) -> tuple[list[str], list[str], list[str]]:
        """Validate assumed confounders against discovered DAG.

        V4.4: Check which confounders from causal_assumptions are actually
        in the DAG, and identify any that are missing (not discovered).

        Args:
            state: Current experiment design state

        Returns:
            Tuple of (validated_confounders, missing_confounders, warnings)
        """
        validated: list[str] = []
        missing: list[str] = []
        warnings: list[str] = []

        dag_nodes = state.get("discovered_dag_nodes", [])
        dag_nodes_set = set(dag_nodes)
        causal_assumptions = state.get("causal_assumptions", [])

        # Extract potential confounder variables from assumptions
        # Assumptions often have format like "Controlled for: specialty, region, ..."
        potential_confounders: list[str] = []
        for assumption in causal_assumptions:
            assumption_lower = assumption.lower()
            if "control" in assumption_lower or "adjust" in assumption_lower:
                # Extract variable names - simple heuristic
                parts = assumption.split(":")
                if len(parts) > 1:
                    vars_part = parts[-1]
                    # Split by common separators
                    for sep in [",", "and", ";", "/"]:
                        vars_part = vars_part.replace(sep, ",")
                    potential_confounders.extend(
                        [v.strip().lower() for v in vars_part.split(",") if v.strip()]
                    )

        # Also check common_causes from dowhy_spec if available
        dowhy_spec = state.get("dowhy_spec")
        if dowhy_spec:
            common_causes = dowhy_spec.get("common_causes", [])
            potential_confounders.extend([c.lower() for c in common_causes])

        # Deduplicate
        potential_confounders = list(set(potential_confounders))

        # Check each confounder against DAG
        dag_nodes_lower = {n.lower(): n for n in dag_nodes}
        for confounder in potential_confounders:
            if confounder in dag_nodes_lower:
                validated.append(dag_nodes_lower[confounder])
            else:
                missing.append(confounder)
                warnings.append(
                    f"Assumed confounder '{confounder}' was NOT discovered in causal DAG. "
                    f"Effect may be spurious or confounder may not be causally relevant."
                )

        return validated, missing, warnings

    def _identify_latent_confounders(self, state: ExperimentDesignState) -> list[str]:
        """Identify latent confounders from FCI bidirected edges.

        V4.4: FCI algorithm detects latent confounders as bidirected edges (↔).
        These indicate unobserved common causes.

        Args:
            state: Current experiment design state

        Returns:
            List of variable pairs with latent confounders
        """
        latent_confounders: list[str] = []
        edge_types = state.get("discovered_dag_edge_types", {})

        for edge_key, edge_type in edge_types.items():
            if edge_type == "BIDIRECTED":
                latent_confounders.append(edge_key)

        return latent_confounders

    def _identify_instrument_candidates(self, state: ExperimentDesignState) -> list[str]:
        """Identify valid instrumental variable candidates from DAG.

        V4.4: A valid IV must:
        1. Have a path to the treatment
        2. NOT have a direct path to the outcome (except through treatment)
        3. NOT share a common cause with the outcome

        Args:
            state: Current experiment design state

        Returns:
            List of potential IV candidates
        """
        candidates: list[str] = []

        dag_adjacency = state.get("discovered_dag_adjacency", [])
        dag_nodes = state.get("discovered_dag_nodes", [])
        treatment_var = state.get("treatment_variable", "")
        outcome_var = state.get("outcome_variable", "")

        if not dag_adjacency or not dag_nodes or not treatment_var or not outcome_var:
            return candidates

        node_to_idx = {node: idx for idx, node in enumerate(dag_nodes)}
        n_nodes = len(dag_nodes)

        treatment_idx = node_to_idx.get(treatment_var)
        outcome_idx = node_to_idx.get(outcome_var)

        if treatment_idx is None or outcome_idx is None:
            return candidates

        # For each node, check IV criteria
        for node_idx, node in enumerate(dag_nodes):
            if node in (treatment_var, outcome_var):
                continue

            # Check 1: Has edge to treatment
            has_edge_to_treatment = dag_adjacency[node_idx][treatment_idx] == 1

            # Check 2: No direct edge to outcome
            has_edge_to_outcome = dag_adjacency[node_idx][outcome_idx] == 1

            # For simplicity, check basic criteria (full IV validation is complex)
            if has_edge_to_treatment and not has_edge_to_outcome:
                candidates.append(node)

        return candidates

    def _identify_effect_modifiers(self, state: ExperimentDesignState) -> list[str]:
        """Identify effect modifiers from DAG structure.

        V4.4: Effect modifiers are variables that may moderate the treatment effect.
        In the DAG, these are variables that are:
        1. Connected to both treatment and outcome
        2. Not on the causal path from treatment to outcome

        Args:
            state: Current experiment design state

        Returns:
            List of potential effect modifiers
        """
        modifiers: list[str] = []

        dag_adjacency = state.get("discovered_dag_adjacency", [])
        dag_nodes = state.get("discovered_dag_nodes", [])
        treatment_var = state.get("treatment_variable", "")
        outcome_var = state.get("outcome_variable", "")

        if not dag_adjacency or not dag_nodes or not treatment_var or not outcome_var:
            return modifiers

        node_to_idx = {node: idx for idx, node in enumerate(dag_nodes)}
        n_nodes = len(dag_nodes)

        treatment_idx = node_to_idx.get(treatment_var)
        outcome_idx = node_to_idx.get(outcome_var)

        if treatment_idx is None or outcome_idx is None:
            return modifiers

        # For each node, check if it could be an effect modifier
        for node_idx, node in enumerate(dag_nodes):
            if node in (treatment_var, outcome_var):
                continue

            # Check if connected to treatment (as cause of treatment OR common cause)
            connected_to_treatment = (
                dag_adjacency[node_idx][treatment_idx] == 1  # Node -> Treatment
                or dag_adjacency[treatment_idx][node_idx] == 1  # Treatment -> Node
            )

            # Check if connected to outcome (not on causal path, but as common cause)
            # Common causes point TO both treatment and outcome
            is_common_cause = (
                dag_adjacency[node_idx][treatment_idx] == 1
                and dag_adjacency[node_idx][outcome_idx] == 1
            )

            if is_common_cause:
                modifiers.append(node)

        return modifiers

    def _perform_dag_validation(
        self, state: ExperimentDesignState
    ) -> tuple[dict[str, Any], list[str]]:
        """Perform comprehensive DAG-based validity validation.

        V4.4: Main entry point for DAG-aware validation.

        Args:
            state: Current experiment design state

        Returns:
            Tuple of (dag_validation_results, warnings)
        """
        results: dict[str, Any] = {}
        all_warnings: list[str] = []

        # Validate confounders
        validated, missing, confounder_warnings = self._validate_confounders_against_dag(state)
        results["confounders_validated"] = validated
        results["confounders_missing"] = missing
        all_warnings.extend(confounder_warnings)

        # Identify latent confounders
        latent = self._identify_latent_confounders(state)
        results["latent_confounders"] = latent
        if latent:
            all_warnings.append(
                f"DAG reveals {len(latent)} latent confounder(s): {', '.join(latent)}. "
                "Consider sensitivity analysis or finding proxies for unmeasured confounders."
            )

        # Identify IV candidates
        iv_candidates = self._identify_instrument_candidates(state)
        results["instrument_candidates"] = iv_candidates
        if iv_candidates:
            all_warnings.append(
                f"DAG suggests potential instrumental variables: {', '.join(iv_candidates)}. "
                "Consider IV design if RCT is not feasible."
            )

        # Identify effect modifiers
        effect_modifiers = self._identify_effect_modifiers(state)
        results["effect_modifiers"] = effect_modifiers
        if effect_modifiers:
            all_warnings.append(
                f"DAG identifies potential effect modifiers: {', '.join(effect_modifiers)}. "
                "Consider stratification or interaction analysis."
            )

        return results, all_warnings
