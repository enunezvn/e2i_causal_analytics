"""Validity Audit Node.

This node performs adversarial validity assessment of the experiment design.
It uses an LLM to red-team the proposed design and identify potential threats
to internal and external validity.

Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md lines 552-706
Contract: .claude/contracts/tier3-contracts.md lines 82-142

V4.4: Added DAG-aware validity validation.
V4.5: Added LangChain ChatAnthropic integration with graceful fallback.
V4.6 (#471): REWIRE — silent MockValidityLLM fallback was a CLAUDE.md
    HARMFUL-NOW anti-mocking violation (prod LangGraph node returning
    plausible-real validity scores when ANTHROPIC_API_KEY missing). The
    fallback now raises ``RuntimeError`` with a diagnostic that
    distinguishes <unset> / <empty-string> / <set,len=N> + reports
    ``.env`` existence; the dev-mode mock path is preserved behind the
    explicit ``EXPERIMENT_DESIGNER_USE_MOCK_LLM=1`` opt-in flag, and the
    mock now emits an in-band ``mock_response_for_dev_only=True`` marker
    so downstream consumers / log readers can distinguish synthetic
    audits from real LLM output.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Protocol, cast, runtime_checkable

from pydantic import SecretStr

from src.agents.experiment_designer.state import (
    ErrorDetails,
    ExperimentDesignState,
    MitigationRecommendation,
    ValidityThreat,
)
from src.utils.env_diagnostics import env_state
from src.utils.llm_content import normalize_llm_content
from src.utils.project_root import find_project_root

logger = logging.getLogger(__name__)

# Opt-in flag for the dev-mode mock path. Set to "1" (or any truthy
# string per ``_flag_enabled``) to allow ``_get_validity_llm()`` to
# return ``MockValidityLLM`` when ``ANTHROPIC_API_KEY`` is missing.
# Without this flag, missing-key raises explicitly to prevent the
# silent-mock anti-pattern (#471).
_MOCK_FLAG_ENV_VAR = "EXPERIMENT_DESIGNER_USE_MOCK_LLM"


def _flag_enabled(var: str) -> bool:
    """Treat ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive) as opt-in.

    Anything else (including ``0``, empty, ``false``) leaves the flag
    OFF. This matches the convention used elsewhere in the codebase
    (see ``ADAPTIVE_VALIDITY_EVALUATOR_ENABLED``).
    """
    raw = os.environ.get(var, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


# ===== LLM INTERFACE =====
@runtime_checkable
class LLMInterface(Protocol):
    """Protocol for LLM implementations."""

    async def ainvoke(self, prompt: str) -> Any:
        """Async invocation of LLM."""
        ...


def _build_missing_key_error(reason: str) -> RuntimeError:
    """Construct the standardized ``RuntimeError`` for missing-key paths.

    Centralized so every raise site emits the same diagnostic envelope:
    actual env state of ``ANTHROPIC_API_KEY`` (distinguishes <unset> /
    <empty-string> / <set,len=N>), existence of the project ``.env``
    file (load-chain ambiguity), and the explicit opt-in escape hatch
    for developers who actually want the mock.
    """
    try:
        dotenv_path: Path | None = find_project_root() / ".env"
    except Exception:  # pragma: no cover - defensive
        dotenv_path = None
    diag = env_state("ANTHROPIC_API_KEY", dotenv_path=dotenv_path)
    return RuntimeError(
        f"validity_audit: cannot initialize LLM — {reason}. "
        f"Diagnostic: {diag}. "
        "If your .env contains ANTHROPIC_API_KEY, ensure load_dotenv() ran "
        "before this module was imported (see #470/#471 audit). "
        f"For development without an API key, set {_MOCK_FLAG_ENV_VAR}=1 "
        "explicitly to opt into MockValidityLLM (returns clearly-fake values "
        "marked with 'mock_response_for_dev_only=True')."
    )


def _get_validity_llm() -> tuple[Any, str, bool]:
    """Get LLM for validity audit.

    Returns a real ``ChatAnthropic`` instance when ``ANTHROPIC_API_KEY``
    is set, the dev-mode ``MockValidityLLM`` when the opt-in flag
    ``EXPERIMENT_DESIGNER_USE_MOCK_LLM=1`` is set, and raises
    ``RuntimeError`` with a precise diagnostic otherwise.

    The previous behavior (silent mock fallback on missing key, missing
    install, or any init error) was a CLAUDE.md anti-mocking
    HARMFUL-NOW violation because the mock returns structured
    ``ValidityFinding`` records that look identical to real audit
    output, and the production LangGraph node ``ValidityAuditNode`` is
    not gated by any feature flag. See issue #471.

    Returns:
        Tuple of (llm_instance, model_name, is_real_llm). When the
        opt-in flag is set and the key is also set, the real LLM still
        wins — the flag only governs missing-key behavior.

    Raises:
        RuntimeError: When ``ANTHROPIC_API_KEY`` is unset / empty AND
            the opt-in flag is not set, OR when an explicit attempt to
            initialize ``ChatAnthropic`` fails (import-missing,
            misconfig, etc.) without the opt-in flag.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    use_mock = _flag_enabled(_MOCK_FLAG_ENV_VAR)

    if not api_key:
        if use_mock:
            logger.info(
                "validity_audit: %s=1 opt-in honored; using MockValidityLLM "
                "(returns clearly-fake values marked dev-only). %s",
                _MOCK_FLAG_ENV_VAR,
                env_state("ANTHROPIC_API_KEY"),
            )
            return MockValidityLLM(), "mock-validity-llm", False
        raise _build_missing_key_error("ANTHROPIC_API_KEY missing or empty")

    try:
        from langchain_anthropic import ChatAnthropic

        # Use Claude Sonnet 4 for validity assessment
        model_name = os.environ.get("VALIDITY_AUDIT_MODEL", "claude-sonnet-4-6")
        llm = ChatAnthropic(  # type: ignore[call-arg]
            model=model_name,
            max_tokens=4096,
            temperature=0.3,  # Lower temperature for structured analysis
            api_key=SecretStr(api_key),
        )
        logger.info(f"Using ChatAnthropic ({model_name}) for validity audit")
        return llm, model_name, True

    except ImportError:
        if use_mock:
            logger.warning(
                "validity_audit: langchain_anthropic not installed; "
                "%s=1 opt-in honored — falling back to MockValidityLLM. "
                "Install with: pip install langchain-anthropic",
                _MOCK_FLAG_ENV_VAR,
            )
            return MockValidityLLM(), "mock-validity-llm", False
        raise _build_missing_key_error(
            "langchain_anthropic is not installed (pip install langchain-anthropic)"
        )

    except Exception as e:
        if use_mock:
            logger.warning(
                "validity_audit: ChatAnthropic init failed (%s); "
                "%s=1 opt-in honored — falling back to MockValidityLLM.",
                e,
                _MOCK_FLAG_ENV_VAR,
            )
            return MockValidityLLM(), "mock-validity-llm", False
        # Preserve original cause for debugging.
        raise _build_missing_key_error(f"ChatAnthropic initialization failed: {e}") from e


class MockValidityLLM:
    """Mock LLM for testing validity audit.

    Used as an OPT-IN dev-mode fallback when ``ChatAnthropic`` is not
    available (missing API key or ``langchain-anthropic`` not
    installed) AND the operator has explicitly set
    ``EXPERIMENT_DESIGNER_USE_MOCK_LLM=1`` to acknowledge the synthetic
    output (#471). Returns a structured response that mirrors the real
    LLM's schema so downstream parsing exercises the full code path,
    but carries an in-band ``mock_response_for_dev_only=True`` marker
    so consumers (and humans inspecting logs / debug dumps) can
    distinguish synthetic audits from real LLM output without
    re-reading the env state.

    The numeric fields are intentionally NOT plausible-real (the
    ``overall_validity_score`` is kept at the historical 0.75 only for
    parser-compat with existing tests; the dev-only marker is the
    primary distinguisher per CLAUDE.md anti-mocking discipline).
    """

    async def ainvoke(self, prompt: str) -> "MockValidityResponse":
        """Mock LLM invocation that returns structured validity audit."""
        await asyncio.sleep(0.1)

        mock_response = {
            # #471: In-band dev-only marker — primary distinguisher per
            # CLAUDE.md anti-mocking discipline. Downstream consumers
            # (and operator log-readers) can grep this string to
            # discover synthetic audits without re-reading env state.
            "mock_response_for_dev_only": True,
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


def _retract_stale_verdict(state: ExperimentDesignState) -> None:
    """Drop a previous iteration's audit findings (#1639).

    The redesign loop re-runs this node against a CHANGED design. When the new
    audit does not complete, iteration N's threats, mitigations, score and
    recommendations describe a design that no longer exists -- and
    ``_create_output`` would publish them beside a "timed_out"/"skipped" status
    as though they applied. Same class as the feasibility verdict in
    power_analysis; a no-op on the first pass, where there is nothing to
    retract.
    """
    state["validity_threats"] = []
    state["mitigations"] = []
    state["overall_validity_score"] = 0.0
    state["redesign_recommendations"] = []

    # The structured verdict is not all the audit wrote. A completed pass also
    # appends its DAG findings to the shared `warnings` list ("Assumed
    # confounder X was NOT discovered in causal DAG"), and those sentences are
    # what the USER reads. Retracting the numbers while leaving the prose is
    # the same half-retraction this function exists to prevent.
    #
    # `dag_validation_warnings` is the exact record of what this node
    # contributed, so the withdrawal is precise -- warnings from other nodes
    # are none of its business and stay.
    _withdraw_previous_dag_warnings(state)
    state["dag_validation_warnings"] = []
    state["dag_missing_confounders"] = []
    state["dag_latent_confounders"] = []
    state["dag_instrument_candidates"] = []
    state["dag_effect_modifiers"] = []


def _withdraw_previous_dag_warnings(state: ExperimentDesignState) -> None:
    """Remove the PREVIOUS audit's DAG prose from the shared warnings list.

    Called on every exit that replaces a verdict -- including a COMPLETED
    rerun, which codex iter-14 caught: iteration 0 appends "Assumed confounder
    region ...", iteration 1 completes with no DAG warning, and the overwrite
    of ``dag_validation_warnings`` destroyed the only record that could ever
    have withdrawn it. The stale sentence then outlived every mechanism built
    to retract it.
    """
    withdrawn = list(state.get("dag_validation_warnings") or [])
    if withdrawn:
        remaining = list(state.get("warnings") or [])
        for message in withdrawn:
            # The LAST occurrence, not the first. `list.remove` takes the
            # first, so an identical sentence already present from another
            # node would be the one deleted -- withdrawing someone else's
            # warning and leaving the stale audit copy in place, the exact
            # inversion of the intent. The audit appends its own copy, so its
            # contribution is the most recent one.
            for index in range(len(remaining) - 1, -1, -1):
                if remaining[index] == message:
                    del remaining[index]
                    break
        state["warnings"] = remaining


class ValidityAuditNode:
    """Adversarial validity assessment for experiment design.

    This node uses an LLM to:
    1. Identify internal validity threats (selection, confounding, measurement, etc.)
    2. Assess external validity limitations
    3. Flag statistical concerns
    4. Recommend mitigations
    5. Determine if redesign is needed

    Model: Claude Sonnet 4 (primary). Fail-closed when ANTHROPIC_API_KEY
    is missing — raises RuntimeError with diagnostic. Dev-mode mock is
    available behind explicit ``EXPERIMENT_DESIGNER_USE_MOCK_LLM=1``
    opt-in flag (#471 anti-mocking REWIRE — pre-fix this was a silent
    fallback returning plausible-real validity scores).
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
            # #1639: an audit that never ran leaves validity_threats=[] and
            # overall_validity_score=0.0 -- byte-identical to an audit that ran
            # and found nothing. Downstream (the pre-registration document) then
            # states "None identified" as fact. Record the reason so a consumer
            # can tell a clean bill of health from an absent one.
            state["validity_audit_status"] = "skipped"
            _retract_stale_verdict(state)
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
                state["validity_audit_status"] = "timed_out"
                _retract_stale_verdict(state)
                state["validity_confidence"] = "low"
                state["redesign_needed"] = False
                state["status"] = "generating"
                return state

            # Parse audit response (AIMessage.content is str | list of blocks, #1358)
            audit = self._parse_audit_response(normalize_llm_content(response.content))

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
            state["validity_audit_status"] = "completed"
            state["mitigations"] = mitigations
            state["overall_validity_score"] = audit.get("overall_validity_score", 0.5)
            state["validity_confidence"] = audit.get("validity_confidence", "medium")
            state["redesign_needed"] = audit.get("redesign_needed", False)
            state["redesign_recommendations"] = audit.get("redesign_recommendations", [])

            # V4.4: DAG-aware validity validation
            if self._has_dag_evidence(state):
                dag_results, dag_warnings = self._perform_dag_validation(state)

                # Withdraw the PREVIOUS audit's prose before overwriting the
                # record of it (#1639). Without this, a completed rerun that
                # finds nothing leaves iteration 0's DAG sentence in
                # `warnings` forever -- and destroys `dag_validation_warnings`,
                # which is what any later retraction would have used.
                _withdraw_previous_dag_warnings(state)

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
            state["validity_audit_status"] = "failed"
            _retract_stale_verdict(state)
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

        # #1639: this prompt asks an LLM whether the design is sound, and its
        # verdict drives the REDESIGN decision. Showing it "Sample Size: 672206"
        # with no duration and no feasibility verdict asks it to judge an
        # uncaveated projection.
        feasibility = state.get("feasibility_warnings") or []
        feasibility_block = "\n".join(f"- {w}" for w in feasibility)
        feasibility_section = (
            f"\n**!! NOT EXECUTABLE AS SPECIFIED !!**\n{feasibility_block}\n" if feasibility else ""
        )

        return f"""You are a methodological critic reviewing an experiment design. Your job is to find weaknesses.

## Proposed Experiment

**Design Type:** {state.get("design_type", "Not specified")}
**Design Rationale:** {state.get("design_rationale", "Not specified")}

**Treatment:**
{treatment_json}

**Outcome:**
{outcome_json}

**Sample Size:** {power_analysis.get("required_sample_size", "Not calculated")}
**Estimated Duration (days):** {state.get("duration_estimate_days", "Not calculated")}
{feasibility_section}**Randomization Unit:** {state.get("randomization_unit", "individual")}
**Randomization Method:** {state.get("randomization_method", "simple")}
**Stratification:** {state.get("stratification_variables", [])}
**Blocking Variables:** {state.get("blocking_variables", [])}
**Causal Assumptions:** {json.dumps(state.get("causal_assumptions", []), indent=2)}

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
                return cast(Dict[str, Any], json.loads(json_match.group(1)))
            except json.JSONDecodeError:
                pass

        # Try to extract bare JSON
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return cast(Dict[str, Any], json.loads(content[start:end]))
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
        set(dag_nodes)
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
        len(dag_nodes)

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
        len(dag_nodes)

        treatment_idx = node_to_idx.get(treatment_var)
        outcome_idx = node_to_idx.get(outcome_var)

        if treatment_idx is None or outcome_idx is None:
            return modifiers

        # For each node, check if it could be an effect modifier
        for node_idx, node in enumerate(dag_nodes):
            if node in (treatment_var, outcome_var):
                continue

            # Check if connected to treatment (as cause of treatment OR common cause)
            (
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
