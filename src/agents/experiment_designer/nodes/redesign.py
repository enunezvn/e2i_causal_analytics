"""Redesign Node.

This node incorporates validity audit feedback to improve the experiment design.
It modifies the design based on identified threats and recommendations,
then sends the design back through power analysis and validity audit.

Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md lines 86-102
Contract: .claude/contracts/tier3-contracts.md lines 82-142
"""

import time
from datetime import datetime, timezone
from typing import Any

from src.agents.experiment_designer.state import (
    DesignIteration,
    ErrorDetails,
    ExperimentDesignState,
)


class RedesignNode:
    """Incorporates validity audit feedback to improve experiment design.

    This node:
    1. Analyzes validity threats and recommendations
    2. Applies mitigations to the current design
    3. Records iteration history
    4. Prepares state for another power analysis cycle

    Pure computation - uses rule-based redesign logic.
    Performance Target: <100ms for redesign
    """

    def __init__(self):
        """Initialize redesign node."""
        self._mitigation_rules: dict[str, dict[str, Any]] = {
            "selection_bias": {
                "action": "add_stratification",
                "variables": ["baseline_engagement", "territory_size"],
            },
            "confounding": {
                "action": "add_blocking",
                "variables": ["region", "competitive_intensity"],
            },
            "contamination": {
                "action": "increase_separation",
                "recommendation": "Use geographic buffer zones between arms",
            },
            "measurement": {
                "action": "add_blinding",
                "recommendation": "Implement outcome assessor blinding",
            },
            "attrition": {
                "action": "increase_sample",
                "multiplier": 1.2,
            },
            "temporal": {
                "action": "extend_baseline",
                "recommendation": "Add pre-randomization observation period",
            },
        }

    async def execute(self, state: ExperimentDesignState) -> ExperimentDesignState:
        """Execute redesign based on validity audit feedback.

        Args:
            state: Current agent state with validity audit results

        Returns:
            Updated state with redesigned experiment
        """
        start_time = time.time()

        # Skip if status is failed
        if state.get("status") == "failed":
            return state

        try:
            # Update status
            state["status"] = "redesigning"

            # Record current iteration
            current_iteration = state.get("current_iteration", 0)
            power_analysis = state.get("power_analysis", {})

            iteration_record: DesignIteration = {
                "iteration_number": current_iteration,
                "design_type": state.get("design_type", "RCT"),
                "validity_threats_identified": len(state.get("validity_threats", [])),
                "critical_threats": sum(
                    1 for t in state.get("validity_threats", []) if t.get("severity") == "critical"
                ),
                "power_achieved": power_analysis.get("achieved_power", 0.0),
                "redesign_reason": self._get_redesign_reason(state),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Add to iteration history
            iteration_history = state.get("iteration_history", [])
            iteration_history.append(iteration_record)
            state["iteration_history"] = iteration_history

            # Increment iteration counter
            state["current_iteration"] = current_iteration + 1

            # Apply mitigations based on validity threats
            state = self._apply_mitigations(state)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            node_latencies = state.get("node_latencies_ms", {})
            node_latencies[f"redesign_{current_iteration}"] = latency_ms
            state["node_latencies_ms"] = node_latencies

            # Set status back to calculating for power analysis
            state["status"] = "calculating"

            # Add warning about redesign
            state["warnings"] = state.get("warnings", []) + [
                f"Design iteration {current_iteration + 1}: Applying mitigations for identified threats"
            ]

        except Exception as e:
            error: ErrorDetails = {
                "node": "redesign",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "recoverable": True,
            }
            state["errors"] = state.get("errors", []) + [error]
            state["warnings"] = state.get("warnings", []) + [
                f"Redesign failed: {str(e)}. Proceeding with current design."
            ]
            # Continue to template generation despite error
            state["status"] = "generating"

        return state

    def _get_redesign_reason(self, state: ExperimentDesignState) -> str:
        """Get the primary reason for redesign.

        Args:
            state: Current agent state

        Returns:
            Redesign reason string
        """
        threats = state.get("validity_threats", [])
        recommendations = state.get("redesign_recommendations", [])

        if recommendations:
            return recommendations[0]

        critical_threats = [t for t in threats if t.get("severity") == "critical"]
        if critical_threats:
            return f"Critical threat: {critical_threats[0].get('threat_name', 'unknown')}"

        high_threats = [t for t in threats if t.get("severity") == "high"]
        if high_threats:
            return f"High severity threat: {high_threats[0].get('threat_name', 'unknown')}"

        return "General validity concerns"

    def _apply_mitigations(self, state: ExperimentDesignState) -> ExperimentDesignState:
        """Apply mitigations based on identified threats.

        Args:
            state: Current agent state

        Returns:
            Updated state with mitigations applied
        """
        threats = state.get("validity_threats", [])
        mitigations = state.get("mitigations", [])

        # Current design parameters
        stratification_vars = list(state.get("stratification_variables", []))
        blocking_variables = list(state.get("blocking_variables", []))
        power_analysis = state.get("power_analysis", {})

        for threat in threats:
            threat_name = threat.get("threat_name", "")
            severity = threat.get("severity", "medium")

            # Only apply mitigations for medium+ severity threats
            if severity in ["low"]:
                continue

            # Apply rule-based mitigation
            if threat_name in self._mitigation_rules:
                rule = self._mitigation_rules[threat_name]
                action = rule.get("action", "")

                if action == "add_stratification":
                    for var in rule.get("variables", []):
                        if var not in stratification_vars:
                            stratification_vars.append(var)

                elif action == "add_blocking":
                    for var in rule.get("variables", []):
                        if var not in blocking_variables:
                            blocking_variables.append(var)

                elif action == "increase_sample":
                    multiplier = rule.get("multiplier", 1.1)
                    if power_analysis:
                        current_n = power_analysis.get("required_sample_size", 500)
                        power_analysis["required_sample_size"] = int(current_n * multiplier)
                        power_analysis["required_sample_size_per_arm"] = int(
                            power_analysis["required_sample_size"] // 2
                        )

        # Also apply explicit mitigation recommendations
        for mitigation in mitigations:
            effectiveness = mitigation.get("effectiveness_rating", "medium")
            if effectiveness == "high":
                # Add implementation note as warning
                steps = mitigation.get("implementation_steps", [])
                if steps:
                    state["warnings"] = state.get("warnings", []) + [
                        f"Mitigation applied: {steps[0]}"
                    ]

        # Update state with mitigations
        state["stratification_variables"] = stratification_vars
        state["blocking_variables"] = blocking_variables
        state["power_analysis"] = power_analysis  # type: ignore[typeddict-item]

        # Add causal assumption about mitigations
        causal_assumptions = list(state.get("causal_assumptions", []))
        causal_assumptions.append(
            f"Redesign iteration {state.get('current_iteration', 0)}: "
            f"Applied mitigations for {len(threats)} identified threats"
        )
        state["causal_assumptions"] = causal_assumptions

        return state
