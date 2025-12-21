"""Sensitivity Analysis Node - E-value calculation for unmeasured confounding.

Computes E-values to quantify robustness to unmeasured confounding.
"""

import time
from typing import Dict

import numpy as np

from src.agents.causal_impact.state import CausalImpactState, SensitivityAnalysis


class SensitivityNode:
    """Performs sensitivity analysis for unmeasured confounding.

    Performance target: <5s
    Type: Standard (computation-light)
    """

    def __init__(self):
        """Initialize sensitivity node."""
        pass

    async def execute(self, state: CausalImpactState) -> Dict:
        """Run sensitivity analysis.

        Args:
            state: Current workflow state with estimation_result

        Returns:
            Updated state with sensitivity_analysis
        """
        start_time = time.time()

        try:
            # Get estimation result
            estimation_result = state.get("estimation_result")
            if not estimation_result:
                raise ValueError("Estimation result not found in state")

            ate = estimation_result["ate"]
            ate_ci_lower = estimation_result["ate_ci_lower"]
            estimation_result["ate_ci_upper"]

            # Calculate E-values
            e_value_point = self._calculate_e_value(ate)
            e_value_ci = self._calculate_e_value(ate_ci_lower)

            # Interpret E-value
            interpretation = self._interpret_e_value(e_value_point)

            # Determine robustness
            robust = e_value_point > 2.0  # Common threshold

            # Classify unmeasured confounder strength needed
            if e_value_point < 1.5:
                strength = "weak"
            elif e_value_point < 3.0:
                strength = "moderate"
            else:
                strength = "strong"

            sensitivity_analysis: SensitivityAnalysis = {
                "e_value": e_value_point,
                "e_value_ci": e_value_ci,
                "interpretation": interpretation,
                "robust_to_confounding": robust,
                "unmeasured_confounder_strength": strength,
            }

            latency_ms = (time.time() - start_time) * 1000

            return {
                **state,
                "sensitivity_analysis": sensitivity_analysis,
                "sensitivity_latency_ms": latency_ms,
                "current_phase": "interpreting",
            }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                **state,
                "sensitivity_error": str(e),
                "sensitivity_latency_ms": latency_ms,
                "status": "failed",
                "error_message": f"Sensitivity analysis failed: {e}",
            }

    def _calculate_e_value(self, effect: float) -> float:
        """Calculate E-value for a given effect estimate.

        E-value is the minimum strength of association (on the risk ratio scale)
        that an unmeasured confounder would need to have with both the treatment
        and outcome to fully explain away the observed effect.

        Formula (VanderWeele & Ding, 2017):
        E-value = RR + sqrt(RR * (RR - 1))

        Where RR is the risk ratio (approximated from effect size).

        Args:
            effect: Effect estimate (ATE)

        Returns:
            E-value (>= 1)
        """
        # Convert effect to approximate risk ratio
        # Assuming standardized effect: RR ≈ exp(effect)
        rr = np.exp(abs(effect))

        # Calculate E-value
        if rr <= 1:
            return 1.0  # No unmeasured confounding needed

        e_value = rr + np.sqrt(rr * (rr - 1))

        return float(e_value)

    def _interpret_e_value(self, e_value: float) -> str:
        """Interpret E-value in natural language.

        Args:
            e_value: Computed E-value

        Returns:
            Human-readable interpretation
        """
        if e_value < 1.25:
            return (
                f"E-value of {e_value:.2f} suggests the effect could be explained "
                "by very weak unmeasured confounding. Exercise caution in causal interpretation."
            )
        elif e_value < 2.0:
            return (
                f"E-value of {e_value:.2f} indicates the effect could be explained "
                "by moderate unmeasured confounding. The causal claim has some robustness "
                "but should be interpreted carefully."
            )
        elif e_value < 3.0:
            return (
                f"E-value of {e_value:.2f} suggests the effect would require fairly "
                "strong unmeasured confounding to be fully explained away. The causal "
                "claim has good robustness."
            )
        else:
            return (
                f"E-value of {e_value:.2f} indicates the effect would require very "
                "strong unmeasured confounding to be fully explained away. The causal "
                "claim has strong robustness to unmeasured confounding."
            )


# Standalone function for LangGraph integration
async def analyze_sensitivity(state: CausalImpactState) -> Dict:
    """Perform sensitivity analysis (standalone function).

    Args:
        state: Current workflow state

    Returns:
        Updated state with sensitivity_analysis
    """
    node = SensitivityNode()
    return await node.execute(state)
