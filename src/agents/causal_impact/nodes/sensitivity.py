"""Sensitivity Analysis Node - E-value calculation for unmeasured confounding.

Computes E-values to quantify robustness to unmeasured confounding.
"""

import time
from typing import Dict, Optional

import numpy as np

from src.agents.causal_impact.state import CausalImpactState, SensitivityAnalysis, spread_safe


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
            ate_ci_upper = estimation_result["ate_ci_upper"]
            # The CI-bound E-value is the CONSERVATIVE one — the bound CLOSEST to
            # the null (smallest |effect|). The previous code read ate_ci_upper
            # but discarded it and used only ate_ci_lower, which is wrong for
            # negative/asymmetric CIs (mirrors the runner's min(|lo|,|hi|)).
            # M-stat1 null-crossing guard: when the CI straddles (or touches) 0
            # the effect is statistically indistinguishable from the null, so the
            # CONSERVATIVE E-value-for-CI must collapse to the null (1.0). Without
            # this, min(|lo|,|hi|) reports a spurious > 1 bound (e.g. (-0.3,0.5)
            # -> 0.3 -> E-value ~ 2.0, falsely robust).
            ci_straddles_null = ate_ci_lower <= 0.0 <= ate_ci_upper
            ci_bound = min(abs(ate_ci_lower), abs(ate_ci_upper))

            # H3: the E-value RR approximation needs a STANDARDIZED effect, so
            # resolve the outcome SD (σ_Y) from the estimation data and divide the
            # raw ATE by it. NOTE: estimation_result["ate_std"] is the ATE's
            # standard error, NOT σ_Y, so it must NOT be used for standardization.
            #
            # Approximation note: this node uses ``RR ≈ exp(d)``; the pipeline
            # runner uses the Chinn(2000) ``RR ≈ exp(0.91·d)`` SMD→OR factor. Both
            # are valid VanderWeele-Ding approximations and BOTH are now
            # scale-invariant (H3); the numeric E-values differ by the 0.91 factor
            # between the two engines by design.
            outcome_std = self._resolve_outcome_std(state)

            # Calculate E-values
            e_value_point = self._calculate_e_value(ate, outcome_std)
            # M-stat1: a null-crossing CI collapses the conservative bound to 1.0.
            e_value_ci = (
                1.0 if ci_straddles_null else self._calculate_e_value(ci_bound, outcome_std)
            )

            if state.get("randomized_design"):
                # DESIGN declaration (dataset spec via the API layer): the
                # treatment is genuinely randomized, so unmeasured confounding
                # of assignment is excluded by construction. Narrating "the
                # effect could be explained by moderate confounding" here is
                # misleading — the E-value numbers stay reported, but as
                # information, and robustness-to-confounding holds by design.
                interpretation = (
                    "Randomized design: treatment assignment is exogenous by "
                    "construction, so unmeasured confounding of assignment is "
                    f"excluded by design. E-value {e_value_point:.2f} "
                    f"(CI bound {e_value_ci:.2f}) is reported for information "
                    "only and does not indicate a validity risk."
                )
                robust = True
                strength = "not_applicable_randomized"
            else:
                # Interpret E-value (point estimate drives the human narrative).
                interpretation = self._interpret_e_value(e_value_point)

                # M-stat2: the robust/strength decision uses the CONSERVATIVE CI
                # E-value, not the point estimate, so a wide / null-crossing CI
                # is correctly flagged as non-robust.
                robust = e_value_ci > 2.0  # Common threshold

                # Classify unmeasured confounder strength needed (off the CI bound).
                if e_value_ci < 1.5:
                    strength = "weak"
                elif e_value_ci < 3.0:
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
                **spread_safe(state),
                "sensitivity_analysis": sensitivity_analysis,
                "sensitivity_latency_ms": latency_ms,
                "current_phase": "interpreting",
            }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                **spread_safe(state),
                "sensitivity_error": str(e),
                "sensitivity_latency_ms": latency_ms,
                "status": "failed",
                "error_message": f"Sensitivity analysis failed: {e}",
            }

    def _resolve_outcome_std(self, state: CausalImpactState) -> Optional[float]:
        """Resolve the outcome SD (σ_Y) for E-value standardization (H3).

        Reads the estimation-data passthrough; returns None when no data /
        outcome column is available (then the E-value is computed on the raw
        effect — scale-dependent, but better than crashing).
        """
        data = state.get("estimation_data")
        outcome_var = state.get("outcome_var")
        if data is None or not outcome_var:
            return None
        try:
            if hasattr(data, "columns") and outcome_var in data.columns:
                sd = float(np.std(np.asarray(data[outcome_var], dtype=float)))
                return sd if np.isfinite(sd) and sd > 0 else None
        except Exception:  # noqa: BLE001 - non-numeric / missing → no standardization
            return None
        return None

    def _calculate_e_value(self, effect: float, outcome_std: Optional[float] = None) -> float:
        """Calculate E-value for a given effect estimate.

        E-value is the minimum strength of association (on the risk ratio scale)
        that an unmeasured confounder would need to have with both the treatment
        and outcome to fully explain away the observed effect.

        Formula (VanderWeele & Ding, 2017):
        E-value = RR + sqrt(RR * (RR - 1))

        Where RR is the risk ratio (approximated from the STANDARDIZED effect).

        Args:
            effect: Effect estimate (ATE), in native outcome units
            outcome_std: Outcome SD (σ_Y) used to standardize the effect (H3); the
                approximation requires a standardized mean difference, so a raw
                ATE in native units would otherwise make the E-value scale-dependent.

        Returns:
            E-value (>= 1)
        """
        # H3: standardize the effect by the outcome SD before the RR step.
        d = abs(effect)
        if outcome_std is not None and np.isfinite(outcome_std) and outcome_std > 0:
            d = d / outcome_std

        # Convert the standardized effect to an approximate risk ratio.
        rr = np.exp(d)

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
