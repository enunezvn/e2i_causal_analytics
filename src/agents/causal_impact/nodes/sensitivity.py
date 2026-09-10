"""Sensitivity Analysis Node - E-value READING for unmeasured confounding.

Spec docs/superpowers/specs/2026-09-10-sensitivity-gate-calibration-design.md §4.6:
the node delegates every number and word to ``src.causal_engine.evalue`` and takes
its benchmark inputs from the SAME helper the refutation node uses
(``sensitivity_inputs.sensitivity_benchmark_inputs``), so the agent narrative, the
refutation runner and the chat tool cannot disagree on a run. A failure inside the
computation surfaces as ``sensitivity_error`` (status failed), never as a reading.
"""

import time
from typing import Dict, Optional

from src.agents.causal_impact.nodes.sensitivity_inputs import sensitivity_benchmark_inputs
from src.agents.causal_impact.state import CausalImpactState, SensitivityAnalysis, spread_safe
from src.causal_engine import evalue


class SensitivityNode:
    """Performs sensitivity analysis for unmeasured confounding.

    Performance target: <5s
    Type: Standard (computation-light)
    """

    def __init__(self):
        """Initialize sensitivity node."""
        pass

    async def execute(self, state: CausalImpactState) -> Dict:
        """Compute the sensitivity reading from the estimation result and the full frame.

        Args:
            state: Current workflow state with estimation_result

        Returns:
            Updated state with sensitivity_analysis
        """
        start_time = time.time()

        try:
            estimation_result = state.get("estimation_result")
            if not estimation_result:
                raise ValueError("Estimation result not found in state")

            ate = float(estimation_result["ate"])
            ci = (
                float(estimation_result["ate_ci_lower"]),
                float(estimation_result["ate_ci_upper"]),
            )
            outcome_std = self._resolve_outcome_std(state)
            inputs = sensitivity_benchmark_inputs(
                estimation_data=state.get("estimation_data"),
                treatment=str(state.get("treatment_var") or ""),
                outcome=str(state.get("outcome_var") or ""),
                estimation_result=estimation_result,
            )
            # n_rows is None only when no frame was looked at; then the estimation
            # node's own count is the honest fallback (a computed zero is a
            # measurement and must NOT be replaced).
            n_rows = (
                inputs.n_rows if inputs.n_rows is not None else estimation_result.get("sample_size")
            )

            reading = evalue.classify(
                ate,
                ci,
                randomized=bool(state.get("randomized_design")),
                baseline_risk=inputs.baseline_risk,
                outcome_std=outcome_std,
                naive_effect=inputs.naive_effect,
                covariate_factors=inputs.covariate_bias_factors,
                n_rows=n_rows,
            )
            randomized = reading.reading == evalue.READING_RANDOMIZED

            sensitivity_analysis: SensitivityAnalysis = {
                "e_value": reading.e_value_point,
                "e_value_ci": reading.e_value_ci,
                "interpretation": (
                    # DESIGN declaration (dataset spec via the API layer): treatment
                    # assignment is exogenous, so the E-value numbers stay reported
                    # but as information, not as a validity risk.
                    "Randomized design: treatment assignment is exogenous by construction, so "
                    "unmeasured confounding of assignment is excluded by design. E-value "
                    f"{reading.e_value_point:.2f} (CI bound {reading.e_value_ci:.2f}) is reported "
                    "for information only and does not indicate a validity risk."
                    if randomized
                    else reading.message
                ),
                # Randomized designs are robust to confounding of assignment by design.
                "robust_to_confounding": randomized or reading.reading == evalue.READING_BEYOND,
                "unmeasured_confounder_strength": reading.reading,
                "reading": reading.reading,
                "headline": reading.headline,
                "rr_point": reading.rr_point,
                "rr_ci": reading.rr_ci,
                "benchmark": reading.benchmark,
                "benchmark_basis": reading.benchmark_basis,
                "conversion": reading.conversion,
            }

            return {
                **spread_safe(state),
                "sensitivity_analysis": sensitivity_analysis,
                "sensitivity_latency_ms": (time.time() - start_time) * 1000,
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
        """Outcome SD (σ_Y) from the estimation-data passthrough; None when unavailable.

        A MISSING frame/column yields None (the reading is served on the raw SMD
        path). Everything else goes through ``evalue.outcome_std_from_frame``, the
        one function all three engines share: it drops the NaN treatment/outcome rows
        the estimation node masked before it fit, so the raw passthrough frame's
        missing values cannot turn a usable estimate into a failure.

        No try/except: a PRESENT but unusable outcome column raises there, and that
        surfaces as ``sensitivity_error`` rather than a reading standardized by
        nothing (spec §5, same rule as the runner).
        """
        data = state.get("estimation_data")
        outcome_var = state.get("outcome_var")
        if data is None or not outcome_var:
            return None
        if not hasattr(data, "columns") or outcome_var not in data.columns:
            return None
        return evalue.outcome_std_from_frame(
            data, outcome_var, treatment=state.get("treatment_var")
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
