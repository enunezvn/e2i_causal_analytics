"""Power Analysis Node.

Thin adapter over src/utils/power_analysis_lib.py. The pure power-calculation
formulas live in that library so Tier 0 (ML Foundation) sufficiency checks can
reuse them without coupling to ExperimentDesignState. This node remains the
Tier 3 entry point; its public interface is unchanged.

Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md lines 417-550
Contract: .claude/contracts/tier3-contracts.md lines 82-142
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.agents.experiment_designer.state import (
    ErrorDetails,
    ExperimentDesignState,
    PowerAnalysisResult,
)
from src.utils.power_analysis_lib import (
    PowerCalculationError,
    PowerResult,
    binary_outcome_power,
    cluster_rct_power,
    continuous_outcome_power,
    sensitivity_variations,
    time_to_event_power,
)


class PowerAnalysisNode:
    """Statistical power analysis for experiment design.

    Pure computation - no LLM needed.
    Performance target: <100ms for power calculations.
    """

    def __init__(self) -> None:
        # Instance attributes preserved verbatim for backward-compat with existing tests
        self._default_alpha = 0.05
        self._default_power = 0.80
        self._default_effect_size = 0.25

    async def execute(self, state: ExperimentDesignState) -> ExperimentDesignState:
        """Execute power analysis."""
        start_time = time.time()

        if state.get("status") == "failed":
            return state

        try:
            state["status"] = "calculating"

            design_type = state.get("design_type", "RCT")
            constraints = state.get("constraints", {})
            outcomes = state.get("outcomes", [])

            outcome_type = "continuous"
            expected_effect: float | None = None
            baseline_value: float | None = None
            for outcome in outcomes:
                if outcome.get("is_primary", False):
                    outcome_type = outcome.get("metric_type", "continuous")
                    expected_effect = outcome.get("expected_effect_size")
                    baseline_value = outcome.get("baseline_value")
                    break

            effect_size = (
                expected_effect
                if expected_effect is not None
                else constraints.get("expected_effect_size", self._default_effect_size)
            )
            alpha = constraints.get("alpha", self._default_alpha)
            power_target = constraints.get("power", self._default_power)

            design_type_lower = design_type.lower().replace("-", "_")
            forward: PowerResult
            if design_type_lower in ("cluster_rct", "cluster"):
                icc = constraints.get("expected_icc", 0.05)
                cluster_size = constraints.get("cluster_size", 20)
                forward = cluster_rct_power(effect_size, alpha, power_target, icc, cluster_size)
            elif outcome_type == "binary":
                baseline_rate = (
                    baseline_value
                    if baseline_value is not None
                    else constraints.get("baseline_rate", 0.3)
                )
                forward = binary_outcome_power(effect_size, alpha, power_target, baseline_rate)
            elif outcome_type == "time_to_event":
                event_rate = constraints.get("event_rate", 0.5)
                forward = time_to_event_power(effect_size, alpha, power_target, event_rate)
            else:
                forward = continuous_outcome_power(effect_size, alpha, power_target)

            accrual_rate = constraints.get("weekly_accrual", 50)
            duration_weeks = max(1, int(np.ceil(forward.sample_size / accrual_rate)))
            duration_days = duration_weeks * 7

            sensitivity = sensitivity_variations(
                effect_size,
                alpha,
                power_target,
                forward.sample_size,
                outcome_type="continuous",  # Sensitivity always uses continuous proxy (legacy parity)
            )

            effect_size_type = self._map_effect_size_type(forward.effect_size_type)
            assumptions = list(forward.assumptions)
            if outcome_type == "binary" and "baseline_rate" in forward.extra:
                # Preserve legacy assumption ordering
                br = forward.extra["baseline_rate"]
                assumptions = [a for a in assumptions if not a.startswith("Baseline rate")]
                assumptions.append(f"Baseline rate: {br:.3f}")

            power_result: PowerAnalysisResult = {
                "required_sample_size": forward.sample_size,
                "required_sample_size_per_arm": forward.sample_size_per_arm,
                "achieved_power": power_target,
                "minimum_detectable_effect": forward.mde,
                "alpha": alpha,
                "effect_size_type": effect_size_type,
                "assumptions": assumptions,
                "sensitivity_analysis": sensitivity,
            }

            latency_ms = int((time.time() - start_time) * 1000)
            node_latencies = state.get("node_latencies_ms", {})
            node_latencies["power_analysis"] = latency_ms

            state["power_analysis"] = power_result
            state["sample_size_justification"] = (
                f"Based on {forward.analysis_type}: "
                f"n={forward.sample_size} provides {power_target * 100:.0f}% power to detect "
                f"effect size of {effect_size:.3f} at alpha={alpha}."
            )
            state["duration_estimate_days"] = duration_days
            state["node_latencies_ms"] = node_latencies

            state["required_sample_size"] = forward.sample_size
            state["statistical_power"] = power_target

            state["status"] = "auditing"

        except (PowerCalculationError, Exception) as e:
            # D2: no plausible-fake fallback. Record the failure honestly.
            error: ErrorDetails = {
                "node": "power_analysis",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "recoverable": False,
            }
            state["errors"] = state.get("errors", []) + [error]
            state["warnings"] = state.get("warnings", []) + [f"Power analysis failed: {str(e)}"]
            state["status"] = "failed"

            latency_ms = int((time.time() - start_time) * 1000)
            node_latencies = state.get("node_latencies_ms", {})
            node_latencies["power_analysis"] = latency_ms
            state["node_latencies_ms"] = node_latencies

        return state

    @staticmethod
    def _map_effect_size_type(
        lib_type: str,
    ) -> Any:
        """Map library effect-size enum to the Tier 3 PowerAnalysisResult enum.

        Library uses "absolute_risk_difference" internally; Tier 3 schema
        expects one of: cohens_d, odds_ratio, rate_ratio, percentage_change.
        """
        mapping = {
            "cohens_d": "cohens_d",
            "rate_ratio": "rate_ratio",
            "hazard_ratio": "rate_ratio",  # closest enum match in PowerAnalysisResult
            "absolute_risk_difference": "rate_ratio",
        }
        return mapping.get(lib_type, "cohens_d")
