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

#: Longest duration a served design may report without being flagged. A trial
#: running beyond a year is a planning decision, not an arithmetic result; beyond
#: that the number is a symptom of an undetectable effect, not a schedule.
#: Deliberately NOT a clamp — see the flagging site for why.
_MAX_PLAUSIBLE_DURATION_DAYS = 365

#: What ``PowerResult.mde`` actually measures, per analysis branch. The Tier 3
#: ``effect_size_type`` enum cannot express this (#1639): it has no member for an
#: absolute risk difference, so binary and time-to-event designs both surface as
#: "rate_ratio" and the MDE's units are lost.
_MDE_SCALES = {
    "cohens_d": "cohens_d",
    "rate_ratio": "absolute_risk_difference",
    "hazard_ratio": "hazard_ratio",
    "absolute_risk_difference": "absolute_risk_difference",
}


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

            # #1639: eval turn 3.6 emitted 94,115 days -- 257.7 years -- and
            # printed it verbatim in its own pre-registration document. The
            # arithmetic was right (ceil(672206/50)*7 == 94115 exactly); it was
            # faithfully propagating an infeasible design, and NOTHING on the
            # output path said so. A 7..365 range does exist at
            # dspy_integration.py:112, but that is a GEPA reward term used to
            # SCORE the optimizer -- it never sees a served design.
            #
            # Flag rather than clamp: the numbers are correct and a silently
            # capped duration would be a fabricated one. The caller gets the real
            # figure plus the reason it cannot be run.
            feasibility_warnings: list[str] = []
            if duration_days > _MAX_PLAUSIBLE_DURATION_DAYS:
                feasibility_warnings.append(
                    f"Estimated duration {duration_days:,} days "
                    f"({duration_days / 365.25:.1f} years) exceeds the "
                    f"{_MAX_PLAUSIBLE_DURATION_DAYS}-day plausibility bound at the "
                    f"assumed accrual of {accrual_rate}/week. The design requires "
                    f"n={forward.sample_size:,}; either the effect size is smaller "
                    f"than the study can practically detect, or accrual must rise."
                )

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
                # #1639: the MDE's own scale, which the payload could not express.
                # ``effect_size_type`` below describes the INPUT effect AND is
                # lossy -- _map_effect_size_type folds both hazard_ratio and
                # absolute_risk_difference into "rate_ratio" because the Tier 3
                # enum has no member for them. So a reader saw
                # minimum_detectable_effect=0.0015 beside a RELATIVE
                # expected_effect_size=0.030 with nothing to say they are measured
                # differently, and read a 20x contradiction where there is none
                # (0.05 baseline x 0.030 relative = 0.0015 absolute, exactly).
                "minimum_detectable_effect_scale": _MDE_SCALES.get(
                    forward.effect_size_type, "unknown"
                ),
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
            # Always set, so a consumer can distinguish "checked, feasible" from
            # "never checked" — the same trap as an empty validity_threats list
            # that is empty because the audit timed out.
            state["feasibility_warnings"] = feasibility_warnings
            # Also raise it on the generic channel every downstream reader
            # already knows. Deliberate duplication: ``feasibility_warnings`` is
            # the structured field, ``warnings`` is the one consumers look at.
            if feasibility_warnings:
                state["warnings"] = state.get("warnings", []) + feasibility_warnings
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
