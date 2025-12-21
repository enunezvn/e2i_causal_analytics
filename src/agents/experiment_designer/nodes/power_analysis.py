"""Power Analysis Node.

This node performs statistical power analysis calculations for experiment design.
It calculates required sample size, minimum detectable effect (MDE), and
estimated experiment duration.

Algorithm: .claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md lines 417-550
Contract: .claude/contracts/tier3-contracts.md lines 82-142
"""

import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy import stats

from src.agents.experiment_designer.state import (
    ErrorDetails,
    ExperimentDesignState,
    PowerAnalysisResult,
)


class PowerAnalysisNode:
    """Statistical power analysis for experiment design.

    This node calculates:
    1. Required sample size for target power
    2. Minimum detectable effect (MDE) given constraints
    3. Estimated experiment duration based on accrual rate
    4. Sensitivity analysis for key parameters

    Pure computation - no LLM needed.
    Performance Target: <100ms for power calculations
    """

    def __init__(self):
        """Initialize power analysis node."""
        self._default_alpha = 0.05
        self._default_power = 0.80
        self._default_effect_size = 0.25

    async def execute(self, state: ExperimentDesignState) -> ExperimentDesignState:
        """Execute power analysis.

        Args:
            state: Current agent state with design outputs

        Returns:
            Updated state with power analysis results
        """
        start_time = time.time()

        # Skip if status is failed
        if state.get("status") == "failed":
            return state

        try:
            # Update status
            state["status"] = "calculating"

            # Extract parameters
            design_type = state.get("design_type", "RCT")
            constraints = state.get("constraints", {})
            outcomes = state.get("outcomes", [])

            # Get outcome type from first primary outcome
            outcome_type = "continuous"
            expected_effect = None
            baseline_value = None
            for outcome in outcomes:
                if outcome.get("is_primary", False):
                    outcome_type = outcome.get("metric_type", "continuous")
                    expected_effect = outcome.get("expected_effect_size")
                    baseline_value = outcome.get("baseline_value")
                    break

            # Get power analysis parameters
            effect_size = expected_effect or constraints.get(
                "expected_effect_size", self._default_effect_size
            )
            alpha = constraints.get("alpha", self._default_alpha)
            power_target = constraints.get("power", self._default_power)

            # Calculate power based on design type (case-insensitive)
            design_type_lower = design_type.lower().replace("-", "_")
            if design_type_lower in ["cluster_rct", "cluster"]:
                result = self._cluster_rct_power(state, effect_size, alpha, power_target)
            elif outcome_type == "binary":
                result = self._binary_outcome_power(
                    state, effect_size, alpha, power_target, baseline_value
                )
            elif outcome_type == "time_to_event":
                result = self._time_to_event_power(state, effect_size, alpha, power_target)
            else:
                result = self._continuous_outcome_power(effect_size, alpha, power_target)

            # Calculate duration estimate
            accrual_rate = constraints.get("weekly_accrual", 50)
            duration_weeks = max(1, int(np.ceil(result["sample_size"] / accrual_rate)))
            duration_days = duration_weeks * 7

            # Run sensitivity analysis
            sensitivity = self._run_sensitivity_analysis(
                effect_size, alpha, power_target, result["sample_size"]
            )

            # Create PowerAnalysisResult
            power_result: PowerAnalysisResult = {
                "required_sample_size": result["sample_size"],
                "required_sample_size_per_arm": result["details"].get(
                    "n_per_arm", result["sample_size"] // 2
                ),
                "achieved_power": power_target,
                "minimum_detectable_effect": result["mde"],
                "alpha": alpha,
                "effect_size_type": result["details"].get("effect_size_type", "cohens_d"),
                "assumptions": result["details"].get("assumptions", []),
                "sensitivity_analysis": sensitivity,
            }

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            node_latencies = state.get("node_latencies_ms", {})
            node_latencies["power_analysis"] = latency_ms

            # Update state
            state["power_analysis"] = power_result
            state["sample_size_justification"] = (
                f"Based on {result['details'].get('analysis_type', 'power analysis')}: "
                f"n={result['sample_size']} provides {power_target*100:.0f}% power to detect "
                f"effect size of {effect_size:.3f} at alpha={alpha}."
            )
            state["duration_estimate_days"] = duration_days
            state["node_latencies_ms"] = node_latencies

            # Update status for next node
            state["status"] = "auditing"

        except Exception as e:
            error: ErrorDetails = {
                "node": "power_analysis",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "recoverable": True,
            }
            state["errors"] = state.get("errors", []) + [error]
            state["warnings"] = state.get("warnings", []) + [
                f"Power analysis failed - using default sample size: {str(e)}"
            ]

            # Set defaults and continue
            state["power_analysis"] = PowerAnalysisResult(
                required_sample_size=500,
                required_sample_size_per_arm=250,
                achieved_power=0.0,
                minimum_detectable_effect=0.0,
                alpha=0.05,
                effect_size_type="unknown",
                assumptions=["Default values used due to calculation error"],
            )
            state["duration_estimate_days"] = 70  # ~10 weeks default
            state["status"] = "auditing"

        return state

    def _continuous_outcome_power(
        self, effect_size: float, alpha: float, power: float
    ) -> dict[str, Any]:
        """Power analysis for continuous outcome (two-sample t-test).

        Formula: n = 2 * ((z_alpha + z_beta) / effect_size)^2

        Args:
            effect_size: Cohen's d effect size
            alpha: Type I error rate
            power: Target power (1 - beta)

        Returns:
            Dictionary with sample_size, mde, and details
        """
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        n_per_arm = int(np.ceil(2 * ((z_alpha + z_beta) / effect_size) ** 2))
        total_n = n_per_arm * 2

        return {
            "sample_size": total_n,
            "mde": effect_size,
            "details": {
                "n_per_arm": n_per_arm,
                "analysis_type": "two_sample_t_test",
                "effect_size_type": "cohens_d",
                "assumptions": [
                    "Equal variance between groups",
                    "Normal distribution of outcome",
                    "Independent observations",
                ],
            },
        }

    def _binary_outcome_power(
        self,
        state: ExperimentDesignState,
        effect_size: float,
        alpha: float,
        power: float,
        baseline_rate: float | None = None,
    ) -> dict[str, Any]:
        """Power analysis for binary outcome (two proportions z-test).

        Args:
            state: Current agent state
            effect_size: Relative effect size (proportion change)
            alpha: Type I error rate
            power: Target power
            baseline_rate: Baseline proportion (p1)

        Returns:
            Dictionary with sample_size, mde, and details
        """
        p1 = baseline_rate or state.get("constraints", {}).get("baseline_rate", 0.3)
        p2 = p1 + effect_size * p1  # Relative effect

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        p_bar = (p1 + p2) / 2
        diff = abs(p2 - p1)

        if diff < 0.001:
            diff = 0.001  # Prevent division by zero

        n_per_arm = int(np.ceil(2 * p_bar * (1 - p_bar) * ((z_alpha + z_beta) / diff) ** 2))

        return {
            "sample_size": n_per_arm * 2,
            "mde": diff,
            "details": {
                "n_per_arm": n_per_arm,
                "baseline_rate": p1,
                "expected_treatment_rate": p2,
                "analysis_type": "two_proportions_z_test",
                "effect_size_type": "rate_ratio",
                "assumptions": [
                    "Independent observations",
                    "Large sample approximation valid",
                    f"Baseline rate: {p1:.3f}",
                ],
            },
        }

    def _cluster_rct_power(
        self, state: ExperimentDesignState, effect_size: float, alpha: float, power: float
    ) -> dict[str, Any]:
        """Power analysis for cluster RCT with ICC adjustment.

        Design effect = 1 + (cluster_size - 1) * ICC

        Args:
            state: Current agent state
            effect_size: Effect size
            alpha: Type I error rate
            power: Target power

        Returns:
            Dictionary with sample_size, mde, and details
        """
        constraints = state.get("constraints", {})
        icc = constraints.get("expected_icc", 0.05)
        cluster_size = constraints.get("cluster_size", 20)

        # Get base sample size
        base_result = self._continuous_outcome_power(effect_size, alpha, power)
        base_n = base_result["sample_size"]

        # Apply design effect
        design_effect = 1 + (cluster_size - 1) * icc
        adjusted_n = int(np.ceil(base_n * design_effect))
        n_clusters = int(np.ceil(adjusted_n / cluster_size))

        return {
            "sample_size": adjusted_n,
            "mde": effect_size,
            "details": {
                "n_clusters_total": n_clusters,
                "n_clusters_per_arm": max(1, n_clusters // 2),
                "cluster_size": cluster_size,
                "icc": icc,
                "design_effect": design_effect,
                "base_sample_size": base_n,
                "analysis_type": "cluster_rct_adjusted",
                "effect_size_type": "cohens_d",
                "assumptions": [
                    f"Intra-cluster correlation (ICC): {icc}",
                    f"Average cluster size: {cluster_size}",
                    f"Design effect: {design_effect:.2f}",
                    "Exchangeable correlation structure within clusters",
                ],
            },
        }

    def _time_to_event_power(
        self, state: ExperimentDesignState, effect_size: float, alpha: float, power: float
    ) -> dict[str, Any]:
        """Power analysis for time-to-event outcome (log-rank test).

        Uses Schoenfeld formula for hazard ratio.

        Args:
            state: Current agent state
            effect_size: Hazard ratio (treatment/control)
            alpha: Type I error rate
            power: Target power

        Returns:
            Dictionary with sample_size, mde, and details
        """
        constraints = state.get("constraints", {})
        event_rate = constraints.get("event_rate", 0.5)
        accrual_period = constraints.get("accrual_weeks", 12)
        follow_up_period = constraints.get("follow_up_weeks", 12)

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        # Schoenfeld formula: d = 4 * (z_alpha + z_beta)^2 / (log(HR))^2
        log_hr = np.log(effect_size) if effect_size > 0 else np.log(0.75)
        if abs(log_hr) < 0.01:
            log_hr = 0.01  # Prevent division by zero

        required_events = int(np.ceil(4 * ((z_alpha + z_beta) / log_hr) ** 2))
        total_n = int(np.ceil(required_events / event_rate))

        return {
            "sample_size": total_n,
            "mde": effect_size,
            "details": {
                "required_events": required_events,
                "expected_event_rate": event_rate,
                "hazard_ratio": effect_size,
                "accrual_weeks": accrual_period,
                "follow_up_weeks": follow_up_period,
                "analysis_type": "log_rank_test",
                "effect_size_type": "hazard_ratio",
                "assumptions": [
                    "Proportional hazards assumption",
                    f"Expected event rate: {event_rate:.2%}",
                    "Exponential survival distribution",
                ],
            },
        }

    def _run_sensitivity_analysis(
        self, effect_size: float, alpha: float, power: float, base_n: int
    ) -> dict[str, Any]:
        """Run sensitivity analysis on key parameters.

        Args:
            effect_size: Base effect size
            alpha: Type I error rate
            power: Target power
            base_n: Base sample size

        Returns:
            Sensitivity analysis results
        """
        sensitivity = {
            "effect_size_variations": {},
            "power_variations": {},
        }

        # Effect size sensitivity
        for multiplier in [0.8, 0.9, 1.1, 1.2]:
            varied_effect = effect_size * multiplier
            result = self._continuous_outcome_power(varied_effect, alpha, power)
            sensitivity["effect_size_variations"][f"{multiplier:.1f}x"] = {
                "effect_size": varied_effect,
                "sample_size": result["sample_size"],
                "change_from_base": result["sample_size"] - base_n,
            }

        # Power sensitivity
        for power_level in [0.70, 0.85, 0.90]:
            result = self._continuous_outcome_power(effect_size, alpha, power_level)
            sensitivity["power_variations"][f"{power_level:.0%}"] = {
                "power": power_level,
                "sample_size": result["sample_size"],
                "change_from_base": result["sample_size"] - base_n,
            }

        return sensitivity
