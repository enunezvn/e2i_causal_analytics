"""Pure power-analysis primitives.

Extracted from src/agents/experiment_designer/nodes/power_analysis.py so they
can be consumed by Tier 0 (ML Foundation) sufficiency checks without coupling
to the Tier 3 ExperimentDesignState.

All functions are stateless and side-effect-free. They raise PowerCalculationError
on impossible inputs rather than returning plausible-fake values (anti-mocking
discipline; see CLAUDE.md).

Forward calculations (n given effect_size, alpha, power):
  - continuous_outcome_power
  - binary_outcome_power
  - cluster_rct_power
  - time_to_event_power

Reverse calculations (effect detectable given n, alpha, power):
  - mde_for_sample_size

Diagnostics:
  - sensitivity_grid
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy import stats


class PowerCalculationError(ValueError):
    """Raised when a power calculation cannot be performed with the given inputs."""


OutcomeType = Literal["continuous", "binary", "time_to_event"]
EffectSizeType = Literal["cohens_d", "rate_ratio", "hazard_ratio", "absolute_risk_difference"]


@dataclass
class PowerResult:
    """Result of a forward power calculation."""

    sample_size: int
    sample_size_per_arm: int
    mde: float
    analysis_type: str
    effect_size_type: EffectSizeType
    assumptions: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def _z_scores(alpha: float, power: float) -> tuple[float, float]:
    if not 0.0 < alpha < 1.0:
        raise PowerCalculationError(f"alpha must be in (0, 1); got {alpha}")
    if not 0.0 < power < 1.0:
        raise PowerCalculationError(f"power must be in (0, 1); got {power}")
    return float(stats.norm.ppf(1 - alpha / 2)), float(stats.norm.ppf(power))


def continuous_outcome_power(effect_size: float, alpha: float, power: float) -> PowerResult:
    """Two-sample t-test power calculation.

    n_per_arm = 2 * ((z_alpha/2 + z_beta) / d)^2

    Args:
        effect_size: Cohen's d (standardized mean difference). Must be non-zero.
        alpha: Type-I error rate.
        power: Target power (1 - beta).
    """
    if effect_size == 0:
        raise PowerCalculationError("effect_size must be non-zero for power calculation")
    z_alpha, z_beta = _z_scores(alpha, power)
    n_per_arm = int(np.ceil(2 * ((z_alpha + z_beta) / abs(effect_size)) ** 2))
    return PowerResult(
        sample_size=n_per_arm * 2,
        sample_size_per_arm=n_per_arm,
        mde=abs(effect_size),
        analysis_type="two_sample_t_test",
        effect_size_type="cohens_d",
        assumptions=[
            "Equal variance between groups",
            "Normal distribution of outcome",
            "Independent observations",
        ],
    )


def binary_outcome_power(
    effect_size: float,
    alpha: float,
    power: float,
    baseline_rate: float,
) -> PowerResult:
    """Two-proportions z-test power calculation.

    effect_size is interpreted as relative change applied to baseline_rate:
        p2 = baseline_rate * (1 + effect_size)
    so effect_size=0.10 with baseline_rate=0.30 means treatment proportion 0.33.

    Args:
        effect_size: Relative effect (e.g., 0.10 = 10% relative increase).
        alpha: Type-I error rate.
        power: Target power.
        baseline_rate: Control proportion (p1). Must be in (0, 1).
    """
    if not 0.0 < baseline_rate < 1.0:
        raise PowerCalculationError(f"baseline_rate must be in (0, 1); got {baseline_rate}")
    p1 = baseline_rate
    p2 = p1 + effect_size * p1
    if not 0.0 < p2 < 1.0:
        raise PowerCalculationError(
            f"Treatment rate (p2={p2}) out of (0,1); check effect_size/baseline_rate"
        )

    z_alpha, z_beta = _z_scores(alpha, power)
    p_bar = (p1 + p2) / 2
    diff = abs(p2 - p1)
    if diff < 1e-9:
        raise PowerCalculationError("Effect size produces zero risk difference")
    n_per_arm = int(np.ceil(2 * p_bar * (1 - p_bar) * ((z_alpha + z_beta) / diff) ** 2))
    return PowerResult(
        sample_size=n_per_arm * 2,
        sample_size_per_arm=n_per_arm,
        mde=diff,
        analysis_type="two_proportions_z_test",
        effect_size_type="rate_ratio",
        assumptions=[
            "Independent observations",
            "Large sample approximation valid",
            f"Baseline rate: {p1:.3f}",
        ],
        extra={"baseline_rate": p1, "expected_treatment_rate": p2},
    )


def cluster_rct_power(
    effect_size: float,
    alpha: float,
    power: float,
    icc: float,
    cluster_size: int,
) -> PowerResult:
    """Cluster-RCT power with design-effect adjustment.

    Design effect = 1 + (cluster_size - 1) * ICC
    n_adjusted = n_base * design_effect

    Args:
        effect_size: Cohen's d.
        alpha: Type-I error rate.
        power: Target power.
        icc: Intra-cluster correlation in [0, 1).
        cluster_size: Average cluster size (>= 1).
    """
    if not 0.0 <= icc < 1.0:
        raise PowerCalculationError(f"icc must be in [0, 1); got {icc}")
    if cluster_size < 1:
        raise PowerCalculationError(f"cluster_size must be >= 1; got {cluster_size}")

    base = continuous_outcome_power(effect_size, alpha, power)
    design_effect = 1 + (cluster_size - 1) * icc
    adjusted_n = int(np.ceil(base.sample_size * design_effect))
    n_clusters = int(np.ceil(adjusted_n / cluster_size))

    return PowerResult(
        sample_size=adjusted_n,
        sample_size_per_arm=adjusted_n // 2,
        mde=abs(effect_size),
        analysis_type="cluster_rct_adjusted",
        effect_size_type="cohens_d",
        assumptions=[
            f"Intra-cluster correlation (ICC): {icc}",
            f"Average cluster size: {cluster_size}",
            f"Design effect: {design_effect:.2f}",
            "Exchangeable correlation structure within clusters",
        ],
        extra={
            "n_clusters_total": n_clusters,
            "n_clusters_per_arm": max(1, n_clusters // 2),
            "cluster_size": cluster_size,
            "icc": icc,
            "design_effect": design_effect,
            "base_sample_size": base.sample_size,
        },
    )


def time_to_event_power(
    hazard_ratio: float,
    alpha: float,
    power: float,
    event_rate: float,
) -> PowerResult:
    """Log-rank test power via Schoenfeld formula.

    required_events = 4 * (z_alpha/2 + z_beta)^2 / (log HR)^2
    total_n = required_events / event_rate

    Args:
        hazard_ratio: HR treatment vs. control. Must be > 0 and != 1.
        alpha: Type-I error rate.
        power: Target power.
        event_rate: Expected event rate over the observation window in (0, 1].
    """
    if hazard_ratio <= 0:
        raise PowerCalculationError(f"hazard_ratio must be > 0; got {hazard_ratio}")
    if not 0.0 < event_rate <= 1.0:
        raise PowerCalculationError(f"event_rate must be in (0, 1]; got {event_rate}")
    log_hr = float(np.log(hazard_ratio))
    if abs(log_hr) < 1e-3:
        raise PowerCalculationError(
            f"hazard_ratio too close to 1.0 ({hazard_ratio}); no detectable effect"
        )
    z_alpha, z_beta = _z_scores(alpha, power)
    required_events = int(np.ceil(4 * ((z_alpha + z_beta) / log_hr) ** 2))
    total_n = int(np.ceil(required_events / event_rate))
    return PowerResult(
        sample_size=total_n,
        sample_size_per_arm=total_n // 2,
        mde=hazard_ratio,
        analysis_type="log_rank_test",
        effect_size_type="hazard_ratio",
        assumptions=[
            "Proportional hazards assumption",
            f"Expected event rate: {event_rate:.2%}",
            "Exponential survival distribution",
        ],
        extra={
            "required_events": required_events,
            "expected_event_rate": event_rate,
            "hazard_ratio": hazard_ratio,
        },
    )


def mde_for_sample_size(
    n: int,
    alpha: float,
    power: float,
    outcome_type: OutcomeType,
    *,
    baseline_rate: float | None = None,
    event_rate: float | None = None,
) -> float:
    """Reverse power calculation: minimum detectable effect given n.

    Inverts the forward formula. Implements MDE Strategy A from the data-sufficiency
    design (answers "given my data, what effect can I detect?").

    Args:
        n: Total sample size across both arms.
        alpha: Type-I error rate.
        power: Target power.
        outcome_type: "continuous", "binary", or "time_to_event".
        baseline_rate: Required when outcome_type="binary".
        event_rate: Required when outcome_type="time_to_event".

    Returns:
        - "continuous": Cohen's d
        - "binary": absolute risk difference
        - "time_to_event": hazard ratio (returned as ratio, not log-HR)
    """
    if n < 2:
        raise PowerCalculationError(f"n must be >= 2 for reverse MDE calculation; got {n}")
    z_alpha, z_beta = _z_scores(alpha, power)
    n_per_arm = n / 2

    if outcome_type == "continuous":
        # d = (z_alpha + z_beta) * sqrt(2 / n_per_arm)
        return float((z_alpha + z_beta) * np.sqrt(2 / n_per_arm))

    if outcome_type == "binary":
        if baseline_rate is None or not 0.0 < baseline_rate < 1.0:
            raise PowerCalculationError(
                f"binary MDE requires baseline_rate in (0, 1); got {baseline_rate}"
            )
        # Solve: diff = (z_alpha + z_beta) * sqrt(2 * p_bar * (1 - p_bar) / n_per_arm)
        # Approximate p_bar by baseline_rate (small-MDE limit)
        variance = baseline_rate * (1 - baseline_rate)
        return float((z_alpha + z_beta) * np.sqrt(2 * variance / n_per_arm))

    if outcome_type == "time_to_event":
        if event_rate is None or not 0.0 < event_rate <= 1.0:
            raise PowerCalculationError(
                f"time_to_event MDE requires event_rate in (0, 1]; got {event_rate}"
            )
        # required_events = 4 * (z_alpha + z_beta)^2 / (log HR)^2
        # => |log HR| = 2 * (z_alpha + z_beta) / sqrt(required_events)
        required_events = n * event_rate
        if required_events < 1:
            raise PowerCalculationError(
                f"Implied required_events ({required_events}) too small for HR estimation"
            )
        log_hr_abs = 2 * (z_alpha + z_beta) / np.sqrt(required_events)
        return float(np.exp(-log_hr_abs))  # HR < 1 (protective); symmetric for HR > 1

    raise PowerCalculationError(f"Unknown outcome_type: {outcome_type}")


def sensitivity_grid(
    n: int,
    alpha: float,
    power: float,
    outcome_type: OutcomeType,
    candidates: list[float],
    *,
    baseline_rate: float | None = None,
    event_rate: float | None = None,
) -> dict[str, Any]:
    """Sensitivity grid: which candidate effect sizes are detectable at given n.

    Implements MDE Strategy C from the design (presents a grid so operator picks
    the threshold rather than the system).

    Args:
        n: Available sample size.
        alpha: Type-I error rate.
        power: Target power.
        outcome_type: Outcome family.
        candidates: Candidate effect sizes to check (units depend on outcome_type).
        baseline_rate: Required when outcome_type="binary".
        event_rate: Required when outcome_type="time_to_event".

    Returns:
        Dict with "detectable_mde_at_n" and "grid" (per-candidate detectability + required n).
    """
    detectable_mde = mde_for_sample_size(
        n, alpha, power, outcome_type, baseline_rate=baseline_rate, event_rate=event_rate
    )

    grid: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            if outcome_type == "continuous":
                req = continuous_outcome_power(candidate, alpha, power)
            elif outcome_type == "binary":
                if baseline_rate is None:
                    raise PowerCalculationError("baseline_rate required for binary")
                req = binary_outcome_power(candidate, alpha, power, baseline_rate)
            elif outcome_type == "time_to_event":
                if event_rate is None:
                    raise PowerCalculationError("event_rate required for time_to_event")
                req = time_to_event_power(candidate, alpha, power, event_rate)
            else:
                raise PowerCalculationError(f"Unknown outcome_type: {outcome_type}")
            grid.append(
                {
                    "candidate_effect": candidate,
                    "detectable_at_current_n": n >= req.sample_size,
                    "required_n": req.sample_size,
                }
            )
        except PowerCalculationError as e:
            grid.append(
                {
                    "candidate_effect": candidate,
                    "detectable_at_current_n": False,
                    "required_n": None,
                    "error": str(e),
                }
            )

    return {
        "detectable_mde_at_n": detectable_mde,
        "grid": grid,
        "n": n,
        "alpha": alpha,
        "power": power,
        "outcome_type": outcome_type,
    }


def sensitivity_variations(
    effect_size: float,
    alpha: float,
    power: float,
    base_n: int,
    outcome_type: OutcomeType = "continuous",
    *,
    baseline_rate: float | None = None,
    event_rate: float | None = None,
) -> dict[str, Any]:
    """Sensitivity analysis: effect size and power variations around a base n.

    Preserves the structure produced by the original
    PowerAnalysisNode._run_sensitivity_analysis() so existing consumers see the
    same shape.
    """

    def _forward(effect: float, p: float) -> PowerResult:
        if outcome_type == "continuous":
            return continuous_outcome_power(effect, alpha, p)
        if outcome_type == "binary":
            if baseline_rate is None:
                raise PowerCalculationError("baseline_rate required for binary")
            return binary_outcome_power(effect, alpha, p, baseline_rate)
        if outcome_type == "time_to_event":
            if event_rate is None:
                raise PowerCalculationError("event_rate required for time_to_event")
            return time_to_event_power(effect, alpha, p, event_rate)
        raise PowerCalculationError(f"Unknown outcome_type: {outcome_type}")

    effect_variations: dict[str, dict[str, Any]] = {}
    for multiplier in (0.8, 0.9, 1.1, 1.2):
        varied = effect_size * multiplier
        try:
            res = _forward(varied, power)
            effect_variations[f"{multiplier:.1f}x"] = {
                "effect_size": varied,
                "sample_size": res.sample_size,
                "change_from_base": res.sample_size - base_n,
            }
        except PowerCalculationError as e:
            effect_variations[f"{multiplier:.1f}x"] = {
                "effect_size": varied,
                "sample_size": None,
                "change_from_base": None,
                "error": str(e),
            }

    power_variations: dict[str, dict[str, Any]] = {}
    for p_level in (0.70, 0.85, 0.90):
        try:
            res = _forward(effect_size, p_level)
            power_variations[f"{p_level:.0%}"] = {
                "power": p_level,
                "sample_size": res.sample_size,
                "change_from_base": res.sample_size - base_n,
            }
        except PowerCalculationError as e:
            power_variations[f"{p_level:.0%}"] = {
                "power": p_level,
                "sample_size": None,
                "change_from_base": None,
                "error": str(e),
            }

    return {
        "effect_size_variations": effect_variations,
        "power_variations": power_variations,
    }
