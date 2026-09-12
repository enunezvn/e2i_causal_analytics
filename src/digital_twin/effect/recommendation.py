"""CI-based three-way pre-screen decision: DEPLOY / REFINE / SKIP."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.digital_twin.effect.errors import EffectDataUnavailable
from src.digital_twin.effect.estimate import EffectEstimate


class Recommendation(str, Enum):
    DEPLOY = "deploy"
    REFINE = "refine"
    SKIP = "skip"


@dataclass
class PolicyThresholds:
    min_effect: float = 0.05  # calibrated (Task 9), not the old fake 0.05
    power: float = 0.80
    alpha: float = 0.05


class RecommendationPolicy:
    """DEPLOY / REFINE / SKIP from the effect's interval against ``min_effect``.

    It does not size the experiment (#2015): it used to return a two-proportion n fed the
    twins' mean ``baseline_propensity`` — a heuristic TREATMENT propensity — as the control
    conversion proportion, while the outcome is a continuous rate. Sizing is
    :func:`experiment_size`, shared by the Digital Twin page and the chat simulator.
    """

    def __init__(self, thresholds: PolicyThresholds) -> None:
        self.t = thresholds

    def decide(self, estimate: EffectEstimate) -> tuple[Recommendation, str]:
        lo, hi, m = estimate.ate_ci_lower, estimate.ate_ci_upper, self.t.min_effect
        if lo > m:
            return (
                Recommendation.DEPLOY,
                f"CI lower bound {lo:.3f} exceeds min effect {m:.3f}.",
            )
        if hi < m:
            return (
                Recommendation.SKIP,
                f"CI upper bound {hi:.3f} is below min effect {m:.3f}.",
            )
        return (
            Recommendation.REFINE,
            f"CI [{lo:.3f}, {hi:.3f}] straddles min effect {m:.3f}; refine or gather more data.",
        )


def experiment_size(
    frame: Any,
    effect: float,
    *,
    regions: Sequence[str] = (),
    thresholds: PolicyThresholds | None = None,
) -> tuple[int | None, str]:
    """Per-arm n for an experiment powered to detect ``effect``, and how it was sized (#2015).

    The one sizing rule for ``POST /digital-twin/simulate`` (via ``SimulationEngine``) and
    the chat ``counterfactual_simulator``, so the two surfaces state the same number for the
    same run. A two-sided, equal-allocation test of a continuous outcome with
    ``power_analysis_lib``, at the policy's power and alpha: Cohen's d = |effect| / the
    outcome SD among the ``frame`` rows in the estimate's comparison arm (at or below the
    median treatment intensity, :func:`control_outcome_sd`), within ``regions`` when given.
    The SD is unadjusted, so the size is conservative next to a covariate-adjusted estimate.

    ``(None, reason)`` when no size can be computed — a zero or non-finite effect, a frame
    without the cohort columns (e.g. the synthetic uplift provider's), an outcome that does
    not vary, or a design under two per arm. There is no fallback formula.
    """
    from src.digital_twin.effect.cohort_causal_estimator import control_outcome_sd
    from src.utils.power_analysis_lib import PowerCalculationError, continuous_outcome_power

    policy = thresholds or PolicyThresholds()
    regions = list(regions)
    scope = f"the targeted regions {regions}" if regions else "the cohort"
    if not math.isfinite(effect) or effect == 0:
        return None, (
            f"recommended_sample_size is not given: the effect is {effect:g}, and no "
            "experiment can be powered to detect a zero effect."
        )
    try:
        sd, n_rows = control_outcome_sd(
            frame.df,
            frame.treatment_var,
            outcome_col=frame.outcome_var,
            confounders=tuple(frame.confounders),
            regions=regions,
        )
    except EffectDataUnavailable as exc:
        return None, f"recommended_sample_size is not given: {exc}"
    if not math.isfinite(sd) or sd <= 0:
        return None, (
            f"recommended_sample_size is not given: {frame.outcome_var} does not vary among "
            f"the {n_rows} comparison-arm rows of {scope}."
        )
    d = abs(effect) / sd
    try:
        per_arm = continuous_outcome_power(d, policy.alpha, policy.power).sample_size_per_arm
    except (PowerCalculationError, ArithmeticError) as exc:
        return None, f"recommended_sample_size is not given: Cohen's d = {d:.3g}: {exc}."
    if per_arm < 2:
        return None, (
            f"recommended_sample_size is not given: Cohen's d = {d:.3g} gives {per_arm} per "
            "arm, below the two per arm a two-arm test needs."
        )
    return per_arm, (
        f"recommended_sample_size = {per_arm} per arm: a two-sided, equal-allocation test at "
        f"power {policy.power:g} and alpha {policy.alpha:g} for Cohen's d = |effect| / SD of "
        f"{frame.outcome_var} ({sd:.4g}, among the {n_rows} rows of {scope} at or below the "
        f"median {frame.treatment_var}) = {d:.3g}. The SD is unadjusted, so this is "
        "conservative next to the covariate-adjusted effect estimate."
    )
