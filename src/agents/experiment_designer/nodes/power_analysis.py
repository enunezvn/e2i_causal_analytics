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
from typing import Any, Optional

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

#: Beyond ANY realistic planning horizon (10 years). Not "long" -- absurd.
#:
#: This was 365 for one round, on the reasoning that a ``7 <= duration <= 365``
#: range already exists at dspy_integration.py:112 and could be reused as an
#: output-path validator. That reasoning was wrong, and the correction matters
#: more than the constant: the GEPA term SCORES an optimizer, where preferring
#: short designs is a legitimate bias, while this site makes an ASSERTION ABOUT
#: REALITY, where a 16-month oncology trial is routine. Measured on this node,
#: effect 0.15 on a 0.30 baseline at 50/week is n=3,388 over 476 days -- a
#: normal study that the 365 bound branded "not executable as specified".
#:
#: A FALSE warning is worse than a missing one: the missing warning leaves the
#: old behaviour, the false one destroys trust in every warning beside it. So
#: the unconditional bound is set where no honest schedule reaches, and a design
#: that is merely long is flagged ONLY against a limit the caller actually
#: stated (``_stated_max_duration_days``).
_IMPLAUSIBLE_DURATION_DAYS = 3650


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


def _append_unique(existing: list[str], additions: list[str]) -> list[str]:
    """Append only messages not already present, preserving order."""
    out = list(existing)
    for message in additions:
        if message not in out:
            out.append(message)
    return out


def _replace_feasibility_verdict(state: ExperimentDesignState, verdict: list[str]) -> None:
    """Publish THIS run's feasibility verdict, retracting the previous one.

    The structured field was already replaced each run, but the copy pushed onto
    the generic ``warnings`` channel was only ever appended -- so a redesign
    that FIXED the design still shipped iteration 1's "94,115 days" sentence in
    the user-visible warnings, and the error path could show "not assessed"
    beside an obsolete verdict for a design that was never assessed.

    Retracts by exact string (the previous verdict is known verbatim), so
    unrelated warnings are untouched.
    """
    superseded = set(state.get("feasibility_warnings") or [])
    remaining = [w for w in state.get("warnings", []) if w not in superseded]
    state["feasibility_warnings"] = verdict
    state["warnings"] = _append_unique(remaining, verdict)


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

    @staticmethod
    def _stated_max_duration_days(constraints: dict[str, Any]) -> Optional[float]:
        """The caller's own duration limit, in days, if they stated one.

        ``constraints`` is a free-form dict and callers state a timeline in THREE
        shapes, two of which predate this code:

        * ``timeline_weeks: 12`` -- the contract's own worked example
        * ``timeline: {"max_duration_days": 90}`` -- tier0_output_mapper.py
        * ``max_duration_days: 180`` -- added by #1639 as an explicit affordance
          and declared in ``validate_constraints`` + the Tier 3 contract, so it
          is a documented key rather than a private convention

        (An earlier round also read ``timeline: {"weeks": 12}``. That shape has
        no caller, no contract entry and no test but my own -- invented, so
        removed. Reading a key nobody writes is not free: it reads as evidence
        the shape is supported.)

        Reading only the last of those (as this first did) means a caller who
        states a limit in a documented shape is told nothing.

        **Free-text ``timeline: "3 months"`` is deliberately NOT parsed**, and
        that decision is the most important thing in this function.

        Five rounds of review went into trying, and every round produced a new
        counter-example in one direction or the other: first-match took a ramp
        phase for the limit; MAX broke on additive composites ("12 months
        recruitment plus 6 months follow-up" is 18, not 12); single-duration
        read "at least 3 months" as a cap; scoping cap words to the duration's
        neighbourhood still mis-read "not under 3 months" as a cap, made
        "no more than" dead (the floor list's "more than" matched first),
        dropped the real cap in "at least 2 and at most 6 months", and ignored
        postposed forms like "3 months maximum".

        Each fix traded one wrong reading for another because the underlying
        task -- inferring a bound's DIRECTION from prose with substring lists --
        is not something this code can do reliably. A false warning is the worst
        outcome available here: it tells a caller their design is unrunnable
        against a limit they never set, and it discredits every real warning
        beside it.

        Free text is also not load-bearing. The defect in #1639 stated no
        timeline at all; it is caught by the plausibility bound. So a stated
        limit is honoured only in the shapes that state it unambiguously, and
        prose yields "no stated limit" -- a missed warning, the direction this
        design accepts.
        """
        timeline = constraints.get("timeline")

        def _num(value: Any) -> Optional[float]:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return float(value)

        direct = _num(constraints.get("max_duration_days"))
        if direct is not None:
            return direct
        if isinstance(timeline, dict):
            nested = _num(timeline.get("max_duration_days"))
            if nested is not None:
                return nested
        weeks_key = _num(constraints.get("timeline_weeks"))
        if weeks_key is not None:
            return weeks_key * 7
        return None

    def _assess_feasibility(
        self,
        duration_days: int,
        sample_size: int,
        accrual_rate: Any,
        constraints: dict[str, Any],
    ) -> list[str]:
        """Reasons this design cannot be run as specified (#1639).

        Two tiers, because they are two different claims:

        * a **stated** limit exceeded is a FACT about the caller's own
          constraint -- flagged however short the overrun;
        * absent any stated limit, only a duration past
          ``_IMPLAUSIBLE_DURATION_DAYS`` is flagged, because without a limit
          "too long" is the caller's judgement to make, not ours.

        Flags rather than clamps: eval turn 3.6's 94,115 days was arithmetically
        exact (``ceil(672206/50)*7 == 94115``), and a silently capped duration
        would be a fabricated number where the real one is merely unwelcome.
        """
        warnings: list[str] = []
        years = duration_days / 365.25
        stated_max = self._stated_max_duration_days(constraints)

        if stated_max is not None and duration_days > stated_max:
            warnings.append(
                f"Estimated duration {duration_days:,} days ({years:.1f} years) exceeds the "
                f"stated maximum of {stated_max:,.0f} days. The design requires "
                f"n={sample_size:,} at the assumed accrual of {accrual_rate}/week; to fit the "
                f"stated window, either accrual must rise or the effect size the study is "
                f"powered to detect must be larger."
            )
        elif stated_max is None and duration_days > _IMPLAUSIBLE_DURATION_DAYS:
            warnings.append(
                f"Estimated duration {duration_days:,} days ({years:.1f} years) is far beyond "
                f"any realistic planning horizon. The design requires n={sample_size:,} at the "
                f"assumed accrual of {accrual_rate}/week; this is a symptom of an effect too "
                f"small for the study to practically detect, not a schedule."
            )
        return warnings

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
            feasibility_warnings = self._assess_feasibility(
                duration_days, forward.sample_size, accrual_rate, constraints
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
            # Also raised on the generic channel every downstream reader already
            # knows. Deliberate duplication: ``feasibility_warnings`` is the
            # structured field, ``warnings`` is the one consumers look at.
            _replace_feasibility_verdict(state, feasibility_warnings)
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
            # Never leave iteration N's verdict attached to iteration N+1's
            # failed, changed design -- on EITHER channel -- and never leave it
            # empty, which would read as "checked, feasible". A failed
            # assessment is not a clean bill of health.
            _replace_feasibility_verdict(
                state, [f"Feasibility was not assessed \u2014 power analysis failed: {str(e)}"]
            )
            state["warnings"] = _append_unique(
                state.get("warnings", []), [f"Power analysis failed: {str(e)}"]
            )
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
