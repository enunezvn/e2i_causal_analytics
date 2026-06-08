"""Golden seed examples for the per-recipient optimizer (Shard 09, option 3).

These are a small, hand-authored supervised set per recipient template field, so
the per-recipient optimizer is runnable + testable WITHOUT each recipient first
emitting training signals (the open design decision in
09-followon-per-recipient-optimizer.md). They are deterministic and static;
swap in self-emitted or shared-pool examples (options 1/2) for stronger
supervision later.

`default_example_provider(agent_name)` returns a callable
(template_field) -> list[dspy.Example] with `.with_inputs(...)` set to the
recipient signature's input fields.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


def _experiment_monitor_seeds() -> Dict[str, List[Any]]:
    """Golden examples keyed by template field for experiment_monitor."""
    try:
        import dspy
    except ImportError:
        return {}

    srm = [
        dspy.Example(
            experiment_name="Kisqali-NE-Q2",
            chi_squared=12.4,
            p_value=0.0004,
            expected_ratio="50/50",
            actual_counts="640/360",
            explanation=(
                "A statistically significant sample ratio mismatch was detected: the "
                "treatment arm received far more units than the 50/50 design specified "
                "(chi-squared=12.4, p=0.0004), which threatens randomization validity."
            ),
            potential_causes=["assignment bug", "differential dropout", "logging gap"],
            recommended_actions=["freeze enrollment", "audit the randomization service"],
        ).with_inputs(
            "experiment_name", "chi_squared", "p_value", "expected_ratio", "actual_counts"
        ),
        dspy.Example(
            experiment_name="Entresto-South-Pilot",
            chi_squared=2.1,
            p_value=0.15,
            expected_ratio="33/33/33",
            actual_counts="120/118/122",
            explanation=(
                "No significant sample ratio mismatch (chi-squared=2.1, p=0.15); arm "
                "counts are within expected sampling variation for the three-arm design."
            ),
            potential_causes=["none — within normal variation"],
            recommended_actions=["continue monitoring at the next checkpoint"],
        ).with_inputs(
            "experiment_name", "chi_squared", "p_value", "expected_ratio", "actual_counts"
        ),
        dspy.Example(
            experiment_name="Cosentyx-West-AB",
            chi_squared=7.8,
            p_value=0.005,
            expected_ratio="50/50",
            actual_counts="430/570",
            explanation=(
                "A significant mismatch favors the control arm (chi-squared=7.8, "
                "p=0.005). Investigate before trusting downstream lift estimates."
            ),
            potential_causes=["bucketing skew", "bot traffic in one arm"],
            recommended_actions=["quarantine results", "re-hash assignment keys"],
        ).with_inputs(
            "experiment_name", "chi_squared", "p_value", "expected_ratio", "actual_counts"
        ),
    ]

    summary = [
        dspy.Example(
            experiments_checked=12,
            healthy_count=9,
            warning_count=2,
            critical_count=1,
            issue_types="SRM, enrollment lag",
            summary=(
                "Monitored 12 experiments: 9 healthy, 2 with warnings, 1 critical. The "
                "critical case is an SRM breach; two warnings are enrollment lag."
            ),
            priority_actions=["triage the SRM-breached experiment first"],
            overall_status="action_required",
        ).with_inputs(
            "experiments_checked", "healthy_count", "warning_count", "critical_count", "issue_types"
        ),
        dspy.Example(
            experiments_checked=5,
            healthy_count=5,
            warning_count=0,
            critical_count=0,
            issue_types="none",
            summary="All 5 monitored experiments are healthy; no warnings or critical issues.",
            priority_actions=["no action required"],
            overall_status="healthy",
        ).with_inputs(
            "experiments_checked", "healthy_count", "warning_count", "critical_count", "issue_types"
        ),
    ]

    alert = [
        dspy.Example(
            experiment_name="Kisqali-NE-Q2",
            alert_type="SRM",
            severity="critical",
            details="chi-squared=12.4, p=0.0004, observed 640/360 vs 50/50",
            message=(
                "CRITICAL SRM on Kisqali-NE-Q2: arm split 640/360 vs the 50/50 design "
                "(p=0.0004). Randomization validity is compromised."
            ),
            recommended_action="Freeze enrollment and audit the assignment service now.",
            urgency_level="immediate",
        ).with_inputs("experiment_name", "alert_type", "severity", "details"),
        dspy.Example(
            experiment_name="Entresto-South-Pilot",
            alert_type="enrollment",
            severity="warning",
            details="enrollment 38% below pace at week 3",
            message=(
                "WARNING on Entresto-South-Pilot: enrollment is 38% behind pace at week 3, "
                "which may delay readout."
            ),
            recommended_action="Review site activation and outreach cadence this week.",
            urgency_level="this_week",
        ).with_inputs("experiment_name", "alert_type", "severity", "details"),
    ]

    return {"srm_template": srm, "summary_template": summary, "alert_template": alert}


_SEEDS_BUILDERS: Dict[str, Callable[[], Dict[str, List[Any]]]] = {
    "experiment_monitor": _experiment_monitor_seeds,
}


def default_example_provider(agent_name: str) -> Callable[[str], List[Any]]:
    """Return a provider mapping a template field -> golden seed examples."""
    builder = _SEEDS_BUILDERS.get(agent_name)
    seeds = builder() if builder else {}
    if not seeds:
        logger.info("No golden seeds registered for recipient %s", agent_name)

    def provider(field: str) -> List[Any]:
        return seeds.get(field, [])

    return provider
