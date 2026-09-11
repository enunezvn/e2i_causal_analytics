"""``power_calculator`` sizes the experiment with the shared power library (#2015).

The tool computed ``n = 16 * (1.96 + 0.84) ** 2 / d**2`` — 3,135 at d=0.2 against 393 per
arm from ``src/utils/power_analysis_lib.continuous_outcome_power`` and statsmodels — and
ignored ``alpha`` and ``power``, echoing the requested power as ``actual_power``. A leader
asking "how many patients do we need?" got a study about four times too large.

The tool now delegates to the library the experiment-designer agent's power node uses, for
every design that library supports, reports per-arm and total n with the analysis type and
its assumptions, and refuses inputs no sample size can be computed from.
"""

from __future__ import annotations

import math

import pytest
from statsmodels.stats.power import TTestIndPower

from src.agents.tool_composer import tool_registrations as tr
from src.agents.tool_composer.errors import ToolInputError
from src.utils import power_analysis_lib as lib


@pytest.mark.parametrize(("d", "per_arm"), [(0.2, 393), (0.5, 63)])
def test_continuous_sample_size_is_the_library_two_sample_calculation(d, per_arm):
    result = tr.power_calculator(effect_size=d)
    assert result.required_n_per_arm == per_arm
    assert result.required_n_total == 2 * per_arm
    assert result.analysis_type == "two_sample_t_test"
    assert result.minimum_detectable_effect_scale == "cohens_d"
    # Independent check: statsmodels' t-test solution, which the library's normal
    # approximation undershoots by less than one subject per arm.
    exact = TTestIndPower().solve_power(effect_size=d, alpha=0.05, power=0.8, ratio=1.0)
    assert abs(result.required_n_per_arm - math.ceil(exact)) <= 1


def test_alpha_and_power_change_the_sample_size():
    default = tr.power_calculator(effect_size=0.2)
    strict = tr.power_calculator(effect_size=0.2, alpha=0.01, power=0.9)
    expected = lib.continuous_outcome_power(0.2, 0.01, 0.9)
    assert strict.required_n_per_arm == expected.sample_size_per_arm
    assert strict.required_n_per_arm > default.required_n_per_arm
    assert (strict.alpha, strict.power) == (0.01, 0.9)


def test_binary_design_uses_the_baseline_rate():
    result = tr.power_calculator(effect_size=0.10, outcome_type="binary", baseline_rate=0.30)
    expected = lib.binary_outcome_power(0.10, 0.05, 0.8, 0.30)
    assert result.required_n_per_arm == expected.sample_size_per_arm
    assert result.required_n_total == expected.sample_size
    assert result.analysis_type == "two_proportions_z_test"
    # The MDE is an absolute risk difference while the input effect is relative (#1639).
    assert result.minimum_detectable_effect == pytest.approx(0.03)
    assert result.minimum_detectable_effect_scale == "absolute_risk_difference"
    assert result.design_details["expected_treatment_rate"] == pytest.approx(0.33)


def test_cluster_design_applies_the_design_effect_in_whole_clusters():
    """codex whole-diff F1: the tool reported 683 in total but 17 clusters of 20 per arm
    (680 subjects). Checked against the formula, not the library's own figures."""
    result = tr.power_calculator(effect_size=0.3, design="cluster", icc=0.05, cluster_size=20)
    individual_total = 2 * math.ceil(2 * (2.8015852 / 0.3) ** 2)  # z(0.975) + z(0.8)
    required = math.ceil(individual_total * 1.95)
    clusters_per_arm = result.design_details["n_clusters_per_arm"]
    assert (individual_total, required) == (350, 683)
    assert result.required_n_total == 2 * result.required_n_per_arm >= required
    assert clusters_per_arm * 2 * 20 >= result.required_n_total
    assert (result.required_n_per_arm, clusters_per_arm) == (342, 18)
    assert result.design_details["n_clusters_total"] == 36
    assert result.design_details["design_effect"] == pytest.approx(1.95)


def test_time_to_event_design_uses_the_event_rate():
    result = tr.power_calculator(
        effect_size=0.7, outcome_type="time_to_event", event_rate=0.4, power=0.9
    )
    expected = lib.time_to_event_power(0.7, 0.05, 0.9, 0.4)
    assert result.required_n_total == expected.sample_size
    assert result.design_details["required_events"] == expected.extra["required_events"]
    assert result.minimum_detectable_effect_scale == "hazard_ratio"


def test_the_output_reports_its_assumptions():
    result = tr.power_calculator(effect_size=0.2)
    assert "Equal variance between groups" in result.assumptions
    assert any("alpha=0.05" in a and "power=0.8" in a for a in result.assumptions)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"effect_size": 0.0}, "effect_size"),
        ({"effect_size": None}, "effect_size"),
        ({"effect_size": float("nan")}, "effect_size"),
        ({"effect_size": "0.2"}, "effect_size"),
        ({"effect_size": True}, "effect_size"),
        ({"effect_size": 0.2, "alpha": 1.5}, "alpha"),
        ({"effect_size": 0.2, "power": 0.0}, "power"),
        ({"effect_size": 0.1, "outcome_type": "binary"}, "baseline_rate"),
        ({"effect_size": 0.1, "outcome_type": "binary", "baseline_rate": 1.2}, "baseline_rate"),
        ({"effect_size": 0.7, "outcome_type": "time_to_event"}, "event_rate"),
        ({"effect_size": 1.0, "outcome_type": "time_to_event", "event_rate": 0.4}, "hazard_ratio"),
        ({"effect_size": 0.3, "design": "cluster"}, "icc"),
        ({"effect_size": 0.3, "design": "cluster", "icc": 0.05}, "cluster_size"),
        (
            {"effect_size": 0.3, "design": "cluster", "icc": 0.05, "cluster_size": 2.5},
            "cluster_size",
        ),
        (
            {
                "effect_size": 0.1,
                "design": "cluster",
                "outcome_type": "binary",
                "baseline_rate": 0.3,
            },
            "cluster",
        ),
        ({"effect_size": 0.2, "outcome_type": "ordinal"}, "outcome_type"),
        ({"effect_size": 0.2, "design": "crossover"}, "design"),
        # A supplied parameter the chosen design does not use would be silently ignored.
        ({"effect_size": 0.2, "baseline_rate": 0.3}, "baseline_rate"),
        ({"effect_size": 0.2, "icc": 0.05}, "icc"),
    ],
)
def test_inputs_no_sample_size_can_be_computed_from_are_refused(kwargs, reason):
    with pytest.raises(ToolInputError, match=reason):
        tr.power_calculator(**kwargs)


def test_an_explicit_null_alpha_or_power_is_the_documented_default():
    assert tr.power_calculator(effect_size=0.2, alpha=None, power=None).required_n_per_arm == 393
