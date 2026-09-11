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


# ---------------------------------------------------------------------------
# A design the library cannot compute is refused, not answered as another design
# (codex whole-diff F2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"ratio": 2}, "ratio"),
        ({"ratio": 0.5}, "ratio"),
        ({"ratio": "2:1"}, "ratio"),
        ({"alternative": "larger"}, "alternative"),
        ({"alternative": "smaller"}, "alternative"),
    ],
)
def test_an_allocation_or_sidedness_the_library_cannot_honour_is_refused(kwargs, reason):
    """``**kwargs`` absorbed ``ratio=2`` and returned the equal-allocation answer. Both names
    are statsmodels' power-API arguments, the vocabulary a planner reaches for."""
    with pytest.raises(ToolInputError, match=reason):
        tr.power_calculator(effect_size=0.2, **kwargs)


@pytest.mark.parametrize("kwargs", [{"ratio": 1}, {"ratio": 1.0}, {"alternative": "two-sided"}])
def test_the_design_the_library_computes_may_be_stated(kwargs):
    assert tr.power_calculator(effect_size=0.2, **kwargs).required_n_per_arm == 393


@pytest.fixture
def _fresh_bounded_pool():
    from src.api.dependencies.compute import _reset_limiter_cache_for_tests

    _reset_limiter_cache_for_tests()
    yield
    _reset_limiter_cache_for_tests()


async def test_an_unequal_allocation_fails_the_planned_step_once(_fresh_bounded_pool):
    from src.agents.tool_composer.executor import PlanExecutor
    from src.agents.tool_composer.models.composition_models import ExecutionStatus
    from tests.unit.test_agents.test_tool_composer.test_sync_tool_bounded_timeout_1592 import (
        _registry_with,
        _single_step_plan,
    )

    calls = []

    def counted_power_calculator(**kwargs):
        calls.append(1)
        return tr.power_calculator(**kwargs)

    executor = PlanExecutor(
        tool_registry=_registry_with("power_calculator", counted_power_calculator),
        enable_caching=False,
        max_retries=2,
        backoff_base_delay=0.01,
    )
    trace = await executor.execute(
        _single_step_plan("power_calculator", {"effect_size": 0.2, "ratio": 2}), context={}
    )
    result = trace.get_result("step_1")
    assert result.status == ExecutionStatus.FAILED
    assert result.output.result is None
    assert "ratio" in (result.output.error or "")
    assert len(calls) == 1


@pytest.mark.parametrize("effect_size", [4.0, -4.0, 1e200, 1e-200])
def test_an_unusable_or_unrepresentable_design_is_refused(effect_size):
    """codex whole-diff #5: 1 or 0 per arm is not a design, and an overflow must not escape
    as an OverflowError the executor retries."""
    with pytest.raises(ToolInputError):
        tr.power_calculator(effect_size=effect_size)


async def test_an_overflowing_design_fails_the_planned_step_once(_fresh_bounded_pool):
    from src.agents.tool_composer.executor import PlanExecutor
    from src.agents.tool_composer.models.composition_models import ExecutionStatus
    from tests.unit.test_agents.test_tool_composer.test_sync_tool_bounded_timeout_1592 import (
        _registry_with,
        _single_step_plan,
    )

    calls = []

    def counted_power_calculator(**kwargs):
        calls.append(1)
        return tr.power_calculator(**kwargs)

    executor = PlanExecutor(
        tool_registry=_registry_with("power_calculator", counted_power_calculator),
        enable_caching=False,
        max_retries=2,
        backoff_base_delay=0.01,
    )
    trace = await executor.execute(
        _single_step_plan("power_calculator", {"effect_size": 1e-200}), context={}
    )
    result = trace.get_result("step_1")
    assert result.status == ExecutionStatus.FAILED and result.output.result is None
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"effect_size": 98, "outcome_type": "binary", "baseline_rate": 0.01, "power": 0.1},
        {"effect_size": 1e-153, "design": "cluster", "icc": 0.5, "cluster_size": 100},
        {"effect_size": 1, "outcome_type": "binary", "baseline_rate": 0.1, "alpha": 1e-200},
        {"effect_size": 100, "outcome_type": "time_to_event", "event_rate": 1.0},
        {"effect_size": 1e100, "outcome_type": "time_to_event", "event_rate": 1.0},
    ],
)
def test_the_remaining_unusable_designs_are_refused(kwargs):
    """codex whole-diff #6: 1 per arm for a binary design at low power, and overflows in the
    cluster inflation and at an extreme alpha."""
    with pytest.raises(ToolInputError):
        tr.power_calculator(**kwargs)


async def test_a_cluster_design_with_one_cluster_per_arm_fails_the_planned_step_once(
    _fresh_bounded_pool,
):
    """codex whole-diff #10: d=2, ICC 0.05, 20 per cluster needs 8 subjects per arm — one
    cluster per arm, which confounds treatment with cluster. Two clusters per arm are the
    minimum for any between-cluster comparison."""
    from src.agents.tool_composer.executor import PlanExecutor
    from src.agents.tool_composer.models.composition_models import ExecutionStatus
    from tests.unit.test_agents.test_tool_composer.test_sync_tool_bounded_timeout_1592 import (
        _registry_with,
        _single_step_plan,
    )

    with pytest.raises(ToolInputError, match="cluster"):
        tr.power_calculator(effect_size=2, design="cluster", icc=0.05, cluster_size=20)
    two_per_arm = tr.power_calculator(effect_size=2, design="cluster", icc=0.05, cluster_size=4)
    assert two_per_arm.design_details["n_clusters_per_arm"] >= 2

    calls = []

    def counted_power_calculator(**kwargs):
        calls.append(1)
        return tr.power_calculator(**kwargs)

    executor = PlanExecutor(
        tool_registry=_registry_with("power_calculator", counted_power_calculator),
        enable_caching=False,
        max_retries=2,
        backoff_base_delay=0.01,
    )
    trace = await executor.execute(
        _single_step_plan(
            "power_calculator",
            {"effect_size": 2, "design": "cluster", "icc": 0.05, "cluster_size": 20},
        ),
        context={},
    )
    result = trace.get_result("step_1")
    assert result.status == ExecutionStatus.FAILED and result.output.result is None
    assert len(calls) == 1
