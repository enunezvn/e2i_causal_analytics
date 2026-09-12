"""The caveat experiment's analysis, on constructed tables with known answers (spec §7.2).

The experiment asks one question: does a reliability caveat in the planning prompt change which
tool the LLM picks? The analysis is what turns its runs into an answer, so it is tested here
against tables whose result is known in advance — the run itself is gated on the owner's
authorization (O2) and is NOT performed by this suite.

The pass rule is deliberately conjunctive: a real effect AND significance AND three
observed-count guards. Each guard is tested failing on its own, because a rule whose guards can
be skipped is not a rule.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from scipy import stats

from scripts.benchmarks.tool_composer.reliability_caveat_experiment import (
    mcnemar_one_sided,
    pass_rule,
    summarize_pairs,
)


def _pair(
    *,
    picked_a: bool,
    picked_b: bool,
    invalid_a: bool = False,
    invalid_b: bool = False,
    total_failure_a: bool = False,
    total_failure_b: bool = False,
    succeeded_a: int = 3,
    succeeded_b: int = 3,
) -> Dict[str, Any]:
    """One frozen item, run in both arms. A = flag off, B = flag on."""
    return {
        "picked_target": {"A": picked_a, "B": picked_b},
        "invalid": {"A": invalid_a, "B": invalid_b},
        "total_failure": {"A": total_failure_a, "B": total_failure_b},
        "succeeded_steps": {"A": succeeded_a, "B": succeeded_b},
    }


def _pairs(b: int, c: int, concordant: int = 0, **over: Any) -> List[Dict[str, Any]]:
    """b items picked the target in A only, c in B only, plus concordant items."""
    items = [_pair(picked_a=True, picked_b=False, **over) for _ in range(b)]
    items += [_pair(picked_a=False, picked_b=True, **over) for _ in range(c)]
    items += [_pair(picked_a=True, picked_b=True, **over) for _ in range(concordant)]
    return items


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("b, c", [(12, 2), (20, 5), (3, 3), (0, 0), (7, 0)])
def test_mcnemar_one_sided_equals_the_exact_binomial(b, c):
    expected = stats.binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c else 1.0

    assert mcnemar_one_sided(b, c) == pytest.approx(expected)


def test_no_discordant_pairs_is_no_evidence_not_a_significant_result():
    assert mcnemar_one_sided(0, 0) == 1.0


# ---------------------------------------------------------------------------
# The pass rule (spec §7.2): effect >= 0.30, p < 0.05, and three guards
# ---------------------------------------------------------------------------


def test_a_clear_effect_with_every_guard_held_passes():
    result = pass_rule(_pairs(b=14, c=1, concordant=5))

    assert result["passed"] is True
    assert result["effect"] == pytest.approx((14 - 1) / 20)
    assert result["p_value"] < 0.05


def test_an_effect_below_the_threshold_fails_even_when_significant():
    # b - c over K is 0.25: significant, but under the 0.30 the rule requires.
    result = pass_rule(_pairs(b=11, c=1, concordant=28))

    assert result["passed"] is False
    assert result["effect"] < 0.30
    assert "effect" in result["failed"]


def test_an_effect_above_the_threshold_fails_without_significance():
    result = pass_rule(_pairs(b=2, c=0, concordant=3))

    assert result["passed"] is False
    assert result["p_value"] >= 0.05
    assert "significance" in result["failed"]


@pytest.mark.parametrize(
    "over, guard",
    [
        ({"invalid_b": True}, "invalid"),
        ({"total_failure_b": True}, "total_failure"),
        ({"succeeded_a": 4, "succeeded_b": 1}, "median_succeeded_steps"),
    ],
)
def test_each_observed_count_guard_fails_the_rule_on_its_own(over, guard):
    """B must not be observed worse on any guard, however strong the primary effect."""
    result = pass_rule(_pairs(b=14, c=1, concordant=5, **over))

    assert result["passed"] is False
    assert guard in result["failed"], result["failed"]


def test_the_guards_are_not_significance_tests():
    """Equal counts pass: these are observed-worse gates, not non-inferiority claims."""
    result = pass_rule(_pairs(b=14, c=1, concordant=5, invalid_a=True, invalid_b=True))

    assert result["passed"] is True


# ---------------------------------------------------------------------------
# Every frozen item stays in the denominator
# ---------------------------------------------------------------------------


def test_an_item_whose_planner_call_errored_counts_as_invalid_in_that_arm():
    pairs = _pairs(b=14, c=1, concordant=5)
    pairs.append(_pair(picked_a=False, picked_b=False, invalid_b=True))

    summary = summarize_pairs(pairs)

    assert summary["n_items"] == 21
    assert summary["invalid"]["B"] == 1
    assert pass_rule(pairs)["passed"] is False


def test_the_denominator_is_every_item_not_only_the_discordant_ones():
    summary = summarize_pairs(_pairs(b=6, c=0, concordant=14))

    assert summary["n_items"] == 20
    assert summary["discordant"] == {"b": 6, "c": 0}
