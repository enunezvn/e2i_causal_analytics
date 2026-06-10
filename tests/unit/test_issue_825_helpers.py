"""Unit tests for the pure helpers introduced by issue #825 (no DB).

* agent_registry tier text-category <-> numeric mapping (real agent_tier enum).
* SRM uniform expected-allocation derivation (ml_experiments persists no
  per-experiment allocation ratio; equal allocation is the correct SRM null).
"""


def test_tier_number_by_category_matches_real_enum():
    """The category->number map covers exactly the real agent_tier_type enum
    labels plus tier-0 ml_foundation, with the canonical 0-5 numbering."""
    from src.repositories.agent_registry import (
        TIER_CATEGORY_BY_NUMBER,
        TIER_NUMBER_BY_CATEGORY,
    )

    assert TIER_NUMBER_BY_CATEGORY == {
        "ml_foundation": 0,
        "coordination": 1,
        "causal_analytics": 2,
        "monitoring": 3,
        "ml_predictions": 4,
        "self_improvement": 5,
    }
    # Bidirectional and consistent.
    for number, category in TIER_CATEGORY_BY_NUMBER.items():
        assert TIER_NUMBER_BY_CATEGORY[category] == number


def test_uniform_expected_ratio_two_variants():
    from src.tasks.ab_testing_tasks import _expected_ratio_for_variants

    assert _expected_ratio_for_variants({"control": 10, "treatment": 12}) == {
        "control": 0.5,
        "treatment": 0.5,
    }


def test_uniform_expected_ratio_three_variants():
    from src.tasks.ab_testing_tasks import _expected_ratio_for_variants

    ratio = _expected_ratio_for_variants({"a": 1, "b": 2, "c": 3})
    assert set(ratio) == {"a", "b", "c"}
    assert all(abs(v - 1 / 3) < 1e-9 for v in ratio.values())
    assert abs(sum(ratio.values()) - 1.0) < 1e-9


def test_uniform_expected_ratio_empty():
    from src.tasks.ab_testing_tasks import _expected_ratio_for_variants

    assert _expected_ratio_for_variants({}) == {}
