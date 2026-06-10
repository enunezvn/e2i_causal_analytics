"""Unit tests for the pure helpers introduced by issue #825 (no DB).

agent_registry tier text-category <-> numeric mapping (real agent_tier enum).
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


def test_tier_number_for_category_unknown_sorts_last():
    """An unknown/None category maps to 99 so unknown agents sort last."""
    from src.repositories.agent_registry import tier_number_for_category

    assert tier_number_for_category(None) == 99
    assert tier_number_for_category("not_a_real_tier") == 99
    assert tier_number_for_category("causal_analytics") == 2
