"""Regression tests for deterministic intent-classifier tie-break (Issue #254).

The previous implementation broke score ties via Python dict iteration order
of ``INTENT_PATTERNS``. Commit ``1dbc18fd`` papered over the symptom by
dropping the affected test queries instead of fixing the tie-break. This file
codifies the fix: a module-level ``INTENT_PRIORITY`` tuple must resolve ties
deterministically and independently of pattern-block insertion order, with
``multi_faceted`` outranking ``performance_gap``, ``segment_analysis``, and
``causal_effect``.
"""

from __future__ import annotations

from src.agents.orchestrator.nodes import intent_classifier as ic_module
from src.agents.orchestrator.nodes.intent_classifier import (
    INTENT_PRIORITY,
    IntentClassifierNode,
)


def _classify(query: str) -> str:
    node = IntentClassifierNode()
    result = node._pattern_classify(query.lower())
    return result.get("primary_intent", "general")


def test_intent_priority_is_a_tuple_of_strings() -> None:
    """INTENT_PRIORITY must be a module-level tuple ordering ties."""
    assert isinstance(INTENT_PRIORITY, tuple)
    assert all(isinstance(name, str) for name in INTENT_PRIORITY)
    # multi_faceted must rank above the three intents it competes with on ties
    mf = INTENT_PRIORITY.index("multi_faceted")
    for rival in ("performance_gap", "segment_analysis", "causal_effect"):
        assert rival in INTENT_PRIORITY, f"{rival} missing from INTENT_PRIORITY"
        assert mf < INTENT_PRIORITY.index(rival), (
            f"multi_faceted must rank above {rival} in INTENT_PRIORITY"
        )


def test_classifier_ties_break_by_explicit_priority() -> None:
    """Construct input scoring equally for multi_faceted + performance_gap.

    Conjunctive marker 'and also' hits multi_faceted; 'gap' + 'improve' hit
    performance_gap (1 pattern each => 0.8 each). multi_faceted wins by
    INTENT_PRIORITY, not by dict iteration order.
    """
    # 'gap' matches performance_gap's first pattern (1 match -> 0.8 + 0.066 ≈ 0.866).
    # 'and also' matches multi_faceted's first pattern (1 match -> same score).
    # Without INTENT_PRIORITY, the earlier-inserted intent (performance_gap) would win.
    query = "show me the gap and also describe it"
    assert _classify(query) == "multi_faceted"


def test_tie_break_independent_of_pattern_block_insertion_order(monkeypatch) -> None:
    """Permuting INTENT_PATTERNS insertion order must not change tie-break winner.

    Falsification: if tie-break read dict iteration order, swapping the
    insertion order of ``performance_gap`` and ``multi_faceted`` would change
    the winner. With explicit INTENT_PRIORITY, it must not.
    """
    original = IntentClassifierNode.INTENT_PATTERNS
    # Build a new dict where multi_faceted appears LAST (worst position for
    # insertion-order tie-break) and performance_gap appears FIRST.
    reordered_keys = [
        "performance_gap",
        "causal_effect",
        "segment_analysis",
        "experiment_design",
        "prediction",
        "resource_allocation",
        "explanation",
        "system_health",
        "drift_check",
        "feedback",
        "experiment_monitor",
        "cohort_definition",
        "multi_faceted",  # LAST — would lose every tie under insertion-order rule
    ]
    reordered = {k: original[k] for k in reordered_keys if k in original}
    # Include any keys we missed at the end
    for k in original:
        if k not in reordered:
            reordered[k] = original[k]
    monkeypatch.setattr(IntentClassifierNode, "INTENT_PATTERNS", reordered)

    # Same tie-inducing query as the priority test above must still resolve to
    # multi_faceted with the reordered patterns.
    node = IntentClassifierNode()
    result = node._pattern_classify("show me the gap and also describe it")
    assert result.get("primary_intent") == "multi_faceted"


# ---------------------------------------------------------------------------
# Reinstated queries dropped by commit 1dbc18fd as positive multi_faceted cases.
# ---------------------------------------------------------------------------


def test_pattern_classifier_detects_multi_faceted_reinstated_queries() -> None:
    """Reinstate the 3 queries commit 1dbc18fd dropped without fixing root cause."""
    assert (
        _classify("compare HCP visits vs prior treatments and also identify high-risk segments")
        == "multi_faceted"
    )
    assert (
        _classify("combine the causal results with the gap analyses for the brand")
        == "multi_faceted"
    )
    assert _classify("show me both effects across regions") == "multi_faceted"


def test_module_exposes_priority_constant() -> None:
    """Public surface: INTENT_PRIORITY accessible as module-level name."""
    assert hasattr(ic_module, "INTENT_PRIORITY")
