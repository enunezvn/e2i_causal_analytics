"""Tie-break determinism for the intent classifier.

When 2+ intents match a query with equal scores, dict-insertion order in
INTENT_PATTERNS is NOT acceptable as the resolver. Commit ``1dbc18fd`` papered
over this by dropping the queries that tied; this test re-asserts the
queries plus the deterministic resolver. See issue #254.
"""

from __future__ import annotations

from src.agents.orchestrator.nodes.intent_classifier import IntentClassifierNode


def _classify(query: str) -> str:
    node = IntentClassifierNode()
    return node._pattern_classify(query.lower()).get("primary_intent", "general")


def test_intent_priority_exists_and_is_documented() -> None:
    """The classifier must expose its tie-break priority so reviewers can
    reason about regressions without reading max() implementation details.

    Per issue #254 acceptance criterion 1.
    """
    from src.agents.orchestrator.nodes.intent_classifier import INTENT_PRIORITY

    assert isinstance(INTENT_PRIORITY, tuple)
    assert all(isinstance(item, str) for item in INTENT_PRIORITY)
    # multi_faceted must rank above performance_gap, segment_analysis, causal_effect
    # so conjunctive queries win ties (issue #254 acceptance 1).
    assert "multi_faceted" in INTENT_PRIORITY
    assert "performance_gap" in INTENT_PRIORITY
    assert "segment_analysis" in INTENT_PRIORITY
    assert "causal_effect" in INTENT_PRIORITY
    assert INTENT_PRIORITY.index("multi_faceted") < INTENT_PRIORITY.index("performance_gap")
    assert INTENT_PRIORITY.index("multi_faceted") < INTENT_PRIORITY.index("segment_analysis")
    assert INTENT_PRIORITY.index("multi_faceted") < INTENT_PRIORITY.index("causal_effect")
    # experiment_monitor must rank above system_health to prevent the
    # "monitor experiment health" → system_health regression (issue #251 F2).
    assert "experiment_monitor" in INTENT_PRIORITY
    assert "system_health" in INTENT_PRIORITY
    assert INTENT_PRIORITY.index("experiment_monitor") < INTENT_PRIORITY.index("system_health")


def test_classifier_ties_break_by_explicit_priority() -> None:
    """Regression test: constructed input scoring equally for multi_faceted +
    performance_gap MUST resolve to multi_faceted via INTENT_PRIORITY.

    Per issue #254 acceptance criterion 4.

    Construction: a query whose only matches are exactly one multi_faceted
    pattern and exactly one performance_gap pattern, so both score 0.867.
    Under the old dict-insertion-order tie-break, performance_gap (defined
    earlier in INTENT_PATTERNS) wins; under priority-based tie-break,
    multi_faceted wins.
    """
    # "untapped" matches performance_gap pattern r"untapped" (single match).
    # "both effects" matches multi_faceted pattern r"(both|multiple) (effects?|...)" (single match).
    # Both score 0.867 → tie. INTENT_PRIORITY ranks multi_faceted higher.
    query = "show me both effects in untapped territories"
    result = _classify(query)
    assert result == "multi_faceted", (
        f"tie-break must favour multi_faceted via INTENT_PRIORITY, got {result!r}"
    )


def test_dropped_query_1_classifies_as_multi_faceted() -> None:
    """Issue #254 acceptance 3, query 1.

    'compare HCP visits vs prior treatments and also identify high-risk segments'
    — dropped in 1dbc18fd; must classify as multi_faceted.
    """
    assert (
        _classify("compare HCP visits vs prior treatments and also identify high-risk segments")
        == "multi_faceted"
    )


def test_dropped_query_2_classifies_as_multi_faceted() -> None:
    """Issue #254 acceptance 3, query 2.

    'combine the causal results with the gap analyses for the brand'
    — dropped in 1dbc18fd; must classify as multi_faceted.
    """
    assert (
        _classify("combine the causal results with the gap analyses for the brand")
        == "multi_faceted"
    )


def test_dropped_query_3_classifies_as_multi_faceted() -> None:
    """Issue #254 acceptance 3, query 3.

    'show me both effects across regions' — dropped in 1dbc18fd; must
    classify as multi_faceted.
    """
    assert _classify("show me both effects across regions") == "multi_faceted"


def test_priority_falls_through_to_general_for_unmatched_intents() -> None:
    """Intents not in INTENT_PRIORITY must lose all ties — but the classifier
    must not crash on them. A query that hits no patterns returns 'general'
    with no priority lookup at all (early-return path).
    """
    # A query with no domain words. Should return 'general' via the
    # max(scores)==0 early-return.
    assert _classify("the quick brown fox") == "general"


def test_existing_single_match_classifications_still_work() -> None:
    """Sanity: changing the tie-break must not regress unambiguous queries."""
    # Unambiguous causal — only causal_effect should match
    assert _classify("what drives discontinuation rates?") == "causal_effect"
    # Unambiguous experiment design
    assert _classify("design an A/B test for the new outreach program") == "experiment_design"
    # Unambiguous health
    assert _classify("system health status") == "system_health"
