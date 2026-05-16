"""Intent classifier pattern coverage for newly wired intents.

These tests check the pattern-matching layer only (no LLM). The async
``execute`` path falls back to the LLM when no pattern hits, which we don't
exercise here.

Scope (Issue #268 codex LOW-5 review):
---------------------------------------
The assertions below verify that the ``multi_faceted`` and
``experiment_monitor`` patterns are present and match the canonical
phrasings. They are **falsifiability-verified for pattern removal**: if the
patterns in ``INTENT_PATTERNS["multi_faceted"]`` or
``INTENT_PATTERNS["experiment_monitor"]`` are deleted, every assertion in
``test_pattern_classifier_detects_multi_faceted`` and
``test_pattern_classifier_detects_experiment_monitor`` falls through to
``"general"`` and trips.

They are **NOT** load-bearing for the deterministic tie-break (Issue #254
INTENT_PRIORITY). The queries here are deliberately constructed to match
ONLY the target intent's patterns, so the tie-break is never invoked. The
tie-break has its own falsifiability-verified test suite at
``tests/unit/test_agents/test_orchestrator/test_intent_classifier_tie_break.py``
— particularly ``test_classifier_ties_break_by_explicit_priority``
(line ~45) and ``test_dropped_query_{1,2,3}_classifies_as_multi_faceted``
(lines ~67-97). Those construct queries that score equally on multiple
intents to force the tie-break path.

If a future contributor adds a tie-prone query here, route it instead to
``test_intent_classifier_tie_break.py`` so the load-bearing intent stays
discoverable in the right file.
"""

from __future__ import annotations

from src.agents.orchestrator.nodes.intent_classifier import IntentClassifierNode


def _classify(query: str) -> str:
    node = IntentClassifierNode()
    result = node._pattern_classify(query.lower())
    return result.get("primary_intent", "general")


def test_pattern_classifier_detects_experiment_monitor() -> None:
    """Verify ``experiment_monitor`` patterns are present.

    Falsifiability: delete the entire ``experiment_monitor`` entry from
    ``INTENT_PATTERNS`` and every assertion below falls through to
    ``general``. Each query has a domain marker (``SRM``,
    ``running trial enrollment``, ``interim analysis``) that is
    unique to the experiment_monitor patterns, so removing any single
    pattern likewise trips the corresponding assertion.
    """
    assert _classify("check all active A/B experiments for SRM") == "experiment_monitor"
    assert _classify("monitor the running trial enrollment") == "experiment_monitor"
    assert (
        _classify("are there interim analysis triggers in the current experiments?")
        == "experiment_monitor"
    )


def test_pattern_classifier_detects_multi_faceted() -> None:
    """Verify ``multi_faceted`` patterns are present.

    Queries are constructed to match ONLY the multi_faceted patterns
    (no overlap with gap_analyzer's ``r"gap"`` or segment_analysis's
    ``r"segment"``), so this test exercises pattern-coverage, not the
    tie-break. The load-bearing tie-break tests live in
    ``test_intent_classifier_tie_break.py`` — see file docstring.

    Falsifiability: deleting the ``multi_faceted`` entry from
    ``INTENT_PATTERNS`` causes all three classifications to fall through
    to ``"general"``.
    """
    assert _classify("synthesize results from multiple analyses then summarize") == "multi_faceted"
    assert _classify("show me both effects across the cohort") == "multi_faceted"
    assert (
        _classify("integrate the findings with the prior research and also report")
        == "multi_faceted"
    )


def test_pattern_classifier_keeps_existing_intents() -> None:
    """Existing intents must remain stable after adding new patterns.

    Falsifiability: each query was hand-picked to be unambiguous for its
    target intent (no tie with the new intents above). Deleting the
    corresponding pattern in ``INTENT_PATTERNS`` trips the matching
    assertion.
    """
    assert _classify("what causes therapy discontinuation?") == "causal_effect"
    assert _classify("where are the biggest performance gaps?") == "performance_gap"
    assert _classify("system health status") == "system_health"
    assert _classify("design an A/B test for the new outreach program") == "experiment_design"
