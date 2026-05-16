"""Baseline coverage for the inline ``multi_faceted`` regexes.

PR #287 deleted the v42 ``MultiFacetedDetector`` in favour of four inline
patterns at ``src/agents/orchestrator/nodes/intent_classifier.py`` under
``IntentClassifierNode.INTENT_PATTERNS['multi_faceted']``. That module's
docstring acknowledges the trade-off explicitly: chaining-phrase and
conjunction-count semantics traded for ~4 simple regexes.

Prior to this test, only post-classification routing (e.g.
``test_router_new_intents.py``) and tie-break determinism
(``test_intent_classifier_tie_break.py``) exercised multi_faceted. Nothing
asserted the regex semantics themselves — so a future contributor could
silently broaden or narrow the patterns and only the tie-break + routing
tests would notice (and only if the tie/route shape happened to flip).

This file locks in the Option-A baseline. The 6 positive cases hit each of
the 4 regexes at least once; the 4 negative cases assert realistic single-
intent queries do NOT trip the conjunctive markers. If a future Option-B
detector revisit lands, deltas here are the observable contract change.

Falsifiability anchor (see commit body for full evidence): replacing any
of the 4 regexes with ``NEVER_MATCHES_ANYTHING_XYZZY`` trips at least the
positive case targeting that regex. Verified locally before merge.

Public callable: ``IntentClassifierNode._pattern_classify`` is the
public-by-convention entry point used across the orchestrator test suite
(see ``test_intent_classifier_new_intents.py``, ``test_intent_classifier_
tie_break.py``). It is the contract surface for the pattern layer; the
private ``re.search`` calls behind it are NOT touched here.
"""

from __future__ import annotations

import pytest

from src.agents.orchestrator.nodes.intent_classifier import IntentClassifierNode


def _classify(query: str) -> str:
    """Mirror the production pre-lowercase + classify call shape."""
    node = IntentClassifierNode()
    return node._pattern_classify(query.lower()).get("primary_intent", "general")


# Each positive case is annotated with the regex it targets. Comments below
# reference the source patterns at intent_classifier.py:168-174.
#   R1: r"and (also|then|additionally|furthermore)"
#   R2: r"compare .* (vs|versus|against|to) .* and"
#   R3: r"(combine|integrate|synthes).*(analyses|results|findings)"
#   R4: r"(both|multiple) (effects?|analyses|perspectives?)"
_POSITIVE_CASES: list[tuple[str, str]] = [
    # R1 — pure: only multi_faceted scores, no tie-break invoked.
    (
        "summarize the brand performance and additionally surface anomalies",
        "R1 conjunctive marker 'and additionally'",
    ),
    # R1 — through tie-break: also hits performance_gap ('gaps') with 1
    # match each. Both score 0.867; multi_faceted wins via INTENT_PRIORITY.
    # NB: queries that hit a single intent at 2 patterns (e.g. causal_effect
    # matching both ``what drives`` and ``what.*driv``) score 0.933 and beat
    # the multi_faceted single-pattern 0.867 BEFORE tie-break is even
    # consulted — see _pattern_classify score formula at
    # intent_classifier.py:245-248. So tie-break cases must be 1-vs-1.
    (
        "explore performance gaps and also surface anomalies",
        "R1 conjunctive marker 'and also' alongside performance_gap",
    ),
    # R2 — pure: only multi_faceted scores. Uses 'versus' alternation.
    (
        "compare regimen alpha versus regimen beta and report the deltas",
        "R2 comparison + conjunction with 'versus'",
    ),
    # R3 — pure: 'synthesize' verb stem + 'findings' object.
    (
        "synthesize the findings from the last quarter",
        "R3 synthesis verb 'synthesize' + 'findings'",
    ),
    # R4 — pure: 'multiple perspectives' (plural form of perspectives?).
    (
        "examine multiple perspectives on the launch strategy",
        "R4 quantifier 'multiple perspectives'",
    ),
    # R4 — through tie-break: also hits performance_gap ('untapped'). This
    # is the load-bearing tie-resolution path from issue #254.
    (
        "show me both effects in untapped territories",
        "R4 'both effects' alongside performance_gap 'untapped'",
    ),
]


_NEGATIVE_CASES: list[tuple[str, str, str]] = [
    # Each tuple: (query, expected_intent, why this is realistic single-intent)
    (
        "what causes therapy discontinuation?",
        "causal_effect",
        "single causal_effect query — no conjunctive markers",
    ),
    (
        "where are the biggest performance gaps?",
        "performance_gap",
        "single performance_gap query — 'gap' alone, no 'and also' / 'both'",
    ),
    (
        "design an A/B test for the new outreach program",
        "experiment_design",
        "single experiment_design query — no synthesis verbs",
    ),
    (
        "executive summary of brand health for the last month",
        "general",
        "simple lookup-style query — no conjunctive or synthesis markers",
    ),
]


@pytest.mark.parametrize(
    "query,why",
    _POSITIVE_CASES,
    ids=[f"pos:{why}" for _q, why in _POSITIVE_CASES],
)
def test_multi_faceted_positive_baseline(query: str, why: str) -> None:
    """Each positive query MUST classify as ``multi_faceted``.

    Falsifiability: replacing the targeted regex with
    ``NEVER_MATCHES_ANYTHING_XYZZY`` in
    ``intent_classifier.py:168-174`` causes the case for that regex to
    fall back to the next-highest-scoring intent (e.g. ``general`` for the
    pure cases, ``performance_gap``/``causal_effect`` for the tie-break
    cases) and the assertion trips.
    """
    result = _classify(query)
    assert result == "multi_faceted", (
        f"multi_faceted regex baseline drift: query targeting {why!r} "
        f"classified as {result!r}; expected 'multi_faceted'. "
        f"If you intentionally narrowed INTENT_PATTERNS['multi_faceted'], "
        f"update this baseline file in the same commit."
    )


@pytest.mark.parametrize(
    "query,expected,why",
    _NEGATIVE_CASES,
    ids=[f"neg:{why}" for _q, _e, why in _NEGATIVE_CASES],
)
def test_multi_faceted_negative_baseline(query: str, expected: str, why: str) -> None:
    """Single-intent queries MUST NOT be over-classified as ``multi_faceted``.

    Each query is a realistic phrasing a user might type — gap analysis,
    causal investigation, A/B planning, executive lookup. None of them
    contains the conjunctive 'and also/then/...' markers, comparison
    'vs/versus/against/to ... and' shape, synthesis verb + analyses/results/
    findings object, or '(both|multiple) (effects|analyses|perspectives)'
    quantifier required by the 4 inline regexes.

    Falsifiability: broadening any of the 4 multi_faceted regexes such that
    one of these realistic single-intent queries also matches would trip
    the corresponding assertion. The asserted ``expected`` intent doubles
    as a sanity check that the negative query is still being classified as
    intended (rather than silently falling back to 'general').
    """
    result = _classify(query)
    assert result != "multi_faceted", (
        f"multi_faceted over-classification: query {query!r} ({why}) "
        f"classified as 'multi_faceted'; expected non-multi_faceted "
        f"(specifically {expected!r}). The inline regexes at "
        f"INTENT_PATTERNS['multi_faceted'] may have been broadened too far."
    )
    assert result == expected, (
        f"baseline drift on negative case: query {query!r} classified as "
        f"{result!r}; expected {expected!r}. If you intentionally changed "
        f"the single-intent classification, update this baseline file."
    )
