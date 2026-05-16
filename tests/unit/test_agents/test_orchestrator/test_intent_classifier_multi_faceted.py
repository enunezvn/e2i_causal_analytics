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


# Each tuple: (test_id, query, expected_intent, why this is realistic single-intent)
#
# ``test_id`` is the compact pytest parametrize id (short, hyphen-cased, stable).
# ``why`` is the long-form explanation that becomes part of the failure message.
# Two fields decouple test-id stability (search/select) from prose richness
# (failure diagnosis) — long ``why`` strings would otherwise produce unwieldy
# ``-k`` selectors.
_NEGATIVE_CASES: list[tuple[str, str, str, str]] = [
    #
    # The first four are "marker-free" strawmen: queries that contain none of
    # the conjunctive/comparison/synthesis/quantifier markers required by R1-R4.
    # They establish the "obviously single-intent" floor.
    (
        "marker-free-causal",
        "what causes therapy discontinuation?",
        "causal_effect",
        "single causal_effect query — no conjunctive markers",
    ),
    (
        "marker-free-perf-gap",
        "where are the biggest performance gaps?",
        "performance_gap",
        "single performance_gap query — 'gap' alone, no 'and also' / 'both'",
    ),
    (
        "marker-free-experiment-design",
        "design an A/B test for the new outreach program",
        "experiment_design",
        "single experiment_design query — no synthesis verbs",
    ),
    (
        "marker-free-lookup",
        "executive summary of brand health for the last month",
        "general",
        "simple lookup-style query — no conjunctive or synthesis markers",
    ),
    #
    # The next three are NEAR-MISS negatives: queries that contain a token
    # superficially resembling a multi_faceted marker but in a role the
    # regexes correctly do not capture, OR where the multi_faceted regex
    # fires but a competing intent wins on score (not on priority). These
    # are the stronger baseline — they pin Option-A's boundary semantics so
    # a future Option-B detector revisit (or a too-eager regex broadening)
    # produces an observable diff.
    (
        "near-miss-and-listjoin",
        "compare growth rates for cohort A and cohort B",
        "general",
        "'and' joins cohort names not intents; R1 requires "
        "'and (also|then|additionally|furthermore)'; R2 "
        "'compare .* (vs|versus|against|to) .* and' requires explicit "
        "vs/versus/against/to before 'and' — neither fires",
    ),
    (
        "near-miss-also-discourse",
        "is this also true for region X",
        "general",
        "'also' is a discourse marker, not an additive-intent conjunction; "
        "R1 requires literal 'and ' immediately before 'also', which is "
        "absent here",
    ),
    (
        "near-miss-single-regex-hit-loses-on-score",
        "what causes discontinuation and what drives switching and also explain trends",
        "causal_effect",
        "R1 'and also' fires (multi_faceted scores 0.867 from 1 hit) but "
        "causal_effect matches 2 patterns ('what.*caus...' + 'what drives') "
        "and scores 0.933, beating multi_faceted before tie-break is "
        "consulted (see formula at intent_classifier.py:245-248). "
        "multi_faceted appears as a secondary intent",
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
    "test_id,query,expected,why",
    _NEGATIVE_CASES,
    ids=[f"neg:{tid}" for tid, _q, _e, _why in _NEGATIVE_CASES],
)
def test_multi_faceted_negative_baseline(test_id: str, query: str, expected: str, why: str) -> None:
    """Single-intent queries MUST NOT be over-classified as ``multi_faceted``.

    The case set has two tiers:

    * **marker-free** (4 cases) — realistic queries that contain none of the
      conjunctive/comparison/synthesis/quantifier markers required by R1-R4.
      Establishes the "obviously single-intent" floor.
    * **near-miss** (3 cases) — queries that contain a token superficially
      resembling a multi_faceted marker but in a role the Option-A regexes
      correctly do not capture (e.g. ``and`` as a list-join, ``also`` as a
      discourse marker), or where a multi_faceted regex DOES fire but a
      competing intent wins on score before tie-break is consulted. These
      pin the boundary semantics so a future Option-B detector revisit (or
      a too-eager regex broadening) produces an observable diff.

    Falsifiability:

    * Broadening any of the 4 multi_faceted regexes such that one of the
      marker-free or near-miss queries also matches would trip the
      corresponding ``!= 'multi_faceted'`` assertion.
    * For the near-miss-single-regex-hit-loses-on-score case specifically,
      narrowing the competing intent's patterns (so its score drops below
      0.867) would flip the outcome to ``multi_faceted`` and trip the same
      assertion — exercising the "loses on score" boundary explicitly.

    The asserted ``expected`` intent doubles as a sanity check that the
    negative query is still being classified as intended (rather than
    silently falling back to ``general``).
    """
    result = _classify(query)
    assert result != "multi_faceted", (
        f"[{test_id}] multi_faceted over-classification: query {query!r} "
        f"({why}) classified as 'multi_faceted'; expected non-multi_faceted "
        f"(specifically {expected!r}). The inline regexes at "
        f"INTENT_PATTERNS['multi_faceted'] may have been broadened too far."
    )
    assert result == expected, (
        f"[{test_id}] baseline drift on negative case: query {query!r} "
        f"classified as {result!r}; expected {expected!r}. If you "
        f"intentionally changed the single-intent classification, update "
        f"this baseline file."
    )
