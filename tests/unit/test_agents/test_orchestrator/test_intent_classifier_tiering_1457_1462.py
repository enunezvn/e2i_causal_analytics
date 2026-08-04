"""#1457 + #1462 — the #1449 tiering gate, constrained by the tiered OBJECT.

These two issues pull in opposite directions on the same regex patterns, which
is why they land together.

#1457 (OVER-REACH, measured): a lone brand/disease token satisfies
``_TIER_POPULATION_ANCHORS``, so when it appears BEFORE the tier container a
tiering ask about a purely COMMERCIAL object ("Rank Kisqali call plans into
high, medium, and low priority tiers") classifies ``cohort_definition``
LLM-free at 0.867+ and lands on ``cohort_profiler`` — which never fails closed,
so the user gets a confident, real-looking per-segment HCP population profile
for a call-plan/budget/territory/creative/sponsorship question. The DISPROVEN
fix is removing brand/disease tokens from the anchor lexicon: that breaks the
genuinely clinical "Break down Remibrutinib NRx by IgE tier (low / medium /
high)." where the brand token is the ONLY population signal. The shipped fix
instead DISQUALIFIES the match when a commercial HEAD NOUN sits inside the
verb->anchor->container / anchor->ladder span — the brand is then a modifier of
a commercial object, not the population being partitioned. Same fail-closed
direction as ``(?<!managed )`` on the payer hole: ambiguity escalates to the
LLM instead of being answered confidently and wrongly.

#1462 (UNDER-REACH, measured): "Segment HCPs into three groups by prescription
volume: high, moderate, and low" carried only ``segment_analysis`` — the
middle-rung alternation had no "moderate", so ladder shape (b) never fired and
the row misrouted to a CATE estimator. Adding "moderate" alone would newly
admit "split the Kisqali sales territories into three groups: high, moderate,
low" (brand anchor + now-matching ladder) — exactly the #1457 class — so the
widening is only safe BECAUSE the head-noun disqualifier lands with it. The
"groups" container stays out of ``_TIER_CONTAINER_NOUNS``: shape (b) needs no
container noun, and admitting the far-too-generic "groups" into shape (a)
(verb + anchor + container, NO ladder required) would re-open a fresh
over-reach class for nothing the acceptance rows need.

Queries are authored out-of-gold adversarial probes — the 337-row gold contains
no brand-modified commercial tiering row, so it scores 0 losses for a build
that misroutes every one of them. No mocks: the real ``_pattern_classify`` ->
``RouterNode.execute`` chain runs.
"""

from __future__ import annotations

import asyncio

import pytest

from src.agents.orchestrator.nodes.intent_classifier import IntentClassifierNode
from src.agents.orchestrator.nodes.router import RouterNode

# Pattern confidence at/above which the real ``execute()`` trusts the pattern
# layer and never calls the LLM (intent_classifier.execute).
PATTERN_TRUST_FLOOR = 0.8


def _classify(query: str) -> dict:
    node = IntentClassifierNode.__new__(IntentClassifierNode)  # skip LLM ctor
    return dict(node._pattern_classify(query.lower()))


def _route(intent: dict) -> list[str]:
    router = RouterNode()
    routed = asyncio.run(router.execute({"query": "", "intent": intent}))
    return sorted(d["agent_name"] for d in routed.get("dispatch_plan", []))


def _classify_and_route(query: str) -> list[str]:
    return _route(_classify(query))


# ---------------------------------------------------------------------------
# #1457 — a brand modifying a COMMERCIAL head noun is not a population anchor.
# ---------------------------------------------------------------------------
class TestBrandModifiedCommercialTieringEscalates:
    @pytest.mark.parametrize(
        "query",
        [
            # Each of these MEASURED cohort_definition on main (0.867+, above
            # the 0.8 trust floor => confident, LLM-free) purely because the
            # brand/disease token precedes the container/ladder. The tiered
            # object is commercial in every one.
            "Rank Kisqali call plans into high, medium, and low priority tiers",
            "Bucket the Kisqali marketing budget into high, medium and low tiers",
            "Split the Fabhalta sales territories into high, medium, and low tiers",
            "Categorize CSU campaign creatives into high, medium, and low performing tiers",
            # Ladder shape (b) with NO container noun after the anchor — the
            # head-noun disqualifier must cover the anchor->ladder span too.
            "Tier the breast cancer conference sponsorships from high to medium to low",
            # codex iter-1 HIGH (2026-08-04): commercial head noun BEFORE the
            # brand anchor. The tempered gaps only guard spans AFTER the
            # anchor, so shape (b) started at "Kisqali" and never saw "call
            # plans" — measured cohort_definition @0.867 on the first lane
            # build. The pre-anchor veto must kill it.
            "Rank call plans for Kisqali into high, medium, and low priority tiers",
            # Reviewer finding (2026-08-04, measured @0.933): "budgetary" is a
            # one-word morphological paraphrase of the issue's own "marketing
            # budget" row that `budgets?` missed (no \w* tolerance).
            "Rank Kisqali budgetary allocations into high, medium, and low priority tiers",
            # codex iter-2 HIGH (2026-08-04): narrowing channels to
            # qualified-only freed UNQUALIFIED commercial channel compounds —
            # "channel tactics/mix/strategy/plan" are commercial objects even
            # without a marketing/sales qualifier.
            "Rank Kisqali channel tactics into high, medium, and low tiers",
            "Bucket the Fabhalta channel mix into high, medium, and low tiers",
        ],
    )
    def test_branded_commercial_tiering_escalates_to_the_llm(self, query: str) -> None:
        """No deterministic cohort claim; below the floor so the LLM is consulted."""
        intent = _classify(query)
        assert intent["primary_intent"] != "cohort_definition"
        assert intent["confidence"] < PATTERN_TRUST_FLOOR

    @pytest.mark.parametrize(
        "query",
        [
            # The brandless variants never matched (no anchor at all) and must
            # stay that way — pinned so the fix cannot trade one hole for another.
            "Rank call plans into high, medium, and low priority tiers",
            "Bucket the marketing budget into high, medium and low tiers",
            "Split the sales territories into high, medium, and low tiers",
        ],
    )
    def test_brandless_commercial_tiering_stays_out(self, query: str) -> None:
        intent = _classify(query)
        assert intent["primary_intent"] != "cohort_definition"
        assert intent["confidence"] < PATTERN_TRUST_FLOOR


# ---------------------------------------------------------------------------
# The DISPROVEN alternative — dropping brand anchors — must stay disproven:
# clinical rows where the brand token is the only population signal must keep
# matching after the head-noun constraint lands.
# ---------------------------------------------------------------------------
class TestBrandAnchoredClinicalTieringStillMatches:
    @pytest.mark.parametrize(
        "query",
        [
            # bench-0142 — the row that disproves removing brand anchors: the
            # brand token is the ONLY population signal, and the tiered object
            # (NRx by IgE) is clinical, not commercial.
            "Break down Remibrutinib NRx by IgE tier (low / medium / high).",
            # bench-0022 — demo 4.3 verbatim, the original #1449 defect row.
            "Segment HCPs by prescription volume into high, medium, and low tiers",
        ],
    )
    def test_clinical_tiering_still_routes_to_cohort_profiler(self, query: str) -> None:
        assert _classify(query)["primary_intent"] == "cohort_definition"
        assert _classify_and_route(query) == ["cohort_profiler"]


# ---------------------------------------------------------------------------
# #1462 — "moderate" is a middle rung; the three-groups phrasing is cohort work.
# ---------------------------------------------------------------------------
class TestModerateMiddleRungIsCohortWork:
    def test_three_groups_moderate_phrasing_matches_cohort_definition(self) -> None:
        """Measured on main: only ``segment_analysis`` fired, so this demo-4.3
        paraphrase misrouted to the CATE estimator. The clinical anchor
        (HCPs) precedes an explicit three-level ladder — shape (b) must fire
        once "moderate" joins the middle-rung alternation."""
        query = "Segment HCPs into three groups by prescription volume: high, moderate, and low"
        intent = _classify(query)
        assert intent["primary_intent"] == "cohort_definition"
        assert intent["confidence"] >= PATTERN_TRUST_FLOOR
        assert _classify_and_route(query) == ["cohort_profiler"]

    def test_moderate_widening_does_not_reopen_the_1457_hole(self) -> None:
        """The interlock row: brand anchor + three-groups ladder with
        "moderate" — WITHOUT the head-noun disqualifier this would newly match
        the moment "moderate" is admitted, and cohort_definition outranks
        segment_analysis in INTENT_PRIORITY, so the commercial ask would flip
        to a confident cohort_profiler misroute. "territories" between the
        anchor and the ladder must disqualify it. (The row itself stays a
        confident segment_analysis on the bare "groups" lexeme — pre-existing
        behaviour this test pins WITHOUT endorsing; the assertion here is only
        that cohort_definition must not capture it.)"""
        intent = _classify(
            "split the Kisqali sales territories into three groups: high, moderate, low"
        )
        assert intent["primary_intent"] != "cohort_definition"
        assert _route(intent) != ["cohort_profiler"]

    def test_positional_guard_is_untouched(self) -> None:
        """ "Rank call-plan tiers by expected ROI for Kisqali" partitions CALL
        PLANS (anchor after the container) — pinned by the #1449 suite and
        re-pinned here because the head-noun lexicon includes call plans."""
        assert (
            _classify("Rank call-plan tiers by expected ROI for Kisqali")["primary_intent"]
            != "cohort_definition"
        )


# ---------------------------------------------------------------------------
# codex iter-1 HIGH/MED (2026-08-04): two head-noun lexemes over-matched
# ordinary clinical/idiomatic vocabulary. Both queries below classified
# cohort_definition @0.933 on main (6e099100) and fell to a confident
# segment_analysis @0.867 (a CATE misroute) on the first lane build — a
# REGRESSION the lexicon narrowing must undo.
# ---------------------------------------------------------------------------
class TestClinicalVocabularyNotDisqualified:
    @pytest.mark.parametrize(
        "query",
        [
            # \baccounts?\b matched the idiom "taking into account".
            "Segment HCPs taking into account prescription volume into high, medium, and low tiers",
            # bare \bchannels?\b matched biological "calcium channel".
            "Segment patients by calcium channel expression into high, medium, and low tiers",
        ],
    )
    def test_idiomatic_and_biological_vocabulary_keeps_matching(self, query: str) -> None:
        intent = _classify(query)
        assert intent["primary_intent"] == "cohort_definition"
        assert intent["confidence"] >= PATTERN_TRUST_FLOOR
        assert _classify_and_route(query) == ["cohort_profiler"]

    @pytest.mark.parametrize(
        "query",
        [
            # The narrowing must NOT free genuinely commercial phrasings:
            # qualified channels and plural accounts stay in the lexicon.
            "Rank the Kisqali marketing channels into high, medium, and low performing tiers",
            "Tier the Kisqali key accounts from high to medium to low",
        ],
    )
    def test_qualified_commercial_vocabulary_still_escalates(self, query: str) -> None:
        intent = _classify(query)
        assert intent["primary_intent"] != "cohort_definition"
        assert intent["confidence"] < PATTERN_TRUST_FLOOR
