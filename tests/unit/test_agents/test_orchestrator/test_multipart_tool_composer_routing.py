"""Multi-part (dependent pipeline) queries must reliably route to ``tool_composer``.

Audited gap (docs/reports/orchestrator-classifier-audit-20260609.md, findings C2/C3):
the orchestrator only recognised "multi-faceted" via the narrowest of three
detectors (``MULTI_FACETED_PATTERNS``), so genuinely multi-part *dependent*
queries — "do A, and which B, **then** C using A/B" — fell through to a single
agent and never reached ``tool_composer`` (which owns real decomposition in
``tool_composer/decomposer.py``). Three coordinated fixes close it:

  Fix 1 (router safety net): when the classifier flags ``requires_multi_agent``
    but the (primary, secondary) pair is NOT one of the hard-coded parallel
    ``MULTI_AGENT_PATTERNS``, route to ``tool_composer`` instead of silently
    collapsing to a single agent. ``router.py``.

  Fix 2 (classifier pipeline promotion): a sequential/dependency marker
    ("then", "after that", "based on that", …) joining 2+ distinct strong
    intents promotes the primary intent to ``multi_faceted`` (→ tool_composer).
    ``intent_classifier.py`` + ``multi_faceted.has_sequential_composition``.

  Fix 3 (broaden detection + LLM backstop): two new SSOT ``MULTI_FACETED_PATTERNS``
    for natural sequential forms (both require a "then"-marker so they cannot
    flip the locked single-intent negatives), plus a deterministic
    ``_needs_llm_disambiguation`` predicate that escalates borderline soft-signal
    queries (facet/topic scorers) to the existing Haiku fallback.

DESIGN GUARD (respects the locked ``near-miss-single-regex-hit-loses-on-score``
case in test_intent_classifier_multi_faceted.py): the discriminator is an
*explicit sequence/dependency marker*, NOT merely "2 intents fired". Compound /
parallel asks ("compare X and Y", "what causes A and what drives B and also
explain C") have no such marker and MUST stay single-agent.

No mocks: the deterministic pattern + routing layers are exercised directly; the
LLM backstop is tested via its decision predicate, not by faking an LLM.
"""

from __future__ import annotations

import asyncio

import pytest

from src.agents.multi_faceted import (
    MULTI_FACETED_PATTERNS,
    has_sequential_composition,
)
from src.agents.orchestrator.nodes.intent_classifier import IntentClassifierNode
from src.agents.orchestrator.nodes.router import RouterNode


# ---------------------------------------------------------------------------
# Helpers — real classify → real route, no mocks
# ---------------------------------------------------------------------------
def _classify(query: str) -> dict:
    node = IntentClassifierNode.__new__(IntentClassifierNode)  # skip LLM ctor
    return dict(node._pattern_classify(query.lower()))


def _route(intent: dict) -> list[str]:
    router = RouterNode()
    state = {"query": "", "intent": intent}
    routed = asyncio.run(router.execute(state))
    return [d["agent_name"] for d in routed.get("dispatch_plan", [])]


def _classify_and_route(query: str) -> list[str]:
    return _route(_classify(query))


# ---------------------------------------------------------------------------
# End-to-end behavioural contract: dependent pipelines → tool_composer
# ---------------------------------------------------------------------------
class TestMultipartRoutesToToolComposer:
    @pytest.mark.parametrize(
        "query",
        [
            # 2 strong intents (segment + experiment) joined by "then" → pipeline
            "What drove the Kisqali uplift, and which segments responded best, "
            "then design a test to confirm it?",
            # comparison + recommendation joined by "then"
            "Compare the causal impact and segment response, then recommend an experiment",
            # single strong intent (prediction) + "then <verb>" → caught by new SSOT pattern
            "Forecast next quarter TRx, then build the target list",
        ],
    )
    def test_dependent_pipeline_routes_to_tool_composer(self, query):
        assert _classify_and_route(query) == ["tool_composer"]


class TestSingleAndParallelStaySingleAgent:
    """Negative guards — these must NOT be over-routed to the expensive
    (180s SLA) tool_composer."""

    @pytest.mark.parametrize(
        "query,expected_agent",
        [
            # clean single causal
            ("What was the impact of the Q3 Kisqali campaign on prescriptions?", "causal_impact"),
            # clean single cohort
            ("Build a cohort of CSU patients eligible for Remibrutinib.", "cohort_constructor"),
            # clean single experiment design (no sequence marker)
            ("Design an A/B test for the new outreach program", "experiment_designer"),
        ],
    )
    def test_single_intent_stays_single_agent(self, query, expected_agent):
        agents = _classify_and_route(query)
        assert "tool_composer" not in agents
        assert expected_agent in agents

    @pytest.mark.parametrize(
        "query",
        [
            # compound/list join, no sequence marker — locked semantics
            "compare growth rates for cohort A and cohort B",
            # two causal probes + 'and also' (additive, NOT sequential) — locked case
            "what causes discontinuation and what drives switching and also explain trends",
        ],
    )
    def test_compound_without_sequence_marker_not_tool_composer(self, query):
        assert "tool_composer" not in _classify_and_route(query)


# ---------------------------------------------------------------------------
# Fix 1 — RouterNode safety net (unit)
# ---------------------------------------------------------------------------
class TestRouterMultiAgentFallback:
    def test_multi_faceted_primary_routes_to_tool_composer(self):
        # A classifier-promoted dependent pipeline → tool_composer (single).
        intent = {
            "primary_intent": "multi_faceted",
            "secondary_intents": ["segment_analysis", "experiment_design"],
            "requires_multi_agent": True,
            "confidence": 0.9,
        }
        assert _route(intent) == ["tool_composer"]

    def test_independent_compound_without_pattern_parallel_delegates(self):
        # 2 strong intents, no dependency, not a hard-coded pair → parallel
        # delegation of BOTH agents (C3 fix: secondary no longer dropped),
        # NOT the 180s tool_composer.
        intent = {
            "primary_intent": "segment_analysis",
            "secondary_intents": ["experiment_design"],
            "requires_multi_agent": True,
            "confidence": 0.93,
        }
        agents = _route(intent)
        assert "tool_composer" not in agents
        assert agents == ["heterogeneous_optimizer", "experiment_designer"]

    def test_known_parallel_pair_preserved_not_tool_composer(self):
        # (causal_effect, segment_analysis) IS a MULTI_AGENT_PATTERN → parallel 2-agent
        intent = {
            "primary_intent": "causal_effect",
            "secondary_intents": ["segment_analysis"],
            "requires_multi_agent": True,
            "confidence": 0.93,
        }
        agents = _route(intent)
        assert "tool_composer" not in agents
        assert agents == ["causal_impact", "heterogeneous_optimizer"]

    def test_single_intent_unaffected(self):
        intent = {
            "primary_intent": "causal_effect",
            "secondary_intents": [],
            "requires_multi_agent": False,
            "confidence": 0.9,
        }
        assert _route(intent) == ["causal_impact"]


# ---------------------------------------------------------------------------
# Fix 2 — classifier sequential-pipeline promotion (unit)
# ---------------------------------------------------------------------------
class TestSequentialPromotion:
    def test_two_intents_plus_then_promotes_to_multi_faceted(self):
        q = "which segment responded best, then design a test"
        assert _classify(q)["primary_intent"] == "multi_faceted"

    def test_two_intents_with_additive_marker_not_promoted(self):
        # locked: 'and also' is additive, not sequential → stays causal_effect
        q = "what causes discontinuation and what drives switching and also explain trends"
        assert _classify(q)["primary_intent"] == "causal_effect"

    def test_single_intent_with_sequence_marker_not_falsely_multi_via_promotion(self):
        # only one strong intent fires here; promotion (which needs 2 distinct
        # strong intents) must NOT fire. (The NEW pattern path is tested
        # separately.) "summarize then" has no second analytical intent.
        q = "summarize the brand health, then summarize it again"
        assert _classify(q)["primary_intent"] != "multi_faceted"

    def test_has_sequential_composition_helper(self):
        assert has_sequential_composition("do A, then do B") is True
        assert has_sequential_composition("after that, estimate the lift") is True
        assert has_sequential_composition("based on that, recommend a plan") is True
        assert has_sequential_composition("compare A and B") is False
        assert has_sequential_composition("X and also Y") is False


# ---------------------------------------------------------------------------
# Fix 3a — broadened SSOT patterns for natural sequential forms (unit)
# ---------------------------------------------------------------------------
class TestBroadenedSsotPatterns:
    def test_pattern_count_is_six(self):
        # Intentional broadening from 4 → 6 (the lock makes this observable).
        assert len(MULTI_FACETED_PATTERNS) == 6

    def test_then_verb_form_classifies_multi_faceted(self):
        # single intent (prediction) + "then build" → new pattern catches it
        assert (
            _classify("forecast next quarter trx, then build the target list")["primary_intent"]
            == "multi_faceted"
        )

    def test_and_which_then_form_classifies_multi_faceted(self):
        assert (
            _classify("show the uplift, and which segments responded, then estimate the roi")[
                "primary_intent"
            ]
            == "multi_faceted"
        )

    def test_new_patterns_do_not_flip_locked_negative(self):
        # the broadened patterns require a 'then'-marker, so this stays single
        assert (
            _classify("compare growth rates for cohort a and cohort b")["primary_intent"]
            != "multi_faceted"
        )


# ---------------------------------------------------------------------------
# Fix 3b — LLM-disambiguation decision predicate (unit, deterministic)
# ---------------------------------------------------------------------------
class TestLlmDisambiguationPredicate:
    def _node(self) -> IntentClassifierNode:
        return IntentClassifierNode.__new__(IntentClassifierNode)

    def test_soft_signal_single_intent_escalates(self):
        # facet-scorer-positive, single strong intent, primary != multi_faceted
        node = self._node()
        q = "why did the causal effect drop, what should we do"
        pr = dict(node._pattern_classify(q.lower()))
        assert pr["primary_intent"] != "multi_faceted"
        assert node._needs_llm_disambiguation(q, pr) is True

    def test_clean_single_intent_does_not_escalate(self):
        node = self._node()
        q = "what is the trx for kisqali"
        pr = dict(node._pattern_classify(q.lower()))
        assert node._needs_llm_disambiguation(q, pr) is False

    def test_already_multi_faceted_does_not_escalate(self):
        node = self._node()
        q = "which segment responded best, then design a test"
        pr = dict(node._pattern_classify(q.lower()))
        assert pr["primary_intent"] == "multi_faceted"
        assert node._needs_llm_disambiguation(q, pr) is False
