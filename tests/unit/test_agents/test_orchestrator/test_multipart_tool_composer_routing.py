"""Multi-part (dependent pipeline) queries must reliably route to ``tool_composer``.

Audited gap (docs/reports/orchestrator-classifier-audit-20260609.md, findings C2/C3):
the orchestrator only recognised "multi-faceted" via the narrowest of three
detectors (``MULTI_FACETED_PATTERNS``), so genuinely multi-part *dependent*
queries — "which B, **then** C using B" — fell through to a single agent and
never reached ``tool_composer`` (which owns real decomposition in
``tool_composer/decomposer.py``). Two coordinated fixes close it:

  Fix 1 (router safety net): when the classifier flags ``requires_multi_agent``
    but the (primary, secondary) pair is NOT one of the hard-coded parallel
    ``MULTI_AGENT_PATTERNS``, parallel-delegate primary + top real-domain
    secondary instead of silently dropping the secondary intent. A classifier-
    promoted ``multi_faceted`` primary routes to ``tool_composer``. ``router.py``.

  Fix 2 (classifier pipeline promotion): a sequential/dependency marker
    ("then", "after that", "based on that/this/these", "using the … results", …)
    joining **>=2 distinct MAPPED strong intents** promotes the primary intent
    to ``multi_faceted`` (→ tool_composer). ``intent_classifier.py`` +
    ``multi_faceted.has_sequential_composition``.

The discriminator is **a dependency marker joining >=2 recognised analytical
intents** — exactly when tool_composer's sub-question decomposition is useful.
This is enforced ONLY by Fix 2's promotion (no extra SSOT ``MULTI_FACETED_PATTERNS``:
a bare "then <verb>" pattern would promote a single mapped intent — "if X
completes, then forecast" — and an LLM-escalation backstop fired on the marker
alone; both over-routed single asks to the 180s tool_composer and were rejected
in review). Consequence (precision over recall): a multi-part query whose
sub-asks the intent regexes do not recognise routes to the best single agent
rather than over-routing to tool_composer.

DESIGN GUARD (respects the locked ``near-miss-single-regex-hit-loses-on-score``
case in test_intent_classifier_multi_faceted.py): compound/parallel asks
("compare X and Y", "what causes A and what drives B and also explain C") and
single asks with an incidental marker phrase MUST stay single-/parallel-agent.

No mocks: the deterministic pattern + routing layers are exercised directly.
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
            # Codex HIGH-1: anaphoric "based on this" + 2 strong intents (segment +
            # experiment) → promoted via the broadened sequence marker.
            "Which HCP segments responded best? Based on this, design a test to confirm it.",
            # Codex 2nd pass: "using the <modifier> results" dependent handoff +
            # 2 strong intents → promoted.
            "Which HCP segments responded best? Using the model results, design a test.",
            # Genuine "and then" pipeline (2 mapped intents) → tool_composer.
            "Which HCP segments responded best and then design a test.",
        ],
    )
    def test_dependent_pipeline_routes_to_tool_composer(self, query):
        # Promotion requires >=2 distinct MAPPED strong intents + a dependency
        # marker (each query above maps segment_analysis + experiment_design).
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
        # Codex HIGH-1: anaphoric "this/these" + "using ... results" forms.
        assert has_sequential_composition("based on this, design a test") is True
        assert has_sequential_composition("based on these results, forecast trx") is True
        assert has_sequential_composition("after this, recommend a plan") is True
        assert has_sequential_composition("using those results, estimate the lift") is True
        # Codex 2nd pass: "using the <modifier> results" must also match.
        assert has_sequential_composition("using the model results, design a test") is True
        assert has_sequential_composition("using the previous results, forecast") is True
        assert has_sequential_composition("compare A and B") is False
        assert has_sequential_composition("X and also Y") is False


# ---------------------------------------------------------------------------
# The SSOT pattern set stays at 4: dependent-pipeline routing is done via
# classifier promotion (>=2 distinct strong intents + a dependency marker), NOT
# extra patterns. A bare "then <verb>" pattern would promote SINGLE asks (Codex
# 3rd pass: "if X completes, then forecast" / "forecast …, then forecast … again").
# ---------------------------------------------------------------------------
class TestNoSsotSequentialPatterns:
    def test_pattern_count_stays_four(self):
        assert len(MULTI_FACETED_PATTERNS) == 4

    def test_single_intent_then_verb_not_promoted(self):
        # 1 mapped intent (prediction) + "then build" → NOT multi_faceted
        assert (
            _classify("forecast next quarter trx, then build the target list")["primary_intent"]
            != "multi_faceted"
        )

    def test_conditional_then_not_promoted(self):
        # "if X completes, then forecast" is a single ask with a conditional
        # precondition, not a 2-step pipeline.
        assert (
            _classify("if the data refresh completes, then forecast next quarter trx")[
                "primary_intent"
            ]
            != "multi_faceted"
        )


# ---------------------------------------------------------------------------
# Single ask + an INCIDENTAL dependency-marker phrase must stay single-agent.
# (Codex HIGH-2, 2026-06-09: an earlier LLM-escalation backstop — "Fix 3b" —
# fired on the marker alone with no second-ask requirement, so single-intent
# queries with a temporal/data-source phrase ["after this week's data refresh",
# "using the results column"] got escalated and could be wrongly promoted to the
# 180s tool_composer. A real probe confirmed that escalation ONLY ever fired on
# single-intent queries — genuine 2-intent pipelines are already promoted by the
# sequential-composition rule first — so the backstop was pure harm + redundant
# and was removed. A dependency marker only routes to tool_composer when it joins
# >=2 analytical asks.)
# ---------------------------------------------------------------------------
class TestSingleIntentWithIncidentalMarkerStaysSingle:
    @pytest.mark.parametrize(
        "query,expected_agent",
        [
            # "after this" is a temporal modifier here, not a 2-step pipeline
            ("After this week's data refresh, forecast next quarter TRx", "prediction_synthesizer"),
            # "using the results column" is a data-source phrase, not a prior step
            (
                "Using the results column from the model table, forecast next quarter TRx",
                "prediction_synthesizer",
            ),
            # single design ask referencing external context (one analytical task)
            ("Design a test based on these results", "experiment_designer"),
            # Codex 3rd pass: conditional precondition + single ask (1 mapped intent)
            (
                "If the data refresh completes, then forecast next quarter TRx",
                "prediction_synthesizer",
            ),
            # Codex 3rd pass: same ask repeated (1 distinct mapped intent)
            (
                "Forecast next quarter TRx, then forecast next quarter TRx again",
                "prediction_synthesizer",
            ),
            # 1 mapped intent (prediction) + "then build the target list" (unmapped 2nd step)
            ("Forecast next quarter TRx, then build the target list", "prediction_synthesizer"),
            # under-mapped: only segment_analysis maps here (causal/experiment phrasings
            # don't hit the intent regexes) -> route to the best single agent, not the
            # 180s tool_composer. Precision over recall (documented limitation).
            (
                "Compare the causal impact and segment response, then recommend an experiment",
                "heterogeneous_optimizer",
            ),
        ],
    )
    def test_single_ask_with_marker_not_tool_composer(self, query, expected_agent):
        agents = _classify_and_route(query)
        assert "tool_composer" not in agents
        assert agents == [expected_agent]
