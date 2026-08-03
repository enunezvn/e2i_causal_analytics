"""#1449 — descriptive HCP/patient TIERING must route to ``cohort_profiler``.

Demo question 4.3 ("Segment HCPs by prescription volume into high, medium, and
low tiers") routed to ``heterogeneous_optimizer`` + its ``gap_analyzer``
fallback on ``/chat/stream`` (request ``70a4b5d1``, 2026-08-03): the legacy
``segment_analysis`` pattern ``(segment|group|heterogen)`` fires on the bare
word "Segment", and ``segment_analysis`` maps to a CATE estimator. Both agents
then failed closed and the orchestrator reported complete failure. Gold routing
(``benchmark_queries_gold.jsonl``, ``demo_meta.question_id == "4.3"``) is
SINGLE_AGENT -> ``cohort_profiler``: a single-domain descriptive partition of a
population is cohort construction, not a treatment-effect estimate — per the
composition ruling, single-domain multi-step stays SINGLE_AGENT no matter how
many internal steps.

THE RISK THIS FILE GUARDS (the #1408 ``\\binterim\\b`` lesson)
--------------------------------------------------------------
A pattern broad enough to catch "segment ... into tiers" is trivially broad
enough to steal genuine CATE asks ("which HCP segments show the strongest
treatment effect") away from ``heterogeneous_optimizer`` — a
higher-priority intent (``cohort_definition`` outranks ``segment_analysis`` in
``INTENT_PRIORITY``) wins the tie CONFIDENTLY and so never reaches the LLM
safety net. So the tiering signal is DOMAIN-GATED three ways:

  1. it requires an explicit tier CONTAINER (a partition verb + tiers/buckets/
     quartiles/deciles/categories) or an explicit 3-level ordinal LADDER
     (high/medium/low, top/middle/bottom) — never a bare "tier" token
     (``"high-decile HCPs"``, bench-0263 gold ``resource_optimizer``, must
     stay out), and
  2. a treatment-effect veto: any causal/CATE/uplift/responder/effect lexeme
     anywhere in the query suppresses the tiering match entirely. The veto can
     only ever DOWNGRADE to today's behaviour, so it is the safe direction.
  3. the classifier's existing clause gate keeps the co-firing
     ``segment_analysis`` from splitting the ask into a 2-agent dispatch.

Queries are verbatim gold rows (bench-NNNN) plus authored out-of-gold
adversarial probes — behavioural pins, not phrase overfitting. No mocks: the
real ``_pattern_classify`` -> ``RouterNode.execute`` chain runs.
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
# The defect: 4.3 and its gold perturbations must reach cohort_profiler.
# ---------------------------------------------------------------------------
class TestDescriptiveTieringRoutesToCohortProfiler:
    @pytest.mark.parametrize(
        "query",
        [
            # bench-0022 — demo 4.3 verbatim (the reported misroute).
            "Segment HCPs by prescription volume into high, medium, and low tiers",
            # bench-0207 — 4.3 paraphrase perturbation (was: escalate -> explainer).
            "Classify healthcare providers into three prescription-volume "
            "categories—top, middle, and bottom performers—based on their Rx output.",
            # bench-0208 — 4.3 fragment perturbation (was: escalate -> explainer).
            "HCP tiers by Rx volume: high/med/low",
            # bench-0142 — same descriptive-tiering shape on a patient attribute
            # (was: "break down" -> explanation -> explainer).
            "Break down Remibrutinib NRx by IgE tier (low / medium / high).",
        ],
    )
    def test_tiering_asks_route_to_cohort_profiler(self, query: str) -> None:
        assert _classify_and_route(query) == ["cohort_profiler"]

    def test_43_is_single_agent_not_a_parallel_split(self) -> None:
        """The co-firing ``segment_analysis`` must not add a second agent.

        ``cohort_definition`` and ``segment_analysis`` both match 4.3; the
        clause gate (#1337) must keep it a SINGLE dispatch rather than a
        PARALLEL cohort_profiler + heterogeneous_optimizer pair.
        """
        intent = _classify("Segment HCPs by prescription volume into high, medium, and low tiers")
        assert intent["primary_intent"] == "cohort_definition"
        assert intent["requires_multi_agent"] is False
        assert _route(intent) == ["cohort_profiler"]

    def test_43_is_decided_deterministically(self) -> None:
        """4.3 must be decided by the pattern layer, not punted to the LLM."""
        intent = _classify("Segment HCPs by prescription volume into high, medium, and low tiers")
        assert intent["confidence"] >= PATTERN_TRUST_FLOOR


# ---------------------------------------------------------------------------
# OUT-OF-GOLD OVER-REACH — the regression class this lane has shipped before.
# Genuine CATE asks must still reach heterogeneous_optimizer.
# ---------------------------------------------------------------------------
class TestGenuineCateAsksStayOnHeterogeneousOptimizer:
    @pytest.mark.parametrize(
        "query",
        [
            # bench-0023 — demo 4.4, human-ratified gold.
            "Which HCP segments show the strongest treatment effect for Remibrutinib?",
            # bench-0005 — demo 1.6.
            "Which HCP segments show the strongest effect?",
            # bench-0173 — 1.6 paraphrase.
            "Which physician segments are demonstrating the most pronounced impact on our metrics?",
            # bench-0209 — 4.4 paraphrase.
            "Which physician segments demonstrate the most significant clinical "
            "response to Remibrutinib therapy?",
        ],
    )
    def test_cate_asks_still_route_to_heterogeneous_optimizer(self, query: str) -> None:
        assert _classify_and_route(query) == ["heterogeneous_optimizer"]

    @pytest.mark.parametrize(
        "query",
        [
            # A tier LADDER glued onto a genuine treatment-effect ask: the veto
            # must suppress the tiering signal so this is never confidently
            # stolen from the CATE estimator.
            "Compare the treatment effect across high, medium, and low volume prescribers",
            # An explicit tier CONTAINER glued onto a CATE ask.
            "Segment HCPs into deciles and report the CATE for Kisqali",
            "Rank the high, medium and low responders by uplift",
            "Bucket prescribers into quartiles by causal impact",
        ],
    )
    def test_effect_lexemes_veto_the_tiering_signal(self, query: str) -> None:
        assert _classify(query)["primary_intent"] != "cohort_definition"

    def test_bare_tier_token_is_not_a_tiering_ask(self) -> None:
        """bench-0263 (gold ``resource_optimizer``): "high-decile HCPs" uses a
        tier word as a MODIFIER, not as a partition to build. A bare tier-token
        pattern captured it — and would have made a currently-escalating row
        CONFIDENTLY wrong, the exact ``\\binterim\\b`` failure. It must stay
        below the pattern-trust floor and keep escalating to the LLM."""
        intent = _classify(
            "What's the optimal call-plan frequency for high-decile HCPs on Remibrutinib?"
        )
        assert intent["primary_intent"] != "cohort_definition"
        assert intent["confidence"] < PATTERN_TRUST_FLOOR


# ---------------------------------------------------------------------------
# Rows the change must leave exactly as they are.
# ---------------------------------------------------------------------------
class TestUnrelatedRowsUnchanged:
    def test_cohort_construction_rows_unchanged(self) -> None:
        # bench-0139 — already correct via the brand+patient cohort pattern.
        assert _classify_and_route(
            "For Remibrutinib, show me the NRx broken down by patient "
            "disease-severity segment (low/medium/high)."
        ) == ["cohort_profiler"]

    def test_plain_kpi_lookup_unchanged(self) -> None:
        assert _classify_and_route("What is TRx for Kisqali?") == ["explainer"]

    def test_plain_segment_ask_unchanged(self) -> None:
        """A bare segment ask with no tier container/ladder stays segment_analysis."""
        assert _classify_and_route("Which segments respond best to rep visits?") == [
            "heterogeneous_optimizer"
        ]
