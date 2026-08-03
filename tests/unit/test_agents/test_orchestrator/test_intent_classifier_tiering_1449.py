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
safety net. It is ALSO broad enough to steal any question that merely uses tier
language — budgets, sales territories, rep rosters, A/B arms. So the tiering
signal is gated FOUR ways:

  1. a clinical POPULATION anchor (hcp/physician/prescriber/provider/patient/
     cohort/population, or a brand/disease name) must be present AND positioned
     as the thing being partitioned — after the partition verb in shape (a),
     before the ladder in shape (b). Without it "Divide the Q3 budget into
     tiers" and "Split the sales territories into tiers" matched CONFIDENTLY;
     see ``TestNonPopulationTieringIsNotCohortWork``. "Anywhere in the query" is
     NOT enough either — "Rank call-plan tiers ... for Kisqali" partitions call
     plans, so the anchor is positional.
  2. it requires an explicit tier CONTAINER (a partition verb + tiers/buckets/
     quartiles/deciles/categories) or an explicit 3-level ordinal LADDER
     (high/medium/low, top/middle/bottom) — never a bare "tier" token
     (``"high-decile HCPs"``, bench-0263 gold ``resource_optimizer``, must
     stay out), and
  3. a treatment-effect veto: any causal/CATE/uplift/responder/effect lexeme
     anywhere in the query suppresses the tiering match entirely. The veto can
     only ever DOWNGRADE to today's behaviour, so it is the safe direction.
  4. the classifier's existing clause gate keeps the co-firing
     ``segment_analysis`` from splitting the ask into a 2-agent dispatch.

Queries are verbatim gold rows (bench-NNNN) plus authored out-of-gold
adversarial probes — behavioural pins, not phrase overfitting. The out-of-gold
probes carry the load here: the 337-row gold contains no non-clinical tiering
row, so it reported 0 losses for a build that misrouted every one of them. No
mocks: the real ``_pattern_classify`` -> ``RouterNode.execute`` chain runs.
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
# OUT-OF-DOMAIN OVER-REACH — tier language is not owned by cohort work.
#
# The first cut of #1449 gated only on the tier CONTAINER/LADDER and never on
# WHAT is being partitioned, so any partition verb + tier noun matched: budgets,
# sales territories, rep rosters, A/B arms, marketing spend. Every one of them
# landed on ``cohort_profiler`` at confidence 0.867-0.933 — above the 0.8
# pattern-trust floor, so the LLM safety net is never consulted. That is the
# ``\binterim\b`` failure (#1408) in full: rows that were previously correct (or
# that safely ESCALATED) turned CONFIDENTLY wrong, and ``cohort_profiler``'s
# resolver deliberately "never fails closed" (it profiles every supported brand
# when none is named), so the user gets a real-looking patient-cohort report for
# a question about budget or territories with no failure signal at all.
#
# The 337-row gold cannot see this class — it contains no budget/territory/rep/
# A-B-arm tiering rows — so these authored probes ARE the regression signal.
# Each expectation below is the MEASURED pre-#1449 (main a6d05a9b) behaviour.
# ---------------------------------------------------------------------------
class TestNonPopulationTieringIsNotCohortWork:
    @pytest.mark.parametrize(
        ("query", "expected_agents"),
        [
            # Resource/finance tiering — main routes these to a forecast or gap read.
            ("Divide the Q3 budget into tiers by expected impact", ["prediction_synthesizer"]),
            ("Rank call-plan tiers by expected ROI for Kisqali", ["prediction_synthesizer"]),
            ("Split the sales territories into tiers based on potential", ["gap_analyzer"]),
            # Experiment-design tiering — the arms of a test are not a patient cohort.
            (
                "Classify the A/B test arms into tiers by statistical power",
                ["experiment_designer"],
            ),
            # A bare ordinal ladder with ZERO pharma/population content anywhere.
            (
                "The forecast confidence is high, medium, or low depending on the scenario.",
                ["prediction_synthesizer"],
            ),
        ],
    )
    def test_non_population_tiering_keeps_its_pre_1449_route(
        self, query: str, expected_agents: list[str]
    ) -> None:
        assert _classify(query)["primary_intent"] != "cohort_definition"
        assert _classify_and_route(query) == expected_agents

    @pytest.mark.parametrize(
        "query",
        [
            # A sales-rep roster is a population, but not a CLINICAL one.
            "Divide the sales reps into top, middle, and bottom performers",
            "Bucket the marketing spend into quartiles",
            "Split the launch budget into high, medium and low priority buckets",
        ],
    )
    def test_non_population_tiering_keeps_escalating_to_the_llm(self, query: str) -> None:
        """These carry no deterministic signal on main (confidence 0.5) and must
        keep escalating. Turning a safely-escalating row into a CONFIDENT wrong
        answer is strictly worse than leaving it to the LLM — the #1408 lesson."""
        intent = _classify(query)
        assert intent["primary_intent"] != "cohort_definition"
        assert intent["confidence"] < PATTERN_TRUST_FLOOR

    def test_population_anchor_must_precede_the_tier_container(self) -> None:
        """A brand named AFTER the container is not the thing being partitioned.

        "Rank call-plan tiers ... for Kisqali" partitions CALL PLANS; the brand is
        incidental trailing context. An anchor test that accepted the token
        anywhere in the query would re-admit exactly this row.
        """
        assert (
            _classify("Rank call-plan tiers by expected ROI for Kisqali")["primary_intent"]
            != "cohort_definition"
        )


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
