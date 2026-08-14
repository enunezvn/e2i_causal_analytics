"""DomainMapper learns the KPI-value-lookup SSOT, narrowed (#1593).

Stage 2 used to have no notion of a KPI value lookup, so ``what is the TRx for
Kisqali`` scored no domain above ``CONFIDENCE_THRESHOLD`` and the whole 4-stage
pipeline abstained — 46 of the 54 KPI-lookup rows in the 337-row #1337 gold set.
This teaches the mapper the SAME ``KPI_VALUE_LOOKUP_RE`` the intent classifier
routes on and the dispatcher binds evidence with (#1475), so the fast path and
the answerability contract cannot drift apart.

The fast path is NARROWED, because in active mode relabeling the classifier's
decision IS a routing change (``RouterNode._dispatch_from_classification``).
Two query shapes measured as active-mode degradations on the gold set must keep
abstaining — the pipeline yields and legacy routing proceeds unchanged:

- **population breakdown** — the ask is a per-segment decomposition, not one
  figure. The explainer resolver binds a single scalar, so the fast path's own
  answerability premise fails; cohort_profiler owns per-segment counts.
  (gold bench-0008 / 0133 / 0139 / 0140 / 0141, all gold ``cohort_profiler``)
- **compound ask** — a second wh-clause after a connector is a second facet a
  lone explainer would silently drop.
  (gold bench-0143, gold ``PARALLEL_DELEGATION[explainer, gap_analyzer]``)

Queries are the verbatim gold rows so the pins track the measured set, not a
paraphrase of it.
"""

from __future__ import annotations

import pytest

from src.agents.orchestrator.classifier import ClassificationPipeline
from src.agents.orchestrator.classifier.domain_mapper import DomainMapper
from src.agents.orchestrator.classifier.feature_extractor import FeatureExtractor
from src.agents.orchestrator.classifier.schemas import Domain
from src.agents.orchestrator.nodes.router import RouterNode

# --- verbatim #1337 gold rows ------------------------------------------------
KPI_LOOKUPS = [
    "What is TRx for Kisqali?",  # bench-0000
    "whats the TRx for kisqali??",  # bench-0164 — double '?', NOT a compound ask
    "What is the conversion rate metric for Remibrutinib across our patient population?",
]

POPULATION_BREAKDOWNS = [
    "Give me an NRx breakdown by patient clinical segment for Remibrutinib",  # bench-0008
    "What are the NRx numbers for different patient segments for Remibrutinib?",  # bench-0133
    (  # bench-0139
        "For Remibrutinib, show me the NRx broken down by patient disease-severity "
        "segment (low/medium/high)."
    ),
    (  # bench-0140
        "Give me the last-30-day NRx breakdown for Remibrutinib split by "
        "biologic-naive vs biologic-experienced patients."
    ),
    (  # bench-0141
        "Give me the last-30-day NRx breakdown for Fabhalta split by "
        "biologic-naive vs biologic-experienced patients."
    ),
]

COMPOUND_ASKS = [
    "What is the current total TRx and which region has the largest gap opportunity?",  # 0143
    "What is TRx and how is it calculated?",  # bench-0064
    "What is the current Total TRx and which brand leads?",  # bench-0135
]

CONTESTED = POPULATION_BREAKDOWNS + COMPOUND_ASKS


def _map(query: str):
    return DomainMapper().map_domains(FeatureExtractor().extract(query))


class TestKpiFastPathReadsTheSSOT:
    def test_imports_the_intent_classifier_pattern_not_a_copy(self):
        """A forked regex would let 'routes to explainer' and 'explainer can
        answer it' drift apart — the #1475 SSOT invariant."""
        from src.agents.orchestrator.classifier import domain_mapper
        from src.agents.orchestrator.nodes.intent_classifier import KPI_VALUE_LOOKUP_RE

        assert domain_mapper._kpi_value_lookup_re() is KPI_VALUE_LOOKUP_RE

    @pytest.mark.parametrize("query", KPI_LOOKUPS)
    def test_kpi_lookup_becomes_primary_explanation(self, query):
        mapping = _map(query)
        assert mapping.primary_domain == Domain.EXPLANATION, query
        top = mapping.domains_detected[0]
        # PatternSelector Rule 2 (explanation override) needs > 0.7.
        assert top.confidence > 0.7, query
        assert "kpi_value_lookup" in top.evidence, query

    def test_forecast_asks_inherit_the_ssot_veto(self):
        """The SSOT's \\A-anchored prediction guard must still apply — a forecast
        ask may not be answered with a current-period figure."""
        mapping = _map("what is the trx for next quarter expected to be?")
        assert Domain.EXPLANATION not in [dm.domain for dm in mapping.domains_detected]

    def test_non_kpi_queries_are_untouched(self):
        mapping = _map("Good morning team")
        assert mapping.domain_count == 0

    def test_fast_path_floors_explanation_never_downgrades_it(self):
        """A query that ALSO scores EXPLANATION on real keyword evidence above
        KPI_LOOKUP_CONFIDENCE must keep the higher score and its evidence — the
        fast path may only ever raise this domain."""
        from src.agents.orchestrator.classifier.domain_mapper import KPI_LOOKUP_CONFIDENCE

        query = "whats TRx mean? explain how"
        keyword_only = DomainMapper()._score_domain(
            Domain.EXPLANATION, FeatureExtractor().extract(query)
        )[0]
        assert keyword_only > KPI_LOOKUP_CONFIDENCE, "fixture no longer exercises the floor"

        top = _map(query).domains_detected[0]
        assert top.domain == Domain.EXPLANATION
        assert top.confidence == pytest.approx(keyword_only, abs=1e-3)
        assert "kpi_value_lookup" in top.evidence
        assert len(top.evidence) > 1, "keyword evidence must survive alongside the marker"

    def test_every_detected_domain_still_carries_evidence(self):
        for query in KPI_LOOKUPS:
            for dm in _map(query).domains_detected:
                assert dm.evidence, f"{dm.domain} detected without evidence on {query!r}"


class TestNarrowingKeepsContestedRowsAbstaining:
    @pytest.mark.parametrize("query", CONTESTED)
    def test_contested_shapes_do_not_take_the_fast_path(self, query):
        mapping = _map(query)
        evidence = [e for dm in mapping.domains_detected for e in dm.evidence]
        assert "kpi_value_lookup" not in evidence, query

    @pytest.mark.parametrize("query", POPULATION_BREAKDOWNS)
    def test_population_breakdown_is_recognised(self, query):
        from src.agents.orchestrator.classifier import domain_mapper

        assert domain_mapper._is_population_breakdown(query) is True, query

    @pytest.mark.parametrize("query", KPI_LOOKUPS)
    def test_plain_lookups_are_not_population_breakdowns(self, query):
        from src.agents.orchestrator.classifier import domain_mapper

        assert domain_mapper._is_population_breakdown(query) is False, query

    def test_compound_veto_reads_clause_structure_not_question_marks(self):
        """bench-0164 ('whats the TRx for kisqali??') has TWO '?' but one ask.
        Counting question marks would veto it; the compound signal must not."""
        feats = FeatureExtractor().extract("whats the TRx for kisqali??")
        assert feats.structural.question_count > 1
        assert feats.structural.has_compound_question is False

        compound = FeatureExtractor().extract(COMPOUND_ASKS[0])
        assert compound.structural.has_compound_question is True


# Measured on the 337-row gold set: every contested row's classification is
# byte-identical with and without the fast path. Spelled out rather than
# derived, so a regression has to edit the table instead of moving with the bug.
# (bench-0139 lands at 0.457, BELOW RouterNode.MIN_ACTIVE_CONFIDENCE, so the
# router declines it too; bench-0064 already routed to a lone explainer on main
# — the narrowing preserves that, it does not create it.)
CONTESTED_EXPECTATIONS = {
    POPULATION_BREAKDOWNS[0]: ("CLARIFICATION_NEEDED", [], None),  # bench-0008
    POPULATION_BREAKDOWNS[1]: ("CLARIFICATION_NEEDED", [], None),  # bench-0133
    POPULATION_BREAKDOWNS[2]: (  # bench-0139
        "PARALLEL_DELEGATION",
        ["explainer", "heterogeneous_optimizer"],
        None,
    ),
    POPULATION_BREAKDOWNS[3]: ("CLARIFICATION_NEEDED", [], None),  # bench-0140
    POPULATION_BREAKDOWNS[4]: ("CLARIFICATION_NEEDED", [], None),  # bench-0141
    COMPOUND_ASKS[0]: ("CLARIFICATION_NEEDED", [], None),  # bench-0143
    COMPOUND_ASKS[1]: ("SINGLE_AGENT", ["explainer"], ["explainer"]),  # bench-0064
    COMPOUND_ASKS[2]: ("CLARIFICATION_NEEDED", [], None),  # bench-0135
}


class TestActiveModeDispatchUnchangedForContestedRows:
    """The narrowing's whole point: these rows must still fall through to
    legacy routing exactly as they did before the fast path existed."""

    @pytest.mark.parametrize("query", CONTESTED)
    async def test_classification_and_plan_match_the_measured_table(self, query):
        pattern, agents, expected_plan = CONTESTED_EXPECTATIONS[query]
        result = await ClassificationPipeline(llm_client=None, enable_llm_layer=False).classify(
            query=query
        )
        assert result.routing_pattern.value == pattern, query
        assert sorted(result.target_agents) == agents, query

        plan = RouterNode()._dispatch_from_classification(
            {
                "routing_pattern": result.routing_pattern.value,
                "target_agents": result.target_agents,
                "confidence": result.confidence,
            }
        )
        if expected_plan is None:
            assert plan is None, f"{query} must abstain to legacy routing"
        else:
            assert plan is not None and [d["agent_name"] for d in plan[0]] == expected_plan, query

    def test_no_contested_row_becomes_a_lone_explainer_via_the_fast_path(self):
        """The measured degradation: a lone explainer standing in for
        cohort_profiler, or a dropped second facet."""
        for query in POPULATION_BREAKDOWNS + [COMPOUND_ASKS[0]]:
            pattern, agents, _ = CONTESTED_EXPECTATIONS[query]
            assert agents != ["explainer"], query

    @pytest.mark.parametrize("query", KPI_LOOKUPS)
    async def test_plain_kpi_lookup_now_dispatches_explainer(self, query):
        result = await ClassificationPipeline(llm_client=None, enable_llm_layer=False).classify(
            query=query
        )
        assert result.routing_pattern.value == "SINGLE_AGENT", query
        assert result.target_agents == ["explainer"], query
        plan = RouterNode()._dispatch_from_classification(
            {
                "routing_pattern": result.routing_pattern.value,
                "target_agents": result.target_agents,
                "confidence": result.confidence,
            }
        )
        assert plan is not None, query
        assert [d["agent_name"] for d in plan[0]] == ["explainer"], query
