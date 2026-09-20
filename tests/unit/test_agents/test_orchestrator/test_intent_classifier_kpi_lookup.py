"""KPI value-lookup classification + LLM-layer observability.

Two defects unmasked by PR #1364 (which revived the previously-dead
``_llm_classify`` parse path):

1. KPI value lookups ("What is TRx for Kisqali?") have NO pattern in
   INTENT_PATTERNS, so they fall to the LLM layer, where haiku
   deterministically answers ``prediction@0.85`` (its intent menu offers no
   metric-lookup category) → router dispatches prediction_synthesizer →
   fail-closed on every chat KPI ask. Pre-#1364 the dead parse fell back to
   general@0.3, and the router's general-default (explainer) was accidentally
   the CORRECT target per the #1337 gold labels (kpi_query → explainer,
   largest gold class at 111/337 rows). The pattern must catch this class
   deterministically so routing no longer depends on that accident.

2. ``_llm_classify`` success is silent — only the parse-failure path logs.
   A prod log grep therefore cannot positively verify the layer works
   (live-verify 2026-07-30 required a docker-exec probe instead).
"""

from __future__ import annotations

import logging
import re
from unittest.mock import AsyncMock

import pytest

from src.agents.orchestrator.nodes.intent_classifier import (
    KPI_VALUE_LOOKUP_RE,
    IntentClassifierNode,
)
from src.kpi.business_metric_vocabulary import KPI_VALUE_LOOKUP_METRIC_PATTERN


def _pattern(query: str):
    node = IntentClassifierNode()
    return node._pattern_classify(query.lower())


class TestKpiValueLookupPattern:
    """Real-traffic KPI value-lookup shapes (verbatim from the 337-query
    #1337 benchmark, gold = SINGLE_AGENT:explainer) must pattern-classify as
    ``explanation`` at >=0.8 so they never reach the LLM layer."""

    @pytest.mark.parametrize(
        "query",
        [
            "What is TRx for Kisqali?",  # bench-0000; live misroute 2026-07-30
            "What is the current TRx volume for Fabhalta?",  # bench-0015
            "What is the market share for Kisqali compared to competitors?",  # bench-0002
            "Show me Kisqali TRx for the last 30 days",  # bench-0075
            "Show me Total TRx for Kisqali brand in US",  # bench-0073
            "whats the current TRx volume for fabhalta rn?",  # bench-0194 (typo shape)
            "what is teh NBRx for kisqali?",  # bench-0100 (typo shape)
            "tell me about the remibrutinib NRx for the past 90 days",  # bench-0092
            "What is the current market share of Remibrutinib compared to Xolair?",  # bench-0128
            "what is the NBRx for Kisqali in the past month?",  # bench-0095
            "What is the current TRx?",  # bench-0121 (no brand)
            "What is NRx?",  # bench-0052 (bare metric definition ask)
            # #2130 follow-up: shapes the old 3-word gap admitted and the constrained
            # target grammar must keep (gold/real traffic, measured 2026-09-19).
            "What is Remibrutinib market share?",  # gold: explainer
            "Can you show me the trend of Remibrutinib NBRx over the past 6 months?",  # gold
            "Can you show me how Kisqali's total prescriptions have evolved throughout the fourth quarter?",  # gold
            "Show me how Remibrutinib's TRx trend over the last 30 days compares to Kisqali and Fabhalta.",  # gold
            "What was the weekly TRx share for Remibrutinib across the Southeast region recently?",  # real traffic
            "What was the weekly TRx trajectory for Remibrutinib across the Southeast during Q2?",  # real traffic
            # codex r4: parse_window accepts hyphenated month ranges; main counts
            # "Jan-Mar" as one word, so the grammar must too.
            "What is Jan-Mar 2025 TRx?",
            "What is January-March 2025 TRx?",
            # codex r5: heads the resolver's governing-head guard binds.
            "What is the current level of TRx for Kisqali?",
            "Give me the latest amount of NRx for Fabhalta",
            "What is the current figure of NBRx for Kisqali?",
            "Show me the sum of TRx for Kisqali",
            "What are the current values of TRx for Kisqali?",
        ],
    )
    def test_kpi_value_lookup_is_explanation(self, query: str) -> None:
        result = _pattern(query)
        assert result["primary_intent"] == "explanation", (
            f"{query!r} classified as {result['primary_intent']!r}; KPI value "
            "lookups must route to explanation → explainer (gold #1337)"
        )
        assert result["confidence"] >= 0.8, (
            f"{query!r} scored {result['confidence']}; must clear the 0.8 "
            "pattern threshold so the LLM layer is never engaged"
        )

    @pytest.mark.parametrize(
        ("query", "expected_intent"),
        [
            # Causal asks that MENTION a KPI keep their specific intent
            # (word-bounded gap in the KPI pattern; INTENT_PRIORITY tie-break
            # ranks causal_effect above explanation as backstop).
            (
                "What is the causal impact of rep visits on TRx for Kisqali?",
                "causal_effect",  # bench-0004
            ),
            (
                "What is driving the drop in Remibrutinib NRx in the northeast region?",
                "causal_effect",  # bench-0010
            ),
            # Forecast asks stay prediction: no value-lookup verb present.
            (
                "Forecast Kisqali TRx volume for the next two quarters",
                "prediction",  # bench-0038 shape
            ),
            (
                "Predict the TRx likelihood for Fabhalta next quarter",
                "prediction",
            ),
        ],
    )
    def test_kpi_mention_does_not_hijack_specific_intents(
        self, query: str, expected_intent: str
    ) -> None:
        result = _pattern(query)
        assert result["primary_intent"] == expected_intent, (
            f"{query!r} classified as {result['primary_intent']!r}, "
            f"expected {expected_intent!r} — the KPI lookup pattern must not "
            "hijack causal/forecast asks that merely mention a metric"
        )

    @pytest.mark.parametrize(
        "query",
        [
            # codex iter-1 MEDIUM: value-lookup opener + forecast noun must NOT
            # co-score explanation — (prediction, explanation) is a deliberate
            # MULTI_AGENT_PATTERNS pair, so a spurious explanation match makes
            # a pure forecast ask double-dispatch [prediction_synthesizer,
            # explainer].
            "show me the TRx forecast for Kisqali",
            "What is the predicted TRx for Kisqali next quarter?",
            "What is the expected TRx next quarter?",
            "what is the trx projection for fabhalta?",
            # codex iter-2 MEDIUM: forecast lexeme AFTER the metric as a
            # participle/adjective — the post-metric guard must cover the
            # same stems as the gap-word guard, not just the noun forms.
            "what is the trx expected next quarter",
            "What is the TRx projected for next quarter?",
            "what is the trx predictive outlook",
            # codex iter-3 MEDIUM: forecast lexeme separated from the metric
            # by punctuation or intervening tokens — token-local guards can't
            # close this family; the pattern must reject any query containing
            # a prediction lexeme ANYWHERE (the whole-query leading guard).
            "what is the trx, expected next quarter?",
            "what is the trx for next quarter expected to be?",
            "what is the market share for next quarter expected to be?",
            "what is the conversion rate for q4 projected to be?",
            "What is the likelihood of TRx growth for Kisqali next quarter?",
            # codex iter-4 LOW: pin the probability phrasing explicitly (the
            # guard's probabilit stem), not just the likelihood form.
            "what is the probability of NRx growth for Fabhalta next month?",
        ],
    )
    def test_forecast_hybrids_stay_single_prediction(self, query: str) -> None:
        result = _pattern(query)
        assert result["primary_intent"] == "prediction", (
            f"{query!r} classified as {result['primary_intent']!r}; forecast "
            "asks that name a metric must stay prediction"
        )
        assert result["requires_multi_agent"] is False, (
            f"{query!r} set requires_multi_agent — the KPI pattern must not "
            "co-score explanation on forecast asks, or the router emits the "
            "(prediction, explanation) parallel pair for a single-intent query"
        )
        assert "explanation" not in result["secondary_intents"], (
            f"{query!r} has explanation in secondary_intents "
            f"{result['secondary_intents']!r}; the KPI lookup pattern matched "
            "a forecast ask"
        )

    def test_llm_prompt_menu_teaches_kpi_lookup(self) -> None:
        """The ``_llm_classify`` intent menu must name KPI/metric value
        lookups under ``explanation`` so ambiguous fragments the pattern
        can't catch ("trx for kisqali") stop landing on ``prediction``."""
        import inspect

        source = inspect.getsource(IntentClassifierNode._llm_classify)
        menu_line = next(
            line for line in source.splitlines() if line.strip().startswith("- explanation:")
        )
        assert "KPI" in menu_line or "metric" in menu_line.lower(), (
            "the explanation menu entry must mention KPI/metric value lookups"
        )


class TestLlmClassifySuccessLogging:
    """``_llm_classify`` must log its outcome on the SUCCESS path (the
    failure path already warns). Absence-of-warning is not positive
    verification; live-verify needed a docker-exec probe because success
    was silent."""

    @pytest.mark.asyncio
    async def test_success_emits_info_log(self, caplog: pytest.LogCaptureFixture) -> None:
        node = IntentClassifierNode()
        response = AsyncMock()
        response.content = (
            '```json\n{"primary_intent": "system_health", "confidence": 0.95,'
            ' "requires_multi_agent": false}\n```'
        )
        node.llm = AsyncMock()
        node.llm.ainvoke = AsyncMock(return_value=response)

        with caplog.at_level(
            logging.INFO, logger="src.agents.orchestrator.nodes.intent_classifier"
        ):
            result = await node._llm_classify("is the model pipeline healthy?")

        assert result["primary_intent"] == "system_health"
        success_logs = [
            r
            for r in caplog.records
            if r.levelno == logging.INFO and "system_health" in r.getMessage()
        ]
        assert success_logs, (
            "expected an INFO log naming the classified intent on the "
            "_llm_classify success path; got none"
        )
        assert "0.95" in success_logs[0].getMessage(), "the success log must include the confidence"


class TestHowManyCannotReachALaterCue:
    """#2130 (codex r4 HIGH-2): a query that says "how many" anywhere may only match
    through the how-many branch. Otherwise the leading `.*?` skips the entity-count
    subject and latches onto a later ordinary cue, and 'How many people asked, "What
    is TRx?"' bound the TRx scalar -- the exact defect this PR exists to remove,
    reached by another route. Widening the entity-noun list would not close it.
    """

    @pytest.mark.parametrize(
        "query",
        [
            'How many people asked, "What is TRx?"',
            "How many pharmacies requested: give me NRx?",
            "How many individuals want to know what is Kisqali TRx?",
            "How many competitors say show me TRx?",
            # codex r5: a NESTED how-many, and spelling variants of the cue itself.
            'How many people asked, "How many TRx?"',
            "How many pharmacies requested: how many NRx?",
            'How\nmany pharmacies asked, "What is TRx?"',
            'How  many pharmacies asked, "What is TRx?"',
            'How-many pharmacies asked, "What is TRx?"',
            # codex r6: the normalised spelling REFUSES but must never MATCH — main's
            # positive cue is the literal "how many", so admitting "How-many TRx?"
            # here would make this grammar a superset of main's.
            "How-many TRx?",
            "HOW-MANY Kisqali TRx?",
        ],
    )
    def test_a_later_cue_does_not_rescue_an_entity_count_ask(self, query: str) -> None:
        assert not KPI_VALUE_LOOKUP_RE.search(query)
        assert _pattern(query)["primary_intent"] != "explanation"

    @pytest.mark.parametrize(
        "query",
        [
            "How many total prescriptions did Kisqali have?",
            "How many new-to-brand prescriptions were there?",
        ],
    )
    def test_a_genuine_how_many_kpi_ask_still_routes(self, query: str) -> None:
        assert KPI_VALUE_LOOKUP_RE.search(query)


class TestOnlyChronologicalMonthRanges:
    """#2130 (codex r5 HIGH-3): parse_window REJECTS a reversed month range, and
    _window_from_query turns that rejection into "no window" — so a reversed range
    would be answered with a DEFAULT-period figure for an explicitly scoped ask
    (measured on main: 'What is March-Jan 2025 TRx?' binds WS3-BI-005 with no window).
    Only chronological ranges are admitted, so the reversed form never reaches the
    deterministic path. The underlying "explicit but invalid window is silently
    dropped" defect is main's and is filed separately.
    """

    @pytest.mark.parametrize(
        "query", ["What is Jan-Mar 2025 TRx?", "What is January-March 2025 TRx?"]
    )
    def test_a_chronological_range_routes(self, query: str) -> None:
        assert KPI_VALUE_LOOKUP_RE.search(query)

    @pytest.mark.parametrize("query", ["What is March-Jan 2025 TRx?", "What is Dec-Feb 2025 TRx?"])
    def test_a_reversed_range_does_not(self, query: str) -> None:
        assert not KPI_VALUE_LOOKUP_RE.search(query)


class TestKpiValueLookupSubsetInvariant:
    """#2130 (codex r3 HIGH-1): the constrained grammar must match a SUBSET of the
    shape main routed — a determiner plus at most three PHYSICAL words before the
    metric. Counting scope-pattern repetitions is NOT that budget: one scope element
    can span several words ("new england", "patient panel", "u.s."), and the branch
    then matched "What is New England and West TRx?", which main refused and whose
    resolver served a NATIONAL figure for a two-region ask.

    The reference below is rebuilt here from the public metric vocabulary, so it stays
    an INDEPENDENT statement of main's shape rather than a copy of the implementation.
    """

    _MAIN_SHAPE = re.compile(
        r"(?s)\A.*?(?:what(?:'?s| is| are| was| were)|show me|tell me about|how many"
        r"|give me)\s+(?:teh\s+|the\s+)?(?:[\w'-]+\s+){0,3}?"
        + KPI_VALUE_LOOKUP_METRIC_PATTERN
        + r"\b",
        re.IGNORECASE,
    )

    @pytest.mark.parametrize(
        "query",
        [
            # codex r3 counterexamples: branch-only matches under the old budget.
            "What is New England and West TRx?",
            "What is New England region total TRx?",
            "What is U.S. TRx?",
            "What is Kisqali vs. TRx?",
            "What is patient panel total current TRx?",
            "What is West Coast's total current TRx?",
            # scoped shapes that must keep matching
            "What is the Northeast region TRx for Kisqali?",
            "Show me New England TRx for Kisqali",
            "What is PNH TRx?",
            "What is March 2026 TRx?",
            "What is 2026-01-01 to 2026-02-01 TRx?",
            "Show me last 30 days NRx for Fabhalta",
            "what is teh currnt TRx for Kisqali?",
            "Show me the competitor comparison market share for Kisqali",
            # entity-count asks: must match neither
            "How many people received new prescriptions?",
            "How many pharmacies filled new prescriptions?",
            "Show me patients receiving new prescriptions",
            'How many people asked, "What is TRx?"',
            "What is Jan-Mar 2025 TRx?",
            "How-many TRx?",
            "HOW-MANY Kisqali TRx?",
        ],
    )
    def test_every_match_is_also_a_main_shape_match(self, query: str) -> None:
        if KPI_VALUE_LOOKUP_RE.search(query):
            assert self._MAIN_SHAPE.search(query), (
                f"{query!r} matches the lookup pattern but NOT main's "
                "determiner-plus-three-words shape; the subset invariant is broken"
            )
