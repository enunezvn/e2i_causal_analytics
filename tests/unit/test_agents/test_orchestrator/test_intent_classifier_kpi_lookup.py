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
from unittest.mock import AsyncMock

import pytest

from src.agents.orchestrator.nodes.intent_classifier import IntentClassifierNode


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
