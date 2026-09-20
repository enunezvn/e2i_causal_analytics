"""Demo 6.5 must reach the tool composer, not a single prediction agent (#2115).

6.5 is *"Forecast Kisqali TRx volume for the next two quarters and tell me the biggest
risk to that forecast"* — two questions about two substrates. The forecast is univariate
and cannot see the risk; the risk lives in the gap and causal views. So the owner's
decision is that it routes to the TOOL COMPOSER, which plans a forecast step alongside
gap/causal steps.

The issue named one cause — ``planner.py`` mapping every PREDICTIVE ask to
``risk_scorer``. MEASURED 2026-09-20, there is an earlier one it did not: the intent
classifier scored 6.5 as ``prediction`` alone, ``requires_multi_agent=False``, because
NO pattern in any intent matched "the biggest risk to that forecast". The composer was
never reached at all, so fixing the planner's map alone would have left 6.5 refusing.

The negative controls matter more than the positive case here. This classifier carries
heavily-tuned precision gates (#1337, #1366, #1409) whose whole purpose is to keep
single asks and additive pairs OFF the composer, and a "risk" pattern is exactly the
kind of broad word that regresses them. Every locked negative below is a case that must
NOT promote.
"""

from __future__ import annotations

import pytest

from src.agents.multi_faceted import has_dependency_composition
from src.agents.orchestrator.nodes.intent_classifier import IntentClassifierNode

DEMO_6_5 = (
    "Forecast Kisqali TRx volume for the next two quarters and tell me the biggest "
    "risk to that forecast"
)


@pytest.fixture(scope="module")
def classify():
    node = IntentClassifierNode()
    return node._pattern_classify


# ------------------------------------------------------------------- the positive case
def test_demo_6_5_is_multi_faceted_so_it_reaches_the_tool_composer(classify):
    result = classify(DEMO_6_5)
    assert result["primary_intent"] == "multi_faceted"
    assert result["requires_multi_agent"] is True


def test_both_halves_of_6_5_are_recognised_as_their_own_facets(classify):
    """The forecast half and the risk half must BOTH be present, or the composer plans
    one step and the answer is half an answer."""
    secondary = set(classify(DEMO_6_5)["secondary_intents"])
    assert "prediction" in secondary, "the forecast half"
    assert "causal_effect" in secondary, "the risk half — what threatens the forecast"


@pytest.mark.parametrize(
    "query",
    [
        DEMO_6_5,
        "Project NRx through Q2 and what is the biggest risk to that projection",
        "Forecast Fabhalta TRx for two quarters and the main risk to that outlook",
    ],
)
def test_an_anaphoric_risk_clause_is_a_dependency_marker(query):
    """'risk to THAT forecast' points back at the previous clause's result. Without the
    marker the promotion gate never fires, however the clause is scored."""
    assert has_dependency_composition(query)


# -------------------------------------------------------------- the locked negatives
@pytest.mark.parametrize(
    "query",
    [
        # A forecast alone is ONE ask. This is the case that must not drift.
        "Forecast Kisqali TRx volume for the next two quarters",
        "What will NRx be by the end of the year?",
        # Entity risk scoring is one ask for one agent, and says nothing about a forecast.
        "Which HCP segments are highest risk of churn?",
        "Score the risk of discontinuation for these patients",
        # 'risk' as an ordinary noun in a single ask.
        "What is the risk of a stockout in the midwest?",
        "Show me the risk scores for Fabhalta prescribers",
        # An additive pair is parallel, not a dependent pipeline.
        "What is Kisqali TRx and what is Fabhalta TRx",
    ],
)
def test_these_still_do_not_promote_to_the_composer(classify, query):
    """A gate that only proved the new route would also pass if EVERYTHING promoted."""
    result = classify(query)
    assert result["primary_intent"] != "multi_faceted", (
        f"{query!r} was promoted to the composer: {result}"
    )


@pytest.mark.parametrize(
    "query",
    [
        "Which HCP segments are highest risk of churn?",
        "What is the risk of a stockout in the midwest?",
        "Show me the risk scores for Fabhalta prescribers",
        "Forecast Kisqali TRx volume for the next two quarters",
    ],
)
def test_a_bare_risk_word_is_not_a_dependency_marker(query):
    """The marker is ANAPHORIC — it must point back at a prior clause's result."""
    assert not has_dependency_composition(query)


def test_the_forecast_only_ask_still_routes_to_prediction(classify):
    """The single-tool path has to keep working: not every forecast is a 6.5."""
    result = classify("Forecast Kisqali TRx volume for the next two quarters")
    assert result["primary_intent"] == "prediction"
    assert result["requires_multi_agent"] is False
