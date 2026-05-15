"""Intent classifier pattern coverage for newly wired intents.

These tests check the pattern-matching layer only (no LLM). The async
``execute`` path falls back to the LLM when no pattern hits, which we don't
exercise here.
"""

from __future__ import annotations

from src.agents.orchestrator.nodes.intent_classifier import IntentClassifierNode


def _classify(query: str) -> str:
    node = IntentClassifierNode()
    result = node._pattern_classify(query.lower())
    return result.get("primary_intent", "general")


def test_pattern_classifier_detects_experiment_monitor() -> None:
    assert _classify("check all active A/B experiments for SRM") == "experiment_monitor"
    assert _classify("monitor the running trial enrollment") == "experiment_monitor"
    assert (
        _classify("are there interim analysis triggers in the current experiments?")
        == "experiment_monitor"
    )


def test_pattern_classifier_detects_multi_faceted() -> None:
    assert (
        _classify("compare HCP visits vs prior treatments and also identify high-risk segments")
        == "multi_faceted"
    )
    assert (
        _classify("combine the causal results with the gap analyses for the brand")
        == "multi_faceted"
    )
    assert _classify("show me both effects across regions") == "multi_faceted"


def test_pattern_classifier_keeps_existing_intents() -> None:
    """Existing intents must remain stable after adding new patterns."""
    assert _classify("what causes therapy discontinuation?") == "causal_effect"
    assert _classify("where are the biggest performance gaps?") == "performance_gap"
    assert _classify("system health status") == "system_health"
    assert _classify("design an A/B test for the new outreach program") == "experiment_design"
