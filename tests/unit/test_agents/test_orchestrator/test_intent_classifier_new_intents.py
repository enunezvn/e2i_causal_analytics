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
    # Use queries that uniquely match multi_faceted, not gap_analyzer's "gap"
    # or segment_analysis's "segment" — otherwise the tie-break (Python
    # dict order in scores) hands the win to the earlier-inserted intent.
    assert _classify("synthesize results from multiple analyses then summarize") == "multi_faceted"
    assert _classify("show me both effects across the cohort") == "multi_faceted"
    assert (
        _classify("integrate the findings with the prior research and also report")
        == "multi_faceted"
    )


def test_pattern_classifier_keeps_existing_intents() -> None:
    """Existing intents must remain stable after adding new patterns."""
    assert _classify("what causes therapy discontinuation?") == "causal_effect"
    assert _classify("where are the biggest performance gaps?") == "performance_gap"
    assert _classify("system health status") == "system_health"
    assert _classify("design an A/B test for the new outreach program") == "experiment_design"
