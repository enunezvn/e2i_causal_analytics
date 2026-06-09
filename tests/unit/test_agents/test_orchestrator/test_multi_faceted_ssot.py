"""SSOT (single source of truth) tests for multi-faceted query detection.

Issue #288 — converge three multi_faceted heuristics into one module so
future drift is observable. Issue #295 — extend convergence to the
fourth detector (copilotkit analytics-labeling helper).

Three algorithms remain distinct in behavior but each has exactly one
implementation:

  - ``MULTI_FACETED_PATTERNS`` — used by ``IntentClassifierNode``.
  - ``is_multi_faceted_facet_score`` — used by both chatbot routes.
  - ``is_multi_faceted_topic_count`` — used by
    ``copilotkit._classify_query_type`` (issue #295).

These tests are deliberately structural (identity / ``is`` checks) so
they fail loudly if a future refactor reintroduces a parallel copy.

Falsifiability: temporarily replace the SSOT export with a stub that
returns a different shape / value and re-run; the structural assertions
trip. Verified manually as part of the PR landing for #288 and #295.
"""

from __future__ import annotations

import pytest

from src.agents.multi_faceted import (
    MULTI_FACETED_PATTERNS,
    TOPIC_COUNT_KEYWORD_GROUPS,
    is_multi_faceted_facet_score,
    is_multi_faceted_topic_count,
)
from src.agents.orchestrator.nodes.intent_classifier import IntentClassifierNode
from src.api.routes import chatbot_dspy, chatbot_graph, copilotkit


class TestPatternsAreSSOT:
    """``IntentClassifierNode.INTENT_PATTERNS['multi_faceted']`` IS the
    SSOT tuple — not a copy with the same content."""

    def test_pattern_list_is_ssot_object(self):
        # Identity, not equality. A duplicate inline list would compare
        # equal but fail this check.
        assert IntentClassifierNode.INTENT_PATTERNS["multi_faceted"] is MULTI_FACETED_PATTERNS

    def test_pattern_count_locked(self):
        # Lock the count so a silent regex addition is observable.
        # 2026-06-09: dependent-pipeline routing is done via classifier promotion
        # on >=2 strong intents + a dependency marker (multi_faceted.
        # has_sequential_composition), NOT extra patterns — a bare "then <verb>"
        # pattern over-routes single asks. So the SSOT pattern set stays at 4.
        assert len(MULTI_FACETED_PATTERNS) == 4

    def test_patterns_are_immutable(self):
        assert isinstance(MULTI_FACETED_PATTERNS, tuple)


class TestChatbotRoutesDelegateToSSOT:
    """Both ``_is_multi_faceted_query`` module-level callables in the
    chatbot routes ARE the SSOT helper — re-exports, not wrappers."""

    def test_chatbot_graph_is_ssot(self):
        assert chatbot_graph._is_multi_faceted_query is is_multi_faceted_facet_score

    def test_chatbot_dspy_is_ssot(self):
        assert chatbot_dspy._is_multi_faceted_query is is_multi_faceted_facet_score


class TestFacetScorerBehaviorLockedIn:
    """Behavior parity with the pre-refactor duplicate. Lock the current
    semantics so a future change to facet weights / words is observable.
    """

    @pytest.mark.parametrize(
        "query,expected",
        [
            # Single facet: conjunction_keywords only → False (need ≥2)
            ("compare brand X to brand Y", False),
            # Two facets: cross_agent + conjunction_keywords ("explain") → True
            ("explain the causal factors behind the drop", True),
            # Two facets: multiple_kpis + conjunction_keywords → True
            ("compare trx and nrx", True),
            # Two facets: multiple_brands + conjunction_keywords ("compare") → True
            ("compare kisqali and fabhalta", True),
            # Two facets: analysis_and_recommendation + cross_agent ("causal") → True
            ("why did the causal effect drop, what should we do", True),
            # Zero facets → False
            ("hello world", False),
        ],
    )
    def test_facet_scorer_known_queries(self, query, expected):
        assert is_multi_faceted_facet_score(query) is expected

    def test_returns_bool_not_int(self):
        # The function previously returned ``sum(...) >= 2`` which is
        # already a bool, but lock the type so a refactor to ``sum(...)``
        # (truthy-int return) is observable.
        result = is_multi_faceted_facet_score("compare trx and nrx")
        assert isinstance(result, bool)


class TestTopicCountScorerBehaviorLockedIn:
    """Issue #295: lock the analytics-label topic-count semantics so a
    future change to the 5 keyword groups is observable. The SSOT
    function ``is_multi_faceted_topic_count`` replaced the inline
    heuristic at ``copilotkit.py:1006-1014``.
    """

    def test_topic_groups_count_locked(self):
        # The pre-#295 inline heuristic was 5 groups; a silent addition
        # or removal of a group is observable.
        assert len(TOPIC_COUNT_KEYWORD_GROUPS) == 5

    def test_topic_groups_are_immutable(self):
        assert isinstance(TOPIC_COUNT_KEYWORD_GROUPS, tuple)
        for group in TOPIC_COUNT_KEYWORD_GROUPS:
            assert isinstance(group, tuple)

    def test_topic_groups_content_locked(self):
        # Lock the exact pre-#295 keyword content so any future change
        # to the analytics-labeling semantics is observable.
        assert TOPIC_COUNT_KEYWORD_GROUPS == (
            ("trx", "nrx", "kpi", "metric", "performance"),
            ("causal", "impact", "effect", "intervention"),
            ("predict", "forecast", "future"),
            ("experiment", "test", "ab test", "a/b"),
            ("drift", "shift", "degradation"),
        )

    @pytest.mark.parametrize(
        "query,expected",
        [
            # Two groups: KPI + causal → True
            ("show trx and the causal effect of the campaign", True),
            # Two groups: KPI + drift → True
            ("forecast trx and detect drift", True),
            # Two groups: experiment + predict → True
            ("run an experiment to predict future trx", True),
            # Single group: KPI only → False
            ("what is the trx for kisqali", False),
            # Single group: drift only → False
            ("any drift in the data", False),
            # Zero groups → False
            ("hello world", False),
        ],
    )
    def test_topic_count_known_queries(self, query, expected):
        assert is_multi_faceted_topic_count(query) is expected

    def test_returns_bool_not_int(self):
        result = is_multi_faceted_topic_count("forecast trx and detect drift")
        assert isinstance(result, bool)


class TestCopilotKitDelegatesToSSOT:
    """Issue #295: ``copilotkit._classify_query_type`` must use the SSOT
    helper rather than re-implementing the topic-count heuristic. We
    assert the imported callable IS the SSOT object (not a wrapper).
    """

    def test_copilotkit_imports_ssot_helper(self):
        assert copilotkit.is_multi_faceted_topic_count is is_multi_faceted_topic_count


class TestMultiFacetedPatternsBehaviorLockedIn:
    """Smoke test that the 4 regexes still match the canonical phrases
    they were written for. Falsifiability: replacing any one regex with
    a never-match sentinel trips the corresponding positive case.
    """

    @pytest.mark.parametrize(
        "phrase",
        [
            "and also",
            "and then",
            "and additionally",
            "and furthermore",
            "compare X vs Y and",
            "combine these analyses",
            "integrate the results",
            "synthesize findings",
            "both effects",
            "multiple analyses",
            "both perspectives",
        ],
    )
    def test_pattern_matches_canonical_phrase(self, phrase):
        import re

        # IGNORECASE mirrors the production call shape at
        # ``intent_classifier._pattern_classify`` so a regression breaking
        # case-insensitive matching would trip this test.
        assert any(re.search(p, phrase, re.IGNORECASE) for p in MULTI_FACETED_PATTERNS), (
            f"No MULTI_FACETED_PATTERNS regex matched canonical phrase: {phrase!r}"
        )
