"""SSOT (single source of truth) tests for multi-faceted query detection.

Issue #288 — converge three multi_faceted heuristics into one module so
future drift is observable.

The two algorithms remain distinct in behavior but each has exactly one
implementation:

  - ``MULTI_FACETED_PATTERNS`` — used by ``IntentClassifierNode``.
  - ``is_multi_faceted_facet_score`` — used by both chatbot routes.

These tests are deliberately structural (identity / ``is`` checks) so
they fail loudly if a future refactor reintroduces a parallel copy.

Falsifiability: temporarily replace the SSOT export with a stub that
returns a different shape / value and re-run; the structural assertions
trip. Verified manually as part of the PR landing for #288.
"""

from __future__ import annotations

import pytest

from src.agents.multi_faceted import (
    MULTI_FACETED_PATTERNS,
    is_multi_faceted_facet_score,
)
from src.agents.orchestrator.nodes.intent_classifier import IntentClassifierNode
from src.api.routes import chatbot_dspy, chatbot_graph


class TestPatternsAreSSOT:
    """``IntentClassifierNode.INTENT_PATTERNS['multi_faceted']`` IS the
    SSOT tuple — not a copy with the same content."""

    def test_pattern_list_is_ssot_object(self):
        # Identity, not equality. A duplicate inline list would compare
        # equal but fail this check.
        assert IntentClassifierNode.INTENT_PATTERNS["multi_faceted"] is MULTI_FACETED_PATTERNS

    def test_pattern_count_locked(self):
        # Lock the count so a silent regex addition is observable.
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
