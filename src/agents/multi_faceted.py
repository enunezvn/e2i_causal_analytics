"""Single source of truth for multi-faceted query detection (issues #288, #295).

Three distinct algorithms participate in classifying queries as
"multi-faceted":

1. ``MULTI_FACETED_PATTERNS`` — 4 conjunctive-marker regexes. Consumed by
   ``IntentClassifierNode.INTENT_PATTERNS['multi_faceted']`` and scored
   alongside every other intent in
   ``src/agents/orchestrator/nodes/intent_classifier.py``. Wins ties via
   ``INTENT_PRIORITY``.

2. ``is_multi_faceted_facet_score`` — 5-facet boolean detector
   (conjunction keywords, multiple KPIs, cross-agent capabilities,
   multiple brands, analysis+recommendation). Returns ``True`` when ≥2
   facets fire. Consumed by ``src/api/routes/chatbot_graph.py`` and
   ``src/api/routes/chatbot_dspy.py`` as a pre-classification flag.

3. ``is_multi_faceted_topic_count`` — 5-topic-group boolean detector
   (KPI/metric, causal/impact, predict/forecast, experiment/AB,
   drift/shift). Returns ``True`` when ≥2 topic groups fire. Consumed
   by ``src/api/routes/copilotkit.py:_classify_query_type`` to flip the
   analytics query_type label to ``"multi_faceted"`` on chat-message
   telemetry. Added by issue #295.

The three algorithms intentionally remain distinct — they're triggered
at different layers of the stack and against different query
distributions. Convergence here is structural (one module,
identity-checked) so that future drift is observable in tests; it is
not semantic (the regex set, the facet scorer, and the topic-count
scorer remain independent).

Background: PR #287 retired the v42 ``MultiFacetedDetector``, leaving
the 4 inline regexes in ``intent_classifier.py`` and the duplicated
``_is_multi_faceted_query`` functions in the two chatbot routes. Issue
#288 surfaced the drift risk; this module is the fix. Issue #295
extended convergence to a fourth detector inside copilotkit's
analytics-labeling helper.

Module location note: this module lives at ``src/agents/`` rather than
``src/agents/orchestrator/`` so that the lightweight chatbot route
modules (``chatbot_dspy.py`` in particular, which has no other
``src.agents.orchestrator`` import) can consume the SSOT without
transitively requiring ``langgraph`` via the orchestrator package
``__init__``.
"""

from __future__ import annotations

import re

MULTI_FACETED_PATTERNS: tuple[str, ...] = (
    # Conjunctive multi-question markers ("and also", "compare X vs Y, then ...").
    r"and (also|then|additionally|furthermore)",
    r"compare .* (vs|versus|against|to) .* and",
    r"(combine|integrate|synthes).*(analyses|results|findings)",
    r"(both|multiple) (effects?|analyses|perspectives?)",
    # Sequential / dependent-pipeline forms (orchestrator multi-part routing,
    # 2026-06-09; audit findings C2/C3). BOTH require an explicit "then"-marker
    # so they cannot flip the locked single-intent negatives in
    # test_intent_classifier_multi_faceted.py — additive/list joins without a
    # sequence marker ("compare A and B", "X and also Y") stay single-agent.
    r",\s+and\s+(what|which|how|why|who|where)\b.*\bthen\b",
    r"\bthen\s+(design|identify|recommend|predict|estimate|compare|find|"
    r"build|simulate|determine|use|forecast|analyz)",
)


# Sequential / dependency connectors that signal a *dependent pipeline*
# ("do A, then B using A"). This is the precise Tool-Composer routing signal —
# deliberately distinct from additive/parallel joins ("and also", "compare X
# and Y"), which are single comparisons or parallel delegations, NOT pipelines.
_SEQUENTIAL_MARKER_REGEX = re.compile(
    r"\b(then|after that|and then|followed by|after determining|"
    r"based on (that|those|the)|use (that|those|the results?)|once (we|you))\b",
    re.IGNORECASE,
)


def has_sequential_composition(query: str) -> bool:
    """Return ``True`` when the query chains a *dependent* second step onto a
    first via an explicit sequence/dependency connector.

    Examples → ``True``: "do A, then B", "after that estimate the lift",
    "based on that recommend a plan".
    Examples → ``False``: "compare A and B", "X and also Y" (additive, not
    sequential).

    Used by ``IntentClassifierNode`` to promote a multi-intent dependent query
    to ``multi_faceted`` (→ ``tool_composer``). Lives in the SSOT module so the
    multi-faceted detection logic has exactly one home (issue #288).
    """
    return bool(_SEQUENTIAL_MARKER_REGEX.search(query))


_FACET_CONJUNCTION_WORDS: tuple[str, ...] = (
    "compare",
    "trends",
    "explain",
    "also",
    "and then",
    "both",
)

_FACET_CROSS_AGENT_WORDS: tuple[str, ...] = (
    "drift",
    "health",
    "causal",
    "experiment",
    "prediction",
)

_FACET_RECOMMENDATION_WORDS: tuple[str, ...] = (
    "recommend",
    "suggest",
    "should",
)

_FACET_KPI_REGEX = re.compile(r"(trx|nrx|market share|conversion|volume|patient starts)")
_FACET_BRAND_REGEX = re.compile(r"(kisqali|fabhalta|remibrutinib|all brands)")


def is_multi_faceted_facet_score(query: str) -> bool:
    """Return ``True`` when ≥2 of 5 facets fire on ``query``.

    Facets:
      - conjunction_keywords: any of {compare, trends, explain, also,
        "and then", both} present.
      - multiple_kpis: >1 occurrence of a KPI token.
      - cross_agent: any of {drift, health, causal, experiment, prediction}.
      - multiple_brands: >1 occurrence of a brand or "all brands".
      - analysis_and_recommendation: ("why" OR "what caused") AND
        (recommend OR suggest OR should).

    Originally duplicated as ``_is_multi_faceted_query`` in
    ``chatbot_graph.py`` and ``chatbot_dspy.py``; both call sites now
    delegate here.
    """
    query_lower = query.lower()

    facets = (
        any(w in query_lower for w in _FACET_CONJUNCTION_WORDS),
        len(_FACET_KPI_REGEX.findall(query_lower)) > 1,
        any(w in query_lower for w in _FACET_CROSS_AGENT_WORDS),
        len(_FACET_BRAND_REGEX.findall(query_lower)) > 1,
        (
            ("why" in query_lower or "what caused" in query_lower)
            and any(w in query_lower for w in _FACET_RECOMMENDATION_WORDS)
        ),
    )

    return sum(facets) >= 2


# ---------------------------------------------------------------------------
# Topic-count detector (issue #295) — analytics-label algorithm used by
# ``src/api/routes/copilotkit.py:_classify_query_type``. Kept distinct from
# the facet-scorer above on purpose: different keyword set, different
# call site, different consumer (analytics field vs Tool Composer dispatch).
# ---------------------------------------------------------------------------

# Each tuple is one topic-group; >=2 firing groups means the query crosses
# multiple analytics topics and should be labelled "multi_faceted" in
# chat-message telemetry. Tuples are immutable; the public name is
# referenced by tests in ``tests/unit/test_agents/test_orchestrator/test_multi_faceted_ssot.py``.
TOPIC_COUNT_KEYWORD_GROUPS: tuple[tuple[str, ...], ...] = (
    ("trx", "nrx", "kpi", "metric", "performance"),
    ("causal", "impact", "effect", "intervention"),
    ("predict", "forecast", "future"),
    ("experiment", "test", "ab test", "a/b"),
    ("drift", "shift", "degradation"),
)


def is_multi_faceted_topic_count(query: str) -> bool:
    """Return ``True`` when ≥2 of 5 analytics topic groups fire on ``query``.

    Topic groups (see ``TOPIC_COUNT_KEYWORD_GROUPS``):

      - KPI/metric/performance (trx, nrx, kpi, metric, performance)
      - causal/impact/effect/intervention
      - predict/forecast/future
      - experiment/A-B test (experiment, test, ab test, a/b)
      - drift/shift/degradation

    Originally inlined at ``src/api/routes/copilotkit.py:1006-1014`` as
    the analytics-label-flipping heuristic in ``_classify_query_type``.
    Surfaced by issue #295; consolidated here so a future change to the
    topic groups is observable in tests.
    """
    query_lower = query.lower()
    return sum(any(kw in query_lower for kw in group) for group in TOPIC_COUNT_KEYWORD_GROUPS) >= 2
