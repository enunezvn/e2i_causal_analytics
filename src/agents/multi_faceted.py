"""Single source of truth for multi-faceted query detection (issue #288).

Two distinct algorithms participate in classifying queries as "multi-faceted":

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

The two algorithms intentionally remain distinct — they're triggered at
different layers of the stack and against different query distributions.
Convergence here is structural (one module, identity-checked) so that
future drift is observable in tests; it is not semantic (the regex set
and the facet scorer remain independent).

Background: PR #287 retired the v42 ``MultiFacetedDetector``, leaving the
4 inline regexes in ``intent_classifier.py`` and the duplicated
``_is_multi_faceted_query`` functions in the two chatbot routes. Issue
#288 surfaces the drift risk and this module is the fix.

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
)


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
