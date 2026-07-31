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
    # NB: "and then" is a DELIBERATE canonical multi_faceted phrase here, locked by
    # test_multi_faceted_ssot.py::test_pattern_matches_canonical_phrase. A review
    # (2026-06-09) noted that on a degenerate repeated single ask ("forecast … and
    # then forecast … again") this over-routes to tool_composer, but the phrase is
    # an intentional product contract, NOT introduced by the multi-part routing
    # fix — so it is left as-is rather than silently broken. (The new sequence-
    # marker promotion is additive; it does not touch these original 4 patterns.)
    r"and (also|then|additionally|furthermore)",
    r"compare .* (vs|versus|against|to) .* and",
    r"(combine|integrate|synthes).*(analyses|results|findings)",
    r"(both|multiple) (effects?|analyses|perspectives?)",
)
# NOTE (2026-06-09, audit C2/C3 + Codex review): dependent-pipeline detection is
# NOT done with extra SSOT patterns. A bare "then <verb>" / ", and <wh> ... then"
# pattern fires multi_faceted from a SINGLE mapped intent ("if X completes, then
# forecast" / "forecast …, then forecast … again"), which wrongly routes single
# asks to the 180s tool_composer. Instead, ``IntentClassifierNode._pattern_classify``
# promotes to multi_faceted only when a sequence/dependency marker
# (``has_sequential_composition``) joins **>=2 distinct strong intents** — i.e.
# >=2 recognised analytical asks, which is exactly when tool_composer's
# sub-question decomposition is useful. This favours precision: a multi-part query
# whose sub-asks the intent regexes do not recognise routes to the best single
# agent rather than over-routing to tool_composer.


# Sequential / dependency connectors that signal a *dependent pipeline*
# ("do A, then B using A"). This is the precise Tool-Composer routing signal —
# deliberately distinct from additive/parallel joins ("and also", "compare X
# and Y"), which are single comparisons or parallel delegations, NOT pipelines.
_SEQUENTIAL_MARKER_REGEX = re.compile(
    r"\b(then|after that|after this|and then|followed by|after determining|"
    r"based on (that|those|this|these|the)|"
    r"once (we|you))\b"
    # "using those" / "using these results" (anaphoric) and "using the [<=3
    # modifier words] results" — e.g. "using the model results", "using the
    # previous results" (Codex 2nd pass HIGH-1).
    r"|\b(use|using) (that|those|this|these)\b"
    r"|\b(use|using) the (?:[\w-]+ ){0,3}results?\b",
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


# Anaphoric/conditional back-references that signal a later step consuming an
# EARLIER step's output WITHOUT an explicit sequence word ("then"/"based on").
# The #1337 gold exposed dependency-linked TOOL_COMPOSER pipelines that
# ``has_sequential_composition`` missed because their dependency is carried by a
# referential phrase, not a sequence connector: "…, and for those regions, what
# is the ROI…" / "…, and if it has, re-run the segment analysis" / "run
# attribution on the worst one, and design a test for the top fix" / "explain
# its root cause, and propose a reallocation to close it".
#
# Precision guard (the #1366 lesson — structural over token-local, and the
# bench-0143 trap): every marker below is ANAPHORIC — it points back at a prior
# clause's result. A bare content superlative ("which region has *the largest
# gap opportunity*", bench-0143, gold PARALLEL) is a NEW ask's descriptor, not a
# back-reference, so the superlative marker is pronoun-anchored ("the worst
# one/ones") and never fires on "the largest gap". These markers only ever
# PROMOTE when the classifier's ">=2 distinct MAPPED strong intents +
# not-a-parallel-pair" gate already holds (intent_classifier.py), so a single
# ask carrying an incidental phrase cannot reach tool_composer.
_DEPENDENCY_MARKER_REGEX = re.compile(
    # "for those regions", "across these segments" — anaphoric object of a step
    r"\b(for|on|in|of|to|across|among|between) (those|these) \w+"
    # "given those findings", "given the results"
    r"|\bgiven (that|those|these|this|the) \w+"
    # "if it has, …" — conditional back-reference to a prior result
    r"|\bif it (has|had|does|did|is|was|were|drifted|shows?|showed)\b"
    # "the worst one", "the top ones" — pronoun-anchored superlative back-ref
    r"|\bthe (worst|best|largest|biggest|top|main|primary|strongest|weakest|"
    r"highest|lowest) (one|ones)\b"
    # "to close it", "to reverse that" — purpose clause acting on a prior result
    r"|\bto (close|reverse|fix|address|protect|mitigate|recover|retain|solve|"
    r"improve) (it|that|them|this|those|these)\b",
    re.IGNORECASE,
)


def has_dependency_composition(query: str) -> bool:
    """Return ``True`` when the query links a later step to an earlier one via a
    sequence connector OR an anaphoric/conditional back-reference.

    Superset of ``has_sequential_composition``: adds the referential dependency
    markers (``_DEPENDENCY_MARKER_REGEX``) the #1337 gold exposed. Kept as a
    distinct function so ``has_sequential_composition``'s locked semantics (its
    helper test) are untouched. Consumed only by ``IntentClassifierNode``'s
    tool_composer promotion, which additionally requires >=2 distinct mapped
    strong intents — so the broader marker set cannot promote single asks.
    """
    return has_sequential_composition(query) or bool(_DEPENDENCY_MARKER_REGEX.search(query))


# Coordinating-clause boundaries for counting genuinely-independent asks. Split
# on conjunctions ("and"/"then"/"plus"/"while"/"whereas"/"also") and clause
# punctuation (comma, semicolon, question mark, dash). NB: this OVER-splits list
# joins ("medium, high, and low severity") on purpose — the caller counts only
# clauses that independently bear a strong intent, so a bare list fragment
# ("high") contributes nothing. Sentence "." is deliberately NOT a delimiter
# (decimals, "Q4.", "vs.") — the conjunction/comma set already separates the
# multi-ask cases in the gold.
_CLAUSE_DELIM_REGEX = re.compile(
    r"\s*(?:[,;?]|—|--|\band\b|\bthen\b|\bplus\b|\bwhile\b|\bwhereas\b|\balso\b)\s*",
    re.IGNORECASE,
)


def split_clauses(query: str) -> list[str]:
    """Split ``query`` into coordinating clauses on conjunctions + punctuation.

    Structural (not semantic): used by ``IntentClassifierNode`` to require a
    genuine SECOND clause before promoting a two-strong-intent query to
    multi-agent — a second intent keyword inside a single clause ("predictive
    model performance", "break down NRx by segment") is an incidental co-match,
    not an independent facet (#1337 PARALLEL over-trigger). Lives in the SSOT
    module so multi-faceted structural analysis has one home (issue #288).
    """
    return [c for c in _CLAUSE_DELIM_REGEX.split(query) if c and c.strip()]


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
