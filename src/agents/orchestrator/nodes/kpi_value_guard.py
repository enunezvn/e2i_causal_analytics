"""Structural guards for KPI value-lookup mentions (#2139/#2141).

The calculator accepts only brand, region, and time context from free text.  A
bare open-class word after a KPI mention therefore cannot be silently ignored:
it either names another quantity (``TRx cost``) or unserved scope (``TRx
patients``).  Closed-class words are enumerable; open-class nouns are not, so
the safe boundary is an allowlist of structure rather than a denylist of known
bad nouns.
"""

from __future__ import annotations

import re
from typing import Optional

_RIGHT_HEAD_FUNCTION_WORDS = frozenset(
    {
        # Prepositions / scope.
        "about",
        "above",
        "across",
        "after",
        "against",
        "along",
        "among",
        "amongst",
        "around",
        "as",
        "at",
        "before",
        "behind",
        "below",
        "beneath",
        "beside",
        "between",
        "beyond",
        "by",
        "despite",
        "down",
        "during",
        "except",
        "for",
        "from",
        "in",
        "including",
        "inside",
        "into",
        "near",
        "of",
        "off",
        "on",
        "onto",
        "out",
        "outside",
        "over",
        "past",
        "per",
        "since",
        "through",
        "throughout",
        "till",
        "to",
        "toward",
        "towards",
        "under",
        "underneath",
        "until",
        "up",
        "upon",
        "via",
        "with",
        "within",
        "without",
        # Coordination / subordination / comparison.
        "and",
        "or",
        "but",
        "nor",
        "yet",
        "so",
        "than",
        "then",
        "versus",
        "vs",
        "compared",
        "relative",
        "if",
        "when",
        "whenever",
        "where",
        "wherever",
        "while",
        "whilst",
        "because",
        "although",
        "though",
        "unless",
        # Determiners / quantifiers / pronouns.
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "its",
        "their",
        "our",
        "his",
        "her",
        "each",
        "every",
        "all",
        "any",
        "both",
        "some",
        "most",
        "more",
        "less",
        "fewer",
        "many",
        "much",
        "no",
        "not",
        "only",
        "same",
        # Temporal modifiers / adverbs.
        "last",
        "next",
        "prior",
        "previous",
        "current",
        "recent",
        "recently",
        "now",
        "today",
        "yesterday",
        "tomorrow",
        "ago",
    }
)

_DETERMINERS = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "its",
        "their",
        "our",
        "his",
        "her",
        "each",
        "every",
        "all",
        "any",
        "both",
        "some",
        "most",
        "many",
        "much",
        "same",
        "last",
        "next",
        "prior",
        "previous",
        "current",
        "recent",
        "past",
    }
)

_PERIOD_TOKEN_RE = re.compile(
    r"^(?:q[1-4]|h[12]|fy\d{2,4}|\d+|"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
    r"day|days|week|weeks|month|months|quarter|quarters|year|years|time|date)$"
)
_TAIL_TOKEN_RE = re.compile(r"[\w'+-]+")
_SENTENCE_BOUNDARY_RE = re.compile(r"[.?!;]")
_METRIC_SUFFIX = r"(?:'s|’s|e?s)?"

# These nouns are established value-lookup vocabulary on this surface even
# when the current registry resolves through a shorter alias (``NRx panel`` and
# the routed benchmark ``current TRx volume``). They are consumed rather than
# accepted as terminal decisions, so ``TRx volume cost`` still reaches and
# refuses on ``cost``.
_KNOWN_KPI_QUALIFIERS = frozenset({"panel", "volume", "volumes"})

_SCOPE_APPOSITIVES = {
    "brand": frozenset({"brand", "brands"}),
    "region": frozenset({"region", "regions"}),
}
_MAX_SCOPE_WINDOW = 3


def _declared_scope_dimension(phrase: str) -> Optional[str]:
    """Return the exact free-text scope dimension declared by the platform."""
    from src.services.query_entities import (
        _REGION_PHRASE_RE,
        INDICATION_TO_BRAND,
        SUPPORTED_BRANDS,
    )

    if _REGION_PHRASE_RE.fullmatch(phrase):
        return "region"
    if any(re.fullmatch(re.escape(brand), phrase, re.I) for brand in SUPPORTED_BRANDS):
        return "brand"
    if any(re.fullmatch(pattern, phrase, re.I) for pattern, _brand in INDICATION_TO_BRAND):
        return "brand"
    return None


def _scope_span(tokens: list[str], index: int) -> tuple[int, Optional[str]]:
    for size in range(min(_MAX_SCOPE_WINDOW, len(tokens) - index), 0, -1):
        dimension = _declared_scope_dimension(" ".join(tokens[index : index + size]))
        if dimension is not None:
            return size, dimension
    return 0, None


def _owned_spans(
    normalized_query: str, kpi_id: str, fallback_start: int, fallback_end: int
) -> list[tuple[int, int]]:
    """All non-overlapping mentions owned by ``kpi_id``, longest phrase wins."""
    from src.services.kpi_resolution import strict_metric_vocabulary

    candidates: set[tuple[int, int, str]] = {(fallback_start, fallback_end, kpi_id)}
    for phrase, owner_id in strict_metric_vocabulary():
        for match in re.finditer(
            rf"(?<![\w'-]){re.escape(phrase)}{_METRIC_SUFFIX}(?![\w'-])",
            normalized_query,
        ):
            candidates.add((match.start(), match.end(), owner_id))

    # Resolve overlapping vocabulary before selecting this KPI.  Without this,
    # the ``TRx`` inside ``TRx share`` would be attributed to two owners.
    owned: list[tuple[int, int, str]] = []
    for start, end, owner_id in sorted(
        candidates, key=lambda item: (-(item[1] - item[0]), item[0], item[2])
    ):
        if any(
            start < accepted_end and accepted_start < end
            for accepted_start, accepted_end, _ in owned
        ):
            continue
        owned.append((start, end, owner_id))
    return sorted((start, end) for start, end, owner_id in owned if owner_id == kpi_id)


def _tail_changes_quantity(
    normalized_query: str, span_end: int, value_heads: frozenset[str]
) -> bool:
    """Whether the bare tail after one KPI occurrence changes what is asked."""
    tail = _SENTENCE_BOUNDARY_RE.split(normalized_query[span_end:], maxsplit=1)[0]
    tokens = _TAIL_TOKEN_RE.findall(tail)
    index = 0
    bound_dimension: Optional[str] = None
    while index < len(tokens):
        token = tokens[index]
        if (
            token in value_heads
            or token in _KNOWN_KPI_QUALIFIERS
            or _PERIOD_TOKEN_RE.fullmatch(token)
        ):
            index += 1
            bound_dimension = None
            continue
        if token in _RIGHT_HEAD_FUNCTION_WORDS:
            # A determiner modifies what follows, so it cannot license a noun
            # hidden behind it (``TRx last two quarters cost``).
            if token in _DETERMINERS:
                index += 1
                bound_dimension = None
                continue
            return False
        if bound_dimension is not None and token in _SCOPE_APPOSITIVES[bound_dimension]:
            index += 1
            bound_dimension = None
            continue
        consumed, dimension = _scope_span(tokens, index)
        if consumed:
            index += consumed
            bound_dimension = dimension
            continue
        return True
    return False


def value_lookup_mentions_supported(
    normalized_query: str, kpi_id: str, match_start: int, match_end: int
) -> bool:
    """True only when every occurrence can honestly be answered as a value."""
    # Lazy import avoids a module cycle: dispatcher imports this helper at the
    # call site while these established head helpers live in dispatcher.
    from src.agents.orchestrator.nodes.dispatcher import (
        _CAUSAL_OF_HEADS,
        _VALUE_OF_HEADS,
        _kpi_governing_of_head,
        _kpi_right_head,
    )

    spans = _owned_spans(normalized_query, kpi_id, match_start, match_end)
    for start, end in spans:
        of_head = _kpi_governing_of_head(normalized_query, start)
        if of_head is not None and of_head not in _VALUE_OF_HEADS:
            return False
        if _kpi_right_head(normalized_query, end) in _CAUSAL_OF_HEADS:
            return False
        # Blank other occurrences without changing coordinates.  This keeps an
        # appositive restatement ("TRx, the total prescriptions") from looking
        # like a new open-class head, while punctuation cannot hide an actual
        # unsupported tail ("TRx, cost").
        head_checked_query = normalized_query
        for other_start, other_end in reversed(spans):
            if (other_start, other_end) != (start, end):
                head_checked_query = (
                    head_checked_query[:other_start]
                    + " " * (other_end - other_start)
                    + head_checked_query[other_end:]
                )
        if _tail_changes_quantity(head_checked_query, end, _VALUE_OF_HEADS):
            return False
    return True
