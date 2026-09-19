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

# These relations can introduce a positive brand/region/window filter that the
# calculator actually applies.  Other function words remain structural tokens
# for classification but are refused: negation, disjunction, comparison and
# conditional grammar cannot be represented by the calculator context.
_SUPPORTED_RELATIONS = frozenset(
    {
        "across",
        "among",
        "amongst",
        "as",
        "at",
        "between",
        "by",
        "during",
        "for",
        "from",
        "in",
        "including",
        "inside",
        "into",
        "near",
        "of",
        "on",
        "onto",
        "over",
        "per",
        "since",
        "through",
        "throughout",
        "to",
        "toward",
        "towards",
        "under",
        "underneath",
        "until",
        "via",
        "with",
        "within",
        "and",
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
_METRIC_SUFFIX = r"(?:'s|’s|e?s)?"
_DISCOURSE_SUFFIXES = frozenset({"please", "thanks"})

# These nouns are established value-lookup vocabulary on this surface even
# when the current registry resolves through a shorter alias (``NRx panel`` and
# the routed benchmark ``current TRx volume``). They are consumed rather than
# accepted as terminal decisions, so ``TRx volume cost`` still reaches and
# refuses on ``cost``.
_KNOWN_KPI_QUALIFIERS = frozenset({"panel", "volume", "volumes"})

# WS3-BI-008 carries a mandatory narrated warning that explicitly says its
# tracked-portfolio share is NOT external competitor market share.  Preserve
# that established, non-silent contract for asks such as "market share ...
# compared to competitors" without allowing the noun on any other KPI.
_WARNED_TAIL_NOUNS = {"WS3-BI-008": frozenset({"competitor", "competitors"})}

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
        region_scan,
    )

    if _REGION_PHRASE_RE.fullmatch(phrase):
        return "region"
    ambiguous = region_scan(phrase).ambiguous_phrase
    if ambiguous is not None and ambiguous.lower() == phrase.lower():
        # The caller's downstream region probe turns this exact phrase into a
        # clarification and never calls the calculator.  Let it reach that
        # safer path instead of collapsing to a generic refusal here.
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


def _window_token_indexes(tokens: list[str]) -> tuple[frozenset[int], bool]:
    """Indexes belonging to one parser-accepted window, plus ambiguity."""
    from src.services.time_window import WindowParseError, parse_window

    candidates: list[tuple[int, int, str, str]] = []
    for size in range(min(4, len(tokens)), 1, -1):
        for start in range(len(tokens) - size + 1):
            try:
                parsed = parse_window(" ".join(tokens[start : start + size]))
            except WindowParseError:
                continue
            if parsed is not None:
                candidates.append((start, start + size, parsed.start_iso, parsed.end_iso))

    # A shorter parse nested inside a longer one is the same phrase (for
    # example the month-year suffix of a range), not a second requested window.
    maximal = [
        span
        for span in candidates
        if not any(
            other != span and other[0] <= span[0] and span[1] <= other[1] for other in candidates
        )
    ]
    distinct = sorted(set(maximal))
    indexes = frozenset(index for start, end, _lo, _hi in distinct for index in range(start, end))
    requested_windows = {(lo, hi) for _start, _end, lo, hi in distinct}
    return indexes, len(requested_windows) > 1


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
    normalized_query: str,
    span_end: int,
    value_heads: frozenset[str],
    *,
    brand_resolved: bool,
    region_resolved_or_clarified: bool,
    warned_tail_nouns: frozenset[str],
) -> bool:
    """Whether the bare tail after one KPI occurrence changes what is asked."""
    tail = normalized_query[span_end:]
    matches = list(_TAIL_TOKEN_RE.finditer(tail))
    tokens = [match.group(0) for match in matches]
    warned_noun_present = bool(warned_tail_nouns.intersection(tokens))
    window_indexes, ambiguous_window = _window_token_indexes(tokens)
    if ambiguous_window:
        return True
    index = 0
    bound_dimension: Optional[str] = None
    needs_object = False
    while index < len(matches):
        token = matches[index].group(0)
        if token in _DISCOURSE_SUFFIXES:
            # A terminal polite suffix changes neither quantity nor scope, with
            # or without punctuation ("TRx please", "TRx? Thanks.").  It is
            # harmless only when EVERY remaining token is also discourse, so
            # "TRx please cost" and "TRx? I mean its cost" still refuse.
            remaining = {match.group(0) for match in matches[index:]}
            terminal = remaining <= _DISCOURSE_SUFFIXES
            return not terminal
        if token in value_heads or token in _KNOWN_KPI_QUALIFIERS or token in warned_tail_nouns:
            index += 1
            bound_dimension = None
            needs_object = False
            continue
        if index in window_indexes:
            index += 1
            bound_dimension = None
            needs_object = False
            continue
        if _PERIOD_TOKEN_RE.fullmatch(token):
            # A time-looking token is scope only when this exact token belongs
            # to a phrase accepted by the consumer's own parser.  A real window
            # elsewhere cannot license a stray Q3/year here.
            return True
        if token in _DETERMINERS:
            index += 1
            bound_dimension = None
            needs_object = True
            continue
        warned_comparison = warned_noun_present and token in {
            "compared",
            "relative",
            "than",
            "versus",
            "vs",
        }
        if token in _SUPPORTED_RELATIONS or warned_comparison:
            # Structure defers the decision; it never ends the safety walk.
            # The next exact scope/time/value token satisfies it, while an
            # unresolved object remains a refusal at end-of-tail.
            index += 1
            bound_dimension = None
            needs_object = True
            continue
        if token in _RIGHT_HEAD_FUNCTION_WORDS:
            # The token is structural but the requested relation is not one
            # the calculator can encode (negation, disjunction, comparison,
            # condition, etc.).  A recognized object must not make it appear
            # supported: "TRx without Kisqali" is not a Kisqali filter.
            return True
        if bound_dimension is not None and token in _SCOPE_APPOSITIVES[bound_dimension]:
            index += 1
            bound_dimension = None
            needs_object = False
            continue
        consumed, dimension = _scope_span(tokens, index)
        if consumed:
            if dimension == "brand" and not brand_resolved:
                return True
            if dimension == "region" and not region_resolved_or_clarified:
                return True
            index += consumed
            bound_dimension = dimension
            needs_object = False
            continue
        return True
    return needs_object


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
    from src.services.query_entities import brand_from_text, region_scan

    spans = _owned_spans(normalized_query, kpi_id, match_start, match_end)
    brand_resolved = brand_from_text(normalized_query) is not None
    region_result = region_scan(normalized_query)
    region_resolved_or_clarified = (
        region_result.region is not None or region_result.ambiguous_phrase is not None
    )
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
                article = re.search(r"\b(?:a|an|the)\s+$", head_checked_query[:other_start])
                mask_start = article.start() if article is not None else other_start
                head_checked_query = (
                    head_checked_query[:mask_start]
                    + " " * (other_end - mask_start)
                    + head_checked_query[other_end:]
                )
        if _tail_changes_quantity(
            head_checked_query,
            end,
            _VALUE_OF_HEADS,
            brand_resolved=brand_resolved,
            region_resolved_or_clarified=region_resolved_or_clarified,
            warned_tail_nouns=_WARNED_TAIL_NOUNS.get(kpi_id, frozenset()),
        ):
            return False
    return True
