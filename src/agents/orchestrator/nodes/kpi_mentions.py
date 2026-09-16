"""Deciding what a KPI mention MEANS before any of it is masked (#2114 codex r9).

The multi-KPI veto masks the recognized span and rescans; whatever is still found
is a SECOND metric and the ask fails closed. Task 11a made that masking cover
every mention the resolved KPI owns, which fixed a false refusal on a repeated
panel phrase — and introduced two regressions that failed OPEN, because

    MASKING IS DESTRUCTIVE, AND IT DESTROYS THE EVIDENCE LATER CHECKS NEED.

11a decided everything from the FIRST mention and then erased them all. Two
per-occurrence facts were gone by the time anything wanted them:

* **who OWNS this occurrence** — "trx share" (WS3-BI-008) was masked out of the
  MIDDLE of "trx share panel" (WS3-BI-014), leaving an orphan "panel" the
  scanner could not recognise, so a genuine two-KPI ask answered with ONE
  number. Fixed in :func:`~src.services.kpi_resolution.vocabulary_occurrences`,
  which settles ownership longest-wins on the INTACT string first.
* **what GOVERNS this occurrence** — the #1475 head guards ran on the first
  match only, so "What is NRx panel and the cost of NRx panel?" masked the
  later mention as a redundant repeat and answered a question nobody asked.
  Fixed here: every owned occurrence is head-checked, not just the first.

Both failures were worse in kind than the one 11a fixed. That one failed CLOSED
— a refusal the user did not deserve. These returned a plausible value for a
question that was never asked.

Lives beside ``dispatcher`` rather than inside it because that module is
ratchet-pinned (tests/unit/test_tests_meta/test_module_size_ratchet.py).
"""

from __future__ import annotations

import re
from typing import Optional

#: Right-head tokens that do NOT turn the KPI mention into a different quantity
#: (#2114 codex r11). Established by ENUMERATION against the real
#: ``_kpi_right_head`` over a spread of realistic causal asks, not guessed.
#:
#: The measured split is structural, and it is the justification for this shape:
#:
#:   legitimate  for in by across among at with from to on per within under between
#:               after before during since over and or than versus vs the this last
#:               next when where if now today recently  ... CLOSED-CLASS function words
#:   the defect  cost accuracy price forecast target volume trend uplift benchmark
#:               ... OPEN-CLASS nouns naming ANOTHER quantity
#:
#: A denylist of those nouns would be a LABELING fix: nouns are an open class and
#: cannot be enumerated, so the next unlisted noun reopens the hole. Function words
#: are a CLOSED class and CAN be. That asymmetry is why the allowlist goes here and
#: not on the other side. A token this set misses causes an OVER-REFUSAL, which is
#: the worse failure — so it is deliberately generous, and the battery in
#: ``test_panel_kpi_consumer_2114.py`` exists to keep it honest.
_RIGHT_HEAD_FUNCTION_WORDS = frozenset(
    {
        # prepositions / scope
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
        # coordination / subordination / comparison
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
        # determiners / quantifiers / pronouns
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
        # temporal modifiers
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

#: A period or plain number reads as scope, never as a new quantity: "in Q3",
#: "since 2026", "year to date".
_PERIOD_TOKEN_RE = re.compile(
    r"^(?:q[1-4]|h[12]|fy\d{2,4}|\d+|"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
    r"day|days|week|weeks|month|months|quarter|quarters|year|years|time|date)$"
)


def _right_head_changes_the_quantity(right_head: Optional[str], causal_heads: frozenset) -> bool:
    """True when the token after the mention makes it a DIFFERENT quantity.

    End-of-string, a causal head, a closed-class function word and a period token
    all leave the KPI as the thing being asked about. Anything else is an
    open-class noun forming a compound -- "NRx panel cost", "TRx accuracy" -- and
    a bare driver answer for the KPI does not answer it.
    """
    if right_head is None:
        return False
    if right_head in causal_heads or right_head in _RIGHT_HEAD_FUNCTION_WORDS:
        return False
    return not _PERIOD_TOKEN_RE.match(right_head)


def causal_masked_or_refusal(
    normalized_query: str, kpi_id: str, start: int, end: int
) -> Optional[str]:
    """The head-checked mask for the CAUSAL path, or ``None`` to refuse.

    The same per-occurrence principle as :func:`masked_or_refusal`, with a
    DIFFERENT accepted head set — which is why the value guard is not reused
    here (#2114 codex r10). On this path:

    * a governing of-head outside ``_CAUSAL_OF_HEADS`` REFUSES -- "the cost of
      NRx panel" names a head the registry does not model, so binding NRx
      panel's drivers would answer a different question;
    * a causal RIGHT-head does NOT refuse -- "NRx panel drivers" IS the ask
      here, where the value path must decline it.

    Checking only the FIRST mention left a later "cost of NRx panel" to be
    masked away as a redundant repeat, and the registry was then asked for
    WS3-BI-012. Measured across the lane: refused on the Task-11 base, ANSWERED
    from 11a onward -- lane-introduced, not pre-existing, and 11b closed only
    the value half because this site masked without checking heads at all.
    """
    from src.agents.orchestrator.nodes.dispatcher import (
        _CAUSAL_OF_HEADS,
        _kpi_governing_of_head,
        _kpi_right_head,
    )
    from src.services.kpi_resolution import mask_spans, owned_mention_spans

    spans = owned_mention_spans(normalized_query, kpi_id, start, end)
    for span_start, span_end in spans:
        of_head = _kpi_governing_of_head(normalized_query, span_start)
        if of_head is not None and of_head not in _CAUSAL_OF_HEADS:
            return None
        # r11: an unsupported RIGHT-head compound ("NRx panel cost") was never
        # checked on this path at all -- not on later occurrences and not on the
        # first. Pre-existing on origin/main for the canonical KPIs; the lane
        # newly exposed it for the four panel KPIs (owner decision #10).
        right_head = _kpi_right_head(normalized_query, span_end)
        if _right_head_changes_the_quantity(right_head, _CAUSAL_OF_HEADS):
            return None
    return mask_spans(normalized_query, spans)


def masked_or_refusal(normalized_query: str, kpi_id: str, start: int, end: int) -> Optional[str]:
    """The head-checked mask for the VALUE-lookup path, or ``None`` to refuse.

    A repeated mention of the same KPI binds only when the repeat is genuinely
    REDUNDANT. "cost of X" and "X drivers" are not redundant: they carry a
    sub-ask a bare value does not answer. So every owned occurrence gets the
    same two #1475 guards the first one gets, and any one of them refusing
    refuses the whole ask.

    The dispatcher's head helpers are imported lazily here because dispatcher
    imports this module — the cycle is real, and every import in that module is
    function-local for the same reason.
    """
    from src.agents.orchestrator.nodes.dispatcher import (
        _CAUSAL_OF_HEADS,
        _VALUE_OF_HEADS,
        _kpi_governing_of_head,
        _kpi_right_head,
    )
    from src.services.kpi_resolution import mask_spans, owned_mention_spans

    spans = owned_mention_spans(normalized_query, kpi_id, start, end)
    for span_start, span_end in spans:
        of_head = _kpi_governing_of_head(normalized_query, span_start)
        if of_head is not None and of_head not in _VALUE_OF_HEADS:
            return None
        if _kpi_right_head(normalized_query, span_end) in _CAUSAL_OF_HEADS:
            return None
    return mask_spans(normalized_query, spans)
