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
#: not on the other side.
#:
#: The set is large because the class is enumerable, NOT because binding is
#: preferred to refusing. The two error directions are INDEPENDENT at this
#: parameter: adding "amongst" here does not let "cost" through, because a token
#: only reaches this set by being a function word in the first place. There is no
#: scale to weight, so no severity argument belongs in this comment — the lane's
#: ordering (fail-open outranks fail-closed) is untouched by anything here. The
#: battery in ``test_panel_kpi_consumer_2114.py`` is what keeps the set honest.
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


#: Determiners that MODIFY whatever follows instead of opening a phrase: "last
#: quarter", "this year", "the q3", "this brand". The walk ALWAYS steps over one
#: and defers the decision to the next token -- requiring a period token to
#: follow was the r12 defect, because "last two quarters cost" then fell through
#: to the function-word return and bound with "cost" never examined.
#:
#: Every member is also in ``_RIGHT_HEAD_FUNCTION_WORDS``, pinned as a test. What
#: that subset proves, stated to match the test rather than the ambition:
#: stepping over a determiner into an end-of-string cannot bind a tail the
#: single-token rule would have refused, because the determiner was already a
#: binding token on its own.
#:
#: ⚠ IT DOES NOT PROVE WHOLE-HELPER MONOTONICITY, and an earlier version of this
#: comment claimed it did ("able only to REFUSE more ... never to bind more").
#: That became false the moment scope acceptance was added: `_scope_span` BINDS
#: tails that used to refuse -- "TRx Kisqali" refused under 11e and binds now, by
#: design. The subset is a local guard on one branch, not a global ordering, and
#: it proves nothing about the grammar being complete (r12).
_PERIOD_MODIFIERS = frozenset(
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

#: The token shape ``_kpi_right_head`` reads, applied to the whole tail -- plus
#: "+", which the RESOLVER needs and a bare ``[\w'-]+`` destroys. "hr+" splits to
#: "hr", and ``brand_from_text('hr')`` is None while ``brand_from_text('hr+')`` is
#: 'Kisqali'; commit b09a3271d in this lane exists to let HR+ ground Kisqali, so
#: the walk was refusing input its own discriminator says must bind (r12).
#:
#: "/" needs nothing: normalisation turns it into a space before the walk sees it
#: ("HR+/HER2-" arrives as 'hr+ her2'). Measured, not assumed.
_TAIL_TOKEN_RE = re.compile(r"[\w'+-]+")

#: Where a clause ends and so where a compound head must stop. "TRx cost" is one
#: quantity; "the TRx, the total prescriptions," is the same one said twice.
_CLAUSE_BOUNDARY_RE = re.compile(r"[,;:()]")

#: The noun that may trail a RESOLVED scope token to name its kind: "west
#: region", "Kisqali brand" -- KEYED TO THE DIMENSION THAT ACTUALLY BOUND, and
#: consumable ONCE, immediately (#2114 11g / #2139; corrected by r12 HIGH-b).
#:
#: An earlier draft was a FLAT set including "cohort", "tier", "axis", "market",
#: "area" and "territory", consumed whenever a scope token had been seen -- and
#: the "seen" flag was never reset, so they chained: "Kisqali tier cohort axis"
#: swallowed three unserved dimension nouns. Both halves were wrong:
#:
#: * neither resolver binds any of them (``brand_from_text`` and
#:   ``region_from_text`` return None for every one), so they were PATIENT_AXES
#:   under a new name -- readmitted one commit after 11h removed them, through a
#:   different door. "territory-level detail" is in the capability catalogue's
#:   NEVER_BLOCK list: no tool serves it at all;
#: * un-keyed, the appositive crossed dimensions -- "Kisqali region" bound the
#:   BRAND and then swallowed "region", the same fail-open in miniature.
#:
#: The key comes from WHICH RESOLVER returned non-None, never from a second word
#: list, so it cannot drift from the registry.
_SCOPE_APPOSITIVES = {
    "brand": frozenset({"brand", "brands"}),
    "region": frozenset({"region", "regions"}),
}


def _scope_span(tokens: list[str], index: int) -> tuple[int, Optional[str]]:
    """How many tokens at ``index`` the PLATFORM ITSELF resolves as scope, and WHICH
    dimension bound them -- ``(0, None)`` when nothing resolves.

    The dimension is returned so the trailing appositive can be keyed to it: see
    ``_SCOPE_APPOSITIVES``. It is read off which resolver answered, so it cannot
    disagree with what the dispatcher will later bind.

    EXACTLY TWO DIMENSIONS, because exactly two are resolvable from query text.
    ``_extract_brand_region`` (dispatcher.py:216-226) asks ``brand_from_text`` and
    ``region_from_text`` and NOTHING ELSE -- an unserved qualifier is not rejected,
    it is never examined. So these two resolvers are not merely the convenient
    vocabulary, they are the whole of what free text can bind, and membership is
    taken from them rather than from a word list copied into this module, which
    would drift from the registry the day a brand is added (#2114 11g / #2139).

    ⚠ PATIENT_AXES IS DELIBERATELY NOT HERE, and an earlier draft of this function
    had it. The four patient axes ("segment", "therapy_line", "biologic",
    "ige_tier") are a SEPARATE CHANNEL that free text does not feed. Measured:

        What is NRx panel segment?  -> calculator context {}   <- axis DROPPED
        What is TRx therapy line?   -> calculator context {}   <- dropped
        What is NRx panel oncology? -> calculator context {}   <- the #2141 defect

    ⚠ CORRECTION to the three lines above, which are NOT all from one tree. Two were
    measured on ``f3663f2d6``; the "oncology" row was measured on ``6d321cbcf``, where
    11g had not yet refused it. A table mixing two trees, written into the commit whose
    own body warns that a measurement is bound to the tree it ran against. Re-measured
    post-window at both, with a control asserting which implementation loaded:

                                      6d321cbcf (pre-11g)   f3663f2d6 (11g)
        What is NRx panel oncology?   {} binds              REFUSED
        What is NRx panel segment?    {} binds              {} binds   <- the 11h defect
        What is NRx panel cost?       {} binds              REFUSED

    THE CORRECTED TABLE IS THE STRONGER ARGUMENT. On ``f3663f2d6``, "segment" and
    "oncology" sit in the SAME dropped-scope condition -- both reach the calculator's
    caller with nothing bound -- yet one answers and the other refuses. The only
    difference between them is that "segment" was on this allowlist. That is the whole
    case against admitting it, and it is visible only once the rows share a tree.

    Admitting them would have whitelisted tokens measured to be ignored -- the very
    thing this fix refuses "oncology" for, and a labeling fix wearing the allowlist's
    hat. "region" is likewise NOT a patient axis: it is the second text channel,
    which is why "in the west region" arrives as ``{'region': 'west'}``.

    Two tokens are tried when one does not resolve, for the multi-word census
    phrases ("new england", "west coast"); one is preferred when it suffices, so
    "Kisqali cost" consumes only "Kisqali" and still refuses on "cost".
    """
    from src.services.query_entities import brand_from_text, region_from_text

    token = tokens[index]
    if brand_from_text(token):
        return 1, "brand"
    if region_from_text(token):
        return 1, "region"
    if index + 1 < len(tokens):
        # ⚠ THE PAIR IS ONLY A PHRASE WHEN THE FIRST TOKEN CHANGES THE ANSWER.
        # These resolvers match a brand or region ANYWHERE in the string handed
        # to them, so "cost kisqali" resolves to Kisqali and would be eaten whole
        # -- carrying the very noun #2139 refuses inside a "scope" span (r12).
        # Comparing the pair against the SECOND TOKEN ALONE separates them:
        #
        #   'new england'  pair=northeast  second=None      -> phrase, accept
        #   'mid west'     pair=midwest    second=west      -> phrase, accept
        #   'cost kisqali' pair=Kisqali    second=Kisqali   -> substring, reject
        #   'target west'  pair=west       second=west      -> substring, reject
        #
        # "reject when the second token resolves at all" was the obvious rule and
        # is WRONG: it would refuse "mid west" and "south west", whose second
        # token resolves to a DIFFERENT region. Measured before choosing.
        following = tokens[index + 1]
        pair = f"{token} {following}"
        pair_resolution = (brand_from_text(pair), region_from_text(pair))
        if pair_resolution != (brand_from_text(following), region_from_text(following)):
            if pair_resolution[0]:
                return 2, "brand"
            if pair_resolution[1]:
                return 2, "region"
    return 0, None


def _tail_changes_the_quantity(
    normalized_query: str, span_end: int, causal_heads: frozenset
) -> bool:
    """True when what follows the mention makes it a DIFFERENT quantity.

    ⚠ THE SCOPE OF THIS GUARD, STATED EXACTLY, BECAUSE AN EARLIER VERSION OF THIS
    DOCSTRING CLAIMED MORE THAN THE CODE DELIVERS (r12). It decides ONE question:
    does a BARE token sitting directly after the mention name a different quantity?
    It does NOT decide whether an ask is scoped. A qualifier behind a PREPOSITION is
    never examined at all, so "What is TRx among new patients?" binds although it
    reaches the calculator with nothing bound -- the same empty context that gets
    "What is TRx patients?" refused one word away.

    That asymmetry is DELIBERATE, not an oversight: judging a preposition's object
    would refuse most legitimately-scoped asks there are, and dropped prepositional
    scope is #2141 by owner decision #13. The honest rule is "a bare noun after the
    mention must resolve", never "an ask must be scoped in order to answer".

    A period token does not settle the question, it DEFERS it: "NRx panel q3" is
    still NRx panel, but "NRx panel q3 cost" is a cost. So the tail is walked --
    period tokens and resolvable scope are consumed, and a determiner is always
    consumed -- until a token turns up that is none of those. That token decides:

    * end-of-string, a causal head or a closed-class function word -> the KPI is
      still the thing being asked about, so BIND;
    * anything else is an open-class noun forming a compound -- "NRx panel cost",
      "TRx quarter forecast" -- so REFUSE.

    ⚠ PREPOSITIONS ARE NOT CONSUMED, and that is the difference between this and
    an over-refusing walk. A preposition opens a SCOPE PHRASE whose object is an
    ordinary noun -- "for Kisqali", "in the west region", "by severity" -- so
    stepping over it and judging its object would refuse every scoped ask there
    is. A determiner opens nothing; the compound head is still ahead of it.

    11e read only the FIRST token and accepted a period outright, which licensed
    whatever stood behind it: "month cost", "quarter forecast", "year target" and
    "2026 revenue" all bound, and those are the defect's own nouns (#2114 11f).
    """
    # A COMPOUND HEAD CANNOT SPAN A CLAUSE BOUNDARY, so the tail is cut at the
    # first one. Without this the determiner rule refuses an APPOSITIVE RESTATEMENT
    # of the same metric: "What is the TRx, the total prescriptions, for Kisqali?"
    # consumed "the" and then judged "total" -- a word belonging to this KPI's own
    # registry name -- and failed closed. Caught by #1475's suite, not by this
    # module's own tests (r12). The same instinct is already in `_OF_HEAD_RE`,
    # whose comment says punctuation breaks the chain.
    tokens = _TAIL_TOKEN_RE.findall(_CLAUSE_BOUNDARY_RE.split(normalized_query[span_end:])[0])
    index = 0
    #: The dimension the PREVIOUS token bound, or None. Holds for exactly one
    #: token, so a matching appositive is consumable once and only immediately
    #: -- "west region" binds, "west region region" and "Kisqali tier cohort"
    #: do not (r12 HIGH-b).
    bound_dimension: Optional[str] = None
    while index < len(tokens):
        token = tokens[index]
        if token in causal_heads:
            return False
        if _PERIOD_TOKEN_RE.match(token):
            index += 1
            bound_dimension = None
            continue
        # A FUNCTION WORD IS DECIDED BEFORE ANY SCOPE LOOKAHEAD, and the order is
        # load-bearing. `_scope_span`'s two-token window would otherwise swallow
        # "for kisqali" whole, carrying the walk PAST the preposition into
        # whatever prose follows -- "TRx for Kisqali, given that access issues
        # ate into field time" then refused on "given". A preposition opens a
        # phrase and ends the walk; it is never part of a scope span.
        if token in _RIGHT_HEAD_FUNCTION_WORDS:
            # A DETERMINER NEVER ENDS THE WALK; it modifies whatever follows, so
            # the decision is DEFERRED to that. Requiring a period token here was
            # the r12 defect: "last two quarters cost" skipped this branch, fell
            # through to the function-word return, and bound with "cost" never
            # examined. Prepositions still end the walk -- they open a phrase
            # whose object must not be judged.
            if token in _PERIOD_MODIFIERS:
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
        # newly exposed it for the four panel KPIs (owner decision #10). 11f: the
        # tail is WALKED, because 11e's one-token read let any period token
        # license the noun behind it.
        if _tail_changes_the_quantity(normalized_query, span_end, _CAUSAL_OF_HEADS):
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
        # 11g / #2139: an unsupported right-head compound ("NRx panel cost") was
        # never checked on this path either. The accepted set is NOT the causal
        # one -- no causal heads here, since "drivers" is declined above -- so the
        # walk is passed an EMPTY accepted-head set. The enumeration is in
        # test_panel_kpi_consumer_2114.py: a bare noun here is usually SCOPE, and
        # the walk defers to the platform's own resolver to tell scope from a
        # second quantity.
        if _tail_changes_the_quantity(normalized_query, span_end, frozenset()):
            return None
    return mask_spans(normalized_query, spans)
