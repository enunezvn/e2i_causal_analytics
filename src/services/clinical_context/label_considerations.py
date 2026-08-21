"""#1775 — the FDA-label facts that bear on a causal analysis, parsed verbatim.

The clinical panel must GROUND the scenario an analyst is interrogating, not just
describe the drug. For a persistence question the grounding is what the label says
about staying on therapy: what has to be monitored, when treatment is interrupted
or reduced, what the dosing schedule demands. For an initiation question it is what
gates the first dose.

All of that is already in the openFDA label we fetch for `approved_indications` —
34 sections come back for ribociclib and we keep three. This module reads the ones
that bear on the analysis and throws nothing else away.

**Every item is VERBATIM label text carrying its own cross-reference.** No
summarisation, no LLM, no derived clinical claim: a fabricated or truncated
consideration would be exactly the plausible-but-wrong value CLAUDE.md forbids in a
user-facing path. The reference (e.g. ``2.2 , 5.3``) lets an analyst open the
prescribing information at the paragraph the sentence came from.

Structure of the source text (verified against the live labels for all three
brands, 2026-08-21): each section opens with a numbered ALL-CAPS header, then the
Highlights bullets as ``Title: detail ( refs )``, then the full prescribing text
beginning at a ``5.1 Title`` subsection. Only the Highlights are parsed — the full
text runs to many thousands of characters per subsection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

WARNINGS_SECTION = "warnings_and_cautions"
DOSAGE_SECTION = "dosage_and_administration"
CONTRAINDICATIONS_SECTION = "contraindications"
BOXED_WARNING_SECTION = "boxed_warning"

# Human-readable names, used when a Highlights bullet carries no title of its own.
SECTION_DISPLAY = {
    WARNINGS_SECTION: "Warnings and precautions",
    DOSAGE_SECTION: "Dosage and administration",
    CONTRAINDICATIONS_SECTION: "Contraindications",
    BOXED_WARNING_SECTION: "Boxed warning",
}

# The ALL-CAPS header each section opens with. Matched EXPLICITLY rather than by a
# heuristic: a greedy "run of capitals" pattern ate the leading capital of the next
# word ("Monitoring" -> "onitoring"), and when the header was followed by an
# all-caps brand token or a digit it found no boundary at all and left
# "2 DOSAGE AND ADMINISTRATION KISQALI tablets ..." inside the first item. A
# truncated title is a fabricated title.
_SECTION_HEADERS = {
    WARNINGS_SECTION: "WARNINGS AND PRECAUTIONS",
    DOSAGE_SECTION: "DOSAGE AND ADMINISTRATION",
    CONTRAINDICATIONS_SECTION: "CONTRAINDICATIONS",
}

# The PI section number each section owns. Measured across the live labels for all
# three brands: EVERY Highlights bullet in section N cites section N, usually
# alongside others ("2.2 , 5.1" inside section 5). Defence in depth — independent of
# how clever the regex is, a "reference" that never names its own section is not one
# we established, so the item is dropped rather than rendered with a citation it
# never carried.
_SECTION_NUMBERS = {
    WARNINGS_SECTION: "5",
    DOSAGE_SECTION: "2",
    CONTRAINDICATIONS_SECTION: "4",
}

# A Highlights bullet ends at its cross-reference marker: "( 2 )", "( 2.2 , 5.3 )".
#
# What makes a parenthetical a TERMINATOR is POSITION, not spacing. A real reference
# ends its bullet, so it is followed by the end of the Highlights region or by the
# next bullet's capitalised title; a number inside the prose ("Assess patients (2)
# weeks after dose") is followed by lowercase continuation.
#
# An earlier version required the internal whitespace real references happen to
# carry. That protected the prose case but did NOT fail closed (codex iter-2 HIGH):
# an unspaced terminal reference simply was not a boundary, so the bullet it ended
# was swallowed into the next one and rendered under the next one's citation —
# fabricating a label item out of two, attributed to a section it never named. The
# positional rule handles both spacings and still rejects prose.
# Two INDEPENDENT signals must both hold, because either alone still let a prose
# number pose as a citation (codex iter-2 and iter-3):
#   - the parenthetical ENDS a sentence — every Highlights bullet across all three
#     live labels closes with "." before its reference;
#   - it is FOLLOWED by the end of the region or the next bullet's capitalised title.
# "Assess patients (1) Patients received therapy" satisfies the second and fails the
# first, which is exactly the case that truncated a bullet and invented "(1)" as its
# label section.
_ITEM = re.compile(
    r"(?P<body>.+?[.])\s*\(\s*(?P<refs>\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)*)\s*\)"
    r"(?=\s*$|\s+[A-Z])",
    re.S,
)

# The full prescribing text starts at a "5.1 Title" subsection header — an N.M NOT
# sitting inside a "( ... )" reference list, so the preceding character is neither
# "(" nor ",".
# `\s*`, not `\s+`: "(5.1)5.1 Full Text Begins" hid the boundary from a
# whitespace-requiring pattern, and the prescribing text behind it was then pulled
# into a consideration under the wrong citation (codex iter-3 HIGH).
_SUBSECTION = re.compile(r"[^(,\s]\s*\d+\.\d+\s+(?=[A-Z])")

# A title longer than this is a run-on sentence that happened to contain ": ",
# not a bullet title.
_MAX_TITLE_CHARS = 90

# "None." — often repeated, as the Highlights summary and the full section both say
# it. Carries no consideration; emitting "Contraindications: None" would be an
# invented clinical item.
_EMPTY_DETAIL = re.compile(r"(?:none[.\s]*)+|[.\s]*", re.I)


@dataclass(frozen=True)
class LabelConsideration:
    """One verbatim Highlights bullet from the prescribing information."""

    title: str
    detail: str
    section: str
    references: str
    source: str = "openfda"


def _references_name_this_section(references: str, section: str) -> bool:
    """True when the parsed reference list names the section it was found in."""
    own = _SECTION_NUMBERS.get(section)
    if own is None:
        return True
    return any(ref.split(".")[0] == own for ref in re.split(r"\s*,\s*", references) if ref)


def _strip_section_header(text: str, section: str) -> str:
    header = _SECTION_HEADERS.get(section)
    if not header:
        return text
    # Whitespace between the header words is arbitrary in the source SPL — matching
    # literal single spaces let "5  WARNINGS  AND\nPRECAUTIONS" leak into the first
    # item's title (codex iter-1 MEDIUM).
    pattern = r"\s+".join(re.escape(word) for word in header.split())
    return re.sub(rf"^\s*\d+\s+{pattern}\s*", "", text, count=1, flags=re.I)


def _highlights_region(text: str, section: str) -> str:
    body = _strip_section_header(text, section)
    match = _SUBSECTION.search(body)
    # +1 keeps the character the lookbehind-free pattern consumed.
    return body[: match.start() + 1] if match else body


def parse_label_considerations(text: Optional[str], section: str) -> Tuple[LabelConsideration, ...]:
    """Verbatim Highlights bullets from one label section.

    Returns an empty tuple when the section is absent or carries no bullets — an
    honest nothing, never an invented item.
    """
    if not text:
        return ()
    out: list[LabelConsideration] = []
    for match in _ITEM.finditer(_highlights_region(text, section)):
        body = " ".join(match.group("body").split())
        references = " ".join(match.group("refs").split())
        if not body:
            continue
        title, separator, detail = body.partition(": ")
        if not separator or len(title) > _MAX_TITLE_CHARS:
            # No bullet title of its own: name it by the section it came from
            # rather than inventing a clinical heading for it.
            title, detail = SECTION_DISPLAY.get(section, section), body
        if not _references_name_this_section(references, section):
            continue
        detail = detail.strip()
        # "4 CONTRAINDICATIONS None. None. ( 4 )" carries no consideration. An empty
        # result is correct; emitting "Contraindications: None" would be an invented
        # clinical item (codex iter-1 LOW).
        if not detail or _EMPTY_DETAIL.fullmatch(detail):
            continue
        out.append(
            LabelConsideration(
                title=title.strip(),
                detail=detail,
                section=section,
                references=references,
            )
        )
    return tuple(out)


def boxed_warning_consideration(text: Optional[str]) -> Optional[LabelConsideration]:
    """The boxed warning as ONE verbatim consideration.

    It is deliberately NOT run through the Highlights parser: it is prose, and doing
    so produced fragments like "] . Life-threatening ...". But it must be visible to
    grounding — Fabhalta's initiation gate (vaccinate against encapsulated bacteria
    before the first dose) lives here and nowhere in the Highlights bullets, so an
    initiation analysis was grounded on nothing at all.
    """
    if not text or not text.strip():
        return None
    return LabelConsideration(
        title=SECTION_DISPLAY[BOXED_WARNING_SECTION],
        detail=" ".join(text.split()),
        section=BOXED_WARNING_SECTION,
        references=SECTION_DISPLAY[BOXED_WARNING_SECTION],
    )
