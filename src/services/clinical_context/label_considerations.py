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

# A Highlights bullet ends at its cross-reference marker. References are one or two
# levels deep — "( 2 )" is as valid as "( 2.2 , 5.3 , 7.1 )".
_ITEM = re.compile(
    r"(?P<body>.+?)\(\s*(?P<refs>\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)*)\s*\)",
    re.S,
)

# The full prescribing text starts at a "5.1 Title" subsection header — an N.M NOT
# sitting inside a "( ... )" reference list, so the preceding character is neither
# "(" nor ",".
_SUBSECTION = re.compile(r"[^(,\s]\s+\d+\.\d+\s+(?=[A-Z])")

# A title longer than this is a run-on sentence that happened to contain ": ",
# not a bullet title.
_MAX_TITLE_CHARS = 90


@dataclass(frozen=True)
class LabelConsideration:
    """One verbatim Highlights bullet from the prescribing information."""

    title: str
    detail: str
    section: str
    references: str
    source: str = "openfda"


def _strip_section_header(text: str, section: str) -> str:
    header = _SECTION_HEADERS.get(section)
    if not header:
        return text
    return re.sub(rf"^\s*\d+\s+{re.escape(header)}\s*", "", text, count=1, flags=re.I)


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
        detail = detail.strip()
        if not detail or detail.strip(". ") == "":
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
