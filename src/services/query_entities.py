"""Shared deterministic query-text entity extraction for chat resolvers (#1351).

Why this module exists
----------------------
The 2026-07-29 empirical routing pass (#1337 Step 0, issue #1351) proved the
orchestrator chat path never populates ``parsed_query`` — no producer exists
(``state.py`` declares it only) — so a brand or region named ONLY in the ask
text was invisible to every dispatcher input resolver, and structurally-sound
dispatches failed closed even when the answer was extractable from the query.
#1356 fixed exactly this failure privately inside cohort_profiler's ``ask.py``
(the q11 "Remibrutinib in the ask text was ignored" case). The owner ruling on
#1351 is resolvers EVERYWHERE, so the proven extraction is lifted here as the
single shared implementation; ``ask.py`` delegates to it.

Design constraints (mirrors ask.py):
* Deterministic regex only — no LLM calls; the resolvers run inline on the
  dispatch path under the 150s chat budget.
* Conservative: a value binds only when the text pins down EXACTLY ONE
  candidate. Two brands (or two regions) named means the ask is ambiguous and
  the caller keeps its honest unscoped/fail-closed behaviour rather than guess.
* Never fabricate: no defaults, ``None`` means "the text does not say".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.services.enum_labels import resolve_region_label

# Canonical brand casing — the KPI brand predicate is case-SENSITIVE
# (``brand::text = $1``), so every caller must receive this exact casing.
SUPPORTED_BRANDS: Tuple[str, ...] = ("Remibrutinib", "Fabhalta", "Kisqali")

# Indication → brand: the three commercial brands map 1:1 to indications on
# this substrate, so an indication mention grounds the brand (#1356 / q11).
INDICATION_TO_BRAND: Tuple[Tuple[str, str], ...] = (
    (r"\bcsu\b|\bchronic\s+spontaneous\s+urticaria\b|\burticaria\b", "Remibrutinib"),
    (r"\bpnh\b|\bparoxysmal\s+nocturnal\b", "Fabhalta"),
    (r"\bbreast\s+cancer\b|\bhr\+\b|\bher2\b", "Kisqali"),
)

# The four canonical region values carried by the data substrate
# (business_metrics.region / patient_journeys.geographic_region — verified
# READ-ONLY 2026-07-31: exactly these four values exist).
SUPPORTED_REGIONS: Tuple[str, ...] = ("northeast", "midwest", "south", "west")


def canonical_brand(raw: Optional[str]) -> Optional[str]:
    """Normalize ``raw`` to the canonical brand casing, else ``None``."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    low = raw.strip().lower()
    for b in SUPPORTED_BRANDS:
        if b.lower() == low:
            return b
    return None


def brand_from_text(query: Optional[str]) -> Optional[str]:
    """Ground the brand from the query text (name first, then indication).

    Returns a brand only when the text pins down EXACTLY ONE — two different
    brands named means the ask is ambiguous and the caller must keep its honest
    unscoped behaviour rather than guess. Behaviour is byte-identical to the
    proven cohort_profiler ``ask.py`` original (#1356).
    """
    if not query:
        return None
    found: List[str] = []
    for b in SUPPORTED_BRANDS:
        if re.search(rf"\b{b}\b", query, re.I) and b not in found:
            found.append(b)
    if not found:
        for pattern, b in INDICATION_TO_BRAND:
            if re.search(pattern, query, re.I) and b not in found:
                found.append(b)
    return found[0] if len(found) == 1 else None


# ---------------------------------------------------------------------------
# Region scanning (#1572)
# ---------------------------------------------------------------------------

#: Region phrases the FREE-TEXT scan may consider (#1572). This is an
#: ALLOWLIST, deliberately tighter than the tool surface's
#: :data:`~src.services.enum_labels.REGION_ALIASES`: the KPI tool's
#: ``region`` argument arrives pre-segmented — the LLM already decided the
#: string names a region — but free prose carries no such evidence, and
#: several shared aliases are wrong-region traps when they appear
#: mid-sentence as locality modifiers (measured while building this scan):
#: "southern california" is census-WEST (alias "southern" -> south),
#: "western pennsylvania" is census-NORTHEAST ("western" -> west),
#: "northwest Indiana" is census-MIDWEST ("northwest" -> west),
#: "Asia-Pacific" is not a US region at all ("pacific" -> west), "per se"
#: contains the token "se", and "central" is an ordinary adjective whose
#: #1565 guard phrase "central coast" must NEVER resolve to midwest. A
#: silently WRONG regional figure is the one outcome this scan may never
#: produce, so free text admits only phrases that name exactly one census
#: region in ordinary usage: the canonical labels (the pre-#1572 scan),
#: their separator variants, and two proper-noun phrases whose geography is
#: unambiguous. Excluding the rest narrows nothing that worked before #1572
#: — none of them ever bound on the free-text path. Every entry still
#: resolves through :func:`resolve_region_label` (pinned by tests), so the
#: two surfaces share ONE vocabulary and can never disagree about what an
#: admitted phrase means.
#: Separated compound directionals ("south east", "south-east", "north west")
#: are admitted AS PHRASES: they resolve through the shared vocabulary
#: (southeast/southwest -> south, northwest -> west), and matching them as
#: units also stops their tokens from leaking into the two passes (on the
#: pre-#1572 scan "south-east" bound south via its "south" token; a scan that
#: masked only "south" would spuriously clarify on the leftover "east" —
#: codex iter-2 HIGH). Their SINGLE-WORD forms ("southeast", "southwest",
#: "northwest") stay excluded: fused compounds are the metro-area modifier
#: idiom — "southeast Michigan" and "northwest Indiana" are census-MIDWEST —
#: so admitting them trades the unchanged-from-main unscoped figure for a
#: silently WRONG regional one.
_FREE_TEXT_REGION_PHRASES: Tuple[str, ...] = (
    "northeast",
    "midwest",
    "south",
    "west",
    "north east",
    "south east",
    "south west",
    "north west",
    "mid west",
    "new england",
    "west coast",
)

#: Collocations where an admitted phrase does NOT name a region: the New
#: England Journal (of Medicine) is everyday pharma vocabulary, and scoping a
#: KPI to the northeast because a journal was cited is exactly the silent
#: mis-scoping this module exists to prevent.
_PHRASE_GUARDS: Dict[str, str] = {"new england": r"(?![\s_-]+journal\b)"}


def _build_region_phrase_re() -> "re.Pattern[str]":
    """Word-boundary pattern over the free-text region allowlist.

    Longest-first alternation lets "west coast" win over "west" at the same
    position; spaces in a phrase tolerate any separator run.
    """
    alternation = "|".join(
        r"[\s_-]+".join(re.escape(part) for part in phrase.split()) + _PHRASE_GUARDS.get(phrase, "")
        for phrase in sorted(_FREE_TEXT_REGION_PHRASES, key=len, reverse=True)
    )
    return re.compile(rf"\b(?:{alternation})\b", re.I)


_REGION_PHRASE_RE = _build_region_phrase_re()

#: "Middle East" / "Far East" name non-US geographies; their spans are masked
#: before the ambiguity probe so they neither bind a region nor clarify.
#: (A mask handles any separator run — a fixed-width lookbehind cannot.)
_NON_US_EAST_RE = re.compile(r"\b(?:middle|far)[\s_-]+east\b", re.I)

#: Region-LIKE phrases that genuinely span more than one census region, so no
#: label can honestly serve them: the Atlantic seaboard runs ME..PA
#: (northeast) and DE..FL (south) — the #1565 ruling that keeps "east coast"
#: out of ``REGION_ALIASES``. "West Coast" resolves (every west-coast state is
#: census-west) while "East Coast" cannot — without this probe, /chat answered
#: it with a silent national figure (#1572). "central coast" (a #1565 guard
#: phrase) is deliberately NOT matched here — a locality mention is not
#: evidence the user meant a census region, so it keeps the honest unscoped
#: behaviour instead of a spurious clarify.
_AMBIGUOUS_REGION_RE = re.compile(
    r"\beast(?:ern)?(?:[\s_-]+(?:coast|seaboard))?\b",
    re.I,
)


@dataclass(frozen=True)
class RegionScan:
    """Outcome of the deterministic region scan over the ask text (#1572).

    ``region``           exactly one canonical label bound, else ``None``.
    ``ambiguous_phrase``  a region-like phrase the shared vocabulary cannot
                          resolve because it spans multiple census regions
                          ("East Coast", bare "East") — a caller that can
                          speak to the user should ask which census region is
                          meant instead of silently answering unscoped.
    """

    region: Optional[str]
    ambiguous_phrase: Optional[str]

    @property
    def needs_clarification(self) -> bool:
        return self.ambiguous_phrase is not None


def region_scan(query: Optional[str]) -> RegionScan:
    """Scan the ask text for region phrases via the shared #1565 vocabulary.

    Two-pass, conservative:

    1. Every allowlisted phrase match is resolved through
       :func:`src.services.enum_labels.resolve_region_label` (synonym mode —
       the same contract the chat KPI tool applies), and its span is masked.
    2. The masked remainder — with non-US "Middle/Far East" spans masked
       too — is probed for phrases that are region-like but unresolvable
       BY DESIGN ("east coast", bare "east").

    ``region`` binds only when pass 1 finds EXACTLY ONE distinct label AND
    pass 2 finds nothing: a resolvable and an unresolvable scope in one ask
    ("Northeast vs the East Coast") is ambiguous as a whole, so nothing binds
    and the clarify signal is raised instead. Never fabricates.
    """
    if not query:
        return RegionScan(region=None, ambiguous_phrase=None)
    labels: List[str] = []
    masked = query
    for match in _REGION_PHRASE_RE.finditer(query):
        label = resolve_region_label(match.group(0), allow_synonyms=True)
        if label is None:  # pragma: no cover — pattern is built from the table
            continue
        if label not in labels:
            labels.append(label)
        masked = (
            masked[: match.start()] + " " * (match.end() - match.start()) + masked[match.end() :]
        )
    masked = _NON_US_EAST_RE.sub(lambda m: " " * (m.end() - m.start()), masked)
    ambiguous = _AMBIGUOUS_REGION_RE.search(masked)
    phrase = ambiguous.group(0) if ambiguous is not None else None
    region = labels[0] if len(labels) == 1 and phrase is None else None
    return RegionScan(region=region, ambiguous_phrase=phrase)


def region_from_text(query: Optional[str]) -> Optional[str]:
    """Ground the region from the query text, else ``None``.

    Since #1572 the scan runs through the shared #1565 alias vocabulary
    (:mod:`src.services.enum_labels`), so unambiguous natural phrasings
    ("West Coast", "New England") bind their canonical label.
    It still binds only when the text pins down EXACTLY ONE region, in the
    substrate's canonical lowercase form; a multi-region phrase ("East
    Coast") binds nothing — callers that can ask the user consult
    :func:`region_scan` for the clarify signal instead.
    """
    return region_scan(query).region
