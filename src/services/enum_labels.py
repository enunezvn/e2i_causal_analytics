"""Single owner of the ``brand_type`` / ``region_type`` enum labels and the one
place a user- or LLM-supplied string is resolved to one of them (#1505).

Why this module exists
----------------------
``business_metrics.brand`` / ``.region`` and ``patient_journeys.brand`` /
``.geographic_region`` are Postgres ENUM columns. A string that is not a real
label cannot merely miss — it raises ``22P02`` and fails the whole query
(#1501, live). Three surfaces independently decided what a valid brand/region
phrase is:

* the chat KPI tool (``src.api.routes.chatbot_tools``) — casefold + separator
  strip + the platform's region synonyms,
* cohort resolution (``src.services.cohort_resolution``) — plain casefold, no
  synonyms, fail-closed,
* entity extraction (``src.rag.entity_extractor``) — the synonym table itself.

They had to be kept in step by hand on every enum change. They now share this
module.

The labels are hard-coded on purpose
------------------------------------
They are a DATABASE contract, verified against
``database/core/e2i_ml_complete_v3_schema.sql`` (no later migration ALTERs
either enum) and the prod DB (#1501: ``SELECT enum_range(NULL::region_type)``).
They are deliberately NOT read from ``config/domain_vocabulary.yaml`` at
runtime: that file is editable config, and a vocabulary edit landing before its
migration would push a non-label straight into an enum cast — the exact defect
#1501 fixed. ``tests/unit/test_services/test_enum_labels.py`` pins the two in
step instead, so drift fails a test rather than a production query.

Two modes, never a default
--------------------------
``resolve_region_label`` takes a REQUIRED ``allow_synonyms`` keyword. The chat
KPI tool passes ``True`` (an LLM emits "Northeast", "NE", "the Pacific").
Cohort resolution passed ``False`` through #1505 so that consolidation could
not widen a fail-closed contract as a side effect; #1517 then made the
widening an explicit product decision (every consumer feeding it passes
chat/LLM-derived or frontend-typed strings — the same input domain the KPI
tool resolves with synonyms), so it now passes ``True`` too. The keyword stays
REQUIRED with no default: a future call site with a different input domain
(e.g. DB-sourced territory names) must answer the question itself — omitting
the keyword is a ``TypeError``, so it can never be answered by accident.

Home chosen by measurement, not taste (#1505): importing
``src.rag.entity_extractor`` (the previous owner of the alias table) costs
``src.services.cohort_resolution`` **+7741 modules / +16.9 s** — ``src/rag``'s
package ``__init__`` eagerly pulls dspy, torch, mlflow, sklearn and supabase.
The reverse edge is free: ``src.services`` is already inside ``src.rag``'s
import chain (measured +0 modules), and ``src/services`` imports nothing from
``src/rag``, so no cycle exists.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# The enum labels (database contract — see module docstring)
# ---------------------------------------------------------------------------

#: ``region_type`` labels. US census regions, lowercase, single concatenated
#: words. NOTE this is NOT "US/EU/APAC".
REGION_ENUM_LABELS: Tuple[str, ...] = ("northeast", "south", "midwest", "west")

#: ``brand_type`` labels, in their real mixed casing.
BRAND_ENUM_LABELS: Tuple[str, ...] = (
    "Remibrutinib",
    "Fabhalta",
    "Kisqali",
    "competitor",
    "other",
)

#: ``brand_type`` labels that are aggregation BUCKETS, not named products.
#: They were added to the vocabulary/enum for DB sync ("resolve ENUM sync
#: issues", commit 9564740d) — valid enum values for storage and groupby, but
#: ordinary English words in chat text. NLP brand extraction
#: (:mod:`src.rag.entity_extractor`) excludes them (#1517): extracting
#: "competitor"/"other" from a phrase like "what other factors…" would scope
#: graph search or analytics to a bucket the user never named. The
#: enum-resolution path (:func:`resolve_brand_label`) accepts them unchanged.
BRAND_BUCKET_LABELS: Tuple[str, ...] = ("competitor", "other")

#: Region phrasings the platform's NLP layer recognizes, keyed by enum label.
#: Owned here and re-exported by :mod:`src.rag.entity_extractor` (which feeds
#: them to ``EntityVocabulary.from_default()``) so entity extraction and the
#: chat KPI tool can never disagree about what "NE" means.
REGION_ALIASES: Dict[str, List[str]] = {
    "northeast": ["northeast", "ne", "north east", "new england"],
    "south": ["south", "southeast", "se", "southwest", "sw", "southern"],
    "midwest": ["midwest", "mw", "mid west", "central"],
    "west": ["west", "pacific", "northwest", "nw", "western"],
}

# Case-insensitive matching uses ``str.casefold()`` (Python's documented
# caseless-matching operation) rather than ``str.lower()``. Exhaustively
# MEASURED over all 0x110000 codepoints, including the folds that EXPAND to
# more than one character: for these label sets exactly three codepoints admit
# a string that ``lower()`` would reject — U+017F LONG S ("ſouth"), U+FB05
# LONG S T and U+FB06 ST ligatures ("northeaﬆ") — yielding 11 inputs in total.
# Every one of them lands on a REAL label; ZERO land on a wrong one, so the
# difference can never resolve to a wrong brand or population. Pinned by
# TestCasefoldSemantics rather than left to chance.
_REGION_LABEL_BY_CASEFOLD: Dict[str, str] = {
    label.casefold(): label for label in REGION_ENUM_LABELS
}
_BRAND_LABEL_BY_CASEFOLD: Dict[str, str] = {label.casefold(): label for label in BRAND_ENUM_LABELS}

_SEPARATORS = re.compile(r"[\s_-]+")


def fold_region_key(value: str) -> str:
    """Casefold and REMOVE separators (spaces / hyphens / underscores).

    ``region_type`` labels are single concatenated words ("northeast"), so
    folding separators to underscores could never match — "North East" must
    become "northeast", not "north_east".
    """
    return _SEPARATORS.sub("", value.strip().casefold())


def _build_region_alias_map() -> Dict[str, str]:
    """folded alias -> ``region_type`` label.

    Only aliases whose canonical key is a verified enum label are admitted, so
    a future vocabulary edit can never push a non-label value into an enum
    cast.
    """
    mapping: Dict[str, str] = {}
    for label, aliases in REGION_ALIASES.items():
        if label not in REGION_ENUM_LABELS:
            continue
        for alias in (label, *aliases):
            mapping[fold_region_key(alias)] = label
    return mapping


#: folded alias -> ``region_type`` label (synonym-tolerant lookup table).
REGION_LABEL_BY_ALIAS: Dict[str, str] = _build_region_alias_map()


def resolve_region_label(region: Optional[str], *, allow_synonyms: bool) -> Optional[str]:
    """Resolve a region string to its ``region_type`` label, else ``None``.

    ``allow_synonyms=False`` (strict): only a real label, in any casing, with
    surrounding whitespace stripped — "NE" and "North East" do NOT resolve.
    For callers whose inputs are already canonical (or must be).

    ``allow_synonyms=True``: additionally accepts separator variants and every
    phrasing the platform's entity extraction recognizes ("NE", "new england",
    "central", "Pacific"). This is the contract of both wired consumers — the
    chat KPI tool (#1501) and, since #1517, cohort resolution — because their
    inputs are chat/LLM-derived forms, and an unresolved value would either
    22P02 the query or fail-close a resolvable ask.

    ``allow_synonyms`` is keyword-only and has NO default: each call site must
    state which contract it wants.
    """
    if not region or not region.strip():
        return None
    if allow_synonyms:
        return REGION_LABEL_BY_ALIAS.get(fold_region_key(region))
    return _REGION_LABEL_BY_CASEFOLD.get(region.strip().casefold())


def resolve_brand_label(brand: Optional[str]) -> Optional[str]:
    """Resolve any casing of a brand ("kisqali") to its ``brand_type`` label.

    Returns the REAL label ("Kisqali", "competitor"), or ``None`` when the
    input matches none — brands have no alias table, and no separator folding
    is applied because ``brand_type`` labels contain no separators.

    There is only one mode: entity extraction's brand aliases ("remi",
    "ribociclib") are a query-understanding concern, never valid values for an
    enum column, so neither consumer may accept them here.
    """
    if not brand or not brand.strip():
        return None
    return _BRAND_LABEL_BY_CASEFOLD.get(brand.strip().casefold())
