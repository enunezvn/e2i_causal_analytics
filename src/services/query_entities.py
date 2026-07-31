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
from typing import List, Optional, Tuple

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


def region_from_text(query: Optional[str]) -> Optional[str]:
    """Ground the region from the query text, else ``None``.

    Matches the four canonical region values word-boundary-anchored; binds only
    when exactly one region is named (same exactly-one semantics as the brand
    scan). Values are returned in the substrate's canonical lowercase form.
    """
    if not query:
        return None
    found: List[str] = []
    for r in SUPPORTED_REGIONS:
        if re.search(rf"\b{r}\b", query, re.I) and r not in found:
            found.append(r)
    return found[0] if len(found) == 1 else None
