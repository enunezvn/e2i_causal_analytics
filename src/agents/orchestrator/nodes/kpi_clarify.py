"""Branch-A clarify payloads: an ambiguous SCOPE ends in a question, never a total.

The dispatcher's KPI value-lookup branch (``_kpi_lookup_evidence``) answers an ask
with the figure the KPI registry's own SQL computes. When the ask names a scope
the substrate cannot honour, the honest fallback is NOT the unscoped figure —
that figure is real, plausible, and answers a different question than the one
asked. Two axes hit this, for the same reason and with the same fix:

* **Region** (#1572/#2191) — "East Coast" spans the northeast AND south census
  regions, and an ask can explicitly name several canonical regions; neither
  shape has one label a scalar lookup can serve. Without a clarify each was
  answered with a silent NATIONAL figure.
* **Brand** (#2114) — an ask naming SEVERAL brands grounds none:
  :func:`src.services.query_entities.brand_from_text` deliberately returns
  ``None`` so callers "keep their honest unscoped behaviour rather than guess".
  For the Rx-volume family that unscoped read is the TRACKED-PORTFOLIO total, a
  number belonging to none of the brands named — the plausible-but-wrong value
  the #1640 scale rules exist to prevent. (The AG-UI tool surface covers its own
  ingress with copilotkit.py's "Multiple brands in play — ambiguous is not
  absent" rule; this module is /chat's multi-agent Branch A.)

Both payloads are shaped like the Branch A value payload (context_assembler
reads "agent" / "analysis_type" / "key_findings" / "confidence" / "warnings"),
with the question in ``key_findings`` so the explainer's deterministic template
narrates it verbatim — and NO ``value``: the misleading figure is never
computed, which is the point.

The two live here rather than in ``dispatcher.py`` because that module is
ratchet-pinned (tests/unit/test_tests_meta/test_module_size_ratchet.py) and
because they are one concern with one rule.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

from src.kpi.volume_family import CANONICAL_VOLUME_KPI_IDS, PANEL_VOLUME_KPI_IDS

logger = logging.getLogger(__name__)

#: A vetted-SQL KPI read is deterministic, not an estimate — and so is the
#: DECISION to clarify (a vocabulary miss / a scan that grounded >1 brand), so
#: both payloads carry the same confidence as the value read.
KPI_LOOKUP_CONFIDENCE = 1.0

#: KPIs whose answer is BRAND-SCOPED, so a multi-brand ask must be clarified
#: rather than answered unscoped (#2114): the Rx-volume family — the canonical
#: business_metrics series (WS3-BI-005..008) and its patient-panel twins
#: (011..014). TRx/NRx/NBRx are brand-ADDITIVE, so their unbranded read is the
#: portfolio total; the two shares are per-brand by definition. Other
#: brand-filterable KPIs (e.g. WS3-BI-009 Conversion Rate) carry the same
#: pre-existing exposure but are not this lane's to change — the boundary is
#: pinned by a test so it stays a decision rather than an accident.
BRAND_CLARIFY_KPI_IDS: frozenset[str] = CANONICAL_VOLUME_KPI_IDS | PANEL_VOLUME_KPI_IDS


def region_clarify_evidence(kpi: Any, phrase: str) -> Dict[str, Any]:
    """Evidence payload that ASKS which census region is meant (#1572/#2191).

    "East Coast" spans the northeast AND south census regions, so no label can
    honestly serve it — the #1565 ruling that keeps it out of the shared alias
    table. The chat KPI tool already pairs that miss with its clarify hint
    (``_REGION_CLARIFY_HINT``, src/api/routes/chatbot_tools.py), but /chat's
    Branch A never passes the phrase to the tool: the free-text scan dropped
    it, so the ask was answered with a silent NATIONAL figure. This mirrors
    the same facts as a direct question to the user.
    """
    from src.services.enum_labels import REGION_ENUM_LABELS

    labels = ", ".join(REGION_ENUM_LABELS[:-1]) + f", or {REGION_ENUM_LABELS[-1]}"
    question = (
        f"Which US census region do you mean: {labels}? "
        f"'{phrase}' spans more than one census region, so {kpi.name} "
        "cannot be scoped to it without your choice."
    )
    return {
        "agent": "kpi_calculator",
        "analysis_type": "kpi_lookup_clarification",
        "key_findings": [question],
        "warnings": [question],
        "confidence": KPI_LOOKUP_CONFIDENCE,
        "needs_clarification": True,
        "unresolved_region_phrase": phrase,
        "kpi_id": kpi.id,
        "kpi_name": kpi.name,
    }


def brand_clarify_evidence(kpi: Any, brands: Sequence[str]) -> Dict[str, Any]:
    """Evidence payload that ASKS which of the named brands is meant (#2114)."""
    named = ", ".join(brands[:-1]) + f", or {brands[-1]}"
    question = (
        f"Which brand do you mean: {named}? "
        f"The ask names more than one, and {kpi.name} is reported per brand — "
        "an unscoped figure would be the whole tracked portfolio, not any one "
        "of the brands named."
    )
    return {
        "agent": "kpi_calculator",
        "analysis_type": "kpi_lookup_clarification",
        "key_findings": [question],
        "warnings": [question],
        "confidence": KPI_LOOKUP_CONFIDENCE,
        "needs_clarification": True,
        "ambiguous_brands": list(brands),
        "kpi_id": kpi.id,
        "kpi_name": kpi.name,
    }


def brand_clarify_for_ask(
    kpi: Any, query: Optional[str], structured_brand: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """The brand clarify for ``query`` on ``kpi``, or ``None`` to carry on.

    Returns a payload ONLY when the KPI is brand-scoped, no STRUCTURED source
    supplied a brand, and the ask grounds more than one. Naming no brand is the
    intentional portfolio ask and keeps computing unscoped; grounding exactly
    one binds it as before.

    ``structured_brand`` — not the text-scanned one — is the gate (#2114 codex
    r2): an explicit ``entities`` / ``user_context`` brand is a decision already
    taken and is never re-asked, but a brand the SCAN merely bound is not a
    decision. Keying off the scanned brand let the mixed shape through, because
    "Kisqali and PNH" binds Kisqali while grounding two scopes.
    """
    if structured_brand or kpi.id not in BRAND_CLARIFY_KPI_IDS:
        return None
    from src.services.query_entities import brand_scan

    scan = brand_scan(query)
    if not scan.is_ambiguous:
        return None
    logger.info(
        "explainer resolver: the ask grounds brands %s and %s is reported per brand "
        "-> returning the brand clarify instead of a figure for one of them.",
        ", ".join(scan.grounded_brands),
        kpi.id,
    )
    return brand_clarify_evidence(kpi, scan.grounded_brands)
