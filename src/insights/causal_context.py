"""Shared causal-registry grounding for strategic-insight surfaces (2026-07-07).

The causal_paths registry models commercial KPIs (commercial grain: TRx / NRx /
NBRx / TRx market share / ROI / intent-to-prescribe). Insight surfaces cite
those modeled drivers through this ONE hop:

* :func:`fetch_commercial_drivers` — server-side read through the same
  repository + provenance gate the chatbot uses (trust boundary is the API;
  callers never post driver claims). Fails soft: any error returns ``[]`` so a
  registry hiccup can never break an insight endpoint.
* :func:`format_causal_context` — figures included, for signatures that
  already consume raw numbers (resource optimization).
* :func:`format_driver_names` — digit-free humanized names ONLY, for the
  executive brief, whose placeholder guard rejects ANY numeric character in
  LM output (``executive_brief._placeholder_violation`` check 2). A node name
  like ``persistent_180d`` fed raw would poison every sample into fallback,
  so unmappable digit-bearing names are DROPPED, never mangled.

These chains are curated synthetic knowledge (surfaced provenance-labeled),
not estimated effects — formatters say so, and consuming signatures instruct
the LM to use them qualitatively, never to invent figures from them.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# Display names for registry node tokens. Digit-bearing nodes MUST map to a
# digit-free label here to be usable by format_driver_names.
_NODE_LABELS: dict[str, str] = {
    "trx_volume": "TRx volume",
    "nrx_volume": "NRx volume",
    "nbrx_volume": "NBRx volume",
    "trx_market_share": "TRx market share",
    "roi": "ROI",
    "intent_to_prescribe": "intent to prescribe",
    "rep_detailing_frequency": "rep detailing frequency",
    "formulary_status": "formulary status",
    "copay_support_program": "copay support program",
    "sample_dropped": "sampling",
    "speaker_program_attendance": "speaker program attendance",
    "hcp_coverage": "HCP coverage",
    "competitor_activity": "competitor activity",
    "persistent_180d": "patient persistence",
    "treatment_initiated": "treatment initiation",
    "discontinued_180d": "treatment discontinuation",
}


def humanize_node(node: str) -> str:
    return _NODE_LABELS.get(node, node.replace("_", " "))


async def fetch_commercial_drivers(
    brand: Optional[str],
    *,
    outcomes: Sequence[str] = ("TRx", "ROI"),
    limit: int = 5,
    repo: Any = None,
    include_synthetic: Optional[bool] = None,
) -> list[dict[str, Any]]:
    """Top modeled drivers for the given outcome KPIs, deduped, confidence-desc.

    ``repo``/``include_synthetic`` are injectable for tests; production callers
    pass neither and get the chatbot's exact read path (CausalPathRepository +
    the platform provenance gate).
    """
    try:
        if repo is None:
            from src.memory.services.factories import get_async_supabase_client
            from src.repositories.causal_path import CausalPathRepository

            repo = CausalPathRepository(await get_async_supabase_client())
        if include_synthetic is None:
            from src.kpi.synthetic_mode import kpi_include_synthetic

            include_synthetic = kpi_include_synthetic()

        seen: dict[str, dict[str, Any]] = {}
        for term in outcomes:
            paths = await repo.search_paths_for_outcome(
                term,
                brand=brand,
                min_confidence=0.7,
                limit=limit,
                include_synthetic=include_synthetic,
            )
            for p in paths:
                pid = str(p.get("path_id"))
                if pid not in seen:
                    seen[pid] = {
                        "start": p.get("start_node", ""),
                        "end": p.get("end_node", ""),
                        "effect": float(p.get("causal_effect_size") or 0.0),
                        "confidence": float(p.get("confidence_level") or 0.0),
                        "synthetic": bool(p.get("is_synthetic", False)),
                    }
        ranked = sorted(seen.values(), key=lambda d: d["confidence"], reverse=True)
        return ranked[:limit]
    except Exception as e:  # noqa: BLE001 — a registry hiccup must never break an insight
        logger.warning("causal-context fetch failed (non-fatal): %s", e)
        return []


def format_causal_context(drivers: list[dict[str, Any]]) -> str:
    """Figure-bearing one-liner for signatures that consume raw numbers."""
    if not drivers:
        return "No modeled causal drivers are available for this scope."
    parts = [
        f"{humanize_node(d['start'])} → {humanize_node(d['end'])} "
        f"(effect {d['effect']:+.2f}, confidence {d['confidence']:.2f})"
        for d in drivers
    ]
    label = (
        "curated synthetic chains, provenance-labeled; directional, not estimated"
        if any(d.get("synthetic") for d in drivers)
        else "from the causal-path registry"
    )
    return f"Registry-modeled causal drivers ({label}): " + "; ".join(parts) + "."


def format_driver_names(drivers: list[dict[str, Any]]) -> list[str]:
    """Digit-free ``cause → effect`` names for the executive brief.

    GUARANTEE: no returned string contains a numeric character (the brief's
    placeholder guard fails closed on any digit outside a token). Drivers
    whose humanized names still carry digits are dropped, not mangled.
    """
    names: list[str] = []
    for d in drivers:
        name = f"{humanize_node(d['start'])} → {humanize_node(d['end'])}"
        if any(ch.isnumeric() for ch in name):
            logger.warning("dropping digit-bearing causal driver name %r for exec brief", name)
            continue
        names.append(name)
    return names
