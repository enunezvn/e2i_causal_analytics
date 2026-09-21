"""Can ``/digital-twin/simulate`` actually serve a brand? — the capability, not a model count.

On 2026-09-21 every brand had an active, loadable twin model and ``/digital-twin/health`` said
``healthy`` while no simulation could run: the cohort's planted treatment channels were gone (a
full-window per-HCP backfill had replaced the rows, and the per-HCP ETL inserts those columns as
NULL). Nothing logged a line, because the availability check ran cleanly and found nothing.

Kept out of ``digital_twin.py`` because that route module is pinned by the module-size ratchet.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger("src.api.routes.digital_twin")

# The page polls /health every 60 s per viewer and one answer costs eight exact counts, so it is
# remembered briefly (per process; after a cohort restore the readout can lag by up to the TTL).
_SIMULABLE_TTL_S = 300.0
_simulable_cache: Dict[str, Tuple[float, bool]] = {}


def _unmeasured(availability: Mapping[str, bool]) -> bool:
    """Nothing usable AND at least one probe errored: unknown, not empty (codex r1)."""
    return not any(availability.values()) and getattr(availability, "n_probe_errors", 0) > 0


async def brand_is_simulable(client: Any, brand: str) -> Optional[bool]:
    """True/False when MEASURED; ``None`` when the cohort probes errored and found nothing.

    ``None`` is never remembered: a connection blip must not read as "cohort gone" for five
    minutes, and the next poll should simply ask again.
    """
    from src.digital_twin.effect.cohort_loader import cohort_treatment_availability

    now = time.monotonic()
    hit = _simulable_cache.get(brand)
    if hit is not None and now - hit[0] < _SIMULABLE_TTL_S:
        return hit[1]
    availability = await cohort_treatment_availability(client, brand)
    if _unmeasured(availability):
        return None
    simulable = any(availability.values())
    _simulable_cache[brand] = (now, simulable)
    return simulable


async def simulable_brands(
    client: Any, models: Sequence[Mapping[str, Any]]
) -> Tuple[List[str], int]:
    """``(brands that have a model, how many of them /simulate can serve)``.

    Only brands that HAVE a model are asked: no model is the other gate, not a cohort problem.
    Logs a WARNING when models exist and none of their brands is simulable.
    """
    model_brands = sorted({str(m["brand"]) for m in models if m.get("brand")})
    answers = [await brand_is_simulable(client, brand) for brand in model_brands]
    n_simulable = sum(1 for a in answers if a)
    if model_brands and n_simulable == 0:
        if any(a is None for a in answers):
            logger.warning(
                "Digital Twin health: cohort effect data could not be measured for %s (the "
                "availability probes errored); not remembered, the next poll asks again",
                ", ".join(b for b, a in zip(model_brands, answers, strict=True) if a is None),
            )
        else:
            logger.warning(
                "Digital Twin health: %d active model(s) for %s but NO brand has a usable cohort "
                "treatment channel; every /simulate will refuse",
                len(models),
                ", ".join(model_brands),
            )
    return model_brands, n_simulable


def warn_model_without_effect_data(brand: str, availability: Mapping[str, bool]) -> None:
    """The state that used to be silent, with the brand and the remedy in the line.

    The remedy is a production write, so it is named only when the shortfall was MEASURED.
    """
    if _unmeasured(availability):
        logger.warning(
            "intervention-types: cohort effect data for %s could not be measured (%d availability "
            "probe(s) errored); reporting every intervention unavailable for this request",
            brand,
            getattr(availability, "n_probe_errors", 0),
        )
        return
    logger.warning(
        "intervention-types: %s has a trained twin model but NO intervention is identified in its "
        "cohort (each planted treatment channel has too few usable per_hcp_rollup rows). "
        "Simulations are unavailable. If a full-window per-HCP backfill just ran, re-run "
        "scripts/backfill_segment_engagement.py --execute.",
        brand,
    )
