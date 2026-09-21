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
from typing import Any, Dict, List, Mapping, Sequence, Tuple

logger = logging.getLogger("src.api.routes.digital_twin")

# The page polls /health every 60 s per viewer and one answer costs eight exact counts, so it is
# remembered briefly (per process; after a cohort restore the readout can lag by up to the TTL).
_SIMULABLE_TTL_S = 300.0
_simulable_cache: Dict[str, Tuple[float, bool]] = {}


async def brand_is_simulable(client: Any, brand: str) -> bool:
    """True when at least one intervention's effect is identified in the brand's cohort."""
    from src.digital_twin.effect.cohort_loader import cohort_treatment_availability

    now = time.monotonic()
    hit = _simulable_cache.get(brand)
    if hit is not None and now - hit[0] < _SIMULABLE_TTL_S:
        return hit[1]
    simulable = any((await cohort_treatment_availability(client, brand)).values())
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
    n_simulable = 0
    for brand in model_brands:
        if await brand_is_simulable(client, brand):
            n_simulable += 1
    if model_brands and n_simulable == 0:
        logger.warning(
            "Digital Twin health: %d active model(s) for %s but NO brand has a usable cohort "
            "treatment channel; every /simulate will refuse",
            len(models),
            ", ".join(model_brands),
        )
    return model_brands, n_simulable


def warn_model_without_effect_data(brand: str) -> None:
    """The state that used to be silent, with the brand and the remedy in the line."""
    logger.warning(
        "intervention-types: %s has a trained twin model but NO intervention is identified in its "
        "cohort (each planted treatment channel has too few usable per_hcp_rollup rows). "
        "Simulations are unavailable. If a full-window per-HCP backfill just ran, re-run "
        "scripts/backfill_segment_engagement.py --execute.",
        brand,
    )
