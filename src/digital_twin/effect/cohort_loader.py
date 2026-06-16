"""Loader for the synthetic-gold per-HCP cohort backing the cohort effect provider.

Phase 2: reads a brand's ``per_hcp_rollup`` rows (``business_metrics``) so
:class:`CohortEffectDataProvider` can estimate a region-standardized treatment
effect for the cohort-estimable interventions. This is **synthetic-gold** data
(``is_synthetic=true``) — the intended showcase substrate before real-world data
is connected. Async DB access lives here (called from the async API route)
because the effect provider / simulation engine run synchronously off the event
loop; the cohort is pre-loaded and handed to the provider as a DataFrame.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from src.digital_twin.effect.provider import (
    COHORT_ESTIMABLE_INTERVENTIONS,
    COHORT_MIN_ROWS,
    INTERVENTION_TREATMENT_MAP,
    CohortEffectDataProvider,
)

logger = logging.getLogger(__name__)

COHORT_TABLE = "business_metrics"
COHORT_METRIC_TYPE = "per_hcp_rollup"
# region (confounder) + the two cohort treatment columns + the outcome.
_COHORT_COLUMNS = "region,engagement_score,call_frequency,conversion_rate"
# Generous cap so we read the full per-brand cohort (~7k rows) past PostgREST's
# default 1000-row page; the estimate only needs a representative sample.
_FETCH_LIMIT = 20000


async def load_cohort_frame(client: Any, brand: str) -> pd.DataFrame:
    """Load the brand's per-HCP cohort (region + treatments + outcome) as a frame.

    Returns an empty DataFrame on no rows. Numeric columns are coerced; row
    filtering / sufficiency checks are the provider's responsibility.
    """
    result = await (
        client.table(COHORT_TABLE)
        .select(_COHORT_COLUMNS)
        .eq("metric_type", COHORT_METRIC_TYPE)
        .eq("brand", brand)
        .limit(_FETCH_LIMIT)
        .execute()
    )
    rows = getattr(result, "data", None) or []
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("engagement_score", "call_frequency", "conversion_rate"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


async def build_cohort_provider_or_none(
    client: Any,
    intervention_type: str,
    brand: str,
) -> Optional[CohortEffectDataProvider]:
    """Return a :class:`CohortEffectDataProvider` if the intervention is
    cohort-estimable AND the brand has enough usable cohort rows; else ``None``
    (the caller falls back to the synthetic provider). Never raises — a DB/shape
    problem degrades to ``None`` (synthetic), never a fabricated effect.
    """
    if intervention_type not in COHORT_ESTIMABLE_INTERVENTIONS:
        return None
    treatment_col = INTERVENTION_TREATMENT_MAP[intervention_type]
    try:
        df = await load_cohort_frame(client, brand)
    except Exception as e:  # DB unreachable / query error → synthetic fallback
        logger.warning("cohort load failed for %s/%s: %s", brand, intervention_type, e)
        return None
    if df.empty or treatment_col not in df.columns:
        return None
    usable = df.dropna(subset=[treatment_col, "conversion_rate", "region"])
    if len(usable) < COHORT_MIN_ROWS:
        logger.info(
            "cohort for %s/%s has %d usable rows (< %d) — using synthetic",
            brand,
            intervention_type,
            len(usable),
            COHORT_MIN_ROWS,
        )
        return None
    return CohortEffectDataProvider(usable)


async def brand_has_cohort(client: Any, brand: str) -> bool:
    """Cheap check for ``/intervention-types`` ``effect_basis``: does the brand
    have a usable cohort (non-null engagement) of at least ``COHORT_MIN_ROWS``?
    Degrades to ``False`` on any error (advisory only).
    """
    try:
        result = await (
            client.table(COHORT_TABLE)
            .select("metric_id", count="exact")
            .eq("metric_type", COHORT_METRIC_TYPE)
            .eq("brand", brand)
            .not_.is_("engagement_score", "null")
            .limit(1)
            .execute()
        )
    except Exception as e:
        logger.warning("brand_has_cohort check failed for %s: %s", brand, e)
        return False
    count = getattr(result, "count", None)
    return bool(count is not None and count >= COHORT_MIN_ROWS)
