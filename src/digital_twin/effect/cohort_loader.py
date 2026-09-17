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

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Union

import pandas as pd

from src.digital_twin.effect.errors import EffectCause
from src.digital_twin.effect.provider import (
    COHORT_CONFOUNDERS,
    COHORT_ESTIMABLE_INTERVENTIONS,
    COHORT_MIN_ROWS,
    INTERVENTION_TREATMENT_MAP,
    CohortEffectDataProvider,
)

logger = logging.getLogger(__name__)

COHORT_TABLE = "business_metrics"
COHORT_METRIC_TYPE = "per_hcp_rollup"
# All planted treatment channels (deduped, stable order for the select string).
_TREATMENT_COLUMNS: tuple[str, ...] = tuple(sorted(set(INTERVENTION_TREATMENT_MAP.values())))
# region (heterogeneity axis) + every treatment channel + outcome + pre-treatment
# confounders (market_share, triggers_total_count) for the direct causal estimate.
_COHORT_COLUMNS = ",".join(
    ["region", "conversion_rate", "market_share", "triggers_total_count", *_TREATMENT_COLUMNS]
)
_NUMERIC_COLUMNS: tuple[str, ...] = (
    "conversion_rate",
    "market_share",
    "triggers_total_count",
    *_TREATMENT_COLUMNS,
)
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
    for col in _NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


async def build_cohort_provider_or_none(
    client: Any,
    intervention_type: str,
    brand: str,
) -> Optional[CohortEffectDataProvider]:
    """Return a :class:`CohortEffectDataProvider` if the intervention is identified in
    the cohort AND the brand has enough usable rows (treatment + outcome + region + the
    required pre-treatment confounders all non-null); else ``None``. When ``None``, the
    caller surfaces an honest "no effect data" (422) — it does NOT fall back to a
    fabricated synthetic effect. Never raises — a DB/shape problem degrades to ``None``.
    """
    if intervention_type not in COHORT_ESTIMABLE_INTERVENTIONS:
        return None
    try:
        df = await load_cohort_frame(client, brand)
    except Exception as e:  # DB unreachable / query error → honest unavailable
        logger.warning("cohort load failed for %s/%s: %s", brand, intervention_type, e)
        return None
    return cohort_provider_from_frame(df, intervention_type)


def build_cohort_provider_or_none_blocking(
    intervention_type: str,
    brand: str,
) -> Optional[CohortEffectDataProvider]:
    """:func:`build_cohort_provider_or_none` for a synchronous caller with no running event
    loop (the experiment-designer pre-screen tool, the Celery simulation worker) (#2025).

    Opens a client scoped to its own event loop, not the cached one: the cached client's
    connection pool outlives the loop ``asyncio.run`` closes. Never raises for a missing
    client or an unusable cohort: ``None`` means no effect can be estimated.
    """
    from src.memory.services.factories import loop_scoped_async_supabase_client

    async def _build() -> Optional[CohortEffectDataProvider]:
        try:
            async with loop_scoped_async_supabase_client() as client:
                return await build_cohort_provider_or_none(client, intervention_type, brand)
        except Exception as e:  # no client configured → honest unavailable
            logger.warning("cohort client unavailable for %s/%s: %s", brand, intervention_type, e)
            return None

    return asyncio.run(_build())


def no_effect_data_reason(intervention_type: str, brand: str) -> str:
    """The reader-facing reason a caller gives when there is no cohort provider (#2025)."""
    return (
        f"No effect data available for intervention '{intervention_type}' and brand "
        f"'{brand}': the connected cohort cannot identify this intervention, so a causal "
        "effect cannot be estimated (no fabricated effect is returned)."
    )


def cohort_provider_from_frame(
    df: pd.DataFrame, intervention_type: str
) -> Optional[CohortEffectDataProvider]:
    """The usability check behind :func:`build_cohort_provider_or_none`, on a loaded frame.

    Split out (#2015) so a caller that must tell an unreachable database (retry) from an
    unusable cohort (refuse) can load the frame itself and still apply the same rule.
    """
    return assess_cohort_frame(df, intervention_type).provider


@dataclass(frozen=True)
class CohortUsability:
    """The outcome of :func:`assess_cohort_frame`: a provider, or why there is none (#2021).

    ``details`` holds counts and flags only, so a refusal can carry them as its reason details.
    """

    provider: Optional[CohortEffectDataProvider]
    cause: Optional[EffectCause] = None
    details: Mapping[str, Union[int, bool]] = field(default_factory=dict)


def assess_cohort_frame(df: pd.DataFrame, intervention_type: str) -> CohortUsability:
    """:func:`cohort_provider_from_frame` with the cause when there is no provider.

    Returns a value rather than raising, so a caller that only needs the provider stays a
    one-line wrapper and the tool's refusal needs no exception handler.
    """
    if intervention_type not in COHORT_ESTIMABLE_INTERVENTIONS:
        return CohortUsability(None, EffectCause.INTERVENTION_NOT_IDENTIFIED)
    treatment_col = INTERVENTION_TREATMENT_MAP[intervention_type]
    n_rows = int(len(df))
    if n_rows == 0:
        return CohortUsability(None, EffectCause.EMPTY_COHORT, {"n_rows": n_rows})
    # Usable rows must have the treatment, outcome, region AND the required confounders
    # non-null — aligned with what the direct estimator needs (it fails closed otherwise),
    # so we never build a provider that /simulate would then reject.
    has_treatment = treatment_col in df.columns
    has_outcome = "conversion_rate" in df.columns
    has_region = "region" in df.columns
    n_missing_confounders = sum(1 for c in COHORT_CONFOUNDERS if c not in df.columns)
    if not (has_treatment and has_outcome and has_region) or n_missing_confounders:
        return CohortUsability(
            None,
            EffectCause.REQUIRED_COLUMN_MISSING,
            {
                "n_rows": n_rows,
                "has_treatment_column": has_treatment,
                "has_outcome_column": has_outcome,
                "has_region_column": has_region,
                "n_missing_confounder_columns": n_missing_confounders,
            },
        )
    required = [treatment_col, "conversion_rate", "region", *COHORT_CONFOUNDERS]
    usable = df.dropna(subset=required)
    if len(usable) < COHORT_MIN_ROWS:
        logger.info(
            "cohort for %s has %d usable rows (< %d) — unavailable",
            intervention_type,
            len(usable),
            COHORT_MIN_ROWS,
        )
        # Which column drives the drop: an all-null channel reads n_null_treatment_rows == n_rows.
        return CohortUsability(
            None,
            EffectCause.TOO_FEW_USABLE_ROWS,
            {
                "n_rows": n_rows,
                "n_usable_rows": int(len(usable)),
                "n_min_usable_rows": COHORT_MIN_ROWS,
                "n_null_treatment_rows": int(df[treatment_col].isna().sum()),
                "n_null_outcome_rows": int(df["conversion_rate"].isna().sum()),
                "n_null_region_rows": int(df["region"].isna().sum()),
                "n_null_confounder_rows": int(
                    df[list(COHORT_CONFOUNDERS)].isna().any(axis=1).sum()
                ),
            },
        )
    return CohortUsability(CohortEffectDataProvider(usable))


async def _treatment_column_usable(client: Any, brand: str, treatment_col: str) -> bool:
    """Does the brand's cohort have >= ``COHORT_MIN_ROWS`` rows usable by the direct
    causal estimator for THIS treatment column — treatment, outcome, region AND the
    required pre-treatment confounders all non-null? Must match what ``/simulate``
    will accept (else the endpoint would advertise an intervention that then 422s).
    """
    query = (
        client.table(COHORT_TABLE)
        .select("metric_id", count="exact")
        .eq("metric_type", COHORT_METRIC_TYPE)
        .eq("brand", brand)
        .not_.is_(treatment_col, "null")
        .not_.is_("conversion_rate", "null")
        .not_.is_("region", "null")
    )
    for col in COHORT_CONFOUNDERS:
        query = query.not_.is_(col, "null")
    result = await query.limit(1).execute()
    count = getattr(result, "count", None)
    return bool(count is not None and count >= COHORT_MIN_ROWS)


async def cohort_treatment_availability(client: Any, brand: str) -> dict[str, bool]:
    """Per-intervention effect availability for a brand: ``{intervention: usable}``.

    An intervention is usable when its planted treatment column has enough usable
    rows (see :func:`_treatment_column_usable`). Drives ``available_for_effect`` in
    ``GET /digital-twin/intervention-types`` — HONEST per channel, so a substrate
    holding only some channels (pre-backfill, or future RWD with partial coverage)
    advertises exactly what ``/simulate`` can estimate. Degrades to ``False`` per
    column on any error (advisory only); never raises.
    """

    async def _safe(col: str) -> bool:
        try:
            return await _treatment_column_usable(client, brand, col)
        except Exception as e:
            logger.warning("cohort availability check failed for %s/%s: %s", brand, col, e)
            return False

    columns = list(_TREATMENT_COLUMNS)
    results = await asyncio.gather(*(_safe(c) for c in columns))
    usable_by_column = dict(zip(columns, results, strict=True))
    return {
        intervention: usable_by_column.get(column, False)
        for intervention, column in INTERVENTION_TREATMENT_MAP.items()
    }
