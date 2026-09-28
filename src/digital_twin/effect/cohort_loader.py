"""Loader for the synthetic-gold per-HCP cohort backing the cohort effect provider.

Phase 2: reads a brand's ``per_hcp_rollup`` rows (``business_metrics``) so
:class:`CohortEffectDataProvider` can estimate a region-standardized treatment
effect for the cohort-estimable interventions. This is **synthetic-gold** data
(``is_synthetic=true``) — the intended showcase substrate before real-world data
is connected. Async DB access lives here (called from the async API route)
because the effect provider / simulation engine run synchronously off the event
loop; the cohort is pre-loaded and handed to the provider as a DataFrame.

Lane T2 (2026-09-28): the estimand is ``hcp_brand_adoption.adopted``, which lives on a
different table at a different grain. The frame is therefore TWO reads: the rollups
(treatment channels, confounders, labels), collapsed to one row per (hcp_id, brand) with
``src.data.per_hcp_cohort_collapse`` — the rules the adoption re-plant built its channel
term from, so the estimator's median contrast is the planted contrast — and the brand's
``adopted`` labels, paged. Both reads take ``is_synthetic = true`` rows only: the estimate
is labelled synthetic-gold (``PROVENANCE_COHORT``), and a real row must not be mixed into it
(a real-world cohort needs its own path and provenance). Both are checked against the
server's exact count. They are left-joined on (hcp_id, brand): a pair without an adoption
row carries a NULL outcome, which the usable-row rule drops, so estimation sees exactly the
inner join while the refusal can still say the OUTCOME is what is missing.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Union

import pandas as pd

from src.data.per_hcp_cohort_collapse import collapse_per_hcp_brand
from src.data.per_hcp_cohort_columns import TWIN_OUTCOME_COLUMN, TWIN_OUTCOME_TABLE
from src.digital_twin.effect.cohort_causal_estimator import treatment_contrast_shortfall
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
# ``specialty`` lives on hcp_profiles, its single source of truth.  The declared FK lets
# PostgREST embed the one matching profile without a second query or an IN-list cap.
# The collapse key (hcp_id, brand) + metric_date (so the collapse carries max_metric_date) +
# region + specialty (heterogeneity axes) + every treatment channel + pre-treatment
# confounders (market_share, triggers_total_count). No outcome: it is on TWIN_OUTCOME_TABLE.
_COHORT_COLUMNS = ",".join(
    [
        "hcp_id",
        "brand",
        "metric_date",
        "region",
        "hcp_profiles(specialty)",
        "market_share",
        "triggers_total_count",
        *_TREATMENT_COLUMNS,
    ]
)
_NUMERIC_COLUMNS: tuple[str, ...] = (
    "market_share",
    "triggers_total_count",
    *_TREATMENT_COLUMNS,
)
# Generous cap so we read the full per-brand cohort (~4.6k rows live) past PostgREST's
# default 1000-row page. Measured 2026-09-23: this read returns the exact per-brand count
# un-paged on the live server (docs/demos/results/2026-09-23_t2_premise_probe/).
_FETCH_LIMIT = 20000
_KEY = ["hcp_id", "brand"]
# The adoption read IS capped at max-rows (5,000 rows per brand live), so it is paged in a
# total order (hcp_id, brand is the table's unique key) and checked against the exact count.
_ADOPTION_PAGE_SIZE = 1000
_ADOPTION_MAX_PAGES = 100


def flatten_specialty_relation(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten PostgREST's embedded ``hcp_profiles`` object into ``specialty``.

    Missing profiles and blank specialty values stay missing; inventing an ``unknown``
    label here would make unavailable source data look like a measured subgroup.
    """
    if "hcp_profiles" not in df.columns:
        return df
    result = df.copy()
    values = result.pop("hcp_profiles").map(
        lambda value: value.get("specialty") if isinstance(value, Mapping) else None
    )
    values = values.map(lambda value: value.strip() if isinstance(value, str) else value)
    result["specialty"] = values.replace("", None)
    return result


async def _load_adoption(client: Any, brand: str) -> pd.DataFrame:
    """Every ``hcp_brand_adoption`` row of ``brand`` as ``hcp_id, brand, adopted``.

    Paged by ``range`` windows in the table's unique-key order, advancing by the rows each
    page actually returned, until the server's exact count is reached or a page comes back
    empty. A total that disagrees with the count, or a repeated key, raises: a partial
    outcome read would silently estimate on a subset of the cohort.
    """
    rows: list[dict[str, Any]] = []
    expected: Optional[int] = None
    for _page in range(_ADOPTION_MAX_PAGES):
        result = await (
            client.table(TWIN_OUTCOME_TABLE)
            .select(f"hcp_id,brand,{TWIN_OUTCOME_COLUMN}", count="exact")
            .eq("brand", brand)
            .eq("is_synthetic", True)
            .order("hcp_id")
            .order("brand")
            .range(len(rows), len(rows) + _ADOPTION_PAGE_SIZE - 1)
            .execute()
        )
        if expected is None:
            expected = getattr(result, "count", None)
            if expected is None:
                raise RuntimeError(
                    f"{TWIN_OUTCOME_TABLE}[{brand}]: the server returned no exact count, so "
                    "the paged read cannot be checked for completeness"
                )
        page = getattr(result, "data", None) or []
        rows.extend(page)
        if not page or len(rows) >= expected:
            break
    else:
        raise RuntimeError(
            f"{TWIN_OUTCOME_TABLE}[{brand}]: paged read hit {_ADOPTION_MAX_PAGES} pages "
            "before exhausting the rows"
        )
    if len(rows) != expected:
        raise RuntimeError(
            f"{TWIN_OUTCOME_TABLE}[{brand}]: paged read returned {len(rows)} rows but the "
            f"server reports {expected}; refusing to estimate on a partial outcome read"
        )
    frame = pd.DataFrame(rows, columns=[*_KEY, TWIN_OUTCOME_COLUMN])
    if frame.duplicated(_KEY).any():
        raise RuntimeError(
            f"{TWIN_OUTCOME_TABLE}[{brand}]: repeated (hcp_id, brand) rows in the paged read"
        )
    frame[TWIN_OUTCOME_COLUMN] = pd.to_numeric(frame[TWIN_OUTCOME_COLUMN], errors="coerce")
    return frame


async def load_cohort_frame(client: Any, brand: str) -> pd.DataFrame:
    """Load the brand's cohort as ONE row per (hcp_id, brand): the collapsed rollup
    (region, specialty, treatments, confounders) left-joined to ``adopted``.

    Returns an empty DataFrame when the brand has no rollup rows. A pair without an
    adoption row has a NULL outcome (see the module docstring). Row filtering and
    sufficiency checks are :func:`assess_cohort_frame`'s responsibility. Database errors
    propagate: :func:`build_cohort_provider_or_none` turns them into ``None``, the chat
    tool retries them.
    """
    result = await (
        client.table(COHORT_TABLE)
        .select(_COHORT_COLUMNS, count="exact")
        .eq("metric_type", COHORT_METRIC_TYPE)
        .eq("brand", brand)
        .eq("is_synthetic", True)
        .limit(_FETCH_LIMIT)
        .execute()
    )
    rows = getattr(result, "data", None) or []
    expected = getattr(result, "count", None)
    if expected is None or len(rows) != expected:
        # A server max-rows cap, or a cohort past _FETCH_LIMIT, would drop HCPs from the
        # collapse silently; so would trusting a read the server did not count.
        raise RuntimeError(
            f"{COHORT_TABLE}[{brand}]: the rollup read returned {len(rows)} rows against a "
            f"server count of {expected} (limit {_FETCH_LIMIT}); refusing to estimate on a "
            "cohort that may be truncated"
        )
    df = flatten_specialty_relation(pd.DataFrame(rows))
    if df.empty:
        return df
    for col in _NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["brand"] = brand  # the read is .eq-filtered on it; the collapse keys on it
    # Collapse only rows that carry at least one treatment channel. The daily per-HCP ETL keeps
    # adding rollup rows after the plant with every channel NULL (219 live on 2026-09-28, dated
    # after the planted window; 168 of their pairs also have planted rows): summed into a pair,
    # their trigger counts and market share would hand the estimator confounders the DGP never
    # collapsed. A row with some channels (a partial substrate) is kept.
    present = [c for c in _TREATMENT_COLUMNS if c in df.columns]
    df = df[df[present].notna().any(axis=1)] if present else df.iloc[0:0]
    if df.empty:
        return df.reset_index(drop=True)
    collapsed = collapse_per_hcp_brand(df)
    adoption = await _load_adoption(client, brand)
    return collapsed.merge(adoption, on=_KEY, how="left", validate="one_to_one")


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
    has_outcome = TWIN_OUTCOME_COLUMN in df.columns
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
    required = [treatment_col, TWIN_OUTCOME_COLUMN, "region", *COHORT_CONFOUNDERS]
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
                "n_null_outcome_rows": int(df[TWIN_OUTCOME_COLUMN].isna().sum()),
                "n_null_region_rows": int(df["region"].isna().sum()),
                "n_null_confounder_rows": int(
                    df[list(COHORT_CONFOUNDERS)].isna().any(axis=1).sum()
                ),
            },
        )
    # The estimator's own median-split rule, on the same rows: a channel it would refuse
    # (NO_TREATMENT_CONTRAST) is not usable here either (codex r2/r3).
    shortfall = treatment_contrast_shortfall(pd.to_numeric(usable[treatment_col], errors="coerce"))
    if shortfall is not None:
        return CohortUsability(None, EffectCause.NO_TREATMENT_CONTRAST, shortfall)
    return CohortUsability(CohortEffectDataProvider(usable))


class ChannelAvailability(dict[str, bool]):
    """``{intervention: usable}``, plus how many column probes ERRORED.

    All-False has two very different causes: every probe MEASURED too few usable rows (the
    cohort's treatment data is gone — the remedy is a re-plant, a production write), or the probes
    could not run (a connection blip — the remedy is to ask again). A plain dict cannot say which,
    and the difference decides whether an operator is told to write to production (codex r1).
    """

    n_probe_errors: int = 0


async def cohort_treatment_availability(client: Any, brand: str) -> ChannelAvailability:
    """Per-intervention effect availability for a brand: ``{intervention: usable}``.

    Drives ``available_for_effect`` in ``GET /digital-twin/intervention-types`` — HONEST
    per channel, so a substrate holding only some channels (pre-backfill, or future RWD
    with partial coverage) advertises exactly what ``/simulate`` can estimate. It loads
    the SAME frame ``/simulate`` loads (:func:`load_cohort_frame`) once for all eight
    channels and applies the SAME rule (:func:`assess_cohort_frame`), so the two cannot
    disagree: the binding constraint is the (hcp_id, brand) join to ``adopted`` (3,354 of
    5,000 HCPs per brand live), which a per-column null count on ``business_metrics``
    cannot see. Degrades to ``False`` for every channel if the load errors (advisory
    only); never raises. ``n_probe_errors`` then counts every channel as unmeasured, so
    "could not measure" is not read as "empty".
    """
    columns = list(_TREATMENT_COLUMNS)
    try:
        df = await load_cohort_frame(client, brand)
    except Exception as e:
        logger.warning("cohort availability check failed for %s: %s", brand, e)
        availability = ChannelAvailability(
            (intervention, False) for intervention in INTERVENTION_TREATMENT_MAP
        )
        availability.n_probe_errors = len(columns)
        return availability
    return ChannelAvailability(
        (intervention, assess_cohort_frame(df, intervention).provider is not None)
        for intervention in INTERVENTION_TREATMENT_MAP
    )
