"""Cohort-resolution service: resolve ``(brand, region)`` to a real cohort frame.

Background
----------
The tool-composer-remediation (issue #770/#774-777) wired the chat caller
``chatbot_tools._resolve_cohort_frame`` so that a real cohort DataFrame could be
threaded into the Tool Composer context (``context["estimation_data"]``). That
wiring only produced data when the caller supplied an explicit ``data_source``
(a parquet/s3 path or table name); an arbitrary ``(brand, region)`` pair could
NOT be resolved to a production frame.

This service is the deferred data-layer follow-up (issue #779). It resolves a
``(brand, region)`` pair to a real cohort DataFrame from the canonical
``patient_journeys`` table (which carries ``brand`` and ``geographic_region``
columns plus the causal variables the composable tools need) WITHOUT requiring
an explicit ``data_source``.

Resolution order (first non-empty frame wins):

1. **Explicit ``data_source``** -> the tier0 ``CohortConstructorAgent`` loader
   (parquet/s3/table). Preserves the exact behavior R4 shipped.
2. **Canonical ``patient_journeys``** filtered by normalized brand +
   ``geographic_region``.

Anti-mocking discipline
-----------------------
This service NEVER fabricates a synthetic cohort. It returns ``None`` (fail
closed) when:

* an explicit ``data_source`` yields nothing,
* a supplied brand/region is not a recognized enum member (so we do not silently
  return a wrong-population cohort), or
* the canonical query returns zero rows.

Callers then honestly proceed without ``estimation_data`` and the composable
tools fail closed in turn (descriptive ``RuntimeError``), rather than returning
plausible-but-fake values.

Both ``chatbot_tools`` (chat path) and the ``cohort_builder`` composable tool
(issue #778) route through :func:`resolve_cohort_frame` so there is a SINGLE
cohort-loading code path, not two divergent ones.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical cohort source. ``patient_journeys`` carries ``brand`` (brand_type),
# ``geographic_region`` (region_type) and the causal variables (engagement_score,
# treatment_initiated, disease_severity, academic_hcp, age_at_diagnosis).
CANONICAL_COHORT_TABLE = "patient_journeys"

# brand_type enum (database/core/e2i_ml_complete_v3_schema.sql). Lowercase key ->
# canonical DB spelling. Drift mode is fail-closed: a newly-added brand resolves
# to None until added here, never a wrong/fabricated cohort.
_BRAND_CANONICAL = {
    b.lower(): b for b in ("Remibrutinib", "Fabhalta", "Kisqali", "competitor", "other")
}

# region_type enum: US census regions, lowercase. NOTE this is NOT "US/EU/APAC".
_REGION_CANONICAL = {r.lower(): r for r in ("northeast", "south", "midwest", "west")}

# PostgREST imposes a configured max row count per request (default 1000). When a
# canonical-table query returns at least this many rows without an explicit
# ``limit``, the cohort may have been silently truncated to a sample.
_POSTGREST_DEFAULT_MAX_ROWS = 1000


def _normalize_brand(brand: Optional[str]) -> Optional[str]:
    """Map a brand string to its canonical ``brand_type`` spelling, else None."""
    if not brand:
        return None
    return _BRAND_CANONICAL.get(brand.strip().lower())


def _normalize_region(region: Optional[str]) -> Optional[str]:
    """Map a region string to its canonical ``region_type`` spelling, else None."""
    if not region:
        return None
    return _REGION_CANONICAL.get(region.strip().lower())


def _load_tier0_agent() -> Any:
    """Return a tier0 ``CohortConstructorAgent`` (indirection eases testing)."""
    from src.agents.cohort_constructor.tier0_integration import (
        CohortConstructorAgent,
    )

    return CohortConstructorAgent()


def _resolve_via_data_source(
    brand: Optional[str],
    region: Optional[str],
    data_source: str,
) -> Optional[pd.DataFrame]:
    """Resolve via the tier0 loader (explicit parquet/s3/table data_source).

    Mirrors the behavior R4 shipped in ``chatbot_tools._resolve_cohort_frame``.
    """
    agent = _load_tier0_agent()
    result = agent.run(
        {
            "scope_spec": {
                "brand": brand or "",
                "indication": "",
                "target_population": region or "",
                "business_objective": "tool_composer_estimation",
            },
            "patient_data_source": data_source,
            "use_existing_config": True,
        }
    )
    frame = result.get("eligible_patients")
    if frame is None or getattr(frame, "empty", True):
        return None
    return frame


def _resolve_via_patient_journeys(
    brand: Optional[str],
    region: Optional[str],
    *,
    supabase_client: Optional[Any] = None,
    limit: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Resolve from the canonical ``patient_journeys`` table.

    Filters by normalized brand and ``geographic_region``. A supplied brand or
    region that is not a recognized enum member returns ``None`` WITHOUT issuing
    a query (we cannot faithfully resolve the requested population, so we fail
    closed rather than return a wrong cohort or trigger a DB enum error).
    """
    norm_brand = _normalize_brand(brand)
    norm_region = _normalize_region(region)

    # A NON-EMPTY brand/region that does not map to a known enum member -> fail
    # closed (never silently widen to an all-population cohort). Empty/whitespace
    # is treated as "not specified" (no filter), identical to None.
    if brand and brand.strip() and norm_brand is None:
        logger.info("cohort_resolution: unrecognized brand %r -> fail closed", brand)
        return None
    if region and region.strip() and norm_region is None:
        logger.info("cohort_resolution: unrecognized region %r -> fail closed", region)
        return None

    client = supabase_client if supabase_client is not None else _default_client()

    query = client.table(CANONICAL_COHORT_TABLE).select("*")
    if norm_brand:
        query = query.eq("brand", norm_brand)
    if norm_region:
        query = query.eq("geographic_region", norm_region)
    if limit:
        query = query.limit(limit)

    response = query.execute()
    rows = getattr(response, "data", None) or []
    if not rows:
        return None
    # No silent caps: if we hit the PostgREST default max without an explicit
    # limit, the cohort may be a truncated sample -- surface it rather than
    # presenting a partial frame as a complete cohort.
    if limit is None and len(rows) >= _POSTGREST_DEFAULT_MAX_ROWS:
        logger.warning(
            "cohort_resolution: patient_journeys returned %d rows (>= PostgREST "
            "default cap %d) for brand=%r region=%r; cohort may be truncated -- "
            "pass limit= explicitly or paginate.",
            len(rows),
            _POSTGREST_DEFAULT_MAX_ROWS,
            norm_brand,
            norm_region,
        )
    return pd.DataFrame(rows)


def _default_client() -> Any:
    """Return the cached service-role Supabase client."""
    from src.repositories import get_supabase_client

    return get_supabase_client()


def resolve_cohort_frame(
    brand: Optional[str],
    region: Optional[str],
    *,
    data_source: Optional[str] = None,
    supabase_client: Optional[Any] = None,
    limit: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Resolve a ``(brand, region)`` pair to a real cohort DataFrame.

    Args:
        brand: Brand context (e.g. ``"Kisqali"``); case-insensitive, mapped to
            the ``brand_type`` enum. Unrecognized -> ``None`` (fail closed).
        region: Region context (e.g. ``"Northeast"``); case-insensitive, mapped
            to the ``region_type`` enum (US census regions). Unrecognized ->
            ``None``.
        data_source: Optional explicit parquet/s3 path or table name. When
            supplied, the tier0 loader is used (preserves R4 behavior) and the
            canonical ``patient_journeys`` path is skipped.
        supabase_client: Optional injected client (testing / reuse). Defaults to
            the cached service-role client.
        limit: Optional row cap for the canonical ``patient_journeys`` path only
            (ignored on the explicit-``data_source`` path, where the tier0 loader
            controls its own bounds). PostgREST also imposes its own configured
            max (default 1000), so very large cohorts may be a sample -- a
            WARNING is logged when that cap is hit without an explicit ``limit``.

    Returns:
        A non-empty ``pd.DataFrame`` on success, else ``None``. NEVER a
        fabricated frame.

    Raises:
        Genuine infrastructure errors (e.g. ``ServiceConnectionError`` from the
        client factory) propagate so the caller can log-and-proceed; the tools
        then fail closed honestly.
    """
    if data_source:
        return _resolve_via_data_source(brand, region, data_source)
    return _resolve_via_patient_journeys(
        brand, region, supabase_client=supabase_client, limit=limit
    )
