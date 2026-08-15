"""
KPI API Routes

REST endpoints for on-demand KPI calculation and management.

Endpoints:
----------
- GET  /api/kpis               - List all KPIs with optional filters
- GET  /api/kpis/{kpi_id}      - Get calculated value for a single KPI
- GET  /api/kpis/{kpi_id}/metadata - Get KPI metadata/definition
- POST /api/kpis/calculate     - Calculate a specific KPI
- POST /api/kpis/batch         - Batch calculate multiple KPIs
- POST /api/kpis/invalidate    - Invalidate cached KPI values
- GET  /api/kpis/workstreams   - List available workstreams
- GET  /api/kpis/health        - KPI system health check

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies.auth import require_admin, require_auth
from src.api.dependencies.supabase_client import get_supabase
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse
from src.api.schemas.kpi import (
    BatchKPICalculationRequest,
    BatchKPICalculationResponse,
    CacheInvalidationRequest,
    CacheInvalidationResponse,
    KPICalculationRequest,
    KPIHealthResponse,
    KPIHistoryCoverageEntry,
    KPIHistoryCoverageResponse,
    KPIHistoryPoint,
    KPIHistoryResponse,
    KPIHistoryScopeEntry,
    KPIHistorySegmentSeries,
    KPIListResponse,
    KPIMetadataResponse,
    KPINowcastHistoryResponse,
    KPINowcastPoint,
    KPIResultResponse,
    KPISegmentedHistoryResponse,
    KPIThresholdResponse,
    WorkstreamInfo,
    WorkstreamListResponse,
)
from src.kpi.calculator import KPICalculator
from src.kpi.calculators.brand_specific import BrandSpecificCalculator
from src.kpi.calculators.business_impact import BusinessImpactCalculator
from src.kpi.calculators.causal_metrics import CausalMetricsCalculator
from src.kpi.calculators.data_quality import DataQualityCalculator
from src.kpi.calculators.model_performance import ModelPerformanceCalculator
from src.kpi.calculators.trigger_performance import TriggerPerformanceCalculator
from src.kpi.models import CausalLibrary, Workstream
from src.kpi.registry import get_registry

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/kpis",
    tags=["KPIs"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        404: {"model": ErrorResponse, "description": "KPI not found"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================


#: Workstream -> per-workstream calculator class. Registered on every
#: KPICalculator so the KPI grid / single-KPI / batch endpoints compute real
#: values instead of falling through to the unimplemented generic table path
#: (which returns honest None+error). The calculators were hardened to FAIL-LOUD
#: on missing data (no fabricated 0.0/0.5 placeholders — #421/#439/#574/#577),
#: so registering them cannot resurface the masked-failure risk that originally
#: kept them unregistered. Under E2I_KPI_INCLUDE_SYNTHETIC they read the
#: synthetic-gold twins (see src/kpi/synthetic_mode.py).
_WORKSTREAM_CALCULATORS = {
    Workstream.WS1_DATA_QUALITY: DataQualityCalculator,
    Workstream.WS1_MODEL_PERFORMANCE: ModelPerformanceCalculator,
    Workstream.WS2_TRIGGERS: TriggerPerformanceCalculator,
    Workstream.WS3_BUSINESS: BusinessImpactCalculator,
    Workstream.BRAND_SPECIFIC: BrandSpecificCalculator,
    Workstream.CAUSAL_METRICS: CausalMetricsCalculator,
}


def get_kpi_calculator() -> KPICalculator:
    """Get KPI calculator instance with all per-workstream calculators registered.

    Returns:
        KPICalculator instance
    """
    # In production, this would be a singleton or use proper DI.
    db = get_supabase()
    calc = KPICalculator(db_connection=db)
    for workstream, calculator_cls in _WORKSTREAM_CALCULATORS.items():
        # Pass the SAME api-layer client (get_supabase supports SUPABASE_KEY, the
        # calculators' lazy get_supabase_client() does NOT — the #845 key-surface
        # trap) so a SUPABASE_KEY-only deployment doesn't fail at lazy client
        # creation. db may be None when unconfigured -> the calculator's lazy
        # db_client property still applies as a fallback. Calculators fail-loud on
        # missing data, never a fabricated placeholder.
        calc.register_calculator(workstream, calculator_cls(db_client=db))
    return calc


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _workstream_from_string(ws_str: str | None) -> Workstream | None:
    """Convert workstream string to enum."""
    if ws_str is None:
        return None

    mapping = {
        "ws1_data_quality": Workstream.WS1_DATA_QUALITY,
        "ws1_model_performance": Workstream.WS1_MODEL_PERFORMANCE,
        "ws2_triggers": Workstream.WS2_TRIGGERS,
        "ws3_business": Workstream.WS3_BUSINESS,
        "brand_specific": Workstream.BRAND_SPECIFIC,
        "causal_metrics": Workstream.CAUSAL_METRICS,
    }
    return mapping.get(ws_str.lower())


def _causal_library_from_string(lib_str: str | None) -> CausalLibrary | None:
    """Convert causal library string to enum."""
    if lib_str is None:
        return None

    mapping = {
        "dowhy": CausalLibrary.DOWHY,
        "econml": CausalLibrary.ECONML,
        "causalml": CausalLibrary.CAUSALML,
        "networkx": CausalLibrary.NETWORKX,
        "none": CausalLibrary.NONE,
    }
    return mapping.get(lib_str.lower())


def _metadata_to_response(kpi: Any) -> KPIMetadataResponse:
    """Convert KPIMetadata to API response."""
    threshold_resp = None
    if kpi.threshold:
        threshold_resp = KPIThresholdResponse(
            target=kpi.threshold.target,
            warning=kpi.threshold.warning,
            critical=kpi.threshold.critical,
            ideal=kpi.threshold.ideal,
            good_tolerance=kpi.threshold.good_tolerance,
            warning_tolerance=kpi.threshold.warning_tolerance,
        )

    return KPIMetadataResponse(
        id=kpi.id,
        name=kpi.name,
        definition=kpi.definition,
        formula=kpi.formula,
        calculation_type=kpi.calculation_type.value,
        workstream=kpi.workstream.value,
        tables=kpi.tables,
        columns=kpi.columns,
        view=kpi.view,
        threshold=threshold_resp,
        unit=kpi.unit,
        value_format=kpi.value_format,
        frequency=kpi.frequency,
        primary_causal_library=kpi.primary_causal_library.value,
        brand=kpi.brand,
        note=kpi.note,
    )


def _result_to_response(result: Any) -> KPIResultResponse:
    """Convert KPIResult to API response."""
    ci = None
    if result.confidence_interval:
        ci = list(result.confidence_interval)

    causal_lib = None
    if result.causal_library_used:
        causal_lib = (
            result.causal_library_used.value
            if hasattr(result.causal_library_used, "value")
            else str(result.causal_library_used)
        )

    # Provenance: 'synthetic' when the value was computed in the
    # E2I_KPI_INCLUDE_SYNTHETIC demo/review mode (stamped by KPICalculator and
    # carried through the cache), else 'database'. Lets the FE badge synthetic
    # figures rather than reading them as real-world data.
    metadata = result.metadata or {}
    data_source = "synthetic" if metadata.get("include_synthetic") else "database"

    # #1640: derived from the registry, and from the calculator's runtime branch
    # when it recorded one (ROI is the case that needs it).
    from src.kpi.measure_basis import measure_basis_for_kpi
    from src.kpi.registry import get_registry

    kpi_meta = get_registry().get(result.kpi_id)
    measure_basis = measure_basis_for_kpi(kpi_meta, metadata) if kpi_meta else None

    return KPIResultResponse(
        kpi_id=result.kpi_id,
        value=result.value,
        measure_basis=measure_basis,
        status=result.status.value if hasattr(result.status, "value") else result.status,
        calculated_at=result.calculated_at,
        cached=result.cached,
        cache_expires_at=result.cache_expires_at,
        error=result.error,
        data_source=data_source,
        # Region provenance (#1538): consumers caption with the region ONLY
        # when region_status == "applied" — every other calculator keeps its
        # global/portfolio value under a region ask.
        region_requested=getattr(result, "region_requested", None),
        region_applied=getattr(result, "region_applied", None),
        region_status=getattr(result, "region_status", "default"),
        causal_library_used=causal_lib,
        confidence_interval=ci,
        p_value=result.p_value,
        effect_size=result.effect_size,
        metadata=result.metadata,
    )


# =============================================================================
# LIST & METADATA ENDPOINTS
# =============================================================================


@router.get(
    "",
    response_model=KPIListResponse,
    summary="List all KPIs",
    description="Get a list of all available KPIs with optional filtering",
    operation_id="list_kpis",
)
async def list_kpis(
    workstream: str | None = Query(
        default=None,
        description="Filter by workstream (e.g., ws1_data_quality)",
    ),
    causal_library: str | None = Query(
        default=None,
        description="Filter by causal library (dowhy, econml, causalml, networkx)",
    ),
    calculator: KPICalculator = Depends(get_kpi_calculator),
) -> KPIListResponse:
    """List all available KPIs with optional filters.

    Args:
        workstream: Filter by workstream
        causal_library: Filter by causal library
        calculator: KPI calculator instance

    Returns:
        List of KPI metadata
    """
    try:
        ws_enum = _workstream_from_string(workstream)
        lib_enum = _causal_library_from_string(causal_library)

        kpis = calculator.list_kpis(workstream=ws_enum, causal_library=lib_enum)

        return KPIListResponse(
            kpis=[_metadata_to_response(kpi) for kpi in kpis],
            total=len(kpis),
            workstream=workstream,
            causal_library=causal_library,
        )

    except Exception as e:
        logger.error(f"Failed to list KPIs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list KPIs: {str(e)}",
        )


@router.get(
    "/workstreams",
    response_model=WorkstreamListResponse,
    summary="List workstreams",
    description="Get a list of all available KPI workstreams",
    operation_id="list_workstreams",
)
async def list_workstreams(
    calculator: KPICalculator = Depends(get_kpi_calculator),
) -> WorkstreamListResponse:
    """List all available workstreams with KPI counts.

    Args:
        calculator: KPI calculator instance

    Returns:
        List of workstream information
    """
    workstream_info = {
        Workstream.WS1_DATA_QUALITY: (
            "WS1: Data Quality",
            "Data completeness, freshness, and validation metrics",
        ),
        Workstream.WS1_MODEL_PERFORMANCE: (
            "WS1: Model Performance",
            "Model accuracy, calibration, and prediction quality",
        ),
        Workstream.WS2_TRIGGERS: (
            "WS2: Trigger Performance",
            "Alert effectiveness, action rates, and trigger validation",
        ),
        Workstream.WS3_BUSINESS: (
            "WS3: Business Impact",
            "ROI, conversion rates, and business outcome metrics",
        ),
        Workstream.BRAND_SPECIFIC: (
            "Brand-Specific KPIs",
            "KPIs specific to Remibrutinib, Fabhalta, or Kisqali",
        ),
        Workstream.CAUSAL_METRICS: (
            "Causal Metrics",
            "Treatment effect estimates and causal inference quality",
        ),
    }

    workstreams = []
    for ws in Workstream:
        kpis = calculator.list_kpis(workstream=ws)
        name, description = workstream_info.get(ws, (ws.value, None))
        workstreams.append(
            WorkstreamInfo(
                id=ws.value,
                name=name,
                kpi_count=len(kpis),
                description=description,
            )
        )

    return WorkstreamListResponse(
        workstreams=workstreams,
        total=len(workstreams),
    )


# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================


@router.get(
    "/health",
    response_model=KPIHealthResponse,
    summary="KPI system health",
    description="Check the health of the KPI calculation system",
    operation_id="kpi_health_check",
)
async def health_check(
    calculator: KPICalculator = Depends(get_kpi_calculator),
) -> KPIHealthResponse:
    """Check KPI system health.

    Args:
        calculator: KPI calculator instance

    Returns:
        System health status
    """
    try:
        get_registry()
        all_kpis = calculator.list_kpis()

        # Determine available workstreams
        workstreams = list({kpi.workstream.value for kpi in all_kpis})

        # Check database connectivity
        db_connected = calculator._db is not None

        # Determine overall status
        status_str = "healthy"
        error = None

        if len(all_kpis) == 0:
            status_str = "degraded"
            error = "No KPIs loaded in registry"
        elif not db_connected:
            status_str = "degraded"
            error = "Database not connected"

        return KPIHealthResponse(
            status=status_str,
            registry_loaded=len(all_kpis) > 0,
            total_kpis=len(all_kpis),
            cache_enabled=calculator._cache.enabled,
            cache_size=calculator._cache.size(),
            database_connected=db_connected,
            workstreams_available=workstreams,
            last_calculation=None,  # Would need tracking in calculator
            error=error,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return KPIHealthResponse(
            status="unhealthy",
            registry_loaded=False,
            total_kpis=0,
            cache_enabled=False,
            cache_size=0,
            database_connected=False,
            workstreams_available=[],
            last_calculation=None,
            error=str(e),
        )


# =============================================================================
# KPI HISTORY COVERAGE (static path — declared BEFORE the /{kpi_id} routes so a
# KPI literally named "history" could never shadow it, and vice versa)
# =============================================================================


@router.get(
    "/history/coverage",
    response_model=KPIHistoryCoverageResponse,
    summary="KPI history coverage map",
    description=(
        "Which KPIs have a real materialized series in kpi_history, and in which "
        "brand scopes ('' = global; per-brand-only KPIs such as WS3-BI-007 NBRx "
        "carry no '' scope by design). KPIs absent from the map have no history — "
        "the Time-Series page uses this to badge its dropdown and scope its brand "
        "selector instead of guessing."
    ),
    operation_id="get_kpi_history_coverage",
)
async def get_kpi_history_coverage() -> KPIHistoryCoverageResponse:
    """Return the per-KPI history coverage summary (empty map when none)."""
    from src.repositories.kpi_history import get_kpi_history_repository

    repo = await get_kpi_history_repository()
    rows = await repo.get_coverage()

    by_kpi: dict[str, KPIHistoryCoverageEntry] = {}
    for row in rows:
        kpi_id = str(row.get("kpi_id") or "")
        if not kpi_id:
            continue
        entry = by_kpi.get(kpi_id)
        if entry is None:
            entry = KPIHistoryCoverageEntry(kpi_id=kpi_id)
            by_kpi[kpi_id] = entry
        brand = str(row.get("brand") if row.get("brand") is not None else "")
        # Pre-126 view rows carry no region key — they are the region='' axis.
        region = str(row.get("region") if row.get("region") is not None else "")
        points = int(row.get("points") or 0)
        first = row.get("first_date")
        last = row.get("last_date")
        entry.scopes.append(
            KPIHistoryScopeEntry(
                brand=brand,
                region=region,
                points=points,
                first_date=str(first) if first else None,
                last_date=str(last) if last else None,
            )
        )
        if region:
            # Region-scoped rows surface through `scopes` only: the brand
            # axis keeps its pre-region meaning (#1536 — folding them in
            # would duplicate brand entries and inflate point counts).
            continue
        entry.brands.append(brand)
        entry.points += points
        if first and (entry.first_date is None or str(first) < entry.first_date):
            entry.first_date = str(first)
        if last and (entry.last_date is None or str(last) > entry.last_date):
            entry.last_date = str(last)
    for entry in by_kpi.values():
        entry.brands.sort()
        entry.scopes.sort(key=lambda s: (s.brand, s.region))
    coverage = sorted(by_kpi.values(), key=lambda e: e.kpi_id)
    return KPIHistoryCoverageResponse(coverage=coverage, total=len(coverage))


# =============================================================================
# KPI METADATA ENDPOINTS
# =============================================================================


@router.get(
    "/{kpi_id}/metadata",
    response_model=KPIMetadataResponse,
    summary="Get KPI metadata",
    description="Get metadata and definition for a specific KPI",
    operation_id="get_kpi_metadata",
)
async def get_kpi_metadata(
    kpi_id: str,
    calculator: KPICalculator = Depends(get_kpi_calculator),
) -> KPIMetadataResponse:
    """Get metadata for a specific KPI.

    Args:
        kpi_id: KPI identifier
        calculator: KPI calculator instance

    Returns:
        KPI metadata

    Raises:
        HTTPException: If KPI not found
    """
    kpi = calculator.get_kpi_metadata(kpi_id)

    if kpi is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"KPI not found: {kpi_id}",
        )

    return _metadata_to_response(kpi)


# =============================================================================
# CALCULATION ENDPOINTS
# =============================================================================


@router.get(
    "/{kpi_id}",
    response_model=KPIResultResponse,
    summary="Get KPI value",
    description="Calculate and return the current value for a KPI",
    operation_id="get_kpi_value",
)
async def get_kpi_value(
    kpi_id: str,
    use_cache: bool = Query(default=True, description="Use cached value if available"),
    force_refresh: bool = Query(default=False, description="Force recalculation"),
    brand: str | None = Query(default=None, description="Brand filter"),
    region: str | None = Query(default=None, description="Geographic region filter"),
    segment: str | None = Query(
        default=None,
        description="Severity tier filter (low_severity, medium_severity, high_severity)",
    ),
    therapy_line: str | None = Query(default=None, description="Line-of-therapy filter ('0'-'3')"),
    biologic: str | None = Query(
        default=None,
        description="Biologic-status filter ('naive'/'experienced'); Remibrutinib only",
    ),
    ige_tier: str | None = Query(
        default=None,
        description="IgE-tertile filter ('low'/'medium'/'high'); Remibrutinib only",
    ),
    calculator: KPICalculator = Depends(get_kpi_calculator),
) -> KPIResultResponse:
    """Get the calculated value for a specific KPI.

    Args:
        kpi_id: KPI identifier
        use_cache: Whether to use cached results
        force_refresh: Force recalculation
        brand: Optional brand filter
        region: Optional geographic region filter
        segment: Optional severity tier filter
        therapy_line: Optional line-of-therapy filter
        biologic: Optional biologic-status filter (Remibrutinib only)
        ige_tier: Optional IgE-tertile filter (Remibrutinib only)
        calculator: KPI calculator instance

    Returns:
        Calculated KPI result

    Raises:
        HTTPException: If KPI not found or calculation fails
    """
    try:
        context: dict[str, Any] = {}
        if brand:
            context["brand"] = brand
        if region:
            context["region"] = region
        if segment:
            context["segment"] = segment
        if therapy_line:
            context["therapy_line"] = therapy_line
        if biologic:
            context["biologic"] = biologic
        if ige_tier:
            context["ige_tier"] = ige_tier

        result = calculator.calculate(
            kpi_id=kpi_id,
            use_cache=use_cache,
            force_refresh=force_refresh,
            context=context,
        )

        if result.error and "not found" in result.error.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.error,
            )

        return _result_to_response(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to calculate KPI {kpi_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate KPI: {str(e)}",
        )


@router.get(
    "/{kpi_id}/history",
    response_model=KPIHistoryResponse,
    summary="Get KPI history (monthly time series)",
    description=(
        "Materialized monthly KPI values (from kpi_history) for the Time-Series "
        "KPI-history view. Returns an EMPTY series for point-in-time KPIs that have "
        "no real history — the UI shows an honest empty-state rather than a "
        "fabricated flat line."
    ),
    operation_id="get_kpi_history",
)
async def get_kpi_history(
    kpi_id: str,
    brand: str | None = Query(default=None, description="Brand filter ('' / omitted = global)"),
    region: str | None = Query(default=None, description="Region filter ('' / omitted = all)"),
    start_date: str | None = Query(default=None, description="Earliest metric_date (YYYY-MM-DD)"),
    end_date: str | None = Query(default=None, description="Latest metric_date (YYYY-MM-DD)"),
) -> KPIHistoryResponse:
    """Return the date-ordered monthly history for a KPI (empty when none exists).

    No segment/therapy_line params: this reads the materialized kpi_history
    table (not the calculator/RPC), which has no patient-segment dimension —
    threading an axis here would silently return unsegmented history.
    """
    from src.repositories.kpi_history import get_kpi_history_repository

    repo = await get_kpi_history_repository()
    rows = await repo.get_history(
        kpi_id, brand=brand, region=region, start_date=start_date, end_date=end_date
    )
    points = [
        KPIHistoryPoint(
            metric_date=str(r.get("metric_date")),
            value=float(r["value"]),
            status=r.get("status"),
        )
        for r in rows
        if r.get("value") is not None and r.get("metric_date")
    ]
    # The chart surface #1640 is about: `renderKpiTrend` plots these points, and
    # the same answer can carry a business_metrics TRx figure (~73x apart).
    from src.kpi.measure_basis import materialized_history_basis

    kpi_meta = get_registry().get(kpi_id)
    return KPIHistoryResponse(
        kpi_id=kpi_id,
        brand=brand or "",
        region=region or "",
        count=len(points),
        points=points,
        measure_basis=materialized_history_basis(kpi_meta, rows=rows) if kpi_meta else None,
    )


@router.get(
    "/{kpi_id}/history/segmented",
    response_model=KPISegmentedHistoryResponse,
    summary="Get KPI history split by patient axis (severity tier / line of therapy)",
    description=(
        "Monthly KPI series per axis bucket, computed live from the vetted "
        "kpi_query registry (migration 110) — NOT from the materialized "
        "kpi_history table, which has no patient-segment dimension. Month "
        "bucketing and partial-edge-month trimming mirror the history "
        "backfill, so the bucket series partition the headline series. Only "
        "the Rx-volume family (WS3-BI-005 TRx, WS3-BI-006 NRx, WS3-BI-007 "
        "NBRx) supports axes."
    ),
    operation_id="get_kpi_history_segmented",
)
async def get_kpi_history_segmented(
    kpi_id: str,
    axis: str = Query(
        ...,
        description="'segment' (severity tier: low/medium/high) or 'therapy_line' (LOT 0-3)",
    ),
    brand: str | None = Query(default=None, description="Brand filter ('' / omitted = global)"),
    value: str | None = Query(
        default=None,
        description=(
            "Restrict to one bucket: a severity tier (low_severity/medium_severity/"
            "high_severity) or a line-of-therapy count ('0'-'3'). Omitted = all buckets."
        ),
    ),
    start_date: str | None = Query(default=None, description="Earliest metric_date (YYYY-MM-DD)"),
    end_date: str | None = Query(default=None, description="Latest metric_date (YYYY-MM-DD)"),
) -> KPISegmentedHistoryResponse:
    """Return per-bucket monthly history for an axis-capable KPI.

    422 (not empty-series) for unsupported KPIs/axes/values: the chat chart
    renderer relays the error honestly instead of drawing an empty chart for
    a request the substrate can never serve.
    """
    from src.kpi.segmented_history import (
        AXIS_SUFFIXES,
        SEGMENTED_KPI_QUERY_FAMILIES,
        canonical_buckets,
        fetch_segmented_rows,
        shape_segmented_series,
    )

    if axis not in AXIS_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown axis '{axis}'. Supported: {sorted(AXIS_SUFFIXES)}",
        )
    if kpi_id not in SEGMENTED_KPI_QUERY_FAMILIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"KPI '{kpi_id}' has no {axis}-level series. Axis-capable KPIs: "
                f"{sorted(SEGMENTED_KPI_QUERY_FAMILIES)}"
            ),
        )
    if value is not None and value not in canonical_buckets(axis):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown {axis} value '{value}'. Expected one of {canonical_buckets(axis)}",
        )

    # '' is the UI's All-Brands scope, but the migration-110 SQL only treats
    # NULL as all-brands ('' would be a literal brand that never matches).
    rows = await fetch_segmented_rows(kpi_id, axis=axis, brand=brand or None)
    series, data_through = shape_segmented_series(
        rows, axis=axis, value=value, start_date=start_date, end_date=end_date
    )
    from src.kpi.measure_basis import registry_query_basis
    from src.kpi.segmented_history import segmented_query_id

    return KPISegmentedHistoryResponse(
        measure_basis=await registry_query_basis(segmented_query_id(kpi_id, axis)),
        kpi_id=kpi_id,
        brand=brand or "",
        axis=axis,
        data_through=data_through,
        count=len(series),
        series=[
            KPIHistorySegmentSeries(
                key=s["key"],
                label=s["label"],
                count=s["count"],
                points=[KPIHistoryPoint(**p) for p in s["points"]],
            )
            for s in series
        ],
    )


@router.get(
    "/{kpi_id}/history/nowcast",
    response_model=KPINowcastHistoryResponse,
    summary="Get KPI history with claims-lag provisional/nowcast overlay",
    description=(
        "Monthly mature / provisional / nowcast series for the Rx-volume family "
        "(WS3-BI-005 TRx, WS3-BI-006 NRx, WS3-BI-007 NBRx), computed live from "
        "the migration-116 claims-arrival lag triangle (backlog #45) — NOT from "
        "the materialized kpi_history table, whose figures stay the mature "
        "values. The completion factor is re-estimated empirically from mature "
        "service months (chain-ladder); when that cannot be done honestly the "
        "response says insufficient_maturity=true with a reason and carries no "
        "nowcast values."
    ),
    operation_id="get_kpi_history_nowcast",
)
async def get_kpi_history_nowcast(
    kpi_id: str,
    brand: str | None = Query(default=None, description="Brand filter ('' / omitted = global)"),
    start_date: str | None = Query(default=None, description="Earliest metric_date (YYYY-MM-DD)"),
    end_date: str | None = Query(default=None, description="Latest metric_date (YYYY-MM-DD)"),
) -> KPINowcastHistoryResponse:
    """Return the provisional/nowcast monthly series for an Rx-volume KPI.

    Unknown KPI ids are 404 FIRST — the same registry lookup /metadata uses
    (``calculator.get_kpi_metadata`` delegates to ``registry.get``). Then 422
    (not empty-series) for known-but-off-family KPIs — the sibling
    ``/history/segmented`` convention for "this KPI can never serve this
    view": the caller relays the error honestly instead of drawing an empty
    chart, and one route family keeps one unsupported-KPI contract.
    """
    from src.kpi.nowcast.completion_factor import (
        NOWCAST_KPI_QUERY_FAMILIES,
        estimate_completion_from_rows,
        fetch_nowcast_rows,
    )

    if get_registry().get(kpi_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"KPI not found: {kpi_id}",
        )

    if kpi_id not in NOWCAST_KPI_QUERY_FAMILIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"KPI '{kpi_id}' has no claims-lag nowcast series. Nowcast-capable "
                f"KPIs (Rx-volume family): {sorted(NOWCAST_KPI_QUERY_FAMILIES)}"
            ),
        )

    # '' is the UI's All-Brands scope, but the migration-116 triangle SQL only
    # treats NULL as all-brands ('' would be a literal brand that never
    # matches — live symptom: reason=no_data disabling the nowcast toggle).
    rows = await fetch_nowcast_rows(kpi_id, brand=brand or None)
    result = estimate_completion_from_rows(rows)
    points = [
        KPINowcastPoint(
            metric_date=p.month.isoformat(),
            mature_value=p.mature_value,
            provisional_value=p.provisional_value,
            provisional=not p.is_mature,
            completion_factor=p.completion_factor,
            nowcast_value=p.nowcast_value,
            nowcast_ci_lower=p.nowcast_ci[0] if p.nowcast_ci else None,
            nowcast_ci_upper=p.nowcast_ci[1] if p.nowcast_ci else None,
        )
        for p in result.months
        # ISO dates compare lexicographically == chronologically.
        if (start_date is None or p.month.isoformat() >= start_date)
        and (end_date is None or p.month.isoformat() <= end_date)
    ]
    from src.kpi.measure_basis import registry_query_basis
    from src.kpi.nowcast.completion_factor import nowcast_query_id

    return KPINowcastHistoryResponse(
        measure_basis=await registry_query_basis(nowcast_query_id(kpi_id)),
        kpi_id=kpi_id,
        brand=brand or "",
        data_through=result.frontier.isoformat() if result.frontier else None,
        insufficient_maturity=result.insufficient_maturity,
        reason=result.reason,
        mature_months_used=len(result.mature_months),
        anchor_cap_month=(result.anchor_cap_month.isoformat() if result.anchor_cap_month else None),
        arrival_plane_coverage=result.arrival_plane_coverage,
        ci_level=result.ci_level,
        count=len(points),
        points=points,
    )


@router.post(
    "/calculate",
    response_model=KPIResultResponse,
    summary="Calculate single KPI",
    description="Calculate a single KPI with full context options",
    operation_id="calculate_kpi",
)
async def calculate_kpi(
    request: KPICalculationRequest,
    calculator: KPICalculator = Depends(get_kpi_calculator),
    user: Dict[str, Any] = Depends(require_auth),
) -> KPIResultResponse:
    """Calculate a single KPI with full context.

    Args:
        request: Calculation request
        calculator: KPI calculator instance

    Returns:
        Calculated KPI result
    """
    try:
        # Build context dict from request
        context: dict[str, Any] = {}
        if request.context:
            if request.context.brand:
                context["brand"] = request.context.brand
            if request.context.region:
                context["region"] = request.context.region
            if request.context.start_date:
                context["start_date"] = request.context.start_date
            if request.context.end_date:
                context["end_date"] = request.context.end_date
            if request.context.territory:
                context["territory"] = request.context.territory
            if request.context.segment:
                context["segment"] = request.context.segment
            if request.context.therapy_line:
                context["therapy_line"] = request.context.therapy_line
            if request.context.biologic:
                context["biologic"] = request.context.biologic
            if request.context.ige_tier:
                context["ige_tier"] = request.context.ige_tier
            context.update(request.context.extra)

        result = calculator.calculate(
            kpi_id=request.kpi_id,
            use_cache=request.use_cache,
            force_refresh=request.force_refresh,
            context=context,
        )

        if result.error and "not found" in result.error.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.error,
            )

        return _result_to_response(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to calculate KPI {request.kpi_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate KPI: {str(e)}",
        )


@router.post(
    "/batch",
    response_model=BatchKPICalculationResponse,
    summary="Batch calculate KPIs",
    description="Calculate multiple KPIs in a single request",
    operation_id="calculate_batch_kpis",
)
async def calculate_batch(
    request: BatchKPICalculationRequest,
    calculator: KPICalculator = Depends(get_kpi_calculator),
    user: Dict[str, Any] = Depends(require_auth),
) -> BatchKPICalculationResponse:
    """Calculate multiple KPIs in batch.

    Args:
        request: Batch calculation request
        calculator: KPI calculator instance

    Returns:
        Batch calculation results
    """
    try:
        # Build context
        context: dict[str, Any] = {}
        if request.context:
            if request.context.brand:
                context["brand"] = request.context.brand
            if request.context.region:
                context["region"] = request.context.region
            if request.context.start_date:
                context["start_date"] = request.context.start_date
            if request.context.end_date:
                context["end_date"] = request.context.end_date
            context.update(request.context.extra)

        # Parse workstream
        ws_enum = _workstream_from_string(request.workstream)

        # Calculate batch
        batch_result = calculator.calculate_batch(
            kpi_ids=request.kpi_ids,
            workstream=ws_enum,
            use_cache=request.use_cache,
            context=context,
        )

        return BatchKPICalculationResponse(
            workstream=request.workstream,
            results=[_result_to_response(r) for r in batch_result.results],
            calculated_at=batch_result.calculated_at,
            total_kpis=batch_result.total_kpis,
            successful=batch_result.successful,
            failed=batch_result.failed,
        )

    except Exception as e:
        logger.error(f"Batch KPI calculation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch calculation failed: {str(e)}",
        )


# =============================================================================
# CACHE MANAGEMENT ENDPOINTS
# =============================================================================


@router.post(
    "/invalidate",
    response_model=CacheInvalidationResponse,
    summary="Invalidate KPI cache",
    description="Invalidate cached KPI values",
    operation_id="invalidate_kpi_cache",
)
async def invalidate_cache(
    request: CacheInvalidationRequest,
    calculator: KPICalculator = Depends(get_kpi_calculator),
    user: Dict[str, Any] = Depends(require_admin),
) -> CacheInvalidationResponse:
    """Invalidate cached KPI values.

    Args:
        request: Cache invalidation request
        calculator: KPI calculator instance

    Returns:
        Invalidation result
    """
    try:
        if request.invalidate_all:
            count = calculator.invalidate_cache()
            return CacheInvalidationResponse(
                invalidated_count=count,
                message="All KPI cache entries invalidated",
            )

        if request.kpi_id:
            count = calculator.invalidate_cache(kpi_id=request.kpi_id)
            return CacheInvalidationResponse(
                invalidated_count=count,
                message=f"Cache invalidated for KPI: {request.kpi_id}",
            )

        if request.workstream:
            ws_enum = _workstream_from_string(request.workstream)
            if ws_enum is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid workstream: {request.workstream}",
                )
            count = calculator.invalidate_cache(workstream=ws_enum)
            return CacheInvalidationResponse(
                invalidated_count=count,
                message=f"Cache invalidated for workstream: {request.workstream}",
            )

        return CacheInvalidationResponse(
            invalidated_count=0,
            message="No invalidation criteria specified",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache invalidation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache invalidation failed: {str(e)}",
        )
