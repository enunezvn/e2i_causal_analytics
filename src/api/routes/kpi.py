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
from src.api.schemas.kpi import (
    BatchKPICalculationRequest,
    BatchKPICalculationResponse,
    CacheInvalidationRequest,
    CacheInvalidationResponse,
    KPICalculationRequest,
    KPIHealthResponse,
    KPIListResponse,
    KPIMetadataResponse,
    KPIResultResponse,
    KPIThresholdResponse,
    WorkstreamInfo,
    WorkstreamListResponse,
)
from src.kpi.calculator import KPICalculator
from src.kpi.models import CausalLibrary, Workstream
from src.kpi.registry import get_registry

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/kpis",
    tags=["KPIs"],
    responses={
        404: {"description": "KPI not found"},
        500: {"description": "Internal server error"},
    },
)


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================


def get_kpi_calculator() -> KPICalculator:
    """Get KPI calculator instance.

    Returns:
        KPICalculator instance
    """
    # In production, this would be a singleton or use proper DI
    return KPICalculator(db_connection=get_supabase())


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

    return KPIResultResponse(
        kpi_id=result.kpi_id,
        value=result.value,
        status=result.status.value if hasattr(result.status, "value") else result.status,
        calculated_at=result.calculated_at,
        cached=result.cached,
        cache_expires_at=result.cache_expires_at,
        error=result.error,
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
# KPI METADATA ENDPOINTS
# =============================================================================


@router.get(
    "/{kpi_id}/metadata",
    response_model=KPIMetadataResponse,
    summary="Get KPI metadata",
    description="Get metadata and definition for a specific KPI",
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
)
async def get_kpi_value(
    kpi_id: str,
    use_cache: bool = Query(default=True, description="Use cached value if available"),
    force_refresh: bool = Query(default=False, description="Force recalculation"),
    brand: str | None = Query(default=None, description="Brand filter"),
    calculator: KPICalculator = Depends(get_kpi_calculator),
) -> KPIResultResponse:
    """Get the calculated value for a specific KPI.

    Args:
        kpi_id: KPI identifier
        use_cache: Whether to use cached results
        force_refresh: Force recalculation
        brand: Optional brand filter
        calculator: KPI calculator instance

    Returns:
        Calculated KPI result

    Raises:
        HTTPException: If KPI not found or calculation fails
    """
    try:
        context = {"brand": brand} if brand else {}

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


@router.post(
    "/calculate",
    response_model=KPIResultResponse,
    summary="Calculate single KPI",
    description="Calculate a single KPI with full context options",
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
            if request.context.start_date:
                context["start_date"] = request.context.start_date
            if request.context.end_date:
                context["end_date"] = request.context.end_date
            if request.context.territory:
                context["territory"] = request.context.territory
            if request.context.segment:
                context["segment"] = request.context.segment
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
