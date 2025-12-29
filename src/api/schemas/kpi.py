"""
KPI API Schemas

Pydantic schemas for KPI API request/response validation.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.kpi.models import CausalLibrary, KPIStatus, Workstream


class KPICalculationContext(BaseModel):
    """Context for KPI calculation."""

    brand: str | None = Field(
        default=None,
        description="Filter by brand (remibrutinib, fabhalta, kisqali)",
    )
    start_date: datetime | None = Field(
        default=None,
        description="Start date for time-based calculations",
    )
    end_date: datetime | None = Field(
        default=None,
        description="End date for time-based calculations",
    )
    territory: str | None = Field(
        default=None,
        description="Territory filter",
    )
    segment: str | None = Field(
        default=None,
        description="Customer segment filter",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context parameters",
    )


class KPICalculationRequest(BaseModel):
    """Request schema for calculating a single KPI."""

    kpi_id: str = Field(
        ...,
        description="KPI identifier (e.g., WS1-DQ-001)",
        examples=["WS1-DQ-001", "WS2-TR-005"],
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use cached results if available",
    )
    force_refresh: bool = Field(
        default=False,
        description="Force recalculation even if cached",
    )
    context: KPICalculationContext | None = Field(
        default=None,
        description="Calculation context (filters, date range, etc.)",
    )


class BatchKPICalculationRequest(BaseModel):
    """Request schema for batch KPI calculation."""

    kpi_ids: list[str] | None = Field(
        default=None,
        description="List of specific KPI IDs to calculate. If None, uses workstream.",
        examples=[["WS1-DQ-001", "WS1-DQ-002", "WS1-MP-001"]],
    )
    workstream: str | None = Field(
        default=None,
        description="Calculate all KPIs for this workstream",
        examples=["ws1_data_quality", "ws2_triggers"],
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use cached results if available",
    )
    context: KPICalculationContext | None = Field(
        default=None,
        description="Calculation context for all KPIs",
    )


class KPIResultResponse(BaseModel):
    """Response schema for a single KPI result."""

    kpi_id: str = Field(..., description="KPI identifier")
    value: float | None = Field(None, description="Calculated KPI value")
    status: str = Field(
        default="unknown",
        description="Status against thresholds",
        examples=["good", "warning", "critical", "unknown"],
    )
    calculated_at: datetime = Field(..., description="Calculation timestamp")
    cached: bool = Field(False, description="Whether result was from cache")
    cache_expires_at: datetime | None = Field(
        None, description="When cache entry expires"
    )
    error: str | None = Field(None, description="Error message if calculation failed")

    # Causal analysis details
    causal_library_used: str | None = Field(
        None, description="Causal library used for calculation"
    )
    confidence_interval: list[float] | None = Field(
        None,
        description="95% confidence interval [lower, upper]",
        examples=[[0.42, 0.58]],
    )
    p_value: float | None = Field(None, description="Statistical p-value")
    effect_size: float | None = Field(None, description="Effect size if applicable")

    # Additional metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional calculation metadata"
    )


class BatchKPICalculationResponse(BaseModel):
    """Response schema for batch KPI calculation."""

    workstream: str | None = Field(None, description="Workstream if specified")
    results: list[KPIResultResponse] = Field(
        default_factory=list, description="List of KPI results"
    )
    calculated_at: datetime = Field(..., description="Batch calculation timestamp")
    total_kpis: int = Field(..., description="Total number of KPIs requested")
    successful: int = Field(..., description="Number of successful calculations")
    failed: int = Field(..., description="Number of failed calculations")


class KPIThresholdResponse(BaseModel):
    """Response schema for KPI thresholds."""

    target: float | None = Field(None, description="Target threshold value")
    warning: float | None = Field(None, description="Warning threshold value")
    critical: float | None = Field(None, description="Critical threshold value")


class KPIMetadataResponse(BaseModel):
    """Response schema for KPI metadata."""

    id: str = Field(..., description="KPI identifier")
    name: str = Field(..., description="Human-readable KPI name")
    definition: str = Field(..., description="KPI definition/description")
    formula: str = Field(..., description="Calculation formula")
    calculation_type: str = Field(
        ...,
        description="direct or derived",
        examples=["direct", "derived"],
    )
    workstream: str = Field(..., description="Workstream this KPI belongs to")
    tables: list[str] = Field(default_factory=list, description="Source tables")
    columns: list[str] = Field(default_factory=list, description="Source columns")
    view: str | None = Field(None, description="Database view name if applicable")
    threshold: KPIThresholdResponse | None = Field(
        None, description="Threshold configuration"
    )
    unit: str | None = Field(None, description="Unit of measurement")
    frequency: str = Field("daily", description="Calculation frequency")
    primary_causal_library: str = Field(
        "none", description="Primary causal library for this KPI"
    )
    brand: str | None = Field(None, description="Brand filter if applicable")
    note: str | None = Field(None, description="Additional notes")


class KPIListResponse(BaseModel):
    """Response schema for listing KPIs."""

    kpis: list[KPIMetadataResponse] = Field(
        default_factory=list, description="List of KPI metadata"
    )
    total: int = Field(..., description="Total number of KPIs")
    workstream: str | None = Field(None, description="Filtered workstream if any")
    causal_library: str | None = Field(
        None, description="Filtered causal library if any"
    )


class WorkstreamInfo(BaseModel):
    """Information about a workstream."""

    id: str = Field(..., description="Workstream identifier")
    name: str = Field(..., description="Human-readable workstream name")
    kpi_count: int = Field(..., description="Number of KPIs in this workstream")
    description: str | None = Field(None, description="Workstream description")


class WorkstreamListResponse(BaseModel):
    """Response schema for listing workstreams."""

    workstreams: list[WorkstreamInfo] = Field(
        default_factory=list, description="List of workstreams"
    )
    total: int = Field(..., description="Total number of workstreams")


class CacheInvalidationRequest(BaseModel):
    """Request schema for cache invalidation."""

    kpi_id: str | None = Field(
        None, description="Specific KPI ID to invalidate (optional)"
    )
    workstream: str | None = Field(
        None, description="Invalidate all KPIs for this workstream (optional)"
    )
    invalidate_all: bool = Field(
        False, description="Invalidate all cached KPIs (use with caution)"
    )


class CacheInvalidationResponse(BaseModel):
    """Response schema for cache invalidation."""

    invalidated_count: int = Field(
        ..., description="Number of cache entries invalidated"
    )
    message: str = Field(..., description="Status message")


class KPIHealthResponse(BaseModel):
    """Response schema for KPI system health."""

    status: str = Field(
        ...,
        description="Overall health status",
        examples=["healthy", "degraded", "unhealthy"],
    )
    registry_loaded: bool = Field(
        ..., description="Whether KPI registry is loaded"
    )
    total_kpis: int = Field(..., description="Total KPIs in registry")
    cache_enabled: bool = Field(..., description="Whether caching is enabled")
    cache_size: int = Field(0, description="Current cache size")
    database_connected: bool = Field(
        ..., description="Whether database is connected"
    )
    workstreams_available: list[str] = Field(
        default_factory=list, description="Available workstreams"
    )
    last_calculation: datetime | None = Field(
        None, description="Timestamp of last calculation"
    )
    error: str | None = Field(None, description="Error message if unhealthy")
