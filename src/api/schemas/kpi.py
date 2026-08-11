"""
KPI API Schemas

Pydantic schemas for KPI API request/response validation.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class KPICalculationContext(BaseModel):
    """Context for KPI calculation."""

    brand: str | None = Field(
        default=None,
        description="Filter by brand (remibrutinib, fabhalta, kisqali)",
    )
    region: str | None = Field(
        default=None,
        description=(
            "Filter by geographic region (e.g. northeast, south, midwest, west); "
            "matched case-insensitively against patient_journeys.geographic_region. "
            "When set, region-scoped KPI query variants (migration 077) are used."
        ),
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
        description=(
            "Filter by patient severity tier (low_severity, medium_severity, "
            "high_severity), matched case-insensitively against "
            "patient_journeys.segment_assignment. Mutually exclusive with "
            "region/therapy_line (the underlying RPC caps positional params at "
            "4). When set, severity-tier-scoped KPI query variants (migration "
            "105) are used."
        ),
    )
    therapy_line: str | None = Field(
        default=None,
        description=(
            "Filter by line of therapy ('0'-'3'), matched against "
            "patient_journeys.prior_therapy_lines. Mutually exclusive with "
            "region/segment (the underlying RPC caps positional params at 4). "
            "When set, line-of-therapy-scoped KPI query variants (migration "
            "105) are used."
        ),
    )
    biologic: str | None = Field(
        default=None,
        description=(
            "Filter by biologic status ('naive' or 'experienced'), from "
            "patient_journeys.biologic_experienced. AVAILABLE FOR REMIBRUTINIB "
            "ONLY (the column is 100% NULL for other brands by design) -- a "
            "biologic breakdown for any other brand fails closed rather than "
            "return a fabricated split. Mutually exclusive with "
            "region/segment/therapy_line/ige_tier (RPC 4-param cap). When set, "
            "biologic-status-scoped KPI query variants (migration 108) are used."
        ),
    )
    ige_tier: str | None = Field(
        default=None,
        description=(
            "Filter by IgE tertile ('low', 'medium', 'high') -- data-driven "
            "tertiles of patient_journeys.ige_level (NOT a clinical threshold). "
            "AVAILABLE FOR REMIBRUTINIB ONLY (100% NULL for other brands); a "
            "breakdown for any other brand fails closed. Mutually exclusive with "
            "region/segment/therapy_line/biologic (RPC 4-param cap). When set, "
            "IgE-tertile-scoped KPI query variants (migration 108) are used."
        ),
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context parameters",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "brand": "kisqali",
                "start_date": "2026-01-01T00:00:00Z",
                "end_date": "2026-01-31T23:59:59Z",
                "territory": "Northeast",
            }
        }
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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "kpi_id": "WS1-DQ-001",
                "use_cache": True,
                "force_refresh": False,
                "context": {"brand": "kisqali", "territory": "Northeast"},
            }
        }
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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "kpi_ids": ["WS1-DQ-001", "WS1-DQ-002", "WS1-MP-001"],
                "use_cache": True,
            }
        }
    )


class KPIResultResponse(BaseModel):
    """Response schema for a single KPI result."""

    kpi_id: str = Field(..., description="KPI identifier")
    value: float | None = Field(None, description="Calculated KPI value")
    status: str = Field(
        default="unknown",
        description=(
            "Status against thresholds; 'informational' = no target by design "
            "(volume/causal metrics), 'unknown' = could not evaluate"
        ),
        examples=["good", "warning", "critical", "informational", "unknown"],
    )
    calculated_at: datetime = Field(..., description="Calculation timestamp")
    cached: bool = Field(False, description="Whether result was from cache")
    cache_expires_at: datetime | None = Field(None, description="When cache entry expires")
    error: str | None = Field(None, description="Error message if calculation failed")
    data_source: str = Field(
        default="database",
        description=(
            "Provenance of the value: 'database' = real (synthetic-excluded) rows; "
            "'synthetic' = computed over synthetic-gold rows in "
            "E2I_KPI_INCLUDE_SYNTHETIC demo/review mode (the FE badges these so a "
            "synthetic figure is never read as real-world data)."
        ),
        examples=["database", "synthetic"],
    )

    # Causal analysis details
    causal_library_used: str | None = Field(None, description="Causal library used for calculation")
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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "kpi_id": "WS1-DQ-001",
                "value": 0.87,
                "status": "good",
                "calculated_at": "2026-02-06T12:00:00Z",
                "cached": False,
                "causal_library_used": "dowhy",
                "confidence_interval": [0.82, 0.92],
                "p_value": 0.003,
                "effect_size": 0.15,
            }
        }
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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "workstream": "ws1_data_quality",
                "calculated_at": "2026-02-06T12:00:00Z",
                "total_kpis": 5,
                "successful": 4,
                "failed": 1,
            }
        }
    )


class KPIThresholdResponse(BaseModel):
    """Response schema for KPI thresholds.

    Monotone mode uses target/warning/critical; band mode (#1117, e.g.
    WS1-MP-006 calibration slope) uses ideal/good_tolerance/warning_tolerance —
    status derives from abs(value - ideal), both directions away are worse.
    """

    target: float | None = Field(None, description="Target threshold value (monotone mode)")
    warning: float | None = Field(None, description="Warning threshold value (monotone mode)")
    critical: float | None = Field(None, description="Critical threshold value (monotone mode)")
    ideal: float | None = Field(
        None, description="Ideal value for deviation-from-ideal KPIs (band mode)"
    )
    good_tolerance: float | None = Field(
        None, description="GOOD when abs(value - ideal) <= this (band mode)"
    )
    warning_tolerance: float | None = Field(
        None, description="WARNING when abs(value - ideal) <= this; CRITICAL beyond (band mode)"
    )

    model_config = ConfigDict(
        json_schema_extra={"example": {"target": 0.90, "warning": 0.75, "critical": 0.60}}
    )


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
    threshold: KPIThresholdResponse | None = Field(None, description="Threshold configuration")
    unit: str | None = Field(None, description="Unit of measurement")
    value_format: str | None = Field(
        None,
        description="Display-format hint: 'percent' (value is a 0-1 ratio shown as %) or None",
    )
    frequency: str = Field("daily", description="Calculation frequency")
    primary_causal_library: str = Field("none", description="Primary causal library for this KPI")
    brand: str | None = Field(None, description="Brand filter if applicable")
    note: str | None = Field(None, description="Additional notes")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "WS1-DQ-001",
                "name": "Data Completeness Rate",
                "definition": "Percentage of required fields populated across patient records",
                "formula": "COUNT(non_null) / COUNT(*) * 100",
                "calculation_type": "direct",
                "workstream": "ws1_data_quality",
                "tables": ["patient_records"],
                "columns": ["hcp_id", "npi", "specialty"],
                "threshold": {"target": 0.95, "warning": 0.85, "critical": 0.70},
                "unit": "%",
                "frequency": "daily",
                "primary_causal_library": "none",
            }
        }
    )


class KPIListResponse(BaseModel):
    """Response schema for listing KPIs."""

    kpis: list[KPIMetadataResponse] = Field(
        default_factory=list, description="List of KPI metadata"
    )
    total: int = Field(..., description="Total number of KPIs")
    workstream: str | None = Field(None, description="Filtered workstream if any")
    causal_library: str | None = Field(None, description="Filtered causal library if any")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total": 2,
                "workstream": "ws1_data_quality",
            }
        }
    )


class WorkstreamInfo(BaseModel):
    """Information about a workstream."""

    id: str = Field(..., description="Workstream identifier")
    name: str = Field(..., description="Human-readable workstream name")
    kpi_count: int = Field(..., description="Number of KPIs in this workstream")
    description: str | None = Field(None, description="Workstream description")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "ws1_data_quality",
                "name": "WS1 Data Quality",
                "kpi_count": 8,
                "description": "Data quality and completeness metrics",
            }
        }
    )


class WorkstreamListResponse(BaseModel):
    """Response schema for listing workstreams."""

    workstreams: list[WorkstreamInfo] = Field(
        default_factory=list, description="List of workstreams"
    )
    total: int = Field(..., description="Total number of workstreams")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total": 6,
            }
        }
    )


class CacheInvalidationRequest(BaseModel):
    """Request schema for cache invalidation."""

    kpi_id: str | None = Field(None, description="Specific KPI ID to invalidate (optional)")
    workstream: str | None = Field(
        None, description="Invalidate all KPIs for this workstream (optional)"
    )
    invalidate_all: bool = Field(False, description="Invalidate all cached KPIs (use with caution)")

    model_config = ConfigDict(json_schema_extra={"example": {"kpi_id": "WS1-DQ-001"}})


class CacheInvalidationResponse(BaseModel):
    """Response schema for cache invalidation."""

    invalidated_count: int = Field(..., description="Number of cache entries invalidated")
    message: str = Field(..., description="Status message")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "invalidated_count": 3,
                "message": "Invalidated 3 cache entries for workstream ws1_data_quality",
            }
        }
    )


class KPIHealthResponse(BaseModel):
    """Response schema for KPI system health."""

    status: str = Field(
        ...,
        description="Overall health status",
        examples=["healthy", "degraded", "unhealthy"],
    )
    registry_loaded: bool = Field(..., description="Whether KPI registry is loaded")
    total_kpis: int = Field(..., description="Total KPIs in registry")
    cache_enabled: bool = Field(..., description="Whether caching is enabled")
    cache_size: int = Field(0, description="Current cache size")
    database_connected: bool = Field(..., description="Whether database is connected")
    workstreams_available: list[str] = Field(
        default_factory=list, description="Available workstreams"
    )
    last_calculation: datetime | None = Field(None, description="Timestamp of last calculation")
    error: str | None = Field(None, description="Error message if unhealthy")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "registry_loaded": True,
                "total_kpis": 48,
                "cache_enabled": True,
                "cache_size": 12,
                "database_connected": True,
                "workstreams_available": [
                    "ws1_data_quality",
                    "ws2_triggers",
                    "ws3_market_performance",
                ],
            }
        }
    )


# =============================================================================
# KPI HISTORY (time-series KPI-history view; migration 079 + history_backfill)
# =============================================================================


class KPIHistoryPoint(BaseModel):
    """One materialized monthly KPI value."""

    metric_date: str = Field(..., description="Month (YYYY-MM-DD, first of month)")
    value: float = Field(..., description="KPI value for that month")
    status: str | None = Field(
        None,
        description="good/warning/critical vs threshold; informational = no target by design",
    )


class KPIHistoryScopeEntry(BaseModel):
    """One (brand, region) scope of a KPI's history (migration 126 lattice)."""

    brand: str = Field("", description="'' = global / all brands")
    region: str = Field("", description="'' = all regions")
    points: int = Field(default=0, description="Points in this scope")
    first_date: str | None = Field(default=None, description="Earliest metric_date in scope")
    last_date: str | None = Field(default=None, description="Latest metric_date in scope")


class KPIHistoryCoverageEntry(BaseModel):
    """History coverage for one KPI: which scopes have a real series.

    ``brands``/``points``/``first_date``/``last_date`` describe the BRAND axis
    (region='' rows only — unchanged semantics from before the region axis);
    ``scopes`` is the full (brand, region) lattice, the source of truth for
    which region series exist per brand (#1536).
    """

    kpi_id: str
    brands: list[str] = Field(
        default_factory=list,
        description="Brand scopes with points; '' = global. Per-brand-only KPIs have no ''.",
    )
    points: int = Field(default=0, description="Total points across brand scopes (region='')")
    first_date: str | None = Field(
        default=None, description="Earliest metric_date across brand scopes"
    )
    last_date: str | None = Field(
        default=None, description="Latest metric_date across brand scopes"
    )
    scopes: list[KPIHistoryScopeEntry] = Field(
        default_factory=list,
        description="Full (brand, region) scope lattice, sorted by (brand, region)",
    )


class KPIHistoryCoverageResponse(BaseModel):
    """Coverage map for the whole registry — KPIs absent here have NO history."""

    coverage: list[KPIHistoryCoverageEntry] = Field(default_factory=list)
    total: int = Field(default=0, description="Number of KPIs with at least one point")


class KPIHistoryResponse(BaseModel):
    """Date-ordered KPI history for one KPI (empty when no real series exists)."""

    kpi_id: str
    brand: str = Field("", description="'' = global / all brands")
    region: str = Field("", description="'' = all regions")
    count: int = Field(..., description="Number of points")
    points: list[KPIHistoryPoint] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "kpi_id": "WS3-BI-010",
                "brand": "",
                "region": "",
                "count": 2,
                "points": [
                    {"metric_date": "2026-05-01", "value": 1.83, "status": "warning"},
                    {"metric_date": "2026-06-01", "value": 1.85, "status": "warning"},
                ],
            }
        }
    )


class KPIHistorySegmentSeries(BaseModel):
    """One axis bucket's monthly series (e.g. the high-severity tier)."""

    key: str = Field(..., description="Bucket key (e.g. 'high_severity', or '2' for LOT)")
    label: str = Field(..., description="Display label (e.g. 'High severity', '2 prior lines')")
    count: int = Field(..., description="Number of points")
    points: list[KPIHistoryPoint] = Field(default_factory=list)


class KPISegmentedHistoryResponse(BaseModel):
    """Per-axis-bucket monthly history for one KPI, computed live (migration 110).

    Unlike ``KPIHistoryResponse`` this is NOT read from the materialized
    ``kpi_history`` table (which has no patient-segment dimension) — it is
    recomputed from treatment_events via the vetted kpi_query registry, with
    identical calendar-month bucketing and partial-edge-month trimming, so
    the bucket series partition the headline series month by month.
    """

    kpi_id: str
    brand: str = Field("", description="'' = global / all brands")
    axis: str = Field(..., description="'segment' (severity tier) or 'therapy_line' (LOT)")
    data_through: str | None = Field(
        None, description="Latest prescription event date backing the series (frontier)"
    )
    count: int = Field(..., description="Number of series (buckets)")
    series: list[KPIHistorySegmentSeries] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "kpi_id": "WS3-BI-005",
                "brand": "Remibrutinib",
                "axis": "segment",
                "data_through": "2026-07-13",
                "count": 3,
                "series": [
                    {
                        "key": "low_severity",
                        "label": "Low severity",
                        "count": 2,
                        "points": [
                            {"metric_date": "2026-05-01", "value": 57.0},
                            {"metric_date": "2026-06-01", "value": 272.0},
                        ],
                    }
                ],
            }
        }
    )


class KPINowcastPoint(BaseModel):
    """One monthly point of the claims-lag nowcast overlay (backlog #45).

    DEDICATED model rather than new optional fields on ``KPIHistoryPoint``:
    Pydantic serializes every declared field, so extending the shared point
    model would inject new ``null``/``false`` keys into every existing
    ``/history`` and ``/history/segmented`` payload (and mutate their generated
    api.ts schema). A separate model keeps those consumers byte-untouched; the
    api.ts delta is purely additive.
    """

    metric_date: str = Field(..., description="Service month (YYYY-MM-DD, first of month)")
    mature_value: float = Field(
        ...,
        description=(
            "The base KPI value over ALL events (the eventual truth — available "
            "because the synthetic substrate is omniscient). Matches /history."
        ),
    )
    provisional_value: float = Field(
        ...,
        description="Events whose claim_available_date <= frontier (the as-of under-count)",
    )
    provisional: bool = Field(
        ...,
        description="True while the month's claims are still maturing (not fully arrived)",
    )
    completion_factor: float | None = Field(
        None,
        description=(
            "Estimated fraction of the month's claims arrived as of the frontier "
            "(empirical chain-ladder CF; None when the month is younger than the "
            "observed lag support)"
        ),
    )
    nowcast_value: float | None = Field(
        None, description="provisional_value / completion_factor (the grossed-up estimate)"
    )
    nowcast_ci_lower: float | None = Field(
        None, description="Bootstrap CI lower bound (provisional months only)"
    )
    nowcast_ci_upper: float | None = Field(
        None, description="Bootstrap CI upper bound (provisional months only)"
    )


class KPINowcastHistoryResponse(BaseModel):
    """Claims-lag provisional/nowcast monthly series for one Rx-volume KPI.

    Computed LIVE from the migration-116 lag-triangle registry queries
    (mirroring the migration-110 segmented-history pattern) — never from the
    materialized ``kpi_history`` table, whose values stay the mature figures.
    When the completion curve cannot be estimated honestly
    (``insufficient_maturity=True``: too few mature months, or the arrival
    plane is not populated yet), ``points`` is EMPTY and ``reason`` says why —
    never a fabricated fallback completion factor.
    """

    kpi_id: str
    brand: str = Field("", description="'' = global / all brands")
    data_through: str | None = Field(
        None, description="Prescription frontier (max event_date) backing the as-of view"
    )
    insufficient_maturity: bool = Field(
        ...,
        description="True when no honest completion curve could be estimated (see reason)",
    )
    reason: str | None = Field(
        None,
        description=(
            "Machine-readable cause when insufficient_maturity "
            "(no_data | arrival_plane_not_populated | arrival_plane_partial | "
            "insufficient_mature_months | no_arrived_claims)"
        ),
    )
    mature_months_used: int = Field(
        0, description="Mature service months backing the completion curve"
    )
    anchor_cap_month: str | None = Field(
        None,
        description=(
            "Frontier month excluded from estimation and output (the #853 anchor-cap pile-up month)"
        ),
    )
    arrival_plane_coverage: float | None = Field(
        None, description="Share of events carrying claim_available_date (1.0 = fully stamped)"
    )
    ci_level: float = Field(0.95, description="Nominal bootstrap CI level")
    count: int = Field(..., description="Number of points")
    points: list[KPINowcastPoint] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "kpi_id": "WS3-BI-005",
                "brand": "Remibrutinib",
                "data_through": "2026-07-21",
                "insufficient_maturity": False,
                "reason": None,
                "mature_months_used": 30,
                "anchor_cap_month": "2026-07-01",
                "arrival_plane_coverage": 1.0,
                "ci_level": 0.95,
                "count": 2,
                "points": [
                    {
                        "metric_date": "2026-05-01",
                        "mature_value": 1322.0,
                        "provisional_value": 1057.0,
                        "provisional": True,
                        "completion_factor": 0.8,
                        "nowcast_value": 1321.25,
                        "nowcast_ci_lower": 1274.0,
                        "nowcast_ci_upper": 1369.0,
                    }
                ],
            }
        }
    )
