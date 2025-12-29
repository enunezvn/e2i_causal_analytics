"""
KPI Data Models

Pydantic models for KPI results, metadata, and thresholds.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CausalLibrary(str, Enum):
    """Causal inference libraries available for KPI calculation."""

    DOWHY = "dowhy"
    ECONML = "econml"
    CAUSALML = "causalml"
    NETWORKX = "networkx"
    NONE = "none"


class Workstream(str, Enum):
    """KPI workstreams."""

    WS1_DATA_QUALITY = "ws1_data_quality"
    WS1_MODEL_PERFORMANCE = "ws1_model_performance"
    WS2_TRIGGERS = "ws2_triggers"
    WS3_BUSINESS = "ws3_business"
    BRAND_SPECIFIC = "brand_specific"
    CAUSAL_METRICS = "causal_metrics"


class CalculationType(str, Enum):
    """How the KPI is calculated."""

    DIRECT = "direct"  # Direct from database view/column
    DERIVED = "derived"  # Requires computation from multiple sources


class KPIStatus(str, Enum):
    """Status of KPI against thresholds."""

    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class KPIThreshold(BaseModel):
    """Threshold configuration for a KPI."""

    target: float | None = None
    warning: float | None = None
    critical: float | None = None

    def evaluate(self, value: float | None, lower_is_better: bool = False) -> KPIStatus:
        """Evaluate a value against thresholds.

        For higher-is-better (default):
            - value >= target: GOOD
            - critical <= value < target: WARNING
            - value < critical: CRITICAL

        For lower-is-better:
            - value <= target: GOOD
            - target < value <= warning: WARNING
            - value > warning: CRITICAL

        Args:
            value: The KPI value to evaluate
            lower_is_better: If True, lower values are better (e.g., error rates)

        Returns:
            KPIStatus indicating the health of this KPI
        """
        if value is None:
            return KPIStatus.UNKNOWN

        if self.target is None:
            return KPIStatus.UNKNOWN

        if lower_is_better:
            # Lower values are better (e.g., error rates, Brier score)
            # target < warning < critical (all define "bad" thresholds going up)
            if value <= self.target:
                return KPIStatus.GOOD
            elif self.warning is not None and value > self.warning:
                return KPIStatus.CRITICAL
            else:
                return KPIStatus.WARNING
        else:
            # Higher values are better (e.g., accuracy, coverage)
            # critical < warning < target (all define "good" thresholds going up)
            if value >= self.target:
                return KPIStatus.GOOD
            elif self.critical is not None and value < self.critical:
                return KPIStatus.CRITICAL
            else:
                return KPIStatus.WARNING


class KPIMetadata(BaseModel):
    """Metadata for a KPI definition."""

    id: str = Field(..., description="Unique KPI identifier (e.g., WS1-DQ-001)")
    name: str = Field(..., description="Human-readable KPI name")
    definition: str = Field(..., description="KPI definition/description")
    formula: str = Field(..., description="Calculation formula")
    calculation_type: CalculationType
    workstream: Workstream
    tables: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    view: str | None = None
    threshold: KPIThreshold | None = None
    unit: str | None = None
    frequency: str = "daily"
    primary_causal_library: CausalLibrary = CausalLibrary.NONE
    secondary_causal_library: CausalLibrary | None = None
    brand: str | None = None
    note: str | None = None


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class KPIResult(BaseModel):
    """Result of a KPI calculation."""

    model_config = ConfigDict(use_enum_values=True)

    kpi_id: str = Field(..., description="KPI identifier")
    value: float | None = Field(None, description="Calculated KPI value")
    status: KPIStatus = Field(KPIStatus.UNKNOWN, description="Status against thresholds")
    calculated_at: datetime = Field(default_factory=_utc_now)
    cached: bool = Field(False, description="Whether result was from cache")
    cache_expires_at: datetime | None = None
    error: str | None = Field(None, description="Error message if calculation failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context")

    # Causal analysis details (if applicable)
    causal_library_used: CausalLibrary | None = None
    confidence_interval: tuple[float, float] | None = None
    p_value: float | None = None
    effect_size: float | None = None


class KPIBatchResult(BaseModel):
    """Result of a batch KPI calculation."""

    workstream: Workstream | None = None
    results: list[KPIResult] = Field(default_factory=list)
    calculated_at: datetime = Field(default_factory=_utc_now)
    total_kpis: int = 0
    successful: int = 0
    failed: int = 0

    def add_result(self, result: KPIResult) -> None:
        """Add a result to the batch."""
        self.results.append(result)
        self.total_kpis += 1
        if result.error is None:
            self.successful += 1
        else:
            self.failed += 1
