"""
KPI Data Models

Pydantic models for KPI results, metadata, and thresholds.
"""

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    """Status of KPI against thresholds.

    INFORMATIONAL: the KPI carries no target BY DESIGN (volume metrics
    tracked for trend/context; causal effect sizes that carry CIs/p-values
    instead of fixed thresholds — see docs/data/06-KPI-REFERENCE.md,
    "Volume and Causal Metrics (No Thresholds)"). UNKNOWN is reserved for
    genuine could-not-evaluate (no value / calculation error).
    """

    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    INFORMATIONAL = "informational"


class KPIThreshold(BaseModel):
    """Threshold configuration for a KPI.

    Two mutually exclusive modes:

    - **Monotone** (``target``/``warning``/``critical``): the value is
      compared directionally (higher- or lower-is-better).
    - **Band** (``ideal``/``good_tolerance``/``warning_tolerance``, #1117):
      the KPI is a deviation-from-ideal metric — e.g. WS1-MP-006 calibration
      slope, where ideal is exactly 1.0 and BOTH directions away are worse.
      Status derives from ``abs(value - ideal)``, never from direction.
    """

    target: float | None = None
    warning: float | None = None
    critical: float | None = None

    # Band mode (#1117): ideal value with symmetric tolerance bands.
    ideal: float | None = None
    good_tolerance: float | None = None
    warning_tolerance: float | None = None

    @model_validator(mode="after")
    def _validate_band_mode(self) -> "KPIThreshold":
        """Fail loudly on malformed band configs instead of mis-evaluating."""
        if self.ideal is not None:
            if self.target is not None:
                raise ValueError(
                    "band mode (ideal) and monotone mode (target) are mutually exclusive"
                )
            if self.good_tolerance is None:
                raise ValueError("band mode requires good_tolerance alongside ideal")
        elif self.good_tolerance is not None or self.warning_tolerance is not None:
            raise ValueError("good_tolerance/warning_tolerance require ideal (band mode)")
        if self.good_tolerance is not None and self.good_tolerance < 0:
            raise ValueError("good_tolerance must be >= 0")
        if self.warning_tolerance is not None:
            if self.warning_tolerance < 0:
                raise ValueError("warning_tolerance must be >= 0")
            if self.good_tolerance is not None and self.warning_tolerance < self.good_tolerance:
                raise ValueError("warning_tolerance must be >= good_tolerance")
        return self

    def evaluate(self, value: float | None, lower_is_better: bool = False) -> KPIStatus:
        """Evaluate a value against thresholds.

        For band mode (``ideal`` set):
            - abs(value - ideal) <= good_tolerance: GOOD
            - abs(value - ideal) <= warning_tolerance: WARNING
            - beyond warning_tolerance: CRITICAL
              (warning_tolerance omitted -> WARNING, mirroring the monotone
              missing-outer-bound behavior)
            ``lower_is_better`` is ignored: the band is direction-symmetric.

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
            lower_is_better: If True, lower values are better (e.g., error
                rates). Ignored in band mode.

        Returns:
            KPIStatus indicating the health of this KPI
        """
        if value is None:
            return KPIStatus.UNKNOWN

        if self.ideal is not None:
            # Band mode (#1117): deviation-from-ideal, direction-symmetric.
            # The subtraction introduces float error at exact boundaries
            # (abs(1.05 - 1.0) > 0.05), so "within tolerance" is <= with an
            # isclose guard to keep the documented inclusive semantics.
            deviation = abs(value - self.ideal)

            def _within(tolerance: float) -> bool:
                return deviation <= tolerance or math.isclose(
                    deviation, tolerance, rel_tol=1e-9, abs_tol=1e-12
                )

            # good_tolerance is guaranteed non-None by the model validator;
            # the guard also narrows the Optional for the type checker.
            if self.good_tolerance is not None and _within(self.good_tolerance):
                return KPIStatus.GOOD
            if self.warning_tolerance is None or _within(self.warning_tolerance):
                return KPIStatus.WARNING
            return KPIStatus.CRITICAL

        if self.target is None:
            # No target on the threshold = no-target-by-design KPI, not an
            # evaluation failure.
            return KPIStatus.INFORMATIONAL

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
    # Display-format hint for value-rendering surfaces: 'percent' => the value is
    # a 0-1 ratio to be shown as NN.N% (×100); None/other => render as-is + `unit`.
    value_format: str | None = None
    frequency: str = "daily"
    primary_causal_library: CausalLibrary = CausalLibrary.NONE
    secondary_causal_library: CausalLibrary | None = None
    brand: str | None = None
    note: str | None = None
    windowable: str = "not_applicable"  # "clean" | "needs_care" | "not_applicable"
    window: dict[str, Any] | None = None  # {column, legs?, look_forward_days?} for windowable KPIs


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

    # Window provenance (spec 2026-06-20). window_status:
    #   "default"        -> no window requested; engine's fixed window used
    #   "applied"        -> requested window honored
    #   "not_applicable" -> KPI has no claims time-dimension; window ignored honestly
    window_requested: dict[str, Any] | None = None
    window_applied: dict[str, Any] | None = None
    window_status: str = Field(default="default", description="default | applied | not_applicable")

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
