"""
E2I Causal Analytics - Business Context Labels
==============================================

Provides business context (brand, segment, region) labels for:
- Model serving predictions
- Agent responses
- Opik/MLflow traces
- Prometheus metrics

This enables filtering and aggregation by business dimensions across
the observability stack.

E2I Business Dimensions:
- Brand: Remibrutinib, Fabhalta, Kisqali, All_Brands
- Region: northeast, south, midwest, west
- Segment: Used for customer/HCP segmentation (custom per analysis)

Usage:
    from src.mlops.business_context import (
        BusinessContext,
        get_context_from_request,
        context_to_labels,
    )

    # Create context
    ctx = BusinessContext(brand="Remibrutinib", region="northeast")

    # Add to Opik span
    span.set_attribute("business.brand", ctx.brand)
    span.set_attribute("business.region", ctx.region)

    # Convert to Prometheus labels
    labels = context_to_labels(ctx)

Author: E2I Causal Analytics Team
Version: 1.0.0 (Phase 4 - G24)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Business Dimension Enums
# =============================================================================


class E2IBrand(str, Enum):
    """E2I brand portfolio."""

    REMIBRUTINIB = "Remibrutinib"  # CSU
    FABHALTA = "Fabhalta"  # PNH
    KISQALI = "Kisqali"  # HR+/HER2- breast cancer
    ALL_BRANDS = "All_Brands"  # Cross-brand analysis


class E2IRegion(str, Enum):
    """E2I geographic regions."""

    NORTHEAST = "northeast"
    SOUTH = "south"
    MIDWEST = "midwest"
    WEST = "west"


class E2ISegmentType(str, Enum):
    """Common segmentation types."""

    HCP_SPECIALTY = "hcp_specialty"  # By HCP specialty
    HCP_TIER = "hcp_tier"  # By HCP tier (high/medium/low value)
    PATIENT_STAGE = "patient_stage"  # By patient journey stage
    PAYER_TYPE = "payer_type"  # By payer category
    ACCOUNT_TYPE = "account_type"  # By account type (hospital, clinic, etc.)
    CUSTOM = "custom"  # Custom segmentation


# String lists for validation (backwards compatible)
VALID_BRANDS = [b.value for b in E2IBrand]
VALID_REGIONS = [r.value for r in E2IRegion]


# =============================================================================
# Business Context Dataclass
# =============================================================================


@dataclass
class BusinessContext:
    """Business context for observability labeling.

    Attributes:
        brand: E2I brand (Remibrutinib, Fabhalta, Kisqali, All_Brands)
        region: Geographic region (northeast, south, midwest, west)
        segment: Customer/HCP segment identifier
        segment_type: Type of segmentation used
        custom_labels: Additional custom labels
        source: Where the context was derived from
        timestamp: When context was captured
    """

    brand: Optional[str] = None
    region: Optional[str] = None
    segment: Optional[str] = None
    segment_type: Optional[str] = None
    custom_labels: Dict[str, str] = field(default_factory=dict)
    source: str = "unknown"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate business context values."""
        if self.brand and self.brand not in VALID_BRANDS:
            logger.warning(f"Unknown brand '{self.brand}', expected one of {VALID_BRANDS}")

        if self.region and self.region not in VALID_REGIONS:
            logger.warning(f"Unknown region '{self.region}', expected one of {VALID_REGIONS}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "brand": self.brand,
            "region": self.region,
            "segment": self.segment,
            "segment_type": self.segment_type,
            "source": self.source,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
        if self.custom_labels:
            result["custom_labels"] = self.custom_labels
        return {k: v for k, v in result.items() if v is not None}

    def to_labels(self) -> Dict[str, str]:
        """Convert to label dict for Prometheus/metrics.

        Returns only non-None string values suitable for metric labels.
        """
        labels = {}
        if self.brand:
            labels["brand"] = self.brand
        if self.region:
            labels["region"] = self.region
        if self.segment:
            labels["segment"] = self.segment
        if self.segment_type:
            labels["segment_type"] = self.segment_type
        return labels

    def to_span_attributes(self) -> Dict[str, str]:
        """Convert to Opik/OpenTelemetry span attributes.

        Uses 'business.' prefix for semantic clarity.
        """
        attrs = {}
        if self.brand:
            attrs["business.brand"] = self.brand
        if self.region:
            attrs["business.region"] = self.region
        if self.segment:
            attrs["business.segment"] = self.segment
        if self.segment_type:
            attrs["business.segment_type"] = self.segment_type
        if self.source:
            attrs["business.context_source"] = self.source
        for key, value in self.custom_labels.items():
            attrs[f"business.custom.{key}"] = value
        return attrs

    def to_mlflow_tags(self) -> Dict[str, str]:
        """Convert to MLflow experiment tags.

        Uses 'e2i.' prefix for E2I-specific tags.
        """
        tags = {}
        if self.brand:
            tags["e2i.brand"] = self.brand
        if self.region:
            tags["e2i.region"] = self.region
        if self.segment:
            tags["e2i.segment"] = self.segment
        if self.segment_type:
            tags["e2i.segment_type"] = self.segment_type
        return tags

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BusinessContext":
        """Create from dictionary."""
        return cls(
            brand=data.get("brand"),
            region=data.get("region"),
            segment=data.get("segment"),
            segment_type=data.get("segment_type"),
            custom_labels=data.get("custom_labels", {}),
            source=data.get("source", "dict"),
        )


# =============================================================================
# Pydantic Models for API
# =============================================================================


class BusinessContextModel(BaseModel):
    """Pydantic model for API requests/responses."""

    brand: Optional[str] = Field(
        default=None,
        description="E2I brand (Remibrutinib, Fabhalta, Kisqali, All_Brands)",
    )
    region: Optional[str] = Field(
        default=None,
        description="Geographic region (northeast, south, midwest, west)",
    )
    segment: Optional[str] = Field(
        default=None,
        description="Customer/HCP segment identifier",
    )
    segment_type: Optional[str] = Field(
        default=None,
        description="Type of segmentation (hcp_specialty, hcp_tier, patient_stage, etc.)",
    )
    custom_labels: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional custom labels",
    )

    @field_validator("brand")
    @classmethod
    def validate_brand(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_BRANDS:
            logger.warning(f"Unknown brand '{v}', expected one of {VALID_BRANDS}")
        return v

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_REGIONS:
            logger.warning(f"Unknown region '{v}', expected one of {VALID_REGIONS}")
        return v

    def to_context(self) -> BusinessContext:
        """Convert to BusinessContext dataclass."""
        return BusinessContext(
            brand=self.brand,
            region=self.region,
            segment=self.segment,
            segment_type=self.segment_type,
            custom_labels=self.custom_labels or {},
            source="api_request",
        )


# =============================================================================
# Context Extraction Functions
# =============================================================================


def get_context_from_request(
    request_data: Dict[str, Any],
    default_brand: Optional[str] = None,
    default_region: Optional[str] = None,
) -> BusinessContext:
    """Extract business context from an API request.

    Looks for context in these locations (in order):
    1. request_data["business_context"] - explicit context object
    2. request_data["brand"], ["region"], etc. - top-level fields
    3. request_data["metadata"]["brand"], etc. - nested in metadata
    4. Defaults provided as arguments

    Args:
        request_data: Request payload dictionary
        default_brand: Default brand if not found
        default_region: Default region if not found

    Returns:
        BusinessContext with extracted or default values
    """
    # Try explicit business_context object first
    if "business_context" in request_data:
        ctx_data = request_data["business_context"]
        if isinstance(ctx_data, dict):
            return BusinessContext.from_dict(ctx_data)

    # Try top-level fields
    brand = request_data.get("brand", default_brand)
    region = request_data.get("region", default_region)
    segment = request_data.get("segment")
    segment_type = request_data.get("segment_type")

    # Try nested in metadata
    metadata = request_data.get("metadata", {})
    if isinstance(metadata, dict):
        brand = brand or metadata.get("brand")
        region = region or metadata.get("region")
        segment = segment or metadata.get("segment")
        segment_type = segment_type or metadata.get("segment_type")

    return BusinessContext(
        brand=brand,
        region=region,
        segment=segment,
        segment_type=segment_type,
        source="request",
    )


def get_context_from_dataframe(
    df: Any,  # pandas DataFrame
    brand_col: str = "brand",
    region_col: str = "region",
) -> BusinessContext:
    """Extract business context from a DataFrame.

    Determines context from the most common values in the DataFrame.

    Args:
        df: pandas DataFrame with business columns
        brand_col: Column name for brand
        region_col: Column name for region

    Returns:
        BusinessContext with most common values
    """
    brand = None
    region = None

    if hasattr(df, "columns"):
        if brand_col in df.columns:
            brand_values = df[brand_col].dropna()
            if len(brand_values) > 0:
                brand = brand_values.mode().iloc[0] if len(brand_values.mode()) > 0 else None

        if region_col in df.columns:
            region_values = df[region_col].dropna()
            if len(region_values) > 0:
                region = region_values.mode().iloc[0] if len(region_values.mode()) > 0 else None

    return BusinessContext(
        brand=brand,
        region=region,
        source="dataframe",
    )


def merge_contexts(*contexts: BusinessContext) -> BusinessContext:
    """Merge multiple contexts, later values override earlier.

    Args:
        *contexts: BusinessContext objects to merge

    Returns:
        Merged BusinessContext
    """
    result = BusinessContext(source="merged")

    for ctx in contexts:
        if ctx.brand:
            result.brand = ctx.brand
        if ctx.region:
            result.region = ctx.region
        if ctx.segment:
            result.segment = ctx.segment
        if ctx.segment_type:
            result.segment_type = ctx.segment_type
        if ctx.custom_labels:
            result.custom_labels.update(ctx.custom_labels)

    return result


# =============================================================================
# Context Propagation Helpers
# =============================================================================


def apply_context_to_span(span_context: Any, business_context: BusinessContext) -> None:
    """Apply business context to an Opik SpanContext.

    Args:
        span_context: SpanContext object (from opik_connector)
        business_context: Business context to apply
    """
    for key, value in business_context.to_span_attributes().items():
        if hasattr(span_context, "set_attribute"):
            span_context.set_attribute(key, value)
        elif hasattr(span_context, "metadata"):
            span_context.metadata[key] = value


def apply_context_to_mlflow(
    run: Any,  # mlflow.ActiveRun
    business_context: BusinessContext,
) -> None:
    """Apply business context as MLflow tags.

    Args:
        run: MLflow active run object
        business_context: Business context to apply
    """
    try:
        import mlflow

        for key, value in business_context.to_mlflow_tags().items():
            mlflow.set_tag(key, value)
    except ImportError:
        logger.debug("MLflow not available, skipping context tags")
    except Exception as e:
        logger.warning(f"Failed to set MLflow tags: {e}")


def context_to_labels(context: BusinessContext) -> Dict[str, str]:
    """Convert business context to metric labels.

    Convenience function that calls context.to_labels().

    Args:
        context: Business context

    Returns:
        Label dictionary for Prometheus metrics
    """
    return context.to_labels()


# =============================================================================
# Response Enrichment
# =============================================================================


def enrich_response_with_context(
    response: Dict[str, Any],
    context: BusinessContext,
    include_timestamp: bool = True,
) -> Dict[str, Any]:
    """Add business context to an API response.

    Args:
        response: Response dictionary to enrich
        context: Business context to add
        include_timestamp: Include context capture timestamp

    Returns:
        Response with business_context field added
    """
    ctx_dict = context.to_dict()
    if not include_timestamp:
        ctx_dict.pop("timestamp", None)

    response["business_context"] = ctx_dict
    return response


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "E2IBrand",
    "E2IRegion",
    "E2ISegmentType",
    # Constants
    "VALID_BRANDS",
    "VALID_REGIONS",
    # Classes
    "BusinessContext",
    "BusinessContextModel",
    # Extraction
    "get_context_from_request",
    "get_context_from_dataframe",
    "merge_contexts",
    # Propagation
    "apply_context_to_span",
    "apply_context_to_mlflow",
    "context_to_labels",
    # Response
    "enrich_response_with_context",
]
