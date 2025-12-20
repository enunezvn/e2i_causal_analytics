"""
Feature Store Data Models

Pydantic models for feature store entities.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class FeatureValueType(str, Enum):
    """Supported feature value types."""

    INT64 = "int64"
    FLOAT64 = "float64"
    STRING = "string"
    BOOL = "bool"
    TIMESTAMP = "timestamp"
    ARRAY_INT64 = "array_int64"
    ARRAY_FLOAT64 = "array_float64"
    ARRAY_STRING = "array_string"


class FreshnessStatus(str, Enum):
    """Feature freshness status."""

    FRESH = "fresh"  # Within SLA
    STALE = "stale"  # Outside SLA but usable
    EXPIRED = "expired"  # Too old, should not be used


class FeatureGroup(BaseModel):
    """Feature group definition."""

    id: Optional[UUID] = None
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    owner: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    source_table: Optional[str] = None
    source_query: Optional[str] = None
    expected_update_frequency_hours: int = Field(default=24, gt=0)
    max_age_hours: int = Field(default=168, gt=0)
    schema_version: str = "1.0.0"
    mlflow_experiment_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class Feature(BaseModel):
    """Individual feature definition."""

    id: Optional[UUID] = None
    feature_group_id: UUID
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    value_type: FeatureValueType
    entity_keys: List[str] = Field(..., min_length=1)
    computation_query: Optional[str] = None
    dependencies: List[UUID] = Field(default_factory=list)
    statistics: Dict[str, Any] = Field(default_factory=dict)
    drift_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    owner: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    version: str = "1.0.0"
    mlflow_run_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

    @field_validator("entity_keys")
    @classmethod
    def validate_entity_keys(cls, v: List[str]) -> List[str]:
        """Ensure entity_keys is not empty."""
        if not v:
            raise ValueError("entity_keys must contain at least one key")
        return v


class FeatureValue(BaseModel):
    """Time-series feature value."""

    id: Optional[UUID] = None
    feature_id: UUID
    entity_values: Dict[str, Any] = Field(..., description="Entity key-value pairs")
    value: Any = Field(..., description="Feature value (any JSON-serializable type)")
    event_timestamp: datetime
    created_timestamp: Optional[datetime] = None
    freshness_status: FreshnessStatus = FreshnessStatus.FRESH
    source_job_id: Optional[str] = None
    version: int = 1

    class Config:
        from_attributes = True

    @field_validator("entity_values")
    @classmethod
    def validate_entity_values(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure entity_values is not empty."""
        if not v:
            raise ValueError("entity_values must contain at least one key-value pair")
        return v


class EntityFeatures(BaseModel):
    """Collection of features for a specific entity."""

    entity_values: Dict[str, Any]
    features: Dict[str, Any] = Field(
        default_factory=dict, description="Feature name -> value mapping"
    )
    metadata: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Feature name -> metadata (timestamp, freshness, etc.)",
    )
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary (entity + features)."""
        return {**self.entity_values, **self.features}


class FeatureStatistics(BaseModel):
    """Feature statistics for monitoring."""

    feature_id: UUID
    feature_name: str
    feature_group: str
    count: int = 0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    percentiles: Dict[str, float] = Field(default_factory=dict)
    null_count: int = 0
    unique_count: Optional[int] = None
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
