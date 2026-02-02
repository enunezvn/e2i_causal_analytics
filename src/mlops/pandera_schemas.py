"""
E2I Causal Analytics - Pandera Schema Definitions
==================================================

Fast DataFrame schema validation using Pandera for E2I data sources.
Runs BEFORE Great Expectations for fast-fail on schema issues.

Components:
-----------
- 6 DataFrameModel schemas for core E2I data sources
- PANDERA_SCHEMA_REGISTRY for schema lookup by data source name
- E2I business constraints (brands, regions, confidence ranges)

Integration:
------------
- Used by data_preparer agent's run_schema_validation node
- Complements Great Expectations (business rules) validation
- Typical execution time: ~10ms

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import logging
from typing import Any, Dict, Optional, Type

import pandas as pd
import pandera.pandas as pa
from pandera import DataFrameModel, Field
from pandera.typing import Series

logger = logging.getLogger(__name__)

# =============================================================================
# E2I Business Constants
# =============================================================================

# Valid brands (from brand_type ENUM)
E2I_BRANDS = ["Remibrutinib", "Fabhalta", "Kisqali", "All_Brands"]

# Valid regions (from region_type ENUM)
E2I_REGIONS = ["northeast", "south", "midwest", "west"]

# Valid prediction types (from prediction_type ENUM)
E2I_PREDICTION_TYPES = ["trigger", "propensity", "risk", "churn"]

# Valid priority types (from priority_type ENUM)
E2I_PRIORITY_TYPES = ["critical", "high", "medium", "low"]

# Valid journey stages (from journey_stage_type ENUM)
E2I_JOURNEY_STAGES = ["diagnosis", "initial_treatment", "treatment_optimization", "maintenance"]

# Valid journey statuses (from journey_status_type ENUM)
E2I_JOURNEY_STATUSES = ["active", "stable", "transitioning", "completed"]

# Valid agent tiers (from agent_tier_type ENUM)
E2I_AGENT_TIERS = ["coordination", "causal_analytics", "monitoring", "ml_predictions", "learning"]


# =============================================================================
# Schema 1: Business Metrics
# =============================================================================


class BusinessMetricsSchema(DataFrameModel):
    """Schema for business_metrics table data.

    Fields validated:
    - metric_id: Unique identifier (string)
    - metric_date: Date of metric (date/datetime)
    - brand: One of E2I brands (nullable)
    - region: One of E2I regions (nullable)
    - value: Numeric metric value (nullable)
    - target: Target value (nullable)
    """

    metric_id: Series[str] = Field(nullable=False, unique=True)
    metric_date: Series[pd.Timestamp] = Field(nullable=False, coerce=True)
    metric_type: Optional[Series[str]] = Field(nullable=True)
    metric_name: Optional[Series[str]] = Field(nullable=True)
    brand: Optional[Series[str]] = Field(nullable=True, isin=E2I_BRANDS + [None])
    region: Optional[Series[str]] = Field(nullable=True, isin=E2I_REGIONS + [None])
    value: Optional[Series[float]] = Field(nullable=True)
    target: Optional[Series[float]] = Field(nullable=True)
    achievement_rate: Optional[Series[float]] = Field(nullable=True, ge=0.0)

    class Config:
        name = "business_metrics"
        strict = False  # Allow extra columns
        coerce = True


# =============================================================================
# Schema 2: Predictions
# =============================================================================


class PredictionsSchema(DataFrameModel):
    """Schema for ml_predictions table data.

    Critical validations:
    - prediction_id: Unique identifier
    - confidence_score: Must be 0.0-1.0
    - prediction_value: Must be 0.0-1.0 (probability)
    """

    prediction_id: Series[str] = Field(nullable=False, unique=True)
    model_version: Optional[Series[str]] = Field(nullable=True)
    model_type: Optional[Series[str]] = Field(nullable=True)
    prediction_type: Optional[Series[str]] = Field(
        nullable=True, isin=E2I_PREDICTION_TYPES + [None]
    )
    prediction_value: Optional[Series[float]] = Field(
        nullable=True, ge=0.0, le=1.0, description="Prediction probability must be between 0 and 1"
    )
    confidence_score: Optional[Series[float]] = Field(
        nullable=True, ge=0.0, le=1.0, description="Confidence score must be between 0 and 1"
    )
    patient_id: Optional[Series[str]] = Field(nullable=True)
    hcp_id: Optional[Series[str]] = Field(nullable=True)

    class Config:
        name = "predictions"
        strict = False
        coerce = True


# =============================================================================
# Schema 3: Triggers
# =============================================================================


class TriggersSchema(DataFrameModel):
    """Schema for triggers table data.

    Critical validations:
    - trigger_id: Unique identifier
    - priority: One of E2I priority types
    - confidence_score: Must be 0.0-1.0
    """

    trigger_id: Series[str] = Field(nullable=False, unique=True)
    patient_id: Series[str] = Field(nullable=False)
    trigger_timestamp: Optional[Series[pd.Timestamp]] = Field(nullable=True, coerce=True)
    trigger_type: Optional[Series[str]] = Field(nullable=True)
    priority: Optional[Series[str]] = Field(nullable=True, isin=E2I_PRIORITY_TYPES + [None])
    confidence_score: Optional[Series[float]] = Field(
        nullable=True, ge=0.0, le=1.0, description="Confidence score must be between 0 and 1"
    )
    lead_time_days: Optional[Series[int]] = Field(nullable=True, ge=0)
    hcp_id: Optional[Series[str]] = Field(nullable=True)

    class Config:
        name = "triggers"
        strict = False
        coerce = True


# =============================================================================
# Schema 4: Patient Journeys
# =============================================================================


class PatientJourneysSchema(DataFrameModel):
    """Schema for patient_journeys table data.

    Critical validations:
    - patient_journey_id: Unique identifier
    - patient_id: Required patient reference
    - brand: One of E2I brands
    - geographic_region: One of E2I regions
    """

    patient_journey_id: Series[str] = Field(nullable=False, unique=True)
    patient_id: Series[str] = Field(nullable=False)
    journey_start_date: Optional[Series[pd.Timestamp]] = Field(nullable=True, coerce=True)
    journey_end_date: Optional[Series[pd.Timestamp]] = Field(nullable=True, coerce=True)
    current_stage: Optional[Series[str]] = Field(nullable=True, isin=E2I_JOURNEY_STAGES + [None])
    journey_status: Optional[Series[str]] = Field(nullable=True, isin=E2I_JOURNEY_STATUSES + [None])
    brand: Optional[Series[str]] = Field(nullable=True, isin=E2I_BRANDS + [None])
    geographic_region: Optional[Series[str]] = Field(nullable=True, isin=E2I_REGIONS + [None])
    age_group: Optional[Series[str]] = Field(nullable=True)
    gender: Optional[Series[str]] = Field(nullable=True)
    source_match_confidence: Optional[Series[float]] = Field(nullable=True, ge=0.0, le=1.0)

    class Config:
        name = "patient_journeys"
        strict = False
        coerce = True


# =============================================================================
# Schema 5: Causal Paths
# =============================================================================


class CausalPathsSchema(DataFrameModel):
    """Schema for causal_paths table data.

    Critical validations:
    - path_id: Unique identifier
    - confidence_level: Must be 0.0-1.0
    - causal_effect_size: Must be -1.0 to 1.0 (effect strength)
    """

    path_id: Series[str] = Field(nullable=False, unique=True)
    discovery_date: Optional[Series[pd.Timestamp]] = Field(nullable=True, coerce=True)
    source_node: Optional[Series[str]] = Field(nullable=True)
    target_node: Optional[Series[str]] = Field(nullable=True)
    path_length: Optional[Series[int]] = Field(nullable=True, ge=1)
    causal_effect_size: Optional[Series[float]] = Field(
        nullable=True, ge=-1.0, le=1.0, description="Causal effect size must be between -1 and 1"
    )
    confidence_level: Optional[Series[float]] = Field(
        nullable=True, ge=0.0, le=1.0, description="Confidence level must be between 0 and 1"
    )
    method_used: Optional[Series[str]] = Field(nullable=True)
    p_value: Optional[Series[float]] = Field(
        nullable=True, ge=0.0, le=1.0, description="P-value must be between 0 and 1"
    )

    class Config:
        name = "causal_paths"
        strict = False
        coerce = True


# =============================================================================
# Schema 6: Agent Activities
# =============================================================================


class AgentActivitiesSchema(DataFrameModel):
    """Schema for agent_activities table data.

    Critical validations:
    - activity_id: Unique identifier
    - agent_tier: One of E2I agent tiers
    - confidence_level: Must be 0.0-1.0
    """

    activity_id: Series[str] = Field(nullable=False, unique=True)
    agent_name: Optional[Series[str]] = Field(nullable=True)
    agent_tier: Optional[Series[str]] = Field(nullable=True, isin=E2I_AGENT_TIERS + [None])
    activity_timestamp: Optional[Series[pd.Timestamp]] = Field(nullable=True, coerce=True)
    activity_type: Optional[Series[str]] = Field(nullable=True)
    confidence_level: Optional[Series[float]] = Field(
        nullable=True, ge=0.0, le=1.0, description="Confidence level must be between 0 and 1"
    )
    impact_estimate: Optional[Series[float]] = Field(nullable=True)
    execution_time_ms: Optional[Series[float]] = Field(nullable=True, ge=0.0)

    class Config:
        name = "agent_activities"
        strict = False
        coerce = True


# =============================================================================
# Schema Registry
# =============================================================================

PANDERA_SCHEMA_REGISTRY: Dict[str, Type[DataFrameModel]] = {
    "business_metrics": BusinessMetricsSchema,
    "predictions": PredictionsSchema,
    "ml_predictions": PredictionsSchema,  # Alias
    "triggers": TriggersSchema,
    "patient_journeys": PatientJourneysSchema,
    "causal_paths": CausalPathsSchema,
    "agent_activities": AgentActivitiesSchema,
}


def get_schema(data_source: str) -> Optional[Type[DataFrameModel]]:
    """Get Pandera schema for a data source.

    Args:
        data_source: Name of the data source (table/view name)

    Returns:
        DataFrameModel class or None if not found

    Example:
        >>> schema = get_schema("business_metrics")
        >>> if schema:
        ...     validated_df = schema.validate(df)
    """
    return PANDERA_SCHEMA_REGISTRY.get(data_source)


def validate_dataframe(df: pd.DataFrame, data_source: str, lazy: bool = True) -> Dict[str, Any]:
    """Validate a DataFrame against its Pandera schema.

    Args:
        df: DataFrame to validate
        data_source: Name of the data source
        lazy: If True, collect all errors; if False, fail on first error

    Returns:
        Dict with validation results:
        - status: "passed", "failed", or "skipped"
        - errors: List of error dicts (if failed)
        - rows_validated: Number of rows validated
        - schema_name: Name of schema used

    Example:
        >>> result = validate_dataframe(df, "business_metrics")
        >>> if result["status"] == "passed":
        ...     print("Schema validation passed!")
    """
    schema = get_schema(data_source)

    if schema is None:
        logger.warning(f"No Pandera schema found for data source: {data_source}")
        return {
            "status": "skipped",
            "errors": [],
            "rows_validated": len(df),
            "schema_name": None,
            "message": f"No schema defined for {data_source}",
        }

    try:
        # Validate with lazy=True to collect all errors
        schema.validate(df, lazy=lazy)

        logger.info(f"Schema validation passed for {data_source} ({len(df)} rows)")
        return {
            "status": "passed",
            "errors": [],
            "rows_validated": len(df),
            "schema_name": schema.Config.name,
        }

    except pa.errors.SchemaErrors as e:
        # Collect all schema errors
        errors = []
        for failure_case in e.failure_cases.to_dict(orient="records"):
            errors.append(
                {
                    "column": failure_case.get("column"),
                    "check": failure_case.get("check"),
                    "failure_case": str(failure_case.get("failure_case")),
                    "index": failure_case.get("index"),
                }
            )

        logger.warning(f"Schema validation failed for {data_source}: {len(errors)} errors")
        return {
            "status": "failed",
            "errors": errors,
            "rows_validated": len(df),
            "schema_name": schema.Config.name,
            "error_count": len(errors),
        }

    except pa.errors.SchemaError as e:
        # Single error (when lazy=False)
        logger.warning(f"Schema validation failed for {data_source}: {e}")
        return {
            "status": "failed",
            "errors": [{"message": str(e)}],
            "rows_validated": len(df),
            "schema_name": schema.Config.name,
            "error_count": 1,
        }

    except Exception as e:
        logger.error(f"Schema validation error for {data_source}: {e}")
        return {
            "status": "error",
            "errors": [{"message": str(e), "type": type(e).__name__}],
            "rows_validated": len(df),
            "schema_name": schema.Config.name if schema else None,
        }


def list_registered_schemas() -> Dict[str, str]:
    """List all registered Pandera schemas.

    Returns:
        Dict mapping data source names to schema class names
    """
    return {name: schema.__name__ for name, schema in PANDERA_SCHEMA_REGISTRY.items()}
