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
from typing import Any, Dict, List, Optional, Sequence, Type

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import DataFrameModel, Field
from pandera.typing.pandas import Series

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

# Valid journey stages (from journey_stage_type ENUM).
# Issue #155 §2: extended to 12 values (5 legacy + 7 PR #152 engagement-funnel
# values: aware / considering / prescribed / first_fill / adherent /
# discontinued / maintained). Migration 035 lands the 7 new values on
# existing Postgres databases.
E2I_JOURNEY_STAGES = [
    "diagnosis",
    "initial_treatment",
    "treatment_optimization",
    "maintenance",
    "treatment_switch",
    "aware",
    "considering",
    "prescribed",
    "first_fill",
    "adherent",
    "discontinued",
    "maintained",
]

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
# Schema 5b: HCP-adoption goldstd view (#2287, migration 162)
# =============================================================================


class HcpAdoptionGoldstdSchema(DataFrameModel):
    """Schema for hcp_adoption_goldstd_v (hcp_brand_adoption LEFT JOIN hcp_profiles).

    The retrain cohort of the hcp_adoption_<brand>_goldstd_lr_v1 champions (migration
    163 contract). Registered so a contract load is checked, not "skipped" (the fail-open
    shape #2320 closed). Checks are the database's own guarantees — the ``adopted`` CHECK
    constraint and the brand / region / split enums, numeric(3,2) for the score — plus
    non-negative counts. Covariates are nullable: the embed is a LEFT join and the
    hcp_profiles columns are nullable.
    """

    hcp_id: Optional[Series[str]] = Field(nullable=False)
    brand: Optional[Series[str]] = Field(
        nullable=False, isin=["Remibrutinib", "Fabhalta", "Kisqali", "competitor", "other"]
    )
    adopted: Series[int] = Field(nullable=False, isin=[0, 1])
    data_split: Optional[Series[str]] = Field(
        nullable=False, isin=["train", "validation", "test", "holdout", "unassigned"]
    )
    is_synthetic: Optional[Series[bool]] = Field(nullable=False)
    peer_influence_score: Optional[Series[float]] = Field(nullable=True, ge=0.0, le=9.99)
    influence_network_size: Optional[Series[float]] = Field(nullable=True, ge=0)
    years_experience: Optional[Series[float]] = Field(nullable=True, ge=0)
    specialty: Optional[Series[str]] = Field(nullable=True)
    geographic_region: Optional[Series[str]] = Field(nullable=True, isin=E2I_REGIONS + [None])

    class Config:
        name = "hcp_adoption_goldstd_v"
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
    "hcp_adoption_goldstd_v": HcpAdoptionGoldstdSchema,  # #2287, migration 162 view
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


def project_schema(model: Type[DataFrameModel], columns: Sequence[str]) -> pa.DataFrameSchema:
    """The schema a COLUMN-SCOPED load of ``model``'s table is held to (#2320).

    A table cohort contract's ``columns`` is a deliberate PROJECTION: the loader SELECTs
    only those columns, so a schema column outside it is NOT APPLICABLE — dropped here
    and logged at INFO, never failed. This mirrors the GE contract suite
    (``data_preparer.nodes.ge_validator._register_contract_suite``, owner decision
    2026-09-23). The projection narrows the schema; it does not weaken it:

    - a schema column inside the projection keeps every check it declares (dtype,
      nullability, uniqueness, ``isin`` / range) and becomes REQUIRED, even when the
      model marks it ``Optional``;
    - every projected column the model does not declare is added as a REQUIRED column
      with no other check, so a projected column missing from the frame still fails
      ``column_in_dataframe``.
    """
    projection: List[str] = list(dict.fromkeys(str(c) for c in columns))
    schema = model.to_schema()
    not_applicable = [c for c in schema.columns if c not in projection]
    if not_applicable:
        logger.info(
            "Pandera schema %r: skipping %s — not in contract projection",
            model.Config.name,
            not_applicable,
        )
        schema = schema.remove_columns(not_applicable)
    declared = [c for c in projection if c in schema.columns]
    if declared:
        schema = schema.update_columns({c: {"required": True} for c in declared})
    undeclared = {
        c: pa.Column(required=True, nullable=True) for c in projection if c not in schema.columns
    }
    if undeclared:
        schema = schema.add_columns(undeclared)
    return schema


def validate_dataframe(
    df: pd.DataFrame,
    data_source: str,
    lazy: bool = True,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Validate a DataFrame against its Pandera schema.

    Args:
        df: DataFrame to validate
        data_source: Name of the data source
        lazy: If True, collect all errors; if False, fail on first error
        columns: A table cohort contract's column projection. When non-empty the
            frame is held to ``project_schema(schema, columns)``; ``None`` or empty
            (the loader SELECTs every column for both) validates the whole schema.

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
        if columns:
            project_schema(schema, columns).validate(df, lazy=lazy)
        else:
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
